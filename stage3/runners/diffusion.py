"""stage3/runners/diffusion.py — 本家 ermongroup/ddim runners/diffusion.py を写し、条件付き（入力の channel 連結）にしたもの。[SpicaV5 改変 2026-10-13]

本家と同じもの（コードを写した）:
  ・get_beta_schedule（そのまま）、dict2namespace（本家 main.py からそのまま）
  ・Diffusion.__init__ の betas / num_timesteps / logvar
  ・train() の 1 step: e = randn_like(x)、antithetic な t（半分を引いて T − t − 1 と対にする）、loss → zero_grad → backward → grad_clip（church.yml には無い = 0）→ step → EMA 更新
  ・Model を torch.nn.DataParallel で包む（本家 train() / sample() と同じ。重みのキーは 'module.' 付き）
  ・sample: EMA の重みを使う（本家 sample(): ema_helper.register → load_state_dict → ema(model)）、step 列（本家 sample_image の skip_type uniform | quad）
  ・resume 時に optimizer の eps を今の設定に戻す（本家 states[1]["param_groups"][0]["eps"] = config.optim.eps）
変えたもの:
  ・データ: x0 = PCD patch、cond = EID-like patch（同位置）。loss は functions/losses.noise_estimation_loss_cond（model に [cond | x_t] を連結）
  ・デバイス: 本家は Diffusion(device=None) で cuda / cpu を自動で選ぶ。ここでは呼び出し側が明示する（machines.yaml の gpu_gen。黙って cpu に落とさない）
  ・ループ・保存・再開は SpicaV5 の train.py / util/run_paths.py の規約（本家の ckpt_{step}.pth 1 ファイルではなく model.pth / ema.pth / state.pth の 1 組を原子的に）
  ・sample は functions/denoising の generalized_steps_cond（patch）と generalized_steps_overlapping（1024 のタイル合成、WeatherDiffusion）。本家の FID / interpolation / sequence は持たない
  ・x_T は CPU の torch.Generator（seed）で作ってからデバイスに送る（学習の乱数列を消費しない。CPU / GPU で同じ x_T）
"""

import argparse
import random
import shutil
import zlib
from pathlib import Path

import numpy as np
import torch

from configs.schema import unflatten
from functions import get_optimizer
from functions.denoising import generalized_steps_cond, generalized_steps_overlapping, overlapping_grid_indices
from functions.losses import noise_estimation_loss_cond
from models.diffusion import Model
from models.ema import EMAHelper
from util.run_paths import CKPT_EMA, CKPT_MODEL, CKPT_STATE, ckpt_dir, replace_retry


# ---------------------------------------------------------------------------
# 本家そのまま
# ---------------------------------------------------------------------------
def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


# ---------------------------------------------------------------------------
# SpicaV5
# ---------------------------------------------------------------------------
def build_config(ddim_flat, device):
    """ddim.yml の実効値（平坦 dict）→ 本家と同じ Namespace（config.model.ch など）。本家 main.py と同じく config.device を持たせる（EMAHelper.ema_copy が使う）。"""
    config = dict2namespace(unflatten(ddim_flat))
    config.device = device
    return config


def timestep_seq(num_timesteps, timesteps, skip_type):
    """本家 sample_image の step 列（generalized）。uniform = range(0, T, T // S)（論文の linear）、quad = (linspace(0, √(0.8T), S))²。"""
    if skip_type == "uniform":
        skip = num_timesteps // timesteps
        seq = range(0, num_timesteps, skip)
    elif skip_type == "quad":
        seq = (
            np.linspace(
                0, np.sqrt(num_timesteps * 0.8), timesteps
            )
            ** 2
        )
        seq = [int(s) for s in list(seq)]
    else:
        raise NotImplementedError(skip_type)
    return list(seq)


def slice_seed(seed, name):
    """x_T の seed をスライス名と組み合わせる（スライスごとに別の雑音・同じ seed なら毎回同じ）。"""
    return (int(seed) * 1_000_003 + zlib.crc32(str(name).encode("utf-8"))) % (2 ** 63)


def initial_noise(shape, seed):
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    return torch.randn(*shape, generator=g)


def _to_tuple(x):
    if isinstance(x, list):
        return tuple(_to_tuple(v) for v in x)
    return x


class CondDiffusion:
    """条件付き DDIM の学習・サンプリング。本家 runners/diffusion.Diffusion の __init__ + train() の中身 + sample() の重み読み込みを 1 つにまとめたもの。"""

    def __init__(self, ddim_flat, device, train=True):
        self.ddim = dict(ddim_flat)
        self.device = device
        self.config = config = build_config(ddim_flat, device)

        # --- 本家 Diffusion.__init__ ---
        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

        # --- 本家 train() の前半 ---
        model = Model(config)
        model = model.to(self.device)
        self.model = torch.nn.DataParallel(model)
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self.optimizer = get_optimizer(config, self.model.parameters()) if train else None
        if config.model.ema:
            self.ema_helper = EMAHelper(mu=config.model.ema_rate)
            self.ema_helper.register(self.model)
        else:
            self.ema_helper = None
        self.grad_clip = float(config.optim.grad_clip)
        self.step, self.epoch = 0, 0
        self.run_config = None  # 保存時の実効設定（train.py が入れる。state.pth に残し、再開の基準にする）

    # ------------------------------------------------------------------ 学習（本家 train() のループの中身）
    def train_step(self, batch):
        """1 step。戻り値 {"loss": 本家の値（(C,H,W) の和の batch 平均）}。"""
        x = batch["target"].to(self.device, non_blocking=True)    # x0 = PCD patch
        cond = batch["input"].to(self.device, non_blocking=True)  # 条件 = EID-like patch
        n = x.size(0)
        self.model.train()
        self.step += 1

        e = torch.randn_like(x)
        b = self.betas

        # antithetic sampling
        t = torch.randint(
            low=0, high=self.num_timesteps, size=(n // 2 + 1,)
        ).to(self.device)
        t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
        loss = noise_estimation_loss_cond(self.model, x, cond, t, e, b)

        self.optimizer.zero_grad()
        loss.backward()

        if self.grad_clip > 0:  # 本家は try: clip_grad_norm_(..., config.optim.grad_clip) except: pass（church.yml にキーが無い = clip しない）
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()

        if self.ema_helper is not None:
            self.ema_helper.update(self.model)
        return {"loss": float(loss.detach())}

    @torch.no_grad()
    def eps_loss(self, cond, x0, model=None):
        """固定の val patch で ε の MSE（本家の損失と同じ量 = (C,H,W) の和の平均）。t は [0, T) に等間隔、雑音は固定（呼び出しごとに同じ）。"""
        model = self.model if model is None else model
        model.eval()
        n = x0.size(0)
        t = torch.from_numpy(np.linspace(0, self.num_timesteps - 1, n).round().astype(np.int64)).to(self.device)
        e = initial_noise(tuple(x0.shape), 777).to(self.device)
        out = noise_estimation_loss_cond(model, x0.to(self.device), cond.to(self.device), t, e, self.betas)
        return float(out)

    def ema_model(self):
        """サンプリング用の EMA の重みのコピー（本家 EMAHelper.ema_copy）。EMA なしなら学習中の model そのもの。"""
        if self.ema_helper is None:
            return self.model
        m = self.ema_helper.ema_copy(self.model)
        m.eval()
        return m

    # ------------------------------------------------------------------ サンプリング
    @torch.no_grad()
    def sample_patches(self, cond, steps, eta, skip_type, seeds, model):
        """cond (n,1,p,p) → x0 (n,1,p,p)。x_T は seeds[i] から 1 枚ずつ作る。"""
        x = torch.cat([initial_noise((1,) + tuple(cond.shape[1:]), s) for s in seeds]).to(self.device)
        seq = timestep_seq(self.num_timesteps, steps, skip_type)
        xs, _ = generalized_steps_cond(x, cond.to(self.device), seq, model, self.betas, eta=eta)
        return xs[-1]

    @torch.no_grad()
    def sample_tiled(self, cond_full, steps, eta, skip_type, seed, stride, tile_batch, model):
        """cond_full (1,1,H,W)（例 1024）→ x0 (1,1,H,W)。image_size のタイルを stride で並べ、毎ステップ重なりの ε を平均する（WeatherDiffusion）。"""
        p = self.config.data.image_size
        _, _, h, w = cond_full.shape
        h_list, w_list = overlapping_grid_indices(h, w, p, stride)
        corners = [(i, j) for i in h_list for j in w_list]
        x = initial_noise(tuple(cond_full.shape), seed).to(self.device)
        seq = timestep_seq(self.num_timesteps, steps, skip_type)
        xs, _ = generalized_steps_overlapping(x, cond_full.to(self.device), seq, model, self.betas, eta=eta,
                                              corners=corners, p_size=p, tile_batch=tile_batch)
        return xs[-1]

    # ------------------------------------------------------------------ checkpoint
    def _state(self):
        np_state = np.random.get_state()
        return {
            "step": int(self.step), "epoch": int(self.epoch),
            "optimizer": self.optimizer.state_dict(),
            "config": self.run_config,  # 保存時の実効設定（ddim / train / machine。平坦キー）。resume はこれを基準にする
            "rng": {
                "python": random.getstate(),
                "torch": torch.get_rng_state(),
                "numpy": (np_state[0], torch.from_numpy(np_state[1].astype(np.int64)), int(np_state[2]), int(np_state[3]), float(np_state[4])),
            },
        }

    def save(self, run_dir, tag):
        """model.pth / ema.pth / state.pth を重みディレクトリに原子的に保存する（<dir>.tmp に書く → 既存を .old に → .tmp を本番に → .old を消す）。"""
        final = ckpt_dir(run_dir, tag)
        tmp, old = final.with_name(final.name + ".tmp"), final.with_name(final.name + ".old")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        torch.save(self.model.state_dict(), tmp / CKPT_MODEL)
        torch.save(self.ema_helper.state_dict() if self.ema_helper is not None else {}, tmp / CKPT_EMA)
        torch.save(self._state(), tmp / CKPT_STATE)
        if old.exists():
            shutil.rmtree(old)
        if final.exists():
            replace_retry(final, old)
        replace_retry(tmp, final)
        if old.exists():
            shutil.rmtree(old)

    def load_weights(self, weight_dir, use_ema):
        """model.pth を strict に読み、use_ema なら EMA の重みを model に写す（本家 sample() と同じ手順）。ema.pth は学習の再開でも読む。"""
        weight_dir = Path(weight_dir)
        self.model.load_state_dict(torch.load(weight_dir / CKPT_MODEL, map_location=self.device, weights_only=True), strict=True)
        if self.ema_helper is not None:
            shadow = torch.load(weight_dir / CKPT_EMA, map_location=self.device, weights_only=True)
            expect = set(self.ema_helper.shadow)
            if set(shadow) != expect:
                raise RuntimeError(f"{weight_dir / CKPT_EMA} のキーが model と合いません（{len(shadow)} vs {len(expect)}）")
            self.ema_helper.load_state_dict(shadow)
            if use_ema:
                self.ema_helper.ema(self.model)
        elif use_ema:
            raise RuntimeError("ddim.yml model.ema が false の run に use_ema: true は使えません")

    def resume(self, state_path):
        """state.pth から optimizer / RNG / 進捗を復元する。lr / eps は今の実効値を再適用（本家も eps を戻す）。"""
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        self.optimizer.load_state_dict(state["optimizer"])
        for g in self.optimizer.param_groups:
            g["lr"] = self.config.optim.lr
            if "eps" in g:
                g["eps"] = self.config.optim.eps
        rng = state["rng"]
        random.setstate(_to_tuple(rng["python"]))
        torch.set_rng_state(rng["torch"].cpu())
        kind, keys, pos, has_gauss, cached = rng["numpy"]
        np.random.set_state((kind, keys.cpu().numpy().astype(np.uint32), pos, has_gauss, cached))
        self.step, self.epoch = int(state["step"]), int(state["epoch"])
        print(f"[resume] {state_path}: step {self.step}, epoch {self.epoch}, lr {self.optimizer.param_groups[0]['lr']:.2e}")
        return state


def describe_seq(num_timesteps, steps, skip_type):
    seq = timestep_seq(num_timesteps, steps, skip_type)
    return f"{len(seq)} steps ({skip_type}: {seq[0]}, {seq[1] if len(seq) > 1 else ''} ... {seq[-1]})"


def count_tiles(h, w, p, stride):
    hl, wl = overlapping_grid_indices(h, w, p, stride)
    return len(hl) * len(wl)


__all__ = ["CondDiffusion", "build_config", "count_tiles", "describe_seq", "get_beta_schedule", "slice_seed", "timestep_seq"]
