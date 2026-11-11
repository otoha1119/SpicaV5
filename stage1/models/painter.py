"""stage1/models/painter.py — 描き手 P の司令塔: EDM の前処理、bf16 / compile の包み、EMA、損失、原子的な保存と再開。
保存・再開・compile の扱いは old/stage2/models/regression_model.py から写した（_swap_dirs、state.pth に実効設定、CUDA Graph 対策の clone）。

精度の決まり（設計書 C-4）:
  FP32 で計算する所 : 度合いの埋め込み、c_in·x、Haar の折り畳み、stem の Conv、head（GroupNorm・Conv・逆 Haar）、x̂0 = c_skip·x + c_out·F、損失、ぼかし、錨
  bf16 autocast の所 : U-Net 本体（net.body）だけ。torch.compile の対象も本体だけ（autocast の切り替えは compile する範囲の外）
  学習の本体の出力は .float() で FP32 に戻し、compile しているときは clone する（max-autotune の CUDA Graph は出力の領域を使い回す）
  val・途中経過の描画・generate は compile していない EMA の重みで通す（state_dict のキーは素のモジュールのまま）
"""

import copy
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from diffusion.cue import CUE_KINDS, center_crop, cue_kind, draw_cue_params, gaussian_blur
from diffusion.edm import N_BINS, draw_sigma, precond, sigma_bin
from models.unet import build_net
from util.run_paths import ckpt_dir


class NonFiniteError(RuntimeError):
    """損失か勾配が NaN / Inf。学習を止める（重みは更新しない）。"""


def _swap_dirs(tmp, final, old, retries=10, wait=3.0):
    """tmp を final に差し替える（Windows のバインドマウントで os.replace が PermissionError になる対策。旧コードと同じ）。
    再試行 → ファイル単位の置き換え → それでも駄目なら警告して学習を続ける（次回の保存で再試行）。"""
    err = None
    for _ in range(retries):
        try:
            if final.exists() and old.exists():
                shutil.rmtree(old)
            if final.exists():
                os.replace(final, old)
            os.replace(tmp, final)
            if old.exists():
                shutil.rmtree(old, ignore_errors=True)
            return True
        except PermissionError as e:
            err = e
            time.sleep(wait)
    try:
        final.mkdir(parents=True, exist_ok=True)
        for f in sorted(tmp.iterdir()):
            os.replace(f, final / f.name)
        shutil.rmtree(tmp, ignore_errors=True)
        if old.exists():
            shutil.rmtree(old, ignore_errors=True)
        print(f"[save] 注意: {final} のフォルダ改名が拒否されたのでファイル単位で置き換えた（{err}）", flush=True)
        return True
    except (PermissionError, FileNotFoundError) as e2:
        print(f"[save] 警告: checkpoint {final} を保存できなかった（{e2}）。学習は続け、次回の保存で再試行する", flush=True)
        return False


def _to_tuple(x):
    if isinstance(x, list):
        return tuple(_to_tuple(v) for v in x)
    return x


def denoise(net, x, sigma, cue, blur_px, noise_x, has, sigma_data, body=None, bf16=False, channels_last=False, clone=False):
    """EDM の前処理つきの 1 回の forward。x (B, 1, H, W)、sigma / blur_px / noise_x (B,)、has (B,) bool。戻り値 (x̂0, F)（FP32）。
    body を渡すとその呼び出し口（compile 済みなど）で本体を通す。autocast の外で呼ぶこと。"""
    x = x.float()
    c_skip, c_out, c_in = precond(sigma, sigma_data)
    emb = net.embed(sigma.float(), blur_px.float(), noise_x.float(), has)
    h = net.stem(c_in * x, cue.float() * has.view(-1, 1, 1, 1).to(torch.float32))  # 手がかり無しの行は 0
    if channels_last:
        h = h.contiguous(memory_format=torch.channels_last)
    with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=bf16):
        h = (net.body if body is None else body)(h, emb)
    h = h.float()
    if clone:
        h = h.clone()
    f = net.head(h)
    return c_skip * x + c_out * f, f


def want_bf16(precision, device):
    return precision == "bf16" and device.type == "cuda"


class Painter:
    def __init__(self, train, model, shared, machine, device):
        self.device = device
        self.train = train
        self.config = {"train": dict(train), "model": dict(model), "shared": dict(shared), "machine": dict(machine)}  # この重みの実効設定（state.pth に入れる。再開の基準）
        self.sigma_data = train["diffusion.sigma_data"]
        self.channels_last = bool(train["optim.channels_last"])
        self.net = build_net(model, train).to(device)
        if self.channels_last:
            self.net = self.net.to(memory_format=torch.channels_last)
        self.ema = copy.deepcopy(self.net).requires_grad_(False)
        self.ema.eval()
        self.n_params = sum(p.numel() for p in self.net.parameters())
        self.use_bf16 = want_bf16(train["optim.precision"], device)
        self.sample_bf16 = want_bf16(train["sample.precision"], device)
        self.compile_enabled = bool(train["optim.compile"]) and device.type == "cuda"
        if self.compile_enabled and not (shutil.which(os.environ.get("CC", "cc")) or shutil.which("gcc")):
            raise RuntimeError("torch.compile に必要な C コンパイラがありません（--compile false で切るか、イメージを確認）")
        self.body_train, self.compiled = self.net.body, False  # compile は maybe_compile が後から差し替える
        self.params = [p for p in self.net.parameters() if p.requires_grad]
        self.lr = train["optim.lr"]
        self.opt = torch.optim.AdamW(self.params, lr=self.lr, betas=(train["optim.beta1"], train["optim.beta2"]), weight_decay=train["optim.weight_decay"])
        self.step, self.elapsed = 0, 0.0  # 累計の step と学習時間（秒。再開をまたぐ）
        self.last = None
        self.reset_window()

    # --- compile ---
    def maybe_compile(self, local_step):
        """この起動の local_step 回目の step の前に呼ぶ。compile_after_steps を過ぎたら本体を torch.compile に差し替える。差し替えたら True。"""
        if not self.compile_enabled or self.compiled or local_step <= self.train["optim.compile_after_steps"]:
            return False
        mode = self.train["optim.compile_mode"]
        self.body_train = torch.compile(self.net.body, mode=mode)
        self.compiled = True
        return True

    # --- 学習 ---
    def reset_window(self):
        d = self.device
        self.win = {"n": 0, "loss": 0.0, "grad": 0.0,
                    "bin_sum": torch.zeros(N_BINS, device=d), "bin_cnt": torch.zeros(N_BINS, device=d),
                    "kind_sum": torch.zeros(len(CUE_KINDS), device=d), "kind_cnt": torch.zeros(len(CUE_KINDS), device=d)}

    def current_lr(self):
        w = self.train["optim.warmup_steps"]
        return self.lr * min(1.0, (self.step + 1) / w) if w > 0 else self.lr

    def train_inputs(self, big):
        """(B, 1, S, S) のパッチ（余白つき）→ 学習の 1 step の入力。くじはすべてここで引く（GPU、FP32）。"""
        t, m = self.train, self.train["data.margin"]
        b = big.shape[0]
        blur_px, noise_x, has = draw_cue_params(b, t, self.device)
        x0 = center_crop(big, m)
        cue = center_crop(gaussian_blur(big, blur_px), m) + noise_x[:, None, None, None] * torch.randn_like(x0)
        cue = cue * has.view(-1, 1, 1, 1)
        sigma = draw_sigma(b, t["diffusion.sigma_min"], t["diffusion.sigma_max"], self.device)
        x = x0 + sigma.view(-1, 1, 1, 1) * torch.randn_like(x0)
        return x0, x, sigma, cue, blur_px, noise_x, has

    def loss_per_sample(self, net, x0, x, sigma, cue, blur_px, noise_x, has, body=None, bf16=False, clone=False):
        """EDM の損失（F の側で重み 1）。サンプルごとの値 (B,) と x̂0 を返す。FP32。"""
        c_skip, c_out, _ = precond(sigma, self.sigma_data)
        target = (x0 - c_skip * x) / c_out
        x0_hat, f = denoise(net, x, sigma, cue, blur_px, noise_x, has, self.sigma_data, body=body, bf16=bf16, channels_last=self.channels_last, clone=clone)
        return ((f - target) ** 2).mean(dim=(1, 2, 3)), x0_hat

    def train_step(self, batch):
        """1 step。損失か勾配が有限でなければ、重みを更新せずに NonFiniteError。戻り値 (loss, grad_norm)。"""
        big = batch.to(self.device, non_blocking=True).float()
        with torch.no_grad():
            x0, x, sigma, cue, blur_px, noise_x, has = self.train_inputs(big)
        for g in self.opt.param_groups:
            g["lr"] = self.current_lr()
        self.net.train()
        per, x0_hat = self.loss_per_sample(self.net, x0, x, sigma, cue, blur_px, noise_x, has, body=self.body_train, bf16=self.use_bf16, clone=self.compiled)
        loss = per.mean()
        loss_v = loss.item()
        if not np.isfinite(loss_v):
            raise NonFiniteError(f"損失が有限でない（{loss_v}）。step {self.step + 1}")
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        grad = float(torch.nn.utils.clip_grad_norm_(self.params, self.train["optim.grad_clip"]))
        if not np.isfinite(grad):
            raise NonFiniteError(f"勾配のノルムが有限でない（{grad}）。step {self.step + 1}")
        self.opt.step()
        self.step += 1
        with torch.no_grad():
            d = min(self.train["optim.ema_decay"], (1.0 + self.step) / (10.0 + self.step))
            for pe, p in zip(self.ema.parameters(), self.net.parameters()):
                pe.lerp_(p.detach(), 1.0 - d)
            per = per.detach()
            w = self.win
            w["n"] += 1; w["loss"] += loss_v; w["grad"] += grad
            ones = torch.ones_like(per)
            b = sigma_bin(sigma, self.train["data.hu_scale"])
            w["bin_sum"].index_add_(0, b, per); w["bin_cnt"].index_add_(0, b, ones)
            k = cue_kind(blur_px, noise_x, has)
            w["kind_sum"].index_add_(0, k, per); w["kind_cnt"].index_add_(0, k, ones)
            n = self.train["log.n_train_patches"]
            self.last = tuple(v[:n].detach() for v in (x0, cue, x, x0_hat, sigma, blur_px, noise_x, has))
        return loss_v, grad

    def window_stats(self):
        """直近の窓の平均（scalar 用）。数が 0 の帯・型は出さない。"""
        w = self.win
        out = {"loss/train": w["loss"] / max(w["n"], 1), "diag/grad_norm": w["grad"] / max(w["n"], 1)}
        for name, s, c, labels in (("loss_bin", w["bin_sum"], w["bin_cnt"], [str(i) for i in range(N_BINS)]), ("loss_cue", w["kind_sum"], w["kind_cnt"], CUE_KINDS)):
            s, c = s.tolist(), c.tolist()
            for i, lab in enumerate(labels):
                if c[i] > 0:
                    out[f"{name}/{lab}"] = s[i] / c[i]
        return out

    # --- val・描画（EMA、compile なし） ---
    @torch.no_grad()
    def evaluate(self, val):
        """val の固定パッチの損失（EMA、FP32）。学習が bf16 のときは、同じ入力を bf16 でも通して帯ごとの比（bf16 ÷ FP32）も返す（設計書 C-4 の確認）。"""
        t, m, bs = self.train, self.train["data.margin"], self.train["optim.batch_size"]
        n = val["big"].shape[0]
        per32, per16, bins = [], [], []
        for i in range(0, n, bs):
            g = {k: v[i:i + bs].to(self.device) for k, v in val.items()}
            x0 = center_crop(g["big"], m)
            cue = (center_crop(gaussian_blur(g["big"], g["blur_px"]), m) + g["noise_x"][:, None, None, None] * g["cue_noise"]) * g["has"].view(-1, 1, 1, 1)
            x = x0 + g["sigma"].view(-1, 1, 1, 1) * g["eps"]
            args = (self.ema, x0, x, g["sigma"], cue, g["blur_px"], g["noise_x"], g["has"])
            per32.append(self.loss_per_sample(*args)[0])
            if self.use_bf16:
                per16.append(self.loss_per_sample(*args, bf16=True)[0])
            bins.append(sigma_bin(g["sigma"], t["data.hu_scale"]))
        per32, bins = torch.cat(per32), torch.cat(bins)
        out = {"val/loss": float(per32.mean())}
        for i in range(N_BINS):
            sel = bins == i
            if bool(sel.any()):
                out[f"val/loss_bin/{i}"] = float(per32[sel].mean())
                if per16:
                    out[f"diag/bf16_val_ratio/bin{i}"] = float(torch.cat(per16)[sel].mean() / per32[sel].mean().clamp_min(1e-12))
        return out

    def ema_denoiser(self):
        """途中経過の描画用: EMA・compile なし・sample.precision。sampler.redraw に渡す関数。"""
        def fn(x, sigma, cue, blur_px, noise_x, has):
            return denoise(self.ema, x, sigma, cue, blur_px, noise_x, has, self.sigma_data, bf16=self.sample_bf16, channels_last=self.channels_last)[0]
        return fn

    # --- checkpoint ---
    def _state(self):
        np_state = np.random.get_state()
        rng = {"python": random.getstate(), "torch": torch.get_rng_state(),
               "numpy": (np_state[0], torch.from_numpy(np_state[1].astype(np.int64)), int(np_state[2]), int(np_state[3]), float(np_state[4]))}
        if self.device.type == "cuda":
            rng["cuda"] = torch.cuda.get_rng_state(self.device)
        return {"step": int(self.step), "elapsed_sec": float(self.elapsed), "optimizer": self.opt.state_dict(),
                "lr": float(self.opt.param_groups[0]["lr"]), "n_params": int(self.n_params), "config": self.config, "rng": rng}

    def save(self, run_dir, tag):
        """net.pth / ema.pth / state.pth を重みディレクトリに原子的に保存する。保存できたら True。"""
        final = ckpt_dir(run_dir, tag)
        tmp, old = final.with_name(final.name + ".tmp"), final.with_name(final.name + ".old")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        torch.save(self.net.state_dict(), tmp / "net.pth")
        torch.save(self.ema.state_dict(), tmp / "ema.pth")
        torch.save(self._state(), tmp / "state.pth")
        return _swap_dirs(tmp, final, old)

    def init_from(self, weight_dir):
        """別の run の EMA の重みを net と EMA の両方に読む（型 X: PCD の描き手を実 EID の描き手から始める）。optimizer・step は新規のまま。"""
        sd = torch.load(Path(weight_dir) / "ema.pth", map_location=self.device, weights_only=True)
        self.net.load_state_dict(sd)
        self.ema.load_state_dict(sd)
        print(f"[init] {weight_dir} の EMA の重みから始める", flush=True)

    def resume(self, state_path):
        """重みディレクトリ（state.pth のある所）から net / ema / optimizer / 乱数 / 進み具合を復元する。lr・betas は今の実効値を再適用。"""
        d = Path(state_path).parent
        self.net.load_state_dict(torch.load(d / "net.pth", map_location=self.device, weights_only=True))
        self.ema.load_state_dict(torch.load(d / "ema.pth", map_location=self.device, weights_only=True))
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        self.opt.load_state_dict(state["optimizer"])
        for g in self.opt.param_groups:
            g["lr"] = self.lr
            g["betas"] = (self.train["optim.beta1"], self.train["optim.beta2"])
            g["weight_decay"] = self.train["optim.weight_decay"]
        rng = state["rng"]
        random.setstate(_to_tuple(rng["python"]))
        torch.set_rng_state(rng["torch"].cpu())
        kind, keys, pos, has_gauss, cached = rng["numpy"]
        np.random.set_state((kind, keys.cpu().numpy().astype(np.uint32), pos, has_gauss, cached))
        if self.device.type == "cuda" and "cuda" in rng:
            torch.cuda.set_rng_state(rng["cuda"].cpu(), self.device)
        self.step, self.elapsed = int(state["step"]), float(state["elapsed_sec"])
        print(f"[resume] {d}: step {self.step}, 累計 {self.elapsed / 3600:.2f} 時間", flush=True)


def load_ema(weight_dir, device):
    """描画用: 重みディレクトリの state.pth の実効設定でネットを組み、ema.pth を読む。戻り値 (net, config, step)。"""
    weight_dir = Path(weight_dir)
    state = torch.load(weight_dir / "state.pth", map_location="cpu", weights_only=True)
    cfg = state["config"]
    net = build_net(cfg["model"], cfg["train"]).to(device)
    net.load_state_dict(torch.load(weight_dir / "ema.pth", map_location=device, weights_only=True))
    if cfg["train"]["optim.channels_last"]:
        net = net.to(memory_format=torch.channels_last)
    net.eval().requires_grad_(False)
    return net, cfg, int(state["step"])
