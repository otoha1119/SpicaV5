"""stage2/models/trainer.py — U と V の司令塔に共通の部品: EMA、学習率、compile の差し替え、原子的な保存と再開。
保存・再開・compile の扱いは stage1/models/painter.py（元は old/stage2/models/regression_model.py）から写した（_swap_dirs、state.pth に実効設定、CUDA Graph 対策の clone）。

重みディレクトリ（util/run_paths.py）: <net 名>.pth（学習中の重み）・<net 名>_ema.pth（EMA）・state.pth（optimizer / 乱数 / step / 累計時間 / **保存時の実効設定 config**）
  + 司令塔ごとの追加ファイル（V は net_U.pth = 使った U の EMA の写し）。<dir>.tmp に揃えてから差し替える（途中で殺されても前の重みが残る）。
"""

import copy
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from util.run_paths import ckpt_dir


class NonFiniteError(RuntimeError):
    """損失か勾配が NaN / Inf。学習を止める（重みは更新しない）。"""


def _swap_dirs(tmp, final, old, retries=10, wait=3.0):
    """tmp を final に差し替える（Windows のバインドマウントで os.replace が PermissionError になる対策。Stage 1 と同じ）。
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


def want_bf16(precision, device):
    return precision == "bf16" and device.type == "cuda"


def check_compiler(what):
    if not (shutil.which(os.environ.get("CC", "cc")) or shutil.which("gcc")):
        raise RuntimeError(f"torch.compile に必要な C コンパイラがありません（{what} false で切るか、イメージを確認）")


class Trainer:
    """共通部品。子クラスが self.net（学習するネット）を作ってから _setup() を呼ぶ。"""

    NET_NAME = "net"  # 重みファイル名の頭（net_U / net_V）

    def _setup(self, cfg, config, device, weight_decay, warmup_steps, grad_clip):
        self.cfg, self.config, self.device = cfg, config, device
        self.channels_last = bool(cfg["optim.channels_last"])
        if self.channels_last:
            self.net = self.net.to(memory_format=torch.channels_last)
        self.ema = copy.deepcopy(self.net).requires_grad_(False)
        self.ema.eval()
        self.n_params = sum(p.numel() for p in self.net.parameters())
        self.use_bf16 = want_bf16(cfg["optim.precision"], device)
        self.compile_enabled = bool(cfg["optim.compile"]) and device.type == "cuda"
        if self.compile_enabled:
            check_compiler("--compile")
        self.compiled = False
        self.params = [p for p in self.net.parameters() if p.requires_grad]
        self.lr, self.warmup_steps, self.grad_clip = cfg["optim.lr"], warmup_steps, grad_clip
        self.betas, self.weight_decay = (cfg["optim.beta1"], cfg["optim.beta2"]), weight_decay
        self.opt = torch.optim.Adam(self.params, lr=self.lr, betas=self.betas, weight_decay=weight_decay)
        self.step, self.elapsed = 0, 0.0  # 累計の step と学習時間（秒。再開をまたぐ）
        self.last = None

    # --- compile ---
    def maybe_compile(self, local_step):
        """この起動の local_step 回目の step の前に呼ぶ。compile_after_steps を過ぎたら学習の呼び出し口を torch.compile に差し替える。差し替えたら True。"""
        if not self.compile_enabled or self.compiled or local_step <= self.cfg["optim.compile_after_steps"]:
            return False
        self._compile(self.cfg["optim.compile_mode"])
        self.compiled = True
        return True

    def _compile(self, mode):
        raise NotImplementedError

    # --- 更新 ---
    def current_lr(self):
        w = self.warmup_steps
        return self.lr * min(1.0, (self.step + 1) / w) if w > 0 else self.lr

    def _update(self, loss):
        """損失 → 勾配 → 更新 → EMA。損失か勾配が有限でなければ、重みを更新せずに NonFiniteError。戻り値 (loss の値, 勾配のノルム)。"""
        loss_v = loss.item()
        if not np.isfinite(loss_v):
            raise NonFiniteError(f"損失が有限でない（{loss_v}）。step {self.step + 1}")
        for g in self.opt.param_groups:
            g["lr"] = self.current_lr()
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        grad = float(torch.nn.utils.clip_grad_norm_(self.params, self.grad_clip if self.grad_clip else float("inf")))
        if not np.isfinite(grad):
            raise NonFiniteError(f"勾配のノルムが有限でない（{grad}）。step {self.step + 1}")
        self.opt.step()
        self.step += 1
        with torch.no_grad():
            d = min(self.cfg["optim.ema_decay"], (1.0 + self.step) / (10.0 + self.step))
            for pe, p in zip(self.ema.parameters(), self.net.parameters()):
                pe.lerp_(p.detach(), 1.0 - d)
        return loss_v, grad

    # --- checkpoint ---
    def _extra_files(self):
        return {}

    def _state(self):
        np_state = np.random.get_state()
        rng = {"python": random.getstate(), "torch": torch.get_rng_state(),
               "numpy": (np_state[0], torch.from_numpy(np_state[1].astype(np.int64)), int(np_state[2]), int(np_state[3]), float(np_state[4]))}
        if self.device.type == "cuda":
            rng["cuda"] = torch.cuda.get_rng_state(self.device)
        return {"step": int(self.step), "elapsed_sec": float(self.elapsed), "optimizer": self.opt.state_dict(),
                "lr": float(self.opt.param_groups[0]["lr"]), "n_params": int(self.n_params), "config": self.config, "rng": rng}

    def save(self, run_dir, tag):
        """重みディレクトリに原子的に保存する。保存できたら True。"""
        final = ckpt_dir(run_dir, tag)
        tmp, old = final.with_name(final.name + ".tmp"), final.with_name(final.name + ".old")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        torch.save(self.net.state_dict(), tmp / f"{self.NET_NAME}.pth")
        torch.save(self.ema.state_dict(), tmp / f"{self.NET_NAME}_ema.pth")
        for name, obj in self._extra_files().items():
            torch.save(obj, tmp / name)
        torch.save(self._state(), tmp / "state.pth")
        return _swap_dirs(tmp, final, old)

    def resume(self, state_path):
        """重みディレクトリ（state.pth のある所）から net / ema / optimizer / 乱数 / 進み具合を復元する。lr・betas は今の実効値を再適用。"""
        d = Path(state_path).parent
        self.net.load_state_dict(torch.load(d / f"{self.NET_NAME}.pth", map_location=self.device, weights_only=True))
        self.ema.load_state_dict(torch.load(d / f"{self.NET_NAME}_ema.pth", map_location=self.device, weights_only=True))
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        self.opt.load_state_dict(state["optimizer"])
        for g in self.opt.param_groups:
            g["lr"], g["betas"], g["weight_decay"] = self.lr, self.betas, self.weight_decay
        rng = state["rng"]
        random.setstate(_to_tuple(rng["python"]))
        torch.set_rng_state(rng["torch"].cpu())
        kind, keys, pos, has_gauss, cached = rng["numpy"]
        np.random.set_state((kind, keys.cpu().numpy().astype(np.uint32), pos, has_gauss, cached))
        if self.device.type == "cuda" and "cuda" in rng:
            torch.cuda.set_rng_state(rng["cuda"].cpu(), self.device)
        self.step, self.elapsed = int(state["step"]), float(state["elapsed_sec"])
        print(f"[resume] {d}: step {self.step}, 累計 {self.elapsed / 3600:.2f} 時間", flush=True)
