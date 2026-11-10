"""stage2/models/u_model.py — U（形を決める回帰。設計書 B）の司令塔。

  forward : f = net(x↑)（net.residual なら x↑ + net(x↑)）、μ = lock(f, x↑) = f + G∗(x↑ − f)
  loss    : mean((μ − X′)²)、X′ = lock(X, x↑) = X + G∗(x↑ − X)。開くと (f − X) − G∗(f − X) = **広い平均を除いた成分だけの MSE**（入力 x↑ は損失から消える）
  精度    : 学習の net の forward だけ bf16 autocast + torch.compile。明るさの固定・損失は FP32（net の出力を .float() にしてから）。val・画像・推論は compile しない EMA を FP32 で
  学習    : Adam、lr 一定、勾配のクリップなし（旧 Stage 2 と同じ）。EMA。パッチの中で反射して G を計算する（設計書 D-3）
"""

import torch

from models.lock import lock
from models.pipeline import Pipeline
from models.trainer import Trainer
from models.unet import build_net


class UModel(Trainer):
    NET_NAME = "net_U"

    def __init__(self, data, u, shared, machine, device):
        self.data, self.sigma_px = data, float(shared["brightness.sigma_px_512"]) * data["data.upsample_scale"]
        self.residual = bool(u["net.residual"])
        self.net = build_net(u).to(device)
        self._setup(u, {"data": dict(data), "u": dict(u), "shared": dict(shared), "machine": dict(machine)}, device, weight_decay=0.0, warmup_steps=0, grad_clip=None)
        self.net_train = self.net
        self.reset_window()

    def _compile(self, mode):
        self.net_train = torch.compile(self.net, mode=mode)

    def reset_window(self):
        self.win = {"n": 0, "loss": 0.0, "grad": 0.0}

    def train_step(self, batch):
        x = batch["x"].to(self.device, non_blocking=True).float()
        t = batch["t"].to(self.device, non_blocking=True).float()
        xin = x.contiguous(memory_format=torch.channels_last) if self.channels_last else x
        self.net.train()
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_bf16):
            f = self.net_train(xin)
        f = f.float()
        if self.compiled:
            f = f.clone()  # max-autotune の CUDA Graph は出力の領域を使い回す
        if self.residual:
            f = x + f
        mu = lock(f, x, self.sigma_px)
        with torch.no_grad():
            target = lock(t, x, self.sigma_px)
        loss = (mu - target).square().mean()
        loss_v, grad = self._update(loss)
        w = self.win
        w["n"] += 1; w["loss"] += loss_v; w["grad"] += grad
        n = self.cfg["log.n_images"]
        self.last = {"x": x[:n].detach(), "mu": mu[:n].detach(), "target": target[:n].detach()}
        return loss_v

    def window_stats(self):
        w = self.win
        n = max(w["n"], 1)
        return {"loss/mse": w["loss"] / n, "diag/train_rmse": (w["loss"] / n) ** 0.5, "diag/grad_norm": w["grad"] / n}

    def pipeline(self):
        """val・画像用: EMA、FP32、compile なし。"""
        return Pipeline(self.data, self.sigma_px, self.ema, self.residual, self.device)
