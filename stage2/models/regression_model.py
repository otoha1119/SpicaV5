"""stage2/models/regression_model.py — Stage 2 の教師あり回帰（U-Net + MSE / L1）の司令塔。junyanz の BaseModel は使わない。

  forward   : pred = net(input)（mode.residual=true なら input + net(input)）
  loss      : mse | l1 を正規化空間 [−1,1] で計算。表示用に HU 換算（1.0 = hu_per_unit HU）の RMSE / MAE も返す
  optimizer : Adam(lr, (beta1, beta2))。lr は一定（論文に減衰の記載なし）→ scheduler は持たない
  checkpoint: 重みディレクトリ（util/run_paths.py）に net_G.pth と state.pth（optimizer / RNG / epoch / total_iters）を**原子的に**保存
              （<dir>.tmp に書く → 既存 <dir> を .old に → .tmp を <dir> に → .old を消す。Stage 1 の save_networks と同じ手順）
  resume    : state.pth から optimizer と RNG（python / torch / numpy）を復元し、lr / betas は今の実効値を再適用（上書きが最優先）
"""

import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from data.ct_io import hu_per_unit
from models.unet import build_net
from util.run_paths import ckpt_dir


def _to_tuple(x):
    """random.setstate は入れ子の tuple を要求する。torch.load 後に list になっている場合に戻す。"""
    if isinstance(x, list):
        return tuple(_to_tuple(v) for v in x)
    return x


class RegressionModel:
    def __init__(self, train, mode, device):
        self.device = device
        self.residual = mode["residual"]
        self.hu_unit = hu_per_unit(train["data.hu_min"], train["data.hu_max"])
        self.net = build_net(mode).to(device)
        if mode["loss"] == "mse":
            self.criterion = nn.MSELoss()
        elif mode["loss"] == "l1":
            self.criterion = nn.L1Loss()
        else:
            raise ValueError(mode["loss"])
        self.loss_name = mode["loss"]
        self.lr, self.betas = train["optim.lr"], (train["optim.beta1"], train["optim.beta2"])
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, betas=self.betas)
        self.cur_epoch, self.epoch_done, self.total_iters = 0, False, 0
        self.n_params = sum(p.numel() for p in self.net.parameters())

    # --- 前向き ---
    def forward(self, x):
        y = self.net(x)
        return x + y if self.residual else y

    def optimize(self, batch):
        """1 step。戻り値は表示用の dict（loss は正規化空間、rmse_hu / mae_hu は HU 換算）。"""
        x = batch["input"].to(self.device, non_blocking=True)
        t = batch["target"].to(self.device, non_blocking=True)
        self.net.train()
        pred = self.forward(x)
        loss = self.criterion(pred, t)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        with torch.no_grad():
            d = pred - t
            return {"loss": float(loss), "rmse_hu": float(d.pow(2).mean().sqrt()) * self.hu_unit, "mae_hu": float(d.abs().mean()) * self.hu_unit}

    @torch.no_grad()
    def predict(self, x):
        """監視・検証用の順伝播（eval、勾配なし）。x は (N,1,H,W) 正規化 tensor（CPU 可）。"""
        self.net.eval()
        return self.forward(x.to(self.device)).cpu()

    @torch.no_grad()
    def evaluate(self, dataset, slices):
        """val のフル画像で RMSE / MAE [HU] を出す。入力そのまま（何もしない場合）の値も参照線として返す。"""
        self.net.eval()
        se, ae, se_in, ae_in, n = 0.0, 0.0, 0.0, 0.0, 0
        for _case, in_path, pcd_path in slices:
            x = dataset.load_full(in_path).to(self.device)
            t = dataset.load_full(pcd_path).to(self.device)
            d = self.forward(x) - t
            d_in = x - t
            se += float(d.pow(2).mean()); ae += float(d.abs().mean())
            se_in += float(d_in.pow(2).mean()); ae_in += float(d_in.abs().mean())
            n += 1
        if n == 0:
            raise RuntimeError("検証スライスがありません")
        u = self.hu_unit
        return {"rmse_hu": (se / n) ** 0.5 * u, "mae_hu": ae / n * u, "rmse_input_hu": (se_in / n) ** 0.5 * u, "mae_input_hu": ae_in / n * u, "n": n}

    # --- checkpoint ---
    def _state(self):
        np_state = np.random.get_state()
        return {
            "epoch": int(self.cur_epoch), "epoch_done": bool(self.epoch_done), "total_iters": int(self.total_iters),
            "optimizer": self.optimizer.state_dict(),
            "lr": float(self.optimizer.param_groups[0]["lr"]),
            "rng": {
                "python": random.getstate(),
                "torch": torch.get_rng_state(),
                "numpy": (np_state[0], torch.from_numpy(np_state[1].astype(np.int64)), int(np_state[2]), int(np_state[3]), float(np_state[4])),
            },
        }

    def save(self, run_dir, tag):
        """net_G.pth と state.pth を重みディレクトリに原子的に保存する。"""
        final = ckpt_dir(run_dir, tag)
        tmp, old = final.with_name(final.name + ".tmp"), final.with_name(final.name + ".old")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        torch.save(self.net.state_dict(), tmp / "net_G.pth")
        torch.save(self._state(), tmp / "state.pth")
        if old.exists():
            shutil.rmtree(old)
        if final.exists():
            os.replace(final, old)
        os.replace(tmp, final)
        if old.exists():
            shutil.rmtree(old)

    def load_weights(self, weight_dir):
        self.net.load_state_dict(torch.load(Path(weight_dir) / "net_G.pth", map_location=self.device, weights_only=True))

    def resume(self, state_path):
        """state.pth から optimizer / RNG / 進捗を復元する。state は CPU に読む（RNG 用 Tensor を CUDA に載せない）。lr / betas は今の実効値を再適用。"""
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        self.optimizer.load_state_dict(state["optimizer"])
        for g in self.optimizer.param_groups:
            g["lr"] = self.lr
            g["betas"] = self.betas
        rng = state["rng"]
        random.setstate(_to_tuple(rng["python"]))
        torch.set_rng_state(rng["torch"].cpu())
        kind, keys, pos, has_gauss, cached = rng["numpy"]
        np.random.set_state((kind, keys.cpu().numpy().astype(np.uint32), pos, has_gauss, cached))
        self.total_iters = int(state["total_iters"])
        print(f"[resume] {state_path}: epoch {state['epoch']} (done={state['epoch_done']}), total_iters {state['total_iters']}, lr {self.optimizer.param_groups[0]['lr']:.7f}")
        return state
