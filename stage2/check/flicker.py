"""stage2/check/flicker.py — 隣り合うスライスのちらつき（設計書 G-4）。確認コマンドだけが使う。

  連続 n 枚を U（+ V）に通して体積（n × H × W）にし、中央の行と列で縦に切った帯（n × W、n × H）を厚み方向に伸ばして保存する。
  乱数は「スライスごとに別」（既定）と「全スライス同じ」の 2 通り。正解 PCD1024 がある系列は同じ帯を並べる。
  flick/adj_corr = 隣のスライスとの「細かい成分」の相関（体内）。参考の数字（PCD の実測は隣で 0.77〜0.80 @PCD512）。
  2D で作るので、スライスごとに別の乱数ならノイズの相関は 0 に近い。**不成立の条件にしない**。見るのは「縁・形がスライスごとに動かないか」。
"""

import numpy as np
import torch

from models.pipeline import cpu_noise
from util import metrics as M


def middle_run(paths, n):
    """中央付近の連続 n 枚。"""
    n = min(n, len(paths))
    a = (len(paths) - n) // 2
    return paths[a:a + n]


@torch.no_grad()
def run_series(pipe, io, paths, case, mode, sigma_s_hu, steps, seed, same_noise, body_hu, should_stop=None):
    """戻り値 {"in", "mu", "out"（mode uv）: (n, H, W) の HU（CPU）, "mask": (n, H, W) bool}。"""
    un = pipe.units
    vols = {"in": [], "mu": [], "mask": []}
    if mode == "uv":
        vols["out"] = []
    for p in paths:
        if should_stop is not None and should_stop():
            raise KeyboardInterrupt
        x = io.load_x(p).to(pipe.device)
        mu, _ = pipe.mu(x)
        xh = un.hu(x)
        vols["in"].append(xh[0, 0].cpu()); vols["mu"].append(un.hu(mu)[0, 0].cpu())
        vols["mask"].append(M.body_mask(xh, pipe.sigma_px, body_hu)[0, 0].cpu())
        if mode == "uv":
            eps = cpu_noise(x.shape[-2:], seed, case, "same" if same_noise else p.name)[None, None]
            vols["out"].append(un.hu(pipe.flow(mu, x, eps, sigma_s_hu, steps)[0])[0, 0].cpu())
    return {k: torch.stack(v) for k, v in vols.items()}


def adj_corr(vol, mask):
    """隣のスライスとの細かい成分の相関の平均。vol (n, H, W) HU、mask (n, H, W)。"""
    f = M.fine(vol[:, None])[:, 0]
    vals = [M.corr(f[i], f[i + 1], mask[i] & mask[i + 1]) for i in range(vol.shape[0] - 1)]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def strips(vol, stretch):
    """体積 (n, H, W) → (中央の行で切った帯 (n·stretch, W), 中央の列で切った帯 (n·stretch, H))。numpy、HU。"""
    n, h, w = vol.shape
    row, col = vol[:, h // 2, :].numpy(), vol[:, :, w // 2].numpy()
    return np.repeat(row, stretch, axis=0), np.repeat(col, stretch, axis=0)
