"""stage2/util/metrics.py — 監視用の指標（2026-09-09 ユーザー決定: 0 に近づく側は MSE、1 に近づく側は SSIM。RMSE / MAE の HU 表示は廃止）。

入力はどれも正規化空間 [−1, 1] の (N,1,H,W) tensor（学習・検証と同じもの）。
  mse  : 正規化空間の平均二乗誤差（学習の損失と同じ量）。表示は √ を取った rmse にする（ユーザー決定 2026-09-09。HU に直すなら rmse × (hu_max − hu_min) / 2 = × 2748）
  ssim : [0, 1] に直してから Wang et al. 2004 の標準（11×11 ガウス窓 σ 1.5、K1 0.01、K2 0.03、data range 1、窓は valid）。1 が完全一致
  psnr : [0, 1] の data range 1 で 10·log10(1 / mse01)。大きいほど良い（dB）。完全一致は 120 dB で頭打ち（0 割を避ける）
どれも N 枚の平均を float で返す。
"""

import math

import torch
import torch.nn.functional as F

_WINDOW = {}


def _to01(x):
    return ((x.float() + 1.0) / 2.0).clamp(0.0, 1.0)


def _gauss_window(size, sigma, device):
    key = (size, sigma, str(device))
    if key not in _WINDOW:
        i = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2.0
        g = torch.exp(-(i**2) / (2 * sigma**2))
        g = g / g.sum()
        _WINDOW[key] = torch.outer(g, g)[None, None]  # (1,1,size,size)
    return _WINDOW[key]


@torch.no_grad()
def mse(a, b):
    return float((a.float() - b.float()).pow(2).mean())


@torch.no_grad()
def psnr(a, b):
    m = float((_to01(a) - _to01(b)).pow(2).mean())
    return 10.0 * math.log10(1.0 / max(m, 1e-12))


@torch.no_grad()
def ssim(a, b, size=11, sigma=1.5, k1=0.01, k2=0.03):
    x, y = _to01(a), _to01(b)
    if x.shape[-1] < size or x.shape[-2] < size:
        raise ValueError(f"SSIM の窓 {size} より画像が小さい: {tuple(x.shape)}")
    w = _gauss_window(size, sigma, x.device)
    c1, c2 = k1**2, k2**2
    mu_x, mu_y = F.conv2d(x, w), F.conv2d(y, w)
    sxx = F.conv2d(x * x, w) - mu_x**2
    syy = F.conv2d(y * y, w) - mu_y**2
    sxy = F.conv2d(x * y, w) - mu_x * mu_y
    s = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (sxx + syy + c2))
    return float(s.mean())
