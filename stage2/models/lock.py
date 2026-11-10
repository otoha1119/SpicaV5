"""stage2/models/lock.py — 明るさの固定（設計書 D、RECONCILE #1・#2）。学習パラメータなし。

  G∗x            = 「広い平均」。Gaussian（σ = configs/shared.yaml の brightness.sigma_px_512 × 補間の倍率 = 1024 で 32 画素）。
                   半径 ceil(3σ) で打ち切って合計 1 に正規化した 1 次元の核を、横 → 縦に掛ける（分離型）。端は reflect。FP32。
  lock(out, ref) = out + G∗(ref − out)      … out の広い平均を捨てて、ref（= 入力 x↑）の広い平均に置き換える

Stage 1 の錨（stage1/diffusion/anchor.py: x̂0 + (G(元) − G(x̂0))）と同じ式・同じ核の作り方（G は線形なので G(ref) − G(out) = G(ref − out)）。
Stage 間で import しない流儀なので、同じ実装をここに複製してある。入れる場所は 3 つ: U の出力 μ = lock(f, x↑)、学習の正解 X′ = lock(X, x↑)、最終出力 y = lock(Z1, x↑)。
Gaussian は「2 回掛けても同じ」ではないので、lock の後の広い平均は ref と完全には一致しない（残り = (G − G∗G)∗(out − ref)。σ よりずっと広い成分はほぼ一致）。
"""

import math

import torch
import torch.nn.functional as F


def gaussian_kernel1d(sigma, device):
    radius = int(math.ceil(3.0 * float(sigma)))
    k = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    w = torch.exp(-0.5 * (k / float(sigma)) ** 2)
    return w / w.sum(), radius


def wide_mean(x, sigma_px):
    """x (B, 1, H, W) → Gaussian でぼかした FP32 の (B, 1, H, W)。σ ≤ 0 はそのまま。autocast の外で呼ぶこと。"""
    if x.dim() != 4 or x.shape[1] != 1:
        raise ValueError(f"(B, 1, H, W) だけ: {tuple(x.shape)}")
    x = x.float()
    if sigma_px <= 0:
        return x
    ker, r = gaussian_kernel1d(sigma_px, x.device)
    if r >= min(x.shape[-2:]):
        raise ValueError(f"核の半径 {r}（σ {sigma_px}）が画像 {tuple(x.shape[-2:])} 以上（reflect で広げられない）")
    y = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), ker.view(1, 1, 1, -1))
    return F.conv2d(F.pad(y, (0, 0, r, r), mode="reflect"), ker.view(1, 1, -1, 1))


def lock(out, ref, sigma_px):
    """out の広い平均を ref のものに差し替える。FP32。"""
    out = out.float()
    return out + wide_mean(ref.float() - out, sigma_px)


def center_crop(x, margin):
    return x[..., margin:x.shape[-2] - margin, margin:x.shape[-1] - margin] if margin > 0 else x
