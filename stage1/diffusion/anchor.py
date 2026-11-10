"""stage1/diffusion/anchor.py — 明るさの錨（設計書 F-2、RECONCILE #1・#2）。学習では使わない。描くときだけ。

  x̂0 ← x̂0 + (Gaussian(元の絵, σ_a) − Gaussian(x̂0, σ_a))

広い平均（σ_a より広い成分）を元の絵に合わせ直す、損失ではない決まった計算。分離型の Gaussian、端は reflect、FP32。
σ_a は configs/shared.yaml の brightness.sigma_px_512（512 の画素。Stage 2 は 1024 で 2 倍を使う。同じ実装を各 Stage に複製する流儀）。
"""

from diffusion.cue import gaussian_blur


def wide_mean(x, sigma_px):
    """広い平均 m = Gaussian(x, σ_a)。x (B, 1, H, W) FP32。"""
    return gaussian_blur(x.float(), sigma_px)


def anchor(x0_hat, m_src, sigma_px):
    """x̂0 の広い平均を m_src（= wide_mean(元の絵)）に合わせる。"""
    return x0_hat + (m_src - wide_mean(x0_hat, sigma_px))
