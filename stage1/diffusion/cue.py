"""stage1/diffusion/cue.py — 手がかりの作り方（設計書 E）。学習でも、描くときでも、同じ関数を通す。

  手がかり = Gaussian ぼかし(元の絵, σ_b) + σ_c × 白色ノイズ          （ぼかした後に足す）

  ぼかし: 半径 ceil(3σ) で打ち切って正規化した 1 次元の核を、横 → 縦に掛ける（分離型）。端は reflect。σ = 0 は何もしない（核が中央 1 点）。
          サンプルごとに違う σ を、サンプルごとの核 + グループ畳み込みで 1 回で掛ける。FFT は使わない。FP32・compile の外。
  くじ : 設計書 E-2。サンプルごとに独立。手がかりを隠す drop_prob、σ_b は blur_zero_prob で 0・残りは [min, max] 画素を対数一様、σ_c も同じ形（HU）。
"""

import math

import torch
import torch.nn.functional as F


def gaussian_kernel1d(sigma, radius):
    """sigma (B,) FP32 → (B, 2·radius + 1)。各行は半径 ceil(3σ) で打ち切って正規化。σ = 0 の行は中央だけ 1。"""
    k = torch.arange(-radius, radius + 1, device=sigma.device, dtype=torch.float32)[None, :]
    s = sigma.float()[:, None]
    w = torch.exp(-0.5 * (k / s.clamp_min(1e-6)) ** 2)
    w = torch.where(k.abs() <= torch.ceil(3.0 * s), w, torch.zeros_like(w))
    w = torch.where(s > 0, w, (k == 0).float().expand_as(w))
    return w / w.sum(dim=1, keepdim=True)


def gaussian_blur(x, sigma):
    """x (B, 1, H, W) を、サンプルごとの σ（(B,) の tensor か 1 つの数。画素）でぼかす。端は reflect。"""
    b, c, h, w = x.shape
    if c != 1:
        raise ValueError(f"1ch だけ: {tuple(x.shape)}")
    if not torch.is_tensor(sigma):
        sigma = torch.full((b,), float(sigma), device=x.device)
    sigma = sigma.to(device=x.device, dtype=torch.float32)
    radius = int(math.ceil(3.0 * float(sigma.max())))
    if radius == 0:
        return x
    if radius >= min(h, w):
        raise ValueError(f"ぼかしの半径 {radius} が画像 {(h, w)} 以上（reflect で広げられない）")
    ker = gaussian_kernel1d(sigma, radius)
    y = x.float().reshape(1, b, h, w)
    y = F.conv2d(F.pad(y, (radius, radius, 0, 0), mode="reflect"), ker[:, None, None, :], groups=b)
    y = F.conv2d(F.pad(y, (0, 0, radius, radius), mode="reflect"), ker[:, None, :, None], groups=b)
    return y.reshape(b, 1, h, w)


def _rand(n, device, generator):
    return torch.rand(n, device=device, generator=generator)


def _loguniform(u, lo, hi):
    return torch.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))


def draw_cue_params(n, train, device, generator=None):
    """学習時のくじ。戻り値: blur_px (n,)、noise_x (n,)（σ_c を x の単位で）、has (n,) bool。手がかり無しの行は σ_b = σ_c = 0。"""
    has = _rand(n, device, generator) >= train["cue.drop_prob"]
    zero = torch.zeros(n, device=device)
    blur = torch.where(_rand(n, device, generator) < train["cue.blur_zero_prob"], zero,
                       _loguniform(_rand(n, device, generator), train["cue.blur_min_px"], train["cue.blur_max_px"]))
    noise_hu = torch.where(_rand(n, device, generator) < train["cue.noise_zero_prob"], zero,
                           _loguniform(_rand(n, device, generator), train["cue.noise_min_hu"], train["cue.noise_max_hu"]))
    return torch.where(has, blur, zero), torch.where(has, noise_hu / train["data.hu_scale"], zero), has


def make_cue(src, blur_px, noise_x, noise):
    """src (B, 1, H, W)（x の単位）→ 手がかり。noise は src と同じ形の白色 Gaussian（呼び出し側が引く。描くときは全段で同じものを使う）。"""
    return gaussian_blur(src, blur_px) + noise_x.float()[:, None, None, None] * noise


def center_crop(x, margin):
    return x[..., margin:x.shape[-2] - margin, margin:x.shape[-1] - margin] if margin > 0 else x


CUE_KINDS = ("none", "clean", "blur_only", "noise_only", "both")  # loss_cue/* の区分。clean = 手がかりあり・σ_b = σ_c = 0（元の絵そのまま）


def cue_kind(blur_px, noise_x, has):
    """(B,) → CUE_KINDS の番号。"""
    kind = 1 + (blur_px > 0).long() + 2 * (noise_x > 0).long()  # 1 clean / 2 blur_only / 3 noise_only / 4 both
    return kind * has.long()
