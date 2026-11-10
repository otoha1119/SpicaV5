"""stage1/diffusion/sampler.py — 描き方（設計書 F-1）。Euler ancestral（毎段に乱数、η）、Karras の配分、明るさの錨を毎段。

1 枚ぶんの手順（PCD でも実 EID でも同じ）:
  1. 元の絵 y（x の単位）
  2. 手がかり c = ぼかし(y, σ_b) + σ_c·n_c（n_c は 1 回だけ引いて全段で同じ）
  3. 広い平均 m = Gaussian(y, σ_a)
  4. 描き始め x = m + σ_0·ε
  5. 段 i = 0…N−1:
       x̂0 = D(x; σ_i, c, 度合い)
       錨: x̂0 ← x̂0 + (m − Gaussian(x̂0, σ_a))
       最後の段（σ_{i+1} = 0）なら x = x̂0 で終わり
       σ_up = η·√(σ_{i+1}²·(σ_i² − σ_{i+1}²) / σ_i²),  σ_down = √(σ_{i+1}² − σ_up²)
       x ← x̂0 + (σ_down / σ_i)·(x − x̂0) + σ_up·（新しい乱数）
     （k-diffusion の sample_euler_ancestral: d = (x − x̂0) / σ_i, x ← x + d·(σ_down − σ_i) + σ_up·ε と同じ式。出典は UPSTREAM.md）
  手がかり無し（has = False）: 手がかりは 0、錨なし、描き始めは σ_0·ε（診断用。設計書 E-3）

乱数は画像ごとの乱数源（util/seeds.py）から n_c → 描き始め → 各段の順に引く。batch の組み方に依らず同じ絵になる。
"""

import torch

from diffusion.anchor import anchor, wide_mean
from diffusion.cue import make_cue
from diffusion.edm import ancestral_step, karras_sigmas


def randn_each(gens, shape, device):
    """画像ごとの乱数源から 1 枚ずつ引いて積む。shape = (1, H, W)。"""
    return torch.stack([torch.randn(shape, generator=g, device=device) for g in gens])


@torch.no_grad()
def redraw(denoise, src, blur_px, noise_x, use_cue, gens, steps, eta, rho, sigma_min, sigma_max, anchor_px):
    """denoise(x, sigma (B,), cue, blur_px, noise_x, has) → x̂0（FP32）。src (B, 1, H, W)（x の単位。手がかり無しのときは形だけ使う）。
    blur_px / noise_x: (B,) tensor。use_cue: bool（batch 全体で同じ）。gens: 画像ごとの torch.Generator（長さ B）。戻り値 (描いた絵, 手がかり)。"""
    device = src.device
    b, _, h, w = src.shape
    shape = (1, h, w)
    has = torch.full((b,), bool(use_cue), device=device, dtype=torch.bool)
    if use_cue:
        cue = make_cue(src, blur_px, noise_x, randn_each(gens, shape, device))
        m = wide_mean(src, anchor_px)
    else:
        cue, m = torch.zeros_like(src), None
        blur_px, noise_x = torch.zeros_like(blur_px), torch.zeros_like(noise_x)
    sig = karras_sigmas(steps, sigma_min, sigma_max, rho, device)
    x = sig[0] * randn_each(gens, shape, device)
    if m is not None:
        x = x + m
    for i in range(steps):
        s, s_next = float(sig[i]), float(sig[i + 1])
        x0 = denoise(x, torch.full((b,), s, device=device), cue, blur_px, noise_x, has)
        if m is not None:
            x0 = anchor(x0, m, anchor_px)
        if s_next == 0:
            x = x0
            break
        down, up = ancestral_step(s, s_next, eta)
        x = x0 + (down / s) * (x - x0)
        if up > 0:
            x = x + up * randn_each(gens, shape, device)
    return x, cue
