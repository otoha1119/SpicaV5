"""stage1/diffusion/ode.py — 型 X（2 つの描き手を同じノイズでつなぐ）の行きと帰り。乱数を使わない確率流 ODE。出典は stage1/UPSTREAM.md。

  EDM（Karras et al. 2022, arXiv:2206.00364）の確率流 ODE:  dx/dσ = (x − D(x; σ)) / σ
  行き（絵 → ノイズ）: σ を σ_min → σ_max へ上げる。帰り（ノイズ → 絵）: σ_max → σ_min → 0 へ下げる。どちらも同じ式・同じ σ の並び（Karras の配分）。
  2 つの領域の拡散モデルを、片方で行き・もう片方で帰りにつなぐ形は DDIB（Su et al., ICLR 2023, arXiv:2203.08382）。

  1 段（σ → σ'）:  d = (x − D(x; σ)) / σ,  x' = x + (σ' − σ)·d                      （Euler）
                   d' = (x' − D(x'; σ')) / σ',  x' = x + (σ' − σ)·(d + d') / 2      （Heun。σ' = 0 の段は Euler）
  行きの始点は絵そのもの（σ_min のノイズは足さない）。条件の絵（手がかり）は行きと帰りで同じものを渡す。
  錨（広い平均の合わせ直し）は帰りの D の出力にだけ掛けられる（anchor_m を渡したとき）。
"""

import torch

from diffusion.anchor import anchor
from diffusion.edm import ancestral_step, karras_sigmas


def _slope(denoise, x, s, cond, anchor_m, anchor_px):
    b = x.shape[0]
    x0 = denoise(x, torch.full((b,), s, device=x.device), *cond)
    if anchor_m is not None:
        x0 = anchor(x0, anchor_m, anchor_px)
    return (x - x0) / s, x0


def _walk(denoise, x, sig, cond, heun, anchor_m, anchor_px, eta=0.0, gens=None):
    for i in range(len(sig) - 1):
        s, s2 = sig[i], sig[i + 1]
        d, x0 = _slope(denoise, x, s, cond, anchor_m, anchor_px)
        if s2 == 0:
            return x0
        if eta > 0:  # 帰りだけ: 各段で新しい乱数を混ぜる（Euler ancestral。sampler.py と同じ式）。Heun は使わない
            down, up = ancestral_step(s, s2, eta)
            x = x0 + (down / s) * (x - x0) + up * torch.stack([torch.randn(x.shape[1:], generator=g, device=x.device) for g in gens])
            continue
        x_e = x + (s2 - s) * d
        if heun:
            d2, _ = _slope(denoise, x_e, s2, cond, anchor_m, anchor_px)
            x = x + (s2 - s) * 0.5 * (d + d2)
        else:
            x = x_e
    return x


@torch.no_grad()
def encode(denoise, src, cond, steps, rho, sigma_min, sigma_max, heun):
    """絵 → ノイズ z（σ_max の水準）。cond = (cue, blur_px, noise_x, has)。"""
    down = karras_sigmas(steps, sigma_min, sigma_max, rho, src.device)[:-1].tolist()  # σ_max … σ_min
    return _walk(denoise, src.float(), down[::-1], cond, heun, None, 0.0)


@torch.no_grad()
def decode(denoise, z, cond, steps, rho, sigma_min, sigma_max, heun, anchor_m=None, anchor_px=0.0, eta=0.0, gens=None):
    """ノイズ z → 絵。eta > 0 なら各段で新しい乱数を混ぜる（0 = 乱数なしの ODE。1 = 描き手の普通の描き方と同じ混ぜ方で、z の情報は早い段で失われる）。"""
    down = karras_sigmas(steps, sigma_min, sigma_max, rho, z.device).tolist()  # σ_max … σ_min, 0
    return _walk(denoise, z, down, cond, heun, anchor_m, anchor_px, eta=eta, gens=gens)
