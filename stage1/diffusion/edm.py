"""stage1/diffusion/edm.py — EDM 型の拡散（設計書 D-1）。式の出典と照合の記録は stage1/UPSTREAM.md。

  Karras, Aittala, Aila, Laine. "Elucidating the Design Space of Diffusion-Based Generative Models", NeurIPS 2022, arXiv:2206.00364
  公式実装 NVlabs/edm の training/networks.py（EDMPrecond）・training/loss.py（EDMLoss）・generate.py（edm_sampler）と 2026-11-11 に照合した。

  ノイズの足し方  x = x0 + σ·ε
  前処理          x̂0 = c_skip·x + c_out·F(c_in·x, …)
                  c_skip = σ_d² / (σ² + σ_d²),  c_out = σ·σ_d / √(σ² + σ_d²),  c_in = 1 / √(σ² + σ_d²),  c_noise = ln(σ) / 4
  損失            λ(σ)·(x̂0 − x0)²,  λ(σ) = (σ² + σ_d²) / (σ·σ_d)² = 1 / c_out²
                  = (F − 正解)²,  正解 = (x0 − c_skip·x) / c_out      （F の側で重み 1。ここではこの形で計算する）
  σ の引き方      公式は ln σ ~ N(−1.2, 1.2²)。ここは設計書の決定で ln σ を [ln σ_min, ln σ_max] で一様（loguniform）
  描くときの並び  σ_i = (σ_max^(1/ρ) + i/(N−1)·(σ_min^(1/ρ) − σ_max^(1/ρ)))^ρ（i = 0…N−1）、最後に 0
"""

import math

import torch

BIN_EDGES_HU = (2.0, 6.0, 20.0, 60.0, 200.0, 2000.0, 40000.0)  # 監視用の σ の帯の境（HU。設計には使わない。設計書 G-3）
N_BINS = len(BIN_EDGES_HU) - 1


def precond(sigma, sigma_data):
    """sigma (B,) FP32 → c_skip, c_out, c_in（各 (B, 1, 1, 1)）。"""
    s = sigma.float().view(-1, 1, 1, 1)
    d2 = s * s + sigma_data * sigma_data
    return sigma_data * sigma_data / d2, s * sigma_data / d2.sqrt(), 1.0 / d2.sqrt()


def draw_sigma(n, sigma_min, sigma_max, device, generator=None, sampling="loguniform", p_mean=None, p_std=None):
    """学習時の σ。loguniform: ln σ を [ln σ_min, ln σ_max] で一様。
    lognormal: ln σ ~ N(p_mean, p_std²)（EDM の標準。Karras et al. 2022 の P_mean = −1.2、P_std = 1.2）。[σ_min, σ_max] で切る（描くときの範囲の外は練習しない）。"""
    if sampling == "loguniform":
        u = torch.rand(n, device=device, generator=generator)
        return torch.exp(math.log(sigma_min) + u * (math.log(sigma_max) - math.log(sigma_min)))
    if sampling == "lognormal":
        z = torch.randn(n, device=device, generator=generator)
        return torch.exp(p_mean + p_std * z).clamp(sigma_min, sigma_max)
    raise ValueError(f"sigma_sampling: {sampling}")


def stratified_sigma(n, sigma_min, sigma_max, generator):
    """val 用: ln σ を n 等分した各区間の中央（どの帯にも必ず入る）を、順番だけ混ぜる。CPU。"""
    u = (torch.arange(n, dtype=torch.float32) + 0.5) / n
    s = torch.exp(math.log(sigma_min) + u * (math.log(sigma_max) - math.log(sigma_min)))
    return s[torch.randperm(n, generator=generator)]


def sigma_bin(sigma, hu_scale):
    """σ (B,) → 帯の番号 0…N_BINS−1（端の外は端の帯に入れる）。"""
    edges = torch.tensor(BIN_EDGES_HU[1:-1], device=sigma.device, dtype=torch.float32) / hu_scale
    return torch.bucketize(sigma.float(), edges)


def karras_sigmas(n, sigma_min, sigma_max, rho, device):
    """描くときの σ の並び（長さ n + 1。最後は 0）。float64 で計算して FP32 に。"""
    if n == 1:
        s = torch.tensor([sigma_max], dtype=torch.float64)
    else:
        i = torch.arange(n, dtype=torch.float64)
        s = (sigma_max ** (1 / rho) + i / (n - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    return torch.cat([s, torch.zeros(1, dtype=torch.float64)]).to(device=device, dtype=torch.float32)


def ancestral_step(sigma_from, sigma_to, eta):
    """Euler ancestral の 1 段の σ_down, σ_up（k-diffusion の get_ancestral_step と同じ式。設計書 F-1 の 5）。python の数で返す。"""
    if eta == 0 or sigma_to == 0:
        return sigma_to, 0.0
    up = min(sigma_to, eta * math.sqrt(sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2))
    return math.sqrt(sigma_to ** 2 - up ** 2), up
