"""eid_noise.py — [SpicaV5] Stage 1 の「解剖の平均画像 μ + 実 EID の NPS を持つ色付きノイズ」（2026-11-05、ブランチ 020、EXP-20261105-01）。

ỹ = μ_θ(s̃) + c(ξ)。s̃ = BM3D(x)（PCD のノイズを除いた解剖の推定、前計算）、μ_θ = 既存の G（決定的、乱数を見ない）、
c(ξ) = IFFT_ortho( amp · FFT_ortho(ξ) )、ξ ~ N(0, 1)。amp は実 EID（B 側 = ImageCAS）の一様部 128 patch の 2D NPS から作り、
spectral 損失と同じ測り方（128 patch、σ6 トレンド除去、鏡の折り返し、帯域の和、正規化空間）で帯域ごとの倍率を校正する（Codex レビュー 2 必須 3）。
乱数はノイズ側にしか入らないので、乱数で解剖は動かない（必須 7）。ノイズは 512² で一度に作って切り出す（patch の継ぎ目・周期の跡なし）。
"""

import glob
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

SIZE = 512
EDGES = (0.03, 0.08, 0.15, 0.25, 0.71)


def _uniform_patches(eid_dir, offset, n_cases, per_case, per_slice, patch=128, seed=0):
    """実 EID の一様部（σ3 平滑の平均 −150〜500 HU、SD < 25）の 128 patch（HU）。"""
    rng = np.random.default_rng(seed); out = []
    cases = sorted(c for c in glob.glob(str(Path(eid_dir) / "*")) if Path(c).is_dir())[:n_cases]
    for c in cases:
        fs = sorted(glob.glob(c + "/*.png"))
        for q in np.linspace(0.2, 0.8, per_case):
            h = cv2.imread(fs[int(len(fs) * q)], cv2.IMREAD_UNCHANGED).astype(np.float32) - offset
            sm = cv2.GaussianBlur(h, (0, 0), 3); got = tries = 0
            while got < per_slice and tries < 4000:
                tries += 1; py, px = rng.integers(0, h.shape[0] - patch, 2); w = sm[py:py + patch, px:px + patch]
                if -150 < w.mean() < 500 and w.std() < 25:
                    out.append(h[py:py + patch, px:px + patch]); got += 1
    if len(out) < 40:
        raise RuntimeError(f"実 EID の一様 patch が少なすぎます（{len(out)} 個）: {eid_dir}")
    return np.stack(out)


def build_filter(eid_dir, offset, hu_min, hu_max, n_cases=120, per_case=4, per_slice=3, iters=4):
    """amp（SIZE², 正規化空間の振幅、正規直交 FFT 用）と report を返す。"""
    from scipy.interpolate import RegularGridInterpolator
    from .spectral_loss import band_power
    sc = 2.0 / (hu_max - hu_min)
    uni = _uniform_patches(eid_dir, offset, n_cases, per_case, per_slice)
    P = uni.shape[-1]
    win = np.outer(np.hanning(P), np.hanning(P)); win /= np.sqrt((win ** 2).mean())
    acc = np.zeros((P, P))
    for a in uni:
        acc += np.abs(np.fft.fft2((a - cv2.GaussianBlur(a, (0, 0), 6)) * win, norm="ortho")) ** 2
    n = acc / len(uni)
    n = 0.5 * (n + np.roll(np.roll(n[::-1, ::-1], 1, 0), 1, 1))  # 中心対称（実数画像の NPS が満たす対称性）だけで平均
    f_p = np.fft.fftshift(np.fft.fftfreq(P)); f_s = np.fft.fftfreq(SIZE)
    itp = RegularGridInterpolator((f_p, f_p), np.fft.fftshift(n), bounds_error=False, fill_value=None)
    FY, FX = np.meshgrid(f_s, f_s, indexing="ij"); FR = np.hypot(FY, FX)
    amp = np.sqrt(np.maximum(itp(np.stack([FY, FX], -1)), 0)) * (1 - np.exp(-(FR / 0.01) ** 4)) * sc   # DC と 0.01 cycles/px 以下を滑らかに抑える
    amp[0, 0] = 0.0
    masks = [(FR >= lo) & (FR < hi) for lo, hi in zip(EDGES[:-1], EDGES[1:])]
    target = band_power(torch.from_numpy((uni * sc).astype(np.float32))[:, None], list(EDGES), window="mirror").numpy()
    gains, hist = np.ones(len(masks)), []
    for it in range(iters):
        g = np.select(masks, gains, 1.0)
        gen = [_make(amp * g, np.random.default_rng(1000 + s)) for s in range(16)]
        t = torch.from_numpy(np.stack([im[y:y + P, x:x + P] for im in gen for y in range(0, SIZE - P + 1, P) for x in range(0, SIZE - P + 1, P)]))[:, None]
        got = band_power(t, list(EDGES), window="mirror").numpy()
        hist.append({"iter": it, "gains": gains.tolist(), "ratio": (got / target).tolist()})
        gains = gains * np.sqrt(target / got)
    amp = (amp * np.select(masks, gains, 1.0)).astype(np.float32)
    # 返す amp そのものの検証（64 枚、別の乱数）: 帯域パワー比、場の SD、局所 SD（σ3 の外）、極低域（σ16）の SD
    ver = [_make(amp, np.random.default_rng(5000 + s)) for s in range(64)]
    t = torch.from_numpy(np.stack([im[y:y + P, x:x + P] for im in ver for y in range(0, SIZE - P + 1, P) for x in range(0, SIZE - P + 1, P)]))[:, None]
    final_ratio = (band_power(t, list(EDGES), window="mirror").numpy() / target).tolist()
    eid_local = float(np.median([(a - cv2.GaussianBlur(a, (0, 0), 3)).std() for a in uni]))
    report = {"eid_dir": str(eid_dir), "n_uniform_patches": int(len(uni)), "target_band_power": target.tolist(), "calibration": hist, "final_gains": gains.tolist(),
              "verify_64": {"band_ratio": final_ratio, "sd_HU": float(np.mean([v.std() for v in ver]) / sc),
                            "local_sd_HU": float(np.mean([(v - cv2.GaussianBlur(v, (0, 0), 3)).std() for v in ver]) / sc), "eid_uniform_local_sd_HU": eid_local / 1.0,
                            "lowpass_sigma16_sd_HU": float(np.mean([cv2.GaussianBlur(v, (0, 0), 16).std() for v in ver]) / sc)},
              "note": "形は Hann 窓の 2D NPS、強さは mirror の帯域パワーで 4 帯域ごとに校正（帯域内の 2D の形・角度は Hann の推定のまま）"}
    return amp, report


def _make(amp, rng):
    xi = rng.standard_normal(amp.shape)
    return np.real(np.fft.ifft2(amp * np.fft.fft2(xi, norm="ortho"), norm="ortho")).astype(np.float32)


class MeanNoiseG(nn.Module):
    """G を包む: forward(x) = net(x) + c(ξ)。self.net が μ_θ（損失は net の出力 μ に掛ける）。amp は buffer（net_G.pth に入るので推論でも同じノイズの形）。
    noise=False にすると μ だけを返す（監視用）。乱数は torch の乱数（推論で再現したいときは torch.manual_seed をスライスごとに）。"""

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.register_buffer("amp", torch.zeros(SIZE, SIZE))
        self.noise = True
        self.generator = None   # forward で使う torch.Generator（学習は model が専用のものを入れる）
        self.seeds = None       # forward で使うサンプルごとの seed のリスト（監視・採点・推論。指定すると generator より優先）

    def noise_like(self, t, generator=None, seeds=None):
        """t と同じ形のノイズ。seeds（長さ N）があればサンプルごとに独立な generator、無ければ generator（None なら torch の既定の乱数）。"""
        n, c, h, w = t.shape
        if h > SIZE or w > SIZE:
            raise ValueError(f"画像 {h}×{w} は {SIZE} を超えています（ノイズの格子は {SIZE}²）")
        if seeds is not None:
            if len(seeds) != n:
                raise ValueError(f"seeds の数 {len(seeds)} とバッチ {n} が違います")
            return torch.cat([self.noise_like(t[i:i + 1], generator=torch.Generator(device=t.device).manual_seed(int(sd))) for i, sd in enumerate(seeds)])
        xi = torch.randn(n, c, SIZE, SIZE, device=t.device, dtype=torch.float32, generator=generator)
        f = torch.fft.ifft2(torch.fft.fft2(xi, norm="ortho") * self.amp, norm="ortho").real
        if h == SIZE and w == SIZE:
            return f.to(t.dtype)
        off = torch.randint(0, SIZE - max(h, w) + 1, (2,), device=t.device, generator=generator).tolist()
        return f[..., off[0]:off[0] + h, off[1]:off[1] + w].to(t.dtype)

    def forward(self, x):
        mu = self.net(x)
        return mu + self.noise_like(mu, generator=self.generator, seeds=self.seeds) if self.noise else mu


def slice_seed(key, salt=0):
    """スライスの識別子（症例/スライス名など）から決まる seed（推論・監視で同じスライスは毎回同じノイズ）。"""
    import zlib
    return (zlib.crc32(str(key).encode("utf-8")) ^ (salt * 0x9E3779B1)) & 0x7FFFFFFF


def save_report(run_dir, report):
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(run_dir) / "eid_noise_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
