"""stage2/data/augment.py — Stage 2 の学習入力（EID-like1024 patch）だけに掛ける augmentation（教師は触らない。2026-11-08、ブランチ 022）。

動機: docs/decisions/20261107_mechanism-analysis.md §5（欠陥 D1 = 入力 u が教師 x1024 の実現値を持つ「漏れ」、D2 = 学習分布が 1 点）。
Stage 1 は決定的（R-S10）で教師と同じ生データから作るので、損失を変えても u の高域には x1024 の実現値が残る。
Stage 2 側の学習入力でその高域を毎回作り直せば、Stage 2 は高域から教師の実現値を読み戻せなくなる（§7 の (A′)）。

  hf_phase_randomize(img, cutoff, rng): cutoff [cycles/px] 以上の帯域の**位相だけ**を一様乱数に置き換える（振幅スペクトルは保つ = ノイズ量・NPS はそのまま、位置情報だけ壊す）。
      rfft2 → 半平面に乱数位相 → irfft2 なので出力は実数でエルミート対称は自動（DC・Nyquist 列は位相を変えない）。帯域の境界は cutoff の ±10 % の raised cosine。
  random_blur(img, sigma, ...): σ [px] の Gaussian（cv2、BORDER_REFLECT）。σ は呼び出し側が U[0, max] で引く（D2: 入力のぼけ量を族にする、R-S26）。
  augment_input(ctx, rng, blur_sigma_max, hf_cutoff): 上の 2 つを「patch の 2 倍の文脈」に掛けてから中央を切る（FFT の周期境界・ぼかしの端を patch の外に出す）。
      順序は ぼかし → 位相ランダム化。どちらも 0.0 なら何もしない（main と同じ）。

すべて numpy / cv2（DataLoader の worker で動く）。正規化空間 [−1,1] でも HU でも線形なので同じ。
"""

import cv2
import numpy as np


def _radial_mask(shape_rfft, h, w, cutoff, width_frac=0.1):
    """rfft2 の格子 (h, w//2+1) 上で、|f| ≥ cutoff を 1、cutoff·(1−width_frac) 以下を 0、間を raised cosine でつなぐ重み。"""
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.rfftfreq(w)[None, :]
    fr = np.hypot(fy, fx)
    lo, hi = cutoff * (1.0 - width_frac), cutoff
    m = np.clip((fr - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    m = 0.5 - 0.5 * np.cos(np.pi * m)
    assert m.shape == shape_rfft
    return m.astype(np.float32)


def hf_phase_randomize(img, cutoff, rng):
    """img (H, W) float32 の cutoff [cycles/px] 以上の成分の位相を一様乱数にする（振幅は保存）。rng は numpy Generator。"""
    h, w = img.shape
    F = np.fft.rfft2(img.astype(np.float32))
    m = _radial_mask(F.shape, h, w, cutoff)
    phase = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, F.shape)).astype(np.complex64)
    # 実数出力のために irfft2 が暗黙に使う列（DC 列と、w が偶数なら Nyquist 列）は自己共役なので位相を触らない
    phase[:, 0] = 1.0
    if w % 2 == 0:
        phase[:, -1] = 1.0
    Fr = np.abs(F) * phase
    out = F * (1.0 - m) + Fr * m
    return np.fft.irfft2(out, s=(h, w)).astype(np.float32)


def random_blur(img, sigma):
    """σ [px] の Gaussian ぼかし。σ が 0.05 未満なら何もしない。"""
    if sigma < 0.05:
        return img
    return cv2.GaussianBlur(img, (0, 0), float(sigma), borderType=cv2.BORDER_REFLECT)


def augment_input(ctx, rng, blur_sigma_max, hf_cutoff):
    """文脈 ctx (C, C) float32（patch の 2 倍。足りないときは画像端で clamp したもの）に augmentation を掛けて返す（切り出しは呼び出し側）。
    rng は numpy Generator。戻り値 (C, C) float32。"""
    out = ctx
    if blur_sigma_max > 0.0:
        out = random_blur(out, float(rng.uniform(0.0, blur_sigma_max)))
    if hf_cutoff > 0.0:
        out = hf_phase_randomize(out, hf_cutoff, rng)
    return out


def context_box(pos, patch, img_hw, factor=2):
    """patch の左上 pos=(y, x) を中心にした factor 倍の文脈の (y0, x0, size) と、その中での patch の相対位置 (dy, dx)。画像端では箱をずらす（大きさは保つ）。"""
    h, w = img_hw
    size = min(patch * factor, h, w)
    y0 = min(max(pos[0] - (size - patch) // 2, 0), h - size)
    x0 = min(max(pos[1] - (size - patch) // 2, 0), w - size)
    return y0, x0, size, pos[0] - y0, pos[1] - x0
