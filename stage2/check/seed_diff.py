"""stage2/check/seed_diff.py — 乱数を変えた出力の差（設計書 G-3 の 4・5）。V が「構造」を作っていないかを見る。

  D = y_a − y_b（同じ入力、乱数の種だけ違う）
  fake_seed_sd_hu     D ÷ √2 の体内の標準偏差（1 回の出力が乱数で揺れる量）
  fake_seed_low_ratio D のパワーのうち 0.04 周期/画素より低い帯の割合
  fake_ref_low_ratio  同じ割合を X − μ（正解と U の出力の差）で出したもの（参照。正解があるスライスだけ）
  絵: [seed a | seed b | D（±display.diff_range_hu）| D を σ 3 でぼかしたもの（±display.diff_blur_range_hu）]
      D に線・点の形が出れば V が構造を作っている。形が無く砂嵐だけなら粒だけを作っている。
  seed を増やしたとき: 画素ごとの平均と標準偏差の絵。標準偏差の絵で、μ の縁から離れた所に線や点が光れば偽の構造。縁が光るのは縁の位置の揺れ。
"""

import math

import torch

from models.lock import wide_mean
from models.pipeline import cpu_noise
from util import metrics as M
from util.views import crop_zoom

BLUR_PX = 3.0


@torch.no_grad()
def sample_seeds(pipe, x, mu, case, name, seeds, sigma_s_hu, steps):
    """同じ x↑・μ から、種ごとの最終出力（n の (1, 1, H, W)）のリスト。"""
    return [pipe.flow(mu, x, cpu_noise(x.shape[-2:], s, case, name)[None, None], sigma_s_hu, steps)[0] for s in seeds]


def seed_stats(ya_hu, yb_hu, mask, tile, target_hu=None, mu_hu=None):
    d = ya_hu - yb_hu
    v = d[mask]
    out = {"fake_seed_sd_hu": float(v.std()) / math.sqrt(2.0) if v.numel() > 1 else float("nan"), "fake_seed_low_ratio": M.low_ratio(d, mask, tile)}
    if target_hu is not None and mu_hu is not None:
        out["fake_ref_low_ratio"] = M.low_ratio(target_hu - mu_hu, mask, tile)
    return out


def diff_row(view, ya_hu, yb_hu, center, size, scale, label):
    """切り抜きの 1 行: [seed a | seed b | D | ぼかした D]。戻り値 (cols, labels)。"""
    d = ya_hu - yb_hu
    cols = [view.img(ya_hu), view.img(yb_hu), view.diff(d), view.diff(wide_mean(d, BLUR_PX), view.blur_range)]
    labels = [f"{label} seed a", "seed b", f"a - b (+-{view.diff_range:g}HU)", f"blur s3 (+-{view.blur_range:g}HU)"]
    return [crop_zoom(c, center, size, scale) for c in cols], labels


def mean_std(ys_hu):
    """種ごとの出力（HU）→ 画素ごとの平均と標準偏差。"""
    s = torch.stack(ys_hu)
    return s.mean(dim=0), s.std(dim=0)
