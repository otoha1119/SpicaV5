"""util/residual_color.py — 残差 R = G(z) − z の表示用カラー化。[SpicaV5]

配色（2026-09-08 ユーザー確定）: 白 = 0 HU（変化なし）、純青 (0,0,255) = −range_hu（G が HU を下げた）、純赤 (255,0,0) = +range_hu（上げた）。
白から純色への線形補間で、|ΔHU| ≥ range_hu は端の色で飽和する。range_hu は train.yaml log.diff_range_hu（推論は run の launch.yaml から同じ値）。

使う所（表示専用。16bit の R（0 HU = 32768、util/monitor.py / inference_dir.py）はデータで、こちらは置き換えない）:
  util/monitor.py   TensorBoard images/current, images/fixed の差分列、images/full/* と output_images/preview_*/ の R 列
  inference_dir.py  <out_dir>/{full,patch}_R_color/<case>/<slice>.png（8bit RGB、512×512 のまま）と <out_dir>/R_colorbar_pm<range>HU.png

戻り値はすべて RGB。cv2.imwrite に渡すときは rgb_to_bgr() を通す。
"""

import numpy as np


def residual_rgb01(delta_hu, range_hu):
    """ΔHU (H,W) → RGB float32 [0,1] (H,W,3)。負: 白→純青、正: 白→純赤、|ΔHU| ≥ range_hu で飽和。"""
    if range_hu <= 0:
        raise ValueError(f"range_hu は正: {range_hu}")
    t = np.clip(np.asarray(delta_hu, dtype=np.float32) / float(range_hu), -1.0, 1.0)
    m = np.abs(t)
    neg = t < 0
    rgb = np.empty(t.shape + (3,), np.float32)
    rgb[..., 0] = np.where(neg, 1.0 - m, 1.0)  # R: 負では落ちる、正では 1
    rgb[..., 1] = 1.0 - m                      # G: どちらでも落ちる
    rgb[..., 2] = np.where(neg, 1.0, 1.0 - m)  # B: 負では 1、正では落ちる
    return rgb


def residual_rgb8(delta_hu, range_hu):
    """ΔHU → RGB uint8 (H,W,3)。"""
    return (residual_rgb01(delta_hu, range_hu) * 255.0).round().astype(np.uint8)


def colorbar_rgb01(range_hu, width, height=10, margin=5):
    """凡例: 左 −range_hu … 中央 白 … 右 +range_hu の横一列。周囲 margin は白。(height + 2*margin, width, 3) float32 [0,1]。"""
    t = np.linspace(-range_hu, range_hu, width, dtype=np.float32)[None, :].repeat(height, 0)
    out = np.ones((height + 2 * margin, width, 3), np.float32)
    out[margin : margin + height] = residual_rgb01(t, range_hu)
    return out


def colorbar_filename(range_hu):
    """凡例ファイル名。範囲を名前に入れる（推論出力の <out_dir>/ 直下に 1 枚）。"""
    return f"R_colorbar_pm{int(range_hu)}HU.png"


def rgb_to_bgr(arr):
    """cv2.imwrite 用（OpenCV は BGR）。"""
    return np.ascontiguousarray(arr[..., ::-1])
