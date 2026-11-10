"""stage2/util/views.py — 表示用の変換（HU の tensor → パネルの列）。表示範囲は HU display.hu_min〜hu_max（−1400〜3000）、差の絵は白 = 0・青 −・赤 +。"""

import cv2
import numpy as np
import torch

from util.panel import gray
from util.residual_color import residual_rgb01


def to_np(x):
    """(1, 1, H, W) / (H, W) の tensor か numpy → float32 (H, W)。"""
    a = x.detach().float().cpu().numpy() if torch.is_tensor(x) else np.asarray(x, np.float32)
    return a.reshape(a.shape[-2], a.shape[-1])


class View:
    def __init__(self, data):
        self.lo, self.hi = data["display.hu_min"], data["display.hu_max"]
        self.diff_range, self.blur_range = data["display.diff_range_hu"], data["display.diff_blur_range_hu"]

    def img(self, hu):
        """HU → 表示用の RGB [0, 1]。"""
        return gray(np.clip((to_np(hu) - self.lo) / float(self.hi - self.lo), 0.0, 1.0))

    def diff(self, d_hu, range_hu=None):
        return residual_rgb01(to_np(d_hu), range_hu or self.diff_range)


def crop_box(center, size, hw):
    """中心 [y, x] と一辺 → 画像に収まるように寄せた左上 (y0, x0)。"""
    h, w = hw
    if size > min(h, w):
        raise ValueError(f"切り抜き {size} が画像 {hw} より大きい")
    y0 = min(max(int(center[0]) - size // 2, 0), h - size)
    x0 = min(max(int(center[1]) - size // 2, 0), w - size)
    return y0, x0


def crop_zoom(img, center, size, scale):
    """(H, W, 3) を中心 [y, x] で size 四方に切り、最近傍で scale 倍。"""
    y0, x0 = crop_box(center, size, img.shape[:2])
    c = img[y0:y0 + size, x0:x0 + size]
    return np.repeat(np.repeat(c, scale, axis=0), scale, axis=1)


def shrink(panel01, n):
    """パネルを 1/n に縮める（TensorBoard 用）。"""
    if n <= 1:
        return panel01
    h, w = panel01.shape[:2]
    return cv2.resize(panel01, (max(1, w // n), max(1, h // n)), interpolation=cv2.INTER_AREA)
