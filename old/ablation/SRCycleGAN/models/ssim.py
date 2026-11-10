"""ablation/SRCycleGAN/models/ssim.py — SSIM 損失。著者公開コードの pytorch_ssim/__init__.py（SpicaV2 commit 741924c）からの移植。

論文 Eq. 3（JMI）/ Eq. 1（SPIE）: SSIM = ((2 μx μy + C1)(2 σxy + C2)) / ((μx² + μy² + C1)(σx² + σy² + C2))、L_S = 1 − SSIM（画像平均）。
移植元との差: C1 / C2 を引数にした（741924c は 0.01² / 0.03² 固定。SPIE 版は "C1 = 0.02 and C2 = 0.06" → ユーザー決定 2026-10-27 でこちら）。
Gaussian window（一辺 window_size、σ 1.5）と、同じ画像を 2 乗して局所統計を取る計算はそのまま。
"""

from math import exp

import torch
import torch.nn.functional as F


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    return _2D_window.expand(channel, 1, window_size, window_size).contiguous()


def _ssim(img1, img2, window, window_size, channel, C1, C2, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    if size_average:
        return ssim_map.mean()
    return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, C1=0.01**2, C2=0.03**2, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.C1, self.C2 = float(C1), float(C2)
        self.channel = 1
        self.register_buffer("window", create_window(window_size, self.channel))

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        window = self.window
        if channel != self.channel or window.dtype != img1.dtype or window.device != img1.device:
            window = create_window(self.window_size, channel).to(device=img1.device, dtype=img1.dtype)
        return _ssim(img1, img2, window, self.window_size, channel, self.C1, self.C2, self.size_average)
