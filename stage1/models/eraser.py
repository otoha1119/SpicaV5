"""stage1/models/eraser.py — 型 Y の「消し手」A: 近くの画素の重みつき平均しかできない小さな CNN（2026-11-11）。

  出力 m(p) = Σ_{q ∈ 近傍(p)} w_p(q) · x(q),  w_p = softmax（画素ごとに足して 1、すべて 0 以上）
  作りで保証されること:
    - 一様な領域の値は変わらない（明るさ・造影の濃さを変えられない）
    - 近傍の最小〜最大を超えない（新しい極値 = 入力に無い構造を作れない、HU の上振れが出ない）
    - 鋭くはできない（平均だけ）
  重みは場所と中身で変わる（縁は狭く、平らな所は広く平均できる）。近傍の広さ（radius）は上限であって、ぼかし量ではない。
  先例: Kernel Prediction Networks（Mildenhall et al., CVPR 2018、画素ごとの重みを予測して平均する）。

  初期値: 最後の Conv の重みを 0、中央の画素の logit だけ center_bias にする（最初は恒等に近い。学習が平均を増やす）。
  入力 x は HU ÷ 1000 の単位。端は reflect。FP32 で計算する（小さいので速度の問題はない）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Block(nn.Module):
    def __init__(self, ch, dilation):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=dilation, dilation=dilation, padding_mode="reflect")
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, padding_mode="reflect")

    def forward(self, h):
        return h + self.c2(F.silu(self.c1(F.silu(h))))


class Eraser(nn.Module):
    def __init__(self, radius, width, depth, center_bias):
        super().__init__()
        self.radius = int(radius)
        k = 2 * self.radius + 1
        self.k2 = k * k
        self.stem = nn.Conv2d(1, width, 3, padding=1, padding_mode="reflect")
        self.blocks = nn.Sequential(*[_Block(width, 2 ** (i % 4)) for i in range(depth)])
        self.out = nn.Conv2d(width, self.k2, 1)
        nn.init.zeros_(self.out.weight)
        with torch.no_grad():
            self.out.bias.zero_()
            self.out.bias[self.k2 // 2] = float(center_bias)

    def weights(self, x):
        """x (B, 1, H, W) → 近傍の重み (B, k², H, W)（softmax 済み）。"""
        return torch.softmax(self.out(F.silu(self.blocks(self.stem(x.float())))), dim=1)

    def forward(self, x, return_weights=False):
        x = x.float()
        b, _, h, w = x.shape
        wts = self.weights(x)
        r = self.radius
        patches = F.unfold(F.pad(x, (r, r, r, r), mode="reflect"), kernel_size=2 * r + 1).view(b, self.k2, h, w)
        m = (wts * patches).sum(dim=1, keepdim=True)
        return (m, wts) if return_weights else m


def build_eraser(cfg):
    return Eraser(cfg["eraser.radius"], cfg["eraser.width"], cfg["eraser.depth"], cfg["eraser.center_bias"])
