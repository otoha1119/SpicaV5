"""stage2/models/haar.py — V の入口と出口の畳み方（設計書 A-2 (i)・C-2。stage1/models/haar.py から複製。Stage 間で import しない）。Haar 1 段の直交変換: (B, C, H, W) ↔ (B, 4C, H/2, W/2)。

2×2 の画素 [[a, b], [c, d]] を「平均 1 本 + 差 3 本」にする（係数 1/2 で直交。情報は減らず、白色ノイズは白色のまま）:
  LL = (a + b + c + d) / 2,  LH = (a − b + c − d) / 2,  HL = (a + b − c − d) / 2,  HH = (a − b − c + d) / 2
逆変換は同じ行列（対称で自己逆）。PyWavelets は使わない（イメージに入っていない）。計算は FP32 で行う（呼び出し側が autocast の外で呼ぶ）。
"""

import torch
import torch.nn.functional as F


def haar_fold(x):
    """(B, C, H, W) → (B, 4C, H/2, W/2)。チャネルの並びは [LL…, LH…, HL…, HH…]。"""
    a, b = x[..., 0::2, 0::2], x[..., 0::2, 1::2]
    c, d = x[..., 1::2, 0::2], x[..., 1::2, 1::2]
    return torch.cat([(a + b + c + d), (a - b + c - d), (a + b - c - d), (a - b - c + d)], dim=1) * 0.5


def haar_unfold(y):
    """(B, 4C, H, W) → (B, C, 2H, 2W)。haar_fold の逆。"""
    ll, lh, hl, hh = torch.chunk(y, 4, dim=1)
    a = (ll + lh + hl + hh) * 0.5
    b = (ll - lh + hl - hh) * 0.5
    c = (ll + lh - hl - hh) * 0.5
    d = (ll - lh - hl + hh) * 0.5
    n, ch, h, w = a.shape
    # pixel_shuffle の並び: 出力チャネル k の 4 枚が [(0,0), (0,1), (1,0), (1,1)] の位置に入る
    return F.pixel_shuffle(torch.stack([a, b, c, d], dim=2).reshape(n, ch * 4, h, w), 2)
