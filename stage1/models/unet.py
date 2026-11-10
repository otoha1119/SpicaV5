"""stage1/models/unet.py — 描き手 P の U-Net（設計書 C-1）。畳み込みだけ（注意機構なし・位置の符号なし。パディングは zeros）。

  PainterNet = embed（度合いの埋め込み、FP32）+ stem（Haar の折り畳み + 最初の Conv、FP32）+ body（U-Net 本体。ここだけ bf16 autocast / torch.compile の対象）
               + head（GroupNorm → SiLU → Conv（重み 0）→ 逆 Haar、FP32）
  3 つに分けてあるのは設計書 C-4 のため: 入口（c_in·x の細かい差）と出口（c_out·F の合成）を bf16 で丸めない。autocast の切り替えを compile する範囲の外に出す。

  body: 4 段（base_ch × ch_mult）。下りは各段 ResBlock × blocks_per_level、段の間は stride 2 の Conv3×3。底に ResBlock × mid_blocks。
        上りは各段 ResBlock × (blocks_per_level + 1)（skip を連結）、段の間は最近傍 ×2 + Conv3×3。
  ResBlock: GroupNorm → SiLU → Conv3×3 → GroupNorm → 埋め込みの scale / shift（FiLM）→ SiLU → Conv3×3（重み 0 で初期化）。幅が変わる所の近道は 1×1 conv。
            設計書の表は「Conv → FiLM → GroupNorm」の順に書いてあるが、FiLM の直後に GroupNorm を置くと scale / shift の大半が正規化で消えるので、
            定番（ADM の scale-shift norm: GroupNorm の後に掛ける）の順にした。UPSTREAM.md に記録。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.embed import CondEmbed
from models.haar import haar_fold, haar_unfold


def zero_(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class ResBlock(nn.Module):
    def __init__(self, cin, cout, emb_dim, groups):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.emb = nn.Linear(emb_dim, 2 * cout)
        self.norm2 = nn.GroupNorm(groups, cout)
        self.conv2 = zero_(nn.Conv2d(cout, cout, 3, padding=1))
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.emb(emb)[:, :, None, None].chunk(2, dim=1)
        h = self.norm2(h) * (1.0 + scale) + shift
        h = self.conv2(F.silu(h))
        return self.skip(x) + h


class Down(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x, emb):
        return self.conv(x)


class Up(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x, emb):
        # 最近傍 ×2。F.interpolate は使わない（bf16 の最近傍補間は torch の版によって CUDA で未実装だったことがある。形の操作だけなら dtype に依らない）
        n, c, h, w = x.shape
        return self.conv(x[:, :, :, None, :, None].expand(n, c, h, 2, w, 2).reshape(n, c, 2 * h, 2 * w))


class UNetBody(nn.Module):
    """stem の出力 (B, ch0, h, w) と埋め込み (B, D) → (B, ch0, h, w)。bf16 autocast と torch.compile はこのモジュールにだけ掛ける。"""

    def __init__(self, model, fp32_top_level):
        super().__init__()
        chs = [model["base_ch"] * m for m in model["ch_mult"]]
        nb, D, G = model["blocks_per_level"], model["emb_dim"], model["groups"]
        self.fp32_top = bool(fp32_top_level)
        self.down, self.down_top = nn.ModuleList(), []
        skip_chs, c = [chs[0]], chs[0]
        for i, ch in enumerate(chs):
            for _ in range(nb):
                self.down.append(ResBlock(c, ch, D, G)); self.down_top.append(i == 0)
                c = ch
                skip_chs.append(c)
            if i < len(chs) - 1:
                self.down.append(Down(c)); self.down_top.append(False)
                skip_chs.append(c)
        self.mid = nn.ModuleList([ResBlock(c, c, D, G) for _ in range(model["mid_blocks"])])
        self.up, self.up_top, self.up_cat = nn.ModuleList(), [], []
        for i in reversed(range(len(chs))):
            for _ in range(nb + 1):
                self.up.append(ResBlock(c + skip_chs.pop(), chs[i], D, G)); self.up_top.append(i == 0); self.up_cat.append(True)
                c = chs[i]
            if i > 0:
                self.up.append(Up(c)); self.up_top.append(False); self.up_cat.append(False)
        assert not skip_chs

    def _run(self, blk, h, emb, top):
        if top and self.fp32_top:  # optim.fp32_top_level: 最上段だけ FP32（既定は false。true のときだけ autocast の切り替えが body の中に入る）
            with torch.autocast(device_type=h.device.type, enabled=False):
                return blk(h.float(), emb.float())
        return blk(h, emb)

    def forward(self, h, emb):
        skips = [h]
        for blk, top in zip(self.down, self.down_top):
            h = self._run(blk, h, emb, top)
            skips.append(h)
        for blk in self.mid:
            h = blk(h, emb)
        for blk, top, cat in zip(self.up, self.up_top, self.up_cat):
            if cat:
                h = torch.cat([h, skips.pop()], dim=1)
            h = self._run(blk, h, emb, top)
        return h


class Stem(nn.Module):
    """[c_in·x, 手がかり]（画素の側で 1ch ずつ）→ それぞれ Haar 1 段 → 連結 8ch → Conv3×3。FP32。"""

    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(8, ch, 3, padding=1)

    def forward(self, x_in, cue):
        return self.conv(torch.cat([haar_fold(x_in), haar_fold(cue)], dim=1))


class Head(nn.Module):
    """GroupNorm → SiLU → Conv3×3（重み 0）→ 4ch → 逆 Haar → 1ch（F）。FP32。"""

    def __init__(self, ch, groups):
        super().__init__()
        self.norm = nn.GroupNorm(groups, ch)
        self.conv = zero_(nn.Conv2d(ch, 4, 3, padding=1))

    def forward(self, h):
        return haar_unfold(self.conv(F.silu(self.norm(h))))


class PainterNet(nn.Module):
    def __init__(self, model, blur_max_px, fp32_top_level):
        super().__init__()
        if model["fold"] != "haar1":
            raise ValueError(model["fold"])
        ch0 = model["base_ch"] * model["ch_mult"][0]
        self.embed = CondEmbed(model, blur_max_px)
        self.stem = Stem(ch0)
        self.body = UNetBody(model, fp32_top_level)
        self.head = Head(ch0, model["groups"])


def build_net(model, train):
    return PainterNet(model, train["cue.blur_max_px"], train["optim.fp32_top_level"])
