"""stage2/models/wavelet_unet.py — ILUMENATE U-Net の Haar wavelet 版。

Stage 2 の比較条件（128/256/512 ch、Conv3x3+ReLU、BN なし、通常の U-Net skip、
最終 Conv3x3）は保ち、解像度を変える経路だけを次のように置き換える。

  MaxPool                  -> HaarDWT: LL は contracting path、LH/HL/HH は同 scale の decoder へ
  ConvTranspose2d(up-conv) -> Conv1x1（LL の channel 射影）+ HaarIDWT(LL, LH/HL/HH)

HaarDWT / HaarIDWT は stage1/models/wavelet_generator.py と同じ正規直交係数（+/-1/2）を
Stage 2 内へ複製したもの。Stage 間では import しない。

例（base_ch=128, n_pool=2）:
  enc:        1 -> 128 -> 128 --DWT1(LL1, HF1)--> 256 -> 256 --DWT2(LL2, HF2)
  bottleneck: 256 -> 512 -> 512 -> 512
  dec2:       512 --1x1->256-- IDWT2(+HF2) --cat enc2--> 256 -> 256
  dec1:       256 --1x1->128-- IDWT1(+HF1) --cat enc1--> 128 -> 128 -> output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet import ConvBlock


def _haar_kernels():
    """正規直交 2x2 Haar kernel (LL, LH, HL, HH)。DWT と IDWT で同じ係数を使う。"""
    low = torch.tensor([1.0, 1.0])
    high = torch.tensor([1.0, -1.0])
    return 0.5 * torch.stack(
        [
            torch.outer(low, low),
            torch.outer(low, high),
            torch.outer(high, low),
            torch.outer(high, high),
        ]
    )


class HaarDWT(nn.Module):
    """x: (N,C,H,W) -> LL: (N,C,H/2,W/2), HF: (N,3C,H/2,W/2)。"""

    def __init__(self):
        super().__init__()
        self.register_buffer("kernel", _haar_kernels().unsqueeze(1))  # (4,1,2,2)

    def forward(self, x):
        n, c, h, w = x.shape
        if h % 2 or w % 2:
            raise ValueError(f"HaarDWT には偶数サイズが必要です: {(h, w)}")
        weight = self.kernel.repeat(c, 1, 1, 1)
        bands = F.conv2d(x, weight, stride=2, groups=c).view(n, c, 4, h // 2, w // 2)
        ll = bands[:, :, 0]
        hf = bands[:, :, 1:].reshape(n, 3 * c, h // 2, w // 2)
        return ll, hf


class HaarIDWT(nn.Module):
    """LL: (N,C,h,w) + HF: (N,3C,h,w) -> (N,C,2h,2w)。HaarDWT の厳密な逆変換。"""

    def __init__(self):
        super().__init__()
        self.register_buffer("kernel", _haar_kernels().unsqueeze(1))  # (4,1,2,2)

    def forward(self, ll, hf):
        n, c, h, w = ll.shape
        expected = (n, 3 * c, h, w)
        if tuple(hf.shape) != expected:
            raise ValueError(f"IDWT の HF shape が不正です: {tuple(hf.shape)} != {expected}")
        bands = torch.cat([ll.unsqueeze(2), hf.reshape(n, c, 3, h, w)], dim=2)
        bands = bands.reshape(n, 4 * c, h, w)
        weight = self.kernel.repeat(c, 1, 1, 1)
        return F.conv_transpose2d(bands, weight, stride=2, groups=c)


class WaveletUNet(nn.Module):
    """ILUMENATE の channel / ConvBlock を保ち、down/up sampling を Haar wavelet 化した U-Net。"""

    def __init__(self, in_nc=1, out_nc=1, base_ch=128, n_pool=2, final_act="linear"):
        super().__init__()
        chans = [base_ch * (2**i) for i in range(n_pool + 1)]  # 128, 256, 512

        self.enc = nn.ModuleList()
        prev = in_nc
        for c in chans[:-1]:
            self.enc.append(ConvBlock(prev, c, n_conv=2))
            prev = c

        self.dwt = HaarDWT()
        self.idwt = HaarIDWT()
        self.bottleneck = ConvBlock(prev, chans[-1], n_conv=3)

        # IDWT の LL channel は同 scale の encoder feature と同じである必要がある。
        # 元 U-Net の up-conv が担っていた channel 縮小を、空間変換の前の 1x1 conv に分離する。
        self.ll_proj = nn.ModuleList()
        self.dec = nn.ModuleList()
        prev = chans[-1]
        for c in reversed(chans[:-1]):
            self.ll_proj.append(nn.Conv2d(prev, c, kernel_size=1))
            self.dec.append(ConvBlock(2 * c, c, n_conv=2))  # cat(IDWT, feature skip) -> c
            prev = c

        self.out = nn.Conv2d(prev, out_nc, kernel_size=3, padding=1)
        if final_act == "linear":
            self.act = nn.Identity()
        elif final_act == "tanh":
            self.act = nn.Tanh()
        else:
            raise ValueError(final_act)
        self.n_pool = n_pool

    def forward(self, x):
        multiple = 2**self.n_pool
        if x.shape[-1] % multiple or x.shape[-2] % multiple:
            raise ValueError(f"入力の H, W は {multiple} の倍数である必要があります: {tuple(x.shape)}")

        skips = []
        for enc in self.enc:
            x = enc(x)
            ll, hf = self.dwt(x)
            skips.append((x, hf))
            x = ll

        x = self.bottleneck(x)
        for project, dec, (skip, hf) in zip(self.ll_proj, self.dec, reversed(skips)):
            x = self.idwt(project(x), hf)
            x = dec(torch.cat([skip, x], dim=1))
        return self.act(self.out(x))
