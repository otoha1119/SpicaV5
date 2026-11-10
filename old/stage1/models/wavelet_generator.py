"""wavelet_generator.py — Park et al. 2019 の generator (deep convolutional framelet) を Fig. 3 どおりに実装する。

対象論文: H. S. Park et al., "Unpaired Image Denoising Using a Generative Adversarial Network in X-Ray CT,"
          IEEE Access 2019 (arXiv:1903.06257v2). Sec. II.C "Network Architecture" と Fig. 3。

【論文の記述 (原文) と実装の対応】
 (a) "consisting of a contracting path and an expansive path with skipped connections and a concatenation (concat) layer"
     → WaveletGenerator.forward の enc / dec ループと torch.cat
 (b) "Each step of the contracting and expansive paths consists of two repeated convolutions (conv) with a 3 × 3 window.
      Each convolution is followed by a batch normalization (bnorm) and a leaky rectified linear unit (LReLU)."
     → ConvBlock = Conv3×3 → BatchNorm2d → LeakyReLU。各段 = ConvBlock ×2
 (c) "Downsampling and upsampling of the features are performed by 2-D Haar wavelet decomposition (wave-dec)
      and recomposition (wave-rec), respectively."
     → HaarDWT / HaarIDWT (固定係数、学習しない)。pooling / strided conv は使わない
 (d) "High pass filters after wavelet decomposition skip directly to the expansive path, while low pass filters
      (marked by 'LF' in Fig. 3) are concatenated with the features in the contracting path during the same step."
     → wave-dec の高周波 3 帯 (HF) は同段の wave-rec へ直結 (skips に保持)。低周波 (LL) は次段へ。
       wave-rec の出力を同段の縮小側特徴 (第 2 conv 出力) と concat
 (e) "At the end, an additional convolution layer is added to generate a grayscale output image."
     → self.final (Fig. 3 の最終矢印は青 = 3×3 conv なので kernel 3)
 (f) "each convolution in our network is performed with zero-padding to match the size of the input and output images"
     → 全 Conv2d は padding = kernel//2 の zero-padding

【Fig. 3 (400 dpi 読み取り) から確定したチャネル構成 (ngf=32, n_levels=3)】
  enc1: 1→32→32   | dwt | enc2: 32→64→64 | dwt | enc3: 64→128→128 | dwt |
  bottleneck: 128→256→128
  dec3: idwt(128, HF3) → 128, cat enc3 → 256 → 128 → 64
  dec2: idwt(64,  HF2) → 64,  cat enc2 → 128 → 64  → 32
  dec1: idwt(32,  HF1) → 32,  cat enc1 → 64  → 32  → final conv → 1
  ※ 各 decoder 段の第 2 conv が一つ上の段の wave-rec の LL 入力になる (チャネルを上段に合わせて減らす)

【論文に記載が無く、引数で切り替えられるようにした点】(docs/reference/park2019_implementation_checklist.md §7)
  Q-G1 lrelu_slope   : G の LReLU の slope。論文は D についてのみ 0.2 と明記。既定 0.2
  Q-G2 wavelet       : Haar の正規化。'ortho' (係数 ±1/2、正規直交、wave-rec が厳密な逆変換) / 'unnormalized' (dec ±1, rec ±1/4)。既定 'ortho'
  Q-G3 final_norm_act: 最終 conv の後ろに bnorm + LReLU を付けるか。既定 False (付けない)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 2-D Haar wavelet (wave-dec / wave-rec)。固定係数、各チャネル独立 (depthwise)。
# ---------------------------------------------------------------------------
def _haar_kernels(mode: str):
    """2×2 Haar カーネル 4 種 (LL, LH, HL, HH) を返す。戻り値 (dec, rec) は各 (4, 2, 2)。

    1-D Haar: L = [1, 1], H = [1, -1]。2-D は外積。
    ortho        : dec = rec = k/2 → 各カーネルのノルム 1、互いに直交。conv_transpose2d(conv2d(x)) == x
    unnormalized : dec = k, rec = k/4 → 同じく完全再構成 (係数のスケールだけ違う)
    """
    L = torch.tensor([1.0, 1.0])
    H = torch.tensor([1.0, -1.0])
    k = torch.stack([torch.outer(L, L), torch.outer(L, H), torch.outer(H, L), torch.outer(H, H)])  # (4, 2, 2)
    if mode == "ortho":
        return k * 0.5, k * 0.5
    if mode == "unnormalized":
        return k, k * 0.25
    raise ValueError(f"unknown wavelet normalization: {mode}")


class HaarDWT(nn.Module):
    """wave-dec: x (N, C, H, W) → LL (N, C, H/2, W/2), HF (N, 3C, H/2, W/2)。HF は [c0: LH, HL, HH, c1: ...] の順。"""

    def __init__(self, mode: str = "ortho"):
        super().__init__()
        dec, _ = _haar_kernels(mode)
        self.register_buffer("kernel", dec.unsqueeze(1))  # (4, 1, 2, 2)

    def forward(self, x):
        n, c, h, w = x.shape
        assert h % 2 == 0 and w % 2 == 0, f"wave-dec には偶数サイズが必要: {(h, w)}"
        weight = self.kernel.repeat(c, 1, 1, 1)  # (4C, 1, 2, 2), groups=C → チャネルごとに 4 帯
        y = F.conv2d(x, weight, stride=2, groups=c).view(n, c, 4, h // 2, w // 2)
        ll = y[:, :, 0]
        hf = y[:, :, 1:].reshape(n, 3 * c, h // 2, w // 2)
        return ll, hf


class HaarIDWT(nn.Module):
    """wave-rec: LL (N, C, h, w) + HF (N, 3C, h, w) → x (N, C, 2h, 2w)。HaarDWT の厳密な逆変換。"""

    def __init__(self, mode: str = "ortho"):
        super().__init__()
        _, rec = _haar_kernels(mode)
        self.register_buffer("kernel", rec.unsqueeze(1))  # (4, 1, 2, 2)

    def forward(self, ll, hf):
        n, c, h, w = ll.shape
        assert hf.shape[1] == 3 * c, f"HF のチャネル数は LL の 3 倍: {hf.shape[1]} != {3 * c}"
        y = torch.cat([ll.unsqueeze(2), hf.view(n, c, 3, h, w)], dim=2).reshape(n, 4 * c, h, w)
        weight = self.kernel.repeat(c, 1, 1, 1)  # (4C, 1, 2, 2)
        return F.conv_transpose2d(y, weight, stride=2, groups=c)


# ---------------------------------------------------------------------------
# 3×3 conv → bnorm → LReLU  (論文 (b))
# ---------------------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, slope: float = 0.2):
        super().__init__()
        # bias=False: 直後の BatchNorm がバイアスを吸収する (junyanz の BN 使用時の慣行と同じ)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)  # zero-padding (論文 (f))
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(slope, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# ---------------------------------------------------------------------------
# Generator (Fig. 3)
# ---------------------------------------------------------------------------
class WaveletGenerator(nn.Module):
    def __init__(
        self,
        input_nc: int = 1,
        output_nc: int = 1,
        ngf: int = 32,
        n_levels: int = 3,
        lrelu_slope: float = 0.2,      # Q-G1
        wavelet: str = "ortho",        # Q-G2
        final_kernel: int = 3,         # Fig. 3 の最終矢印は青 (3×3 conv)
        final_norm_act: bool = False,  # Q-G3
    ):
        super().__init__()
        chs = [ngf * (2**i) for i in range(n_levels)]  # ngf=32, n_levels=3 → [32, 64, 128]
        self.n_levels = n_levels

        # contracting path: 各段 ConvBlock ×2、段の間は wave-dec
        self.enc = nn.ModuleList()
        for k in range(n_levels):
            in_ch = input_nc if k == 0 else chs[k - 1]
            self.enc.append(nn.Sequential(ConvBlock(in_ch, chs[k], lrelu_slope), ConvBlock(chs[k], chs[k], lrelu_slope)))
        self.dwt = HaarDWT(wavelet)
        self.idwt = HaarIDWT(wavelet)

        # bottleneck (Fig. 3: 128 → 256 → 128)。第 2 conv で最下段 wave-rec の LL 入力チャネル数に戻す
        self.bottleneck = nn.Sequential(ConvBlock(chs[-1], 2 * chs[-1], lrelu_slope), ConvBlock(2 * chs[-1], chs[-1], lrelu_slope))

        # expansive path: wave-rec → concat → ConvBlock ×2 (第 2 conv は一つ上の段のチャネル数へ)
        self.dec = nn.ModuleList()
        for k in range(n_levels - 1, 0, -1):  # k = n_levels-1 ... 1
            self.dec.append(nn.Sequential(ConvBlock(2 * chs[k], chs[k], lrelu_slope), ConvBlock(chs[k], chs[k - 1], lrelu_slope)))
        # 最上段 (k = 0): ConvBlock(64 → 32) + 出力 conv (論文 (e) "an additional convolution layer")
        self.dec_top = ConvBlock(2 * chs[0], chs[0], lrelu_slope)
        final = [nn.Conv2d(chs[0], output_nc, kernel_size=final_kernel, padding=final_kernel // 2)]
        if final_norm_act:
            final += [nn.BatchNorm2d(output_nc), nn.LeakyReLU(lrelu_slope, inplace=True)]
        self.final = nn.Sequential(*final)

    def forward(self, x):
        assert x.shape[-1] % (2**self.n_levels) == 0 and x.shape[-2] % (2**self.n_levels) == 0, f"入力サイズは {2**self.n_levels} の倍数が必要: {tuple(x.shape[-2:])}"
        skips = []  # 各段の (縮小側特徴 e_k, 高周波 HF_k)
        h = x
        for k in range(self.n_levels):
            h = self.enc[k](h)          # ConvBlock ×2
            e = h
            ll, hf = self.dwt(h)        # wave-dec: LL は次段へ、HF は同段の wave-rec へ (論文 (d))
            skips.append((e, hf))
            h = ll
        h = self.bottleneck(h)
        for i, k in enumerate(range(self.n_levels - 1, 0, -1)):
            e, hf = skips[k]
            h = self.idwt(h, hf)        # wave-rec (LL = 下段からの出力, HF = 同段の skip)
            h = torch.cat([h, e], dim=1)  # skip & concat (論文 (a), (d))
            h = self.dec[i](h)
        e, hf = skips[0]
        h = self.idwt(h, hf)
        h = torch.cat([h, e], dim=1)
        h = self.dec_top(h)
        return self.final(h)
