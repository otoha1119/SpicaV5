"""stage2/models/discriminator.py — 実 EID 枝の D（PatchGAN）。stage1/models/paper_discriminator.py（PaperDiscriminator）を 2026-10-13 に**コピー**した（Stage 間で import しない）。
入力は U-Net の出力（1 ch、正規化空間 [−1,1]）の小片。本物側はペア枝の教師（PCD1024 の patch）、偽物側は実 EID 小片 → ×scale → U-Net の出力。
全畳み込みなので本物と偽物の大きさが違っても動く（patch map の大きさが変わるだけ。adv.crop_eid × scale ≠ data.patch_size でも可）。

初期化は Stage 1 の D と同じ normal(0, 0.01)（Park 2019 "Gaussian distribution with a mean of 0 … standard deviation of 0.01"、junyanz init_net の normal と同じ掛け方:
Conv の weight ~ N(0, gain)、BatchNorm の weight ~ N(1, gain)、bias 0）。U-Net の mode.init_type（xavier_uniform = Keras 既定）とは**合わせない**
（U-Net は ILUMENATE、D は Park 2019 / Stage 1 の流儀。GAN の均衡は Stage 1 で λ を校正した D と同じ初期値のほうが見通しが良い。2026-10-13 Claude 決定）。

--- 以下、stage1/models/paper_discriminator.py の冒頭コメントをそのまま写す ---
paper_discriminator.py — Park et al. 2019 の discriminator (PatchGAN) を Sec. II.C / Fig. 3 どおりに実装する。

対象論文: H. S. Park et al., "Unpaired Image Denoising Using a Generative Adversarial Network in X-Ray CT,"
          IEEE Access 2019 (arXiv:1903.06257v2)。

【論文の記述 (原文) と実装の対応】
 (a) "we adopt the PatchGAN classifier as a discriminator, which tries to classify whether each patch in an image is real or fake"
     → 全畳み込み。出力は (N, 1, h, w) の patch map (画像 1 枚につき 1 値ではない)
 (b) "The discriminator contains three convolution layers with a 4 × 4 window and strides of two in each direction of the domain
      and each layer is followed by a batch normalization and a leaky ReLU with a slope of 0.2."
     → [Conv4×4 s2 → BatchNorm2d → LeakyReLU(0.2)] ×3。**1 層目にも bnorm** (CycleGAN の D は 1 層目 norm なしだが、本論文は "each layer")
 (c) "As the final stage of the architecture, a 1 × 1 convolution layer is added to generate 1-dimensional output data."
     → Conv1×1 (128 → 1)。活性化は本文に記載なし → 付けない (D の値域制約は損失側 (フェーズ 4) で扱う)
 (d) Fig. 3 のチャネル: 32, 32, 128
 (e) "each convolution in our network is performed with zero-padding" (Sec. II.C、G の段落)
     → 4×4 s2 に padding=1 (zero)。128 → 64 → 32 → 16。D への適用は明記されていないが (チェックリスト Q-D1)、
       stride 2 でサイズを正確に半分にする標準的な選択でもある

【受容野】 4 → 4+3·2=10 → 10+3·4=22。1×1 conv は増やさない → 22×22
"""

import torch
import torch.nn as nn

D_INIT_GAIN = 0.01   # Stage 1 train.yaml network.init_gain と同じ（Park 2019 "standard deviation of 0.01"）
D_MIN_INPUT = 8      # 4×4 s2 ×3 で 1×1 の patch map が出る最小の一辺（これ未満は Conv が失敗する）


class PaperDiscriminator(nn.Module):
    def __init__(self, input_nc: int = 1, ndf: int = 32, lrelu_slope: float = 0.2, padding: int = 1):
        """
        input_nc : 入力チャネル数 (グレースケール CT なら 1)
        ndf      : 1 層目のフィルタ数。Fig. 3 は (32, 32, 128) = (ndf, ndf, 4·ndf) with ndf=32
        """
        super().__init__()
        chs = [ndf, ndf, 4 * ndf]  # Fig. 3: 32, 32, 128
        layers = []
        in_ch = input_nc
        for out_ch in chs:
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=padding, bias=False),  # bias は直後の bnorm が吸収
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(lrelu_slope, inplace=True),
            ]
            in_ch = out_ch
        layers += [nn.Conv2d(in_ch, 1, kernel_size=1, stride=1, padding=0)]  # (c) 1×1 conv → 1 ch。活性化なし
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def init_weights_D(net, gain=D_INIT_GAIN):
    """junyanz networks.init_weights(init_type="normal") と同じ: Conv の weight ~ N(0, gain)、bias 0。BatchNorm の weight ~ N(1, gain)、bias 0。"""
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0.0, gain)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, gain)
            nn.init.zeros_(m.bias)


def build_D(in_nc=1):
    """実 EID 枝の D を組む（構成は固定: Park 2019 Fig. 3。入力 1 ch）。"""
    net = PaperDiscriminator(input_nc=in_nc)
    init_weights_D(net)
    return net


class GANLoss(nn.Module):
    """junyanz networks.GANLoss の lsgan / vanilla を写したもの（Stage 1 の ablation 用損失と同じ式。wgangp は持ち込まない）+ hinge（2026-10-13）。
    D の出力（patch map、活性化なし）に対し、lsgan は MSE（本物 1 / 偽物 0）、vanilla は BCEWithLogits。
    hinge（Lim & Ye 2017 / SAGAN）: D は max(0, 1 − D(real)) + max(0, 1 + D(fake))、G は −D(fake)。G の勾配が D の出力の大きさに比例しないので、D が勝っても G の勾配が爆発しない。"""

    def __init__(self, gan_mode):
        super().__init__()
        if gan_mode == "lsgan":
            self.loss = nn.MSELoss()
        elif gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == "hinge":
            self.loss = None
        else:
            raise ValueError(gan_mode)
        self.gan_mode = gan_mode

    def forward(self, prediction, target_is_real, for_D=True):
        """for_D: D の損失（True）か G の損失（False）か。lsgan / vanilla では区別しない（junyanz と同じ）。hinge だけ式が違う。"""
        if self.gan_mode == "hinge":
            if not for_D:
                return -prediction.mean()
            return torch.relu(1.0 - prediction).mean() if target_is_real else torch.relu(1.0 + prediction).mean()
        target = torch.full_like(prediction, 1.0 if target_is_real else 0.0)
        return self.loss(prediction, target)
