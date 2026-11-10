"""paper_discriminator.py — Park et al. 2019 の discriminator (PatchGAN) を Sec. II.C / Fig. 3 どおりに実装する。

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

import torch.nn as nn


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
