"""stage2/models/discriminator.py — Stage 2 の出力側 D（2026-11-09、ブランチ 023）。

動機: 研究者の決定（2026-10-12「Stage 2 は MSE + 実 EID 出力側 D」、2026-11-08「病院 EID → Stage 2 の出力がぼけないように」）。
MSE は EID-like（ペア）の経路しか見ないので、実 EID の出力は学習信号を持たず条件付き平均（ぼけ）に落ちる（機構分析 20261107 §3）。
実 EID を入れたときの出力を「偽物」、教師の PCD1024 patch を「本物」として D に見せ、実 EID の経路にも PCD らしさの信号を与える。

D は junyanz の 70×70 PatchGAN（NLayerDiscriminator、ndf 64、n_layers 3、BatchNorm）をそのまま写した（Stage 1 の paper D と同系）。
GAN 損失は lsgan（既定）| vanilla。
"""

import torch
import torch.nn as nn


class PatchDiscriminator(nn.Module):
    def __init__(self, input_nc=1, ndf=64, n_layers=3):
        super().__init__()
        kw, padw = 4, 1
        seq = [nn.Conv2d(input_nc, ndf, kw, 2, padw), nn.LeakyReLU(0.2, True)]
        nf, nf_prev = 1, 1
        for n in range(1, n_layers):
            nf_prev, nf = nf, min(2 ** n, 8)
            seq += [nn.Conv2d(ndf * nf_prev, ndf * nf, kw, 2, padw, bias=False), nn.BatchNorm2d(ndf * nf), nn.LeakyReLU(0.2, True)]
        nf_prev, nf = nf, min(2 ** n_layers, 8)
        seq += [nn.Conv2d(ndf * nf_prev, ndf * nf, kw, 1, padw, bias=False), nn.BatchNorm2d(ndf * nf), nn.LeakyReLU(0.2, True)]
        seq += [nn.Conv2d(ndf * nf, 1, kw, 1, padw)]
        self.model = nn.Sequential(*seq)
        for m in self.modules():  # junyanz の init normal 0.02
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.model(x)


class GANLoss(nn.Module):
    def __init__(self, gan_mode):
        super().__init__()
        if gan_mode == "lsgan":
            self.loss = nn.MSELoss()
        elif gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(gan_mode)

    def forward(self, pred, target_is_real):
        target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
        return self.loss(pred, target)
