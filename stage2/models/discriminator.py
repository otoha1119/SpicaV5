"""stage2/models/discriminator.py — Stage 2 の出力側 D（2026-11-09、ブランチ 023）。

動機: 研究者の決定（2026-10-12「Stage 2 は MSE + 実 EID 出力側 D」、2026-11-08「病院 EID → Stage 2 の出力がぼけないように」）。
MSE は EID-like（ペア）の経路しか見ないので、実 EID の出力は学習信号を持たず条件付き平均（ぼけ）に落ちる（機構分析 20261107 §3）。
実 EID を入れたときの出力を「偽物」、教師の PCD1024 patch を「本物」として D に見せ、実 EID の経路にも PCD らしさの信号を与える。

D は junyanz の 70×70 PatchGAN（NLayerDiscriminator、ndf 64、n_layers 3、BatchNorm）をそのまま写した（Stage 1 の paper D と同系）。
GAN 損失は lsgan（既定）| vanilla。
norm（adv.d_norm、2026-11-09 ブランチ 026）: batch = 上記そのまま | spectral = BatchNorm を置かず全 Conv（1 層目・最終層も）に torch.nn.utils.spectral_norm。
  BatchNorm は本物・偽物の群の差を batch 統計で消す / batch 構成に依る（Codex レビュー 11-09）ので、構成（ndf / n_layers / 70×70 / LeakyReLU 0.2）はそのままに正規化だけ替える。
  state_dict のキーが変わる（weight_orig / weight_u / weight_v、BN 無し）ので、d_norm が違う run からは resume できない（run_train.py が止める）。
"""

import torch
import torch.nn as nn

D_NORMS = ("batch", "spectral")


class PatchDiscriminator(nn.Module):
    def __init__(self, input_nc=1, ndf=64, n_layers=3, norm="batch"):
        super().__init__()
        if norm not in D_NORMS:
            raise ValueError(f"norm は {D_NORMS}: {norm!r}")
        kw, padw = 4, 1
        bn = norm == "batch"   # BN の後の Conv は bias 無し（junyanz）。spectral は BN が無いので bias 有り

        def conv(i, o, stride, bias=True):
            c = nn.Conv2d(i, o, kw, stride, padw, bias=bias)
            nn.init.normal_(c.weight, 0.0, 0.02)   # junyanz の init normal 0.02（spectral_norm で包む前に weight_orig へ入れる）
            if c.bias is not None:
                nn.init.zeros_(c.bias)
            return c if bn else nn.utils.spectral_norm(c)

        def block(i, o, stride):
            return [conv(i, o, stride, bias=not bn)] + ([nn.BatchNorm2d(o)] if bn else []) + [nn.LeakyReLU(0.2, True)]

        seq = [conv(input_nc, ndf, 2), nn.LeakyReLU(0.2, True)]
        nf, nf_prev = 1, 1
        for n in range(1, n_layers):
            nf_prev, nf = nf, min(2 ** n, 8)
            seq += block(ndf * nf_prev, ndf * nf, 2)
        nf_prev, nf = nf, min(2 ** n_layers, 8)
        seq += block(ndf * nf_prev, ndf * nf, 1)
        seq += [conv(ndf * nf, 1, 1)]
        self.model = nn.Sequential(*seq)
        for m in self.modules():  # junyanz の init normal 0.02（BN）
            if isinstance(m, nn.BatchNorm2d):
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
