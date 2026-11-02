"""wavelet_discriminator.py — [SpicaV5] wavelet 入力の discriminator（2026-11-02、ブランチ 015-f_014-stage1_wavelet_D）。

動機: 通常の conv D は stride 2 の初段で高周波を捨てやすく、wavelet 型 G と組むと高周波帯に D の勾配が届かず不安定になる
（SWAGAN, Gal et al. SIGGRAPH 2021 の対照実験: wavelet G + 通常 D で FID 5.22 → 12.27。docs/surveys/20261102_wavelet-discriminator.md）。
PC1 run 2026_1102_1806（packet G + paper D）でも EID-like は「ぼけるが PCD の粒が残る」で、D が高周波の質感差を判定に使えていなかった。

構成: 固定 Haar 1 段（wavelet_generator.HaarDWT、学習しない）で入力を [LL | LH, HL, HH]（各 H/2 × W/2）に分け、
チャネルとして積んで Park Fig. 3 の PaperDiscriminator（4×4 s2 ×3 + 1×1、BN 固定）に入れる。D の構造・損失は変えず、1 層目が見る表現だけ変える。
  bands = "all": 4·input_nc ch（LL も見る。HU の水準も D の判定対象のまま）
  bands = "hf" : 3·input_nc ch（高周波だけ見る。LL = HU の水準は fidelity に任せ、D が非ペアのコホート差（HU）を学ぶ経路を塞ぐ。先行例の相場: WGSR CVPR 2024、Kim et al. PLoS ONE 2022）
受容野: paper D の 22 px は wavelet 係数上なので画像では 44 px（ノイズ相関長 6 px と血管の縁を同時に見る）。入力は偶数サイズが必要。
"""

import torch
import torch.nn as nn

from .paper_discriminator import PaperDiscriminator
from .wavelet_generator import HaarDWT

BANDS = ("all", "hf")


class WaveletPaperDiscriminator(nn.Module):
    def __init__(self, input_nc: int = 1, ndf: int = 32, bands: str = "all", wavelet: str = "ortho"):
        super().__init__()
        if bands not in BANDS:
            raise ValueError(f"bands は {BANDS}: {bands}")
        self.bands = bands
        self.dwt = HaarDWT(wavelet)
        self.model = PaperDiscriminator((4 if bands == "all" else 3) * input_nc, ndf=ndf)

    def forward(self, x):
        ll, hf = self.dwt(x)
        return self.model(torch.cat([ll, hf], dim=1) if self.bands == "all" else hf)
