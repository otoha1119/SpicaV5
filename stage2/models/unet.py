"""stage2/models/unet.py — ILUMENATE の U-Net（Koons et al., Med Phys 2025, Fig. 2）。

論文の記述:
  "ILUMENATE employed a modified U-Net Architecture with 2 max pooling and 2 up-convolutional layers (128 filters at the first layer
   doubling at each max pooling layer), rectified linear unit (ReLU) activation function"
  Fig. 2: CNN Input → [128 → 128] → pool → [256 → 256] → pool → [512 → 512 → 512] → up → [256 → 256] → up → [128 → 128] → CNN Output
          （凡例: Conv 3x3 + RELU / Max Pooling / Up Convolutional Layer / Skip Connection）

ここでの読み（論文に無い点。docs/plans/20260908_stage2-unet-implementation-spec.md §4）:
  ・畳み込みは zero padding（入力 128² patch → 出力 128²。論文は patch 対で学習しているので "same" 以外は考えにくい）
  ・BatchNorm / dropout は記載が無いので入れない
  ・up-convolution = ConvTranspose2d(2×2, stride 2)（Ronneberger 2015 の "up-convolution"）。skip は concat
  ・bottleneck は Fig. 2 どおり Conv3×3 を 3 回（他の段は 2 回）
  ・最終層は Conv3×3（base_ch → 1）。活性化は final_act（linear = 無し、tanh = 変種）。図に 1ch への層は描かれていないのでここは読み
  ・初期化は init_type（xavier_uniform = Keras 既定 Glorot uniform + bias 0 / torch = PyTorch 既定）
n_pool を一般化してあるが、論文は 2。base_ch=128, n_pool=2 で約 9.8M パラメータ（bottleneck の Conv3×3 ×3 が約半分）。
"""

import torch
import torch.nn as nn


def conv_relu(in_ch, out_ch):
    return [nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1), nn.ReLU(inplace=True)]


class ConvBlock(nn.Module):
    """Conv3×3 + ReLU を n_conv 回。"""

    def __init__(self, in_ch, out_ch, n_conv=2):
        super().__init__()
        layers = conv_relu(in_ch, out_ch)
        for _ in range(n_conv - 1):
            layers += conv_relu(out_ch, out_ch)
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNetILUMENATE(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, base_ch=128, n_pool=2, final_act="linear"):
        super().__init__()
        chans = [base_ch * (2 ** i) for i in range(n_pool + 1)]  # 128, 256, 512
        self.enc = nn.ModuleList()
        prev = in_nc
        for c in chans[:-1]:
            self.enc.append(ConvBlock(prev, c, n_conv=2))
            prev = c
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(prev, chans[-1], n_conv=3)
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        prev = chans[-1]
        for c in reversed(chans[:-1]):
            self.up.append(nn.ConvTranspose2d(prev, c, kernel_size=2, stride=2))
            self.dec.append(ConvBlock(c * 2, c, n_conv=2))  # concat(skip, up) → c
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
        m = 2 ** self.n_pool
        if x.shape[-1] % m or x.shape[-2] % m:
            raise ValueError(f"入力の H, W は {m} の倍数である必要があります: {tuple(x.shape)}")
        skips = []
        for enc in self.enc:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.up, self.dec, reversed(skips)):
            x = dec(torch.cat([skip, up(x)], dim=1))
        return self.act(self.out(x))


def init_weights(net, init_type):
    """xavier_uniform: Keras 既定（Glorot uniform、bias 0）に合わせる。torch: PyTorch 既定のまま（何もしない）。"""
    if init_type == "torch":
        return
    if init_type != "xavier_uniform":
        raise ValueError(init_type)
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def build_net(mode, in_nc=1, out_nc=1):
    """mode.yaml の実効値からネットワークを組む。"""
    if mode["arch"] != "unet_ilumenate":
        raise ValueError(mode["arch"])
    net = UNetILUMENATE(in_nc, out_nc, mode["base_ch"], mode["n_pool"], mode["final_act"])
    init_weights(net, mode["init_type"])
    return net
