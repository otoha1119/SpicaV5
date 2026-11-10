"""ablation/SRCycleGAN/models/networks.py — SR-CycleGAN のネットワーク。著者公開コード（SpicaV2 commit 741924c、models/networks.py）からの移植。

移植元（741924c）との対応:
  Clinical2MicroGenerator   : 論文 G1（LR → HR。downblock / upblock を外し、末尾に PixelShuffle）。741924c 1066〜1136 行。変更は upsampling_times を scale の log2 から決めることと print の削除だけ
  Micro2ClinicalGenerator   : 論文 G2（HR → LR。junyanz の ResnetGenerator の後に strided conv の downblock）。741924c 1478〜1544 行。同上（downsampling_times）
  ResnetBlock               : junyanz 本家と同一（741924c 502 行）
  NLayerDiscriminator       : 70×70 PatchGAN（741924c 756 行。本家と同一。"SpectralNorm" の print は削除）
  GANLoss                   : junyanz 本家（vanilla = BCEWithLogits = 論文 Eq. 2 の log 形、lsgan = コードの既定）
  get_norm_layer / init_weights / init_net / linear scheduler : junyanz 本家（stage1/models/networks.py から複製）

論文（JMI 2022）との照合:
  Sec. 2.3.1 "we modified conventional CycleGAN's ... generator G1 to an SR neural network by removing downblocks/upblocks ... and adding pixel-shuffling layers"
  Fig. 3     "G2 is a generator that shrinks an input image of 256×256 pixels into an image of 32×32 pixels. We added three downsample blocks (downblocks) to generator G2"
             "Each downblock contains a convolution layer that scales down the image to 1/2 of its original size" + batch normalization + activation
  Sec. 4.2   G1 は 9 ResBlocks
  D1 / D2 は "a CNN-based discriminator" とだけ書かれている → コードの basic（70×70 PatchGAN、n_layers 3）
正規化層はユーザー決定（2026-10-27）で batch（論文）。コードの既定は instance。train.yaml network.norm で選ぶ。
"""

import functools
import math

import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type="instance"):
    """batch | instance | none（junyanz 本家）。BatchNorm は affine + running 統計、InstanceNorm は affine なし・統計なし。"""
    if norm_type == "batch":
        return functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    if norm_type == "instance":
        return functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    if norm_type == "none":
        return lambda x: Identity()
    raise NotImplementedError("normalization layer [%s] is not found" % norm_type)


def _use_bias(norm_layer):
    if type(norm_layer) == functools.partial:
        return norm_layer.func == nn.InstanceNorm2d
    return norm_layer == nn.InstanceNorm2d


def log2_int(scale):
    """scale（2 のべき）→ 段数。2 → 1、8 → 3（論文）。"""
    n = int(round(math.log2(scale)))
    if scale < 2 or 2**n != scale:
        raise ValueError(f"scale は 2 以上の 2 のべき: {scale}")
    return n


def init_weights(net, init_type="normal", init_gain=0.02):
    """junyanz 本家。Conv / Linear の重みを init_type で、BatchNorm2d の weight を N(1, gain)・bias 0 に。"""

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError("initialization method [%s] is not implemented" % init_type)
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


def init_net(net, init_type, init_gain, device):
    net.to(device)
    init_weights(net, init_type, init_gain)
    return net


def linear_scheduler(optimizer, epoch_count, n_epochs, n_epochs_decay):
    """junyanz の lr_policy=linear: 最初の n_epochs は一定、次の n_epochs_decay で 0 へ線形（論文 Sec. 3.3 と同じ）。
    LambdaLR は epoch_count を起点に 0 から数えるので、再開時は optimizer の lr を戻してから作り直す（Stage 1 と同じ扱い）。"""

    def lambda_rule(epoch):
        return 1.0 - max(0, epoch + epoch_count - n_epochs) / float(n_epochs_decay + 1)

    return lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)


###############################################################################
# ResnetBlock（junyanz 本家 = 741924c 502 行）
###############################################################################
class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


###############################################################################
# G1: Clinical2MicroGenerator（741924c 1066〜1136 行）
###############################################################################
class Clinical2MicroGenerator(nn.Module):
    """論文 G1（LR → HR、×scale）。ReflectionPad(3) + 7×7 conv → norm → PReLU → ResnetBlock × n_blocks → ReflectionPad(1) + 3×3 conv → norm → PReLU
    → [3×3 conv ngf→2ngf → norm → PixelShuffle(2) → PReLU] × log2(scale)（チャネルは段ごとに 1/2）→ ReflectionPad(3) + 7×7 conv → Tanh。
    741924c からの変更: upsampling_times を scale から決める。層の並び・名前（self.model の Sequential の index）はそのまま。"""

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=9, padding_type="reflect", scale=8):
        assert n_blocks >= 0
        super(Clinical2MicroGenerator, self).__init__()
        use_bias = _use_bias(norm_layer)
        upsampling_times = log2_int(scale)  # 741924c: upsampling_times（論文 ×8 → 3）
        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias), norm_layer(ngf), nn.PReLU()]
        for i in range(n_blocks):  # add ResNet blocks
            model += [ResnetBlock(ngf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        model += [nn.ReflectionPad2d(1), nn.Conv2d(ngf, ngf, kernel_size=3, padding=0, bias=use_bias), norm_layer(ngf), nn.PReLU()]
        # upsampling layers: pixel shuffle（741924c "使用pixel shuffle方式进行upsample"）
        for i in range(upsampling_times):
            model += [nn.Conv2d(int(ngf), int(ngf * 2), kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(int(ngf * 2)), nn.PixelShuffle(2), nn.PReLU()]  # channel 数は 1/4 → 実効 ngf/2
            ngf = ngf / 2
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(int(ngf), output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


###############################################################################
# G2: Micro2ClinicalGenerator（741924c 1478〜1544 行）
###############################################################################
class Micro2ClinicalGenerator(nn.Module):
    """論文 G2（HR → LR、1/scale）。junyanz ResnetGenerator（7×7 conv → 2 downsampling → ResnetBlock × n_blocks → 2 upsampling）の後に
    downblock [3×3 conv stride 2 → norm → ReLU] × log2(scale)（チャネルは段ごとに 2 倍）→ ReflectionPad(3) + 7×7 conv → Tanh。
    741924c からの変更: downsampling_times を scale から決める。"""

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=9, padding_type="reflect", scale=8):
        assert n_blocks >= 0
        super(Micro2ClinicalGenerator, self).__init__()
        use_bias = _use_bias(norm_layer)
        downsampling_times = log2_int(scale)  # 741924c: downsampling_times（論文 ×8 → 3）
        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias), norm_layer(ngf), nn.ReLU(True)]
        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias), norm_layer(ngf * mult * 2), nn.ReLU(True)]
        mult = 2**n_downsampling
        for i in range(n_blocks):  # add ResNet blocks
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias), norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]
        # downsampling to clinical scale（741924c "downsampling to clinical scale" = 論文の downblock）
        for i in range(downsampling_times):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias), norm_layer(ngf * mult * 2), nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf * (2**downsampling_times), output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


###############################################################################
# D: NLayerDiscriminator（741924c 756 行 = junyanz 本家の 70×70 PatchGAN）
###############################################################################
class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias), norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)]
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias), norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


###############################################################################
# GANLoss（junyanz 本家）
###############################################################################
class GANLoss(nn.Module):
    """vanilla = BCEWithLogits（論文 Eq. 2 の log 形。D の出力は logit）| lsgan = MSE（コード 741924c の既定）。"""

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer("real_label", torch.tensor(target_real_label))
        self.register_buffer("fake_label", torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == "lsgan":
            self.loss = nn.MSELoss()
        elif gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError("gan mode %s not implemented" % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        target_tensor = self.real_label if target_is_real else self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        return self.loss(prediction, self.get_target_tensor(prediction, target_is_real))


###############################################################################
# 組み立て
###############################################################################
def build_G1(cfg, device):
    """cfg = train の平坦 dict（network.*）。論文 G1（A: LR → B: HR）。"""
    norm_layer = get_norm_layer(cfg["network.norm"])
    net = Clinical2MicroGenerator(1, 1, cfg["network.ngf"], norm_layer=norm_layer, use_dropout=False, n_blocks=cfg["network.n_blocks"], scale=cfg["network.scale"])
    return init_net(net, cfg["network.init_type"], cfg["network.init_gain"], device)


def build_G2(cfg, device):
    """論文 G2（B: HR → A: LR）。"""
    norm_layer = get_norm_layer(cfg["network.norm"])
    net = Micro2ClinicalGenerator(1, 1, cfg["network.ngf"], norm_layer=norm_layer, use_dropout=False, n_blocks=cfg["network.n_blocks"], scale=cfg["network.scale"])
    return init_net(net, cfg["network.init_type"], cfg["network.init_gain"], device)


def build_D(cfg, device):
    """D1 / D2（同じ構造。入力は 1ch）。"""
    norm_layer = get_norm_layer(cfg["network.norm"])
    net = NLayerDiscriminator(1, cfg["network.ndf"], n_layers=cfg["network.n_layers_D"], norm_layer=norm_layer)
    return init_net(net, cfg["network.init_type"], cfg["network.init_gain"], device)


def count_params(net):
    return sum(p.numel() for p in net.parameters())
