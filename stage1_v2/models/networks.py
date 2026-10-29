"""stage1_v2/models/networks.py — CycleGAN の generator / discriminator。
2026-10-27 に必要分だけ複製した。構造は変えていない。

  ResnetGenerator        : 本家 resnet_9blocks / resnet_6blocks（7×7 conv → 2 段 downsampling → ResnetBlock × n → 2 段 upsampling（ConvTranspose）→ 7×7 conv → Tanh）。論文 Sec. 7.2（256×256 用に 9 blocks）
  ResnetBlock            : 本家
  NLayerDiscriminator    : 本家 basic（70×70 PatchGAN、n_layers 3）。論文 Sec. 7.2
  GANLoss                : 本家（lsgan = MSE が既定 = 論文 Sec. 4、vanilla = BCEWithLogits）
  get_norm_layer / init_weights / init_net / linear scheduler : 本家
"""

import functools

import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type="instance"):
    """batch | instance | none（本家）。BatchNorm は affine + running 統計、InstanceNorm は affine なし・統計なし。"""
    if norm_type == "batch":
        return functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    if norm_type == "instance":
        return functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    if norm_type == "none":
        return lambda x: Identity()
    raise NotImplementedError("normalization layer [%s] is not found" % norm_type)


def init_weights(net, init_type="normal", init_gain=0.02):
    """本家。Conv / Linear の重みを init_type で、BatchNorm2d の weight を N(1, gain)・bias 0 に。"""

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
    """本家 lr_policy=linear: 最初の n_epochs は一定、次の n_epochs_decay で 0 へ線形（論文 Sec. 4）。"""

    def lambda_rule(epoch):
        return 1.0 - max(0, epoch + epoch_count - n_epochs) / float(n_epochs_decay + 1)

    return lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)


###############################################################################
# ResnetGenerator（本家）
###############################################################################
class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations."""

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type="reflect"):
        assert n_blocks >= 0
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
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
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block（本家）"""

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
# D: NLayerDiscriminator（本家 basic = 70×70 PatchGAN）
###############################################################################
class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias), norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)]
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias), norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


###############################################################################
# GANLoss（本家）
###############################################################################
class GANLoss(nn.Module):
    """lsgan = MSE（本家既定 = 論文 Sec. 4）| vanilla = BCEWithLogits。D の出力は生の logit。"""

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
# 組み立て（本家 define_G / define_D の必要分）
###############################################################################
N_BLOCKS = {"resnet_9blocks": 9, "resnet_6blocks": 6}


def build_G(cfg, device):
    """cfg = train の平坦 dict（network.*）。G_A / G_B とも同じ構造（1ch → 1ch、同解像度）。本家は no_dropout=True（論文も dropout 無し）。"""
    norm_layer = get_norm_layer(cfg["network.norm"])
    net = ResnetGenerator(1, 1, cfg["network.ngf"], norm_layer=norm_layer, use_dropout=False, n_blocks=N_BLOCKS[cfg["network.netG"]])
    return init_net(net, cfg["network.init_type"], cfg["network.init_gain"], device)


def build_D(cfg, device):
    norm_layer = get_norm_layer(cfg["network.norm"])
    net = NLayerDiscriminator(1, cfg["network.ndf"], n_layers=cfg["network.n_layers_D"], norm_layer=norm_layer)
    return init_net(net, cfg["network.init_type"], cfg["network.init_gain"], device)


def count_params(net):
    return sum(p.numel() for p in net.parameters())
