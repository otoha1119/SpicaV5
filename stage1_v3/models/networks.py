"""stage1_v3/models/networks.py — G（学習残差 R）と D。stage1/ の Park 2019 実装を複製し、ablation 用に junyanz の resnet / basic も置く。

  WaveletGenerator    : 論文 Fig. 3 の deep convolutional framelet（stage1/models/wavelet_generator.py の複製。3×3 conv → BN → LReLU、Haar wave-dec / wave-rec、HF skip、最終 3×3 conv、活性化なし）
  PaperDiscriminator  : 論文 Fig. 3 の PatchGAN（stage1/models/paper_discriminator.py の複製。4×4 s2 conv → BN → LReLU ×3、1×1 conv。受容野 22×22、R-S11）
  ResnetGenerator     : junyanz resnet_9blocks / resnet_6blocks（ablation。残差 R を出すので最終 Tanh は付けない）
  NLayerDiscriminator : junyanz basic 70×70 PatchGAN（ablation）
  ResidualGenerator   : out = in + R(in)。R = 包んだ generator の生出力（stage1 の residual: true と同じ。常に使う）
  GANLoss             : lsgan / vanilla（ablation。fgan_kl は models/projection_model.py が D の生出力 v から直接計算する）
"""

import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.optim import lr_scheduler


# ---------------------------------------------------------------------------
# 共通
# ---------------------------------------------------------------------------
class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type="instance"):
    if norm_type == "batch":
        return functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    if norm_type == "instance":
        return functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    if norm_type == "none":
        return lambda x: Identity()
    raise NotImplementedError("normalization layer [%s] is not found" % norm_type)


def init_weights(net, init_type="normal", init_gain=0.02):
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
    def lambda_rule(epoch):
        return 1.0 - max(0, epoch + epoch_count - n_epochs) / float(n_epochs_decay + 1)

    return lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)


class ResidualGenerator(nn.Module):
    """out = in + R(in)。state_dict のキーに inner. が付く。"""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        return x + self.inner(x)

    def residual(self, x):
        return self.inner(x)


# ---------------------------------------------------------------------------
# Park 2019 G: 2-D Haar wave-dec / wave-rec（固定係数、depthwise）
# ---------------------------------------------------------------------------
def _haar_kernels(mode: str):
    L = torch.tensor([1.0, 1.0])
    H = torch.tensor([1.0, -1.0])
    k = torch.stack([torch.outer(L, L), torch.outer(L, H), torch.outer(H, L), torch.outer(H, H)])
    if mode == "ortho":
        return k * 0.5, k * 0.5
    if mode == "unnormalized":
        return k, k * 0.25
    raise ValueError(f"unknown wavelet normalization: {mode}")


class HaarDWT(nn.Module):
    def __init__(self, mode: str = "ortho"):
        super().__init__()
        dec, _ = _haar_kernels(mode)
        self.register_buffer("kernel", dec.unsqueeze(1))

    def forward(self, x):
        n, c, h, w = x.shape
        assert h % 2 == 0 and w % 2 == 0, f"wave-dec には偶数サイズが必要: {(h, w)}"
        weight = self.kernel.repeat(c, 1, 1, 1)
        y = F.conv2d(x, weight, stride=2, groups=c).view(n, c, 4, h // 2, w // 2)
        return y[:, :, 0], y[:, :, 1:].reshape(n, 3 * c, h // 2, w // 2)


class HaarIDWT(nn.Module):
    def __init__(self, mode: str = "ortho"):
        super().__init__()
        _, rec = _haar_kernels(mode)
        self.register_buffer("kernel", rec.unsqueeze(1))

    def forward(self, ll, hf):
        n, c, h, w = ll.shape
        assert hf.shape[1] == 3 * c
        y = torch.cat([ll.unsqueeze(2), hf.view(n, c, 3, h, w)], dim=2).reshape(n, 4 * c, h, w)
        weight = self.kernel.repeat(c, 1, 1, 1)
        return F.conv_transpose2d(y, weight, stride=2, groups=c)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, slope=0.2):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(slope, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class WaveletGenerator(nn.Module):
    """論文 Fig. 3（ngf=32, n_levels=3: enc 32/64/128、bottleneck 128→256→128、dec、最終 3×3 conv → 1ch、活性化なし）。"""

    def __init__(self, input_nc=1, output_nc=1, ngf=32, n_levels=3, lrelu_slope=0.2, wavelet="ortho", final_kernel=3):
        super().__init__()
        chs = [ngf * (2**i) for i in range(n_levels)]
        self.n_levels = n_levels
        self.enc = nn.ModuleList()
        for k in range(n_levels):
            in_ch = input_nc if k == 0 else chs[k - 1]
            self.enc.append(nn.Sequential(ConvBlock(in_ch, chs[k], lrelu_slope), ConvBlock(chs[k], chs[k], lrelu_slope)))
        self.dwt = HaarDWT(wavelet)
        self.idwt = HaarIDWT(wavelet)
        self.bottleneck = nn.Sequential(ConvBlock(chs[-1], 2 * chs[-1], lrelu_slope), ConvBlock(2 * chs[-1], chs[-1], lrelu_slope))
        self.dec = nn.ModuleList()
        for k in range(n_levels - 1, 0, -1):
            self.dec.append(nn.Sequential(ConvBlock(2 * chs[k], chs[k], lrelu_slope), ConvBlock(chs[k], chs[k - 1], lrelu_slope)))
        self.dec_top = ConvBlock(2 * chs[0], chs[0], lrelu_slope)
        self.final = nn.Conv2d(chs[0], output_nc, kernel_size=final_kernel, padding=final_kernel // 2)

    def forward(self, x):
        assert x.shape[-1] % (2**self.n_levels) == 0 and x.shape[-2] % (2**self.n_levels) == 0, f"入力サイズは {2**self.n_levels} の倍数が必要: {tuple(x.shape[-2:])}"
        skips = []
        h = x
        for k in range(self.n_levels):
            h = self.enc[k](h)
            e = h
            ll, hf = self.dwt(h)
            skips.append((e, hf))
            h = ll
        h = self.bottleneck(h)
        for i, k in enumerate(range(self.n_levels - 1, 0, -1)):
            e, hf = skips[k]
            h = self.idwt(h, hf)
            h = torch.cat([h, e], dim=1)
            h = self.dec[i](h)
        e, hf = skips[0]
        h = self.idwt(h, hf)
        h = torch.cat([h, e], dim=1)
        h = self.dec_top(h)
        return self.final(h)


# ---------------------------------------------------------------------------
# Park 2019 D
# ---------------------------------------------------------------------------
class PaperDiscriminator(nn.Module):
    """[Conv4×4 s2 → BN → LReLU(0.2)] ×3（32, 32, 128）→ Conv1×1 → 1ch（活性化なし。D := 1 − exp(v) は損失側）。受容野 22×22。"""

    def __init__(self, input_nc=1, ndf=32, lrelu_slope=0.2, padding=1):
        super().__init__()
        chs = [ndf, ndf, 4 * ndf]
        layers = []
        in_ch = input_nc
        for out_ch in chs:
            layers += [nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=padding, bias=False), nn.BatchNorm2d(out_ch), nn.LeakyReLU(lrelu_slope, inplace=True)]
            in_ch = out_ch
        layers += [nn.Conv2d(in_ch, 1, kernel_size=1, stride=1, padding=0)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# ---------------------------------------------------------------------------
# junyanz resnet G / basic D（ablation）
# ---------------------------------------------------------------------------
class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer, use_bias):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias), norm_layer(dim), nn.ReLU(True),
            nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias), norm_layer(dim))

    def forward(self, x):
        return x + self.conv_block(x)


class ResnetGenerator(nn.Module):
    """junyanz resnet。残差 R を出すので最終 Tanh は付けない。"""

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, n_blocks=9):
        super().__init__()
        use_bias = (norm_layer.func == nn.InstanceNorm2d) if isinstance(norm_layer, functools.partial) else (norm_layer == nn.InstanceNorm2d)
        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias), norm_layer(ngf), nn.ReLU(True)]
        for i in range(2):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias), norm_layer(ngf * mult * 2), nn.ReLU(True)]
        mult = 4
        model += [ResnetBlock(ngf * mult, norm_layer, use_bias) for _ in range(n_blocks)]
        for i in range(2):
            mult = 2 ** (2 - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias), norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super().__init__()
        use_bias = (norm_layer.func == nn.InstanceNorm2d) if isinstance(norm_layer, functools.partial) else (norm_layer == nn.InstanceNorm2d)
        kw, padw = 4, 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev, nf_mult = nf_mult, min(2**n, 8)
            sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias), norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)]
        nf_mult_prev, nf_mult = nf_mult, min(2**n_layers, 8)
        sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias), norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)


class GANLoss(nn.Module):
    """lsgan = MSE | vanilla = BCEWithLogits（ablation）。"""

    def __init__(self, gan_mode):
        super().__init__()
        self.register_buffer("real_label", torch.tensor(1.0))
        self.register_buffer("fake_label", torch.tensor(0.0))
        if gan_mode == "lsgan":
            self.loss = nn.MSELoss()
        elif gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError("gan mode %s not implemented" % gan_mode)

    def __call__(self, prediction, target_is_real):
        target = (self.real_label if target_is_real else self.fake_label).expand_as(prediction)
        return self.loss(prediction, target)


# ---------------------------------------------------------------------------
# 組み立て
# ---------------------------------------------------------------------------
N_BLOCKS = {"resnet_9blocks": 9, "resnet_6blocks": 6}


def build_G(cfg, device):
    """cfg = train の平坦 dict（network.*）。常に ResidualGenerator で包む（out = in + R）。"""
    netG = cfg["network.netG"]
    if netG == "wavelet":
        inner = WaveletGenerator(1, 1, ngf=cfg["network.ngf"])
    elif netG in N_BLOCKS:
        inner = ResnetGenerator(1, 1, cfg["network.ngf"], norm_layer=get_norm_layer(cfg["network.norm"]), n_blocks=N_BLOCKS[netG])
    else:
        raise ValueError(f"network.netG: {netG}")
    return init_net(ResidualGenerator(inner), cfg["network.init_type"], cfg["network.init_gain"], device)


def build_D(cfg, device):
    netD = cfg["network.netD"]
    if netD == "paper":
        net = PaperDiscriminator(1, ndf=cfg["network.ndf"])
    elif netD == "basic":
        net = NLayerDiscriminator(1, cfg["network.ndf"], n_layers=3, norm_layer=get_norm_layer(cfg["network.norm"]))
    else:
        raise ValueError(f"network.netD: {netD}")
    return init_net(net, cfg["network.init_type"], cfg["network.init_gain"], device)


def count_params(net):
    return sum(p.numel() for p in net.parameters())
