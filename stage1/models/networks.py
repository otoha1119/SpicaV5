import math
import zlib

import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from .wavelet_generator import WaveletGenerator  # [SpicaV5] Park et al. 2019 Fig. 3 の generator
from .paper_discriminator import PaperDiscriminator  # [SpicaV5] Park et al. 2019 Fig. 3 の discriminator


###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


class ResidualGenerator(nn.Module):
    """[SpicaV5] Residual 化（段階 5-1、2026-09-11）: G(z) = z + F(z)。F は包んだ generator（線形出力）で、その生出力が残差 R。
    損失は変えない（fidelity ‖G(z) − z‖² = ‖F(z)‖²）。入出力のチャネル数が同じときだけ使える。state_dict のキーには inner. が付く
    （非 residual の重みを residual 構成に strict load すると明示的に落ちる）。mode.yaml の residual で切替、false なら包まない。
    noise_input（段階 5-5、2026-10-12）と併用するときは inner が NoiseInputGenerator で、G(z) = z + F(cat[z, n])（足すのは z だけ。n は inner が連結する）。
    forward のキーワード引数（n=）は inner に素通しする。"""

    def __init__(self, inner, input_nc, output_nc):
        super().__init__()
        if input_nc != output_nc:
            raise ValueError(f"residual は input_nc == output_nc が必要です: {input_nc} != {output_nc}")
        self.inner = inner

    def forward(self, x, **kw):
        return x + self.inner(x, **kw)


class NoiseInputGenerator(nn.Module):
    """[SpicaV5] 乱数画像 n の入力（段階 5-5、2026-10-12、ユーザー採用。論文 Park 2019 には無い独自機構）:
    F(cat[x, n])。n = randn(B,1,H,W) × sigma（1 ch の白色ガウス、x と同じ大きさ）を x にチャネル連結して包んだ generator F に入れる。
    F の 1 層目の入力は input_nc + 1（define_G が F をこのチャネル数で組む）。residual と併用するときは ResidualGenerator が外側で z を足す。
    損失は不変（fidelity ‖G(z) − z‖² は n を含む G の出力で自動的に計算される）。state_dict のキーには inner. が付く。

    sigma は HU で設定し（train.yaml noise.sigma_hu_min / max）、正規化空間では sigma = sigma_hu / ((hu_max − hu_min) / 2)。
      学習（forward に n を渡さない）: サンプルごとに sigma_hu を log-uniform [min, max] から引き、n も毎回引く（torch のグローバル RNG。学習は毎回ランダム）
      推論・preview（再現性）: 呼ぶ側が slice_noise(name, hw, device, ...) で **スライス名から決まる seed**（zlib.crc32。Python の hash() はプロセスごとに変わるので使わない）
        と、その seed から引いた sigma_hu（sigma_mode random）または固定値（fixed）で n を作り、forward(x, n=n) で渡す。
        n は CPU の torch.Generator で作る（cuda / cpu で同じ n）。draw k（n_draws 用）は seed + k。sigma は draw によらず同じ（n の効きだけを見る）。
        patch 推論は 1 スライスにつき 512 全体の n を 1 枚作り、各 patch はその対応領域を切り出して使う（patch ごとに独立に引くと重なりの平均で粒が薄まる）。
    直近の学習 batch の sigma_hu は last_sigma_hu（TensorBoard diag/sigma_hu 用）。"""

    def __init__(self, inner, sigma_hu_min, sigma_hu_max, hu_min, hu_max):
        super().__init__()
        if not (0.0 < sigma_hu_min <= sigma_hu_max):
            raise ValueError(f"noise.sigma_hu_min / max は 0 < min ≤ max: {sigma_hu_min} / {sigma_hu_max}")
        if hu_min >= hu_max:
            raise ValueError(f"hu_min < hu_max: {hu_min} / {hu_max}")
        self.inner = inner
        self.sigma_hu_min = float(sigma_hu_min)
        self.sigma_hu_max = float(sigma_hu_max)
        self.hu_per_unit = (float(hu_max) - float(hu_min)) / 2.0  # 正規化 [−1,1] の 1.0 が何 HU か
        self.last_sigma_hu = None

    # ---- 再現用 API（推論・preview・crop が使う）----
    @staticmethod
    def slice_seed(name):
        """スライス名（例 PCD-002-215。拡張子なしの stem）→ 決定的な 32bit 整数。"""
        return zlib.crc32(str(name).encode("utf-8")) & 0xFFFFFFFF

    def _sigma_hu_from_u(self, u):
        """u ∈ [0,1) → log-uniform [sigma_hu_min, sigma_hu_max]。"""
        lo, hi = math.log(self.sigma_hu_min), math.log(self.sigma_hu_max)
        return math.exp(lo + u * (hi - lo))

    def slice_noise(self, name, hw, device, sigma_mode="random", sigma_hu_fixed=None, draw=0):
        """スライス名から決定的に n を作る。戻り値 (n (1,1,H,W) on device, seed, sigma_hu)。
        generator は seed + draw で初期化し、先頭の rand 1 個を sigma 用の u（draw 0 のときだけ使う。sigma は draw によらず draw 0 のもの）、続く randn を n に使う。
        sigma_mode: random = draw 0 の u から log-uniform | fixed = sigma_hu_fixed（n の乱数列は random と同じなので、違いは振幅だけ）。"""
        seed = self.slice_seed(name)
        g0 = torch.Generator(device="cpu").manual_seed(seed)
        u0 = torch.rand(1, generator=g0).item()
        if sigma_mode == "random":
            sigma_hu = self._sigma_hu_from_u(u0)
        elif sigma_mode == "fixed":
            if sigma_hu_fixed is None or sigma_hu_fixed <= 0:
                raise ValueError(f"sigma_mode fixed には正の sigma_hu_fixed が必要です: {sigma_hu_fixed!r}")
            sigma_hu = float(sigma_hu_fixed)
        else:
            raise ValueError(f"sigma_mode は random | fixed: {sigma_mode!r}")
        g = g0 if draw == 0 else torch.Generator(device="cpu").manual_seed(seed + int(draw))
        if draw != 0:
            torch.rand(1, generator=g)  # draw 0 と同じ位置から randn を取る（先頭の u は捨てる）
        h, w = hw
        n = torch.randn(1, 1, int(h), int(w), generator=g) * (sigma_hu / self.hu_per_unit)
        return n.to(device), seed, sigma_hu

    def random_noise(self, x):
        """学習用: x (B,1,H,W) と同じ形の n を、サンプルごとの log-uniform sigma で引く（グローバル RNG、x のデバイス）。sigma_hu を last_sigma_hu に残す。"""
        b = x.shape[0]
        lo, hi = math.log(self.sigma_hu_min), math.log(self.sigma_hu_max)
        sigma_hu = torch.exp(lo + torch.rand(b, device=x.device, dtype=x.dtype) * (hi - lo))  # (B,)
        self.last_sigma_hu = sigma_hu.detach()
        n = torch.randn(b, 1, x.shape[-2], x.shape[-1], device=x.device, dtype=x.dtype) * (sigma_hu / self.hu_per_unit).view(b, 1, 1, 1)
        return n

    def forward(self, x, n=None):
        """n を渡さなければ毎回ランダム（学習）。渡せばそれを使う（推論・preview の再現用。形は x と同じ (B,1,H,W)）。"""
        if n is None:
            n = self.random_noise(x)
        elif n.shape != (x.shape[0], 1, x.shape[-2], x.shape[-1]):
            raise ValueError(f"n の形 {tuple(n.shape)} が x {tuple(x.shape)} と合いません（(B,1,H,W)）")
        return self.inner(torch.cat([x, n.to(x.dtype)], dim=1))


def noise_wrapper(net):
    """G の包みの中から NoiseInputGenerator を探して返す（無ければ None）。DDP / torch.compile / ResidualGenerator の包みを剥がして .inner を辿る。
    呼ぶ側（monitor / inference_dir / crop_patches）はこれで「n あり」を判別し、再現用の n を slice_noise で作って netG(x, n=n) に渡す。"""
    net = net.module if hasattr(net, "module") else net
    net = net._orig_mod if hasattr(net, "_orig_mod") else net
    while net is not None:
        if isinstance(net, NoiseInputGenerator):
            return net
        net = getattr(net, "inner", None)
    return None


def get_norm_layer(norm_type="instance"):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == "batch":
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == "syncbatch":
        norm_layer = functools.partial(nn.SyncBatchNorm, affine=True, track_running_stats=True)
    elif norm_type == "instance":
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == "none":

        def norm_layer(x):
            return Identity()

    else:
        raise NotImplementedError("normalization layer [%s] is not found" % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == "linear":

        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == "step":
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == "plateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == "cosine":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError("learning rate policy [%s] is not implemented", opt.lr_policy)
    return scheduler


def init_weights(net, init_type="normal", init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
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
        elif classname.find("BatchNorm2d") != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print("initialize network with %s" % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type="normal", init_gain=0.02):
    """Initialize a network: 1. register CPU/GPU device; 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.

    Return an initialized network.
    """
    import os

    if torch.cuda.is_available():
        if "LOCAL_RANK" in os.environ:
            local_rank = int(os.environ["LOCAL_RANK"])
            net.to(local_rank)
            print(f"Initialized with device cuda:{local_rank}")
        else:
            net.to(0)
            print("Initialized with device cuda:0")
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm="batch", use_dropout=False, init_type="normal", init_gain=0.02, final_norm_act=False, residual=False, noise_input=False, sigma_hu=None, hu=None):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_128 | unet_256
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.

    [SpicaV5] noise_input（段階 5-5、2026-10-12）: true なら F の 1 層目を input_nc + 1 ch で組み、NoiseInputGenerator で包む（sigma_hu = (min, max) [HU]、hu = (hu_min, hu_max) が必須）。
    residual と併用するときの包みの順は ResidualGenerator(NoiseInputGenerator(F))（G(z) = z + F(cat[z, n])。state_dict のキーは inner.inner.…）。

    Returns a generator
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    f_in = input_nc + 1 if noise_input else input_nc  # [SpicaV5] n を 1 ch 連結する分

    if netG == "resnet_9blocks":
        net = ResnetGenerator(f_in, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == "resnet_6blocks":
        net = ResnetGenerator(f_in, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == "unet_128":
        net = UnetGenerator(f_in, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == "unet_256":
        net = UnetGenerator(f_in, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == "wavelet":  # [SpicaV5] Park et al. 2019 (deep convolutional framelet)。論文どおり BatchNorm 固定のため norm_layer / use_dropout は使わない
        net = WaveletGenerator(f_in, output_nc, ngf=ngf, final_norm_act=final_norm_act)
    else:
        raise NotImplementedError("Generator model name [%s] is not recognized" % netG)
    if noise_input:  # [SpicaV5] 段階 5-5: F(cat[z, n])（mode.yaml noise_input）
        if sigma_hu is None or hu is None:
            raise ValueError("noise_input には sigma_hu=(sigma_hu_min, sigma_hu_max) と hu=(hu_min, hu_max) が必要です")
        net = NoiseInputGenerator(net, sigma_hu[0], sigma_hu[1], hu[0], hu[1])
    if residual:  # [SpicaV5] 段階 5-1: G(z) = z + F(z)（mode.yaml residual）。noise_input のときは z + F(cat[z, n])
        net = ResidualGenerator(net, input_nc, output_nc)
    return net


def define_D(input_nc, ndf, netD, n_layers_D=3, norm="batch", init_type="normal", init_gain=0.02):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == "basic":  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == "n_layers":  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == "pixel":  # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    elif netD == "paper":  # [SpicaV5] Park et al. 2019: 4×4 s2 ×3 (ndf, ndf, 4ndf) + 1×1、受容野 22。論文どおり BatchNorm 固定のため norm_layer は使わない
        net = PaperDiscriminator(input_nc, ndf=ndf)
    else:
        raise NotImplementedError("Discriminator model name [%s] is not recognized" % netD)
    return net


##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer("real_label", torch.tensor(target_real_label))
        self.register_buffer("fake_label", torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == "lsgan":
            self.loss = nn.MSELoss()
        elif gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ["wgangp"]:
            self.loss = None
        else:
            raise NotImplementedError("gan mode %s not implemented" % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ["lsgan", "vanilla"]:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == "wgangp":
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type="mixed", constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == "real":  # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == "fake":
            interpolatesv = fake_data
        elif type == "mixed":
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError(f"{type} not implemented")
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv, grad_outputs=torch.ones(disc_interpolates.size()).to(device), create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp  # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type="reflect"):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
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
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
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
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):  # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
    X -------------------identity----------------------
    |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:  # add skip connections
            return torch.cat([x, self.model(x)], 1)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

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
        """Standard forward."""
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias),
        ]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)
