"""stage2/models/flow_net.py — V のネットワーク（設計書 C-2）。畳み込みだけ（注意機構なし。パディングは zeros）。

  FlowNet.forward(z, t, sigma_s)   … **引数はこの 3 つだけ**（設計書 C-3 の担保: U の出力は乱数に埋めた z としてだけ入る。EID・μ を別に受け取る口は無い）
    z       (B, 1, H, W)  V の中の値（HU ÷ flow.unit_hu）。H, W は 2 × 2^(段数 − 1) の倍数（[1, 2, 4, 4] なら 16）
    t       (B,)          時刻 0〜1
    sigma_s (B,)          乱数の強さ（z と同じ単位）

  embed : t × 1000 と 250 × ln(σs ÷ 100 HU) を、それぞれ正弦波 emb_freq_dim 次元 → MLP（→ emb_dim → emb_dim、SiLU）→ 足す。FP32
  stem  : Haar 1 段（1ch・H×W → 4ch・H/2×W/2 [LL, LH, HL, HH]）→ Conv3×3（4 → base_ch）。FP32
  body  : ResBlock U-Net（base_ch × ch_mult）。下りは各段 ResBlock × blocks_per_level、段の間は stride 2 の Conv3×3。中央 ResBlock × mid_blocks。
          上りは各段 ResBlock × blocks_per_level（1 個目で同じ段の下りの出力を concat）、段の間は最近傍 ×2 + Conv3×3。
          **bf16 autocast と torch.compile はこのモジュールにだけ掛ける**（autocast の切り替えを compile する範囲の外に出す）
  head  : LN → SiLU → Conv3×3（base_ch → 4、ゼロ初期化）→ 逆 Haar → 1ch。FP32。学習開始時の出力は 0（= 動かない）

  ResBlock: LN → SiLU → Conv3×3 → + Linear(埋め込み) → LN → SiLU → Conv3×3（ゼロ初期化）+ 近道（幅が変わる所は 1×1 conv）
  LN（LayerNorm2d）: **画素ごと**にチャネル方向で平均 0・分散 1（+ チャネルごとの重みと偏り）。画像の広い範囲の統計を使わないので、256 パッチ（学習）と 1024 一枚（推論）で同じ関数になる。
                     統計は FP32 で計算する（bf16 は有効 8 ビット。平均を引く計算を丸めない）。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.haar import haar_fold, haar_unfold

T_SCALE = 1000.0      # 埋め込みに入れる t の倍率（設計書 C-2）
SIGMA_SCALE = 250.0   # 同 ln(σs ÷ 100 HU) の倍率（25〜400 HU が ±347 になる。設計書は倍率を決めていないので、t と同じ桁になるように置いた。UPSTREAM.md）
SIGMA_REF_HU = 100.0


def zero_(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


def sinusoid(v, dim, max_period=10000.0):
    """v (B,) → (B, dim)。拡散モデルで標準の正弦波の埋め込み（周波数は 1 から 1 / max_period まで対数で等間隔）。"""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=v.device, dtype=torch.float32) / half)
    ang = v.float()[:, None] * freqs[None, :]
    return torch.cat([ang.cos(), ang.sin()], dim=1)


class LayerNorm2d(nn.Module):
    """画素ごと（チャンネル方向）の LayerNorm。weight・bias は (ch,) で持ち、使うときに (1, ch, 1, 1) に並べる。
    (1, ch, 1, 1) のまま持つと、channels_last + torch.compile（inductor）で backward が落ちる
    （CompiledFunctionBackward returned an invalid gradient: got [ch] but expected [1, ch, 1, 1]。PC3・torch 2.10 で切り分け済み。
    bf16 の有無・compile の mode には依らず、aot_eager では通る = inductor 側の不具合。2026-11-13、EXP-20261112-01）。
    前の形で保存した重み（2026-11-13 までの V）は読み込むときに (ch,) に直す。"""

    def __init__(self, ch, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ch))
        self.bias = nn.Parameter(torch.zeros(ch))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        for k in ("weight", "bias"):
            v = state_dict.get(prefix + k)
            if v is not None and v.dim() == 4:
                state_dict[prefix + k] = v.reshape(-1)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, x):
        x = x.float()
        m = x.mean(dim=1, keepdim=True)
        d = x - m
        return d * torch.rsqrt(d.square().mean(dim=1, keepdim=True) + self.eps) * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class Embed(nn.Module):
    def __init__(self, freq_dim, dim, sigma_ref):
        super().__init__()
        self.freq_dim, self.sigma_ref = freq_dim, float(sigma_ref)
        self.t_mlp = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.s_mlp = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t, sigma_s):
        return self.t_mlp(sinusoid(t.float() * T_SCALE, self.freq_dim)) + self.s_mlp(sinusoid((sigma_s.float() / self.sigma_ref).log() * SIGMA_SCALE, self.freq_dim))


class ResBlock(nn.Module):
    def __init__(self, cin, cout, emb_dim):
        super().__init__()
        self.norm1 = LayerNorm2d(cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.emb = nn.Linear(emb_dim, cout)
        self.norm2 = LayerNorm2d(cout)
        self.conv2 = zero_(nn.Conv2d(cout, cout, 3, padding=1))
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb(emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return self.skip(x) + h


class Down(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Up(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        # 最近傍 ×2。F.interpolate は使わない（bf16 の最近傍補間は torch の版によって CUDA で未実装だったことがある。Stage 1 と同じ形の操作）
        n, c, h, w = x.shape
        return self.conv(x[:, :, :, None, :, None].expand(n, c, h, 2, w, 2).reshape(n, c, 2 * h, 2 * w))


class FlowBody(nn.Module):
    """stem の出力 (B, ch0, h, w) と埋め込み (B, D) → (B, ch0, h, w)。"""

    def __init__(self, net):
        super().__init__()
        chs = [net["net.base_ch"] * m for m in net["net.ch_mult"]]
        nb, D = net["net.blocks_per_level"], net["net.emb_dim"]
        self.down_blocks, self.downs = nn.ModuleList(), nn.ModuleList()
        c = chs[0]
        for i, ch in enumerate(chs):
            self.down_blocks.append(nn.ModuleList([ResBlock(c if j == 0 else ch, ch, D) for j in range(nb)]))
            c = ch
            if i < len(chs) - 1:
                self.downs.append(Down(c))
        self.mid = nn.ModuleList([ResBlock(c, c, D) for _ in range(net["net.mid_blocks"])])
        self.ups, self.up_blocks = nn.ModuleList(), nn.ModuleList()
        for i in reversed(range(len(chs))):
            if i < len(chs) - 1:
                self.ups.append(Up(c))
            self.up_blocks.append(nn.ModuleList([ResBlock(c + chs[i] if j == 0 else chs[i], chs[i], D) for j in range(nb)]))
            c = chs[i]

    def forward(self, h, emb):
        skips = []
        for i, blocks in enumerate(self.down_blocks):
            for blk in blocks:
                h = blk(h, emb)
            skips.append(h)
            if i < len(self.downs):
                h = self.downs[i](h)
        for blk in self.mid:
            h = blk(h, emb)
        for i, blocks in enumerate(self.up_blocks):
            if i > 0:
                h = self.ups[i - 1](h)
            h = torch.cat([h, skips.pop()], dim=1)
            for blk in blocks:
                h = blk(h, emb)
        return h


class Stem(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(4, ch, 3, padding=1)

    def forward(self, z):
        return self.conv(haar_fold(z))


class Head(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.norm = LayerNorm2d(ch)
        self.conv = zero_(nn.Conv2d(ch, 4, 3, padding=1))

    def forward(self, h):
        return haar_unfold(self.conv(F.silu(self.norm(h))))


class FlowNet(nn.Module):
    def __init__(self, net, unit_hu):
        super().__init__()
        ch0 = net["net.base_ch"] * net["net.ch_mult"][0]
        self.multiple = 2 * 2 ** (len(net["net.ch_mult"]) - 1)
        self.embed = Embed(net["net.emb_freq_dim"], net["net.emb_dim"], SIGMA_REF_HU / float(unit_hu))
        self.stem = Stem(ch0)
        self.body = FlowBody(net)
        self.head = Head(ch0)

    def forward(self, z, t, sigma_s):
        """素の 1 回の forward（全部同じ精度。CPU・FP32 の検算用）。学習・推論は flow_forward を通す。"""
        return flow_forward(self, z, t, sigma_s)


def flow_forward(net, z, t, sigma_s, body=None, bf16=False, channels_last=False, clone=False):
    """V の 1 回の forward。入口・出口・埋め込みは FP32、本体だけ bf16 autocast（bf16=True のとき）。body を渡すとその呼び出し口（compile 済みなど）で本体を通す。
    autocast の外で呼ぶこと。戻り値は FP32 の (B, 1, H, W)（進む向き。z と同じ単位）。"""
    z = z.float()
    if z.shape[-1] % net.multiple or z.shape[-2] % net.multiple:
        raise ValueError(f"V の入力の H, W は {net.multiple} の倍数である必要があります: {tuple(z.shape)}")
    emb = net.embed(t, sigma_s)
    h = net.stem(z)
    if channels_last:
        h = h.contiguous(memory_format=torch.channels_last)
    with torch.autocast(device_type=z.device.type, dtype=torch.bfloat16, enabled=bf16):
        h = (net.body if body is None else body)(h, emb)
    h = h.float()
    if clone:
        h = h.clone()
    return net.head(h)


def build_flow(v):
    return FlowNet(v, v["flow.unit_hu"])
