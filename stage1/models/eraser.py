"""stage1/models/eraser.py — 型 Y の「消し手」A: 近くの画素の重みつき平均しかできない小さな CNN（2026-11-11）。

  出力 m(p) = Σ_{q ∈ 近傍(p)} w_p(q) · x(q),  w_p = softmax（画素ごとに足して 1、すべて 0 以上）
  作りで保証されること:
    - 一様な領域の値は変わらない（明るさ・造影の濃さを変えられない）
    - 近傍の最小〜最大を超えない（新しい極値 = 入力に無い構造を作れない、HU の上振れが出ない）
    - 鋭くはできない（平均だけ）
  重みは場所と中身で変わる（縁は狭く、平らな所は広く平均できる）。近傍の広さ（radius）は上限であって、ぼかし量ではない。
  先例: Kernel Prediction Networks（Mildenhall et al., CVPR 2018、画素ごとの重みを予測して平均する）。

  初期値: 最後の Conv の重みを 0、中央の画素の logit だけ center_bias にする（最初は恒等に近い。学習が平均を増やす）。
  入力 x は HU ÷ 1000 の単位。端は reflect。FP32 で計算する（小さいので速度の問題はない）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Block(nn.Module):
    def __init__(self, ch, dilation):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=dilation, dilation=dilation, padding_mode="reflect")
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, padding_mode="reflect")

    def forward(self, h):
        return h + self.c2(F.silu(self.c1(F.silu(h))))


class Eraser(nn.Module):
    def __init__(self, radius, width, depth, center_bias):
        super().__init__()
        self.radius = int(radius)
        k = 2 * self.radius + 1
        self.k2 = k * k
        self.stem = nn.Conv2d(1, width, 3, padding=1, padding_mode="reflect")
        self.blocks = nn.Sequential(*[_Block(width, 2 ** (i % 4)) for i in range(depth)])
        self.out = nn.Conv2d(width, self.k2, 1)
        nn.init.zeros_(self.out.weight)
        with torch.no_grad():
            self.out.bias.zero_()
            self.out.bias[self.k2 // 2] = float(center_bias)

    def weights(self, x):
        """x (B, 1, H, W) → 近傍の重み (B, k², H, W)（softmax 済み）。"""
        return torch.softmax(self.out(F.silu(self.blocks(self.stem(x.float())))), dim=1)

    def forward(self, x, return_weights=False):
        x = x.float()
        b, _, h, w = x.shape
        wts = self.weights(x)
        r = self.radius
        patches = F.unfold(F.pad(x, (r, r, r, r), mode="reflect"), kernel_size=2 * r + 1).view(b, self.k2, h, w)
        m = (wts * patches).sum(dim=1, keepdim=True)
        return (m, wts) if return_weights else m


class IsoEraser(Eraser):
    """向きのないぼかしだけの消し手（2026-11-11 夕方、EXP-20261111-03 の 2 回目の調整）。
    画素ごとに広さ σ(p) だけを選ぶ: w_p(q) ∝ exp(−|q − p|² / (2σ(p)²))（近傍 (2r+1)² で正規化）。σ(p) = r · sigmoid(logit)。
    重みは p を中心に左右上下対称なので、明るい側を片側へ広げる・細い構造を動かすことができない（自由な重みの Eraser は 1 回目にそれをした）。
    σ → 0 で恒等。初期値は logit = −center_bias（σ ≈ r · e^−center_bias、ほぼ恒等）。"""

    def __init__(self, radius, width, depth, center_bias):
        super().__init__(radius, width, depth, center_bias)
        self.out = nn.Conv2d(width, 1, 1)
        nn.init.zeros_(self.out.weight)
        with torch.no_grad():
            self.out.bias.fill_(-float(center_bias))
        r = self.radius
        yy, xx = torch.meshgrid(torch.arange(-r, r + 1.0), torch.arange(-r, r + 1.0), indexing="ij")
        self.register_buffer("d2", (yy ** 2 + xx ** 2).reshape(1, -1, 1, 1), persistent=False)

    def sigma_px(self, x):
        return self.radius * torch.sigmoid(self.out(F.silu(self.blocks(self.stem(x.float())))))

    def weights(self, x):
        s = self.sigma_px(x).clamp_min(1e-3)
        return torch.softmax(-self.d2 / (2.0 * s * s), dim=1)


class SteeredEraser(IsoEraser):
    """縁に沿った向きのぼかし（2026-11-12 夜、EXP-20261112-01 の調整 4c）。画素ごとに 2 つの広さ σ_u, σ_v と向き θ を選ぶ Gaussian の重み:
      w_p(q) ∝ exp(−½ (d_u² / σ_u² + d_v² / σ_v²)),  d = q − p,  d_u = d を θ 方向へ射影, d_v = 直交方向
    重みは p について点対称（w(d) = w(−d)）なので、平均の中心は動かない（1 回目の自由な重みのように明るい側を片側へ広げることはできない）。
    向きのないぼかし（isotropic）は σ_u = σ_v の特別な場合。
    根拠（分析 job 032423）: 向きのないぼかしでは、縁の上で「PCD の粒を消す」と「縁を太らせない」が両立せず、縁の幅が 5.6 px（実 EID 4.5）に広がった。
    縁に沿う向きにだけ広く平均すれば、粒を消しつつ縁を横切る向きのぼけを小さくできる（steering kernel: Takeda, Farsiu, Milanfar, IEEE TIP 2007）。
    向きは (c, s) = (cos 2θ, sin 2θ) を正規化せずに出し、|(c, s)| < 1 の分は向きのない平均に近づく（c = s = 0 で d_u² = d_v² = |d|²/2）。
    初期値: logit σ_u, σ_v = −center_bias ± 0.3（r = 4、center_bias 2.2 で 0.52 / 0.30 px。ほぼ恒等）、c = s = 0。
    σ_u = σ_v だと向きの勾配が 0 になり（d_u² と d_v² の重みが同じ）、向きのないぼかしの中から出られないので、少しだけ違えて始める。"""

    def __init__(self, radius, width, depth, center_bias):
        super().__init__(radius, width, depth, center_bias)
        self.out = nn.Conv2d(width, 4, 1)   # logit σ_u, logit σ_v, c, s
        nn.init.zeros_(self.out.weight)
        with torch.no_grad():
            self.out.bias.zero_()
            self.out.bias[0] = -float(center_bias) + 0.3   # σ_u と σ_v を少しだけ違えて始める。同じだと向き (c, s) の勾配が 0 で、向きのないぼかしから出られない
            self.out.bias[1] = -float(center_bias) - 0.3
        r = self.radius
        yy, xx = torch.meshgrid(torch.arange(-r, r + 1.0), torch.arange(-r, r + 1.0), indexing="ij")
        self.register_buffer("dxx", (xx * xx).reshape(1, -1, 1, 1), persistent=False)
        self.register_buffer("dyy", (yy * yy).reshape(1, -1, 1, 1), persistent=False)
        self.register_buffer("dxy", (xx * yy).reshape(1, -1, 1, 1), persistent=False)

    def params(self, x):
        o = self.out(F.silu(self.blocks(self.stem(x.float()))))
        su = (self.radius * torch.sigmoid(o[:, 0:1])).clamp_min(1e-3)
        sv = (self.radius * torch.sigmoid(o[:, 1:2])).clamp_min(1e-3)
        n = torch.sqrt(o[:, 2:3] ** 2 + o[:, 3:4] ** 2 + 1e-2)   # |(c, s)| < 1（柔らかい正規化。0 付近でも勾配が有限）
        return su, sv, o[:, 2:3] / n, o[:, 3:4] / n

    def sigma_px(self, x):
        su, sv, _, _ = self.params(x)
        return torch.sqrt(su * sv)

    def weights(self, x):
        su, sv, c, s = self.params(x)
        du2 = 0.5 * (1 + c) * self.dxx + 0.5 * (1 - c) * self.dyy + s * self.dxy
        dv2 = 0.5 * (1 - c) * self.dxx + 0.5 * (1 + c) * self.dyy - s * self.dxy
        return torch.softmax(-0.5 * (du2 / (su * su) + dv2 / (sv * sv)), dim=1)


class GlobalEraser(nn.Module):
    """画像全体で 1 つの広さ σ の向きのないぼかし（2026-11-13、EXP-20261113-01 の検証）。学習する数は θ の 1 個だけ: σ = r · sigmoid(θ)。
    重みの式は IsoEraser と同じ（w(q) ∝ exp(−|q − p|² / (2σ²))、近傍 (2r+1)² で正規化）で、σ がどの画素でも同じ。入力の絵は見ない（CNN は無い。width・depth は使わない）。
    根拠: IsoEraser の学習後の σ(p) は縁 約 1.7 px・平ら 約 1.9 px とほぼ一様で、場所ごとに σ を選ぶ理由が弱い（EXP-20261112-01 の分析）。
    初期値は θ = −center_bias（IsoEraser の初期値と同じ σ ≈ r · sigmoid(−center_bias)）。"""

    def __init__(self, radius, width, depth, center_bias):
        super().__init__()
        self.radius = int(radius)
        k = 2 * self.radius + 1
        self.k2 = k * k
        self.theta = nn.Parameter(torch.tensor(-float(center_bias)))
        r = self.radius
        yy, xx = torch.meshgrid(torch.arange(-r, r + 1.0), torch.arange(-r, r + 1.0), indexing="ij")
        self.register_buffer("d2", (yy ** 2 + xx ** 2).reshape(1, -1, 1, 1), persistent=False)

    def sigma_px(self, x):
        """(B, 1, H, W)。どの画素も同じ値（ログと作成時の a_sigma_scale が IsoEraser と同じ形で働くように、絵の形に広げて返す）。"""
        return (self.radius * torch.sigmoid(self.theta)).expand(x.shape[0], 1, x.shape[2], x.shape[3])

    def weights(self, x):
        s = self.sigma_px(x).clamp_min(1e-3)
        return torch.softmax(-self.d2 / (2.0 * s * s), dim=1)

    forward = Eraser.forward


def build_eraser(cfg):
    cls = {"free": Eraser, "isotropic": IsoEraser, "steered": SteeredEraser, "global": GlobalEraser}[cfg["eraser.kind"]]
    return cls(cfg["eraser.radius"], cfg["eraser.width"], cfg["eraser.depth"], cfg["eraser.center_bias"])


def fill_legacy(cfg):
    """eraser.kind が無い重み（2026-11-11 17:07 の run など、足す前に作ったもの）は free。足したら表示する。"""
    if "eraser.kind" not in cfg:
        cfg["eraser.kind"] = "free"
        print("[eraser] 重みの設定に eraser.kind が無い（足す前の run）→ free として読む", flush=True)
    return cfg
