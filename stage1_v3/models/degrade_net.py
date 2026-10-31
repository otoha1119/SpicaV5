"""stage1_v3/models/degrade_net.py — 学習型の確率的劣化 P_θ（degrade.mode: learned。PDM 型、Luo et al., CVPR 2022 の定式化を CT 用に組み直したもの）。

  u = (x ⊛ k(z_k)) + σ(x) ⊙ (h(z_k) ⊛ ε)
    k(z_k) : ぼかしカーネル（K×K、softmax で非負・和 1）。乱数 z_k（z_dim）から MLP で生成。スライス（サンプル）ごとに別。重心が中心に留まる正則化（COM）を掛ける
    h(z_k) : ノイズの整形フィルタ（Kh×Kh、L2 ノルム 1）。同じ z_k から生成。NPS の形を担う
    ε      : 白色 Gaussian 画像（毎回引く）。ノイズの実現値はここからだけ来る
    σ(x)   : ノイズ振幅マップ [0, sigma_max]（正規化単位）。x から小さな CNN で作り、8×8 の平均プーリング → 双線形で戻して低周波に限定する（振幅は減弱経路に依存してよいが、微細構造を符号化できないように）

  なぜこの因子分解か（固定 P からの転換理由と、学習型で漏れを防ぐ条件。docs/decisions/20261030_stage1-v3-learned-degradation.md）:
    ・ノイズの実現値は ε だけから来るので、u は x の決定的関数にならない（R-S21）。E[u | x] = x ⊛ k で、ノイズの条件付き平均は 0（C2N の stabilizing と同じ性質）
    ・fidelity はカーネル側 ‖x ⊛ k − x‖² にだけ掛け、ノイズには掛けない（掛けると ε 間の分散が罰されてノイズが消える。005 の n 無視の再来）
    ・ノイズネットに x を直接入れて実現値を出させると、z を無視して内容依存の決定的テクスチャ（漏れ）を出せる。振幅 σ(x) を低周波に限定し、実現値は ε ⊛ h に固定することでそれを塞ぐ
    ・量の校正は D 任せにせず、util/validator.py（R-S22 の指標）で実 EID と照合する

  診断値: k の 2 次モーメント（等価 σ [px]）、σ(x) の体内平均 [HU]、|u − x| の RMS [HU]

【v4（pipeline.mode v4、2026-10-31）】 同じ部品を順序を変えて使う（models/projection_model.py）: v = G(x_c) ⊛ k(z)、u = v + σ(x_c) ⊙ (h(z) ⊛ ε)。
  blur(x, z) と noise(x_ref, z, eps) に分けて呼ぶ。σ(x) = σ_min + (σ_max − σ_min)·sigmoid(·)（sigma_min_norm。v3 は 0 で従来どおり σ_max·sigmoid）。
  forward（v3 の u = x ⊛ k + σ(x) ⊙ (h ⊛ ε)）は変えない
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class KernelNet(nn.Module):
    """z_k → (k: N×K×K の非負・和 1 カーネル, h: N×Kh×Kh の L2 ノルム 1 フィルタ)。"""

    def __init__(self, z_dim, ksize, hsize, hidden=128):
        super().__init__()
        self.ksize, self.hsize = ksize, hsize
        self.body = nn.Sequential(nn.Linear(z_dim, hidden), nn.LeakyReLU(0.2, True), nn.Linear(hidden, hidden), nn.LeakyReLU(0.2, True))
        self.head_k = nn.Linear(hidden, ksize * ksize)
        self.head_h = nn.Linear(hidden, hsize * hsize)
        with torch.no_grad():  # 初期は k ≈ δ（中心が大きい）、h ≈ δ（白色）になるように bias を置く
            self.head_k.bias.zero_()
            self.head_k.bias[(ksize * ksize) // 2] = 4.0
            self.head_h.weight.mul_(0.1)
            self.head_h.bias.zero_()
            self.head_h.bias[(hsize * hsize) // 2] = 1.0
        yy, xx = torch.meshgrid(torch.arange(ksize, dtype=torch.float32), torch.arange(ksize, dtype=torch.float32), indexing="ij")
        c = (ksize - 1) / 2.0
        self.register_buffer("dy", (yy - c).reshape(1, -1))
        self.register_buffer("dx", (xx - c).reshape(1, -1))

    def forward(self, z):
        f = self.body(z)
        k = torch.softmax(self.head_k(f), dim=1)  # (N, K*K)
        h = self.head_h(f)
        h = h / (h.norm(dim=1, keepdim=True) + 1e-8)  # (N, Kh*Kh)、単位ノルム → 白色 ε に掛けても分散 1
        return k, h

    def kernel_stats(self, k):
        """重心（COM）の二乗と等価 σ [px]（2 次モーメントの平方根）。k: (N, K*K)。"""
        my, mx = (k * self.dy).sum(1), (k * self.dx).sum(1)
        com2 = my ** 2 + mx ** 2
        var = (k * (self.dy ** 2 + self.dx ** 2)).sum(1) - com2
        return com2, var.clamp(min=0).sqrt() / 2.0 ** 0.5  # 2 次元の等方 σ に換算


class AmplitudeNet(nn.Module):
    """x → σ(x) ∈ [sigma_min, sigma_max]（低周波限定のノイズ振幅マップ）。sigma_min = 0（v3）なら従来の sigma_max·sigmoid と同じ値。"""

    def __init__(self, ch, sigma_max, pool=8, sigma_min=0.0):
        super().__init__()
        self.pool, self.sigma_max, self.sigma_min = pool, sigma_max, sigma_min
        self.net = nn.Sequential(
            nn.Conv2d(1, ch, 3, padding=1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ch, ch, 3, padding=1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ch, 1, 3, padding=1))
        with torch.no_grad():
            self.net[-1].bias.fill_(-1.0)  # 初期振幅 ≈ sigma_max · sigmoid(−1) ≈ 0.27 sigma_max

    def forward(self, x):
        a = self.net(x)
        a = F.avg_pool2d(a, self.pool)
        a = F.interpolate(a, size=x.shape[-2:], mode="bilinear", align_corners=False)
        if self.sigma_min == 0.0:  # v3: 従来の式のまま（bit 同一）
            return torch.sigmoid(a) * self.sigma_max
        return self.sigma_min + (self.sigma_max - self.sigma_min) * torch.sigmoid(a)


def _conv_per_sample(x, k, ksize):
    """x: (N,1,H,W)、k: (N, ksize*ksize) → サンプルごとのカーネルで畳み込み（reflect padding、groups=N）。"""
    n, _, h, w = x.shape
    pad = ksize // 2
    xp = F.pad(x, (pad, pad, pad, pad), mode="reflect").view(1, n, h + 2 * pad, w + 2 * pad)
    y = F.conv2d(xp, k.view(n, 1, ksize, ksize), groups=n)
    return y.view(n, 1, h, w)


class LearnedDegrade(nn.Module):
    def __init__(self, z_dim, ksize, hsize, amp_ch, sigma_max_norm, sigma_min_norm=0.0):
        super().__init__()
        if ksize % 2 == 0 or hsize % 2 == 0:
            raise ValueError("learned_kernel_size / learned_noise_kernel_size は奇数")
        if not 0.0 <= sigma_min_norm < sigma_max_norm:
            raise ValueError(f"0 ≤ sigma_min ({sigma_min_norm}) < sigma_max ({sigma_max_norm})")
        self.z_dim, self.ksize, self.hsize = z_dim, ksize, hsize
        self.kernel = KernelNet(z_dim, ksize, hsize)
        self.amp = AmplitudeNet(amp_ch, sigma_max_norm, sigma_min=sigma_min_norm)

    def sample_z(self, n, device, generator=None):
        return torch.randn(n, self.z_dim, device=device, generator=generator)

    def forward(self, x, z=None, eps=None, generator=None):
        """x: (N,1,H,W) 正規化済み → (u, info)。z / eps を省略すれば引く（generator で再現可）。"""
        n = x.shape[0]
        if z is None:
            z = self.sample_z(n, x.device, generator)
        if eps is None:
            eps = torch.randn(x.shape, device=x.device, generator=generator)
        k, h = self.kernel(z)
        blurred = _conv_per_sample(x, k, self.ksize)
        shaped = _conv_per_sample(eps, h, self.hsize)
        sigma = self.amp(x)
        u = blurred + sigma * shaped
        com2, k_sigma = self.kernel.kernel_stats(k)
        return u, {"blurred": blurred, "sigma": sigma, "com2": com2, "k_sigma_px": k_sigma, "z": z}

    # ------------------------------------------------------------------ v4: 分離した呼び方（順序は呼び出し側）
    def blur(self, x, z):
        """x ⊛ k(z)。戻り値 (blurred, info)。info: k, h, com2, k_sigma_px, z（同じ z で noise を呼ぶ）。"""
        k, h = self.kernel(z)
        blurred = _conv_per_sample(x, k, self.ksize)
        com2, k_sigma = self.kernel.kernel_stats(k)
        return blurred, {"k": k, "h": h, "com2": com2, "k_sigma_px": k_sigma, "z": z}

    def noise(self, x_ref, z, eps, h=None):
        """σ(x_ref) ⊙ (h(z) ⊛ ε)。戻り値 (noise, sigma)。h を渡せば kernel(z) を再計算しない。"""
        if h is None:
            _, h = self.kernel(z)
        shaped = _conv_per_sample(eps, h, self.hsize)
        sigma = self.amp(x_ref)
        return sigma * shaped, sigma
