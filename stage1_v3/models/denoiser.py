"""stage1_v3/models/denoiser.py — v4 の Dn（PCD512 の denoiser）: U-Net と 3 方式（dn.method）の学習則 / 固定 denoiser。

  n2n（Neighbor2Neighbor）: Huang, Li, Chen, Xu, Liu, "Neighbor2Neighbor: Self-Supervised Denoising from Single Noisy Images," CVPR 2021（arXiv 2101.02824）。
      本研究の変更（docs/decisions/20261031_stage1-v3-noise-path-analysis.md §6.2）: CT の再構成ノイズは隣接画素で相関するので、
      セル幅 s と、セル内で選ぶ 2 画素の最小 Chebyshev 距離 d₀ を実測のノイズ相関長（measure_dir.py の noise_corr_d0）から決める。それ以外は論文どおり。
      初回（cell 4 / min_dist 2）は実 PCD512 で不合格（残留 SD 44 HU、edge_ratio p90 0.68）: 学習は 4 倍 sub-sample 画像（ノイズが白く見える）、推論はフル解像度
      （ノイズは 1〜2 px の塊、ρ(1) = 0.56, ρ(2) = 0.13）なので、ネットは相関成分を構造として残す
  blindspot（構造化 blind-spot、本命）: Krull, Buchholz, Jug, "Noise2Void - Learning Denoising from Single Noisy Images," CVPR 2019（arXiv 1811.10980）のマスク方式
      （中心画素を隠し、その位置の損失だけ取る）を、Broaddus, Krull, Weigert, Schmidt, Myers, "Removing Structured Noise with Self-Supervised Blind-Spot Networks,"
      ISBI 2020（arXiv 2001.01689）の構造化盲点（中心の周りの相関画素もまとめて隠す）に拡張し、盲点を半径 r の円盤にしたもの。フル解像度で学習・推論するので n2n の不一致が無い。
      blindspot_mask / blindspot_fill / blindspot_loss
  bm3d（固定、対照）: Mäkinen, Azzari, Foi, "Collaborative Filtering of Correlated Noise: Exact Transform-Domain Variance for Improved Shrinkage and Patch Matching,"
      IEEE TIP 2020（PyPI bm3d）。stats で測ったノイズ PSD（util/metrics.noise_psd）を sigma_psd に渡す。BM3DDenoiser。学習ステップ内では回せない（CPU、数秒 / 512²）ので denoise_dir.py で先に掛ける

  DenoiserUNet(ch, n_levels): 素の U-Net（3×3 conv、LeakyReLU(0.2)、正規化層なし、avg-pool で 1/2・bilinear + conv で 2 倍）。出力はきれいな画像そのもの（残差ではない）。
                              入出力とも 1ch の正規化単位 [−1, 1]（train.yaml / denoise.yaml の hu_min / hu_max）
  receptive_field(net, device): 出力 1 画素が見る入力の一辺 [px]（勾配で実測。blindspot では 2r+1 より十分大きいこと）
  neighbor_subsample(y, s, d0, generator): (N,1,H,W) → g1(y), g2(y)（各 (N,1,H/s,W/s)）と index。各 s×s セルから Chebyshev 距離 ≥ d₀ の 2 画素を
                              有効な組の中から一様に選ぶ（組の一覧は (s, d₀) ごとに 1 回だけ作る）。画像ごと・セルごとに別の乱択
  apply_subsample(img, idx) : 同じ乱択を別の画像（f(y)）に適用する
  n2n_loss(net, y, s, d0, gamma, generator, net_full): §6.2 の L = L_rec + γ·L_reg
      L_rec = ‖f(g1(y)) − g2(y)‖²、Δ = g1(f(y)) − g2(f(y))（勾配停止、フル解像度で denoise してから sub-sample）、L_reg = ‖f(g1(y)) − g2(y) − Δ‖²
      compile 対策（9bd30de と同じ CUDA Graph の出力上書き）: 勾配側の forward の出力は直後に diff = f(g1(y)) − g2(y) と L_rec に消費してから、
      勾配なしのフル解像度 forward は compile していない net_full（同じ重み）で行う。compile した module を 1 step に 2 回呼ばない
  blindspot_mask(shape, r, stride, generator): 画像ごとに一様乱数の全体オフセットで格子を置き（環 r < d ≤ 2r+1 が画像内に収まる中心だけ）、中心 C と円盤 M（d ≤ r）の bool マスクを返す。
                              stride ≥ 2r + 2 なので円盤は重ならない。GPU でベクトル化（中心ごとの Python ループなし）
  blindspot_fill(y, M, C, r, stride, generator): ỹ = y の M の画素を、同じ中心の環から画素ごとに乱択した y の値で置き換える（N2V の random-neighbour 方式）。
                              環のオフセットは (r, stride) ごとに 1 回だけ作り、隣の円盤（格子の 4 近傍）に入りうるオフセットは除く → 置換値が隠した画素由来になることは無い
                              （verify=True で M の画素が置換元に無いことと、M の画素がすべて置き換わったことを確かめる。学習の最初の step で 1 回呼ぶ）
  blindspot_loss(net, y, r, stride, loss, generator): f(ỹ) と y の差を中心 C だけで平均（l2 | l1）。推論は f(y)（マスクなし、N2V と同じ）
  n2n_loss(net, y, s, d0, gamma, generator, net_full): §6.2 の L = L_rec + γ·L_reg
      L_rec = ‖f(g1(y)) − g2(y)‖²、Δ = g1(f(y)) − g2(f(y))（勾配停止、フル解像度で denoise してから sub-sample）、L_reg = ‖f(g1(y)) − g2(y) − Δ‖²
      compile 対策（9bd30de と同じ CUDA Graph の出力上書き）: 勾配側の forward の出力は直後に diff = f(g1(y)) − g2(y) と L_rec に消費してから、
      勾配なしのフル解像度 forward は compile していない net_full（同じ重み）で行う。compile した module を 1 step に 2 回呼ばない
  BM3DDenoiser(psd_path, profile, hu_min, hu_max, sigma_scale): HU の 2 次元配列 → bm3d.bm3d(hu, psd, profile) → [hu_min, hu_max] に clip。PSD は画像と同じ H×W
      （bm3d の規約: sigma_psd[f] = E|FFT2(noise)|²。白色 σ なら一定値 σ²·H·W = bm3d が float σ を内部で展開する値と同じ。DC が [0,0]、fftshift しない）
  net_Dn.pth の形式: {"model": {"ch", "n_levels"}, "state_dict"}（save_denoiser / load_denoiser。読む側が denoise.yaml を持たなくても組める）
"""

import functools
import hashlib
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# U-Net（正規化層なし）
# ---------------------------------------------------------------------------
def _block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.LeakyReLU(0.2, True),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.LeakyReLU(0.2, True))


class DenoiserUNet(nn.Module):
    def __init__(self, ch, n_levels):
        super().__init__()
        if ch < 1 or n_levels < 1:
            raise ValueError("ch ≥ 1, n_levels ≥ 1")
        self.ch, self.n_levels = int(ch), int(n_levels)
        widths = [ch * (2 ** i) for i in range(n_levels + 1)]
        self.inc = _block(1, widths[0])
        self.down = nn.ModuleList([_block(widths[i], widths[i + 1]) for i in range(n_levels)])
        self.up_conv = nn.ModuleList([nn.Conv2d(widths[i + 1], widths[i], 3, padding=1) for i in reversed(range(n_levels))])
        self.up = nn.ModuleList([_block(widths[i] * 2, widths[i]) for i in reversed(range(n_levels))])
        self.outc = nn.Conv2d(widths[0], 1, 3, padding=1)

    def forward(self, x):
        skips = [self.inc(x)]
        h = skips[0]
        for blk in self.down:
            h = blk(F.avg_pool2d(h, 2))
            skips.append(h)
        h = skips.pop()
        for conv, blk in zip(self.up_conv, self.up):
            h = conv(F.interpolate(h, scale_factor=2.0, mode="bilinear", align_corners=False))
            h = blk(torch.cat([h, skips.pop()], dim=1))
        return self.outc(h)


def receptive_field(net, device, size=256):
    """出力の中央 1 画素の勾配が届く入力の範囲（一辺 [px]）。重みは今のまま、FP32、勾配は bias 経由で LeakyReLU(0.2) の負側も通る。"""
    was = net.training
    net.eval()
    x = torch.zeros(1, 1, size, size, device=device, requires_grad=True)
    with torch.autocast(device_type=device.type, enabled=False):
        y = net(x.float())
    y[0, 0, size // 2, size // 2].backward()
    g = (x.grad[0, 0].abs() > 0)
    net.train(was)
    rows, cols = torch.where(g.any(dim=1))[0], torch.where(g.any(dim=0))[0]
    if rows.numel() == 0:
        return 0
    return int(max(rows.max() - rows.min(), cols.max() - cols.min())) + 1


def save_denoiser(net, path):
    torch.save({"model": {"ch": net.ch, "n_levels": net.n_levels}, "state_dict": net.state_dict()}, path)


def load_denoiser(path, device):
    """net_Dn.pth → DenoiserUNet（eval、requires_grad False）。形式が違えばエラー。"""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if not (isinstance(ck, dict) and isinstance(ck.get("model"), dict) and "state_dict" in ck):
        raise ValueError(f"net_Dn.pth の形式が不正です（models/denoiser.py の save_denoiser で書いたものを指定）: {path}")
    net = DenoiserUNet(int(ck["model"]["ch"]), int(ck["model"]["n_levels"]))
    net.load_state_dict(ck["state_dict"])
    net.to(device).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


# ---------------------------------------------------------------------------
# Neighbor2Neighbor の sub-sample
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def valid_pairs(s, d0):
    """s×s セル内の 2 画素 (i1,j1),(i2,j2) で max(|i1−i2|,|j1−j2|) ≥ d0 の組（順序あり）を (P,4) の LongTensor で。"""
    if s < 2 or d0 < 1 or d0 >= s:
        raise ValueError(f"cell_size {s} ≥ 2, 1 ≤ min_dist {d0} < cell_size")
    pairs = [(i1, j1, i2, j2) for i1 in range(s) for j1 in range(s) for i2 in range(s) for j2 in range(s)
             if (i1, j1) != (i2, j2) and max(abs(i1 - i2), abs(j1 - j2)) >= d0]
    if not pairs:
        raise ValueError(f"cell_size {s}, min_dist {d0} では有効な画素の組がありません")
    return torch.tensor(pairs, dtype=torch.long)


def neighbor_subsample(y, s, d0, generator=None):
    """y: (N,1,H,W)（H, W は s の倍数）→ (g1, g2, idx)。idx = (flat1, flat2) は (N, Hc*Wc) の gather index（apply_subsample 用）。"""
    n, c, h, w = y.shape
    if c != 1 or h % s or w % s:
        raise ValueError(f"(N,1,H,W) で H, W は cell_size {s} の倍数: {tuple(y.shape)}")
    hc, wc = h // s, w // s
    pairs = valid_pairs(s, d0).to(y.device)
    choice = torch.randint(pairs.shape[0], (n, hc, wc), device=y.device, generator=generator)
    sel = pairs[choice]  # (N, hc, wc, 4)
    cy = (torch.arange(hc, device=y.device) * s).view(1, hc, 1)
    cx = (torch.arange(wc, device=y.device) * s).view(1, 1, wc)
    flat1 = ((cy + sel[..., 0]) * w + (cx + sel[..., 1])).view(n, -1)
    flat2 = ((cy + sel[..., 2]) * w + (cx + sel[..., 3])).view(n, -1)
    idx = (flat1, flat2, hc, wc)
    return apply_subsample(y, idx, 0), apply_subsample(y, idx, 1), idx


def apply_subsample(img, idx, which):
    """idx の which（0 = g1, 1 = g2）の乱択を img (N,1,H,W) に適用して (N,1,Hc,Wc) を返す。"""
    flat1, flat2, hc, wc = idx
    flat = flat1 if which == 0 else flat2
    n = img.shape[0]
    return torch.gather(img.reshape(n, -1), 1, flat).view(n, 1, hc, wc)


def n2n_loss(net, y, s, d0, gamma, generator=None, net_full=None):
    """§6.2 の損失。戻り値 (L, L_rec, L_reg)。L_reg は γ = 0 でも値として返す（TensorBoard 用）。
    net = 勾配側（compile 済みでよい）、net_full = 勾配なしのフル解像度 forward（compile していない同じ重み。None なら net）。"""
    g1, g2, idx = neighbor_subsample(y, s, d0, generator)
    pred = net(g1)
    diff = (pred - g2).float()  # compile 済み net の出力はここで消費し切る（後続の forward に上書きされない）
    l_rec = (diff ** 2).mean()
    with torch.no_grad():
        full = (net_full if net_full is not None else net)(y).float()
        delta = (apply_subsample(full, idx, 0) - apply_subsample(full, idx, 1)).clone()
    l_reg = F.mse_loss(diff, delta)
    return l_rec + gamma * l_reg, l_rec, l_reg


# ---------------------------------------------------------------------------
# 構造化 blind-spot（Noise2Void のマスク + Structured N2V の円盤盲点）
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def _disk_offsets(r):
    """ユークリッド距離 ≤ r のオフセット (dy, dx)。"""
    return [(dy, dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1) if dy * dy + dx * dx <= r * r]


@functools.lru_cache(maxsize=None)
def annulus_offsets(r, stride):
    """環 r < d ≤ 2r+1 のオフセット (K,2) の LongTensor（CPU）。格子の隣（(±stride,0),(0,±stride)）の円盤に入りうる位置は除く
    （stride ≥ 2r + 2 なら対角の隣は距離 ≥ 2.8r + 2.8 > (2r+1) + r で届かない）。"""
    if r < 1 or stride < 2 * r + 2:
        raise ValueError(f"radius {r} ≥ 1, stride {stride} ≥ 2·radius + 2")
    out = []
    for dy in range(-(2 * r + 1), 2 * r + 2):
        for dx in range(-(2 * r + 1), 2 * r + 2):
            d2 = dy * dy + dx * dx
            if not (r * r < d2 <= (2 * r + 1) ** 2):
                continue
            if any((dy - ny) ** 2 + (dx - nx) ** 2 <= r * r for ny, nx in ((stride, 0), (-stride, 0), (0, stride), (0, -stride))):
                continue
            out.append((dy, dx))
    if not out:
        raise ValueError(f"radius {r}, stride {stride} では環に候補がありません")
    return torch.tensor(out, dtype=torch.long)


@functools.lru_cache(maxsize=None)
def _disk_kernel(r):
    k = torch.zeros(1, 1, 2 * r + 1, 2 * r + 1)
    for dy, dx in _disk_offsets(r):
        k[0, 0, dy + r, dx + r] = 1.0
    return k


def blindspot_mask(shape, r, stride, generator=None, device=None):
    """shape = (N, 1, H, W) → (C, M, origin)。C: 中心、M: 円盤（d ≤ r）の bool マスク。origin: (N, 2) の格子原点（oy, ox）。
    格子は margin = 2r+1 から始め（環が画像内に収まる）、画像ごとに [0, stride) の一様乱数でずらす。"""
    n, c, h, w = shape
    if c != 1:
        raise ValueError(f"(N,1,H,W): {tuple(shape)}")
    margin = 2 * r + 1
    if h < 2 * margin + stride or w < 2 * margin + stride:
        raise ValueError(f"画像 {h}×{w} が小さすぎます（≥ 2·(2r+1) + stride = {2 * margin + stride}）")
    origin = torch.randint(stride, (n, 2), device=device, generator=generator) + margin  # (N, 2)
    ny, nx = (h - 2 * margin - (stride - 1) - 1) // stride + 1, (w - 2 * margin - (stride - 1) - 1) // stride + 1
    if ny < 1 or nx < 1:
        raise ValueError(f"画像 {h}×{w} に中心を置けません（radius {r}, stride {stride}）")
    iy = origin[:, 0:1] + torch.arange(ny, device=device).view(1, -1) * stride  # (N, ny)
    ix = origin[:, 1:2] + torch.arange(nx, device=device).view(1, -1) * stride  # (N, nx)
    C = torch.zeros(n, h * w, dtype=torch.bool, device=device)
    flat = (iy.view(n, ny, 1) * w + ix.view(n, 1, nx)).view(n, -1)
    C.scatter_(1, flat, True)
    C = C.view(n, 1, h, w)
    M = F.conv2d(C.float(), _disk_kernel(r).to(device), padding=r) > 0.5
    return C, M, origin


def blindspot_fill(y, M, origin, r, stride, generator=None, verify=False):
    """ỹ: M の画素を、その画素が属する円盤の中心の環から画素ごとに乱択した y の値で置き換える。中心 = 最も近い格子点（円盤は重ならないので一意）。"""
    n, _, h, w = y.shape
    dev = y.device
    offs = annulus_offsets(r, stride).to(dev)  # (K, 2)
    yy = torch.arange(h, device=dev).view(1, h, 1)
    xx = torch.arange(w, device=dev).view(1, 1, w)
    oy, ox = origin[:, 0].view(n, 1, 1), origin[:, 1].view(n, 1, 1)
    cy = oy + torch.div(yy - oy + stride // 2, stride, rounding_mode="floor") * stride  # 最も近い格子点（M の外では意味を持たない）
    cx = ox + torch.div(xx - ox + stride // 2, stride, rounding_mode="floor") * stride
    pick = torch.randint(offs.shape[0], (n, h, w), device=dev, generator=generator)
    sy = (cy + offs[pick, 0]).clamp_(0, h - 1)  # 環は margin のおかげで画像内。clamp は保険（M の外の画素にしか効かない）
    sx = (cx + offs[pick, 1]).clamp_(0, w - 1)
    src = (sy * w + sx).view(n, -1)
    filled = torch.gather(y.view(n, -1), 1, src).view(n, 1, h, w)
    y_tilde = torch.where(M, filled, y)
    if verify:
        m_flat = M.view(n, -1)
        src_in_mask = torch.gather(m_flat, 1, src).view(n, 1, h, w) & M
        if bool(src_in_mask.any()):
            raise RuntimeError("blindspot: 置換元が隠した画素に含まれています（情報漏れ）")
        self_idx = (yy * w + xx).expand(n, h, w).reshape(n, -1)  # 値の一致ではなく index で判定（FOV 外の一定値 −1400 は置換後も同じ値になるため）
        if bool((src == self_idx).view(n, 1, h, w)[M].any()):
            raise RuntimeError("blindspot: 隠した画素の置換元が自分自身です（マスクが効いていない）")
    return y_tilde


def blindspot_loss(net, y, r, stride, loss, generator=None, verify=False):
    """y: (N,1,H,W)。戻り値 (L, n_centres)。L は中心画素だけの平均（l2 | l1）。"""
    C, M, origin = blindspot_mask(y.shape, r, stride, generator, y.device)
    y_tilde = blindspot_fill(y, M, origin, r, stride, generator, verify=verify)
    pred = net(y_tilde)
    diff = (pred - y).float()[C]  # compile 済み net の出力はここで消費し切る
    if loss == "l2":
        return (diff ** 2).mean(), int(C.sum())
    if loss == "l1":
        return diff.abs().mean(), int(C.sum())
    raise ValueError(f"blindspot.loss は l2 | l1: {loss!r}")


# ---------------------------------------------------------------------------
# BM3D（固定 denoiser、学習なし）
# ---------------------------------------------------------------------------
def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class BM3DDenoiser:
    """bm3d.bm3d(hu, psd, profile) を HU のまま掛ける（BM3D の閾値は sqrt(PSD) に比例するので単位はほぼ無関係。合成データで HU のまま と /1000 に縮めて掛けた結果の差は最大 3 HU、残差 RMS 54 HU）。出力は [hu_min, hu_max] に clip。
    psd: util/metrics.noise_psd が書いた (H, W) の .npy（bm3d の規約 = E|FFT2(noise)|²、DC が [0,0]）。画像サイズと一致しなければエラー。
    sigma_scale: PSD を sigma_scale² 倍して渡す（= ノイズ振幅を sigma_scale 倍と見なす）。1.0 で実測どおり、> 1 で強めに除去（残留 SD を下げる代わりに細い構造を削るおそれ。dncheck の残像で見る）。"""

    def __init__(self, psd_path, profile, hu_min, hu_max, sigma_scale):
        try:
            import bm3d  # noqa: F401
        except ImportError as e:
            raise ImportError("bm3d が入っていません（docker/requirements-*.txt に追加済み。down → build で image を作り直す。scratch では pip install bm3d）") from e
        self.psd_path = Path(psd_path)
        if not self.psd_path.is_file():
            raise FileNotFoundError(f"bm3d.psd_path がありません: {self.psd_path}（bash start1_v3.sh <machine> stats --dirs <PCD512> が noise_psd_<名前>.npy を書く）")
        psd = np.load(self.psd_path)
        if psd.ndim != 2 or psd.shape[0] != psd.shape[1] or not np.isfinite(psd).all() or (psd < 0).any():
            raise ValueError(f"PSD は有限・非負の正方 2 次元配列: {self.psd_path} {psd.shape}")
        if not (sigma_scale > 0):
            raise ValueError(f"bm3d.sigma_scale は正: {sigma_scale}")
        self.sigma_scale = float(sigma_scale)
        self.psd = psd.astype(np.float64) * self.sigma_scale ** 2  # 振幅を sigma_scale 倍と見なす（> 1 で強めに除去。BM3D の閾値は sqrt(PSD) に比例）
        self.psd_sha256 = sha256_of(self.psd_path)
        self.profile = profile
        self.hu_min, self.hu_max = float(hu_min), float(hu_max)
        self.sigma_white_hu = float(np.sqrt(self.psd.mean() / (psd.shape[0] * psd.shape[1])))  # 同じ総パワーの白色ノイズの σ（表示用）

    def info(self):
        return {"method": "bm3d", "psd_path": str(self.psd_path), "psd_sha256": self.psd_sha256, "psd_shape": list(self.psd.shape), "profile": self.profile,
                "sigma_scale": self.sigma_scale, "sigma_white_equiv_hu": self.sigma_white_hu, "hu_min": self.hu_min, "hu_max": self.hu_max}

    def __call__(self, hu):
        import bm3d
        if hu.shape != self.psd.shape:
            raise ValueError(f"画像 {hu.shape} と PSD {self.psd.shape} のサイズが違います（stats を同じサイズの入力で取り直す）")
        z = np.clip(np.asarray(hu, dtype=np.float64), self.hu_min, self.hu_max)
        with warnings.catch_warnings(), np.errstate(divide="ignore", over="ignore", invalid="ignore"):  # bm4d の変換行列の正規化で出る 0 除算の警告（結果には影響しない列）
            warnings.simplefilter("ignore", RuntimeWarning)
            out = bm3d.bm3d(z, self.psd, self.profile)
        return np.clip(out, self.hu_min, self.hu_max).astype(np.float32)

