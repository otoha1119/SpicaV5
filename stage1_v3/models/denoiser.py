"""stage1_v3/models/denoiser.py — v4 の Dn（PCD512 の自己教師あり denoiser）: U-Net と Neighbor2Neighbor の sub-sample・損失。

  出典: Huang, Li, Chen, Xu, Liu, "Neighbor2Neighbor: Self-Supervised Denoising from Single Noisy Images," CVPR 2021（arXiv 2101.02824）。
  本研究の変更（docs/decisions/20261031_stage1-v3-noise-path-analysis.md §6.2）: CT の再構成ノイズは隣接画素で相関するので、
  セル幅 s と、セル内で選ぶ 2 画素の最小 Chebyshev 距離 d₀ を実測のノイズ相関長（measure_dir.py の noise_corr_d0）から決める。それ以外は論文どおり。

  DenoiserUNet(ch, n_levels): 素の U-Net（3×3 conv、LeakyReLU(0.2)、正規化層なし、avg-pool で 1/2・bilinear + conv で 2 倍）。出力はきれいな画像そのもの（残差ではない）。
                              入出力とも 1ch の正規化単位 [−1, 1]（train.yaml / denoise.yaml の hu_min / hu_max）
  neighbor_subsample(y, s, d0, generator): (N,1,H,W) → g1(y), g2(y)（各 (N,1,H/s,W/s)）と index。各 s×s セルから Chebyshev 距離 ≥ d₀ の 2 画素を
                              有効な組の中から一様に選ぶ（組の一覧は (s, d₀) ごとに 1 回だけ作る）。画像ごと・セルごとに別の乱択
  apply_subsample(img, idx) : 同じ乱択を別の画像（f(y)）に適用する
  n2n_loss(net, y, s, d0, gamma, generator, net_full): §6.2 の L = L_rec + γ·L_reg
      L_rec = ‖f(g1(y)) − g2(y)‖²、Δ = g1(f(y)) − g2(f(y))（勾配停止、フル解像度で denoise してから sub-sample）、L_reg = ‖f(g1(y)) − g2(y) − Δ‖²
      compile 対策（9bd30de と同じ CUDA Graph の出力上書き）: 勾配側の forward の出力は直後に diff = f(g1(y)) − g2(y) と L_rec に消費してから、
      勾配なしのフル解像度 forward は compile していない net_full（同じ重み）で行う。compile した module を 1 step に 2 回呼ばない
  net_Dn.pth の形式: {"model": {"ch", "n_levels"}, "state_dict"}（save_denoiser / load_denoiser。読む側が denoise.yaml を持たなくても組める）
"""

import functools

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
