"""stage2/util/metrics.py — 確認用の数字（設計書 G-1〜G-4）。**どれも「結果の確認」にだけ使う**（合格線・設計の的にしない。ネットワークにも入れない）。

入力は HU の (1, 1, H, W) tensor（FP32）。mask は「体内」の印（入力の広い平均 > display.body_hu。bool の (1, 1, H, W)）。
ここの固定値（細かい成分 = σ 2 でぼかした残り、局所の窓 9×9、縁 = 上位 2 %、帯の境）は設計書 G-1 の測り方そのもので、過去の「帯域パワー比」と同じ帯。

  fine(x)          細かい成分 = x − Gaussian(x, σ 2)
  fine_sd          細かい成分の 9×9 の局所の標準偏差の、体内での中央値（HU）               … ノイズ感（平らな所が中央値を決める）
  edge_ratio       縁の画素（σ 3 でぼかした勾配が体内の上位 2 %）で、σ 1 の勾配 ÷ σ 3 の勾配の平均 … 縁の鋭さ（造影の濃さに依らない比）
  band_power       tile 四方のタイル（体内が半分以上）のパワーを 5 帯に分けた平均（1024 の周期/画素で 0.015–0.04 / 0.04–0.075 / 0.075–0.125 / 0.125–0.25 / 0.25–0.5）
  low_ratio        パワーのうち 0.04 周期/画素より低い帯の割合（乱数の差に形があるか）
  noise_explained  1 − median(局所分散(fine(X − μ))) ÷ median(局所分散(fine(X)))             … U が正解のノイズをどれだけ当てているか（0 に近い = 正直）
"""

import torch
import torch.nn.functional as F

from models.lock import wide_mean

FINE_PX = 2.0
LOCAL_WIN = 9
EDGE_TOP = 0.02
BAND_EDGES = (0.015, 0.04, 0.075, 0.125, 0.25, 0.5)
LOW_CUT = 0.04
N_BANDS = len(BAND_EDGES) - 1


def body_mask(x_hu, sigma_px, body_hu):
    return wide_mean(x_hu, sigma_px) > body_hu


def fine(x):
    return x - wide_mean(x, FINE_PX)


def local_var(x, win=LOCAL_WIN):
    p = win // 2
    xp = F.pad(x, (p, p, p, p), mode="reflect")
    m = F.avg_pool2d(xp, win, stride=1)
    return (F.avg_pool2d(xp * xp, win, stride=1) - m * m).clamp_min(0.0)


def _median(v):
    return float(v.median()) if v.numel() else float("nan")


def fine_sd(x, mask):
    return _median(local_var(fine(x)).sqrt()[mask])


def grad_mag(x):
    xp = F.pad(x, (1, 1, 1, 1), mode="replicate")
    gx = (xp[..., 1:-1, 2:] - xp[..., 1:-1, :-2]) * 0.5
    gy = (xp[..., 2:, 1:-1] - xp[..., :-2, 1:-1]) * 0.5
    return (gx * gx + gy * gy).sqrt()


def edge_ratio(x, mask):
    g3 = grad_mag(wide_mean(x, 3.0))
    v = g3[mask]
    if v.numel() < 100:
        return float("nan")
    k = max(1, int(round(v.numel() * EDGE_TOP)))
    thr = torch.topk(v.flatten(), k).values[-1]
    sel = mask & (g3 >= thr)
    return float((grad_mag(wide_mean(x, 1.0))[sel] / g3[sel].clamp_min(1e-6)).mean())


def _tiles(x, mask, tile):
    """(1, 1, H, W) → 体内が半分以上のタイル (n, tile, tile)。"""
    h, w = x.shape[-2:]
    if h < tile or w < tile:
        return x.new_zeros((0, tile, tile))
    t = x[0, 0, :h // tile * tile, :w // tile * tile].unfold(0, tile, tile).unfold(1, tile, tile).reshape(-1, tile, tile)
    m = mask[0, 0, :h // tile * tile, :w // tile * tile].float().unfold(0, tile, tile).unfold(1, tile, tile).reshape(-1, tile, tile)
    return t[m.mean(dim=(1, 2)) >= 0.5]


def _power(tiles):
    """タイルごとに平均を引き、Hann 窓を掛けてパワー。戻り値 (パワー (n, T, T), 周波数の大きさ (T, T) 周期/画素)。"""
    n, t, _ = tiles.shape
    win = torch.hann_window(t, periodic=False, device=tiles.device)
    d = (tiles - tiles.mean(dim=(1, 2), keepdim=True)) * (win[:, None] * win[None, :])
    p = torch.fft.fft2(d).abs().square()
    f = torch.fft.fftfreq(t, device=tiles.device)
    return p, (f[:, None] ** 2 + f[None, :] ** 2).sqrt()


def band_power(x, mask, tile):
    """5 帯の平均パワー (5,) の tensor。使えるタイルが無ければ None。"""
    tiles = _tiles(x, mask, tile)
    if tiles.shape[0] == 0:
        return None
    p, fr = _power(tiles)
    p = p.mean(dim=0)
    out = []
    for lo, hi in zip(BAND_EDGES[:-1], BAND_EDGES[1:]):
        sel = (fr >= lo) & (fr < hi)
        out.append(p[sel].mean() if bool(sel.any()) else p.new_tensor(float("nan")))
    return torch.stack(out)


def low_ratio(x, mask, tile):
    """パワー（直流を除く）のうち LOW_CUT 周期/画素より低い帯の割合。"""
    tiles = _tiles(x, mask, tile)
    if tiles.shape[0] == 0:
        return float("nan")
    p, fr = _power(tiles)
    p = p.mean(dim=0)
    total = p[fr > 0].sum()
    return float(p[(fr > 0) & (fr < LOW_CUT)].sum() / total.clamp_min(1e-12))


def rms(x, mask):
    v = x[mask]
    return float(v.square().mean().sqrt()) if v.numel() else float("nan")


def corr(a, b, mask):
    a, b = a[mask], b[mask]
    if a.numel() < 2:
        return float("nan")
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).mean() / (a.std() * b.std()).clamp_min(1e-12))


def noise_explained(target, mu, mask):
    den = _median(local_var(fine(target))[mask])
    return 1.0 - _median(local_var(fine(target - mu))[mask]) / max(den, 1e-12)
