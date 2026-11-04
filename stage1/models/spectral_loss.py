"""spectral_loss.py — [SpicaV5] 画素 fidelity の代わりの損失（2026-11-04、ブランチ 015-f_014-stage1_wavelet_D）。

背景（docs/plans/20261104_math-step-by-step.md）: 画素 fidelity ‖G(x) − x‖² は、出力の強さが D に決められているとき
「入力を縮めたコピー」を必ず選ばせる（損失 = 1 − 2a + P が伝達 a の減少関数）。その結果、本物の EID で失われている帯域の情報が
縮んだまま残り、Stage 2 が読み戻す（血管の縁の帯域 0.15–0.25 cycles/px で本物の EID の 13 倍の情報）。

ここでは、バッチの patch の FFT から帯域ごとに
  a(band)  = Re S_yx / S_xx                 伝達（入力が平均して何倍で出力に写ったか）
  γ²(band) = |S_yx|² / (S_xx · S_yy)        コヒーレンス（出力のうち入力で説明できる割合）
を求め、本物の EID の値（a_target = h_E·S_s/S_x、γ²_EID = h_E²S_s / (h_E²S_s + N_E)）に合わせる。
  L_T = Σ_band w · (a − a_target)²          強さを本物に合わせる
  L_⊥ = Σ_band w · max(0, γ² − γ²_EID − δ)  情報量の上限
低域の 1 枚ずつの錨は lp_fidelity（両方を Gaussian σ でぼかしてからの MSE）。本物の EID が情報をほぼ全部持つ帯域なのでコピーしても漏れにならない。
前処理（σ6 の Gaussian でトレンド除去 + Hann 窓）は、目標値を測った手順（docs/experiments/EXP-20261103-01_scripts/m_targets128.py）と同じ。
"""

import torch
import torch.nn.functional as F


def gaussian_blur(x, sigma):
    """固定の分離 Gaussian（reflect）。sigma ≤ 0 なら何もしない。"""
    if sigma <= 0:
        return x
    r = max(1, int(round(3 * sigma)))
    t = torch.arange(-r, r + 1, dtype=torch.float32, device=x.device)
    k = torch.exp(-t ** 2 / (2 * sigma ** 2)); k = (k / k.sum()).to(x.dtype)
    x = F.pad(x, (r, r, r, r), mode="reflect")
    x = F.conv2d(x, k.view(1, 1, 1, -1))
    return F.conv2d(x, k.view(1, 1, -1, 1))


def band_transfer(x, y, edges, detrend_sigma=6.0):
    """x, y: (N,1,H,W)。帯域ごとの伝達 a と コヒーレンス γ² を返す（tensor、y について微分可能）。edges は cycles/px の境界（昇順）。"""
    n, _, h, w = x.shape
    win = torch.outer(torch.hann_window(h, periodic=False, device=x.device), torch.hann_window(w, periodic=False, device=x.device))
    a = (x - gaussian_blur(x, detrend_sigma)).float() * win
    b = (y - gaussian_blur(y, detrend_sigma)).float() * win
    X, Y = torch.fft.fft2(a), torch.fft.fft2(b)
    sxx = (X.abs() ** 2).mean(0)[0]; syy = (Y.abs() ** 2).mean(0)[0]; syx = (Y * X.conj()).mean(0)[0]
    fy = torch.fft.fftfreq(h, device=x.device).abs()[:, None]; fx = torch.fft.fftfreq(w, device=x.device).abs()[None, :]
    fr = torch.sqrt(fy ** 2 + fx ** 2)
    av, gv = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (fr >= lo) & (fr < hi)
        s_xx, s_yy, s_yx = sxx[m].sum(), syy[m].sum(), syx[m].sum()  # 帯域内で足してから比を取る（推定のばらつきを抑える）
        av.append(s_yx.real / (s_xx + 1e-12))
        gv.append(s_yx.abs() ** 2 / (s_xx * s_yy + 1e-12))
    return torch.stack(av), torch.stack(gv)


def spectral_losses(x, y, edges, a_target, g2_target, weights, delta):
    """L_T, L_⊥ と、監視用の a / γ²（float のリスト）。"""
    a, g2 = band_transfer(x, y, edges)
    at = torch.tensor(a_target, device=x.device, dtype=a.dtype); gt = torch.tensor(g2_target, device=x.device, dtype=a.dtype)
    wv = torch.tensor(weights, device=x.device, dtype=a.dtype)
    l_t = (wv * (a - at) ** 2).sum()
    l_perp = (wv * torch.relu(g2 - gt - delta)).sum()
    return l_t, l_perp, [float(v) for v in a.detach()], [float(v) for v in g2.detach()]


def parse_floats(s, name):
    """'0.1,0.2' → [0.1, 0.2]。設定の文字列（schema は str）を検査して読む。"""
    try:
        v = [float(t) for t in str(s).split(",") if t.strip() != ""]
    except ValueError:
        raise ValueError(f"{name} はカンマ区切りの数値: {s!r}")
    if not v:
        raise ValueError(f"{name} が空です")
    return v
