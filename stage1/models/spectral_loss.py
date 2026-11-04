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
  L_N = Σ_band (log(S_yy/S_xx) − log r_target)²   出力の強さの錨（敵対的でない）。L_T・L_⊥ は入力との比しか見ないので、
        入力と無関係なゴミを足しても罰されない（PC3 run 2026_1104_1949 で epoch 1 にノイズ SD 426 HU、epoch 2 で崩壊）。r_target = S_EID / S_PCD（実測）
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


def _spectra(z, detrend_sigma, window):
    """トレンド除去した z（N,1,H,W）の FFT と周波数の半径。window: hann（Hann 窓。patch の端の重みがほぼ 0）| mirror（鏡で折り返して 2 倍に広げ、窓なし。全画素が同じ重み）。"""
    a = (z - gaussian_blur(z, detrend_sigma)).float()
    n, _, h, w = a.shape
    if window == "hann":
        a = a * torch.outer(torch.hann_window(h, periodic=False, device=z.device), torch.hann_window(w, periodic=False, device=z.device))
    elif window == "mirror":  # 端で値が途切れないので窓が要らない（DCT と同じ考え方）。patch の端も損失に入る（hann では端の数画素が無拘束になり、512 推論の外周に市松模様の枠が出た。PC3 2026_1104_1956）
        a = torch.cat([a, a.flip(-1)], -1); a = torch.cat([a, a.flip(-2)], -2); h, w = 2 * h, 2 * w
    else:
        raise ValueError(f"window は hann | mirror: {window}")
    fy = torch.fft.fftfreq(h, device=z.device).abs()[:, None]; fx = torch.fft.fftfreq(w, device=z.device).abs()[None, :]
    return torch.fft.fft2(a), torch.sqrt(fy ** 2 + fx ** 2)


def band_power(z, edges, detrend_sigma=6.0, window="mirror"):
    """帯域ごとのパワー（目標値の実測用。損失と同じ前処理）。"""
    Z, fr = _spectra(z, detrend_sigma, window)
    s = (Z.abs() ** 2).mean(0)[0]
    return torch.stack([s[(fr >= lo) & (fr < hi)].sum() for lo, hi in zip(edges[:-1], edges[1:])])


def band_transfer(x, y, edges, detrend_sigma=6.0, window="mirror"):
    """x, y: (N,1,H,W)。帯域ごとの伝達 a、コヒーレンス γ²、パワー比 S_yy/S_xx を返す（tensor、y について微分可能）。edges は cycles/px の境界（昇順）。
    半径周波数は斜め方向で最大 √2/2 ≈ 0.707 になる。上端を 0.5 にすると角の成分（市松模様など）がどの帯域にも入らず無拘束になる（PC3 で崩壊）ので、上端は 0.71 にする。"""
    X, fr = _spectra(x, detrend_sigma, window); Y, _ = _spectra(y, detrend_sigma, window)
    sxx = (X.abs() ** 2).mean(0)[0]; syy = (Y.abs() ** 2).mean(0)[0]; syx = (Y * X.conj()).mean(0)[0]
    av, gv, pv = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (fr >= lo) & (fr < hi)
        s_xx, s_yy, s_yx = sxx[m].sum(), syy[m].sum(), syx[m].sum()  # 帯域内で足してから比を取る（推定のばらつきを抑える）
        av.append(s_yx.real / (s_xx + 1e-12))
        gv.append(s_yx.abs() ** 2 / (s_xx * s_yy + 1e-12))
        pv.append(s_yy / (s_xx + 1e-12))
    return torch.stack(av), torch.stack(gv), torch.stack(pv)


def spectral_losses(x, y, edges, a_target, g2_target, weights, delta, p_target, window="mirror"):
    """L_T, L_⊥, L_N と、監視用の a / γ² / パワー比（float のリスト）。"""
    a, g2, pr = band_transfer(x, y, edges, window=window)
    at = torch.tensor(a_target, device=x.device, dtype=a.dtype); gt = torch.tensor(g2_target, device=x.device, dtype=a.dtype)
    wv = torch.tensor(weights, device=x.device, dtype=a.dtype)
    l_t = (wv * (a - at) ** 2).sum()
    l_perp = (wv * torch.relu(g2 - gt - delta)).sum()
    pt = torch.tensor(p_target, device=x.device, dtype=a.dtype)
    l_n = ((torch.log(pr + 1e-12) - torch.log(pt)) ** 2).sum()
    return l_t, l_perp, l_n, [float(v) for v in a.detach()], [float(v) for v in g2.detach()], [float(v) for v in pr.detach()]


def parse_floats(s, name):
    """'0.1,0.2' → [0.1, 0.2]。設定の文字列（schema は str）を検査して読む。"""
    try:
        v = [float(t) for t in str(s).split(",") if t.strip() != ""]
    except ValueError:
        raise ValueError(f"{name} はカンマ区切りの数値: {s!r}")
    if not v:
        raise ValueError(f"{name} が空です")
    return v
