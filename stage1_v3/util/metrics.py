"""stage1_v3/util/metrics.py — R-S22 の計測（学習なし）。measure_dir.py（stats）と util/validator.py（epoch ごとの自動採点）が共用する。

  analyze_slice(hu, thr, rng): 1 スライスの HU から
    (a) 一様 ROI（平滑化画像の gradient energy が最小の 32×32 窓 × N_ROI）の 2 次トレンド除去後の局所 SD [HU] と radial NPS の f_peak / f_av [cycles/px]
        と、同じ ROI のノイズ自己相関（d = 1..8 の Chebyshev 距離）:
          noise_rho[d]     = 軸 2 方向 (d,0),(0,d) と対角 2 方向 (d,d),(d,−d) の平均（報告用。noise_rho_1..4）
          noise_rho_max[d] = Chebyshev 殻 max(|dy|,|dx|) = d 上の全オフセット（混合 (d,j), (j,d), 0 < j < d を含む）の |ρ| の最大
          noise_corr_d0    = noise_rho_max[d] < 0.05 になる最小 d（保守的。denoiser の min_dist は軸方向の距離 d₀ の組も許すので、殻全体で独立を要求する）
        （v4 の Dn の Neighbor2Neighbor のセル幅 / 最小距離を決める。決定 §6.1。8 以内に無ければ NaN）。roi_yx に ROI の左上座標も返す（残差の計測で同じ ROI を使う）
    (b) 体内の強いエッジの勾配 [HU/px]（edge_grad_p90 / p95 / p99, edge_grad_top1_mean）
    (c) 体内 HU の p50 / p90 / p99、体外の平均
  body_mask(hu, thr): 体内（浸食済み）の bool マスク
  residual_edge_ratio(resid, body, roi_yx): 残差 x − Dn(x) の「残像」指標（決定 §6.4）= 体内 edge_grad p99 ÷ 一様 ROI の SD。戻り値 (ratio, resid_sd, resid_edge_p99)
    この比は純ノイズ場でも 0 にならず（Gaussian 場の |∇| p99 / SD はスペクトル形で決まり、PCD の相関ノイズで ≈ 3.5）、縁沿いの残像が体内の 1 % 未満なら p99 は動かない。参考値
  residual_edge_energy_ratio(resid, xc, body, top, band): 残像の判定（決定 §6.4 改訂）= Dn(x) の縁（体内 |∇xc| 上位 top）での RMS(resid) ÷ 縁から band px 以上離れた体内の RMS(resid)。
    Dn がノイズだけを除いていれば残差は縁と無関係で ≈ 1。Dn が縁を鈍らせる（細い血管を消す）と残差が縁に集中して > 1（合成で確認: 縁の残像 RMS 8 HU → 1.07、27 HU → 1.55）
  noise_psd(resid_rois, size): 一様 ROI のトレンド除去後ノイズ（analyze_slice の noise_rois、(n, ROI, ROI)）の平均ペリオドグラム → bm3d の規約の (size, size) PSD
    bm3d（PyPI、Mäkinen / Azzari / Foi）の sigma_psd は E|FFT2(noise)|²（fftshift なし、DC が [0,0]）。白色 σ なら一定 σ²·N·M（bm3d が float σ を内部で展開する値と同じ。
    実測で確認: float σ と σ²·N·M の PSD で出力が一致）。ROI の 32×32 ペリオドグラム P32 = mean|FFT2(roi)|² は同じ規約の 32×32 版なので、
    fftshift → (size, size) に線形補間（cycles/px の軸は不変）→ ifftshift → × size²/32² で画像サイズの PSD になる。
    DC（トレンド除去で 0）は隣接 4 bin の平均で埋める（0 のままだと BM3D が DC 成分を「ノイズ無し」と扱う。補間の種にするためだけ）
  summarize(rows, key): 中央値と四分位
"""

import cv2
import numpy as np

ROI = 32
N_ROI = 8
CORR_MAX_D = 8
CORR_THRESHOLD = 0.05


def body_mask(hu, thr):
    body = hu > thr
    return cv2.erode(body.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)


def _shell_offsets(d):
    """Chebyshev 距離 d の殻のオフセット (dy, dx)。ρ(dy,dx) = ρ(−dy,−dx) なので半分だけ: dy = 0 は dx = d、0 < dy < d は dx = ±d、dy = d は dx = −d..d。"""
    return [(0, d)] + [(dy, dx) for dy in range(1, d) for dx in (-d, d)] + [(d, dx) for dx in range(-d, d + 1)]


def _rho_offset(r, dy, dx):
    """1 ROI の正規化相関 ρ(dy, dx)（dy ≥ 0）。"""
    h, w = r.shape
    a = r[dy:, max(dx, 0) : w + min(dx, 0)]
    b = r[: h - dy, max(-dx, 0) : w + min(-dx, 0)]
    return float((a * b).mean())


def noise_autocorr(resid, max_d=CORR_MAX_D):
    """resid: (n, ROI, ROI) の平均 0 ノイズ → (rho, rho_max)。d = 1..max_d。
    rho[d−1]     = 軸 (d,0),(0,d) と対角 (d,d),(d,−d) の 4 方向の平均（ROI ごとの正規化相関を全体で平均）
    rho_max[d−1] = Chebyshev 殻 max(|dy|,|dx|) = d の全オフセット（混合 (d,j) を含む）について、ROI 平均の |ρ| の最大"""
    rho, rho_max = [], []
    for d in range(1, max_d + 1):
        per_offset = {}
        for r in resid:
            v = float((r * r).mean())
            if v <= 0:
                continue
            for off in _shell_offsets(d):
                per_offset.setdefault(off, []).append(_rho_offset(r, *off) / v)
        if not per_offset:
            rho.append(float("nan")); rho_max.append(float("nan"))
            continue
        mean = {off: float(np.mean(v)) for off, v in per_offset.items()}
        rho.append(float(np.mean([mean[(d, 0)], mean[(0, d)], mean[(d, d)], mean[(d, -d)]])))
        rho_max.append(float(max(abs(v) for v in mean.values())))
    return rho, rho_max


def corr_d0(rho_max, threshold=CORR_THRESHOLD):
    """殻全体で |ρ| < threshold になる最小 d。8 以内に無ければ NaN。"""
    for d, r in enumerate(rho_max, start=1):
        if np.isfinite(r) and r < threshold:
            return float(d)
    return float("nan")


def gradient_magnitude(hu):
    gx, gy = cv2.Sobel(hu, cv2.CV_32F, 1, 0, ksize=3) / 8.0, cv2.Sobel(hu, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    return np.sqrt(gx * gx + gy * gy)


def residual_edge_ratio(resid, body, roi_yx):
    """残差（HU）の残像指標。体内の |∇resid| p99 ÷ 一様 ROI（analyze_slice の roi_yx）での残差 SD。"""
    sd = float(np.mean([resid[y : y + ROI, x : x + ROI].std() for y, x in roi_yx]))
    p99 = float(np.percentile(gradient_magnitude(np.ascontiguousarray(resid, dtype=np.float32))[body], 99))
    return p99 / max(sd, 1e-6), sd, p99


def residual_edge_energy_ratio(resid, xc, body, top=0.05, band=2):
    """残差の縁への集中度。xc = Dn(x)（HU）の体内 |∇| 上位 top（割合）を縁とし、RMS(resid | 縁) ÷ RMS(resid | 縁から band px 以上離れた体内)。戻り値 (ratio, rms_edge, rms_far)。"""
    gm = gradient_magnitude(np.ascontiguousarray(xc, dtype=np.float32))
    thr = np.percentile(gm[body], 100.0 * (1.0 - top))
    edge = (gm >= thr) & body
    far = body & ~cv2.dilate(edge.astype(np.uint8), np.ones((2 * band + 1, 2 * band + 1), np.uint8)).astype(bool)
    if edge.sum() < 16 or far.sum() < 16:
        return float("nan"), float("nan"), float("nan")
    rms_edge, rms_far = float(np.sqrt((resid[edge] ** 2).mean())), float(np.sqrt((resid[far] ** 2).mean()))
    return rms_edge / max(rms_far, 1e-6), rms_edge, rms_far


def noise_psd(resid_rois, size):
    """resid_rois: (n, ROI, ROI) の平均 0 ノイズ → (size, size) float64 の PSD（bm3d の規約）。戻り値 (psd, sigma_white_equiv) で sigma = sqrt(mean(psd) / size²) [HU]。"""
    rois = np.asarray(resid_rois, dtype=np.float64)
    if rois.ndim != 3 or rois.shape[0] == 0:
        raise ValueError(f"(n, ROI, ROI) のノイズ ROI が要る: {rois.shape}")
    n, h, w = rois.shape
    p32 = np.zeros((h, w), np.float64)
    for r in rois:
        r = r - r.mean()
        p32 += np.abs(np.fft.fft2(r)) ** 2
    p32 /= n
    p32[0, 0] = 0.25 * (p32[0, 1] + p32[1, 0] + p32[0, -1] + p32[-1, 0])
    sh = np.fft.fftshift(p32)
    big = cv2.resize(sh.astype(np.float32), (int(size), int(size)), interpolation=cv2.INTER_LINEAR).astype(np.float64)
    psd = np.fft.ifftshift(big) * (float(size) ** 2 / float(h * w))
    psd = np.maximum(psd, 0.0)
    return psd, float(np.sqrt(psd.mean() / (float(size) ** 2)))


def radial_nps(patches):
    """patches: (n, ROI, ROI) HU（平均を引いた一様 ROI）→ 1D radial NPS と周波数軸 [cycles/px]。NPS = |FFT|² · (Δx Δy / (Nx Ny))、Δ = 1 px。"""
    n = patches.shape[0]
    ps = np.zeros((ROI, ROI), np.float64)
    for p in patches:
        p = p - p.mean()
        f = np.fft.fftshift(np.fft.fft2(p))
        ps += np.abs(f) ** 2 / (ROI * ROI)
    ps /= max(n, 1)
    yy, xx = np.indices((ROI, ROI))
    c = ROI // 2
    r = np.sqrt((yy - c) ** 2 + (xx - c) ** 2)
    bins = np.arange(0, c + 1)
    radial = np.array([ps[(r >= b) & (r < b + 1)].mean() if ((r >= b) & (r < b + 1)).any() else 0.0 for b in bins])
    freq = bins / float(ROI)
    return radial, freq


def analyze_slice(hu, thr, rng):
    body = body_mask(hu, thr)
    out = {}
    if body.sum() < ROI * ROI * 4:
        return None
    # (c) HU
    b = hu[body]
    out["hu_p50"], out["hu_p90"], out["hu_p99"] = (float(np.percentile(b, q)) for q in (50, 90, 99))
    air = hu[~(hu > thr)]
    out["air_mean"] = float(air.mean()) if air.size else float("nan")
    # (a) 一様 ROI: 平滑化画像の gradient energy が最小の窓
    sm = cv2.GaussianBlur(hu, (0, 0), 2.0)
    gx, gy = cv2.Sobel(sm, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(sm, cv2.CV_32F, 0, 1, ksize=3)
    ge = gx * gx + gy * gy
    H, W = hu.shape
    cands = []
    for y in range(0, H - ROI + 1, ROI // 2):
        for x in range(0, W - ROI + 1, ROI // 2):
            if body[y : y + ROI, x : x + ROI].all():
                cands.append((float(ge[y : y + ROI, x : x + ROI].mean()), y, x))
    if len(cands) < N_ROI:
        return None
    cands.sort()
    out["roi_yx"] = [(int(y), int(x)) for _, y, x in cands[:N_ROI]]
    rois = np.stack([hu[y : y + ROI, x : x + ROI] for _, y, x in cands[:N_ROI]]).astype(np.float64)
    # 2 次多項式のトレンドを除去してから SD
    yy, xx = np.indices((ROI, ROI))
    A = np.stack([np.ones(ROI * ROI), yy.ravel(), xx.ravel(), yy.ravel() ** 2, xx.ravel() ** 2, (yy * xx).ravel()], axis=1)
    resid = []
    for r in rois:
        if r.std() < 1e-6:  # 完全に一様な ROI（合成データなど）。lstsq が発散するので残差 0
            resid.append(np.zeros((ROI, ROI), np.float64))
            continue
        coef, *_ = np.linalg.lstsq(A, r.ravel(), rcond=None)
        resid.append((r.ravel() - A @ coef).reshape(ROI, ROI))
    resid = np.stack(resid)
    out["noise_rois"] = resid.astype(np.float32)  # noise_psd 用（stats が PSD を書く）
    out["local_sd"] = float(np.mean([r.std() for r in resid]))
    radial, freq = radial_nps(resid)
    if radial[1:].sum() > 0:
        out["nps_f_peak"] = float(freq[1:][np.argmax(radial[1:])])
        out["nps_f_av"] = float((freq[1:] * radial[1:]).sum() / radial[1:].sum())
    else:
        out["nps_f_peak"] = out["nps_f_av"] = float("nan")
    out["nps_radial"] = radial.tolist()
    rho, rho_max = noise_autocorr(resid)
    out["noise_rho"], out["noise_rho_max"] = rho, rho_max
    for d in range(1, 5):
        out[f"noise_rho_{d}"] = rho[d - 1]
        out[f"noise_rho_max_{d}"] = rho_max[d - 1]
    out["noise_corr_d0"] = corr_d0(rho_max)
    # (b) 強いエッジの勾配
    gxr, gyr = cv2.Sobel(hu, cv2.CV_32F, 1, 0, ksize=3) / 8.0, cv2.Sobel(hu, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    gm = np.sqrt(gxr * gxr + gyr * gyr)[body]
    out["edge_grad_p90"] = float(np.percentile(gm, 90))  # 中コントラストの構造（冠動脈・心筋境界）の鮮鋭度。p99 は骨・空気の強いエッジで決まる
    out["edge_grad_p95"] = float(np.percentile(gm, 95))
    out["edge_grad_p99"] = float(np.percentile(gm, 99))
    out["edge_grad_top1_mean"] = float(gm[gm >= np.percentile(gm, 99)].mean())
    return out


def summarize(rows, key):
    v = np.array([r[key] for r in rows if r is not None and np.isfinite(r[key])], dtype=np.float64)
    if v.size == 0:
        return {"median": float("nan"), "q25": float("nan"), "q75": float("nan"), "n": 0}
    return {"median": float(np.median(v)), "q25": float(np.percentile(v, 25)), "q75": float(np.percentile(v, 75)), "n": int(v.size)}


