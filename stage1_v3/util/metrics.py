"""stage1_v3/util/metrics.py — R-S22 の計測（学習なし）。measure_dir.py（stats）と util/validator.py（epoch ごとの自動採点）が共用する。

  analyze_slice(hu, thr, rng): 1 スライスの HU から
    (a) 一様 ROI（平滑化画像の gradient energy が最小の 32×32 窓 × N_ROI）の 2 次トレンド除去後の局所 SD [HU] と radial NPS の f_peak / f_av [cycles/px]
    (b) 体内の強いエッジの勾配 [HU/px]（edge_grad_p99, edge_grad_top1_mean）
    (c) 体内 HU の p50 / p90 / p99、体外の平均
  summarize(rows, key): 中央値と四分位
"""

import cv2
import numpy as np

ROI = 32
N_ROI = 8


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
    body = hu > thr
    body = cv2.erode(body.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
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
    rois = np.stack([hu[y : y + ROI, x : x + ROI] for _, y, x in cands[:N_ROI]]).astype(np.float64)
    # 2 次多項式のトレンドを除去してから SD
    yy, xx = np.indices((ROI, ROI))
    A = np.stack([np.ones(ROI * ROI), yy.ravel(), xx.ravel(), yy.ravel() ** 2, xx.ravel() ** 2, (yy * xx).ravel()], axis=1)
    resid = []
    for r in rois:
        coef, *_ = np.linalg.lstsq(A, r.ravel(), rcond=None)
        resid.append((r.ravel() - A @ coef).reshape(ROI, ROI))
    resid = np.stack(resid)
    out["local_sd"] = float(np.mean([r.std() for r in resid]))
    radial, freq = radial_nps(resid)
    if radial[1:].sum() > 0:
        out["nps_f_peak"] = float(freq[1:][np.argmax(radial[1:])])
        out["nps_f_av"] = float((freq[1:] * radial[1:]).sum() / radial[1:].sum())
    else:
        out["nps_f_peak"] = out["nps_f_av"] = float("nan")
    out["nps_radial"] = radial.tolist()
    # (b) 強いエッジの勾配
    gxr, gyr = cv2.Sobel(hu, cv2.CV_32F, 1, 0, ksize=3) / 8.0, cv2.Sobel(hu, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    gm = np.sqrt(gxr * gxr + gyr * gyr)[body]
    out["edge_grad_p99"] = float(np.percentile(gm, 99))
    out["edge_grad_top1_mean"] = float(gm[gm >= np.percentile(gm, 99)].mean())
    return out


def summarize(rows, key):
    v = np.array([r[key] for r in rows if r is not None and np.isfinite(r[key])], dtype=np.float64)
    if v.size == 0:
        return {"median": float("nan"), "q25": float("nan"), "q75": float("nan"), "n": 0}
    return {"median": float(np.median(v)), "q25": float(np.percentile(v, 25)), "q75": float(np.percentile(v, 75)), "n": int(v.size)}


