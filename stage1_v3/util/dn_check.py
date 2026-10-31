"""stage1_v3/util/dn_check.py — v4 の Dn 単体の合否（docs/decisions/20261031_stage1-v3-noise-path-analysis.md §6.4 の表）。
check_denoiser.py（dncheck）と train_denoiser.py（epoch ごとの check/*）が共用する。

  evaluate_denoiser(net, hus, device, hu, thr) → rows（スライスごと）と summary（中央値）
    (a) 残留ノイズ  resid_sd_HU        = Dn(x) の一様 ROI 局所 SD（util/metrics.analyze_slice と同じ ROI の取り方）
    (b) 縁の保持    edge_ratio_p90/95/99 = Dn(x) の edge_grad pXX ÷ x の同値
    (c) 残像        resid_edge_ratio   = 残差 x − Dn(x) の体内 edge_grad p99 ÷ 残差の一様 ROI SD（ROI は x で選んだもの）
                    dn_resid_rms_HU    = 体内の RMS(x − Dn(x))（参考）
  judge(summary, dn) → {指標: (値, 閾値, 合否)}。全部 PASS で合格
  denoise_slice(net, hu, device, hu_norm) → Dn(x) の HU（推論はフル解像度、sub-sample なし）
  計測はすべて FP32（autocast なし）。正規化単位 1 = 2748 HU なので bf16 の量子化段差は軟部組織で 5〜11 HU になり、6 HU の閾値に対して無視できない
"""

import numpy as np
import torch

from data.dataset import hu_to_norm
from util.metrics import analyze_slice, body_mask, residual_edge_ratio

EDGE_KEYS = ("edge_grad_p90", "edge_grad_p95", "edge_grad_p99")
SUMMARY_KEYS = ("resid_sd_HU", "edge_ratio_p90", "edge_ratio_p95", "edge_ratio_p99", "resid_edge_ratio", "dn_resid_rms_HU", "x_local_sd_HU")


@torch.no_grad()
def denoise_slice(net, hu, device, hu_norm):
    """hu: (H,W) float32 → Dn(x) の HU（clip 窓の中）。FP32（autocast 無効）で通す。"""
    offset, hu_min, hu_max = hu_norm
    del offset
    x = torch.from_numpy(hu_to_norm(hu, hu_min, hu_max))[None, None].to(device)
    with torch.autocast(device_type=device.type, enabled=False):
        y = net(x.float())
    return (y.float().cpu().numpy()[0, 0] + 1.0) / 2.0 * (hu_max - hu_min) + hu_min


def evaluate_denoiser(net, hus, device, hu_norm, thr, seed=0, return_images=False):
    """hus: HU の ndarray のリスト。戻り値 (rows, summary[, images])。rows は None（体内が足りない）を含みうる。"""
    rng = np.random.default_rng(seed)
    rows, images = [], []
    hu_min, hu_max = hu_norm[1], hu_norm[2]
    for hu in hus:
        xc = denoise_slice(net, hu, device, hu_norm)
        x = np.clip(hu, hu_min, hu_max).astype(np.float32)
        resid = x - xc
        ax, ac = analyze_slice(x, thr, rng), analyze_slice(xc, thr, rng)
        if ax is None or ac is None:
            rows.append(None)
        else:
            body = body_mask(x, thr)
            ratio, sd, p99 = residual_edge_ratio(resid, body, ax["roi_yx"])
            row = {"resid_sd_HU": ac["local_sd"], "x_local_sd_HU": ax["local_sd"], "resid_edge_ratio": ratio, "resid_roi_sd_HU": sd, "resid_edge_p99": p99,
                   "dn_resid_rms_HU": float(np.sqrt((resid[body] ** 2).mean()))}
            for k in EDGE_KEYS:
                row["edge_ratio_" + k.split("_")[-1]] = ac[k] / max(ax[k], 1e-6)
                row["x_" + k] = ax[k]
                row["dn_" + k] = ac[k]
            rows.append(row)
        if return_images:
            images.append((x, xc, resid))
    valid = [r for r in rows if r is not None]
    summary = {k: (float(np.median([r[k] for r in valid])) if valid else float("nan")) for k in SUMMARY_KEYS}
    summary["n_valid"] = len(valid)
    summary["n_slices"] = len(hus)
    return (rows, summary, images) if return_images else (rows, summary)


def judge(summary, dn):
    """§6.4 の表。{名前: (値, 閾値の説明, PASS)}。"""
    res = {}
    v = summary["resid_sd_HU"]
    res["resid_sd_HU"] = (v, f"<= {dn['check.resid_sd_max_hu']}", bool(np.isfinite(v) and v <= dn["check.resid_sd_max_hu"]))
    for q in ("p90", "p95", "p99"):
        v = summary["edge_ratio_" + q]
        res["edge_ratio_" + q] = (v, f">= {dn['check.edge_ratio_min']}", bool(np.isfinite(v) and v >= dn["check.edge_ratio_min"]))
    v = summary["resid_edge_ratio"]
    res["resid_edge_ratio"] = (v, f"<= {dn['check.resid_edge_ratio_max']}", bool(np.isfinite(v) and v <= dn["check.resid_edge_ratio_max"]))
    return res


def format_table(res):
    lines = ["指標".ljust(20) + "値".rjust(10) + "  合格条件".ljust(14) + "  判定"]
    for k, (v, cond, ok) in res.items():
        lines.append(k.ljust(20) + f"{v:10.3f}" + f"  {cond}".ljust(14) + ("  PASS" if ok else "  FAIL"))
    lines.append("結果: " + ("PASS（全項目）" if all(ok for _, _, ok in res.values()) else "FAIL"))
    return "\n".join(lines)
