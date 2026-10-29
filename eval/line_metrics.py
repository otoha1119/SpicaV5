"""手で引いた線（eval/annotate_lines.py の lines_<tag>.json）に沿って、入力 EID・各モデル出力・参照 PCD の edge の鋭さを測る（2026-10-28 試作）。

線は血管を外側から外側まで横切っている前提で、線の両端それぞれの外側 edge（外 → 高吸収）について、手前の谷から次の山までを 1 段の edge とし、
谷と山の値のあいだで 10 % / 90 % を越える距離の差を直接測る（主指標、1024 画素。2026-10-28 に erf 当てはめから変更:
小さい石灰は上に平らな部分が無く、erf だと幅を大きく見積もるため）。参考に erf 当てはめ
hu(r) = a + b * erf((r - r0) / (sqrt(2) * s))、d = 2.563 * s も残す
あわせて線上のピーク HU と、外側基準の半値幅（FWHM。大きさ依存なので同じ撮像の中の比だけに使う）を出す。

EID の線は EID1024 と同じ画像格子の出力（PCD-like1024）にそのまま当てる。PCD は PCD 側の線で測る（2 年前の別撮像。位置は合っていない）。

    python3 eval/line_metrics.py --lines ~/Downloads/PCD-plaque-eval/lines/lines_day1.json \
        --infer_dir ~/Downloads/weight_infer_results/full_png --models stage1_sr,stage2,stage2+D,stage3 \
        --out_dir ~/Downloads/weight_infer_results/line_metrics_day1

infer_dir/<model>/<EID スライス名>.png が無い病変は EID と PCD だけ測る。画素値 stored = HU + 1400（lines の hu_offset）。
"""

import argparse
import csv
import json
import math
import os

import numpy as np
from PIL import Image

SAMPLE_STEP = 0.1          # 線上の sample 間隔（画素）
DETECT_SIGMA_PX = 1.0      # edge の窓を決めるときだけ使う平滑化（画素）
S_GRID = np.arange(0.15, 6.0, 0.025)   # erf の s の探索範囲（画素）。上限に張り付いたら fit 不良として印を付ける


def erf_vec(x):
    return np.vectorize(math.erf, otypes=[float])(x)


def bilinear(a, x, y):
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    h, w = a.shape

    def g(xx, yy):
        return a[np.clip(yy, 0, h - 1), np.clip(xx, 0, w - 1)]
    return (g(x0, y0) * (1 - fx) + g(x0 + 1, y0) * fx) * (1 - fy) + (g(x0, y0 + 1) * (1 - fx) + g(x0 + 1, y0 + 1) * fx) * fy


def profile(a_hu, line):
    L = math.hypot(line["x1"] - line["x0"], line["y1"] - line["y0"])
    n = int(L / SAMPLE_STEP) + 1
    t = np.linspace(0.0, 1.0, n)
    x = line["x0"] + t * (line["x1"] - line["x0"])
    y = line["y0"] + t * (line["y1"] - line["y0"])
    return t * L, bilinear(a_hu, x, y)


def smooth(v, sigma_samples):
    k = int(3 * sigma_samples)
    g = np.exp(-0.5 * (np.arange(-k, k + 1) / sigma_samples) ** 2)
    g /= g.sum()
    return np.convolve(np.pad(v, k, mode="edge"), g, mode="valid")


def edge_window(d, hu):
    """始点側の外側 edge の窓（sample の添字 [i0, i1)）。平滑化した profile で、
    始点の水準と全体の最大の半値を最初に越える点を見つけ、その手前の谷から次の山までを 1 段の edge とする
    （手前に小さい石灰などの段があっても、主な edge だけを当てるため。2026-10-28）"""
    hs = smooth(hu, DETECT_SIGMA_PX / SAMPLE_STEP)
    lo = float(np.mean(hs[: int(1.0 / SAMPLE_STEP)]))
    hi = float(hs.max())
    half = lo + 0.5 * (hi - lo)
    i_half = int(np.argmax(hs >= half))
    i0 = i_half
    while i0 > 0 and hs[i0 - 1] <= hs[i0]:
        i0 -= 1
    i1 = i_half
    while i1 + 1 < len(hs) and hs[i1 + 1] >= hs[i1]:
        i1 += 1
    return i0, i1 + 1


def rise_direct(d, hu, hs, i0, i1):
    """窓 [i0, i1) の中で、谷（平滑化した profile の i0）と山（i1 − 1）の値を 0 % / 100 % として、
    生の profile が 10 % と 90 % を最初に越える距離の差（線形補間）。平らな部分が無い小さい石灰でも使える"""
    lo, hi = float(hs[i0]), float(hs[i1 - 1])
    if hi - lo <= 0:
        return None

    def cross(level):
        seg = hu[i0:i1]
        k = int(np.argmax(seg >= level))
        if seg[k] < level:
            return None
        if k == 0:
            return float(d[i0])
        x0, x1, y0, y1 = d[i0 + k - 1], d[i0 + k], seg[k - 1], seg[k]
        return float(x0 + (level - y0) * (x1 - x0) / (y1 - y0))
    a, b = cross(lo + 0.1 * (hi - lo)), cross(lo + 0.9 * (hi - lo))
    return None if a is None or b is None else dict(d1090=b - a, lo=lo, hi=hi)


def fit_erf(d, hu):
    best = None
    n = len(d)
    r0_grid = np.arange(d[0], d[-1], 0.05)
    for s in S_GRID:
        E = erf_vec((d[None, :] - r0_grid[:, None]) / (math.sqrt(2) * s))   # (n_r0, n)
        Em, Hm = E.mean(1), hu.mean()
        var = ((E - Em[:, None]) ** 2).sum(1)
        b = ((E - Em[:, None]) * (hu - Hm)[None, :]).sum(1) / np.maximum(var, 1e-12)
        a = Hm - b * Em
        res = ((a[:, None] + b[:, None] * E - hu[None, :]) ** 2).sum(1)
        k = int(np.argmin(res))
        if best is None or res[k] < best["res"]:
            best = dict(res=float(res[k]), s=float(s), r0=float(r0_grid[k]), a=float(a[k]), b=float(b[k]))
    rmse = math.sqrt(best["res"] / n)
    contrast = 2 * abs(best["b"])
    return dict(d1090=2.563 * best["s"], r0=best["r0"], lo=best["a"] - best["b"], hi=best["a"] + best["b"],
                rmse=rmse, rel_rmse=rmse / max(contrast, 1e-6),
                at_bound=bool(best["s"] >= S_GRID[-1] - 1e-9 or best["s"] <= S_GRID[0] + 1e-9))


def measure(a_hu, line):
    d, hu = profile(a_hu, line)
    out = {"len": float(d[-1]), "peak": float(hu.max()), "min": float(hu.min())}
    for side, (dd, hh) in (("start", (d, hu)), ("end", (d[-1] - d[::-1], hu[::-1]))):
        i0, i1 = edge_window(dd, hh)
        hs = smooth(hh, DETECT_SIGMA_PX / SAMPLE_STEP)
        out[side] = rise_direct(dd, hh, hs, i0, i1) if i1 - i0 >= 5 else None
        out[side + "_erf"] = fit_erf(dd[i0:i1], hh[i0:i1]) if i1 - i0 >= 10 else None
    base = 0.5 * (np.mean(hu[: int(1.0 / SAMPLE_STEP)]) + np.mean(hu[-int(1.0 / SAMPLE_STEP):]))
    half = base + 0.5 * (hu.max() - base)
    above = np.nonzero(hu >= half)[0]
    out["fwhm"] = float(d[above[-1]] - d[above[0]]) if len(above) else float("nan")
    return out, (d, hu)


def load_hu(path, off):
    a = np.array(Image.open(path))
    if a.dtype != np.uint16 or a.ndim != 2:
        raise SystemExit(f"16bit 1ch ではない: {path}")
    return a.astype(np.float64) - off


def mean_d(m):
    """主指標: 両端の直接法 10–90 % 幅の平均（取れた側だけ）"""
    vals = [m[s]["d1090"] for s in ("start", "end") if m[s] is not None]
    return float(np.mean(vals)) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", required=True)
    ap.add_argument("--infer_dir", required=True, help="<model>/<EID スライス名>.png がある所")
    ap.add_argument("--models", required=True, help="カンマ区切り")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--pcd_d_ref", type=float, default=None,
                    help="PCD 側に線が無いときの PCD の 10–90 %% 幅の基準（1024 画素）。例: PCD-plaque-eval day1 の PCD 全線の中央値 2.50")
    args = ap.parse_args()

    ann = json.load(open(os.path.expanduser(args.lines)))
    off = ann["hu_offset"]
    data_dir = ann["data_dir"]
    infer_dir = os.path.expanduser(args.infer_dir)
    models = [m for m in args.models.split(",") if m]
    out_dir = os.path.expanduser(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    recs = []
    plots = {}
    for rid, r in ann["rows"].items():
        srcs = [("EID", os.path.join(data_dir, r["eid_image"]), "eid")]
        for m in models:
            p = os.path.join(infer_dir, m, r["eid_name"] + ".png")
            if os.path.isfile(p):
                srcs.append((m, p, "eid"))
        srcs.append(("PCD", os.path.join(data_dir, r["pcd_image"]), "pcd"))
        plots[rid] = {"r": r, "srcs": [], "has_infer": len(srcs) > 2}
        for name, path, side in srcs:
            a = load_hu(path, off)
            prof = []
            for k, line in enumerate(r[side + "_lines"]):
                m, pr = measure(a, line)
                prof.append(pr)
                row = {"lesion": rid, "image": name, "line": k + 1, "len_px": round(m["len"], 1),
                       "peak_HU": round(m["peak"]), "min_HU": round(m["min"]), "fwhm_px": round(m["fwhm"], 2), "d1090_mean_px": round(mean_d(m), 2)}
                for s in ("start", "end"):
                    f, e = m[s], m[s + "_erf"]
                    row[f"d1090_{s}_px"] = round(f["d1090"], 2) if f else ""
                    row[f"erf_d1090_{s}_px"] = round(e["d1090"], 2) if e else ""
                    row[f"erf_relrmse_{s}"] = round(e["rel_rmse"], 3) if e else ""
                    row[f"erf_bound_{s}"] = int(e["at_bound"]) if e else ""
                recs.append(row)
            plots[rid]["srcs"].append((name, a, side, prof))

    keys = list(recs[0].keys())
    with open(os.path.join(out_dir, "line_metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(recs)

    # 病変ごとの要約（線の平均）と判定 |d_out − d_PCD| < |d_EID − d_PCD|
    summ = {}
    for row in recs:
        summ.setdefault(row["lesion"], {}).setdefault(row["image"], []).append(row)
    lines_out = ["lesion | image | d1090 [px] (線平均) | 始点側 | 終点側 | peak HU | line min HU | FWHM 比 (/EID) | PCD に近づいたか（線平均で判定）"]
    for rid, imgs in summ.items():
        def agg(name, key):
            v = [x[key] for x in imgs[name] if isinstance(x[key], (int, float)) and not math.isnan(x[key])]
            return float(np.mean(v)) if v else float("nan")
        dE = agg("EID", "d1090_mean_px")
        dP = agg("PCD", "d1090_mean_px") if "PCD" in imgs else float("nan")
        if math.isnan(dP) and args.pcd_d_ref is not None:
            dP = args.pcd_d_ref
        fE = agg("EID", "fwhm_px")
        for name in imgs:
            dd = agg(name, "d1090_mean_px")
            verdict = ""
            if name not in ("EID", "PCD") and not any(math.isnan(x) for x in (dd, dE, dP)):
                verdict = "yes" if abs(dd - dP) < abs(dE - dP) else "no"
            ratio = agg(name, "fwhm_px") / fE if name != "PCD" else float("nan")
            lines_out.append(f"{rid} | {name} | {dd:.2f} | {agg(name, 'd1090_start_px'):.2f} | {agg(name, 'd1090_end_px'):.2f} | "
                             f"{agg(name, 'peak_HU'):.0f} | {agg(name, 'min_HU'):.0f} | {ratio:.2f} | {verdict}")
        if "PCD" not in imgs and args.pcd_d_ref is not None:
            lines_out.append(f"{rid} | PCD（基準値 --pcd_d_ref） | {args.pcd_d_ref:.2f} | | | | | |")
    txt = "\n".join(lines_out)
    open(os.path.join(out_dir, "summary.txt"), "w").write(txt + "\n")
    print(txt)

    # 図: 推論のある病変だけ。上段 = crop に線、下段 = プロファイル
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for rid, P in plots.items():
        if not P["has_infer"]:
            continue
        r = P["r"]
        n = len(P["srcs"])
        fig, axes = plt.subplots(2, n, figsize=(2.6 * n, 5.6), gridspec_kw={"height_ratios": [1, 0.9]})
        for j, (name, a, side, prof) in enumerate(P["srcs"]):
            cx, cy = r[side + "_center"]
            h = 48
            x0, y0 = int(round(cx)) - h, int(round(cy)) - h
            ax = axes[0, j]
            ax.imshow(a[y0:y0 + 2 * h, x0:x0 + 2 * h], cmap="gray", vmin=-400, vmax=1600,
                      extent=(x0 - 0.5, x0 + 2 * h - 0.5, y0 + 2 * h - 0.5, y0 - 0.5), interpolation="nearest")
            ax.set_title(name, fontsize=9)
            ax.axis("off")
            for k, line in enumerate(r[side + "_lines"]):
                col = ["r", "y"][k % 2]
                ax.annotate("", xy=(line["x1"], line["y1"]), xytext=(line["x0"], line["y0"]),
                            arrowprops=dict(arrowstyle="->", color=col, lw=1.2))
                axes[1, j].plot(prof[k][0], prof[k][1], col, lw=1)
            axes[1, j].set_ylim(-500, 3500)
            axes[1, j].tick_params(labelsize=7)
            dvals = [x["d1090_mean_px"] for x in summ[rid].get(name, [])]
            axes[1, j].set_title(("d10-90 " + ", ".join(f"{v:.2f}" for v in dvals)) if dvals else "no line", fontsize=8)
        fig.suptitle(f"lesion {rid}: {r['eid_name']} / {r['pcd_name']}  (window -400..1600 HU, x = 1024 px)", fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"lesion{rid}.png"), dpi=90)
        plt.close(fig)
    print("出力:", out_dir)


if __name__ == "__main__":
    main()
