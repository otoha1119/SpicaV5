"""stage3/eval_nps.py — 推論の出力フォルダで NPS（noise power spectrum）を入力・出力・教師の 3 つで比べる。[SpicaV5 新規 2026-10-13、決定 7 の評価 (ii)]
ホストからは  bash start3.sh nps [--flag value]  （nps_stage3.sh の INFER_DIR = <repo>/output/<run>_<重み>_<入力名>_<時刻>/）。方式は configs/nps.yaml。

対象（同じ ROI）:
  input   推論の入力 1024: <INFER_DIR>/full_input1024/<slice>（512 入力で保存してあれば）→ なければ infer.yaml の input_dir/<slice>（1024 ならそのまま、512 なら推論と同じ方式で補間）
  output  <INFER_DIR>/full/<slice>
  teacher pcd1024_dir/<slice>（推論が teacher: true のときだけ。infer.yaml の machine.pcd1024_dir）
計算（ICRU Report 87 の 2D NPS を画素単位で）: ROI（roi_size²、HU）ごとに detrend（mean | poly2）→ 2D DFT → NPS(fx, fy) = |DFT|² / (N_x N_y)。画素間隔は 1（PNG に無い）
  → HU²·px²、周波数は cycles/pixel（1024 格子。0.5 がナイキスト）。ROI で平均 → radial 平均（n_bins 個、0〜0.5）。
  ・variance  = Σ NPS / (N_x N_y)（Parseval。detrend 後の ROI 内分散 HU²）、std = √variance
  ・f_peak    = radial NPS が最大のビンの中心
  ・f_mean    = Σ |f| NPS / Σ NPS（|f| ≤ 0.5 の 2D 平均。ピークより安定）
  決定 7 の Go 条件 (b)「NPS のピーク周波数が入力より PCD に近い」を |f_peak(output) − f_peak(teacher)| < |f_peak(input) − f_peak(teacher)| で判定（f_mean でも同じ比較を出す）。
出力: <INFER_DIR>/nps_<時刻>/ に nps.txt（表と判定）、nps_radial.csv（ビン中心と 3 つの radial NPS）、nps.yaml（使った設定と ROI）、roi_<n>_<set>.png（ROI の切り出し、確認用 16bit）
"""

import argparse
import datetime
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from configs.schema import MACHINE, NPS, ConfigError, apply_overrides, check_nps_values, validate, validate_shared  # noqa: E402
from data.ct_io import read_stored, upsample, write_png  # noqa: E402

SETS = ("input", "output", "teacher")


def load_yaml(path):
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"ファイルが見つかりません: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def detrend(roi, how):
    if how == "mean":
        return roi - roi.mean()
    n_y, n_x = roi.shape
    yy, xx = np.mgrid[0:n_y, 0:n_x].astype(np.float64)
    yy, xx = yy / n_y - 0.5, xx / n_x - 0.5
    A = np.stack([np.ones_like(xx), xx, yy, xx * xx, xx * yy, yy * yy], axis=-1).reshape(-1, 6)
    coef, *_ = np.linalg.lstsq(A, roi.reshape(-1), rcond=None)
    return roi - (A * coef).sum(axis=1).reshape(roi.shape)


def nps2d(rois_hu, how):
    """ROI（HU、float64）のリスト → 平均 2D NPS（fftshift 前、HU²·px²）。"""
    acc = None
    for r in rois_hu:
        d = detrend(r, how)
        p = np.abs(np.fft.fft2(d)) ** 2 / d.size
        acc = p if acc is None else acc + p
    return acc / len(rois_hu)


def radial(nps, n_bins):
    n_y, n_x = nps.shape
    fy, fx = np.meshgrid(np.fft.fftfreq(n_y), np.fft.fftfreq(n_x), indexing="ij")
    f = np.sqrt(fx ** 2 + fy ** 2)
    edges = np.linspace(0.0, 0.5, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    prof = np.full(n_bins, np.nan)
    for i in range(n_bins):
        m = (f >= edges[i]) & (f < edges[i + 1]) if i < n_bins - 1 else (f >= edges[i]) & (f <= edges[i + 1])
        if m.any():
            prof[i] = nps[m].mean()
    inside = f <= 0.5
    f_mean = float((f[inside] * nps[inside]).sum() / nps[inside].sum()) if nps[inside].sum() > 0 else float("nan")
    return centers, prof, f_mean


def main():
    p = argparse.ArgumentParser(description="Stage 3 の推論出力で NPS を比べる")
    p.add_argument("--machine", required=True)
    p.add_argument("--nps", required=True, help="configs/nps.yaml")
    p.add_argument("--machines", required=True)
    p.add_argument("--infer_dir", required=True, help="bash start3.sh infer の出力（infer.yaml がある所）")
    p.add_argument("overrides", nargs=argparse.REMAINDER)
    a = p.parse_args()
    overrides = a.overrides[1:] if a.overrides and a.overrides[0] == "--" else a.overrides
    toks, infer_dir = [], Path(a.infer_dir)
    i = 0
    while i < len(overrides):
        if overrides[i] == "--infer_dir":
            if i + 1 >= len(overrides):
                print("[nps] 設定エラー: --infer_dir には値が必要です", file=sys.stderr)
                sys.exit(2)
            infer_dir = Path(overrides[i + 1]); i += 2
        else:
            toks.append(overrides[i]); i += 1
    try:
        cfg = validate("nps", load_yaml(a.nps), NPS)
        machines = load_yaml(a.machines)
        if not isinstance(machines, dict) or a.machine not in machines:
            raise ConfigError(f"machines.yaml にエントリ '{a.machine}' がありません")
        machine = validate_shared(f"machines.{a.machine}", machines[a.machine], MACHINE)
        applied = apply_overrides(toks, {"nps": (cfg, NPS), "machine": (machine, MACHINE)})
        check_nps_values(cfg)
        info = load_yaml(infer_dir / "infer.yaml")
        try:
            input_dir, teacher = Path(info["input_dir"]), bool(info["infer"]["teacher"])
            pcd1024_dir = Path(info["machine"]["pcd1024_dir"])
            offset = int(info["hu"]["data.hu_offset"])
            scale, interp, size = int(info["upsample"]["scale"]), str(info["upsample"]["interp"]), int(info["upsample"]["image_size"])
        except (KeyError, TypeError, ValueError) as e:
            raise ConfigError(f"{infer_dir / 'infer.yaml'} を読めません（Stage 3 の推論出力か確認）: {e}")
        n = cfg["roi_size"]
        plan = []
        for k, r in enumerate(cfg["rois"]):
            rel = Path(r["slice"])
            out = infer_dir / "full" / rel
            if not out.is_file():
                raise ConfigError(f"ROI {k} のスライスが推論の出力にありません: {out}（推論した症例・スライスから選ぶ）")
            if r["y"] + n > size or r["x"] + n > size:
                raise ConfigError(f"ROI {k} {r} が画像 {size} をはみ出します（roi_size {n}）")
            saved_in = infer_dir / "full_input1024" / rel
            src_in = saved_in if saved_in.is_file() else input_dir / rel
            if not src_in.is_file():
                raise ConfigError(f"ROI {k} の入力がありません: {src_in}")
            src_t = pcd1024_dir / rel if teacher else None
            if src_t is not None and not src_t.is_file():
                raise ConfigError(f"ROI {k} の教師がありません: {src_t}")
            plan.append((k, r, {"input": src_in, "output": out, "teacher": src_t}))
    except ConfigError as e:
        print(f"[nps] 設定エラー: {e}", file=sys.stderr)
        sys.exit(2)

    jst = datetime.timezone(datetime.timedelta(hours=9), name="JST")
    now = datetime.datetime.now(jst)
    out_dir = infer_dir / f"nps_{now.strftime('%Y_%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True)
    rois = {s: [] for s in SETS}
    for k, r, src in plan:
        for s in SETS:
            if src[s] is None:
                continue
            img = read_stored(src[s])
            if img.shape == (size // scale, size // scale) and s == "input":
                img = upsample(img, scale, interp)
            if img.shape != (size, size):
                raise ValueError(f"{s} の大きさ {img.shape} が {size} ではありません: {src[s]}")
            crop = img[r["y"]: r["y"] + n, r["x"]: r["x"] + n]
            write_png(out_dir / f"roi_{k:02d}_{s}.png", crop)
            rois[s].append(crop.astype(np.float64) - offset)
    res = {}
    for s in SETS:
        if not rois[s]:
            continue
        nps = nps2d(rois[s], cfg["detrend"])
        centers, prof, f_mean = radial(nps, cfg["n_bins"])
        var = float(nps.sum() / nps.size)
        res[s] = {"prof": prof, "f_peak": float(centers[int(np.nanargmax(prof))]), "f_mean": f_mean, "variance": var, "std": var ** 0.5, "n": len(rois[s])}
    with open(out_dir / "nps_radial.csv", "w", encoding="utf-8") as f:
        f.write("f_cycles_per_pixel," + ",".join(res) + "\n")
        for i, c in enumerate(centers):
            f.write(f"{c:.5f}," + ",".join(f"{res[s]['prof'][i]:.6g}" for s in res) + "\n")
    lines = [f"# Stage 3 NPS: {infer_dir}  ROI {len(plan)} 個 × {n}²、detrend {cfg['detrend']}、周波数 cycles/pixel（1024 格子）、NPS は HU²·px²",
             "set\tn_roi\tstd_HU\tvariance_HU2\tf_peak\tf_mean"]
    for s, v in res.items():
        lines.append(f"{s}\t{v['n']}\t{v['std']:.2f}\t{v['variance']:.2f}\t{v['f_peak']:.4f}\t{v['f_mean']:.4f}")
    if "teacher" in res:
        for key in ("f_peak", "f_mean"):
            d_out, d_in = abs(res["output"][key] - res["teacher"][key]), abs(res["input"][key] - res["teacher"][key])
            lines.append(f"[判定 {key}] |output − teacher| = {d_out:.4f} vs |input − teacher| = {d_in:.4f} → " + ("出力の方が PCD に近い" if d_out < d_in else "出力は PCD に近づいていない"))
    else:
        lines.append("[判定] 教師なし（teacher: false の推論）なので比較しない")
    text = "\n".join(lines) + "\n"
    (out_dir / "nps.txt").write_text(text, encoding="utf-8")
    with open(out_dir / "nps.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"timestamp": now.isoformat(timespec="seconds"), "infer_dir": str(infer_dir), "nps": cfg, "applied": applied,
                        "sources": [{"roi": k, **{s: (str(v) if v else None) for s, v in src.items()}} for k, _, src in plan]}, f, allow_unicode=True, sort_keys=False)
    print(text + f"[nps] -> {out_dir}")


if __name__ == "__main__":
    main()
