"""Stage 2 U-Net の評価出力。選択した01/03を全体推論し、全体と144px局所を同じ列で保存する。"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from data.ct_io import denormalize, normalize, read_stored, upsample, write_png  # noqa: E402
from inference_dir import build_generator, resolve_device  # noqa: E402
from util.panel import build_panel  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Stage 2 固定プラーク領域の全体・局所評価")
    req = lambda *x, **k: p.add_argument(*x, required=True, **k)  # noqa: E731
    req("--weight_dir"); req("--out_dir"); req("--eidlike1024_dir"); req("--pcd1024_dir"); req("--eid_dir")
    req("--arch"); req("--base_ch", type=int); req("--n_pool", type=int); req("--final_act")
    p.add_argument("--residual", action="store_true")
    req("--hu_offset", type=int); req("--hu_min", type=int); req("--hu_max", type=int)
    req("--scale", type=int); req("--interp"); req("--image_size", type=int)
    req("--display_hu_min", type=int); req("--display_hu_max", type=int); req("--preview_bits", type=int, choices=(8, 16))
    req("--device", choices=("cpu", "cuda")); req("--crop_size", type=int); req("--panel_scale", type=int)
    req("--pcd_index", type=int); req("--eid_index", type=int); req("--diff_max_hu", type=int)
    req("--hist_hu_min", type=int); req("--hist_hu_max", type=int); req("--hist_bin_hu", type=int); req("--hist_y_max_pct", type=float)
    p.add_argument("--pcd_case", nargs=4, action="append", default=[], metavar=("TAG", "SLICE", "X", "Y"))
    p.add_argument("--eid_case", nargs=4, action="append", default=[], metavar=("TAG", "SLICE", "X", "Y"))
    return p.parse_args()


def cut(a, x, y, n, what):
    if x < 0 or y < 0 or x + n > a.shape[1] or y + n > a.shape[0]:
        raise ValueError(f"{what}: ({x}, {y}) から {n}px が画像 {a.shape[1]}x{a.shape[0]} をはみ出します")
    return a[y:y+n, x:x+n]


def heatmap_abs(a, b, max_hu):
    """|a-b|: 0 HU=青、max_hu以上=赤の anomaly-map 風 RGB float01。"""
    u8 = np.clip(np.abs(a.astype(np.int32) - b.astype(np.int32)) / float(max_hu), 0, 1)
    u8 = np.rint(u8 * 255).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_JET)[..., ::-1].astype(np.float32) / 255.0


def _point_on_polyline(points, lengths, distance):
    i = min(np.searchsorted(lengths, distance, side="right") - 1, len(points) - 2)
    i = max(i, 0)
    span = lengths[i + 1] - lengths[i]
    ratio = 0.0 if span == 0 else (distance - lengths[i]) / span
    return np.rint(points[i] + ratio * (points[i + 1] - points[i])).astype(np.int32)


def styled_line(canvas, points, color, style):
    """論文図向けの黒い実線・破線・細かい点線を、曲線に沿って一定間隔で描く。"""
    points = np.asarray(points, dtype=np.float64)
    if style == "solid":
        cv2.polylines(canvas, [np.rint(points).astype(np.int32)], False, color, 2, cv2.LINE_AA)
        return
    segment = np.linalg.norm(np.diff(points, axis=0), axis=1)
    lengths = np.concatenate([[0.0], np.cumsum(segment)])
    total = lengths[-1]
    if total <= 0:
        return
    if style == "dashed":
        for start in np.arange(0.0, total, 18.0):
            end = min(start + 11.0, total)
            cv2.line(canvas, tuple(_point_on_polyline(points, lengths, start)),
                     tuple(_point_on_polyline(points, lengths, end)), color, 2, cv2.LINE_AA)
    elif style == "dotted":
        for distance in np.arange(0.0, total, 4.0):
            cv2.circle(canvas, tuple(_point_on_polyline(points, lengths, distance)), 1, color, -1, cv2.LINE_AA)
    else:
        raise ValueError(f"未知の線種: {style}")


def hu_distribution(arrays, labels, colors, styles, offset, hu_min, hu_max, bin_hu, y_max_pct, title):
    """同じbin・軸でHU分布を重ねた、OpenCVだけで書ける8bit BGRグラフ。"""
    width, height = 1400, 900
    left, right, top, bottom = 125, 45, 85, 115
    x0, x1, y0, y1 = left, width - right, height - bottom, top
    canvas = np.full((height, width, 3), 255, np.uint8)
    font, black, grid = cv2.FONT_HERSHEY_SIMPLEX, (25, 25, 25), (220, 220, 220)
    edges = np.arange(hu_min, hu_max + bin_hu, bin_hu, dtype=np.float64)
    centers = (edges[:-1] + edges[1:]) / 2.0
    for hu in range(hu_min, hu_max + 1, 512):
        px = round(x0 + (hu - hu_min) / (hu_max - hu_min) * (x1 - x0))
        cv2.line(canvas, (px, y1), (px, y0), grid, 1, cv2.LINE_AA)
        text = str(hu); tw = cv2.getTextSize(text, font, 0.55, 1)[0][0]
        cv2.putText(canvas, text, (px - tw // 2, y0 + 28), font, 0.55, black, 1, cv2.LINE_AA)
    y_step = 2.0 if y_max_pct >= 6 else y_max_pct / 5.0
    for pct in np.arange(0.0, y_max_pct + y_step * 0.5, y_step):
        py = round(y0 - pct / y_max_pct * (y0 - y1))
        cv2.line(canvas, (x0, py), (x1, py), grid, 1, cv2.LINE_AA)
        text = f"{pct:g}"; tw = cv2.getTextSize(text, font, 0.55, 1)[0][0]
        cv2.putText(canvas, text, (x0 - tw - 14, py + 5), font, 0.55, black, 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (x0, y1), (x1, y0), black, 2)
    for stored, color, style in zip(arrays, colors, styles):
        hu = stored.astype(np.float32) - offset
        hist, _ = np.histogram(hu, bins=edges)
        pct = hist.astype(np.float64) * (100.0 / hu.size)
        px = x0 + (centers - hu_min) / (hu_max - hu_min) * (x1 - x0)
        py = y0 - np.clip(pct, 0, y_max_pct) / y_max_pct * (y0 - y1)
        points = np.rint(np.column_stack([px, py])).astype(np.int32)
        styled_line(canvas, points, color, style)
    cv2.putText(canvas, title, (x0, 45), font, 0.9, black, 2, cv2.LINE_AA)
    cv2.putText(canvas, f"Pixels per {bin_hu} HU bin (% of all pixels; outside range omitted)", (x0, 72), font, 0.55, black, 1, cv2.LINE_AA)
    xlabel = f"HU (stored PNG value - {offset})"
    tw = cv2.getTextSize(xlabel, font, 0.7, 2)[0][0]
    cv2.putText(canvas, xlabel, ((width - tw) // 2, height - 35), font, 0.7, black, 2, cv2.LINE_AA)
    lx = x1 - 360
    for i, (label, color, style) in enumerate(zip(labels, colors, styles)):
        ly = y1 + 28 + i * 32
        styled_line(canvas, [(lx + j, ly - 6) for j in range(0, 51, 5)], color, style)
        cv2.putText(canvas, label, (lx + 65, ly), font, 0.58, black, 1, cv2.LINE_AA)
    return canvas


def main():
    a = parse_args(); device = resolve_device(a.device)
    net, weight_path = build_generator(a, device); forward = (lambda x: x + net(x)) if a.residual else net
    hu = (a.hu_offset, a.hu_min, a.hu_max); root, weight = Path(a.out_dir), Path(a.weight_dir).name
    lin = lambda s: np.clip((s.astype(np.float32) - a.hu_offset - a.display_hu_min) / float(a.display_hu_max - a.display_hu_min), 0, 1)  # noqa: E731
    quant = ((lambda z: (np.clip(z, 0, 1) * 65535).round().astype(np.uint16)) if a.preview_bits == 16 else (lambda z: (np.clip(z, 0, 1) * 255).round().astype(np.uint8)))
    rgb = lambda g: np.repeat(g[..., None], 3, 2)  # noqa: E731

    for kind, rows in (("pcd", a.pcd_case), ("eid", a.eid_case)):
        for tag, rel, xs, ys in rows:
            x, y = int(xs), int(ys)
            if kind == "pcd":
                inp = read_stored(Path(a.eidlike1024_dir) / rel); teacher = read_stored(Path(a.pcd1024_dir) / rel)
                if inp.shape != (a.image_size, a.image_size) or teacher.shape != inp.shape: raise ValueError(f"PCD case の画像サイズが不正: {rel}")
                diff_label = f"|PCD-like - PCD|  0..{a.diff_max_hu} HU"
            else:
                src = read_stored(Path(a.eid_dir) / rel); expect = a.image_size // a.scale
                if src.shape != (expect, expect): raise ValueError(f"実 EID は {expect}px を期待: {rel} {src.shape}")
                inp, teacher = upsample(src, a.scale, a.interp), None
            cut(inp, x, y, a.crop_size, rel)
            xn = normalize(inp, *hu)
            with torch.inference_mode(): yn = forward(torch.from_numpy(xn)[None, None].to(device))[0, 0].cpu().numpy()
            output = denormalize(yn, *hu)
            base = root / f"{tag}_{kind}_{Path(rel).stem}_x{x}_y{y}"
            for view, arrays in (("full", (inp, output, teacher)),
                                 ("local", (cut(inp, x, y, a.crop_size, rel), cut(output, x, y, a.crop_size, rel + " output"),
                                            cut(teacher, x, y, a.crop_size, rel + " teacher") if teacher is not None else None))):
                in_v, out_v, teacher_v = arrays
                if kind == "pcd":
                    gray = [lin(in_v), lin(out_v), lin(teacher_v)]
                    names = ["1_EID-like1024", "2_PCD-like1024", "3_PCD1024"]
                    labels = ["EID-like1024  (input)", f"PCD-like1024  (output)   {weight}", "PCD1024  (teacher)"]
                    hist_labels = ["EID-Like", "PCD-Like", "PCD"]
                    hist_arrays = [in_v, out_v, teacher_v]
                    hist_colors = [(20, 20, 20)] * 3
                    hist_styles = ["dotted", "dashed", "solid"]
                    diff = heatmap_abs(out_v, teacher_v, a.diff_max_hu)
                else:
                    gray = [lin(in_v), lin(out_v)]
                    names = ["1_EID1024", "2_PCD-like1024"]
                    labels = [f"EID1024  (real EID, x{a.scale} {a.interp})", f"PCD-like1024  (output)   {weight}"]
                    hist_labels = ["EID", "PCD-Like"]
                    hist_arrays = [in_v, out_v]
                    hist_colors = [(20, 20, 20)] * 2
                    hist_styles = ["dotted", "dashed"]
                d = base / view
                for name, z in zip(names, gray): write_png(d / f"{name}.png", quant(z))
                hist = hu_distribution(hist_arrays, hist_labels, hist_colors, hist_styles, a.hu_offset, a.hist_hu_min, a.hist_hu_max,
                                       a.hist_bin_hu, a.hist_y_max_pct, f"HU distribution - {tag} {view}")
                write_png(d / "hu_distribution.png", hist)
                cols, panel_labels = [rgb(z) for z in gray], labels
                if kind == "pcd":
                    diff_name = f"4_diff_abs_0_{a.diff_max_hu}HU.png"
                    write_png(d / diff_name, np.ascontiguousarray(quant(diff)[..., ::-1]))
                    cols, panel_labels = cols + [diff], labels + [diff_label]
                if view == "local" and a.panel_scale != 1:
                    cols = [cv2.resize(c, None, fx=a.panel_scale, fy=a.panel_scale, interpolation=cv2.INTER_NEAREST) for c in cols]
                panel = build_panel(cols, panel_labels)
                write_png(d / "panel.png", np.ascontiguousarray(quant(panel)[..., ::-1]))
            print(f"[crop] {tag} {kind} {rel}: full + local {a.crop_size}px -> {base}")
    print(f"[crop] done: weights={weight_path} -> {root}")


if __name__ == "__main__": main()
