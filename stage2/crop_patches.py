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
                    diff = heatmap_abs(out_v, teacher_v, a.diff_max_hu)
                else:
                    gray = [lin(in_v), lin(out_v)]
                    names = ["1_EID1024", "2_PCD-like1024"]
                    labels = [f"EID1024  (real EID, x{a.scale} {a.interp})", f"PCD-like1024  (output)   {weight}"]
                d = base / view
                for name, z in zip(names, gray): write_png(d / f"{name}.png", quant(z))
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
