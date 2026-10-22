"""Stage 3 DDIM の固定プラーク領域。256px文脈でサンプルし、指定144pxだけを保存する。"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from configs.schema import DDIM, validate  # noqa: E402
from data.ct_io import denormalize, normalize, read_stored, upsample, write_png  # noqa: E402
from runners.diffusion import CondDiffusion, slice_seed  # noqa: E402
from util.panel import build_panel  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Stage 3 固定プラーク領域の単独サンプリング")
    req = lambda *x, **k: p.add_argument(*x, required=True, **k)  # noqa: E731
    req("--weight_dir"); req("--launch"); req("--out_dir"); req("--eidlike1024_dir"); req("--pcd1024_dir"); req("--eid_dir")
    req("--hu_offset", type=int); req("--hu_min", type=int); req("--hu_max", type=int)
    req("--scale", type=int); req("--interp"); req("--image_size", type=int); req("--patch_size", type=int); req("--crop_size", type=int)
    req("--display_hu_min", type=int); req("--display_hu_max", type=int); req("--preview_bits", type=int, choices=(8, 16))
    req("--device", choices=("cpu", "cuda")); req("--panel_scale", type=int)
    p.add_argument("--use_ema", action="store_true")
    req("--sample_steps", type=int); req("--eta", type=float); req("--skip_type", choices=("uniform", "quad")); req("--seed", type=int)
    p.add_argument("--pcd_case", nargs=3, action="append", default=[], metavar=("SLICE", "X", "Y"))
    p.add_argument("--eid_case", nargs=3, action="append", default=[], metavar=("SLICE", "X", "Y"))
    return p.parse_args()


def resolve_device(name):
    if name == "cuda":
        if not torch.cuda.is_available(): raise RuntimeError("--device cuda ですが CUDA が使えません")
        return torch.device("cuda:0")
    return torch.device("cpu")


def cut(a, x, y, n, what):
    if x < 0 or y < 0 or x + n > a.shape[1] or y + n > a.shape[0]:
        raise ValueError(f"{what}: ({x}, {y}) から {n}px が画像 {a.shape[1]}x{a.shape[0]} をはみ出します")
    return a[y:y+n, x:x+n]


def context(a, x, y, crop_size, patch_size, what):
    margin = (patch_size - crop_size) // 2
    cx = min(max(x - margin, 0), a.shape[1] - patch_size)
    cy = min(max(y - margin, 0), a.shape[0] - patch_size)
    return cut(a, cx, cy, patch_size, what + " context"), x - cx, y - cy


def main():
    a = parse_args(); device = resolve_device(a.device)
    with open(a.launch, encoding="utf-8") as f: ddim = validate("ddim(launch)", yaml.safe_load(f)["ddim"], DDIM)
    diffusion = CondDiffusion(ddim, device, train=False); diffusion.load_weights(a.weight_dir, use_ema=a.use_ema)
    model = diffusion.model.eval(); hu = (a.hu_offset, a.hu_min, a.hu_max)
    out, weight = Path(a.out_dir), Path(a.weight_dir).name
    lin = lambda s: np.clip((s.astype(np.float32) - a.hu_offset - a.display_hu_min) / float(a.display_hu_max - a.display_hu_min), 0, 1)  # noqa: E731
    quant = ((lambda z: (np.clip(z, 0, 1) * 65535).round().astype(np.uint16)) if a.preview_bits == 16 else (lambda z: (np.clip(z, 0, 1) * 255).round().astype(np.uint8)))
    rgb = lambda g: np.repeat(g[..., None], 3, 2)  # noqa: E731
    panel_up = lambda im: cv2.resize(im, None, fx=a.panel_scale, fy=a.panel_scale, interpolation=cv2.INTER_NEAREST)  # noqa: E731

    plan, conds, seeds = [], [], []
    for kind, rows in (("pcd", a.pcd_case), ("eid", a.eid_case)):
        for rel, xs, ys in rows:
            x, y = int(xs), int(ys)
            if kind == "pcd":
                full = read_stored(Path(a.eidlike1024_dir) / rel); teacher = read_stored(Path(a.pcd1024_dir) / rel)
                if full.shape != (a.image_size, a.image_size) or teacher.shape != full.shape: raise ValueError(f"PCD case の画像サイズが不正: {rel}")
            else:
                src = read_stored(Path(a.eid_dir) / rel); expect = a.image_size // a.scale
                if src.shape != (expect, expect): raise ValueError(f"実 EID は {expect}px を期待: {rel} {src.shape}")
                full, teacher = upsample(src, a.scale, a.interp), None
            ctx, ox, oy = context(full, x, y, a.crop_size, a.patch_size, rel)
            cut(full, x, y, a.crop_size, rel)
            if teacher is not None: cut(teacher, x, y, a.crop_size, rel + " teacher")
            plan.append((kind, rel, x, y, full, teacher, ox, oy))
            conds.append(torch.from_numpy(normalize(ctx, *hu))[None, None])
            seeds.append(slice_seed(a.seed, f"{kind}:{rel}"))

    torch.manual_seed(a.seed)
    with torch.inference_mode():
        sampled = diffusion.sample_patches(torch.cat(conds), a.sample_steps, a.eta, a.skip_type, seeds, model).cpu().clamp(-1, 1)
    for i, ((kind, rel, x, y, full, teacher, ox, oy), yn) in enumerate(zip(plan, sampled), 1):
        generated = denormalize(yn[0].numpy(), *hu)
        inp_crop = cut(full, x, y, a.crop_size, rel); out_crop = cut(generated, ox, oy, a.crop_size, rel + " output")
        d = out / f"{i:02d}_{kind}_{Path(rel).stem}_x{x}_y{y}"
        if kind == "pcd":
            arrays = [inp_crop, out_crop, cut(teacher, x, y, a.crop_size, rel + " teacher")]
            names = ["1_EID-like1024", "2_PCD-like1024", "3_PCD1024"]
            labels = ["EID-like1024  (input)", f"PCD-like1024  (output)   {weight}", "PCD1024  (teacher)"]
        else:
            arrays = [inp_crop, out_crop]; names = ["1_EID1024", "2_PCD-like1024"]
            labels = [f"EID1024  (real EID, x{a.scale} {a.interp})", f"PCD-like1024  (output)   {weight}"]
        shown = [lin(z) for z in arrays]
        for name, z in zip(names, shown): write_png(d / f"{name}.png", quant(z))
        panel = build_panel([panel_up(rgb(z)) for z in shown], labels)
        write_png(d / "panel.png", np.ascontiguousarray(quant(panel)[..., ::-1]))
        print(f"[crop] {kind} {rel} ({x},{y}) crop {a.crop_size}px / context {a.patch_size}px -> {d}")
    print(f"[crop] done: {len(plan)} regions -> {out}")


if __name__ == "__main__": main()
