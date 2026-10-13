"""stage3/inference_dir.py — 学習済みの条件付き DDIM で EID-like1024（または EID512 → 補間）→ PCD-like1024 をフォルダ丸ごと書き出す。
stage2/inference_dir.py から 2026-10-13 に複製し、U-Net の 1 回の forward を「1024 を image_size のタイルに分けた DDIM サンプリング」に置き換えた。
通常は run_infer.py（ホストからは bash start3.sh infer）から**全引数明示**で exec される。

サンプリング: runners/diffusion.CondDiffusion.sample_tiled（WeatherDiffusion 型。毎ステップ重なるタイルの ε を画素ごとに平均して全体で 1 回 DDIM 更新）。
  重みは run の model.pth（--use_ema なら ema.pth を写す。本家 sample() と同じ）、ネットと拡散の設定は run の launch（--launch）の ddim。
  x_T はスライスごとに --seed とスライスの相対パスから決める（同じ seed なら同じ結果）。所要時間は「タイル数 × sample_steps」に比例する（1 スライスずつ）。
入力（--input_dir 直下に症例フォルダ、中に <slice>.png。uint16 1ch）: 1 枚目の大きさで決める。image_size（1024）はそのまま、image_size / scale（512）は学習と同じ方式で補間。
  --cases で症例を絞り、--slices_per_case で症例ごとに等間隔に間引き、--max_slices で上限（この順）。
出力（out_dir = <repo>/output/<run>_<重み>_<入力名>_<時刻>/。入力からの相対パスを保つ）: stage2 と同じ
  full/<case>/<slice>.png（PCD-like1024 16bit）、full_input1024/（512 入力 かつ --save_input1024）、full_panel/（--save_panel、5 列）、eid/ または ref/（パネルの相方の 16bit）、
  metrics.txt（--teacher。rmse / ssim / psnr と入力そのままの参照値 *_input。出力は保存する画像と同じく [−1, 1] に収めてから測る）、infer.yaml（run_infer.py が書く）、slices.txt（推論したスライスと秒数）
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from configs.schema import DDIM, validate  # noqa: E402
from data.ct_io import INTERP_FLAGS, CaseIndex, denormalize, normalize, read_stored, upsample, write_png  # noqa: E402
from runners.diffusion import CondDiffusion, count_tiles, describe_seq, slice_seed  # noqa: E402
from util.metrics import mse, psnr, ssim  # noqa: E402
from util.panel import build_panel, full_labels  # noqa: E402

DEVICES = ("cuda", "cpu")
METRIC_COLS = ("rmse", "ssim", "psnr", "rmse_input", "ssim_input", "psnr_input")


def parse_args():
    p = argparse.ArgumentParser(description="Stage 3（条件付き DDIM）推論。引数はすべて必須")
    req = lambda *a, **k: p.add_argument(*a, required=True, **k)  # noqa: E731
    req("--weight_dir"); req("--input_dir"); req("--out_dir")
    req("--device", choices=DEVICES)
    req("--launch", help="run の launch（ddim の実効値を読む）")
    req("--index_cache_dir"); req("--pcd1024_dir"); req("--eid_dir"); req("--eidlike1024_dir")
    req("--hu_offset", type=int); req("--hu_min", type=int); req("--hu_max", type=int)
    req("--scale", type=int); req("--interp", choices=tuple(INTERP_FLAGS)); req("--image_size", type=int)
    req("--display_hu_min", type=int); req("--display_hu_max", type=int); req("--preview_bits", type=int, choices=(8, 16))
    # infer.yaml
    req("--input", choices=("eidlike", "eid"))
    req("--cases", help="カンマ区切りの症例名（空文字で全部）")
    req("--max_slices", type=int); req("--slices_per_case", type=int)
    p.add_argument("--teacher", action="store_true"); p.add_argument("--save_input1024", action="store_true"); p.add_argument("--save_panel", action="store_true")
    req("--eid_slice"); req("--pcd_slice")
    p.add_argument("--use_ema", action="store_true")
    req("--sample_steps", type=int); req("--eta", type=float); req("--skip_type", choices=("uniform", "quad")); req("--seed", type=int)
    req("--tile_stride", type=int); req("--tile_batch", type=int)
    return p.parse_args()


def resolve_device(name):
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda が指定されたが CUDA が使えない（machines.yaml の gpu_gen を 0 にすれば cpu で動く）")
        return torch.device("cuda:0")
    return torch.device("cpu")


def write_metrics(path, rows, weight_path, input_dir, pcd1024_dir, hu, note):
    """metrics.txt: スライスごとの指標と末尾に平均（stage2 と同じ形式）。"""
    hu_per_unit = (hu[2] - hu[1]) / 2.0
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Stage 3 infer: 出力 vs 教師（PCD1024）。rmse は正規化空間 [−1,1]（HU に直すなら × {hu_per_unit:.0f}）、ssim は 1 へ、psnr は dB。*_input は入力そのまま vs 教師（参照値）\n")
        f.write(f"# weights={weight_path} input={input_dir} teacher={pcd1024_dir} {note}\n")
        f.write("# slice\t" + "\t".join(METRIC_COLS) + "\n")
        for rel, r in rows:
            f.write(rel + "\t" + "\t".join(f"{r[k]:.6f}" if k.startswith("rmse") else f"{r[k]:.4f}" for k in METRIC_COLS) + "\n")
        n = len(rows)
        mean = {k: sum(r[k] for _, r in rows) / n for k in ("mse", "ssim", "psnr", "mse_input", "ssim_input", "psnr_input")}
        agg = {"rmse": mean["mse"] ** 0.5, "ssim": mean["ssim"], "psnr": mean["psnr"], "rmse_input": mean["mse_input"] ** 0.5, "ssim_input": mean["ssim_input"], "psnr_input": mean["psnr_input"]}
        f.write(f"# MEAN (n={n})\t" + "\t".join(f"{agg[k]:.6f}" if k.startswith("rmse") else f"{agg[k]:.4f}" for k in METRIC_COLS) + "\n")
    return agg


def select_paths(idx, cases, per_case, max_slices):
    """--cases → --slices_per_case（等間隔）→ --max_slices の順に絞る。"""
    use = cases if cases else idx.cases
    missing = [c for c in use if c not in idx.slices]
    if missing:
        raise FileNotFoundError(f"--cases の症例に PNG がありません: {missing}")
    out = []
    for c in use:
        ps = idx.slices[c]
        if per_case and len(ps) > per_case:
            ps = [ps[i] for i in np.linspace(0, len(ps) - 1, per_case).round().astype(int)]
        out += ps
    return out if max_slices == 0 else out[:max_slices]


def main():
    a = parse_args()
    device = resolve_device(a.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    input_dir, out_dir = Path(a.input_dir), Path(a.out_dir)
    hu = (a.hu_offset, a.hu_min, a.hu_max)
    if a.scale < 2 or a.image_size % a.scale:
        raise ValueError(f"--image_size {a.image_size} は --scale {a.scale}（≥ 2）の倍数")
    small = a.image_size // a.scale
    with open(a.launch, encoding="utf-8") as f:
        ddim = validate("ddim(launch)", yaml.safe_load(f)["ddim"], DDIM)

    diffusion = CondDiffusion(ddim, device, train=False)
    diffusion.load_weights(a.weight_dir, use_ema=a.use_ema)
    model = diffusion.model
    model.eval()
    p = ddim["data.image_size"]
    n_tiles = count_tiles(a.image_size, a.image_size, p, a.tile_stride)
    def sample(cond, name):
        sd = slice_seed(a.seed, name)
        torch.manual_seed(sd)  # η > 0 のときの途中の雑音（torch.randn_like）もスライスごとに再現できるように。η = 0 なら x_T だけで決まる
        return diffusion.sample_tiled(cond, a.sample_steps, a.eta, a.skip_type, sd, a.tile_stride, a.tile_batch, model)

    idx = CaseIndex(input_dir, a.index_cache_dir)
    cases = [c for c in a.cases.split(",") if c] if a.cases else []
    paths = select_paths(idx, cases, a.slices_per_case, a.max_slices)
    first = read_stored(paths[0]).shape
    if first == (a.image_size, a.image_size):
        up = False
    elif first == (small, small):
        up = True
    else:
        raise ValueError(f"入力の大きさ {first} が学習画像 {a.image_size} でも補間元 {small} でもありません: {paths[0]}")
    note = (f"ema={a.use_ema} DDIM eta={a.eta} {describe_seq(diffusion.num_timesteps, a.sample_steps, a.skip_type)} tiles={n_tiles} (p {p}, r {a.tile_stride}) seed={a.seed}")
    print(f"[infer] device={device} {diffusion.n_params / 1e6:.1f}M params weights={a.weight_dir} {note}")
    print(f"[infer] input={input_dir}（{len(paths)} 枚: cases {cases or 'all'}, slices_per_case {a.slices_per_case}, max {a.max_slices}）size {first[0]}"
          + (f" → x{a.scale} {a.interp} → {a.image_size}" if up else "") + f" input={a.input} teacher={a.teacher} -> {out_dir}")

    teacher_of = {}
    if a.teacher:
        missing = []
        for pth in paths:
            rel = Path(pth).relative_to(input_dir)
            t = Path(a.pcd1024_dir) / rel
            if t.is_file():
                teacher_of[pth] = t
            else:
                missing.append(str(rel))
        if missing:
            raise FileNotFoundError(f"--teacher ですが教師 {a.pcd1024_dir} に無いスライスが {len(missing)} 枚あります: " + ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else ""))

    lin01 = lambda stored: np.repeat(np.clip((stored.astype(np.float32) - a.hu_offset - a.display_hu_min) / float(a.display_hu_max - a.display_hu_min), 0.0, 1.0)[..., None], 3, axis=2)  # noqa: E731
    quant = ((lambda x01: (np.clip(x01, 0, 1) * 65535.0).round().astype(np.uint16)) if a.preview_bits == 16 else (lambda x01: (np.clip(x01, 0, 1) * 255.0).round().astype(np.uint8)))  # noqa: E731
    bgr = lambda rgb: np.ascontiguousarray(rgb[..., ::-1])  # noqa: E731
    to_t = lambda stored: torch.from_numpy(normalize(stored, *hu))[None, None]  # noqa: E731
    out16 = lambda y: denormalize(y[0, 0].cpu().numpy(), *hu)  # noqa: E731

    def load_1024(path, what):
        s = read_stored(path)
        if s.shape == (small, small):
            return upsample(s, a.scale, a.interp), True
        if s.shape != (a.image_size, a.image_size):
            raise ValueError(f"{what} の大きさ {s.shape} が {a.image_size} でも {small} でもありません: {path}")
        return s, False

    ref_right, eid_info, ref_left, ref_stem = [], None, [], None
    weight_name = Path(a.weight_dir).name
    if a.save_panel and a.input == "eidlike" and a.eid_slice:
        e_path = Path(a.eid_dir) / a.eid_slice
        e16, e_up = load_1024(e_path, "--eid_slice")
        e_out16 = out16(sample(to_t(e16), e_path.name))
        write_png(out_dir / "eid" / f"{e_path.stem}_eid1024.png", e16)
        write_png(out_dir / "eid" / f"{e_path.stem}_pcdlike.png", e_out16)
        eid_info = (e_path.stem, a.scale if e_up else None, a.interp if e_up else None)
        ref_right = [lin01(e_out16), lin01(e16)]
    if a.save_panel and a.input == "eid" and a.pcd_slice:
        r_in_path, r_t_path = Path(a.eidlike1024_dir) / a.pcd_slice, Path(a.pcd1024_dir) / a.pcd_slice
        r16, _ = load_1024(r_in_path, "--pcd_slice")
        rt16 = read_stored(r_t_path)
        r_out16 = out16(sample(to_t(r16), r_in_path.name))
        ref_stem = r_in_path.stem
        write_png(out_dir / "ref" / f"{ref_stem}_eidlike.png", r16)
        write_png(out_dir / "ref" / f"{ref_stem}_pcdlike.png", r_out16)
        write_png(out_dir / "ref" / f"{ref_stem}_pcd1024.png", rt16)
        ref_left = [lin01(r16), lin01(r_out16), lin01(rt16)]
    panel_labels = full_labels(weight_name, None, a.teacher, eid_info, upsample=(a.scale, a.interp) if up else None) if a.input == "eidlike" else None

    metric_rows, times = [], []
    t0 = time.time()
    for pth in tqdm(paths, unit="slice", dynamic_ncols=True):
        rel = Path(pth).relative_to(input_dir)
        in16 = read_stored(pth)
        if in16.shape != first:
            raise ValueError(f"入力の大きさが揃っていません: {in16.shape} != {first}: {pth}")
        if up:
            in16 = upsample(in16, a.scale, a.interp)
        x = to_t(in16)
        ts = time.time()
        y = sample(x, rel.as_posix()).cpu().clamp(-1.0, 1.0)  # 本家どおりサンプラでは x0 を clip しない。保存（denormalize）と同じ値域にしてから指標を取る
        times.append((rel.as_posix(), time.time() - ts))
        y16 = out16(y)
        write_png(out_dir / "full" / rel, y16)
        if up and a.save_input1024:
            write_png(out_dir / "full_input1024" / rel, in16)
        t16 = None
        if pth in teacher_of:
            t16 = read_stored(teacher_of[pth])
            if t16.shape != in16.shape:
                raise ValueError(f"教師の大きさが入力（補間後）と違います: {t16.shape} != {in16.shape}: {teacher_of[pth]}")
            tt = to_t(t16)
            row = {"mse": mse(y, tt), "ssim": ssim(y, tt), "psnr": psnr(y, tt), "mse_input": mse(x, tt), "ssim_input": ssim(x, tt), "psnr_input": psnr(x, tt)}
            row["rmse"], row["rmse_input"] = row["mse"] ** 0.5, row["mse_input"] ** 0.5
            metric_rows.append((rel.as_posix(), row))
        if a.save_panel:
            if a.input == "eidlike":
                cols, labels = [lin01(in16), lin01(y16)] + ([lin01(t16)] if t16 is not None else []) + ref_right, panel_labels
            else:
                cols = ref_left + [lin01(y16), lin01(in16)]
                labels = full_labels(weight_name, ref_stem, True, (rel.stem, a.scale if up else None, a.interp if up else None))
                if not ref_left:
                    labels = labels[3:]
            write_png(out_dir / "full_panel" / rel, bgr(quant(build_panel(cols, labels))))
    elapsed = time.time() - t0
    with open(out_dir / "slices.txt", "w", encoding="utf-8") as f:
        f.write(f"# {note}\n# slice\tsec\n" + "".join(f"{r}\t{s:.1f}\n" for r, s in times))
    if metric_rows:
        agg = write_metrics(out_dir / "metrics.txt", metric_rows, a.weight_dir, input_dir, a.pcd1024_dir, hu, note)
        print(f"[infer] metrics (n={len(metric_rows)}): rmse {agg['rmse']:.6f} (input {agg['rmse_input']:.6f}) ssim {agg['ssim']:.4f} (input {agg['ssim_input']:.4f}) "
              f"psnr {agg['psnr']:.2f} dB (input {agg['psnr_input']:.2f}) -> {out_dir / 'metrics.txt'}")
    print(f"[infer] done: {len(paths)} slices, {elapsed:.1f}s ({elapsed / max(len(paths), 1):.1f} s/slice) -> {out_dir}")


if __name__ == "__main__":
    main()
