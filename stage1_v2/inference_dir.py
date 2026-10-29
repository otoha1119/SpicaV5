"""stage1_v2/inference_dir.py — 学習済み G_A で PCD512 → EID-like512 を書き出す。

通常は run_infer.py（ホストからは bash start1_v2.sh <マシン> infer）から全引数明示で exec される。
  入力: <input_dir>/<case>/<slice>.png（PCD512、1ch uint16）
  出力: <out_dir>/full/<case>/<slice>.png（EID-like512）
        <out_dir>/full_R/（R = G_A(z) − z、0 HU = 32768）, full_R_color/（8bit RGB）, R_colorbar_pm<range>HU.png   --save_residual
        <out_dir>/full_panel/（[EID-like512 | PCD512 | R + ゲージ]） --save_panel
  G_A は512フル画像を一発で通す。patch推論・DICOMは未対応。
"""

import argparse
import collections
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from data.dataset import CaseIndex, denormalize, normalize, read_stored, residual_stored, write_png  # noqa: E402
from models import networks  # noqa: E402
from util.panel import build_panel  # noqa: E402
from util.residual_color import colorbar_filename, colorbar_rgb01, residual_rgb01, residual_rgb8, rgb_to_bgr  # noqa: E402
from util.run_paths import INFER_WEIGHT  # noqa: E402

DEVICES = ("cuda", "cpu")


def parse_args():
    p = argparse.ArgumentParser(description="CycleGAN 推論: PCD512 → EID-like512")
    req = lambda *a, **k: p.add_argument(*a, required=True, **k)  # noqa: E731
    req("--weight_dir", help=f"重みディレクトリ（{INFER_WEIGHT} がある所）")
    req("--input_dir")
    req("--out_dir")
    req("--device", choices=DEVICES)
    req("--index_cache_dir")
    # G_A の構成（run の launch.yaml から）
    req("--netG")
    req("--ngf", type=int)
    req("--norm")
    req("--hu_offset", type=int)
    req("--hu_min", type=int)
    req("--hu_max", type=int)
    req("--input_size", type=int)
    # 表示（現在の configs/train.yaml の log）
    req("--diff_range_hu", type=int)
    req("--display_hu_min", type=int)
    req("--display_hu_max", type=int)
    req("--preview_bits", type=int, choices=(8, 16))
    # 方式（configs/infer.yaml）
    req("--max_slices", type=int)
    req("--batch_slices", type=int)
    p.add_argument("--save_residual", action="store_true")
    p.add_argument("--save_panel", action="store_true")
    req("--read_workers", type=int)
    req("--write_workers", type=int)
    return p.parse_args()


def resolve_device(name):
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda ですが CUDA が使えません")
    return torch.device(name)


def build_generator(a, device):
    cfg = {"network.norm": a.norm, "network.ngf": a.ngf, "network.netG": a.netG, "network.init_type": "normal", "network.init_gain": 0.02}
    net = networks.build_G(cfg, device)
    path = Path(a.weight_dir) / INFER_WEIGHT
    if not path.is_file():
        raise FileNotFoundError(f"G_A の重みがありません: {path}")
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    net.to(device).eval()
    return net, path


def main():
    a = parse_args()
    device = resolve_device(a.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    input_dir, out_dir = Path(a.input_dir), Path(a.out_dir)
    hu = (a.hu_offset, a.hu_min, a.hu_max)
    size = a.input_size
    net, weight_path = build_generator(a, device)
    idx = CaseIndex(input_dir, a.index_cache_dir)
    paths = idx.all_paths if a.max_slices == 0 else idx.all_paths[: a.max_slices]
    print(f"[infer] device={device} G_A={weight_path}")
    print(f"[infer] input={input_dir} ({idx.n_cases} cases, {idx.n_slices} slices, 推論 {len(paths)} 枚) PCD{size} -> EID-like{size}: {out_dir}")
    if (a.save_residual or a.save_panel) and a.diff_range_hu <= 0:
        raise ValueError(f"--diff_range_hu は正: {a.diff_range_hu}")
    if a.save_panel and a.display_hu_min >= a.display_hu_max:
        raise ValueError("--display_hu_min < --display_hu_max にしてください")
    panel_labels = [f"EID-like{size}  G_A(z)   {Path(a.weight_dir).name}", f"PCD{size}  z", "R = G_A(z) - z   [HU]"]
    panel_gray01 = lambda stored: np.repeat(np.clip((stored.astype(np.float32) - a.hu_offset - a.display_hu_min) / float(a.display_hu_max - a.display_hu_min), 0.0, 1.0)[..., None], 3, axis=2)  # noqa: E731
    quant_panel = (lambda x01: (np.clip(x01, 0, 1) * 65535.0).round().astype(np.uint16)) if a.preview_bits == 16 else (lambda x01: (np.clip(x01, 0, 1) * 255.0).round().astype(np.uint8))  # noqa: E731
    if a.save_residual:
        out_dir.mkdir(parents=True, exist_ok=True)
        write_png(out_dir / colorbar_filename(a.diff_range_hu), rgb_to_bgr((colorbar_rgb01(a.diff_range_hu, 512) * 255.0).round().astype(np.uint8)))

    reader = ThreadPoolExecutor(max_workers=a.read_workers) if a.read_workers > 0 else None
    writer = ThreadPoolExecutor(max_workers=a.write_workers) if a.write_workers > 0 else None
    pending = []
    tm = {"read": 0.0, "g": 0.0, "write": 0.0}

    def load(p):
        stored = read_stored(p)
        if stored.shape != (size, size):
            raise ValueError(f"入力は PCD{size} のみです: {stored.shape} {p}")
        return stored, normalize(stored, *hu)

    def drain(limit):
        t = time.time()
        while len(pending) > limit:
            pending.pop(0).result()
        tm["write"] += time.time() - t

    def put(path, arr):
        if writer is None:
            write_png(path, arr)
        else:
            pending.append(writer.submit(write_png, path, arr))
            drain(4 * a.write_workers + 16)

    queue = collections.deque()
    it = iter(paths)
    prefetch = 2 * a.batch_slices + (a.read_workers if reader is not None else 0)

    def take(n):
        items = []
        for _ in range(n):
            if reader is None:
                p = next(it, None)
                if p is None:
                    break
                t = time.time()
                stored, x_np = load(p)
                tm["read"] += time.time() - t
            else:
                while len(queue) < prefetch:
                    p = next(it, None)
                    if p is None:
                        break
                    queue.append((p, reader.submit(load, p)))
                if not queue:
                    break
                p, fut = queue.popleft()
                t = time.time()
                stored, x_np = fut.result()
                tm["read"] += time.time() - t
            items.append((p, stored, x_np))
        return items

    t0 = time.time()
    bar = tqdm(total=len(paths), unit="slice", dynamic_ncols=True)
    try:
        while True:
            items = take(a.batch_slices)
            if not items:
                break
            xs = torch.cat([torch.from_numpy(x_np).unsqueeze(0).unsqueeze(0) for _, _, x_np in items]).to(device)
            t = time.time()
            with torch.inference_mode():
                ys = net(xs).cpu()
            tm["g"] += time.time() - t
            for k, (p, stored, x_np) in enumerate(items):
                rel = Path(p).relative_to(input_dir)
                out16 = denormalize(ys[k, 0].numpy(), *hu)
                in16 = denormalize(x_np, *hu)  # 正規化窓でクリップした入力（R の基準）
                put(out_dir / "full" / rel, out16)
                if a.save_residual or a.save_panel:
                    delta_hu = out16.astype(np.int32) - in16.astype(np.int32)
                if a.save_residual:
                    put(out_dir / "full_R" / rel, residual_stored(in16, out16))
                    put(out_dir / "full_R_color" / rel, rgb_to_bgr(residual_rgb8(delta_hu, a.diff_range_hu)))
                if a.save_panel:
                    panel01 = build_panel([panel_gray01(out16), panel_gray01(in16), residual_rgb01(delta_hu, a.diff_range_hu)], panel_labels, a.diff_range_hu)
                    put(out_dir / "full_panel" / rel, rgb_to_bgr(quant_panel(panel01)))
                bar.update(1)
        drain(0)
    finally:
        bar.close()
        if reader is not None:
            reader.shutdown(wait=False, cancel_futures=True)
        if writer is not None:
            writer.shutdown(wait=True)
    elapsed = time.time() - t0
    print(f"[infer] time: 読み待ち {tm['read']:.1f}s | G_A {tm['g']:.1f}s | 書き待ち {tm['write']:.1f}s（先読み {a.read_workers} / 同時スライス {a.batch_slices} / 書き {a.write_workers}）")
    print(f"[infer] done: {len(paths)} slices, {elapsed:.1f}s ({elapsed / max(len(paths), 1):.2f} s/slice) -> {out_dir}")


if __name__ == "__main__":
    main()
