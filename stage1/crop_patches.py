"""crop_patches.py — パッチ切り出し（bash start.sh crop → crop_stage1.sh → run_crop.py → ここ）。引数はすべて必須（run_crop.py が組む）。

指定した PCD スライスを 512 のフル推論にかけ、左上 (x, y) から patch 四方を切り出す。EID は別スライスの指定座標から同じ大きさで切る（PCD と EID はペア）。
保存は表示用のみ（16bit 生データは無し。2026-09-08 ユーザー指示）。preview_fixed_*/ と同じ流儀（表示範囲 display_hu_min..max で線形、preview_bits の深度、R は util/residual_color のカラー）。

出力（<out_dir> = <repo>/output/<run>_<重みディレクトリ名>_crop_<実行時刻>/、run_crop.py が作る）:
  case<N>_<pcd>_<eid>/
    1_EID_<eid>_x<X>_y<Y>.png       EID の代表パッチ（グレー 1ch、patch 四方、等倍）
    2_EID-like_<pcd>_x<X>_y<Y>.png  G(z) の同じ場所（グレー 1ch）
    3_PCD_<pcd>_x<X>_y<Y>.png       入力 z（グレー 1ch）
    4_R_color_<pcd>_x<X>_y<Y>.png   R = G(z) − z のカラー（RGB）
    panel.png                       [EID | EID-like | PCD | R + ゲージ] を panel_scale 倍（最近傍）に拡大して並べ、各列の下にラベル（util/panel.py。余白・ラベル・ゲージは列の高さに比例し、preview_*/ の 512 パネルと相似形）
BN は eval。学習中に走らせてよい（device cpu なら VRAM を使わない。出力は学習の書き込み先と別）。
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from data.ct_dataset import denormalize, normalize, read_stored, write_png  # noqa: E402
from inference_dir import DEVICES, build_generator, infer_full, resolve_device  # noqa: E402
from util.panel import build_panel  # noqa: E402
from util.residual_color import residual_rgb01, rgb_to_bgr  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Stage 1 (FE-GAN) パッチ切り出し: 512 フル推論 → patch 四方を切り出して EID の代表と並べる。引数はすべて必須")
    req = lambda *a, **k: p.add_argument(*a, required=True, **k)  # noqa: E731
    req("--weight_dir", help="重みディレクトリ（net_G.pth がある所）")
    req("--pcd_dir", help="PCD の根（machines.yaml pcd_dir）。<pcd_dir>/<PCD-nnn>/<PCD-nnn-sss>.png")
    req("--eid_dir", help="EID の根（machines.yaml eid_dir）")
    req("--out_dir", help="出力先（<repo>/output/<run>_<重みディレクトリ名>_crop_<実行時刻>/）")
    req("--device", choices=DEVICES)
    # G の構成（run の launch.yaml から）
    req("--netG"); req("--ngf", type=int); req("--input_nc", type=int); req("--output_nc", type=int); req("--norm")
    p.add_argument("--final_norm_act", action="store_true")
    p.add_argument("--residual", action="store_true")
    req("--hu_offset", type=int); req("--hu_min", type=int); req("--hu_max", type=int)
    # 表示（run の launch.yaml の log.*）
    req("--display_hu_min", type=int); req("--display_hu_max", type=int); req("--preview_bits", type=int, choices=(8, 16)); req("--diff_range_hu", type=int)
    # crop.yaml
    req("--patch", type=int); req("--panel_scale", type=int)
    req("--case", nargs=6, action="append", metavar=("PCD", "PCD_X", "PCD_Y", "EID", "EID_X", "EID_Y"), help="pcd スライス名 x y eid スライス名 x y（複数可）")
    return p.parse_args()


def slice_path(root, name):
    """スライス名 PCD-002-236 → <root>/PCD-002/PCD-002-236.png（症例フォルダ = 末尾の番号を除いた部分）。無ければエラー。"""
    case = name.rsplit("-", 1)[0]
    path = Path(root) / case / f"{name}.png"
    if not path.is_file():
        raise FileNotFoundError(f"スライスがありません: {path}（crop.yaml の cases を確認）")
    return path


def crop(arr, x, y, size, what):
    h, w = arr.shape[:2]
    if x + size > w or y + size > h:
        raise ValueError(f"{what}: 左上 ({x}, {y}) から {size} 四方が画像 {w}×{h} をはみ出します")
    return arr[y : y + size, x : x + size]


def main():
    a = parse_args()
    device = resolve_device(a.device)
    hu = (a.hu_offset, a.hu_min, a.hu_max)
    out_root = Path(a.out_dir)
    net, weight_path = build_generator(a, device)
    weight_name = Path(a.weight_dir).name
    lin = lambda stored: np.clip((stored.astype(np.float32) - a.hu_offset - a.display_hu_min) / float(a.display_hu_max - a.display_hu_min), 0.0, 1.0)  # noqa: E731
    quant = (lambda x01: (np.clip(x01, 0, 1) * 65535.0).round().astype(np.uint16)) if a.preview_bits == 16 else (lambda x01: (np.clip(x01, 0, 1) * 255.0).round().astype(np.uint8))
    up = lambda img01: cv2.resize(np.ascontiguousarray(img01, dtype=np.float32), None, fx=a.panel_scale, fy=a.panel_scale, interpolation=cv2.INTER_NEAREST)  # noqa: E731
    labels = ["EID", f"EID-like  G(z)   {weight_name}", "PCD  z", "R = G(z) - z   [HU]"]
    print(f"[crop] device={device} G={weight_path} patch={a.patch} panel_scale={a.panel_scale} cases={len(a.case)}")

    # 先に全ケースのファイルと座標を検査する（途中で止まって中途半端な出力を残さないため）
    plan = []
    for i, (pcd_name, px, py, eid_name, ex, ey) in enumerate(a.case, 1):
        px, py, ex, ey = int(px), int(py), int(ex), int(ey)
        pcd_path, eid_path = slice_path(a.pcd_dir, pcd_name), slice_path(a.eid_dir, eid_name)
        crop(read_stored(pcd_path), px, py, a.patch, f"case{i} PCD {pcd_name}")
        crop(read_stored(eid_path), ex, ey, a.patch, f"case{i} EID {eid_name}")
        plan.append((i, pcd_name, px, py, eid_name, ex, ey, pcd_path, eid_path))

    for i, pcd_name, px, py, eid_name, ex, ey, pcd_path, eid_path in plan:
        stored = read_stored(pcd_path)
        x_np = normalize(stored, *hu)
        with torch.no_grad():
            y = infer_full(net, torch.from_numpy(x_np).unsqueeze(0).unsqueeze(0).to(device))
        pcd16 = denormalize(x_np, *hu)                      # 正規化窓でクリップした入力（R の基準。inference_dir と同じ）
        eid16 = denormalize(y[0, 0].cpu().numpy(), *hu)     # EID-like
        R = eid16.astype(np.int32) - pcd16.astype(np.int32)  # ΔHU（推論の *_R/ と同じ差）
        eid_ref = read_stored(eid_path)
        c_pcd, c_eid_like, c_R = crop(pcd16, px, py, a.patch, f"case{i} PCD {pcd_name}"), crop(eid16, px, py, a.patch, f"case{i} EID-like"), crop(R, px, py, a.patch, f"case{i} R")
        c_eid = crop(eid_ref, ex, ey, a.patch, f"case{i} EID {eid_name}")
        g_eid, g_eid_like, g_pcd = lin(c_eid), lin(c_eid_like), lin(c_pcd)
        rgb_R = residual_rgb01(c_R, a.diff_range_hu)

        d = out_root / f"case{i}_{pcd_name}_{eid_name}"
        tag_p, tag_e = f"{pcd_name}_x{px}_y{py}", f"{eid_name}_x{ex}_y{ey}"
        write_png(d / f"1_EID_{tag_e}.png", quant(g_eid))
        write_png(d / f"2_EID-like_{tag_p}.png", quant(g_eid_like))
        write_png(d / f"3_PCD_{tag_p}.png", quant(g_pcd))
        write_png(d / f"4_R_color_{tag_p}.png", rgb_to_bgr(quant(rgb_R)))
        cols = [up(np.repeat(g[..., None], 3, 2)) for g in (g_eid, g_eid_like, g_pcd)] + [up(rgb_R)]
        write_png(d / "panel.png", rgb_to_bgr(quant(build_panel(cols, labels, a.diff_range_hu))))
        print(f"[crop] case{i}: PCD {pcd_name} ({px},{py}) / EID {eid_name} ({ex},{ey}) | ΔHU min {int(c_R.min())} max {int(c_R.max())} mean {float(c_R.mean()):.1f} -> {d}")
    print(f"[crop] done: {len(a.case)} cases -> {out_root}")


if __name__ == "__main__":
    main()
