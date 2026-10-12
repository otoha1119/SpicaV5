"""crop_patches.py — パッチ切り出し（bash start.sh crop → crop_stage1.sh → run_crop.py → ここ）。引数はすべて必須（run_crop.py が組む）。

指定した PCD スライスを 512 のフル推論にかけ、左上 (x, y) から patch 四方を切り出す。EID は別スライスの指定座標から同じ大きさで切る（PCD と EID はペア）。
保存は表示用のみ（16bit 生データは無し。2026-09-08 ユーザー指示）。preview_fixed_*/ と同じ流儀（表示範囲 display_hu_min..max で線形、preview_bits の深度、R は util/residual_color のカラー）。
task（run の launch.yaml の mode.task。run_crop.py が --task で渡す）[2026-10-12]: crop.yaml の pcd / eid はモダリティ名のまま、どちらが入力かが task で決まる
  degrade : 入力 = pcd（pcd_dir = PCD512、座標 pcd_x/y が入力・G(z)・R の切り出し）、参照 = eid（eid_dir = EID512、eid_x/y）。出力 1_EID / 2_EID-like / 3_PCD / 4_R_color
  sr      : 入力 = eid（eid_dir = EID1024、座標 eid_x/y が入力・G(z)・R の切り出し。1024 の画素で書く）、参照 = pcd（pcd_dir = PCD1024、pcd_x/y）。出力 1_PCD / 2_PCD-like / 3_EID / 4_R_color

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
from inference_dir import DEVICES, add_generator_args, build_generator, infer_full, resolve_device  # noqa: E402
from models.networks import noise_wrapper  # noqa: E402
from util.panel import build_panel  # noqa: E402
from util.residual_color import residual_rgb01, rgb_to_bgr  # noqa: E402
from util.task_names import TASKS, names as task_names  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Stage 1 (FE-GAN) パッチ切り出し: 512 フル推論 → patch 四方を切り出して EID の代表と並べる。引数はすべて必須")
    req = lambda *a, **k: p.add_argument(*a, required=True, **k)  # noqa: E731
    req("--weight_dir", help="重みディレクトリ（net_G.pth がある所）")
    req("--pcd_dir", help="PCD の根（machines.yaml pcd_dir）。<pcd_dir>/<PCD-nnn>/<PCD-nnn-sss>.png")
    req("--eid_dir", help="EID の根（machines.yaml eid_dir）")
    req("--out_dir", help="出力先（<repo>/output/<run>_<重みディレクトリ名>_crop_<実行時刻>/）")
    req("--device", choices=DEVICES)
    # G の構成（run の launch.yaml から。inference_dir.py と共通。noise_input のときは --sigma_hu_min / max も）
    add_generator_args(p)
    req("--task", choices=TASKS, help="run の launch.yaml の mode.task（degrade | sr）。どちらのモダリティが入力かと表示名")
    p.add_argument("--sr_scale", type=int, default=None); p.add_argument("--sr_interp", type=str, default=None); p.add_argument("--sr_input_size", type=int, default=None)  # run_infer.generator_argv が sr で付ける（crop は 1024 を直接読むので使わない。記録用）
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
    nm = task_names(a.task)
    labels = [nm["B"], f"{nm['GEN']}  G(z)   {weight_name}", f"{nm['A']}  z", "R = G(z) - z   [HU]"]
    print(f"[crop] task={a.task} device={device} G={weight_path} patch={a.patch} panel_scale={a.panel_scale} cases={len(a.case)}")

    # 先に全ケースのファイルと座標を検査する（途中で止まって中途半端な出力を残さないため）
    # 入力（A、G に通す側）と参照（B、並べるだけ）を task で決める: degrade は pcd が入力、sr は eid が入力
    plan = []
    for i, (pcd_name, px, py, eid_name, ex, ey) in enumerate(a.case, 1):
        px, py, ex, ey = int(px), int(py), int(ex), int(ey)
        pcd_path, eid_path = slice_path(a.pcd_dir, pcd_name), slice_path(a.eid_dir, eid_name)
        if a.task == "sr":
            in_name, ix, iy, in_path, ref_name, rx, ry, ref_path = eid_name, ex, ey, eid_path, pcd_name, px, py, pcd_path
        else:
            in_name, ix, iy, in_path, ref_name, rx, ry, ref_path = pcd_name, px, py, pcd_path, eid_name, ex, ey, eid_path
        crop(read_stored(in_path), ix, iy, a.patch, f"case{i} {nm['A']} {in_name}")
        crop(read_stored(ref_path), rx, ry, a.patch, f"case{i} {nm['B']} {ref_name}")
        plan.append((i, in_name, ix, iy, in_path, ref_name, rx, ry, ref_path))

    nw = noise_wrapper(net)  # noise_input の run だけ非 None
    for i, in_name, ix, iy, in_path, ref_name, rx, ry, ref_path in plan:
        stored = read_stored(in_path)
        x_np = normalize(stored, *hu)
        x = torch.from_numpy(x_np).unsqueeze(0).unsqueeze(0).to(device)
        noise_note = ""
        with torch.no_grad():
            if nw is None:
                y = infer_full(net, x)
            else:  # 再現用: スライス名の seed と、その seed からの sigma（推論の sigma_mode random・学習の preview と同じ n）
                n, seed, sigma_hu = nw.slice_noise(in_name, x.shape[-2:], device)
                y = infer_full(net, x, n=n)
                noise_note = f" | n: seed {seed} sigma {sigma_hu:.1f}HU"
        in16 = denormalize(x_np, *hu)                       # 正規化窓でクリップした入力 A（R の基準。inference_dir と同じ）
        gen16 = denormalize(y[0, 0].cpu().numpy(), *hu)     # G(z)
        R = gen16.astype(np.int32) - in16.astype(np.int32)  # ΔHU（推論の *_R/ と同じ差）
        ref = read_stored(ref_path)
        c_in, c_gen, c_R = crop(in16, ix, iy, a.patch, f"case{i} {nm['A']} {in_name}"), crop(gen16, ix, iy, a.patch, f"case{i} {nm['GEN']}"), crop(R, ix, iy, a.patch, f"case{i} R")
        c_ref = crop(ref, rx, ry, a.patch, f"case{i} {nm['B']} {ref_name}")
        g_ref, g_gen, g_in = lin(c_ref), lin(c_gen), lin(c_in)
        rgb_R = residual_rgb01(c_R, a.diff_range_hu)

        d = out_root / f"case{i}_{in_name}_{ref_name}"
        tag_i, tag_r = f"{in_name}_x{ix}_y{iy}", f"{ref_name}_x{rx}_y{ry}"
        write_png(d / f"1_{nm['B']}_{tag_r}.png", quant(g_ref))
        write_png(d / f"2_{nm['GEN']}_{tag_i}.png", quant(g_gen))
        write_png(d / f"3_{nm['A']}_{tag_i}.png", quant(g_in))
        write_png(d / f"4_R_color_{tag_i}.png", rgb_to_bgr(quant(rgb_R)))
        cols = [up(np.repeat(g[..., None], 3, 2)) for g in (g_ref, g_gen, g_in)] + [up(rgb_R)]
        write_png(d / "panel.png", rgb_to_bgr(quant(build_panel(cols, labels, a.diff_range_hu))))
        print(f"[crop] case{i}: {nm['A']} {in_name} ({ix},{iy}) / {nm['B']} {ref_name} ({rx},{ry}) | ΔHU min {int(c_R.min())} max {int(c_R.max())} mean {float(c_R.mean()):.1f}{noise_note} -> {d}")
    print(f"[crop] done: {len(a.case)} cases -> {out_root}")


if __name__ == "__main__":
    main()
