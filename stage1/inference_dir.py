"""inference_dir.py — 学習済み G で PCD → EID-like をディレクトリ丸ごと書き出す（フェーズ 7、症例丸ごと推論）。[SpicaV5 新規]

通常は run_infer.py（ホストからは bash start.sh infer。重みディレクトリと入力フォルダは infer_stage1.sh の変数）から**全引数明示**で exec される。
引数はすべて必須（既定値なし）。設計: docs/plans/20260907_inference-plan.md

mode
  full  : H×W をそのまま G に通す。G は全畳み込み + Haar 3 段なので H, W が 8 の倍数なら一発。満たさなければエラー（padding はしない）
  patch : patch_size の patch を patch_stride 刻みの**均等格子**で切り（nulmil image_cut と同じ「過不足なく被覆」。
          (H − p) が stride で割り切れれば普通の格子と一致）、patch_batch_size 枚ずつ G に通し、重なりを窓 patch_blend で重み付き平均する
  both  : 両方を保存し、スライスごとの |full − patch| の mean / max [HU] を diff_stats.txt に記録する（Q-I1 の実測用）

出力（out_dir = <repo>/output/<run>_<重みディレクトリ名>_<入力フォルダ名>_<実行時刻>/。2026-09-09 に run の下から移動）。入力ディレクトリからの相対パスをそのまま保つので、出力は学習データと同じ <症例>/<slice>.png 構造になる
  output_format（infer_stage1.sh の OUTPUT_FORMAT）
    png   : 16bit PNG（入力と同じ規約 stored = HU + hu_offset）
    dicom : 前処理を戻して DICOM（util/dicom_io.py。元 DICOM ルート --dicom_dir の参照ヘッダを継承し、HU を参照の RescaleSlope/Intercept で格納値に）
    both  : 両方
  <out_dir>/full/<case>/<slice>.png        EID-like（uint16、stored = HU + hu_offset。学習データと同じ規約）
  <out_dir>/full_dicom/<case>/<slice>.dcm  EID-like（DICOM）
  <out_dir>/full_R/<case>/<slice>.png      残差 R = G(z) − z（uint16、0 HU = 32768）。--save_residual のとき。eidlike = pcd + (R − 32768) が厳密に成り立つ。R は PNG のみ
  <out_dir>/full_R_color/<case>/<slice>.png  同じ R の表示用カラー（8bit RGB、512×512 のまま。白 = 0、純青 = −diff_range_hu、純赤 = +。util/residual_color.py）。--save_residual のとき
  <out_dir>/R_colorbar_pm<range>HU.png    上の凡例 1 枚（--diff_range_hu は現在の configs/train.yaml の log.diff_range_hu。run_infer.py が渡す。run の launch.yaml ではない）
  <out_dir>/patch/..., patch_dicom/..., patch_R/..., patch_R_color/...
  <out_dir>/diff_stats.txt                 mode = both のとき

BN は eval（running 統計）。学習の checkpoint 時のフル画像（util/monitor.py save_full_images）と同じ。
速度（2026-09-09）: 読み込み・デコードは --read_workers 本のスレッドで先読み、--batch_slices 枚のスライスをまとめて G に通す（full はそのまま 1 回の forward、patch は全スライスの patch を束ねて --patch_batch_size ずつ。BN は eval なので 1 枚ずつと同じ結果）、
PNG の書き込みは --write_workers 本のスレッドで非同期。GPU は数 ms で終わるので、律速は PNG の読み書き（HDD + WSL 越しだと 1 枚 0.1〜0.3 s）。終了時に内訳（読み待ち / full / patch / 書き待ち）を表示する。
デバイスは --device cuda | cpu で明示（run_infer.py が machines.yaml の gpu_gen から決める）。cuda 指定で使えなければエラー。
"""

import argparse
import collections
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pydicom
import torch
from pydicom.uid import generate_uid
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from data.ct_dataset import CaseIndex, denormalize, normalize, read_stored, residual_stored, write_png  # noqa: E402
from util.residual_color import colorbar_filename, colorbar_rgb01, residual_rgb8, rgb_to_bgr  # noqa: E402
from models import networks  # noqa: E402
from util.dicom_io import DicomIndex, check_pixels, check_rescale, check_series_unique, write_like_reference  # noqa: E402

MODES = ("full", "patch", "both")
BLENDS = ("uniform", "hann")
OUTPUT_FORMATS = ("png", "dicom", "both")
DEVICES = ("cuda", "cpu")
LEVELS = 3  # Haar 3 段 → H, W は 2**3 の倍数


def parse_args():
    p = argparse.ArgumentParser(description="Stage 1 (FE-GAN) 推論: PCD → EID-like を uint16 PNG で書き出す。引数はすべて必須")
    req = lambda *a, **k: p.add_argument(*a, required=True, **k)  # noqa: E731
    # 場所・デバイス
    req("--weight_dir", help="重みディレクトリ（net_G.pth がある所: <run>/latest | <run>/best | <run>/weights/epoch_NNN）")
    req("--input_dir", help="処理する PNG 群のフォルダ（直下に症例フォルダ、または PNG 直置き）")
    req("--out_dir", help="出力先（通常 <run>/infer/<重みディレクトリ名>/<入力フォルダ名>）")
    req("--device", choices=DEVICES, help="cuda | cpu（machines.yaml の gpu_gen から。cuda が使えなければエラー）")
    req("--index_cache_dir", help="入力フォルダの列挙結果キャッシュ（<checkpoints_dir>/.case_index。学習と共通）")
    # G の構成（run の launch.yaml から。学習時と同じ G を組む）
    req("--netG")
    req("--ngf", type=int)
    req("--input_nc", type=int)
    req("--output_nc", type=int)
    req("--norm")
    p.add_argument("--final_norm_act", action="store_true")
    req("--hu_offset", type=int)
    req("--hu_min", type=int)
    req("--hu_max", type=int)
    req("--diff_range_hu", type=int, help="R のカラー表示の ±範囲 HU（run の launch.yaml の log.diff_range_hu。学習時の TensorBoard と同じ）")
    # 方式（configs/infer.yaml）
    req("--mode", choices=MODES)
    req("--patch_size", type=int)
    req("--patch_stride", type=int)
    req("--patch_blend", choices=BLENDS)
    req("--patch_batch_size", type=int)
    req("--max_slices", type=int)
    p.add_argument("--save_residual", action="store_true")
    req("--batch_slices", type=int, help="同時に G に通すスライス数（full はそのまま、patch は全スライスの patch を束ねて patch_batch_size ずつ）")
    req("--read_workers", type=int, help="PNG の読み込み・デコードを先読みするスレッド数（0 = 直列）")
    req("--write_workers", type=int, help="PNG の書き込みを非同期にするスレッド数（0 = 直列）")
    # 出力形式（infer_stage1.sh の OUTPUT_FORMAT / DICOM_DIR）
    req("--output_format", choices=OUTPUT_FORMATS, help="png | dicom | both")
    req("--dicom_dir", help="元 DICOM ルート（dicom / both のとき参照ヘッダに使う。png のときは使わない）")
    req("--rescale_slope", type=float, help="infer.yaml dicom.rescale_slope（参照 DICOM と一致しなければエラー）")
    req("--rescale_intercept", type=float, help="infer.yaml dicom.rescale_intercept")
    return p.parse_args()


# ---------------------------------------------------------------------------
# G
# ---------------------------------------------------------------------------
def build_generator(a, device):
    """networks.define_G で学習時と同じ G を組み、<weight_dir>/net_G.pth を strict に読む。"""
    net = networks.define_G(a.input_nc, a.output_nc, a.ngf, a.netG, a.norm, False, "normal", 0.02, final_norm_act=a.final_norm_act)
    path = Path(a.weight_dir) / "net_G.pth"
    if not path.is_file():
        raise FileNotFoundError(f"G の重みがありません: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if hasattr(state, "_metadata"):
        del state._metadata
    net.load_state_dict(state)
    net.to(device).eval()
    return net, path


# ---------------------------------------------------------------------------
# full / patch
# ---------------------------------------------------------------------------
def infer_full(net, x):
    """x: (1,1,H,W)。H, W が 2^LEVELS の倍数でなければエラー。"""
    h, w = x.shape[-2:]
    m = 2**LEVELS
    if h % m or w % m:
        raise ValueError(f"full モードは H, W が {m} の倍数である必要があります: {h}×{w}")
    return net(x)


def grid_positions(length, patch, stride):
    """patch の左上座標を stride 刻みで、端まで過不足なく被覆するよう均等配置する（nulmil image_cut の考え方）。
    (length − patch) が stride で割り切れれば 0, stride, 2·stride, ... の普通の格子と一致する。"""
    if length < patch:
        raise ValueError(f"画像 ({length}) が patch ({patch}) より小さい")
    n = math.ceil((length - patch) / stride) + 1
    return np.round(np.linspace(0, length - patch, n)).astype(int).tolist()


def blend_window(patch, blend, device):
    """重なり合成の 2D 窓 (patch, patch)。uniform = 単純平均。hann = (i+0.5)/p で評価した Hann（端でも 0 にならない）。"""
    if blend == "uniform":
        w1 = torch.ones(patch)
    elif blend == "hann":
        i = torch.arange(patch, dtype=torch.float32)
        w1 = 0.5 - 0.5 * torch.cos(2.0 * math.pi * (i + 0.5) / patch)
    else:
        raise ValueError(blend)
    return torch.outer(w1, w1).to(device)


def infer_patch(net, x, patch, stride, window, batch_size):
    """x: (K,1,H,W)。K 枚のスライスの patch を全部まとめて切り出し、batch_size 枚ずつ G に通し、窓で重み付き平均して (K,1,H,W) を返す。
    2026-09-09: patch ごとの Python ループを廃止。切り出しは平坦 index の gather 1 回、合成は index_add_（重なりは加算）で、
    バッチ 1 回につき G の呼び出し + 数カーネルだけ。スライスをまたいで束ねるので、スライス末尾の半端なバッチと GPU 同期も K 枚に 1 回になる。
    旧実装は 1 枚ずつ、patch ごとに 2 回の小さな GPU 演算（2,401 patch で約 5,000 回/枚）を 1 本の Python スレッドで発行していて、
    GPU も CPU も遊んだまま 0.5 s/枚かかっていた（PC3、patch.batch_size 256 でも変わらず）。
    index_add_ の重なりの加算順は CUDA では不定なので float の下位桁が変わりうる（uint16 化後に 1 違うことがある程度）。"""
    k, _, h, w = x.shape
    dev = x.device
    ys, xs = grid_positions(h, patch, stride), grid_positions(w, patch, stride)
    base1 = torch.tensor([y * w + xx for y in ys for xx in xs], dtype=torch.int64, device=dev)  # (N,) 1 枚の中の各 patch の左上の平坦 index
    base = (torch.arange(k, device=dev)[:, None] * (h * w) + base1[None, :]).reshape(-1)  # (K*N,) スライス k の分は k*H*W だけずらす
    offs = (torch.arange(patch, device=dev)[:, None] * w + torch.arange(patch, device=dev)[None, :]).reshape(-1)  # (p*p,) patch 内の相対 index
    flat = x.reshape(-1)
    wflat = window.reshape(-1).to(x.dtype)
    out = torch.zeros(k * h * w, dtype=x.dtype, device=dev)
    wsum = torch.zeros(k * h * w, dtype=x.dtype, device=dev)
    for s in range(0, base.shape[0], batch_size):
        idx = base[s : s + batch_size, None] + offs[None, :]  # (b, p*p)
        o = net(flat[idx].view(-1, 1, patch, patch))  # (b,1,p,p)。gather 1 回で切り出し
        out.index_add_(0, idx.reshape(-1), (o.reshape(idx.shape[0], -1) * wflat).reshape(-1))  # 重み付きで重なりを加算
        wsum.index_add_(0, idx.reshape(-1), wflat.expand(idx.shape[0], -1).reshape(-1))
    # F-08: 被覆されない画素（wsum = 0 → 0/0 = NaN）や非有限値を黙って出さない（stride ≤ size は schema でも検査。二重防御）
    if bool((wsum <= 0).any()):
        raise RuntimeError(f"patch 合成で被覆されない画素があります（patch {patch}, stride {stride}）。patch_stride ≤ patch_size にしてください")
    y_out = (out / wsum).view(k, 1, h, w)
    if not bool(torch.isfinite(y_out).all()):
        raise RuntimeError("patch 合成の結果に NaN / inf が含まれます")
    return y_out


# ---------------------------------------------------------------------------
def resolve_device(name):
    """--device を検証して torch.device にする。cuda 指定で使えなければエラー（cpu に落とさない）。"""
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda が指定されたが CUDA が使えない（machines.yaml の gpu_gen を 0 にすれば cpu で動く）")
        return torch.device("cuda:0")
    return torch.device("cpu")


def main():
    a = parse_args()
    device = resolve_device(a.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # 入力の形が固定なので cuDNN に最速アルゴリズムを選ばせる（2026-09-09）
    write_png_out = a.output_format in ("png", "both")
    write_dcm = a.output_format in ("dicom", "both")
    dicom_index = DicomIndex(a.dicom_dir) if write_dcm else None  # 無ければここで止まる
    input_dir = Path(a.input_dir)
    out_dir = Path(a.out_dir)
    hu = (a.hu_offset, a.hu_min, a.hu_max)
    hu_per_unit = (a.hu_max - a.hu_min) / 2.0  # 正規化 [−1,1] の 1.0 が何 HU か

    net, weight_path = build_generator(a, device)
    modes = ["full", "patch"] if a.mode == "both" else [a.mode]
    window = blend_window(a.patch_size, a.patch_blend, device) if "patch" in modes else None
    if "patch" in modes and a.patch_size % (2**LEVELS):
        raise ValueError(f"patch_size は {2**LEVELS} の倍数である必要があります: {a.patch_size}")

    idx = CaseIndex(input_dir, a.index_cache_dir)
    paths = idx.all_paths if a.max_slices == 0 else idx.all_paths[: a.max_slices]
    n_patches = None
    print(f"[infer] device={device} G={weight_path}")
    print(f"[infer] input={input_dir} ({idx.n_cases} cases, {idx.n_slices} slices, 推論 {len(paths)} 枚) mode={a.mode} output={a.output_format} -> {out_dir}")
    if write_dcm:
        rels_by_case = {}
        for p in paths:  # 先に全部の対応と Rescale の一致・症例内の Series 一意性を確認してから回す（途中で止まらないように）
            ref = dicom_index.reference_for(Path(p).stem)
            check_rescale(pydicom.dcmread(ref, stop_before_pixels=True), a.rescale_slope, a.rescale_intercept, ref)
            rels_by_case.setdefault(str(Path(p).relative_to(input_dir).parent), []).append(Path(p).stem)
        check_series_unique(dicom_index, rels_by_case)  # F-18c
        series_uid = {}  # (mode, case) → SeriesInstanceUID（症例ごと・モードごとに 1 本）
        run_name = Path(a.weight_dir).parent.name if Path(a.weight_dir).parent.name != "weights" else Path(a.weight_dir).parent.parent.name
        print(f"[infer] dicom: 参照ルート {dicom_index.root}（{len(dicom_index.cases)} 症例）")

    if a.save_residual:  # R のカラー表示の凡例（<out_dir>/ 直下に 1 枚。範囲を名前に入れる）
        if a.diff_range_hu <= 0:
            raise ValueError(f"--diff_range_hu は正: {a.diff_range_hu}")
        out_dir.mkdir(parents=True, exist_ok=True)
        write_png(out_dir / colorbar_filename(a.diff_range_hu), rgb_to_bgr((colorbar_rgb01(a.diff_range_hu, 512) * 255.0).round().astype(np.uint8)))
    # --- パイプライン（2026-09-09）: 読み込みはスレッドで先読み → batch_slices 枚をまとめて forward（full はそのまま、patch は全スライスの patch を束ねる）→ PNG 書き込みはスレッドで非同期 ---
    #     GPU は数 ms で終わるので、律速は PNG の読み書き（PC2 は HDD + WSL 越しで 1 枚 0.1〜0.3 s）。読み・計算・書きを同時に動かして読み書きの上限まで詰める。
    #     順序は保つ（先読みは deque に submit 順で積み、先頭から取る）。DICOM の照合・書き出しは pydicom なので主スレッドで直列のまま。
    reader = ThreadPoolExecutor(max_workers=a.read_workers) if a.read_workers > 0 else None
    writer = ThreadPoolExecutor(max_workers=a.write_workers) if a.write_workers > 0 else None
    pending = []  # 書き込みの future（例外は drain で表に出す。黙って失敗させない）
    tm = {"read": 0.0, "full": 0.0, "patch": 0.0, "write": 0.0}

    def load(p):
        stored = read_stored(p)
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
            drain(4 * a.write_workers + 16)  # 溜めすぎない（メモリ）

    batch_n = a.batch_slices
    prefetch = 2 * batch_n + (a.read_workers if reader is not None else 0)
    queue = collections.deque()
    it = iter(paths)

    def take(n):
        """次の n 枚を順序どおり [(path, stored, x_np)] で返す（末尾は n 未満）。"""
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

    diff_lines = []
    t0 = time.time()
    bar = tqdm(total=len(paths), unit="slice", dynamic_ncols=True)
    try:
        while True:
            items = take(batch_n)
            if not items:
                break
            outs_full = None
            if "full" in modes:
                xs = [torch.from_numpy(x_np).unsqueeze(0).unsqueeze(0) for _, _, x_np in items]
                t = time.time()
                with torch.inference_mode():
                    if all(x.shape == xs[0].shape for x in xs):
                        outs_full = list(infer_full(net, torch.cat(xs).to(device)).split(1))  # まとめて 1 回（形が同じときだけ）
                    else:
                        outs_full = [infer_full(net, x.to(device)) for x in xs]
                tm["full"] += time.time() - t
            outs_patch = None
            if "patch" in modes:
                xs = [torch.from_numpy(x_np).unsqueeze(0).unsqueeze(0) for _, _, x_np in items]
                if n_patches is None:
                    n_patches = len(grid_positions(xs[0].shape[-2], a.patch_size, a.patch_stride)) * len(grid_positions(xs[0].shape[-1], a.patch_size, a.patch_stride))
                    tqdm.write(f"[infer] patch: {a.patch_size}px stride {a.patch_stride} → {n_patches} patch/枚, blend={a.patch_blend}, patch_batch={a.patch_batch_size}, slices/forward={a.batch_slices}")
                t = time.time()
                with torch.inference_mode():
                    if all(x.shape == xs[0].shape for x in xs):
                        outs_patch = list(infer_patch(net, torch.cat(xs).to(device), a.patch_size, a.patch_stride, window, a.patch_batch_size).split(1))  # K 枚の patch を束ねて
                    else:
                        outs_patch = [infer_patch(net, x.to(device), a.patch_size, a.patch_stride, window, a.patch_batch_size) for x in xs]
                tm["patch"] += time.time() - t
            for k, (p, stored, x_np) in enumerate(items):
                rel = Path(p).relative_to(input_dir)
                outs = {}
                if outs_full is not None:
                    outs["full"] = outs_full[k]
                if outs_patch is not None:
                    outs["patch"] = outs_patch[k]
                pcd16 = denormalize(x_np, *hu)  # 正規化窓でクリップした入力（R の基準）
                ref_ds = None
                if write_dcm:  # F-18a: このスライスの参照 DICOM が本当に入力 PNG の元か、画素で照合（違えばここで止まる）
                    ref_ds = check_pixels(dicom_index.reference_for(rel.stem), stored, a.hu_offset, str(rel))
                for m, y in outs.items():
                    eid16 = denormalize(y[0, 0].cpu().numpy(), *hu)
                    if write_png_out:
                        put(out_dir / m / rel, eid16)
                    if write_dcm:
                        key = (m, rel.parent)
                        if key not in series_uid:
                            series_uid[key] = generate_uid()
                        write_like_reference(dicom_index.reference_for(rel.stem), eid16.astype(np.float64) - a.hu_offset,
                                             out_dir / f"{m}_dicom" / rel.with_suffix(".dcm"), series_uid[key],
                                             f"SpicaV5 EID-like {run_name}/{Path(a.weight_dir).name} {m}", a.rescale_slope, a.rescale_intercept, ds=ref_ds)
                    if a.save_residual:
                        put(out_dir / f"{m}_R" / rel, residual_stored(pcd16, eid16))
                        delta_hu = eid16.astype(np.int32) - pcd16.astype(np.int32)  # = R_stored − 32768（保存した 16bit と同じ差）
                        put(out_dir / f"{m}_R_color" / rel, rgb_to_bgr(residual_rgb8(delta_hu, a.diff_range_hu)))
                if a.mode == "both":
                    d = (outs["full"] - outs["patch"]).abs() * hu_per_unit
                    diff_lines.append((str(rel), float(d.mean()), float(d.max())))
                bar.update(1)
        drain(0)  # 書き込みを全部待つ（失敗があればここで例外）
    finally:
        bar.close()
        if reader is not None:
            reader.shutdown(wait=False, cancel_futures=True)
        if writer is not None:
            writer.shutdown(wait=True)

    elapsed = time.time() - t0
    if a.mode == "both":
        with open(out_dir / "diff_stats.txt", "w", encoding="utf-8") as f:
            f.write(f"# |full − patch| [HU]  patch={a.patch_size} stride={a.patch_stride} blend={a.patch_blend}  G={weight_path}\n")
            f.write("# slice\tmean\tmax\n")
            for rel, mean, mx in diff_lines:
                f.write(f"{rel}\t{mean:.3f}\t{mx:.3f}\n")
            if diff_lines:
                f.write(f"# ALL\tmean_of_means={np.mean([m for _, m, _ in diff_lines]):.3f}\tmax={max(x for _, _, x in diff_lines):.3f}\n")
        if diff_lines:
            print(f"[infer] |full − patch|: mean {np.mean([m for _, m, _ in diff_lines]):.3f} HU, max {max(x for _, _, x in diff_lines):.3f} HU -> {out_dir / 'diff_stats.txt'}")
    print(f"[infer] time: 読み待ち {tm['read']:.1f}s | full {tm['full']:.1f}s | patch {tm['patch']:.1f}s | 書き待ち {tm['write']:.1f}s（先読み {a.read_workers} / 同時スライス {a.batch_slices} / patch_batch {a.patch_batch_size} / 書き {a.write_workers}）")
    print(f"[infer] done: {len(paths)} slices, {elapsed:.1f}s ({elapsed / max(len(paths), 1):.2f} s/slice) -> {out_dir}")


if __name__ == "__main__":
    main()
