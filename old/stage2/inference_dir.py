"""stage2/inference_dir.py — 学習済み U-Net で EID-like1024（または EID512 → 補間）→ PCD-like1024 をフォルダ丸ごと書き出す。[SpicaV5 新規 2026-09-10]
Stage 1 の inference_dir.py と同じ流儀。通常は run_infer.py（ホストからは bash start2.sh infer。重みディレクトリと入力フォルダは infer_stage2.sh の変数）から
**全引数明示**で exec される。引数はすべて必須（既定値なし）。

入力（--input_dir 直下に症例フォルダ、中に <slice>.png。uint16 1ch、stored = HU + hu_offset）。1 枚目の大きさで決める（全スライス同じでなければエラー）:
  image_size（学習画像 1024）  : そのまま U-Net に通す（EID-like1024 = bash start2.sh dataset の出力。PCD のテスト症例など）
  image_size / scale（512）    : data/ct_io.upsample（データセット作成・学習中の実 EID テストと同じ関数、run と同じ scale / interp）で 1024 にしてから通す
                                 （実 EID512 = EID_v5、Stage 1 の推論出力 full/ など）
入力の種類（--input。大きさからは区別できないので明示する）:
  eidlike : PCD 症例の EID-like（1024 / 512）。教師（--teacher）と指標が使える。パネルでは 1〜3 列目（入力 / 出力 / 教師）に入り、4・5 列目は --eid_slice の実 EID
  eid     : 実 EID512（EID_v5）。教師は無い。パネルでは 4・5 列目（出力 / 入力）に入り、1〜3 列目は --pcd_slice の参照 PCD 症例（EID-like1024 / 出力 / 教師）
教師（--teacher）: pcd1024_dir/<case>/<slice>.png を名前で読む。全スライスに無ければ開始前にエラー（黙って飛ばさない）。
  指標は util/metrics.py（学習の val と同じ: rmse = √mse は正規化空間 [−1,1]・0 へ、ssim は 1 へ、psnr は dB）+ 入力そのまま vs 教師の参照値 *_input。

出力（out_dir = <repo>/output/<run>_<重みディレクトリ名>_<入力フォルダ名>_<実行時刻>/。入力からの相対パスをそのまま保つので <case>/<slice>.png 構造になる）
  <out_dir>/full/<case>/<slice>.png            PCD-like1024（uint16、stored = HU + hu_offset。学習データと同じ規約）
  <out_dir>/full_input1024/<case>/<slice>.png  512 を補間した 1024 入力（uint16）。--save_input1024 かつ 512 入力のとき
  <out_dir>/full_panel/<case>/<slice>.png      表示用パネル。学習の固定パネルと同じ 5 列 [EID-like1024 | PCD-like1024 | PCD1024 | 実 EID → PCD-like1024 | 実 EID1024]
                                               （display_hu_min..max で線形、preview_bits の深度。util/panel.py full_labels）。--save_panel のとき。無い列（教師なし・参照なし）は詰める
  <out_dir>/eid/<eid_slice>_{eid1024,pcdlike}.png       input=eidlike: --eid_slice（eid_dir からの相対パス。512 なら補間）を 1 回だけ通した 1024 入力と出力（uint16）
  <out_dir>/ref/<pcd_slice>_{eidlike,pcdlike,pcd1024}.png  input=eid: --pcd_slice（eidlike1024_dir / pcd1024_dir からの相対パス）を 1 回だけ通した入力・出力・教師（uint16）
  <out_dir>/metrics.txt                       --teacher のとき。スライスごとの rmse / ssim / psnr と *_input、末尾に平均（rmse は mse の平均の √ = 学習の val と同じ）
  <out_dir>/infer.yaml                         解決済み設定（run_infer.py が書く）
速度: Stage 1 と同じ。読み込み（512 なら補間も）は --read_workers 本のスレッドで先読み、--batch_slices 枚をまとめて 1 回の forward、PNG の書き込みは --write_workers 本で非同期。
デバイスは --device cuda | cpu で明示（cuda 指定で使えなければエラー。cpu に落とさない）。
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
from data.ct_io import INTERP_FLAGS, CaseIndex, denormalize, normalize, read_stored, upsample, write_png  # noqa: E402
from models.unet import build_net  # noqa: E402
from util.metrics import mse, psnr, ssim  # noqa: E402
from util.panel import build_panel, full_labels  # noqa: E402

DEVICES = ("cuda", "cpu")
METRIC_COLS = ("rmse", "ssim", "psnr", "rmse_input", "ssim_input", "psnr_input")


def parse_args():
    p = argparse.ArgumentParser(description="Stage 2（U-Net）推論: EID-like1024 / EID512 → PCD-like1024 を uint16 PNG で書き出す。引数はすべて必須")
    req = lambda *a, **k: p.add_argument(*a, required=True, **k)  # noqa: E731
    # 場所・デバイス
    req("--weight_dir", help="重みディレクトリ（net_G.pth がある所: <run>/latest | <run>/best | <run>/weights/epoch_NNN）")
    req("--input_dir", help="処理する PNG 群のフォルダ（直下に症例フォルダ、または PNG 直置き）")
    req("--out_dir", help="出力先（<repo>/output/<run>_<重み>_<入力名>_<時刻>/）")
    req("--device", choices=DEVICES, help="cuda | cpu（machines.yaml の gpu_gen から。cuda が使えなければエラー）")
    req("--index_cache_dir", help="入力フォルダの列挙結果キャッシュ（<stage2_checkpoints_dir>/.case_index。学習と共通）")
    req("--pcd1024_dir", help="教師 PCD1024 の根（machines.yaml）。--teacher のときだけ読む")
    req("--eid_dir", help="実 EID512 の根（machines.yaml の eid_dir）。--save_panel かつ input=eidlike かつ --eid_slice のときだけ読む")
    req("--eidlike1024_dir", help="EID-like1024 の根（machines.yaml）。--save_panel かつ input=eid かつ --pcd_slice のときだけ読む（参照の 1 列目）")
    # U-Net の構成（run の launch.yaml の mode。学習時と同じネットを組む）
    req("--arch")
    req("--base_ch", type=int)
    req("--n_pool", type=int)
    req("--final_act")
    p.add_argument("--residual", action="store_true", help="mode.residual（出力 = 入力 + net(入力)）")
    req("--hu_offset", type=int)
    req("--hu_min", type=int)
    req("--hu_max", type=int)
    # 512 入力の補間（run の dataset_info.yaml = 学習入力の manifest と同じ方式）
    req("--scale", type=int, help="補間の倍率（512 → 1024 は 2）")
    req("--interp", choices=tuple(INTERP_FLAGS), help="補間の方式（cv2.resize）")
    req("--image_size", type=int, help="学習画像の一辺（1024）。入力がこれならそのまま、image_size / scale なら補間")
    # 表示（現在の configs/train.yaml の log）
    req("--display_hu_min", type=int, help="表示用の線形範囲の下限 HU。--save_panel にだけ使う")
    req("--display_hu_max", type=int, help="表示用の線形範囲の上限 HU。--save_panel にだけ使う")
    req("--preview_bits", type=int, choices=(8, 16), help="パネルのビット深度。--save_panel にだけ使う")
    # 方式（configs/infer.yaml）
    req("--input", choices=("eidlike", "eid"), help="入力の種類: eidlike（PCD 症例の EID-like）| eid（実 EID512）")
    req("--max_slices", type=int)
    req("--batch_slices", type=int, help="同時に U-Net に通すスライス数")
    p.add_argument("--teacher", action="store_true")
    p.add_argument("--save_input1024", action="store_true")
    p.add_argument("--save_panel", action="store_true")
    req("--eid_slice", help="input=eidlike のとき、パネルの 4・5 列目に並べる実 EID（eid_dir からの相対パス）。空文字で無し")
    req("--pcd_slice", help="input=eid のとき、パネルの 1〜3 列目に並べる参照 PCD 症例（eidlike1024_dir / pcd1024_dir からの相対パス）。空文字で無し")
    req("--read_workers", type=int, help="PNG の読み込み・デコード（と補間）を先読みするスレッド数（0 = 直列）")
    req("--write_workers", type=int, help="PNG の書き込みを非同期にするスレッド数（0 = 直列）")
    return p.parse_args()


# ---------------------------------------------------------------------------
def build_generator(a, device):
    """models/unet.build_net で学習時と同じ U-Net を組み、<weight_dir>/net_G.pth を strict に読む（形が違えばここで止まる）。"""
    mode = {"arch": a.arch, "base_ch": a.base_ch, "n_pool": a.n_pool, "final_act": a.final_act, "init_type": "torch"}  # 重みを読むので初期化はしない
    net = build_net(mode)
    path = Path(a.weight_dir) / "net_G.pth"
    if not path.is_file():
        raise FileNotFoundError(f"U-Net の重みがありません: {path}")
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    net.to(device).eval()
    return net, path


def resolve_device(name):
    """--device を検証して torch.device にする。cuda 指定で使えなければエラー（cpu に落とさない）。"""
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda が指定されたが CUDA が使えない（machines.yaml の gpu_gen を 0 にすれば cpu で動く）")
        return torch.device("cuda:0")
    return torch.device("cpu")


def write_metrics(path, rows, weight_path, input_dir, pcd1024_dir, hu):
    """metrics.txt: スライスごとの指標と末尾に平均。rmse の平均は mse の平均の √（学習の val と同じ）。"""
    hu_per_unit = (hu[2] - hu[1]) / 2.0
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Stage 2 infer: 出力 vs 教師（PCD1024）。rmse は正規化空間 [−1,1]（HU に直すなら × {hu_per_unit:.0f}）、ssim は 1 へ、psnr は dB（大きいほど良い）。*_input は入力そのまま vs 教師（参照値）\n")
        f.write(f"# weights={weight_path} input={input_dir} teacher={pcd1024_dir}\n")
        f.write("# slice\t" + "\t".join(METRIC_COLS) + "\n")
        for rel, r in rows:
            f.write(rel + "\t" + "\t".join(f"{r[k]:.6f}" if k.startswith("rmse") else f"{r[k]:.4f}" for k in METRIC_COLS) + "\n")
        n = len(rows)
        mean = {k: sum(r[k] for _, r in rows) / n for k in ("mse", "ssim", "psnr", "mse_input", "ssim_input", "psnr_input")}
        agg = {"rmse": mean["mse"] ** 0.5, "ssim": mean["ssim"], "psnr": mean["psnr"], "rmse_input": mean["mse_input"] ** 0.5, "ssim_input": mean["ssim_input"], "psnr_input": mean["psnr_input"]}
        f.write(f"# MEAN (n={n})\t" + "\t".join(f"{agg[k]:.6f}" if k.startswith("rmse") else f"{agg[k]:.4f}" for k in METRIC_COLS) + "\n")
    return agg


def main():
    a = parse_args()
    device = resolve_device(a.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # 入力の形が固定なので cuDNN に最速アルゴリズムを選ばせる（学習と同じ）
    input_dir, out_dir = Path(a.input_dir), Path(a.out_dir)
    hu = (a.hu_offset, a.hu_min, a.hu_max)
    if a.hu_min >= a.hu_max:
        raise ValueError(f"--hu_min < --hu_max: {a.hu_min} >= {a.hu_max}")
    if a.scale < 2 or a.image_size % a.scale:
        raise ValueError(f"--image_size {a.image_size} は --scale {a.scale}（≥ 2）の倍数")
    if a.save_panel and a.display_hu_min >= a.display_hu_max:
        raise ValueError(f"--display_hu_min < --display_hu_max にしてください: {a.display_hu_min} >= {a.display_hu_max}")
    small = a.image_size // a.scale

    net, weight_path = build_generator(a, device)
    forward = (lambda x: x + net(x)) if a.residual else net  # RegressionModel.forward と同じ

    idx = CaseIndex(input_dir, a.index_cache_dir)
    all_paths = [p for c in idx.cases for p in idx.slices[c]]
    paths = all_paths if a.max_slices == 0 else all_paths[: a.max_slices]
    first = read_stored(paths[0]).shape
    if first == (a.image_size, a.image_size):
        up = False
    elif first == (small, small):
        up = True
    else:
        raise ValueError(f"入力の大きさ {first} が学習画像 {a.image_size} でも補間元 {small}（= {a.image_size} / scale {a.scale}）でもありません: {paths[0]}")
    in_shape = first
    print(f"[infer] device={device} net={a.arch} base_ch {a.base_ch} n_pool {a.n_pool} final_act {a.final_act} residual {a.residual} weights={weight_path}")
    print(f"[infer] input={input_dir} ({idx.n_cases} cases, {idx.n_slices} slices, 推論 {len(paths)} 枚) size {first[0]}"
          + (f" → x{a.scale} {a.interp} → {a.image_size}" if up else "（そのまま）") + f" input={a.input} teacher={a.teacher} batch_slices={a.batch_slices} -> {out_dir}")

    teacher_of = {}
    if a.teacher:  # 先に全スライスの教師の存在を確認してから回す（途中で止まらないように。Stage 1 の DICOM 照合と同じ流儀）
        missing = []
        for p in paths:
            rel = Path(p).relative_to(input_dir)
            t = Path(a.pcd1024_dir) / rel
            if t.is_file():
                teacher_of[p] = t
            else:
                missing.append(str(rel))
        if missing:
            raise FileNotFoundError(f"--teacher ですが教師 {a.pcd1024_dir} に無いスライスが {len(missing)} 枚あります（実 EID など教師の無い入力は --teacher false）: "
                                    + ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else ""))
        print(f"[infer] teacher: {a.pcd1024_dir}（{len(teacher_of)} 枚すべて対応あり）→ metrics.txt")

    # --save_panel: 表示用パネル [入力1024 | PCD-like1024 | 教師 | 実 EID → PCD-like1024 | 実 EID1024]（学習の固定パネルと同じ並び。util/monitor.py の _lin01 / _rgb と同じ正規化表示）
    lin01 = lambda stored: np.repeat(np.clip((stored.astype(np.float32) - a.hu_offset - a.display_hu_min) / float(a.display_hu_max - a.display_hu_min), 0.0, 1.0)[..., None], 3, axis=2)  # noqa: E731
    quant = (  # noqa: E731
        (lambda x01: (np.clip(x01, 0, 1) * 65535.0).round().astype(np.uint16)) if a.preview_bits == 16
        else (lambda x01: (np.clip(x01, 0, 1) * 255.0).round().astype(np.uint8))
    )
    bgr = lambda rgb: np.ascontiguousarray(rgb[..., ::-1])  # noqa: E731  cv2.imwrite は BGR
    def load_1024(path, what):
        """参照スライスを読む。512 なら run の方式で補間、1024 ならそのまま。戻り値 (stored 1024, 補間したか)。"""
        s = read_stored(path)
        if s.shape == (small, small):
            return upsample(s, a.scale, a.interp), True
        if s.shape != (a.image_size, a.image_size):
            raise ValueError(f"{what} の大きさ {s.shape} が {a.image_size} でも {small} でもありません: {path}")
        return s, False

    def forward16(stored):
        with torch.inference_mode():
            return denormalize(forward(torch.from_numpy(normalize(stored, *hu))[None, None].to(device))[0, 0].cpu().numpy(), *hu)

    # パネルの相方（学習の固定パネルと同じ 5 列にする。1 回だけ通して全スライスに同じものを並べる）
    ref_right, eid_info = [], None      # input=eidlike: 4・5 列目 = --eid_slice の実 EID（出力 | 入力）
    ref_left, ref_stem = [], None       # input=eid    : 1〜3 列目 = --pcd_slice の参照 PCD 症例（EID-like1024 | 出力 | 教師）
    weight_name = Path(a.weight_dir).name
    if a.save_panel and a.input == "eidlike" and a.eid_slice:
        e_path = Path(a.eid_dir) / a.eid_slice
        e16, e_up = load_1024(e_path, "--eid_slice")
        e_out16 = forward16(e16)
        write_png(out_dir / "eid" / f"{e_path.stem}_eid1024.png", e16)
        write_png(out_dir / "eid" / f"{e_path.stem}_pcdlike.png", e_out16)
        eid_info = (e_path.stem, a.scale if e_up else None, a.interp if e_up else None)
        ref_right = [lin01(e_out16), lin01(e16)]
        print(f"[infer] panel: 4・5 列目 = 実 EID {e_path}" + (f"（x{a.scale} {a.interp}）" if e_up else "") + " → eid/ に 16bit")
    if a.save_panel and a.input == "eid" and a.pcd_slice:
        r_in_path, r_t_path = Path(a.eidlike1024_dir) / a.pcd_slice, Path(a.pcd1024_dir) / a.pcd_slice
        r16, _ = load_1024(r_in_path, "--pcd_slice")
        rt16 = read_stored(r_t_path)
        if rt16.shape != r16.shape:
            raise ValueError(f"--pcd_slice の教師の大きさが入力と違います: {rt16.shape} != {r16.shape}: {r_t_path}")
        r_out16 = forward16(r16)
        ref_stem = r_in_path.stem
        write_png(out_dir / "ref" / f"{ref_stem}_eidlike.png", r16)
        write_png(out_dir / "ref" / f"{ref_stem}_pcdlike.png", r_out16)
        write_png(out_dir / "ref" / f"{ref_stem}_pcd1024.png", rt16)
        ref_left = [lin01(r16), lin01(r_out16), lin01(rt16)]
        print(f"[infer] panel: 1〜3 列目 = 参照 PCD 症例 {r_in_path}（教師 {r_t_path}）→ ref/ に 16bit")
    # input=eidlike のラベルは全スライス共通。input=eid は 4・5 列目の名前がスライスごとに変わるのでループ内で作る
    panel_labels = full_labels(weight_name, None, a.teacher, eid_info, upsample=(a.scale, a.interp) if up else None) if a.input == "eidlike" else None

    # --- パイプライン（Stage 1 と同じ）: 読み込み（+ 補間 + 教師）はスレッドで先読み → batch_slices 枚をまとめて forward → PNG 書き込みはスレッドで非同期。順序は保つ ---
    reader = ThreadPoolExecutor(max_workers=a.read_workers) if a.read_workers > 0 else None
    writer = ThreadPoolExecutor(max_workers=a.write_workers) if a.write_workers > 0 else None
    pending = []  # 書き込みの future（例外は drain で表に出す。黙って失敗させない）
    tm = {"read": 0.0, "forward": 0.0, "write": 0.0}

    def load(p):
        stored = read_stored(p)
        if stored.shape != in_shape:
            raise ValueError(f"入力の大きさが揃っていません: {stored.shape} != {in_shape}: {p}")
        if up:
            stored = upsample(stored, a.scale, a.interp)
        t16 = None
        if p in teacher_of:
            t16 = read_stored(teacher_of[p])
            if t16.shape != stored.shape:
                raise ValueError(f"教師の大きさが入力（補間後）と違います: {t16.shape} != {stored.shape}: {teacher_of[p]}")
        return stored, normalize(stored, *hu), t16

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
            drain(4 * a.write_workers + 16)  # 溜めすぎない（1024 の uint16 は 2 MB/枚）

    batch_n = a.batch_slices
    prefetch = 2 * batch_n + (a.read_workers if reader is not None else 0)
    queue = collections.deque()
    it = iter(paths)

    def take(n):
        """次の n 枚を順序どおり [(path, in16, x_np, t16)] で返す（末尾は n 未満）。"""
        items = []
        for _ in range(n):
            if reader is None:
                p = next(it, None)
                if p is None:
                    break
                t = time.time()
                item = load(p)
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
                item = fut.result()
                tm["read"] += time.time() - t
            items.append((p,) + item)
        return items

    metric_rows = []
    t0 = time.time()
    bar = tqdm(total=len(paths), unit="slice", dynamic_ncols=True)
    try:
        while True:
            items = take(batch_n)
            if not items:
                break
            t = time.time()
            with torch.inference_mode():
                xb = torch.cat([torch.from_numpy(x_np)[None, None] for _, _, x_np, _ in items]).to(device)  # (K,1,H,W)。大きさは in_shape で揃えてある
                yb = forward(xb)
            tm["forward"] += time.time() - t
            for k, (p, in16, _x_np, t16) in enumerate(items):
                rel = Path(p).relative_to(input_dir)
                y = yb[k : k + 1]
                y16 = denormalize(y[0, 0].cpu().numpy(), *hu)
                put(out_dir / "full" / rel, y16)
                if up and a.save_input1024:
                    put(out_dir / "full_input1024" / rel, in16)
                if t16 is not None:
                    with torch.inference_mode():
                        tt = torch.from_numpy(normalize(t16, *hu))[None, None].to(device)
                        x = xb[k : k + 1]
                        row = {"mse": mse(y, tt), "ssim": ssim(y, tt), "psnr": psnr(y, tt), "mse_input": mse(x, tt), "ssim_input": ssim(x, tt), "psnr_input": psnr(x, tt)}
                    row["rmse"], row["rmse_input"] = row["mse"] ** 0.5, row["mse_input"] ** 0.5
                    metric_rows.append((str(rel), row))
                if a.save_panel:
                    if a.input == "eidlike":
                        cols, labels = [lin01(in16), lin01(y16)] + ([lin01(t16)] if t16 is not None else []) + ref_right, panel_labels
                    else:  # eid: [参照 EID-like1024 | 参照 PCD-like | 参照 PCD1024 | この EID → PCD-like | この EID]
                        cols = ref_left + [lin01(y16), lin01(in16)]
                        labels = full_labels(weight_name, ref_stem, True, (rel.stem, a.scale if up else None, a.interp if up else None))
                        if not ref_left:
                            labels = labels[3:]
                    put(out_dir / "full_panel" / rel, bgr(quant(build_panel(cols, labels))))
                bar.update(1)
        drain(0)  # 書き込みを全部待つ（失敗があればここで例外）
    finally:
        bar.close()
        if reader is not None:
            reader.shutdown(wait=False, cancel_futures=True)
        if writer is not None:
            writer.shutdown(wait=True)

    elapsed = time.time() - t0
    if metric_rows:
        agg = write_metrics(out_dir / "metrics.txt", metric_rows, weight_path, input_dir, a.pcd1024_dir, hu)
        print(f"[infer] metrics (n={len(metric_rows)}): rmse {agg['rmse']:.6f} (input {agg['rmse_input']:.6f}) ssim {agg['ssim']:.4f} (input {agg['ssim_input']:.4f}) "
              f"psnr {agg['psnr']:.2f} dB (input {agg['psnr_input']:.2f}) -> {out_dir / 'metrics.txt'}")
    print(f"[infer] time: 読み待ち {tm['read']:.1f}s | forward {tm['forward']:.1f}s | 書き待ち {tm['write']:.1f}s（先読み {a.read_workers} / 同時スライス {a.batch_slices} / 書き {a.write_workers}）")
    print(f"[infer] done: {len(paths)} slices, {elapsed:.1f}s ({elapsed / max(len(paths), 1):.2f} s/slice) -> {out_dir}")


if __name__ == "__main__":
    main()
