"""inference_dir.py — 学習済み G で A → G(z)（degrade: PCD → EID-like、sr: EID1024 → PCD-like1024）をディレクトリ丸ごと書き出す（フェーズ 7、症例丸ごと推論）。[SpicaV5 新規]

task（run の launch.yaml の mode.task。run_infer.py が --task で渡す）[2026-10-12]
  degrade : 従来どおり。入力 = PCD512 など、出力 = EID-like
  sr      : 入力は 1 枚ごとに大きさで判定: 一辺 = sr_input_size（512 = 実 EID_v5）なら launch の sr.upsample の方式（scale / interp = 学習入力 EID1024 の manifest）で
            補間してから G に通し、その 1024 入力も <mode>_input/<case>/<slice>.png に 16bit で残す。一辺 = input_size × scale（1024）ならそのまま。他の大きさはエラー。
            出力 = PCD-like1024（<mode>/）。R = G(z) − z は補間後の入力に対する差。DICOM 出力は未対応（参照 DICOM が 512 で行列が変わるため。--output_format png）

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
  <out_dir>/full_panel/<case>/<slice>.png 表示用パネル [EID-like | PCD | R + ゲージ]（学習の output_images/preview_* と同じ正規化表示。display_hu_min..max で線形、preview_bits の深度、util/panel.py）。--save_panel のとき
  <out_dir>/R_colorbar_pm<range>HU.png    上の凡例 1 枚（--diff_range_hu は現在の configs/train.yaml の log.diff_range_hu。run_infer.py が渡す。run の launch.yaml ではない）。--save_residual のとき
  <out_dir>/patch/..., patch_dicom/..., patch_R/..., patch_R_color/..., patch_panel/...
  <out_dir>/diff_stats.txt                 mode = both のとき
  <out_dir>/noise.yaml                     乱数画像 n の記録（noise_input / sigma_hu_min / max / sigma_mode / sigma_hu_fixed / n_draws とスライスごとの seed・sigma_hu）。常に書く（n の無い run は noise_input: false だけ）
  <out_dir>/{full,patch}_std/<case>/<slice>.png   --noise_n_draws ≥ 2 のとき: n_draws 通りの n の出力の画素ごとの標準偏差 [HU]（uint16、stored = round(std_HU)。0 HU = 0、オフセット無し）
  <out_dir>/std_stats.txt                  同じく、スライスごとの標準偏差の平均 / 最大 [HU]

乱数画像 n（run の launch の mode.noise_input = true。段階 5-5、2026-10-12。networks.NoiseInputGenerator）
  seed はスライス名（stem、例 PCD-002-215）から zlib.crc32 で決める（決定的。同じスライスは何回推論しても同じ出力）。sigma_hu は --noise_sigma_mode random なら seed から log-uniform [--sigma_hu_min, max]、
  fixed なら --noise_sigma_hu_fixed。1 スライスにつき H×W 全体の n を 1 枚作り、full はそのまま、patch は各 patch の対応領域を切り出して使う（full と patch で同じ n。
  patch ごとに独立に引くと重なりの平均で粒が薄まる）。--noise_n_draws k ≥ 2 なら draw 0..k−1（seed + draw）で k 回推論し、平均を通常の出力（<mode>/）に、標準偏差を <mode>_std/ に書く。

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
import yaml
from pydicom.uid import generate_uid
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from data.ct_dataset import CaseIndex, denormalize, normalize, read_stored, residual_stored, upsample, write_png  # noqa: E402
from util.residual_color import colorbar_filename, colorbar_rgb01, residual_rgb01, residual_rgb8, rgb_to_bgr  # noqa: E402
from util.panel import build_panel  # noqa: E402
from util.task_names import TASKS, names as task_names  # noqa: E402
from models import networks  # noqa: E402
from models.networks import noise_wrapper  # noqa: E402
from util.dicom_io import DicomIndex, check_pixels, check_rescale, check_series_unique, write_like_reference  # noqa: E402

MODES = ("full", "patch", "both")
BLENDS = ("uniform", "hann")
OUTPUT_FORMATS = ("png", "dicom", "both")
DEVICES = ("cuda", "cpu")
SIGMA_MODES = ("random", "fixed")
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
    # G の構成（run の launch.yaml から。学習時と同じ G を組む。crop_patches.py と共通）
    add_generator_args(p)
    req("--task", choices=TASKS, help="run の launch.yaml の mode.task（degrade | sr）。表示名と、sr のときの 512 入力の補間を決める")
    p.add_argument("--sr_scale", type=int, default=None, help="sr: launch の sr.upsample.scale（512 入力をこの倍率で補間）")
    p.add_argument("--sr_interp", type=str, default=None, help="sr: launch の sr.upsample.interp")
    p.add_argument("--sr_input_size", type=int, default=None, help="sr: launch の sr.upsample.input_size（この一辺の入力を補間する）")
    req("--diff_range_hu", type=int, help="R のカラー表示の ±範囲 HU（現在の configs/train.yaml の log.diff_range_hu。学習時の TensorBoard と同じ）")
    req("--display_hu_min", type=int, help="表示用の線形範囲の下限 HU（現在の configs/train.yaml の log.display_hu_min）。--save_panel のパネルにだけ使う")
    req("--display_hu_max", type=int, help="表示用の線形範囲の上限 HU（現在の configs/train.yaml の log.display_hu_max）。--save_panel のパネルにだけ使う")
    req("--preview_bits", type=int, choices=(8, 16), help="パネルのビット深度（現在の configs/train.yaml の log.preview_bits）。--save_panel のパネルにだけ使う")
    # 方式（configs/infer.yaml）
    req("--mode", choices=MODES)
    req("--patch_size", type=int)
    req("--patch_stride", type=int)
    req("--patch_blend", choices=BLENDS)
    req("--patch_batch_size", type=int)
    req("--max_slices", type=int)
    p.add_argument("--save_residual", action="store_true")
    p.add_argument("--save_panel", action="store_true")
    req("--batch_slices", type=int, help="同時に G に通すスライス数（full はそのまま、patch は全スライスの patch を束ねて patch_batch_size ずつ）")
    # 乱数画像 n（configs/infer.yaml の noise.*。run が noise_input でなければ n_draws 1 / sigma_mode random 以外はエラー）
    req("--noise_sigma_mode", choices=SIGMA_MODES, help="random（スライス名の seed から log-uniform）| fixed（全スライス --noise_sigma_hu_fixed）")
    req("--noise_sigma_hu_fixed", type=float, help="sigma_mode fixed のときの sigma [HU]")
    req("--noise_n_draws", type=int, help="1 = 通常。k ≥ 2 で k 通りの n の平均を出力し、標準偏差を <mode>_std/ と std_stats.txt に")
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
def add_generator_args(p):
    """G の構成のフラグ（run の launch.yaml から run_infer.generator_argv が組む）。inference_dir.py と crop_patches.py で共通。
    noise_input（2026-10-12）のときは --sigma_hu_min / max（launch の train の noise.*）が必須（build_generator で検査）。"""
    req = lambda *a, **k: p.add_argument(*a, required=True, **k)  # noqa: E731
    req("--netG")
    req("--ngf", type=int)
    req("--input_nc", type=int)
    req("--output_nc", type=int)
    req("--norm")
    p.add_argument("--final_norm_act", action="store_true")
    p.add_argument("--residual", action="store_true")
    p.add_argument("--noise_input", action="store_true", help="run が乱数画像 n あり（G = F(cat[z, n])、networks.NoiseInputGenerator。F の 1 層目は input_nc + 1 ch）")
    p.add_argument("--sigma_hu_min", type=float, default=None, help="noise_input: 学習時の sigma の下限 [HU]（launch の train.noise.sigma_hu_min。sigma_mode random の範囲）")
    p.add_argument("--sigma_hu_max", type=float, default=None, help="noise_input: 学習時の sigma の上限 [HU]")
    req("--hu_offset", type=int)
    req("--hu_min", type=int)
    req("--hu_max", type=int)


def build_generator(a, device):
    """networks.define_G で学習時と同じ G を組み、<weight_dir>/net_G.pth を strict に読む（noise_input なら F の 1 層目が input_nc + 1 ch なので、n の無い重みは明示的に落ちる）。"""
    if a.noise_input and (a.sigma_hu_min is None or a.sigma_hu_max is None):
        raise ValueError("--noise_input には --sigma_hu_min / --sigma_hu_max が必要です（run_infer.py が launch の train の noise.* から渡す）")
    net = networks.define_G(a.input_nc, a.output_nc, a.ngf, a.netG, a.norm, False, "normal", 0.02, final_norm_act=a.final_norm_act, residual=a.residual,
                            noise_input=a.noise_input, sigma_hu=(a.sigma_hu_min, a.sigma_hu_max), hu=(a.hu_min, a.hu_max))
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
def infer_full(net, x, n=None):
    """x: (K,1,H,W)。H, W が 2^LEVELS の倍数でなければエラー。n: noise_input のときの乱数画像 (K,1,H,W)（None なら G が毎回引く = 再現しない）。"""
    h, w = x.shape[-2:]
    m = 2**LEVELS
    if h % m or w % m:
        raise ValueError(f"full モードは H, W が {m} の倍数である必要があります: {h}×{w}")
    return net(x) if n is None else net(x, n=n)


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


def infer_patch(net, x, patch, stride, window, batch_size, n=None):
    """x: (K,1,H,W)。K 枚のスライスの patch を全部まとめて切り出し、batch_size 枚ずつ G に通し、窓で重み付き平均して (K,1,H,W) を返す。
    n（noise_input、2026-10-12）: x と同じ形 (K,1,H,W) のスライス全体の乱数画像。各 patch は x と同じ index で n の対応領域を切り出して G に渡す（full と同じ n。patch ごとに引き直さない）。
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
    if n is not None and n.shape != x.shape:
        raise ValueError(f"n の形 {tuple(n.shape)} が x {tuple(x.shape)} と合いません")
    nflat = n.reshape(-1) if n is not None else None
    wflat = window.reshape(-1).to(x.dtype)
    out = torch.zeros(k * h * w, dtype=x.dtype, device=dev)
    wsum = torch.zeros(k * h * w, dtype=x.dtype, device=dev)
    for s in range(0, base.shape[0], batch_size):
        idx = base[s : s + batch_size, None] + offs[None, :]  # (b, p*p)
        xin = flat[idx].view(-1, 1, patch, patch)  # (b,1,p,p)。gather 1 回で切り出し
        o = net(xin) if nflat is None else net(xin, n=nflat[idx].view(-1, 1, patch, patch))  # n も同じ index で切り出す
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
    nm = task_names(a.task)
    sr_up = None
    if a.task == "sr":
        if any(getattr(a, k) is None for k in ("sr_scale", "sr_interp", "sr_input_size")):
            raise ValueError("--task sr には --sr_scale / --sr_interp / --sr_input_size が必要です（run_infer.py が launch の sr.upsample から渡す）")
        if write_dcm:
            raise ValueError("task: sr の DICOM 出力は未対応です（参照 DICOM が 512 で出力が 1024。--output_format png）")
        sr_up = (a.sr_scale, a.sr_interp, a.sr_input_size)
    n_upsampled = [0]
    dicom_index = DicomIndex(a.dicom_dir) if write_dcm else None  # 無ければここで止まる
    input_dir = Path(a.input_dir)
    out_dir = Path(a.out_dir)
    hu = (a.hu_offset, a.hu_min, a.hu_max)
    hu_per_unit = (a.hu_max - a.hu_min) / 2.0  # 正規化 [−1,1] の 1.0 が何 HU か

    net, weight_path = build_generator(a, device)
    nw = noise_wrapper(net)  # noise_input の run だけ非 None
    n_draws = a.noise_n_draws
    if n_draws < 1:
        raise ValueError(f"--noise_n_draws ≥ 1: {n_draws}")
    if nw is None and (n_draws != 1 or a.noise_sigma_mode != "random"):
        raise ValueError(f"この run は noise_input: false（n なし）なので --noise_n_draws 1 / --noise_sigma_mode random 以外は指定できません: n_draws={n_draws}, sigma_mode={a.noise_sigma_mode}")
    noise_records = {}  # rel → {"seed", "sigma_hu"}（noise.yaml）
    std_lines = []      # (mode, rel, sigma_hu, mean_std_hu, max_std_hu)（std_stats.txt。n_draws ≥ 2）
    modes = ["full", "patch"] if a.mode == "both" else [a.mode]
    window = blend_window(a.patch_size, a.patch_blend, device) if "patch" in modes else None
    if "patch" in modes and a.patch_size % (2**LEVELS):
        raise ValueError(f"patch_size は {2**LEVELS} の倍数である必要があります: {a.patch_size}")

    # --save_panel: 表示用パネル [G(z) | A | R + ゲージ]（degrade: EID-like | PCD | R、sr: PCD-like | EID | R。crop_patches.py・util/monitor.py と同じ正規化表示）
    panel_labels = [f"{nm['GEN']}  G(z)   {Path(a.weight_dir).name}", f"{nm['A']}  z", "R = G(z) - z   [HU]"]
    panel_gray01 = lambda stored: np.repeat(np.clip((stored.astype(np.float32) - a.hu_offset - a.display_hu_min) / float(a.display_hu_max - a.display_hu_min), 0.0, 1.0)[..., None], 3, axis=2)  # noqa: E731
    quant_panel = (  # noqa: E731
        (lambda x01: (np.clip(x01, 0, 1) * 65535.0).round().astype(np.uint16)) if a.preview_bits == 16
        else (lambda x01: (np.clip(x01, 0, 1) * 255.0).round().astype(np.uint8))
    )

    idx = CaseIndex(input_dir, a.index_cache_dir)
    paths = idx.all_paths if a.max_slices == 0 else idx.all_paths[: a.max_slices]
    n_patches = None
    print(f"[infer] device={device} G={weight_path}" + (f" | n: sigma_mode={a.noise_sigma_mode}" + (f" sigma_hu_fixed={a.noise_sigma_hu_fixed}" if a.noise_sigma_mode == "fixed" else f" sigma_hu [{a.sigma_hu_min}, {a.sigma_hu_max}]") + f" n_draws={n_draws}（seed = crc32(スライス名)）" if nw is not None else " | n: なし（noise_input false）"))
    print(f"[infer] task={a.task} input={input_dir} ({idx.n_cases} cases, {idx.n_slices} slices, 推論 {len(paths)} 枚) mode={a.mode} output={a.output_format} -> {out_dir}"
          + (f" | sr: 一辺 {sr_up[2]} の入力は ×{sr_up[0]} {sr_up[1]} で補間、{sr_up[2] * sr_up[0]} はそのまま" if sr_up is not None else ""))
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

    if (a.save_residual or a.save_panel) and a.diff_range_hu <= 0:
        raise ValueError(f"--diff_range_hu は正: {a.diff_range_hu}")
    if a.save_panel and a.display_hu_min >= a.display_hu_max:
        raise ValueError(f"--display_hu_min < --display_hu_max にしてください: {a.display_hu_min} >= {a.display_hu_max}")
    if a.save_residual:  # R のカラー表示の凡例（<out_dir>/ 直下に 1 枚。範囲を名前に入れる）
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
        if sr_up is not None:  # sr: 512（input_size）なら学習入力と同じ方式で補間、1024（input_size × scale）ならそのまま。他はエラー
            scale, interp, in_size = sr_up
            if stored.shape == (in_size, in_size):
                stored = upsample(stored, scale, interp)
                n_upsampled[0] += 1
                return stored, normalize(stored, *hu), True
            if stored.shape != (in_size * scale, in_size * scale):
                raise ValueError(f"task sr の入力は一辺 {in_size}（補間する）か {in_size * scale}（そのまま）: {stored.shape} {p}")
        return stored, normalize(stored, *hu), False

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
                stored, x_np, up = load(p)
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
                stored, x_np, up = fut.result()
                tm["read"] += time.time() - t
            items.append((p, stored, x_np, up))
        return items

    diff_lines = []
    t0 = time.time()
    bar = tqdm(total=len(paths), unit="slice", dynamic_ncols=True)
    try:
        while True:
            items = take(batch_n)
            if not items:
                break
            xs = [torch.from_numpy(x_np).unsqueeze(0).unsqueeze(0) for _, _, x_np, _ in items]
            same_shape = all(x.shape == xs[0].shape for x in xs)
            # 乱数画像 n（noise_input のときだけ）: スライスごとに H×W 全体の n を draw 0..n_draws−1 分作る（seed = crc32(スライス名)、draw k は seed + k。sigma は draw によらず同じ）。full / patch で同じ n を使う
            noises = None  # [draw][item] → (1,1,H,W)
            if nw is not None:
                noises = [[None] * len(items) for _ in range(n_draws)]
                for i, (p, _, x_np, _) in enumerate(items):
                    rel_i = str(Path(p).relative_to(input_dir))
                    for k in range(n_draws):
                        n_k, seed, sigma_hu = nw.slice_noise(Path(p).stem, x_np.shape, device, a.noise_sigma_mode, a.noise_sigma_hu_fixed, draw=k)
                        noises[k][i] = n_k
                    noise_records[rel_i] = {"seed": int(seed), "sigma_hu": float(sigma_hu)}

            def forward_mode(m, k):
                """モード m（full | patch）で items を draw k の n で通し、item ごとの (1,1,H,W) のリストを返す。形が同じときは束ねて 1 回。"""
                ns = noises[k] if noises is not None else None
                if m == "full":
                    if same_shape:
                        return list(infer_full(net, torch.cat(xs).to(device), None if ns is None else torch.cat(ns)).split(1))  # まとめて 1 回（形が同じときだけ）
                    return [infer_full(net, x.to(device), None if ns is None else ns[i]) for i, x in enumerate(xs)]
                if same_shape:
                    return list(infer_patch(net, torch.cat(xs).to(device), a.patch_size, a.patch_stride, window, a.patch_batch_size, None if ns is None else torch.cat(ns)).split(1))  # K 枚の patch を束ねて
                return [infer_patch(net, x.to(device), a.patch_size, a.patch_stride, window, a.patch_batch_size, None if ns is None else ns[i]) for i, x in enumerate(xs)]

            outs_by_mode, stds_by_mode = {}, {}  # mode → [item] → (1,1,H,W)。std は n_draws ≥ 2 のときだけ（HU）
            for m in modes:
                if m == "patch" and n_patches is None:
                    n_patches = len(grid_positions(xs[0].shape[-2], a.patch_size, a.patch_stride)) * len(grid_positions(xs[0].shape[-1], a.patch_size, a.patch_stride))
                    tqdm.write(f"[infer] patch: {a.patch_size}px stride {a.patch_stride} → {n_patches} patch/枚, blend={a.patch_blend}, patch_batch={a.patch_batch_size}, slices/forward={a.batch_slices}")
                t = time.time()
                with torch.inference_mode():
                    draws = [forward_mode(m, k) for k in range(n_draws)]  # [draw][item]
                    if n_draws == 1:
                        outs_by_mode[m] = draws[0]
                    else:  # n の効きの確認: draw 間の平均を出力、画素ごとの標準偏差（母標準偏差、HU）を std に
                        stacks = [torch.cat([draws[k][i] for k in range(n_draws)]) for i in range(len(items))]  # item → (n_draws,1,H,W)
                        outs_by_mode[m] = [st.mean(0, keepdim=True) for st in stacks]
                        stds_by_mode[m] = [st.std(0, unbiased=False, keepdim=True) * hu_per_unit for st in stacks]
                tm[m] += time.time() - t
            for k, (p, stored, x_np, up) in enumerate(items):
                rel = Path(p).relative_to(input_dir)
                outs = {m: outs_by_mode[m][k] for m in modes}
                pcd16 = denormalize(x_np, *hu)  # 正規化窓でクリップした入力（R の基準）
                ref_ds = None
                if write_dcm:  # F-18a: このスライスの参照 DICOM が本当に入力 PNG の元か、画素で照合（違えばここで止まる）
                    ref_ds = check_pixels(dicom_index.reference_for(rel.stem), stored, a.hu_offset, str(rel))
                for m, y in outs.items():
                    eid16 = denormalize(y[0, 0].cpu().numpy(), *hu)
                    if write_png_out:
                        put(out_dir / m / rel, eid16)
                        if up:  # sr: 補間した 1024 入力も残す（学習入力 EID1024 と同じ画像。R の基準）
                            put(out_dir / f"{m}_input" / rel, stored)
                    if write_dcm:
                        key = (m, rel.parent)
                        if key not in series_uid:
                            series_uid[key] = generate_uid()
                        write_like_reference(dicom_index.reference_for(rel.stem), eid16.astype(np.float64) - a.hu_offset,
                                             out_dir / f"{m}_dicom" / rel.with_suffix(".dcm"), series_uid[key],
                                             f"SpicaV5 {nm['GEN']} {run_name}/{Path(a.weight_dir).name} {m}", a.rescale_slope, a.rescale_intercept, ds=ref_ds)
                    if m in stds_by_mode:  # n_draws ≥ 2: 画素ごとの標準偏差 [HU] を uint16 で（stored = round(std_HU)。オフセット無し）
                        std_hu = stds_by_mode[m][k][0, 0].cpu().numpy()
                        put(out_dir / f"{m}_std" / rel, np.clip(np.round(std_hu), 0, 65535).astype(np.uint16))
                        std_lines.append((m, str(rel), noise_records[str(rel)]["sigma_hu"], float(std_hu.mean()), float(std_hu.max())))
                    if a.save_residual or a.save_panel:
                        delta_hu = eid16.astype(np.int32) - pcd16.astype(np.int32)  # = R_stored − 32768（保存した 16bit と同じ差）
                    if a.save_residual:
                        put(out_dir / f"{m}_R" / rel, residual_stored(pcd16, eid16))
                        put(out_dir / f"{m}_R_color" / rel, rgb_to_bgr(residual_rgb8(delta_hu, a.diff_range_hu)))
                    if a.save_panel:
                        cols = [panel_gray01(eid16), panel_gray01(pcd16), residual_rgb01(delta_hu, a.diff_range_hu)]
                        panel01 = build_panel(cols, panel_labels, a.diff_range_hu)
                        put(out_dir / f"{m}_panel" / rel, rgb_to_bgr(quant_panel(panel01)))
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
    # 乱数画像 n の記録（常に書く。n の無い run は noise_input: false だけ）
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "noise.yaml", "w", encoding="utf-8") as f:
        rec = {"noise_input": nw is not None}
        if nw is not None:
            rec.update({"sigma_hu_min": float(a.sigma_hu_min), "sigma_hu_max": float(a.sigma_hu_max), "sigma_mode": a.noise_sigma_mode,
                        "sigma_hu_fixed": float(a.noise_sigma_hu_fixed) if a.noise_sigma_mode == "fixed" else None, "n_draws": n_draws,
                        "seed_rule": "zlib.crc32(slice stem); draw k uses seed + k (sigma_hu is that of draw 0); n = randn(1,1,H,W) * sigma_hu / ((hu_max - hu_min) / 2) on a CPU torch.Generator; full and patch share the same n per slice",
                        "slices": noise_records})
        yaml.safe_dump(rec, f, allow_unicode=True, sort_keys=False)
    if std_lines:
        with open(out_dir / "std_stats.txt", "w", encoding="utf-8") as f:
            f.write(f"# n の効き: n_draws={n_draws} 通りの n（seed + draw）の出力の画素ごとの標準偏差 [HU]（母標準偏差。<mode>_std/ の stored = round(std_HU)）  sigma_mode={a.noise_sigma_mode}  G={weight_path}\n")
            f.write("# mode\tslice\tsigma_hu\tmean_std\tmax_std\n")
            for m, rel, sg, mean_sd, max_sd in std_lines:
                f.write(f"{m}\t{rel}\t{sg:.2f}\t{mean_sd:.3f}\t{max_sd:.3f}\n")
            for m in modes:
                rows = [x for x in std_lines if x[0] == m]
                if rows:
                    f.write(f"# ALL {m}\tmean_of_means={np.mean([r[3] for r in rows]):.3f}\tmax={max(r[4] for r in rows):.3f}\n")
        for m in modes:
            rows = [x for x in std_lines if x[0] == m]
            if rows:
                print(f"[infer] std({m}, n_draws={n_draws}): mean {np.mean([r[3] for r in rows]):.3f} HU, max {max(r[4] for r in rows):.3f} HU -> {out_dir / 'std_stats.txt'}")
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
    if sr_up is not None:
        print(f"[infer] sr: 補間した入力 {n_upsampled[0]} 枚（<mode>_input/ に保存）、そのまま {len(paths) - n_upsampled[0]} 枚")
    print(f"[infer] done: {len(paths)} slices, {elapsed:.1f}s ({elapsed / max(len(paths), 1):.2f} s/slice) -> {out_dir}")


if __name__ == "__main__":
    main()
