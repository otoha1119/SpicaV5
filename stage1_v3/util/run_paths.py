"""stage1_v3/util/run_paths.py — 射影型 Stage 1 の run レイアウト（stage1_v2 と同じ規則、ネットは G / D の 2 つ）。

<repo>/stage1_v3/checkpoints/<run>/
  launch.yaml                 解決済み設定（train / machine / degrade の範囲）。resume 時は launch_resume_<日時秒>.yaml を追加
  dataset_info.yaml           PCD / EID の画像サイズ、症例・スライス数、sampling、P の設定
  loss_log.txt
  latest/ best/               重みディレクトリ: net_G.pth（G = in + R、推論に使う）, net_D.pth, state.pth（optimizer ×2 / RNG / 進捗 / 実効設定）, net_P.pth（degrade.mode learned のときだけ。推論の P にも使う）,
                              net_Dn.pth（pipeline.mode v4 のときだけ。凍結 denoiser の複製。推論の Dn にも使う）
  net_Dn.pth                  v4: pipeline.dn_weights の複製（run の自己完結用。resume / infer はこちらを優先）
  best.txt
  weights/epoch_NNN/          save_epoch_freq ごとの checkpoint
  output_images/epoch_NNN/<slice>_pcd.png, _u.png（P(x)）, _eidlike.png, _R.png, <fixed>_std.png（n_draws の標準偏差 [HU]）, <eid>_idt.png（G(y)）, _Ry.png
  output_images/preview_fixed_<slice>/ epoch_NNN_panel.png（[EID代表 | EID-like | P(x) | PCD | R]）、epoch_NNN_std.png
  output_images/preview_identity_<eid>/ epoch_NNN_panel.png（[EID y | G(y) | R(y)]）
  output_images/preview_random/epoch_NNN_<slice>.png
  tb/                         TensorBoard（stage1 と共通の tb_port）
  tensorboard.log

v4 の Dn（bash start1_v3.sh <machine> dn、train_denoiser.py）は <repo>/stage1_v3/checkpoints/denoiser/<run>/:
  launch.yaml, dataset_info.yaml, loss_log.txt, latest/net_Dn.pth（毎 epoch）, weights/epoch_NNN/net_Dn.pth（save_epoch_freq）, tb/, output_images/epoch_NNN/<slice>_panel.png（[x | Dn(x) | x − Dn(x)]）
  <repo>/stage1_v3/checkpoints/denoiser/latest/  最新の dn run の net_Dn.pth と launch.yaml の複製（train.yaml の pipeline.dn_weights の既定の参照先。symlink ではなく複製 = Windows バインドマウント対策）
dncheck（check_denoiser.py）の出力は <repo>/output/stage1_v3/dncheck_<時刻>/summary.yaml と <slice>_panel.png。

推論の出力は <repo>/output/stage1_v3/<run>_<重みディレクトリ名>_<入力フォルダ名>_<実行時刻>/。
    full/<case>/<slice>.png            出力（uint16、stored = HU + 1400）。apply_degrade なら draw 0 の P を通した出力
    full_u/                            P(x)（apply_degrade かつ save_residual）
    full_R/, full_R_color/             R = 出力 − G の入力（save_residual）
    full_std/                          n_draws 間の出力の標準偏差 [HU]（uint16。n_draws > 1）
    full_eps0/                         v4: ε = 0 の u（指紋。infer.eps_zero）。full_xc/ = Dn(x)（v4 かつ save_residual。v3 の full_u/ に相当）
    manifest.yaml                      重み（G / P / Dn）のパスと sha256
    full_panel/                        [出力 | P(x) | 入力 | R + ゲージ]（save_panel）
    R_colorbar_pm<range>HU.png, infer.yaml, draws_summary.yaml
P 単独生成（bash start1_v3.sh <machine> ponly）は <repo>/output/stage1_v3/ponly_<入力名>_<実行時刻>/full/ と manifest.yaml。
"""

import os
import re
from pathlib import Path

STAGE = "stage1_v3"
RUN_NAME_FORMAT = "%Y_%m%d_%H%M"
RUN_NAME_RE = re.compile(r"^\d{4}_\d{4}_\d{4}$")
TOP_TAGS = ("latest", "best")
WEIGHTS_DIR = "weights"
LAUNCH_FILE = "launch.yaml"
DATASET_INFO_FILE = "dataset_info.yaml"
LOSS_LOG_FILE = "loss_log.txt"
TB_DIR = "tb"
OUTPUT_IMAGES_DIR = "output_images"
NET_NAMES = ("G", "D")                 # 常にある。degrade.mode learned の run はさらに "P"（学習型劣化 P_θ、net_P.pth）、pipeline.mode v4 はさらに "Dn"（凍結 denoiser、net_Dn.pth）
OPTIONAL_NET_NAMES = ("P", "Dn")
DN_WEIGHT = "net_Dn.pth"
DENOISER_DIR = "denoiser"
CKPT_KINDS = tuple(f"net_{n}.pth" for n in NET_NAMES) + ("state.pth",)   # 必須
OPTIONAL_CKPT_KINDS = tuple(f"net_{n}.pth" for n in OPTIONAL_NET_NAMES)
INFER_WEIGHT = "net_G.pth"
BEST_NOTE = "best.txt"


def checkpoints_root(machine):
    """<repo>/stage1_v3/checkpoints/。machine は API 互換のため受け取る。
    環境変数 SPICA_V3_CHECKPOINTS_ROOT があればそこ（リポジトリ外の scratch で煙試験をするためだけ。通常運用では設定しない）。"""
    del machine
    env = os.environ.get("SPICA_V3_CHECKPOINTS_ROOT")
    return Path(env) if env else repo_root() / STAGE / "checkpoints"


def run_name(now):
    return now.strftime(RUN_NAME_FORMAT)


def epoch_dirname(epoch):
    return f"epoch_{int(epoch):03d}"


def ckpt_dir(run_dir, tag):
    tag = str(tag)
    if tag in TOP_TAGS:
        return Path(run_dir) / tag
    if tag.isdigit():
        return Path(run_dir) / WEIGHTS_DIR / epoch_dirname(tag)
    return Path(run_dir) / WEIGHTS_DIR / tag


def ckpt_path(run_dir, tag, kind):
    return ckpt_dir(run_dir, tag) / kind


def saved_tags(run_dir):
    run_dir = Path(run_dir)
    tags = [t for t in TOP_TAGS if (run_dir / t / "state.pth").is_file()]
    epochs = []
    for d in (run_dir / WEIGHTS_DIR).glob("*"):
        if d.name.endswith((".tmp", ".old")):
            continue
        if (d / "state.pth").is_file():
            m = re.fullmatch(r"epoch_(\d+)", d.name)
            epochs.append(str(int(m.group(1))) if m else d.name)
    return tags + sorted(epochs, key=lambda t: (not t.isdigit(), int(t) if t.isdigit() else t))


def latest_launch(run_dir):
    run_dir = Path(run_dir)
    resumes = sorted(run_dir.glob("launch_resume_*.yaml"))
    return resumes[-1] if resumes else run_dir / LAUNCH_FILE


def find_run_dir(weight_dir):
    p = Path(weight_dir).resolve()
    for _ in range(3):
        p = p.parent
        if (p / LAUNCH_FILE).is_file():
            return p
    return None


def epoch_images_dir(run_dir, epoch):
    return Path(run_dir) / OUTPUT_IMAGES_DIR / epoch_dirname(epoch)


def preview_dir(run_dir, name):
    return Path(run_dir) / OUTPUT_IMAGES_DIR / f"preview_{name}"


OUTPUT_DIRNAME = "output"
INFER_STAMP_FORMAT = "%Y_%m%d_%H%M%S"


def repo_root():
    return Path(__file__).resolve().parents[2]


def output_root(root=None):
    return (Path(root) if root else repo_root() / OUTPUT_DIRNAME) / STAGE


def infer_output_dir(run_dir, weight_dir, input_dir, now, root=None):
    return output_root(root) / f"{Path(run_dir).name}_{Path(weight_dir).name}_{Path(input_dir).name}_{now.strftime(INFER_STAMP_FORMAT)}"


def ponly_output_dir(input_dir, now, root=None):
    return output_root(root) / f"ponly_{Path(input_dir).name}_{now.strftime(INFER_STAMP_FORMAT)}"


def stats_output_dir(now, root=None):
    return output_root(root) / f"stats_{now.strftime(INFER_STAMP_FORMAT)}"


def dncheck_output_dir(now, root=None):
    return output_root(root) / f"dncheck_{now.strftime(INFER_STAMP_FORMAT)}"


# ---------------------------------------------------------------------------
# v4 の Dn（train_denoiser.py）
# ---------------------------------------------------------------------------
def denoiser_root(machine):
    """<checkpoints_root>/denoiser/"""
    return checkpoints_root(machine) / DENOISER_DIR


def denoiser_run_dir(machine, name):
    return denoiser_root(machine) / name


def denoiser_latest_dir(machine):
    """最新の dn run の net_Dn.pth と launch.yaml の複製（train.yaml pipeline.dn_weights の既定の参照先）。"""
    return denoiser_root(machine) / "latest"
