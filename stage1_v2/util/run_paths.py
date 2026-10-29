"""stage1_v2/util/run_paths.py — CycleGAN 版 Stage 1 の run レイアウト。

run 名は起動時刻（JST）を yyyy_mmdd_HHMM にしたもの。checkpoint は「重みディレクトリ」単位（G_A / G_B / D_A / D_B と state の 5 ファイルが 1 組）。

<repo>/stage1_v2/checkpoints/<run>/
  launch.yaml                 解決済み設定。resume 時は launch_resume_<日時秒>.yaml を追加
  dataset_info.yaml           PCD / EID の画像サイズ、症例・スライス数、sampling
  loss_log.txt
  latest/ best/               重みディレクトリ: net_G_A.pth（G: A → B、推論に使う）, net_G_B.pth（F: B → A）, net_D_A.pth（D_Y）, net_D_B.pth（D_X）, state.pth（optimizer ×2 / RNG / 進捗 / 実効設定）
  best.txt                    bash start1_v2.sh <マシン> best <run> <epoch> で選んだ checkpoint の記録
  weights/epoch_NNN/          save_epoch_freq ごとの checkpoint（保存は <dir>.tmp → rename で原子的）
  output_images/epoch_NNN/<slice>_pcd.png, _eidlike.png, _R.png
  output_images/preview_fixed_<slice>/ epoch_NNN_panel.png（[EID代表 | EID-like | PCD | R]）
  output_images/preview_random/epoch_NNN_<slice>.png   ランダムスライスの同じパネル
  tb/                         TensorBoard（Stage 1 と共通の tb_port）
  tensorboard.log

推論の出力は <repo>/output/stage1_v2/<run>_<重みディレクトリ名>_<入力フォルダ名>_<実行時刻>/。
    full/<case>/<slice>.png            EID-like512（uint16、stored = HU + 1400）
    full_R/, full_R_color/             R = G_A(z) − z（save_residual）
    full_panel/                        [EID-like512 | PCD512 | R + ゲージ]（save_panel）
    R_colorbar_pm<range>HU.png, infer.yaml
"""

import re
from pathlib import Path

STAGE = "stage1_v2"
RUN_NAME_FORMAT = "%Y_%m%d_%H%M"
RUN_NAME_RE = re.compile(r"^\d{4}_\d{4}_\d{4}$")
TOP_TAGS = ("latest", "best")
WEIGHTS_DIR = "weights"
LAUNCH_FILE = "launch.yaml"
DATASET_INFO_FILE = "dataset_info.yaml"
LOSS_LOG_FILE = "loss_log.txt"
TB_DIR = "tb"
OUTPUT_IMAGES_DIR = "output_images"
NET_NAMES = ("G_A", "G_B", "D_A", "D_B")          # 本家 cycle_gan_model と同じ（論文 G / F / D_Y / D_X）
CKPT_KINDS = tuple(f"net_{n}.pth" for n in NET_NAMES) + ("state.pth",)
INFER_WEIGHT = "net_G_A.pth"                      # 推論に使う G（A → B）
BEST_NOTE = "best.txt"


def checkpoints_root(machine):
    """<repo>/stage1_v2/checkpoints/。machine は API 互換のため受け取る。"""
    del machine
    return repo_root() / STAGE / "checkpoints"


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
    """その run にある checkpoint の tag 一覧（state.pth がある重みディレクトリから）。"""
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
    """重みディレクトリから run ディレクトリ（launch.yaml がある所）を探す。<run>/best → 1 つ上、<run>/weights/epoch_NNN → 2 つ上。見つからなければ None。"""
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


# --- 推論の出力 ---
OUTPUT_DIRNAME = "output"
INFER_STAMP_FORMAT = "%Y_%m%d_%H%M%S"


def repo_root():
    """リポジトリ直下（このファイルは <repo>/stage1_v2/util/run_paths.py）。"""
    return Path(__file__).resolve().parents[2]


def output_root(root=None):
    """<repo>/output/stage1_v2/。scratch 実行のときだけ root で差し替える。"""
    return (Path(root) if root else repo_root() / OUTPUT_DIRNAME) / STAGE


def infer_output_dir(run_dir, weight_dir, input_dir, now, root=None):
    return output_root(root) / f"{Path(run_dir).name}_{Path(weight_dir).name}_{Path(input_dir).name}_{now.strftime(INFER_STAMP_FORMAT)}"
