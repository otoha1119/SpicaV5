"""ablation/CycleGAN/util/run_paths.py — この ablation の run ディレクトリのレイアウトの唯一の正。ablation/CycleGAN/util/run_paths.py から 2026-10-27 に複製（同解像度に合わせて説明を変更）。

run 名は起動時刻（JST）を yyyy_mmdd_HHMM にしたもの。checkpoint は「重みディレクトリ」単位（G_A / G_B / D_A / D_B と state の 5 ファイルが 1 組）。

<ablation_checkpoints_dir>/CycleGAN/<run>/          （machines.yaml の ablation_checkpoints_dir の下に ablation 名）
  launch.yaml                 解決済み設定（実効値 = yaml + sh の上書き、+ upsample = eid1024_dir/manifest.yaml の方式）。resume 時は launch_resume_<日時秒>.yaml が追加される（記録用。再開の基準は checkpoint の state.pth の config）
  dataset_info.yaml           train.py が起動時に書く: 症例分割の実効リスト、A / B の画像サイズ・patch・枚数（推論が入力の一辺を確認するのに使う）
  loss_log.txt
  latest/ best/               重みディレクトリ: net_G_A.pth（G: A → B、推論に使う）, net_G_B.pth（F: B → A）, net_D_A.pth（D_Y）, net_D_B.pth（D_X）, state.pth（optimizer ×2 / RNG / 進捗 / 実効設定）
  best.txt                    bash start_ablation.sh <マシン> CycleGAN best <run> <epoch> で weights/epoch_NNN/ をコピーしたときの記録（判定指標は無いので目視で手動）
  weights/epoch_NNN/          save_epoch_freq ごとの checkpoint（保存は <dir>.tmp → rename で原子的）
  output_images/epoch_NNN/<slice>_eid.png（1024 生）, _pcdlike.png（1024 生）, _R.png（G_A(z) − z、0 HU = 32768）   固定 + ランダム
  output_images/preview_fixed_<slice>/  00_eid_<slice>.png, 01_pcd_<pcd_slice>.png（代表。初回だけ）, epoch_NNN_pcdlike.png, epoch_NNN_R_color.png, epoch_NNN_panel.png（[PCD | PCD-like | EID1024 | R + ゲージ]）
  output_images/preview_random/epoch_NNN_<slice>.png   ランダムスライスの同じパネル
  tb/                         TensorBoard（ablation_tb_port）
  tensorboard.log

推論の出力は <repo>/output/ablation/CycleGAN/<run>_<重みディレクトリ名>_<入力フォルダ名>_<実行時刻>/（.gitignore 済みの output/ の下。コンテナでは /workspace/output/...）
    full/<case>/<slice>.png            PCD-like1024（uint16、stored = HU + 1400）
    full_input/<case>/<slice>.png      512 入力を補間した 1024（入力が 512 のときだけ）
    full_R/, full_R_color/             R = G_A(z) − z（save_residual）
    full_panel/                        [PCD-like1024 | EID1024 | R + ゲージ]（save_panel）
    R_colorbar_pm<range>HU.png, infer.yaml
"""

import re
from pathlib import Path

ABLATION = "CycleGAN"                           # ablation 名 = このディレクトリ名。checkpoints と output の下のサブディレクトリ名
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
    """<ablation_checkpoints_dir>/CycleGAN/"""
    return Path(machine["ablation_checkpoints_dir"]) / ABLATION


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
    """リポジトリ直下（このファイルは <repo>/ablation/CycleGAN/util/run_paths.py）。コンテナでは /workspace。"""
    return Path(__file__).resolve().parents[3]


def output_root(root=None):
    """<repo>/output/ablation/CycleGAN/。scratch 実行のときだけ root で差し替える。"""
    return (Path(root) if root else repo_root() / OUTPUT_DIRNAME) / "ablation" / ABLATION


def infer_output_dir(run_dir, weight_dir, input_dir, now, root=None):
    return output_root(root) / f"{Path(run_dir).name}_{Path(weight_dir).name}_{Path(input_dir).name}_{now.strftime(INFER_STAMP_FORMAT)}"
