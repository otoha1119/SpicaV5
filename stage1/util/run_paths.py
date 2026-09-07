"""util/run_paths.py — run ディレクトリ（<checkpoints_dir>/<run>/）のレイアウトの唯一の正。[SpicaV5]

run 名は起動時刻（JST）を yyyy_mmdd_HHMM にしたもの（例 2026_0907_1742）。'/' は使わない（引数や TensorBoard で階層に化けるため）。
checkpoint は「重みディレクトリ」単位で扱う（G と D と optimizer state の 3 ファイルが 1 組なので）。

<checkpoints_dir>/<run>/
  launch.yaml                 解決済み設定（resume 時は launch_resume_<日時>.yaml が追加）
  train_opt.txt, loss_log.txt
  latest/net_G.pth, net_D.pth, state.pth        直下に置く重みディレクトリは latest と best だけ
  best/net_G.pth,   net_D.pth, state.pth        bash start.sh best <run> <epoch> で weights/epoch_NNN/ をコピー
  best.txt                    best がどの epoch か（判定基準がまだ無いので当面は目視で手動指定）
  weights/epoch_NNN/net_G.pth, net_D.pth, state.pth   save_epoch_freq ごとの checkpoint
  output_images/samples/<total_iters>_{current,fixed}.png    学習中の 128 patch グリッド（8bit、TensorBoard と同じ表示用）
  output_images/epoch_NNN/<slice>_{pcd,eidlike,R}.png        checkpoint ごとのフル 512（uint16、入力と同じ規約）
  infer/<重みディレクトリ名>/<入力フォルダ名>/   bash start.sh infer の出力（inference_dir.py。uint16 PNG）
  tb/                         TensorBoard
"""

import re
from pathlib import Path

RUN_NAME_FORMAT = "%Y_%m%d_%H%M"                 # datetime.strftime 用
RUN_NAME_RE = re.compile(r"^\d{4}_\d{4}_\d{4}$")  # 2026_0907_1742
TOP_TAGS = ("latest", "best")                    # run 直下に置く重みディレクトリ。それ以外（epoch 番号、iter_N）は weights/
WEIGHTS_DIR = "weights"
OUTPUT_IMAGES_DIR = "output_images"
SAMPLES_DIR = "samples"
INFER_DIR = "infer"
BEST_NOTE = "best.txt"
LAUNCH_FILE = "launch.yaml"
CKPT_KINDS = ("net_G.pth", "net_D.pth", "state.pth")  # 重みディレクトリの中身


def run_name(now):
    """起動時刻 → run 名。"""
    return now.strftime(RUN_NAME_FORMAT)


def epoch_dirname(epoch):
    return f"epoch_{int(epoch):03d}"


def ckpt_dir(run_dir, tag):
    """tag（'latest' | 'best' | epoch 番号 | 'iter_N'）→ 重みディレクトリ。
    latest / best は <run>/<tag>/、epoch 番号は <run>/weights/epoch_NNN/、それ以外の文字列は <run>/weights/<tag>/。"""
    tag = str(tag)
    if tag in TOP_TAGS:
        return Path(run_dir) / tag
    if tag.isdigit():
        return Path(run_dir) / WEIGHTS_DIR / epoch_dirname(tag)
    return Path(run_dir) / WEIGHTS_DIR / tag


def ckpt_path(run_dir, tag, kind):
    """重みディレクトリの中のファイル。kind は 'net_G.pth' | 'net_D.pth' | 'state.pth'。"""
    return ckpt_dir(run_dir, tag) / kind


def saved_tags(run_dir):
    """その run にある checkpoint の tag 一覧（net_G.pth がある重みディレクトリから）。"""
    run_dir = Path(run_dir)
    tags = [t for t in TOP_TAGS if (run_dir / t / "net_G.pth").is_file()]
    epochs = []
    for d in (run_dir / WEIGHTS_DIR).glob("*"):
        if (d / "net_G.pth").is_file():
            m = re.fullmatch(r"epoch_(\d+)", d.name)
            epochs.append(str(int(m.group(1))) if m else d.name)
    return tags + sorted(epochs, key=lambda t: (not t.isdigit(), int(t) if t.isdigit() else t))


def find_run_dir(weight_dir):
    """重みディレクトリから run ディレクトリ（launch.yaml がある所）を探す。<run>/best → 1 つ上、<run>/weights/epoch_NNN → 2 つ上。
    見つからなければ None。"""
    p = Path(weight_dir).resolve()
    for _ in range(3):
        p = p.parent
        if (p / LAUNCH_FILE).is_file():
            return p
    return None


def samples_dir(run_dir):
    return Path(run_dir) / OUTPUT_IMAGES_DIR / SAMPLES_DIR


def epoch_images_dir(run_dir, epoch):
    return Path(run_dir) / OUTPUT_IMAGES_DIR / epoch_dirname(epoch)


def infer_dir(run_dir, weight_dir, input_dir):
    """推論の出力先: <run>/infer/<重みディレクトリ名>/<入力フォルダ名>/（同じ重みで別の入力を処理しても衝突しない）。"""
    return Path(run_dir) / INFER_DIR / Path(weight_dir).name / Path(input_dir).name
