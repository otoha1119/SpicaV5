"""stage2/util/run_paths.py — Stage 2 の run ディレクトリ（<stage2_checkpoints_dir>/<run>/）のレイアウトの唯一の正。
stage1/util/run_paths.py（2026-09-07 第 2 版）から 2026-09-08 に複製し、Stage 2 に合わせた（D が無いので checkpoint は 2 ファイル、画像出力は未実装）。

run 名は起動時刻（JST）を yyyy_mmdd_HHMM にしたもの（例 2026_0908_2130）。'/' は使わない。
checkpoint は「重みディレクトリ」単位で扱う（net_G.pth と state.pth の 2 ファイルが 1 組）。

<stage2_checkpoints_dir>/<run>/
  launch.yaml                 解決済み設定（実効値 = yaml + sh の上書き）。resume 時は launch_resume_<日時秒>.yaml が追加され、最新のものが次の再開の基準
  dataset_info.yaml           train.py が起動時に書く: 症例分割の実効リスト、症例ごとの入力 / 教師 / ペア枚数（切り捨て）
  loss_log.txt
  latest/net_G.pth, state.pth            直下に置く重みディレクトリは latest と best だけ
  best/net_G.pth,   state.pth            （best の指定は別フェーズ）
  weights/epoch_NNN/net_G.pth, state.pth save_epoch_freq ごとの checkpoint（保存は <dir>.tmp → rename で原子的。.tmp / .old が残っていたら中断の痕跡）
  tb/                         TensorBoard（scalar のみ。画像・preview は別フェーズ）
  tensorboard.log
"""

import re
from pathlib import Path

RUN_NAME_FORMAT = "%Y_%m%d_%H%M"                 # datetime.strftime 用
RUN_NAME_RE = re.compile(r"^\d{4}_\d{4}_\d{4}$")  # 2026_0908_2130
TOP_TAGS = ("latest", "best")                    # run 直下に置く重みディレクトリ。それ以外（epoch 番号、iter_N）は weights/
WEIGHTS_DIR = "weights"
LAUNCH_FILE = "launch.yaml"
DATASET_INFO_FILE = "dataset_info.yaml"
LOSS_LOG_FILE = "loss_log.txt"
TB_DIR = "tb"
CKPT_KINDS = ("net_G.pth", "state.pth")          # 重みディレクトリの中身


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
    """重みディレクトリの中のファイル。kind は 'net_G.pth' | 'state.pth'。"""
    return ckpt_dir(run_dir, tag) / kind


def saved_tags(run_dir):
    """その run にある checkpoint の tag 一覧（net_G.pth がある重みディレクトリから）。"""
    run_dir = Path(run_dir)
    tags = [t for t in TOP_TAGS if (run_dir / t / "net_G.pth").is_file()]
    epochs = []
    for d in (run_dir / WEIGHTS_DIR).glob("*"):
        if d.name.endswith((".tmp", ".old")):
            continue
        if (d / "net_G.pth").is_file():
            m = re.fullmatch(r"epoch_(\d+)", d.name)
            epochs.append(str(int(m.group(1))) if m else d.name)
    return tags + sorted(epochs, key=lambda t: (not t.isdigit(), int(t) if t.isdigit() else t))


def latest_launch(run_dir):
    """run の実効設定ファイル。再開のたびに launch_resume_<日時>.yaml が増えるので、あれば最新（名前順の末尾）、無ければ launch.yaml。"""
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
