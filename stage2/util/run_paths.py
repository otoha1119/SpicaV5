"""stage2/util/run_paths.py — 新 Stage 2 の run フォルダのレイアウトの唯一の正（設計書 H-2、RECONCILE #5）。

置き場: <flow_checkpoints_dir>/<種類>/<run>/ = /workspace/runs/stage2/u/<run>/（U）、/workspace/runs/stage2/v/<run>/（V）。run 名は起動時刻（JST）の yyyy_mmdd_HHMM。
重みは「重みディレクトリ」単位（保存は <dir>.tmp → 差し替えで原子的）。U: net_U.pth / net_U_ema.pth / state.pth。V: net_V.pth / net_V_ema.pth / net_U.pth（使った U の EMA の写し）/ state.pth。

<run>/
  launch.yaml, launch_resume_<日時秒>.yaml   実効設定（yaml + sh の上書き）。再開の基準は選んだ重みの state.pth の中の実効設定
  dataset_info.yaml                          疑似 EID の組と manifest、組の数、パラメータ数、実測の速度
  loss_log.txt
  latest/                                    log.save_latest_steps ごと + epoch 末 + 止まるとき（上限・SIGTERM）
  best/  + best.txt                          上限（max_steps / max_hours）で止まったときの最後の EMA（rule: last）。手で選び直すのは bash start2.sh best
  weights/step_NNNNNNN/                      log.save_weights_epochs epoch ごと（設計書は U を epoch_NNN と書いているが、U・V とも step 番号にそろえた）
  output_images/<印>/                        epoch 末の生の 16bit（stored = HU + 1400）。<印> = epoch_NNN（最初の確認は first_step_NNNNNNN）
  preview/<名前>/<印>.png                    epoch 末のパネル（表示用 8bit、等倍）
  check/<印>/magnifier.md                    epoch 末の虫眼鏡の数字。確認コマンド（bash start2.sh check）の出力は check/<重み>_<時刻>/
  tb/                                        TensorBoard（scalar と画像）
  tensorboard.log
"""

import datetime
import re
from pathlib import Path

RUN_NAME_FORMAT = "%Y_%m%d_%H%M"
KINDS = ("u", "v")
TOP_TAGS = ("latest", "best")
WEIGHTS_DIR = "weights"
LAUNCH_FILE = "launch.yaml"
DATASET_INFO_FILE = "dataset_info.yaml"
LOSS_LOG_FILE = "loss_log.txt"
TB_DIR = "tb"
OUTPUT_IMAGES_DIR = "output_images"
PREVIEW_DIR = "preview"
CHECK_DIR = "check"
CKPT_KINDS = {"u": ("net_U.pth", "net_U_ema.pth", "state.pth"), "v": ("net_V.pth", "net_V_ema.pth", "net_U.pth", "state.pth")}
BEST_NOTE = "best.txt"
JST = datetime.timezone(datetime.timedelta(hours=9), name="JST")


def run_name(now):
    return now.strftime(RUN_NAME_FORMAT)


def kind_root(machine, kind):
    if kind not in KINDS:
        raise ValueError(kind)
    return Path(machine["flow_checkpoints_dir"]) / kind


def step_dirname(step):
    return f"step_{int(step):07d}"


def ckpt_dir(run_dir, tag):
    """tag（'latest' | 'best' | step 番号）→ 重みディレクトリ。"""
    tag = str(tag)
    if tag in TOP_TAGS:
        return Path(run_dir) / tag
    if tag.isdigit():
        return Path(run_dir) / WEIGHTS_DIR / step_dirname(tag)
    return Path(run_dir) / WEIGHTS_DIR / tag


def missing_files(weight_dir, kind):
    return [k for k in CKPT_KINDS[kind] if not (Path(weight_dir) / k).is_file()]


def saved_tags(run_dir, kind):
    """その run にある重みの tag 一覧（ファイルが揃っているものだけ）。"""
    run_dir = Path(run_dir)
    tags = [t for t in TOP_TAGS if not missing_files(run_dir / t, kind)]
    steps = []
    for d in sorted((run_dir / WEIGHTS_DIR).glob("step_*")):
        m = re.fullmatch(r"step_(\d+)", d.name)
        if m and not missing_files(d, kind):
            steps.append(str(int(m.group(1))))
    return tags + steps


def images_dir(run_dir, mark):
    return Path(run_dir) / OUTPUT_IMAGES_DIR / mark


def preview_path(run_dir, name, mark):
    return Path(run_dir) / PREVIEW_DIR / name / f"{mark}.png"


def check_dir(run_dir, mark):
    return Path(run_dir) / CHECK_DIR / mark


def write_best_note(run_dir, step, rule, source, note=""):
    lines = [f"step: {int(step)}", f"rule: {rule}", f"source: {source}", f"note: {note}",
             f"written_at: {datetime.datetime.now(JST).isoformat(timespec='minutes')}"]
    with open(Path(run_dir) / BEST_NOTE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
