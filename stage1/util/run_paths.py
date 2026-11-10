"""stage1/util/run_paths.py — 新 Stage 1 の run フォルダ（<painter_checkpoints_dir>/<run>/ = /workspace/runs/stage1/<run>/）のレイアウトの唯一の正。

run 名は起動時刻（JST）の yyyy_mmdd_HHMM。重みは「重みディレクトリ」単位（net.pth / ema.pth / state.pth の 3 つ 1 組。保存は <dir>.tmp → 差し替えで原子的）。

<run>/
  launch.yaml, launch_resume_<日時秒>.yaml   実効設定（yaml + sh の上書き）。再開の基準は選んだ重みの state.pth の中の実効設定
  dataset_info.yaml                          症例の分割、枚数、測った平均・標準偏差、パラメータ数、実測の速度
  loss_log.txt
  latest/   net.pth, ema.pth, state.pth      log.save_latest_steps ごと + 上限で止まるとき + SIGTERM のとき
  best/     + best.txt                       手で付けたときだけ（bash start.sh best <run> <step>）
  weights/step_NNNNNNN/                      log.save_weights_steps ごと
  output_images/step_NNNNNNN/                途中経過の生の 16bit（stored = HU + 1400）
  preview/<名前>/step_NNNNNNN.png            途中経過のパネル（表示用 8bit）
  tb/                                        TensorBoard（scalar と画像）
  tensorboard.log
"""

import datetime
import re
from pathlib import Path

RUN_NAME_FORMAT = "%Y_%m%d_%H%M"
TOP_TAGS = ("latest", "best")
WEIGHTS_DIR = "weights"
LAUNCH_FILE = "launch.yaml"
DATASET_INFO_FILE = "dataset_info.yaml"
LOSS_LOG_FILE = "loss_log.txt"
TB_DIR = "tb"
OUTPUT_IMAGES_DIR = "output_images"
PREVIEW_DIR = "preview"
CKPT_KINDS = ("net.pth", "ema.pth", "state.pth")
BEST_NOTE = "best.txt"
JST = datetime.timezone(datetime.timedelta(hours=9), name="JST")


def run_name(now):
    return now.strftime(RUN_NAME_FORMAT)


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


def saved_tags(run_dir):
    """その run にある重みの tag 一覧（3 ファイルが揃っているものだけ）。"""
    run_dir = Path(run_dir)
    ok = lambda d: all((d / k).is_file() for k in CKPT_KINDS)
    tags = [t for t in TOP_TAGS if ok(run_dir / t)]
    steps = []
    for d in sorted((run_dir / WEIGHTS_DIR).glob("step_*")):
        m = re.fullmatch(r"step_(\d+)", d.name)
        if m and ok(d):
            steps.append(str(int(m.group(1))))
    return tags + steps


def step_images_dir(run_dir, step):
    return Path(run_dir) / OUTPUT_IMAGES_DIR / step_dirname(step)


def preview_path(run_dir, name, step):
    return Path(run_dir) / PREVIEW_DIR / name / f"{step_dirname(step)}.png"


def write_best_note(run_dir, step, source, note):
    lines = [f"step: {int(step)}", "how: manual", f"source: {source}", f"note: {note}",
             f"written_at: {datetime.datetime.now(JST).isoformat(timespec='minutes')}"]
    with open(Path(run_dir) / BEST_NOTE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
