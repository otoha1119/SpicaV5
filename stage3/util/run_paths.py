"""stage3/util/run_paths.py — Stage 3 の run ディレクトリ（<stage3_checkpoints_dir>/<run>/）のレイアウトの唯一の正。
stage2/util/run_paths.py から 2026-10-13 に複製し、拡散モデルに合わせた（checkpoint は model / ema / state の 3 ファイル、区切りは epoch ではなく step = 本家と同じ）。
Windows の rename 再試行 replace_retry は 003 ブランチ（2026-10-13）の stage2 と同じもの。

run 名は起動時刻（JST）を yyyy_mmdd_HHMM にしたもの（例 2026_1013_2130）。'/' は使わない。
checkpoint は「重みディレクトリ」単位（3 ファイル 1 組。本家は ckpt_{step}.pth の 1 ファイル = [model, optimizer, epoch, step, ema] のリスト）。

<stage3_checkpoints_dir>/<run>/
  launch.yaml                 解決済み設定（ddim / train / machine の実効値）。resume で launch_resume_<日時秒>.yaml が増える（記録用。再開の基準は checkpoint の state.pth）
  dataset_info.yaml           train.py が起動時に書く: 症例分割、ペア枚数、画像の大きさ、実 EID の補間方式（推論が読む）
  loss_log.txt
  latest/model.pth, ema.pth, state.pth      ddim.yml training.snapshot_freq ごと（本家と同じ間隔。本家は step 1 でも保存する → 同じ）
  best/  ...                                手動だけ: bash start3.sh best <run> <iter>（weights/iter_NNNNNNN/ をコピー、best.txt に記録）
  weights/iter_NNNNNNN/model.pth, ema.pth, state.pth   train.yaml log.save_weights_freq ごと（1 組 ≈ 1.8 GB）
  output_images/iter_NNNNNNN/                  log.full_freq ごと。生 16bit（stored = HU + 1400）
      <full_slice>_eidlike.png, _pcd1024.png, _pcdlike.png      固定スライスの入力 / 教師 / 出力（タイル合成のサンプル）
      <eid_slice>_eid1024.png, _pcdlike.png                    実 EID テスト（512 を補間した 1024 / 出力）
  output_images/preview_fixed_<slice>/         表示用（Stage 2 と同じ）: 00_ / 01_ / 02_（代表、初回だけ）、iter_NNNNNNN_pcdlike.png、iter_NNNNNNN_eid_pcdlike.png、iter_NNNNNNN_panel.png（5 列）
  tb/                         TensorBoard
  tensorboard.log

推論の出力は Stage 1 / 2 と同じ根 <repo>/output/<run>_<重みディレクトリ名>_<入力フォルダ名>_<実行時刻>/（inference_dir.py の説明を参照）。
"""

import re
from pathlib import Path

RUN_NAME_FORMAT = "%Y_%m%d_%H%M"
RUN_NAME_RE = re.compile(r"^\d{4}_\d{4}_\d{4}$")
TOP_TAGS = ("latest", "best")
WEIGHTS_DIR = "weights"
LAUNCH_FILE = "launch.yaml"
DATASET_INFO_FILE = "dataset_info.yaml"
LOSS_LOG_FILE = "loss_log.txt"
TB_DIR = "tb"
OUTPUT_IMAGES_DIR = "output_images"
CKPT_MODEL, CKPT_EMA, CKPT_STATE = "model.pth", "ema.pth", "state.pth"
CKPT_KINDS = (CKPT_MODEL, CKPT_EMA, CKPT_STATE)
BEST_NOTE = "best.txt"


def write_best_note(run_dir, step, how, source=None):
    """best.txt を書く（manual: mark_best.py）。上書き。"""
    import datetime
    jst = datetime.timezone(datetime.timedelta(hours=9), name="JST")
    lines = [f"iter: {int(step)}", f"how: {how}"]
    if source is not None:
        lines.append(f"source: {source}")
    lines.append(f"written_at: {datetime.datetime.now(jst).isoformat(timespec='minutes')}")
    with open(Path(run_dir) / BEST_NOTE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run_name(now):
    return now.strftime(RUN_NAME_FORMAT)


def iter_dirname(step):
    return f"iter_{int(step):07d}"


def ckpt_dir(run_dir, tag):
    """tag（'latest' | 'best' | step 番号）→ 重みディレクトリ。latest / best は <run>/<tag>/、番号は <run>/weights/iter_NNNNNNN/。"""
    tag = str(tag)
    if tag in TOP_TAGS:
        return Path(run_dir) / tag
    if tag.isdigit():
        return Path(run_dir) / WEIGHTS_DIR / iter_dirname(tag)
    raise ValueError(f"checkpoint の tag は latest | best | step 番号: {tag!r}")


def ckpt_path(run_dir, tag, kind):
    return ckpt_dir(run_dir, tag) / kind


def saved_tags(run_dir):
    """その run にある checkpoint の tag 一覧（3 ファイルが揃っている重みディレクトリ）。"""
    run_dir = Path(run_dir)
    ok = lambda d: all((d / k).is_file() for k in CKPT_KINDS)  # noqa: E731
    tags = [t for t in TOP_TAGS if ok(run_dir / t)]
    steps = []
    for d in (run_dir / WEIGHTS_DIR).glob("iter_*"):
        m = re.fullmatch(r"iter_(\d+)", d.name)
        if m and ok(d):
            steps.append(int(m.group(1)))
    return tags + [str(s) for s in sorted(steps)]


def latest_launch(run_dir):
    run_dir = Path(run_dir)
    resumes = sorted(run_dir.glob("launch_resume_*.yaml"))
    return resumes[-1] if resumes else run_dir / LAUNCH_FILE


def iter_images_dir(run_dir, step):
    return Path(run_dir) / OUTPUT_IMAGES_DIR / iter_dirname(step)


def preview_dir(run_dir, name):
    return Path(run_dir) / OUTPUT_IMAGES_DIR / f"preview_{name}"


def find_run_dir(weight_dir):
    """重みディレクトリから run ディレクトリ（launch.yaml がある所）を探す。<run>/best → 1 つ上、<run>/weights/iter_N → 2 つ上。"""
    p = Path(weight_dir).resolve()
    for _ in range(3):
        p = p.parent
        if (p / LAUNCH_FILE).is_file():
            return p
    return None


# --- 推論の出力（Stage 1 / 2 と同じ根 <repo>/output/） ---
OUTPUT_DIRNAME = "output"
INFER_STAMP_FORMAT = "%Y_%m%d_%H%M%S"


def repo_root():
    """リポジトリ直下（このファイルは <repo>/stage3/util/run_paths.py）。コンテナでは /workspace。"""
    return Path(__file__).resolve().parents[2]


def output_root(root=None):
    return Path(root) if root else repo_root() / OUTPUT_DIRNAME


def infer_output_dir(run_dir, weight_dir, input_dir, now, root=None):
    """推論の出力先: <repo>/output/<run>_<重みディレクトリ名>_<入力フォルダ名>_<yyyy_mmdd_HHMMSS>/。"""
    return output_root(root) / f"{Path(run_dir).name}_{Path(weight_dir).name}_{Path(input_dir).name}_{now.strftime(INFER_STAMP_FORMAT)}"


def replace_retry(src, dst, tries=20, wait=0.5):
    """os.replace の再試行版。Windows では Defender や検索インデックスが書いたばかりの .pth を掴んで PermissionError（WinError 5 / 32）に
    なることがあるので、最大 tries 回 wait 秒おきに繰り返す。最後まで失敗したら例外をそのまま上げる（stage2 と同じ、2026-10-13）。"""
    import os
    import time
    for i in range(tries):
        try:
            os.replace(src, dst)
            return
        except (PermissionError, OSError):
            if i == tries - 1:
                raise
            time.sleep(wait)
