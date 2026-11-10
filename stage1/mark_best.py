"""stage1/mark_best.py — 目で選んだ重みを run 直下の best/ に写し、best.txt に記録する（設計書 G-2: best は自動で選ばない）。

ホストからは  bash start.sh best <run> <step|latest>  。
  python stage1/mark_best.py --machine PC1 --machines configs/machines.yaml --run 2026_1112_0030 --tag 150000 [--note "理由"]
best.tmp/ に 3 ファイルを揃えてから差し替える（途中で止まっても best/ に新旧が混ざらない）。
"""

import argparse
import shutil
import sys
from pathlib import Path

import torch
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from configs.schema import MACHINE, ConfigError, validate_shared  # noqa: E402
from models.painter import _swap_dirs  # noqa: E402
from util.run_paths import CKPT_KINDS, LAUNCH_FILE, ckpt_dir, saved_tags, write_best_note  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="重みを best として印を付ける（新 Stage 1）")
    p.add_argument("--machine", required=True)
    p.add_argument("--machines", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--tag", required=True, help="weights/ にある step 番号、または latest")
    p.add_argument("--note", default="", help="best.txt に残す理由")
    a = p.parse_args()
    try:
        if not (a.tag.isdigit() or a.tag == "latest"):
            raise ConfigError(f"tag は step 番号か latest: {a.tag!r}")
        with open(a.machines, encoding="utf-8") as f:
            machines = yaml.safe_load(f)
        if not isinstance(machines, dict) or a.machine not in machines:
            raise ConfigError(f"machines.yaml にエントリ '{a.machine}' がありません")
        machine = validate_shared(f"machines.{a.machine}", machines[a.machine], MACHINE)
        run_dir = Path(machine["painter_checkpoints_dir"]) / a.run
        if not (run_dir / LAUNCH_FILE).is_file():
            raise ConfigError(f"run が見つかりません: {run_dir}")
        src = ckpt_dir(run_dir, a.tag)
        missing = [k for k in CKPT_KINDS if not (src / k).is_file()]
        if missing:
            raise ConfigError(f"{src} に {missing} がありません（保存済み tag: {saved_tags(run_dir)}）")
    except ConfigError as e:
        print(f"[best] 設定エラー: {e}", file=sys.stderr)
        sys.exit(2)
    final = ckpt_dir(run_dir, "best")
    tmp, old = final.with_name("best.tmp"), final.with_name("best.old")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    for kind in CKPT_KINDS:
        shutil.copy2(src / kind, tmp / kind)
    step = int(torch.load(tmp / "state.pth", map_location="cpu", weights_only=True)["step"])
    if not _swap_dirs(tmp, final, old):
        sys.exit(1)
    write_best_note(run_dir, step, str(src.relative_to(run_dir)), a.note)
    print(f"[best] run {run_dir.name}: best = step {step}（{src.relative_to(run_dir)} を写した。best.txt に記録）")


if __name__ == "__main__":
    main()
