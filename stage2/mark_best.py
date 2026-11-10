"""stage2/mark_best.py — 目で選んだ重みを run 直下の best/ に写し、best.txt に記録する（設計書 B-4・C-5: best は自動では選ばない。学習が上限まで進むと最後の EMA が best になる）。

ホストからは  bash start2.sh best <u|v> <run> <step|latest>  。
  python stage2/mark_best.py --machine PC1 --machines configs/machines.yaml --kind u --run 2026_1113_0900 --tag 32000 [--note "理由"]
best.tmp/ にファイルを揃えてから差し替える（途中で止まっても best/ に新旧が混ざらない）。**病院 EID の出力を見て選ばない**（評価専用。設計書 F-1）。
"""

import argparse
import shutil
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from configs.schema import ConfigError  # noqa: E402
from models.trainer import _swap_dirs  # noqa: E402
from run_infer import find_weights  # noqa: E402
from util.launcher import load_machine  # noqa: E402
from util.run_paths import CKPT_KINDS, KINDS, ckpt_dir, write_best_note  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="重みを best として印を付ける（新 Stage 2）")
    p.add_argument("--machine", required=True)
    p.add_argument("--machines", required=True)
    p.add_argument("--kind", required=True, choices=KINDS)
    p.add_argument("--run", required=True)
    p.add_argument("--tag", required=True, help="weights/ にある step 番号、または latest")
    p.add_argument("--note", default="", help="best.txt に残す理由")
    a = p.parse_args()
    try:
        if not (a.tag.isdigit() or a.tag == "latest"):
            raise ConfigError(f"tag は step 番号か latest: {a.tag!r}")
        run_dir, src = find_weights(load_machine(a.machines, a.machine), a.kind, a.run, a.tag)
    except ConfigError as e:
        print(f"[best] 設定エラー: {e}", file=sys.stderr)
        sys.exit(2)
    final = ckpt_dir(run_dir, "best")
    tmp, old = final.with_name("best.tmp"), final.with_name("best.old")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    for name in CKPT_KINDS[a.kind]:
        shutil.copy2(src / name, tmp / name)
    step = int(torch.load(tmp / "state.pth", map_location="cpu", weights_only=True)["step"])
    if not _swap_dirs(tmp, final, old):
        sys.exit(1)
    write_best_note(run_dir, step, "manual", str(src.relative_to(run_dir)), a.note)
    print(f"[best] {a.kind} の run {run_dir.name}: best = step {step}（{src.relative_to(run_dir)} を写した。best.txt に記録）")


if __name__ == "__main__":
    main()
