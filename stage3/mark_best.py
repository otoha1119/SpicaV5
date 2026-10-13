"""stage3/mark_best.py — weights/iter_NNNNNNN/{model,ema,state}.pth を run 直下の best/ にコピーし、best.txt に記録する（手動指定）。stage2/mark_best.py から複製（epoch → step）。

ホストからは  bash start3.sh best <run> <step>  。Stage 3 は best を自動では書かない（val の指標を学習中に取らない。決定 7 の評価は推論 + metrics.txt / NPS で行う）。
"""

import argparse
import shutil
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from configs.schema import MACHINE, ConfigError, validate_shared  # noqa: E402
from util.run_paths import CKPT_KINDS, LAUNCH_FILE, ckpt_dir, ckpt_path, replace_retry, saved_tags, write_best_note  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="step の checkpoint を best としてマークする（Stage 3）")
    p.add_argument("--machine", required=True)
    p.add_argument("--machines", required=True)
    p.add_argument("--run", required=True, help="run 名（stage3_checkpoints_dir 配下）か run ディレクトリのパス")
    p.add_argument("--step", required=True, help="weights/ にある step 番号")
    a = p.parse_args()
    try:
        if not a.step.isdigit():
            raise ConfigError(f"step は番号で指定してください: {a.step!r}")
        with open(a.machines, encoding="utf-8") as f:
            machines = yaml.safe_load(f)
        if not isinstance(machines, dict) or a.machine not in machines:
            raise ConfigError(f"machines.yaml にエントリ '{a.machine}' がありません")
        machine = validate_shared(f"machines.{a.machine}", machines[a.machine], MACHINE)
        run_dir = Path(a.run) if (Path(a.run) / LAUNCH_FILE).is_file() else Path(machine["stage3_checkpoints_dir"]) / a.run
        if not (run_dir / LAUNCH_FILE).is_file():
            raise ConfigError(f"run が見つかりません: {run_dir}")
        srcs = [ckpt_path(run_dir, a.step, k) for k in CKPT_KINDS]
        missing = [str(s) for s in srcs if not s.is_file()]
        if missing:
            raise ConfigError(f"step {a.step} の checkpoint が揃っていません: {missing}（保存済み tag: {saved_tags(run_dir)}）")
    except ConfigError as e:
        print(f"[best] 設定エラー: {e}", file=sys.stderr)
        sys.exit(2)

    final = ckpt_dir(run_dir, "best")
    tmp, old = final.with_name("best.tmp"), final.with_name("best.old")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    for kind, src in zip(CKPT_KINDS, srcs):
        shutil.copy2(src, tmp / kind)
        print(f"[best] {src.relative_to(run_dir)} -> best/{kind}")
    if old.exists():
        shutil.rmtree(old)
    if final.exists():
        replace_retry(final, old)
    replace_retry(tmp, final)
    if old.exists():
        shutil.rmtree(old)
    write_best_note(run_dir, a.step, "manual", source=", ".join(str(s.relative_to(run_dir)) for s in srcs))
    print(f"[best] run {run_dir.name}: best = step {a.step}（手動。best.txt に記録）")


if __name__ == "__main__":
    main()
