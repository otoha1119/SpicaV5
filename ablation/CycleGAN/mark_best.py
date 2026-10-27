"""ablation/CycleGAN/mark_best.py — weights/epoch_NNN/ の 5 ファイルを run 直下の best/ にコピーし best.txt に記録する（stage1/mark_best.py の複製）。
ホストからは  bash start_ablation.sh <マシン> CycleGAN best <run> <epoch>。判定指標が無いので目視で選んだ epoch を手で指定する。"""

import argparse
import datetime
import os
import shutil
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from configs.schema import MACHINE, ConfigError, validate_shared  # noqa: E402
from util.run_paths import BEST_NOTE, CKPT_KINDS, LAUNCH_FILE, checkpoints_root, ckpt_dir, ckpt_path, saved_tags  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="epoch の checkpoint を best としてマークする")
    p.add_argument("--machine", required=True)
    p.add_argument("--machines", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--epoch", required=True)
    a = p.parse_args()
    try:
        if not a.epoch.isdigit():
            raise ConfigError(f"epoch は番号で指定してください: {a.epoch!r}")
        machines = yaml.safe_load(open(a.machines, encoding="utf-8"))
        if not isinstance(machines, dict) or a.machine not in machines:
            raise ConfigError(f"machines.yaml にエントリ '{a.machine}' がありません")
        machine = validate_shared(f"machines.{a.machine}", machines[a.machine], MACHINE)
        run_dir = Path(a.run) if (Path(a.run) / LAUNCH_FILE).is_file() else checkpoints_root(machine) / a.run
        if not (run_dir / LAUNCH_FILE).is_file():
            raise ConfigError(f"run が見つかりません: {run_dir}")
        srcs = [ckpt_path(run_dir, a.epoch, k) for k in CKPT_KINDS]
        missing = [str(s) for s in srcs if not s.is_file()]
        if missing:
            raise ConfigError(f"epoch {a.epoch} の checkpoint が揃っていません: {missing}（保存済み tag: {saved_tags(run_dir)}）")
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
        os.replace(final, old)
    os.replace(tmp, final)
    if old.exists():
        shutil.rmtree(old)
    jst = datetime.timezone(datetime.timedelta(hours=9), name="JST")
    with open(run_dir / BEST_NOTE, "w", encoding="utf-8") as f:
        f.write(f"epoch: {a.epoch}\nmarked_at: {datetime.datetime.now(jst).isoformat(timespec='minutes')}\nsource: {', '.join(str(s.relative_to(run_dir)) for s in srcs)}\nnote: 判定基準（定量指標）が未実装のため目視で手動指定\n")
    print(f"[best] run {run_dir.name}: best = epoch {a.epoch}")


if __name__ == "__main__":
    main()
