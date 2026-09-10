"""mark_best.py — weights/epoch_NNN/{net_G,state}.pth を run 直下の best/ にコピーし、best.txt に記録する（手動指定）。[SpicaV5 新規。stage1/mark_best.py から複製]

ホストからは  bash start2.sh best <run> <epoch>  。
通常は学習中に val の log.best_metric が最良を更新するたびに best/ が自動で書かれる（models/regression_model.update_best）。
これは目視などで別の epoch を best にしたいときの上書き用。自動更新の比較対象は state.pth の best 記録なので、
この後にその run を resume すると、コピーした epoch の state に入っている記録（その時点までの自動 best）が基準になる。

使い方:
  python mark_best.py --machine PC1 --machines ../configs/machines.yaml --run 2026_0909_1200 --epoch 30
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from configs.schema import MACHINE, ConfigError, validate_shared  # noqa: E402
from util.run_paths import CKPT_KINDS, LAUNCH_FILE, ckpt_dir, ckpt_path, saved_tags, write_best_note  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="epoch の checkpoint を best としてマークする（Stage 2）")
    p.add_argument("--machine", required=True)
    p.add_argument("--machines", required=True)
    p.add_argument("--run", required=True, help="run 名（yyyy_mmdd_HHMM。machines.yaml の stage2_checkpoints_dir 配下）か run ディレクトリのパス")
    p.add_argument("--epoch", required=True, help="weights/ にある epoch 番号")
    a = p.parse_args()
    try:
        if not a.epoch.isdigit():
            raise ConfigError(f"epoch は番号で指定してください: {a.epoch!r}")
        with open(a.machines, encoding="utf-8") as f:
            machines = yaml.safe_load(f)
        if not isinstance(machines, dict) or a.machine not in machines:
            raise ConfigError(f"machines.yaml にエントリ '{a.machine}' がありません")
        machine = validate_shared(f"machines.{a.machine}", machines[a.machine], MACHINE)
        run_dir = Path(a.run) if (Path(a.run) / LAUNCH_FILE).is_file() else Path(machine["stage2_checkpoints_dir"]) / a.run
        if not (run_dir / LAUNCH_FILE).is_file():
            raise ConfigError(f"run が見つかりません: {run_dir}（run 名か、launch.yaml のある run ディレクトリのパス）")
        srcs = [ckpt_path(run_dir, a.epoch, k) for k in CKPT_KINDS]
        missing = [str(s) for s in srcs if not s.is_file()]
        if missing:
            raise ConfigError(f"epoch {a.epoch} の checkpoint が揃っていません: {missing}（保存済み tag: {saved_tags(run_dir)}）")
    except ConfigError as e:
        print(f"[best] 設定エラー: {e}", file=sys.stderr)
        sys.exit(2)

    # 原子的に差し替える: best.tmp/ に 2 ファイルを揃えてから rename。途中で止まっても best/ に新旧が混ざらない
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
    write_best_note(run_dir, a.epoch, "manual", source=", ".join(str(s.relative_to(run_dir)) for s in srcs))
    print(f"[best] run {run_dir.name}: best = epoch {a.epoch}（手動。best.txt に記録）")


if __name__ == "__main__":
    main()
