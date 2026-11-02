"""mark_best.py — weights/epoch_NNN/{net_G,net_D,state}.pth を run 直下の best_eNNN/ にコピーし、best.txt に記録する（手動の best。自動は util/validator.py）。[SpicaV5 新規]

ホストからは  bash start.sh best <run> <epoch>  。
「最も良かった epoch」を自動で決める定量指標がまだ無いので、当面は目視で選んだ epoch を手で指定する（指標ができたら自動化）。
best/ は latest/ と同じく run 直下の重みディレクトリ（レイアウトは util/run_paths.py）。resume の tag に best を、infer の WEIGHT_DIR に <run>/best を指定できる。

使い方:
  python mark_best.py --machine PC1 --machines ../configs/machines.yaml --run 2026_0907_1742 --epoch 30
"""

import argparse
import datetime
import os
import shutil
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from configs.schema import MACHINE, ConfigError, validate  # noqa: E402
from util.run_paths import BEST_NOTE, CKPT_KINDS, LAUNCH_FILE, best_dirname, ckpt_path, retire_other_bests, saved_tags  # noqa: E402

KINDS = CKPT_KINDS


def main():
    p = argparse.ArgumentParser(description="epoch の checkpoint を best としてマークする")
    p.add_argument("--machine", required=True)
    p.add_argument("--machines", required=True)
    p.add_argument("--run", required=True, help="run 名（yyyy_mmdd_HHMM。machines.yaml の checkpoints_dir 配下）か run ディレクトリのパス")
    p.add_argument("--epoch", required=True, help="weights/ にある epoch 番号")
    a = p.parse_args()
    try:
        if not a.epoch.isdigit():
            raise ConfigError(f"epoch は番号で指定してください: {a.epoch!r}")
        machines = yaml.safe_load(open(a.machines, encoding="utf-8"))
        if not isinstance(machines, dict) or a.machine not in machines:
            raise ConfigError(f"machines.yaml にエントリ '{a.machine}' がありません")
        machine = validate(f"machines.{a.machine}", machines[a.machine], MACHINE)
        run_dir = Path(a.run) if (Path(a.run) / LAUNCH_FILE).is_file() else Path(machine["checkpoints_dir"]) / a.run  # run 名 or run ディレクトリのパス
        if not (run_dir / LAUNCH_FILE).is_file():
            raise ConfigError(f"run が見つかりません: {run_dir}（run 名か、launch.yaml のある run ディレクトリのパス）")
        srcs = [ckpt_path(run_dir, a.epoch, k) for k in KINDS]
        missing = [str(s) for s in srcs if not s.is_file()]
        if missing:
            raise ConfigError(f"epoch {a.epoch} の checkpoint が揃っていません: {missing}（保存済み tag: {saved_tags(run_dir)}）")
    except ConfigError as e:
        print(f"[best] 設定エラー: {e}", file=sys.stderr)
        sys.exit(2)

    # 原子的に差し替える（F-14）: best_eNNN.tmp/ に 3 ファイルを揃えてから rename。途中で止まっても新旧が混ざらない。他の best_e*（と旧 best/）は消す（best は常に 1 つ）
    final = run_dir / best_dirname(a.epoch)
    tmp, old = final.with_name(final.name + ".tmp"), final.with_name(final.name + ".old")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    for kind, src in zip(KINDS, srcs):
        shutil.copy2(src, tmp / kind)
        print(f"[best] {src.relative_to(run_dir)} -> {final.name}/{kind}")
    if old.exists():
        shutil.rmtree(old)
    if final.exists():
        os.replace(final, old)
    os.replace(tmp, final)
    if old.exists():
        shutil.rmtree(old)
    retire_other_bests(run_dir, final)
    jst = datetime.timezone(datetime.timedelta(hours=9), name="JST")
    note = run_dir / BEST_NOTE
    with open(note, "w", encoding="utf-8") as f:
        f.write(f"epoch: {a.epoch}\ndir: {final.name}\nmarked_at: {datetime.datetime.now(jst).isoformat(timespec='minutes')}\n"
                f"source: manual ({', '.join(str(s.relative_to(run_dir)) for s in srcs)})\n")
    print(f"[best] run {run_dir.name}: best = epoch {a.epoch} → {final.name}/ ({note})")


if __name__ == "__main__":
    main()
