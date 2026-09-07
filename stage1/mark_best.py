"""mark_best.py — weights/epoch_NNN/{net_G,net_D,state}.pth を run 直下の best/ にコピーし、best.txt に記録する。[SpicaV5 新規]

ホストからは  bash start.sh best <run> <epoch>  。
「最も良かった epoch」を自動で決める定量指標がまだ無いので、当面は目視で選んだ epoch を手で指定する（指標ができたら自動化）。
best/ は latest/ と同じく run 直下の重みディレクトリ（レイアウトは util/run_paths.py）。resume の tag に best を、infer の WEIGHT_DIR に <run>/best を指定できる。

使い方:
  python mark_best.py --machine PC1 --machines ../configs/machines.yaml --run 2026_0907_1742 --epoch 30
"""

import argparse
import datetime
import shutil
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from configs.schema import MACHINE, ConfigError, validate  # noqa: E402
from util.run_paths import BEST_NOTE, CKPT_KINDS, LAUNCH_FILE, ckpt_dir, ckpt_path, saved_tags  # noqa: E402

KINDS = CKPT_KINDS


def main():
    p = argparse.ArgumentParser(description="epoch の checkpoint を best としてマークする")
    p.add_argument("--machine", required=True)
    p.add_argument("--machines", required=True)
    p.add_argument("--run", required=True, help="run 名（yyyy_mmdd_HHMM）")
    p.add_argument("--epoch", required=True, help="weights/ にある epoch 番号")
    a = p.parse_args()
    try:
        if not a.epoch.isdigit():
            raise ConfigError(f"epoch は番号で指定してください: {a.epoch!r}")
        machines = yaml.safe_load(open(a.machines, encoding="utf-8"))
        if not isinstance(machines, dict) or a.machine not in machines:
            raise ConfigError(f"machines.yaml にエントリ '{a.machine}' がありません")
        machine = validate(f"machines.{a.machine}", machines[a.machine], MACHINE)
        run_dir = Path(machine["checkpoints_dir"]) / a.run
        if not (run_dir / LAUNCH_FILE).is_file():
            raise ConfigError(f"run が見つかりません: {run_dir}")
        srcs = [ckpt_path(run_dir, a.epoch, k) for k in KINDS]
        missing = [str(s) for s in srcs if not s.is_file()]
        if missing:
            raise ConfigError(f"epoch {a.epoch} の checkpoint が揃っていません: {missing}（保存済み tag: {saved_tags(run_dir)}）")
    except ConfigError as e:
        print(f"[best] 設定エラー: {e}", file=sys.stderr)
        sys.exit(2)

    ckpt_dir(run_dir, "best").mkdir(parents=True, exist_ok=True)
    for kind, src in zip(KINDS, srcs):
        dst = ckpt_path(run_dir, "best", kind)
        shutil.copy2(src, dst)
        print(f"[best] {src.relative_to(run_dir)} -> {dst.relative_to(run_dir)}")
    jst = datetime.timezone(datetime.timedelta(hours=9), name="JST")
    note = run_dir / BEST_NOTE
    with open(note, "w", encoding="utf-8") as f:
        f.write(f"epoch: {a.epoch}\nmarked_at: {datetime.datetime.now(jst).isoformat(timespec='minutes')}\n"
                f"source: {', '.join(str(s.relative_to(run_dir)) for s in srcs)}\n"
                "note: 判定基準（定量指標）が未実装のため目視で手動指定\n")
    print(f"[best] run {a.run}: best = epoch {a.epoch} ({note})")


if __name__ == "__main__":
    main()
