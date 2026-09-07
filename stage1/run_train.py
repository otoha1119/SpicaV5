"""run_train.py — Stage 1 学習の起動器（Docker 内で動く）。

設計: docs/plans/20260905_config-design-plan.md
  1. configs/train.yaml, configs/mode.yaml, ../configs/machines.yaml を読み、configs/schema.py で検証する
     （必須キー欠落・未知キー・型違いはエラー。既定値は無い）
  2. schema の順に train.py（junyanz 本家）の引数列を**全パラメータ明示**で組む
  3. 実験名 --name を起動時刻 yyyy_mmdd_HHMM（JST 固定。コンテナは UTC のため）から生成する（同一分の再起動は上書き。レイアウトは util/run_paths.py）
  4. sh から渡された上書き引数を末尾に付ける（argparse は後勝ち → sh が最優先）。schema に無いフラグはエラー
  5. 解決済み設定を checkpoints/<name>/launch.yaml に保存してから train.py を exec する

使い方（通常は ../train_stage1.sh 経由）:
  python run_train.py --machine PC1 --train configs/train.yaml --mode configs/mode.yaml --machines ../configs/machines.yaml -- --n_epochs 50
再開:
  python run_train.py --machine PC1 ... --resume 2026_0905_1234 [--resume_tag latest|best|30] -- [--n_epochs 400]
  → run の launch.yaml の設定を使い、重みディレクトリ（<run>/latest | <run>/best | <run>/weights/epoch_NNN）の state.pth（optimizer / RNG / 進捗）と net_G/D.pth を復元して同じディレクトリに続きを保存
"""

import argparse
import datetime
import os
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from configs.schema import MACHINE, MODE, TRAIN, ConfigError, to_argv, validate  # noqa: E402
from util.run_paths import ckpt_path, run_name, saved_tags  # noqa: E402

# Stage 1 で固定の引数（チューニング対象ではないので設定ファイルには置かない）
FIXED_ARGV = ["--model", "fidelity_gan", "--dataset_mode", "ct"]


def load_yaml(path):
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"設定ファイルが見つかりません: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_overrides(tokens):
    """sh からの上書きは schema にあるフラグだけ許す。"""
    allowed = {k.flag for s in (TRAIN, MODE, MACHINE) for k in s.values() if k.flag}
    bad = [t for t in tokens if t.startswith("--") and t.split("=")[0] not in allowed]
    if bad:
        raise ConfigError("上書きできないフラグです（schema.py に無い）: " + ", ".join(bad))


RESUME_FLAGS = {"--continue_train", "--epoch_count", "--epoch", "--resume_state"}  # 以前の再開で付いた引数は捨てて付け直す


def build_resume_argv(a, machine, train, mode, overrides, now):
    """再開: run の launch.yaml に記録された argv（起動時の設定）をそのまま使い、
    --continue_train / --epoch_count / --epoch / --resume_state を付け足す。今の yaml との差は警告として表示するだけ。"""
    import torch  # コンテナ内でのみ使う（state.pth の進捗を読む）

    run_dir = Path(machine["checkpoints_dir"]) / a.resume
    launch = run_dir / "launch.yaml"
    if not launch.is_file():
        raise ConfigError(f"再開する run が見つかりません: {launch}")
    saved = yaml.safe_load(open(launch, encoding="utf-8"))
    for section, cur in (("train", train), ("mode", mode), ("machine", machine)):
        diff = {k: (saved[section].get(k), v) for k, v in cur.items() if saved[section].get(k) != v}
        if diff:
            print(f"[run_train] 注意: 今の {section} 設定は run 起動時と異なる（起動時の値を使う）: {diff}")
    state_path = ckpt_path(run_dir, a.resume_tag, "state.pth")
    if not state_path.is_file():
        raise ConfigError(f"再開用 state がありません: {state_path}（保存済み tag: {saved_tags(run_dir)}）")
    for kind in ("net_G.pth", "net_D.pth"):
        if not ckpt_path(run_dir, a.resume_tag, kind).is_file():
            raise ConfigError(f"再開用の重みがありません: {ckpt_path(run_dir, a.resume_tag, kind)}")
    st = torch.load(state_path, map_location="cpu", weights_only=True)
    epoch_count = st["epoch"] + 1 if st["epoch_done"] else st["epoch"]  # epoch 末保存なら次の epoch から、途中保存ならその epoch を頭から

    base = list(saved["argv"])
    argv = []
    skip = False
    for i, t in enumerate(base):  # 以前の再開フラグを除去
        if skip:
            skip = False
            continue
        if t in RESUME_FLAGS:
            skip = t != "--continue_train"
            continue
        argv.append(t)
    argv += ["--continue_train", "--epoch_count", str(epoch_count), "--epoch", str(a.resume_tag), "--resume_state", str(state_path)]
    argv += list(overrides)
    print(f"[run_train] resume {a.resume} from tag '{a.resume_tag}': saved epoch {st['epoch']} (done={st['epoch_done']}), total_iters {st['total_iters']} → epoch_count {epoch_count}")
    return a.resume, argv, f"launch_resume_{now.strftime('%Y%m%d_%H%M')}.yaml"


def main():
    p = argparse.ArgumentParser(description="Stage 1 (FE-GAN) 学習の起動器")
    p.add_argument("--machine", required=True, help="configs/machines.yaml のエントリ名（PC1, mac, ...）")
    p.add_argument("--train", required=True, help="学習パラメータ YAML")
    p.add_argument("--mode", required=True, help="モード切替 YAML")
    p.add_argument("--machines", required=True, help="マシン定義 YAML")
    p.add_argument("--resume", default=None, metavar="RUN", help="再開する run 名（yyyy_mmdd_HHMM）。その run の launch.yaml の設定をそのまま使い、同じディレクトリに続きを保存する")
    p.add_argument("--resume_tag", default="latest", help="再開に使う checkpoint の tag（latest | best | epoch 番号）")
    p.add_argument("overrides", nargs=argparse.REMAINDER, help="'--' の後に train.py の引数（schema にあるフラグのみ）")
    a = p.parse_args()
    overrides = a.overrides[1:] if a.overrides and a.overrides[0] == "--" else a.overrides

    try:
        train = validate("train", load_yaml(a.train), TRAIN)
        mode = validate("mode", load_yaml(a.mode), MODE)
        machines = load_yaml(a.machines)
        if not isinstance(machines, dict) or a.machine not in machines:
            raise ConfigError(f"machines.yaml にエントリ '{a.machine}' がありません。候補: {list(machines) if isinstance(machines, dict) else '(不正な形式)'}")
        machine = validate(f"machines.{a.machine}", machines[a.machine], MACHINE)
        check_overrides(overrides)
    except ConfigError as e:
        print(f"[run_train] 設定エラー: {e}", file=sys.stderr)
        sys.exit(2)

    # コンテナ内は UTC なので、実験名の時刻は JST (+9) 固定で生成する（ホストの時計と一致させる）
    jst = datetime.timezone(datetime.timedelta(hours=9), name="JST")
    now = datetime.datetime.now(jst)
    if a.resume:
        try:
            name, argv, launch_name = build_resume_argv(a, machine, train, mode, overrides, now)
        except ConfigError as e:
            print(f"[run_train] 設定エラー: {e}", file=sys.stderr)
            sys.exit(2)
    else:
        name = run_name(now)  # yyyy_mmdd_HHMM
        argv = ["--name", name] + FIXED_ARGV + to_argv(train, TRAIN) + to_argv(mode, MODE) + to_argv(machine, MACHINE) + list(overrides)
        launch_name = "launch.yaml"

    out_dir = Path(machine["checkpoints_dir"]) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / launch_name, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {"timestamp": now.isoformat(timespec="minutes"), "machine_name": a.machine, "train": train, "mode": mode, "machine": machine,
             "overrides": list(overrides), "resume": a.resume, "resume_tag": a.resume_tag if a.resume else None, "argv": argv},
            f, allow_unicode=True, sort_keys=False,
        )

    cmd = [sys.executable, str(HERE / "train.py")] + argv
    print("[run_train] " + " ".join(cmd), flush=True)
    os.chdir(HERE)
    os.execv(sys.executable, cmd)


if __name__ == "__main__":
    main()
