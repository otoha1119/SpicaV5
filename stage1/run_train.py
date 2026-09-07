"""run_train.py — Stage 1 学習の起動器（Docker 内で動く）。

設計: docs/plans/20260905_config-design-plan.md、修正: docs/plans/20260907_review-fix-list.md（F-02 / F-03 / F-07 / F-10 / F-11）
  1. configs/train.yaml, configs/mode.yaml, ../configs/machines.yaml を読み、configs/schema.py で検証する
     （必須キー欠落・未知キー・型違いはエラー。既定値は無い）
  2. sh から渡された上書き引数を **平坦 dict に反映**してから（後勝ち。schema に無いフラグはエラー）、値域・相互条件を検証する
  3. schema の順に train.py（junyanz 本家）の引数列を**全パラメータ明示**で組む（実効値から。argv に生の上書きは足さない）
  4. 実験名 --name を起動時刻 yyyy_mmdd_HHMM（JST 固定。コンテナは UTC のため）から生成する。同名の run が既にあればエラー（F-11）
  5. 解決済み設定（実効値）を <checkpoints_dir>/<run>/launch.yaml に保存してから train.py を exec する。
     推論（run_infer.py）と再開はこの実効値を読むので、上書きした G の構造や HU 正規化が引き継がれる

使い方（通常は ../train_stage1.sh 経由）:
  python run_train.py --machine PC1 --train configs/train.yaml --mode configs/mode.yaml --machines ../configs/machines.yaml -- --n_epochs 50
再開:
  python run_train.py --machine PC1 ... --resume 2026_0905_1234 [--resume_tag latest|best|30] -- [--n_epochs 400] [--lr 0.0001]
  → run の**最新の** launch（launch_resume_*.yaml があればそれ、無ければ launch.yaml）の実効設定を基準に、今回の上書きを反映し、
    重みディレクトリ（<run>/latest | best | weights/epoch_NNN）の state.pth（optimizer / RNG / 進捗）と net_G/D.pth を復元して同じディレクトリに続きを保存する。
    設定は launch_resume_<日時秒>.yaml に保存され、次の再開・推論の基準になる
"""

import argparse
import datetime
import os
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from configs.schema import MACHINE, MODE, TRAIN, ConfigError, apply_overrides, check_values, to_argv, validate  # noqa: E402
from util.run_paths import LAUNCH_FILE, ckpt_path, latest_launch, run_name, saved_tags  # noqa: E402

# Stage 1 で固定の引数（チューニング対象ではないので設定ファイルには置かない）
FIXED_ARGV = ["--model", "fidelity_gan", "--dataset_mode", "ct"]
RESUME_ONLY_FLAGS = ("--continue_train", "--epoch_count")  # 再開の内部状態。sh からは上書きできない（run_train が決める）


def load_yaml(path):
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"設定ファイルが見つかりません: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_argv(name, train, mode, machine, extra):
    """実効値（平坦 dict）から train.py の引数列を組む。"""
    return ["--name", name] + FIXED_ARGV + to_argv(train, TRAIN) + to_argv(mode, MODE) + to_argv(machine, MACHINE) + list(extra)


def sections(train, mode, machine):
    return {"train": (train, TRAIN), "mode": (mode, MODE), "machine": (machine, MACHINE)}


def prepare_new(a, train, mode, machine, overrides, now):
    """新規 run: yaml の値に上書きを反映 → 検証 → run 名と保存先。"""
    applied = apply_overrides(overrides, sections(train, mode, machine))
    if any(f in applied for f in RESUME_ONLY_FLAGS):
        raise ConfigError(f"{' / '.join(RESUME_ONLY_FLAGS)} は再開の内部フラグです。再開は --resume <run> で行ってください")
    if train["log.continue_train"] or train["log.epoch_count"] != 1:
        raise ConfigError("新規 run では log.continue_train = false, log.epoch_count = 1 にしてください（再開は --resume <run>）")
    check_values(train, mode, machine)
    name = run_name(now)
    out_dir = Path(machine["checkpoints_dir"]) / name
    if out_dir.exists():
        raise ConfigError(f"run ディレクトリが既に存在します: {out_dir}（同一分の再起動。1 分待つか、不要なら消してから起動）")
    return name, out_dir, [], LAUNCH_FILE, applied, None


def prepare_resume(a, train_yaml, mode_yaml, machine_yaml, overrides, now):
    """再開: run の最新 launch の実効設定を基準に、今回の上書きを反映 → 検証 → 再開フラグ。"""
    import torch  # コンテナ内でのみ使う（state.pth の進捗を読む）

    # run の場所: 今の machines.yaml の checkpoints_dir に、今回の上書き（--checkpoints_dir があれば）を反映して探す
    probe = (dict(train_yaml), dict(mode_yaml), dict(machine_yaml))
    apply_overrides(overrides, sections(*probe))
    run_dir = Path(probe[2]["checkpoints_dir"]) / a.resume
    base = latest_launch(run_dir)
    if not base.is_file():
        raise ConfigError(f"再開する run が見つかりません: {run_dir / LAUNCH_FILE}")
    saved = load_yaml(base)
    try:
        train = validate("train(run)", saved["train"], TRAIN)
        mode = validate("mode(run)", saved["mode"], MODE)
        machine = validate("machine(run)", saved["machine"], MACHINE)
    except (KeyError, TypeError) as e:
        raise ConfigError(f"{base} の形式が不正です: {e}")
    for section, cur, sv in (("train", train_yaml, train), ("mode", mode_yaml, mode), ("machine", machine_yaml, machine)):
        diff = {k: (sv.get(k), v) for k, v in cur.items() if sv.get(k) != v and k not in ("log.continue_train", "log.epoch_count")}
        if diff:
            print(f"[run_train] 注意: 今の {section} 設定は run の実効設定と異なる（run の値を使う。変えるなら --flag で上書き）: {diff}")
    applied = apply_overrides(overrides, sections(train, mode, machine))
    if any(f in applied for f in RESUME_ONLY_FLAGS):
        raise ConfigError(f"{' / '.join(RESUME_ONLY_FLAGS)} は再開の内部フラグです（run_train が決めます）")
    if Path(machine["checkpoints_dir"]) / a.resume != run_dir:
        raise ConfigError(f"再開時に checkpoints_dir を変えることはできません: {machine['checkpoints_dir']} != {run_dir.parent}")
    state_path = ckpt_path(run_dir, a.resume_tag, "state.pth")
    if not state_path.is_file():
        raise ConfigError(f"再開用 state がありません: {state_path}（保存済み tag: {saved_tags(run_dir)}）")
    for kind in ("net_G.pth", "net_D.pth"):
        if not ckpt_path(run_dir, a.resume_tag, kind).is_file():
            raise ConfigError(f"再開用の重みがありません: {ckpt_path(run_dir, a.resume_tag, kind)}")
    st = torch.load(state_path, map_location="cpu", weights_only=True)
    epoch_count = st["epoch"] + 1 if st["epoch_done"] else st["epoch"]  # epoch 末保存なら次の epoch から、途中保存ならその epoch を頭から
    train["log.continue_train"] = True
    train["log.epoch_count"] = epoch_count
    check_values(train, mode, machine)
    extra = ["--epoch", str(a.resume_tag), "--resume_state", str(state_path)]
    print(f"[run_train] resume {a.resume} from tag '{a.resume_tag}' (base: {base.name}): saved epoch {st['epoch']} (done={st['epoch_done']}), total_iters {st['total_iters']} → epoch_count {epoch_count}, n_epochs {train['optim.n_epochs']}")
    return a.resume, run_dir, extra, f"launch_resume_{now.strftime('%Y%m%d_%H%M%S')}.yaml", applied, base, train, mode, machine


def main():
    p = argparse.ArgumentParser(description="Stage 1 (FE-GAN) 学習の起動器")
    p.add_argument("--machine", required=True, help="configs/machines.yaml のエントリ名（PC1, mac, ...）")
    p.add_argument("--train", required=True, help="学習パラメータ YAML")
    p.add_argument("--mode", required=True, help="モード切替 YAML")
    p.add_argument("--machines", required=True, help="マシン定義 YAML")
    p.add_argument("--resume", default=None, metavar="RUN", help="再開する run 名（yyyy_mmdd_HHMM）。その run の最新 launch の実効設定を基準にし、同じディレクトリに続きを保存する")
    p.add_argument("--resume_tag", default="latest", help="再開に使う checkpoint の tag（latest | best | epoch 番号）")
    p.add_argument("overrides", nargs=argparse.REMAINDER, help="'--' の後に train.py の引数（schema にあるフラグのみ。bool は --flag true|false も可）")
    a = p.parse_args()
    overrides = a.overrides[1:] if a.overrides and a.overrides[0] == "--" else a.overrides

    # コンテナ内は UTC なので、実験名の時刻は JST (+9) 固定で生成する（ホストの時計と一致させる）
    jst = datetime.timezone(datetime.timedelta(hours=9), name="JST")
    now = datetime.datetime.now(jst)
    try:
        train = validate("train", load_yaml(a.train), TRAIN)
        mode = validate("mode", load_yaml(a.mode), MODE)
        machines = load_yaml(a.machines)
        if not isinstance(machines, dict) or a.machine not in machines:
            raise ConfigError(f"machines.yaml にエントリ '{a.machine}' がありません。候補: {list(machines) if isinstance(machines, dict) else '(不正な形式)'}")
        machine = validate(f"machines.{a.machine}", machines[a.machine], MACHINE)
        if a.resume:
            name, out_dir, extra, launch_name, applied, base, train, mode, machine = prepare_resume(a, train, mode, machine, overrides, now)
        else:
            name, out_dir, extra, launch_name, applied, base = prepare_new(a, train, mode, machine, overrides, now)
    except ConfigError as e:
        print(f"[run_train] 設定エラー: {e}", file=sys.stderr)
        sys.exit(2)

    argv = build_argv(name, train, mode, machine, extra)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / launch_name, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {"timestamp": now.isoformat(timespec="seconds"), "machine_name": a.machine,
             "train": train, "mode": mode, "machine": machine,  # 実効値（yaml + 上書き）。推論・再開はこれを読む
             "overrides": list(overrides), "applied": applied,
             "resume": a.resume, "resume_tag": a.resume_tag if a.resume else None, "base_launch": str(base) if base else None,
             "argv": argv},
            f, allow_unicode=True, sort_keys=False,
        )

    cmd = [sys.executable, str(HERE / "train.py")] + argv
    print("[run_train] " + " ".join(cmd), flush=True)
    os.chdir(HERE)
    os.execv(sys.executable, cmd)


if __name__ == "__main__":
    main()
