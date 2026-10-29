"""stage1_v2/run_train.py — CycleGAN 版 Stage 1 の学習起動器。

  1. configs/train.yaml と ../configs/machines.yaml（使うキーだけ）を configs/schema.py で検証する（既定値なし）
  2. sh の上書きを反映し、PCD512 / EID512 と固定スライスを検査する
  3. run 名 = 起動時刻 yyyy_mmdd_HHMM（JST）。保存先 stage1_v2/checkpoints/<run>/
  4. 実効値を launch.yaml に保存し、train.py に渡す
  5. start1_v2.sh 経由なら、この run の tb/ を logdir にして TensorBoard を起動する

再開: --resume <run> [--resume_tag latest|best|<epoch>]。基準は選んだ checkpoint の state.pth に入っている実効設定。
"""

import argparse
import datetime
import os
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from configs.schema import MACHINE, TRAIN, ConfigError, apply_overrides, check_train_values, validate, validate_shared  # noqa: E402
from util.run_paths import STAGE, CKPT_KINDS, LAUNCH_FILE, checkpoints_root, ckpt_path, latest_launch, run_name, saved_tags  # noqa: E402

SHAPE_KEYS = ("network.netG", "network.ngf", "network.ndf", "network.norm", "network.n_layers_D")  # 重みの形。再開で変えられない


def load_yaml(path):
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"設定ファイルが見つかりません: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def start_tensorboard(run_dir, port):
    """この run の tb/ だけを logdir にして TensorBoard を起動する。"""
    import shutil
    import signal
    import subprocess

    exe = shutil.which("tensorboard") or (str(Path(sys.executable).parent / "tensorboard") if (Path(sys.executable).parent / "tensorboard").exists() else None)
    if exe is None:
        raise ConfigError("tensorboard が見つかりません")
    proc_dir = Path("/proc")
    if proc_dir.is_dir():
        for d in proc_dir.glob("[0-9]*"):
            try:
                cmd = (d / "cmdline").read_bytes()
            except OSError:
                continue
            if b"tensorboard" in cmd and b"--logdir" in cmd and (b"--port\x00" + str(port).encode() + b"\x00") in cmd and int(d.name) != os.getpid():
                try:
                    os.kill(int(d.name), signal.SIGTERM)
                except OSError:
                    pass
    tb_dir = run_dir / "tb"
    tb_dir.mkdir(parents=True, exist_ok=True)
    log = open(run_dir / "tensorboard.log", "ab")
    subprocess.Popen([exe, "--logdir", str(tb_dir), "--port", str(port), "--bind_all", "--reload_interval", "5"], stdout=log, stderr=log, start_new_session=True)
    print(f"[run_train] TensorBoard: http://localhost:{port} → {tb_dir}（全 run は bash start1_v2.sh <マシン> tb）", flush=True)


def sections(train, machine):
    return {"train": (train, TRAIN), "machine": (machine, MACHINE)}


def check_paths(train, machine):
    """空の run を残す前に Stage 1 degrade と同じ入力を確認する。"""
    for k in ("pcd_dir", "eid_dir"):
        if not Path(machine[k]).is_dir():
            raise ConfigError(f"machines.yaml の {k} がありません: {machine[k]}")
    if not (Path(machine["pcd_dir"]) / train["log.full_slice"]).is_file():
        raise ConfigError(f"log.full_slice が pcd_dir の下にありません: {Path(machine['pcd_dir']) / train['log.full_slice']}")
    if not (Path(machine["eid_dir"]) / train["log.eid_slice"]).is_file():
        raise ConfigError(f"log.eid_slice が eid_dir の下にありません: {Path(machine['eid_dir']) / train['log.eid_slice']}")


def prepare_new(train, machine, overrides, now):
    applied = apply_overrides(overrides, sections(train, machine))
    check_train_values(train, machine)
    check_paths(train, machine)
    name = run_name(now)
    run_dir = checkpoints_root(machine) / name
    if run_dir.exists():
        raise ConfigError(f"run ディレクトリが既に存在します: {run_dir}（同一分の再起動。1 分待つか、不要なら消してから起動）")
    return name, run_dir, 1, None, LAUNCH_FILE, applied, None, train, machine


def prepare_resume(a, train_yaml, machine_yaml, overrides, now):
    import torch

    probe = (dict(train_yaml), dict(machine_yaml))
    apply_overrides(overrides, sections(*probe))
    run_dir = checkpoints_root(probe[1]) / a.resume
    if not run_dir.is_dir():
        raise ConfigError(f"再開する run が見つかりません: {run_dir}")
    state_path = ckpt_path(run_dir, a.resume_tag, "state.pth")
    if not state_path.is_file():
        raise ConfigError(f"再開用 state がありません: {state_path}（保存済み tag: {saved_tags(run_dir)}）")
    for kind in CKPT_KINDS:
        if not ckpt_path(run_dir, a.resume_tag, kind).is_file():
            raise ConfigError(f"再開用の重みがありません: {ckpt_path(run_dir, a.resume_tag, kind)}")
    st = torch.load(state_path, map_location="cpu", weights_only=True)
    if isinstance(st.get("config"), dict) and all(k in st["config"] for k in ("train", "machine")):
        saved, base = {k: dict(st["config"][k]) for k in ("train", "machine")}, state_path
    else:
        base = latest_launch(run_dir)
        if not base.is_file():
            raise ConfigError(f"{state_path} に設定が無く、{run_dir / LAUNCH_FILE} もありません")
        print(f"[run_train] 注意: {state_path.name} に実効設定が無いので、最新の launch（{base.name}）を基準にする")
        saved = load_yaml(base)
    try:
        for section, cur, schema in (("train", train_yaml, TRAIN), ("machine", machine_yaml, MACHINE)):
            missing = [k for k in schema if k not in saved[section]]
            if missing:
                print(f"[run_train] 注意: run の {section} に無いキー（run 作成後に追加）は今の yaml の値を使う: " + ", ".join(f"{k}={cur[k]!r}" for k in missing))
                for k in missing:
                    saved[section][k] = cur[k]
            obsolete = [k for k in saved[section] if k not in schema]
            if obsolete:
                print(f"[run_train] 注意: run の {section} にある廃止キーは無視する: " + ", ".join(f"{k}={saved[section][k]!r}" for k in obsolete))
                for k in obsolete:
                    del saved[section][k]
        train = validate("train(run)", saved["train"], TRAIN)
        machine = validate_shared("machine(run)", saved["machine"], MACHINE)
    except (KeyError, TypeError) as e:
        raise ConfigError(f"{base} の形式が不正です: {e}")
    for section, cur, sv in (("train", train_yaml, train), ("machine", machine_yaml, machine)):
        diff = {k: (sv.get(k), v) for k, v in cur.items() if sv.get(k) != v}
        if diff:
            print(f"[run_train] 注意: 今の {section} 設定は checkpoint の実効設定と異なる（checkpoint の値を使う。変えるなら --flag で上書き）: {diff}")
    saved_shape = {k: train[k] for k in SHAPE_KEYS}
    applied = apply_overrides(overrides, sections(train, machine))
    changed = {k: (saved_shape[k], train[k]) for k in SHAPE_KEYS if train[k] != saved_shape[k]}
    if changed:
        raise ConfigError(f"再開時に重みの形を決めるキーは変えられません: {changed}")
    check_train_values(train, machine)
    check_paths(train, machine)
    if checkpoints_root(machine) / a.resume != run_dir:
        raise ConfigError(f"再開先が stage1_v2/checkpoints と一致しません: {checkpoints_root(machine)} != {run_dir.parent}")
    epoch_count = st["epoch"] + 1 if st["epoch_done"] else st["epoch"]
    total = train["optim.n_epochs"] + train["optim.n_epochs_decay"]
    if epoch_count > total:
        raise ConfigError(f"run は既に epoch {st['epoch']} まで終わっています（n_epochs + n_epochs_decay = {total}）。延ばすなら --n_epochs_decay 等を上書き")
    print(f"[run_train] resume {a.resume} from tag '{a.resume_tag}' (base: {'state.pth の config' if base == state_path else base.name}): saved epoch {st['epoch']} (done={st['epoch_done']}), total_iters {st['total_iters']} → epoch_count {epoch_count} / {total}")
    return a.resume, run_dir, epoch_count, state_path, f"launch_resume_{now.strftime('%Y%m%d_%H%M%S')}.yaml", applied, base, train, machine


def main():
    p = argparse.ArgumentParser(description=f"{STAGE} 学習の起動器")
    p.add_argument("--machine", required=True)
    p.add_argument("--train", required=True)
    p.add_argument("--machines", required=True)
    p.add_argument("--resume", default=None, metavar="RUN")
    p.add_argument("--resume_tag", default="latest")
    p.add_argument("overrides", nargs=argparse.REMAINDER)
    a = p.parse_args()
    overrides = a.overrides[1:] if a.overrides and a.overrides[0] == "--" else a.overrides

    jst = datetime.timezone(datetime.timedelta(hours=9), name="JST")
    now = datetime.datetime.now(jst)
    try:
        train = validate("train", load_yaml(a.train), TRAIN)
        machines = load_yaml(a.machines)
        if not isinstance(machines, dict) or a.machine not in machines:
            raise ConfigError(f"machines.yaml にエントリ '{a.machine}' がありません。候補: {list(machines) if isinstance(machines, dict) else '(不正な形式)'}")
        machine = validate_shared(f"machines.{a.machine}", machines[a.machine], MACHINE)
        if a.resume:
            name, run_dir, epoch_count, state_path, launch_name, applied, base, train, machine = prepare_resume(a, train, machine, overrides, now)
        else:
            name, run_dir, epoch_count, state_path, launch_name, applied, base, train, machine = prepare_new(train, machine, overrides, now)
    except ConfigError as e:
        print(f"[run_train] 設定エラー: {e}", file=sys.stderr)
        sys.exit(2)

    run_dir.mkdir(parents=True, exist_ok=True)
    launch_path = run_dir / launch_name
    argv = ["--launch", str(launch_path), "--run_dir", str(run_dir), "--epoch_count", str(epoch_count)]
    if state_path:
        argv += ["--resume_state", str(state_path)]
    if machine["gpu_gen"] != 0:
        argv.append("--require_cuda")
    with open(launch_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {"stage": STAGE, "timestamp": now.isoformat(timespec="seconds"), "machine_name": a.machine,
             "train": train, "machine": machine,
             "overrides": list(overrides), "applied": applied,
             "resume": a.resume, "resume_tag": a.resume_tag if a.resume else None, "base_launch": str(base) if base else None,
             "argv": argv},
            f, allow_unicode=True, sort_keys=False,
        )
    print(f"[run_train] {STAGE} run {name} → {launch_path}", flush=True)
    if os.environ.get("SPICA_TB_PORT"):
        try:
            start_tensorboard(run_dir, int(os.environ["SPICA_TB_PORT"]))
        except ConfigError as e:
            print(f"[run_train] 設定エラー: {e}", file=sys.stderr)
            sys.exit(2)
    os.chdir(HERE)
    os.execv(sys.executable, [sys.executable, str(HERE / "train.py")] + argv)


if __name__ == "__main__":
    main()
