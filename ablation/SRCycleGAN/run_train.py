"""ablation/SRCycleGAN/run_train.py — SR-CycleGAN 学習の起動器（Docker 内で動く）。stage2/run_train.py の流儀。

  1. configs/train.yaml と ../../configs/machines.yaml（使うキーだけ）を configs/schema.py で検証する（既定値なし）
  2. sh の上書き（--flag value）を実効値に反映 → 値域検査 → ディレクトリ・split・固定スライスの存在を確認（run ディレクトリを作る前に止める）
  3. run 名 = 起動時刻 yyyy_mmdd_HHMM（JST）。保存先 <ablation_checkpoints_dir>/SRCycleGAN/<run>/。同名があればエラー
  4. 実効値を launch.yaml に保存し、train.py にそのファイルを渡して exec する
  5. start_ablation.sh 経由（SPICA_TB_PORT あり）なら、この run の tb/ を logdir にして TensorBoard を起動する

再開: --resume <run> [--resume_tag latest|best|<epoch>]。基準は**選んだ checkpoint の state.pth に入っている実効設定**。重みの形を決めるキー（scale / n_blocks / ngf / ndf / norm / n_layers_D）は変更不可。
"""

import argparse
import datetime
import os
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from configs.schema import MACHINE, TRAIN, ConfigError, apply_overrides, check_split, check_train_values, validate, validate_shared  # noqa: E402
from util.run_paths import ABLATION, CKPT_KINDS, LAUNCH_FILE, checkpoints_root, ckpt_path, latest_launch, run_name, saved_tags  # noqa: E402

REPO_ROOT = HERE.parents[1]  # <repo>（split_file の相対パスの基準。コンテナでは /workspace）
SHAPE_KEYS = ("network.scale", "network.n_blocks", "network.ngf", "network.ndf", "network.norm", "network.n_layers_D")  # 重みの形。再開で変えられない


def load_yaml(path):
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"設定ファイルが見つかりません: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def start_tensorboard(run_dir, port):
    """この run の tb/ だけを logdir にして TensorBoard を起動する。同じポート（ablation_tb_port）の前のものは止める。"""
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
    print(f"[run_train] TensorBoard: http://localhost:{port} → {tb_dir}（この run だけ。全 run は bash start_ablation.sh <マシン> {ABLATION} tb）", flush=True)


def sections(train, machine):
    return {"train": (train, TRAIN), "machine": (machine, MACHINE)}


def resolve_split(train):
    """split_file をリポジトリ直下基準の絶対パスにし、中身を検査する。"""
    sf = Path(train["data.split_file"])
    sf = sf if sf.is_absolute() else REPO_ROOT / sf
    if not sf.is_file():
        raise ConfigError(f"data.split_file がありません: {sf}（Stage 1 の bash start.sh split で作る）")
    check_split(load_yaml(sf), f"split_file {sf}")
    train["data.split_file"] = str(sf)


def check_paths(train, machine):
    """run ディレクトリを作る前に、データディレクトリ・固定スライス・代表スライスの存在を確認する（空の run を残さない）。症例単位の検査は dataset が起動時に行う。"""
    for k in ("eid_dir", "pcd1024_dir"):
        if not Path(machine[k]).is_dir():
            raise ConfigError(f"machines.yaml の {k} がありません: {machine[k]}")
    if not (Path(machine["eid_dir"]) / train["log.full_slice"]).is_file():
        raise ConfigError(f"log.full_slice が eid_dir の下にありません: {Path(machine['eid_dir']) / train['log.full_slice']}")
    if not (Path(machine["pcd1024_dir"]) / train["log.pcd_slice"]).is_file():
        raise ConfigError(f"log.pcd_slice が pcd1024_dir の下にありません: {Path(machine['pcd1024_dir']) / train['log.pcd_slice']}")


def prepare_new(train, machine, overrides, now):
    applied = apply_overrides(overrides, sections(train, machine))
    check_train_values(train, machine)
    resolve_split(train)
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
    resolve_split(train)
    check_paths(train, machine)
    if checkpoints_root(machine) / a.resume != run_dir:
        raise ConfigError(f"再開時に ablation_checkpoints_dir を変えることはできません: {checkpoints_root(machine)} != {run_dir.parent}")
    epoch_count = st["epoch"] + 1 if st["epoch_done"] else st["epoch"]
    total = train["optim.n_epochs"] + train["optim.n_epochs_decay"]
    if epoch_count > total:
        raise ConfigError(f"run は既に epoch {st['epoch']} まで終わっています（n_epochs + n_epochs_decay = {total}）。延ばすなら --n_epochs_decay 等を上書き")
    print(f"[run_train] resume {a.resume} from tag '{a.resume_tag}' (base: {'state.pth の config' if base == state_path else base.name}): saved epoch {st['epoch']} (done={st['epoch_done']}), total_iters {st['total_iters']} → epoch_count {epoch_count} / {total}")
    return a.resume, run_dir, epoch_count, state_path, f"launch_resume_{now.strftime('%Y%m%d_%H%M%S')}.yaml", applied, base, train, machine


def main():
    p = argparse.ArgumentParser(description=f"ablation {ABLATION} 学習の起動器")
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
            {"ablation": ABLATION, "timestamp": now.isoformat(timespec="seconds"), "machine_name": a.machine,
             "train": train, "machine": machine,
             "overrides": list(overrides), "applied": applied,
             "resume": a.resume, "resume_tag": a.resume_tag if a.resume else None, "base_launch": str(base) if base else None,
             "argv": argv},
            f, allow_unicode=True, sort_keys=False,
        )
    print(f"[run_train] {ABLATION} run {name} → {launch_path}", flush=True)
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
