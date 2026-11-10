"""stage1/run_train.py — 新 Stage 1 学習の起動器（Docker 内で動く。ホストからは bash start.sh）。流れは old/stage2/run_train.py と同じ。

  1. train.yaml / model.yaml / configs/shared.yaml / configs/machines.yaml を読み、configs/schema.py で検証（必須キー欠落・未知キー・型違いはエラー。既定値なし）
  2. sh からの上書き（-- の後の --flag value）を反映してから、値域・相互条件・症例の分割・固定表示のスライスの有無を検査（run フォルダを作る前）
  3. run 名 = 起動時刻 yyyy_mmdd_HHMM（JST）。同名があればエラー
  4. 実効値を <painter_checkpoints_dir>/<run>/launch.yaml に保存し、train.py を exec する:
       python stage1/train.py --run_dir /workspace/runs/stage1/<run> --launch <run>/launch.yaml
  5. SPICA_TB_PORT があれば、この run の tb/ だけを見る TensorBoard を起動する

再開:  python run_train.py ... --resume <run> [--resume_tag latest|best|<step>] -- [--max_steps 300000]
  基準は**選んだ重みの state.pth に入っている実効設定**（train / model / shared）。今回の --flag だけが上に乗る。machine（データと run の置き場）は今の machines.yaml。
  ネットワーク（model.*）と FROZEN_ON_RESUME のキー（HU の変換、σ の範囲、σ_data、cue.blur_max_px）は変えられない。設定は launch_resume_<日時秒>.yaml に残す。
"""

import argparse
import datetime
import os
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from configs.schema import (FROZEN_ON_RESUME, MACHINE, MODEL, SHARED, TRAIN, ConfigError, apply_overrides, check_train_values,  # noqa: E402
                            validate, validate_shared)
from data.eid_dataset import resolve_split  # noqa: E402
from util.run_paths import CKPT_KINDS, JST, LAUNCH_FILE, ckpt_dir, run_name, saved_tags  # noqa: E402


def start_tensorboard(run_dir, port):
    """この run の tb/ だけを logdir にして TensorBoard を起動する。同じポートで動いている前の TensorBoard は止める。ログは <run>/tensorboard.log。"""
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
    print(f"[run_train] TensorBoard: http://localhost:{port} → {tb_dir}（この run だけ。全 run は bash start.sh tb）", flush=True)


def load_yaml(path):
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"設定ファイルが見つかりません: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def sections(train, model, shared, machine):
    return {"train": (train, TRAIN), "model": (model, MODEL), "shared": (shared, SHARED), "machine": (machine, MACHINE)}


def check_on_disk(train, machine):
    """症例の分割と、途中経過のパネルに使うスライスの有無。run フォルダを作る前に止める（空の run を残さない。何時間も進んでから落ちない）。"""
    try:
        split = resolve_split(train, machine, cache_dir=Path(machine["painter_checkpoints_dir"]) / ".case_index")
    except RuntimeError as e:
        raise ConfigError(str(e))
    missing = [f"{k}: {p}" for k in ("log.eid_slices", "log.pcd_slices") for p in train[k] if not Path(p).is_file()]
    if missing:
        raise ConfigError("途中経過のパネルに使うスライスがありません: " + "; ".join(missing) + "（そのマシンに無いものは --eid_slices / --pcd_slices で外す）")
    learned = [p for p in train["log.eid_slices"] if Path(p).parent.name in set(split["train"])]
    if learned:
        raise ConfigError(f"log.eid_slices に、学習に使う症例（train 側）のスライスがあります: {learned}（val / test 側か評価専用の症例にする）")


def prepare_new(train, model, shared, machine, overrides, now):
    applied = apply_overrides(overrides, sections(train, model, shared, machine))
    check_train_values(train, model, shared, machine)
    check_on_disk(train, machine)
    name = run_name(now)
    run_dir = Path(machine["painter_checkpoints_dir"]) / name
    if run_dir.exists():
        raise ConfigError(f"run フォルダが既にあります: {run_dir}（同じ分の再起動。1 分待つか、不要なら消してから起動）")
    return name, run_dir, None, LAUNCH_FILE, applied, train, model, shared, machine


def check_weights_match(weight_path, model, train):
    """net.pth の各テンソルの形が、実効設定で組んだネットと一致するか（launch を書く前に止める）。"""
    import torch
    from models.unet import build_net

    saved = torch.load(weight_path, map_location="cpu", weights_only=True)
    expect = build_net(model, train).state_dict()
    bad = [f"{k}: 重み {tuple(saved[k].shape) if k in saved else '無し'} ≠ 設定 {tuple(v.shape)}" for k, v in expect.items() if k not in saved or tuple(saved[k].shape) != tuple(v.shape)]
    extra = [k for k in saved if k not in expect]
    if bad or extra:
        raise ConfigError(f"{weight_path} の形が実効設定と合いません: " + "; ".join(bad[:3] + [f"余分 {k}" for k in extra[:3]]))


def prepare_resume(a, train_yaml, model_yaml, shared_yaml, machine, overrides, now):
    import torch

    run_dir = Path(machine["painter_checkpoints_dir"]) / a.resume
    if not run_dir.is_dir():
        raise ConfigError(f"再開する run が見つかりません: {run_dir}")
    wdir = ckpt_dir(run_dir, a.resume_tag)
    missing = [k for k in CKPT_KINDS if not (wdir / k).is_file()]
    if missing:
        raise ConfigError(f"再開用の重みが揃っていません: {wdir} に {missing} が無い（保存済み tag: {saved_tags(run_dir)}）")
    state_path = wdir / "state.pth"
    st = torch.load(state_path, map_location="cpu", weights_only=True)
    saved = st.get("config")
    if not isinstance(saved, dict) or any(k not in saved for k in ("train", "model", "shared")):
        raise ConfigError(f"{state_path} に実効設定がありません")
    out = {}
    for sec, cur, schema in (("train", train_yaml, TRAIN), ("model", model_yaml, MODEL), ("shared", shared_yaml, SHARED)):
        sv = dict(saved[sec])
        new = [k for k in schema if k not in sv]
        if new:
            print(f"[run_train] 注意: 重みの {sec} に無いキー（run を作った後に足された）は今の yaml の値を使う: " + ", ".join(f"{k}={cur[k]!r}" for k in new))
            for k in new:
                sv[k] = cur[k]
        gone = [k for k in sv if k not in schema]
        if gone:
            print(f"[run_train] 注意: 重みの {sec} にある廃止キーは無視する: {gone}")
        sv = {k: v for k, v in sv.items() if k in schema}
        diff = {k: (sv[k], cur[k]) for k in schema if sv[k] != cur[k]}
        if diff:
            print(f"[run_train] 注意: 今の {sec} の yaml は重みの実効設定と違う（重みの値を使う。変えるなら --flag）: {diff}")
        out[sec] = (validate if sec != "shared" else validate_shared)(f"{sec}(state)", sv, schema)
    train, model, shared = out["train"], out["model"], out["shared"]
    frozen = {k: train[k] for k in FROZEN_ON_RESUME}
    frozen.update({f"model.{k}": v for k, v in model.items()})
    applied = apply_overrides(overrides, sections(train, model, shared, machine))
    after = {k: train[k] for k in FROZEN_ON_RESUME}
    after.update({f"model.{k}": v for k, v in model.items()})
    changed = [k for k in frozen if frozen[k] != after[k]]
    if changed:
        raise ConfigError(f"再開では変えられない設定です（重みの意味が変わる）: {changed}")
    check_train_values(train, model, shared, machine)
    check_on_disk(train, machine)
    check_weights_match(wdir / "net.pth", model, train)
    if st["step"] >= train["optim.max_steps"]:
        raise ConfigError(f"run は既に step {st['step']} まで終わっています（optim.max_steps {train['optim.max_steps']}）。延ばすなら --max_steps を上書き")
    if st["elapsed_sec"] / 3600.0 >= train["optim.max_hours"]:
        raise ConfigError(f"run は既に累計 {st['elapsed_sec'] / 3600:.1f} 時間学習しています（optim.max_hours {train['optim.max_hours']}）。延ばすなら --max_hours を上書き")
    print(f"[run_train] resume {a.resume} / {a.resume_tag}: step {st['step']}、累計 {st['elapsed_sec'] / 3600:.2f} 時間 → 上限 step {train['optim.max_steps']} / {train['optim.max_hours']} 時間")
    return a.resume, run_dir, state_path, f"launch_resume_{now.strftime('%Y%m%d_%H%M%S')}.yaml", applied, train, model, shared, machine


def main():
    p = argparse.ArgumentParser(description="新 Stage 1 学習の起動器")
    p.add_argument("--machine", required=True, help="configs/machines.yaml のエントリ名")
    p.add_argument("--train", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--shared", required=True)
    p.add_argument("--machines", required=True)
    p.add_argument("--resume", default=None, metavar="RUN")
    p.add_argument("--resume_tag", default="latest", help="latest | best | step 番号")
    p.add_argument("overrides", nargs=argparse.REMAINDER, help="'--' の後に上書き（schema のフラグのみ）")
    a = p.parse_args()
    overrides = a.overrides[1:] if a.overrides and a.overrides[0] == "--" else a.overrides

    now = datetime.datetime.now(JST)
    try:
        train = validate("train", load_yaml(a.train), TRAIN)
        model = validate("model", load_yaml(a.model), MODEL)
        shared = validate_shared("shared", load_yaml(a.shared), SHARED)
        machines = load_yaml(a.machines)
        if not isinstance(machines, dict) or a.machine not in machines:
            raise ConfigError(f"machines.yaml にエントリ '{a.machine}' がありません。候補: {list(machines) if isinstance(machines, dict) else '(不正な形式)'}")
        machine = validate_shared(f"machines.{a.machine}", machines[a.machine], MACHINE)
        if a.resume:
            name, run_dir, state_path, launch_name, applied, train, model, shared, machine = prepare_resume(a, train, model, shared, machine, overrides, now)
        else:
            name, run_dir, state_path, launch_name, applied, train, model, shared, machine = prepare_new(train, model, shared, machine, overrides, now)
    except ConfigError as e:
        print(f"[run_train] 設定エラー: {e}", file=sys.stderr)
        sys.exit(2)

    run_dir.mkdir(parents=True, exist_ok=True)
    launch_path = run_dir / launch_name
    argv = ["--run_dir", str(run_dir), "--launch", str(launch_path)]
    if state_path:
        argv += ["--resume_state", str(state_path)]
    if machine["gpu_gen"] != 0:
        argv.append("--require_cuda")
    with open(launch_path, "w", encoding="utf-8") as f:
        yaml.safe_dump({"timestamp": now.isoformat(timespec="seconds"), "stage": "stage1_painter", "machine_name": a.machine,
                        "train": train, "model": model, "shared": shared, "machine": machine,  # 実効値（yaml + 上書き、平坦キー）
                        "overrides": list(overrides), "applied": applied,
                        "resume": a.resume, "resume_tag": a.resume_tag if a.resume else None, "argv": argv},
                       f, allow_unicode=True, sort_keys=False)
    print(f"[run_train] run {name} → {launch_path}", flush=True)
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
