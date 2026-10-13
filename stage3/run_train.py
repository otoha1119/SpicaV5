"""stage3/run_train.py — Stage 3（条件付き DDIM）学習の起動器（Docker 内で動く）。stage2/run_train.py から 2026-10-13 に複製し、mode.yaml の代わりに ddim.yml を読む。

  1. configs/ddim.yml, configs/train.yaml, ../configs/machines.yaml を読み、configs/schema.py で検証する（必須キー欠落・未知キー・型違いはエラー。既定値は無い）
  2. sh から渡された上書き引数を反映してから（後勝ち。schema に無いフラグはエラー）、値域・相互条件を検証する
  3. 実験名を起動時刻 yyyy_mmdd_HHMM（JST）から作る。同名の run があればエラー
  4. 実効値を <stage3_checkpoints_dir>/<run>/launch.yaml に保存し、そのファイルを train.py に渡して exec する
  5. start3.sh 経由（SPICA_TB_PORT あり）なら、この run の tb/ を logdir にして TensorBoard を起動する

再開:
  python run_train.py ... --resume 2026_1013_2130 [--resume_tag latest|best|<step>] -- [--n_iters 2000000]
  → 選んだ checkpoint の state.pth に入っている実効設定を基準に、今回の上書きを反映して続きを学習する（stage2 と同じ）。
    重み（model.pth）の形が実効設定と合わなければ launch を書く前に止める。
"""

import argparse
import datetime
import os
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from configs.schema import DDIM, MACHINE, TRAIN, ConfigError, apply_overrides, check_train_values, validate, validate_shared  # noqa: E402
from data.pair_dataset import check_split_on_disk  # noqa: E402
from util.run_paths import CKPT_KINDS, CKPT_MODEL, CKPT_STATE, LAUNCH_FILE, ckpt_path, latest_launch, run_name, saved_tags  # noqa: E402

SECTIONS = (("ddim", DDIM), ("train", TRAIN), ("machine", MACHINE))


def start_tensorboard(run_dir, port):
    """この run の tb/ だけを logdir にして TensorBoard を起動する。同じポート（stage3_tb_port）で動いている前の TensorBoard は止め、他の Stage のポートは触らない。"""
    import shutil
    import signal
    import subprocess

    exe = shutil.which("tensorboard") or (str(Path(sys.executable).parent / "tensorboard") if (Path(sys.executable).parent / "tensorboard").exists() else None)
    if exe is None:
        raise ConfigError("tensorboard が見つかりません（docker/requirements-*.txt に tensorboard があるか確認）")
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
    subprocess.Popen([exe, "--logdir", str(tb_dir), "--port", str(port), "--bind_all", "--reload_interval", "5"],
                     stdout=log, stderr=log, start_new_session=True)
    print(f"[run_train] TensorBoard: http://localhost:{port} → {tb_dir}（この run だけ。全 run は bash start3.sh tb）", flush=True)


def load_yaml(path):
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"設定ファイルが見つかりません: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def sections(ddim, train, machine):
    return {"ddim": (ddim, DDIM), "train": (train, TRAIN), "machine": (machine, MACHINE)}


def _check_split(train, machine):
    try:
        check_split_on_disk(train, machine)
    except RuntimeError as e:
        raise ConfigError(str(e))


def prepare_new(ddim, train, machine, overrides, now):
    applied = apply_overrides(overrides, sections(ddim, train, machine))
    check_train_values(train, ddim, machine)
    _check_split(train, machine)
    name = run_name(now)
    run_dir = Path(machine["stage3_checkpoints_dir"]) / name
    if run_dir.exists():
        raise ConfigError(f"run ディレクトリが既に存在します: {run_dir}（同一分の再起動。1 分待つか、不要なら消してから起動）")
    return name, run_dir, None, LAUNCH_FILE, applied, None, ddim, train, machine


def check_weights_match(weight_path, ddim):
    """model.pth の各テンソルの形が、実効 ddim で組んだネットと一致するか（--ch 等を変えた再開をここで止める。launch を書く前）。"""
    import torch
    from models.diffusion import Model
    from runners.diffusion import build_config

    saved = torch.load(weight_path, map_location="cpu", weights_only=True)
    expect = {"module." + k: v for k, v in Model(build_config(ddim, "cpu")).state_dict().items()}  # DataParallel のキー
    bad = [f"{k}: 重み {tuple(saved[k].shape) if k in saved else '無し'} ≠ 設定 {tuple(v.shape)}" for k, v in expect.items() if k not in saved or tuple(saved[k].shape) != tuple(v.shape)]
    extra = [k for k in saved if k not in expect]
    if bad or extra:
        raise ConfigError(f"{weight_path} の形が実効設定（ddim.yml の model.* / data.image_size）と合いません: " + "; ".join(bad[:3] + [f"余分 {k}" for k in extra[:3]]) + ("" if len(bad) + len(extra) <= 3 else " ..."))


def prepare_resume(a, ddim_yaml, train_yaml, machine_yaml, overrides, now):
    """再開の基準は選んだ checkpoint の state.pth に入っている実効設定（stage2 と同じ）。"""
    import torch

    probe = (dict(ddim_yaml), dict(train_yaml), dict(machine_yaml))
    apply_overrides(overrides, sections(*probe))
    run_dir = Path(probe[2]["stage3_checkpoints_dir"]) / a.resume
    if not run_dir.is_dir():
        raise ConfigError(f"再開する run が見つかりません: {run_dir}")
    try:
        paths = {k: ckpt_path(run_dir, a.resume_tag, k) for k in CKPT_KINDS}
    except ValueError as e:
        raise ConfigError(str(e))
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise ConfigError(f"再開用の checkpoint が揃っていません: {missing}（保存済み tag: {saved_tags(run_dir)}）")
    st = torch.load(paths[CKPT_STATE], map_location="cpu", weights_only=True)
    if not (isinstance(st.get("config"), dict) and all(k in st["config"] for k, _ in SECTIONS)):
        raise ConfigError(f"{paths[CKPT_STATE]} に実効設定がありません")
    saved = {k: dict(st["config"][k]) for k, _ in SECTIONS}
    try:
        for (section, schema), cur in zip(SECTIONS, (ddim_yaml, train_yaml, machine_yaml)):
            missing_k = [k for k in schema if k not in saved[section]]
            if missing_k:
                print(f"[run_train] 注意: run の {section} に無いキー（run 作成後に追加）は今の設定の値を使う: " + ", ".join(f"{k}={cur[k]!r}" for k in missing_k))
                for k in missing_k:
                    saved[section][k] = cur[k]
            obsolete = [k for k in saved[section] if k not in schema]
            if obsolete:
                print(f"[run_train] 注意: run の {section} にある廃止キーは無視する: " + ", ".join(f"{k}={saved[section][k]!r}" for k in obsolete))
                for k in obsolete:
                    del saved[section][k]
        ddim = validate("ddim(run)", saved["ddim"], DDIM)
        train = validate("train(run)", saved["train"], TRAIN)
        machine = validate_shared("machine(run)", saved["machine"], MACHINE)
    except (KeyError, TypeError) as e:
        raise ConfigError(f"{paths[CKPT_STATE]} の設定の形式が不正です: {e}")
    for (section, _), cur, sv in zip(SECTIONS, (ddim_yaml, train_yaml, machine_yaml), (ddim, train, machine)):
        diff = {k: (sv.get(k), v) for k, v in cur.items() if sv.get(k) != v}
        if diff:
            print(f"[run_train] 注意: 今の {section} 設定は checkpoint の実効設定と異なる（checkpoint の値を使う。変えるなら --flag で上書き）: {diff}")
    applied = apply_overrides(overrides, sections(ddim, train, machine))
    check_train_values(train, ddim, machine)
    _check_split(train, machine)
    if Path(machine["stage3_checkpoints_dir"]) / a.resume != run_dir:
        raise ConfigError(f"再開時に stage3_checkpoints_dir を変えることはできません: {machine['stage3_checkpoints_dir']} != {run_dir.parent}")
    check_weights_match(paths[CKPT_MODEL], ddim)
    if st["step"] >= ddim["training.n_iters"]:
        raise ConfigError(f"run は既に step {st['step']} まで終わっています（n_iters {ddim['training.n_iters']}）。延ばすなら --n_iters を上書き")
    print(f"[run_train] resume {a.resume} from tag '{a.resume_tag}': step {st['step']} → n_iters {ddim['training.n_iters']}")
    return a.resume, run_dir, paths[CKPT_STATE], f"launch_resume_{now.strftime('%Y%m%d_%H%M%S')}.yaml", applied, paths[CKPT_STATE], ddim, train, machine


def main():
    p = argparse.ArgumentParser(description="Stage 3 学習の起動器")
    p.add_argument("--machine", required=True, help="configs/machines.yaml のエントリ名（PC1, mac, ...）")
    p.add_argument("--ddim", required=True, help="拡散モデルの設定 YAML（本家 church.yml を写したもの）")
    p.add_argument("--train", required=True, help="学習パラメータ YAML")
    p.add_argument("--machines", required=True, help="マシン定義 YAML")
    p.add_argument("--resume", default=None, metavar="RUN", help="再開する run 名（yyyy_mmdd_HHMM）")
    p.add_argument("--resume_tag", default="latest", help="再開に使う checkpoint の tag（latest | best | step 番号）")
    p.add_argument("overrides", nargs=argparse.REMAINDER, help="'--' の後に上書き（schema のフラグのみ）")
    a = p.parse_args()
    overrides = a.overrides[1:] if a.overrides and a.overrides[0] == "--" else a.overrides

    jst = datetime.timezone(datetime.timedelta(hours=9), name="JST")
    now = datetime.datetime.now(jst)
    try:
        ddim = validate("ddim", load_yaml(a.ddim), DDIM)
        train = validate("train", load_yaml(a.train), TRAIN)
        machines = load_yaml(a.machines)
        if not isinstance(machines, dict) or a.machine not in machines:
            raise ConfigError(f"machines.yaml にエントリ '{a.machine}' がありません。候補: {list(machines) if isinstance(machines, dict) else '(不正な形式)'}")
        machine = validate_shared(f"machines.{a.machine}", machines[a.machine], MACHINE)
        if a.resume:
            name, run_dir, state_path, launch_name, applied, base, ddim, train, machine = prepare_resume(a, ddim, train, machine, overrides, now)
        else:
            name, run_dir, state_path, launch_name, applied, base, ddim, train, machine = prepare_new(ddim, train, machine, overrides, now)
    except ConfigError as e:
        print(f"[run_train] 設定エラー: {e}", file=sys.stderr)
        sys.exit(2)

    run_dir.mkdir(parents=True, exist_ok=True)
    launch_path = run_dir / launch_name
    argv = ["--launch", str(launch_path), "--run_dir", str(run_dir)]
    if state_path:
        argv += ["--resume_state", str(state_path)]
    if machine["gpu_gen"] != 0:
        argv.append("--require_cuda")
    with open(launch_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {"timestamp": now.isoformat(timespec="seconds"), "machine_name": a.machine,
             "ddim": ddim, "train": train, "machine": machine,  # 実効値（平坦キー）
             "overrides": list(overrides), "applied": applied,
             "resume": a.resume, "resume_tag": a.resume_tag if a.resume else None, "base": str(base) if base else None,
             "argv": argv},
            f, allow_unicode=True, sort_keys=False,
        )
    print(f"[run_train] run {name} → {launch_path}（latest は {latest_launch(run_dir).name}）", flush=True)
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
