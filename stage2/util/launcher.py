"""stage2/util/launcher.py — U・V の学習の起動器の本体（run_train_u.py / run_train_v.py が呼ぶ）。流れは stage1/run_train.py と同じ。

  1. data.yaml / u.yaml（または v.yaml）/ configs/shared.yaml / configs/machines.yaml を読み、configs/schema.py で検証（必須キー欠落・未知キー・型違いはエラー。既定値なし）
  2. sh からの上書き（-- の後の --flag value）を反映してから、値域・相互条件と、ディスクの検査（run フォルダを作る前）:
       疑似 EID の組と manifest、明るさの広さの一致、症例の分割、表示用のスライス、確認用の実 EID が train 側でないこと。V は U の run と重み、U と data の一致
  3. run 名 = 起動時刻 yyyy_mmdd_HHMM（JST）。同名があればエラー
  4. 実効値を <flow_checkpoints_dir>/<u|v>/<run>/launch.yaml に保存し、学習ループを exec する:
       python stage2/train_u.py --run_dir /workspace/runs/stage2/u/<run> --launch <run>/launch.yaml
  5. SPICA_TB_PORT があれば、この run の tb/ だけを見る TensorBoard を起動する

再開:  … --resume <run> [--resume_tag latest|best|<step>] -- [--max_steps 60000]
  基準は**選んだ重みの state.pth に入っている実効設定**（data / u|v / shared）。今回の --flag だけが上に乗る。machine（データと run の置き場）は今の machines.yaml。
  ネットワーク（net.*）と、HU の物差し・補間・V の単位・使う U は変えられない。設定は launch_resume_<日時秒>.yaml に残す。
"""

import argparse
import datetime
import os
import sys
from pathlib import Path

import yaml

from configs.schema import (DATA, FROZEN_DATA, FROZEN_U, FROZEN_V, MACHINE, MUST_MATCH_U, SHARED, U, V, ConfigError, apply_overrides, check_u_values, check_v_values,
                            validate, validate_shared)
from data.pair_dataset import check_before_launch
from util.run_paths import JST, LAUNCH_FILE, ckpt_dir, kind_root, missing_files, run_name, saved_tags

STAGE2 = Path(__file__).resolve().parents[1]
SCHEMAS = {"u": U, "v": V}
FROZEN = {"u": FROZEN_U, "v": FROZEN_V}
CHECK = {"u": check_u_values, "v": check_v_values}


def load_yaml(path):
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"設定ファイルが見つかりません: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_machine(machines_path, name):
    machines = load_yaml(machines_path)
    if not isinstance(machines, dict) or name not in machines:
        raise ConfigError(f"machines.yaml にエントリ '{name}' がありません。候補: {list(machines) if isinstance(machines, dict) else '(不正な形式)'}")
    return validate_shared(f"machines.{name}", machines[name], MACHINE)


def start_tensorboard(run_dir, port, tag="run_train"):
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
    tb_dir = Path(run_dir) / "tb"
    tb_dir.mkdir(parents=True, exist_ok=True)
    log = open(Path(run_dir) / "tensorboard.log", "ab")
    subprocess.Popen([exe, "--logdir", str(tb_dir), "--port", str(port), "--bind_all", "--reload_interval", "5"], stdout=log, stderr=log, start_new_session=True)
    print(f"[{tag}] TensorBoard: http://localhost:{port} → {tb_dir}（この run だけ。全 run は bash start2.sh tb）", flush=True)


def sections(kind, data, cfg, shared, machine):
    return {"data": (data, DATA), kind: (cfg, SCHEMAS[kind]), "shared": (shared, SHARED), "machine": (machine, MACHINE)}


def cache_dir(machine):
    return Path(machine["flow_checkpoints_dir"]) / ".case_index"


def check_on_disk(data, cfg, shared, machine):
    try:
        return check_before_launch(data, cfg, shared, machine, cache_dir(machine))
    except RuntimeError as e:
        raise ConfigError(str(e))


def load_u_source(v, machine):
    """V が固定して使う U: run と重みがあること。戻り値 (重みディレクトリ, U の実効設定 {"data", "u", "shared"}, 記録)。"""
    import torch

    run_dir = kind_root(machine, "u") / v["u.run"]
    if not (run_dir / LAUNCH_FILE).is_file():
        raise ConfigError(f"U の run が見つかりません: {run_dir}（--u_run <run>。先に bash start2.sh u で学習する）")
    wdir = ckpt_dir(run_dir, v["u.tag"])
    miss = missing_files(wdir, "u")
    if miss:
        raise ConfigError(f"U の重みが揃っていません: {wdir} に {miss} が無い（保存済み tag: {saved_tags(run_dir, 'u')}。best は U が上限まで学習し終えると出来る。途中の重みを使うなら --u_tag latest か step 番号）")
    st = torch.load(wdir / "state.pth", map_location="cpu", weights_only=True)
    cfg = st.get("config")
    if not isinstance(cfg, dict) or any(k not in cfg for k in ("data", "u", "shared")):
        raise ConfigError(f"{wdir / 'state.pth'} に実効設定がありません")
    return wdir, {k: dict(cfg[k]) for k in ("data", "u", "shared")}, {"run": v["u.run"], "tag": v["u.tag"], "step": int(st["step"]), "weight_dir": str(wdir)}


def check_v_against_u(data, v, shared, u_config):
    """U と V で絵の物差し・補間・分割・明るさの広さが同じこと（違うと μ の意味が変わる）。疑似 EID の組が違うのは注意だけ。"""
    diff = {k: (u_config["data"][k], data[k]) for k in MUST_MATCH_U if u_config["data"].get(k) != data[k]}
    if diff:
        raise ConfigError(f"U を学習したときの data と違います（U の値, 今の値）: {diff}")
    su, sv = float(u_config["shared"]["brightness.sigma_px_512"]), float(shared["brightness.sigma_px_512"])
    if abs(su - sv) > 1e-9:
        raise ConfigError(f"明るさを平均する広さが U を学習したとき（{su:g}）と今の configs/shared.yaml（{sv:g}）で違います。U を学習し直すか shared.yaml を戻す")
    m = 2 ** u_config["u"]["net.n_pool"]
    size = v["patch.size"] + 2 * v["patch.margin"]
    if size % m:
        raise ConfigError(f"patch.size + 2 × patch.margin（{size}）は U の入力の倍数 {m} で割り切れること")
    if u_config["data"]["data.pseudo_eid_set"] != data["data.pseudo_eid_set"]:
        print(f"[run_train_v] 注意: U を学習した疑似 EID の組（{u_config['data']['data.pseudo_eid_set']}）と V の組（{data['data.pseudo_eid_set']}）が違います", flush=True)


def prepare_new(kind, data, cfg, shared, machine, overrides, now):
    applied = apply_overrides(overrides, sections(kind, data, cfg, shared, machine))
    CHECK[kind](data, cfg, shared, machine)
    extra = {}
    if kind == "v":
        wdir, u_config, u_source = load_u_source(cfg, machine)
        check_v_against_u(data, cfg, shared, u_config)
        extra = {"u_weight_dir": str(wdir), "u_source": u_source}
    check_on_disk(data, cfg, shared, machine)
    name = run_name(now)
    run_dir = kind_root(machine, kind) / name
    if run_dir.exists():
        raise ConfigError(f"run フォルダが既にあります: {run_dir}（同じ分の再起動。1 分待つか、不要なら消してから起動）")
    return name, run_dir, None, LAUNCH_FILE, applied, data, cfg, shared, extra


def check_weights_match(kind, wdir, cfg, u_config):
    """重みの各テンソルの形が、実効設定で組んだネットと一致するか（launch を書く前に止める）。"""
    import torch

    from models.flow_net import build_flow
    from models.unet import build_net

    pairs = [("net_U.pth", build_net(cfg))] if kind == "u" else [("net_V.pth", build_flow(cfg)), ("net_U.pth", build_net(u_config["u"]))]
    for fname, net in pairs:
        saved = torch.load(wdir / fname, map_location="cpu", weights_only=True)
        expect = net.state_dict()
        bad = [f"{k}: 重み {tuple(saved[k].shape) if k in saved else '無し'} ≠ 設定 {tuple(v.shape)}" for k, v in expect.items() if k not in saved or tuple(saved[k].shape) != tuple(v.shape)]
        extra = [k for k in saved if k not in expect]
        if bad or extra:
            raise ConfigError(f"{wdir / fname} の形が実効設定と合いません: " + "; ".join(bad[:3] + [f"余分 {k}" for k in extra[:3]]))


def prepare_resume(kind, a, data_yaml, cfg_yaml, shared_yaml, machine, overrides, now):
    import torch

    run_dir = kind_root(machine, kind) / a.resume
    if not run_dir.is_dir():
        raise ConfigError(f"再開する run が見つかりません: {run_dir}")
    wdir = ckpt_dir(run_dir, a.resume_tag)
    miss = missing_files(wdir, kind)
    if miss:
        raise ConfigError(f"再開用の重みが揃っていません: {wdir} に {miss} が無い（保存済み tag: {saved_tags(run_dir, kind)}）")
    state_path = wdir / "state.pth"
    st = torch.load(state_path, map_location="cpu", weights_only=True)
    saved = st.get("config")
    if not isinstance(saved, dict) or any(k not in saved for k in ("data", kind, "shared")):
        raise ConfigError(f"{state_path} に実効設定がありません")
    out = {}
    for sec, cur, schema in (("data", data_yaml, DATA), (kind, cfg_yaml, SCHEMAS[kind]), ("shared", shared_yaml, SHARED)):
        sv = dict(saved[sec])
        new = [k for k in schema if k not in sv]
        if new:
            print(f"[run_train_{kind}] 注意: 重みの {sec} に無いキー（run を作った後に足された）は今の yaml の値を使う: " + ", ".join(f"{k}={cur[k]!r}" for k in new))
            for k in new:
                sv[k] = cur[k]
        gone = [k for k in sv if k not in schema]
        if gone:
            print(f"[run_train_{kind}] 注意: 重みの {sec} にある廃止キーは無視する: {gone}")
        sv = {k: v for k, v in sv.items() if k in schema}
        diff = {k: (sv[k], cur[k]) for k in schema if sv[k] != cur[k]}
        if diff:
            print(f"[run_train_{kind}] 注意: 今の {sec} の yaml は重みの実効設定と違う（重みの値を使う。変えるなら --flag）: {diff}")
        out[sec] = (validate_shared if sec == "shared" else validate)(f"{sec}(state)", sv, schema)
    data, cfg, shared = out["data"], out[kind], out["shared"]

    def frozen():
        f = {f"data:{k}": data[k] for k in FROZEN_DATA}
        f.update({f"{kind}:{k}": cfg[k] for k in cfg if k.startswith("net.") or k in FROZEN[kind]})
        return f

    before = frozen()
    applied = apply_overrides(overrides, sections(kind, data, cfg, shared, machine))
    changed = [k for k, v in frozen().items() if before[k] != v]
    if changed:
        raise ConfigError(f"再開では変えられない設定です（重みの意味が変わる）: {changed}")
    CHECK[kind](data, cfg, shared, machine)
    u_config = saved.get("u_config") if kind == "v" else None
    if kind == "v":
        if not isinstance(u_config, dict):
            raise ConfigError(f"{state_path} に U の実効設定（u_config）がありません")
        check_v_against_u(data, cfg, shared, u_config)
    check_on_disk(data, cfg, shared, machine)
    check_weights_match(kind, wdir, cfg, u_config)
    if st["step"] >= cfg["optim.max_steps"]:
        raise ConfigError(f"run は既に step {st['step']} まで終わっています（optim.max_steps {cfg['optim.max_steps']}）。延ばすなら --max_steps を上書き")
    if st["elapsed_sec"] / 3600.0 >= cfg["optim.max_hours"]:
        raise ConfigError(f"run は既に累計 {st['elapsed_sec'] / 3600:.1f} 時間学習しています（optim.max_hours {cfg['optim.max_hours']}）。延ばすなら --max_hours を上書き")
    print(f"[run_train_{kind}] resume {a.resume} / {a.resume_tag}: step {st['step']}、累計 {st['elapsed_sec'] / 3600:.2f} 時間 → 上限 step {cfg['optim.max_steps']} / {cfg['optim.max_hours']} 時間")
    extra = {"u_source": saved.get("u_source")} if kind == "v" else {}
    return a.resume, run_dir, state_path, f"launch_resume_{now.strftime('%Y%m%d_%H%M%S')}.yaml", applied, data, cfg, shared, extra


def main(kind):
    p = argparse.ArgumentParser(description=f"新 Stage 2 の {kind.upper()} の学習の起動器")
    p.add_argument("--machine", required=True, help="configs/machines.yaml のエントリ名")
    p.add_argument("--data", required=True)
    p.add_argument("--cfg", required=True, help="u.yaml / v.yaml")
    p.add_argument("--shared", required=True)
    p.add_argument("--machines", required=True)
    p.add_argument("--resume", default=None, metavar="RUN")
    p.add_argument("--resume_tag", default="latest", help="latest | best | step 番号")
    p.add_argument("overrides", nargs=argparse.REMAINDER, help="'--' の後に上書き（schema のフラグのみ）")
    a = p.parse_args()
    overrides = a.overrides[1:] if a.overrides and a.overrides[0] == "--" else a.overrides
    tag = f"run_train_{kind}"
    now = datetime.datetime.now(JST)
    try:
        data = validate("data", load_yaml(a.data), DATA)
        cfg = validate(kind, load_yaml(a.cfg), SCHEMAS[kind])
        shared = validate_shared("shared", load_yaml(a.shared), SHARED)
        machine = load_machine(a.machines, a.machine)
        if a.resume:
            name, run_dir, state_path, launch_name, applied, data, cfg, shared, extra = prepare_resume(kind, a, data, cfg, shared, machine, overrides, now)
        else:
            name, run_dir, state_path, launch_name, applied, data, cfg, shared, extra = prepare_new(kind, data, cfg, shared, machine, overrides, now)
    except ConfigError as e:
        print(f"[{tag}] 設定エラー: {e}", file=sys.stderr)
        sys.exit(2)

    run_dir.mkdir(parents=True, exist_ok=True)
    launch_path = run_dir / launch_name
    argv = ["--run_dir", str(run_dir), "--launch", str(launch_path)]
    if state_path:
        argv += ["--resume_state", str(state_path)]
    if machine["gpu_gen"] != 0:
        argv.append("--require_cuda")
    with open(launch_path, "w", encoding="utf-8") as f:
        yaml.safe_dump({"timestamp": now.isoformat(timespec="seconds"), "stage": f"stage2_{kind}", "machine_name": a.machine,
                        "data": data, kind: cfg, "shared": shared, "machine": machine,  # 実効値（yaml + 上書き、平坦キー）
                        **extra, "overrides": list(overrides), "applied": applied,
                        "resume": a.resume, "resume_tag": a.resume_tag if a.resume else None, "argv": argv},
                       f, allow_unicode=True, sort_keys=False)
    print(f"[{tag}] run {name} → {launch_path}", flush=True)
    if os.environ.get("SPICA_TB_PORT"):
        try:
            start_tensorboard(run_dir, int(os.environ["SPICA_TB_PORT"]), tag)
        except ConfigError as e:
            print(f"[{tag}] 設定エラー: {e}", file=sys.stderr)
            sys.exit(2)
    os.chdir(STAGE2)
    os.execv(sys.executable, [sys.executable, str(STAGE2 / f"train_{kind}.py")] + argv)
