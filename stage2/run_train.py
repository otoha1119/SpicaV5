"""stage2/run_train.py — Stage 2 学習の起動器（Docker 内で動く）。Stage 1 の run_train.py と同じ流れ。

  1. configs/train.yaml, configs/mode.yaml, ../configs/machines.yaml を読み、configs/schema.py で検証する（必須キー欠落・未知キー・型違いはエラー。既定値は無い）
  2. sh から渡された上書き引数を平坦 dict に反映してから（後勝ち。schema に無いフラグはエラー）、値域・相互条件を検証する
  3. 実験名を起動時刻 yyyy_mmdd_HHMM（JST 固定）から生成する。同名の run が既にあればエラー
  4. 解決済み設定（実効値）を <stage2_checkpoints_dir>/<run>/launch.yaml に保存し、**そのファイルを train.py に渡して** exec する
     （Stage 1 は junyanz の argparse に合わせて全引数を argv にしたが、Stage 2 の train.py は自前なので launch.yaml を読む。症例リストも渡せる）
  5. start2.sh 経由（SPICA_TB_PORT あり）なら、この run の tb/ を logdir にして TensorBoard を起動する

使い方（通常は ../train_stage2.sh 経由）:
  python run_train.py --machine PC1 --train configs/train.yaml --mode configs/mode.yaml --machines ../configs/machines.yaml -- --n_epochs 50
再開:
  python run_train.py --machine PC1 ... --resume 2026_0908_2130 [--resume_tag latest|best|30] -- [--n_epochs 200]
  → **選んだ checkpoint の state.pth に入っている実効設定**（その重みを作った設定）を基準に、今回の上書きを反映し、state.pth（optimizer / RNG / 進捗）と
    net_G.pth を復元して同じディレクトリに続きを保存する。設定は launch_resume_<日時秒>.yaml に保存される（記録用。次の再開の基準は checkpoint 側）。
    重みの形が実効設定と合わなければ launch を書く前に止める（レビュー指摘 2026-09-09: 失敗した起動の設定を次回に引き継がない）
"""

import argparse
import datetime
import os
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from configs.schema import MACHINE, MODE, TRAIN, ConfigError, apply_overrides, check_train_values, validate, validate_shared  # noqa: E402
from data.pair_dataset import check_split_on_disk  # noqa: E402
from util.run_paths import LAUNCH_FILE, ckpt_path, latest_launch, run_name, saved_tags  # noqa: E402


def start_tensorboard(run_dir, port):
    """この run の tb/ だけを logdir にして TensorBoard を起動する。**同じポート（stage2_tb_port）**で動いている前の TensorBoard は止め、Stage 1 のポートのものは触らない。ログは <run>/tensorboard.log。"""
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
    print(f"[run_train] TensorBoard: http://localhost:{port} → {tb_dir}（この run だけ。全 run は bash start2.sh tb）", flush=True)


def load_yaml(path):
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"設定ファイルが見つかりません: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def sections(train, mode, machine):
    return {"train": (train, TRAIN), "mode": (mode, MODE), "machine": (machine, MACHINE)}


def _check_split(train, machine):
    """症例リストとディスクの照合（未割当・欠落）。run ディレクトリを作る前に止める（空の run を残さない）。"""
    try:
        check_split_on_disk(train, machine)
    except RuntimeError as e:
        raise ConfigError(str(e))


def prepare_new(train, mode, machine, overrides, now):
    applied = apply_overrides(overrides, sections(train, mode, machine))
    check_train_values(train, mode, machine)
    _check_split(train, machine)
    name = run_name(now)
    run_dir = Path(machine["stage2_checkpoints_dir"]) / name
    if run_dir.exists():
        raise ConfigError(f"run ディレクトリが既に存在します: {run_dir}（同一分の再起動。1 分待つか、不要なら消してから起動）")
    return name, run_dir, 1, None, LAUNCH_FILE, applied, None, train, mode, machine


def check_weights_match(weight_path, mode):
    """net_G.pth の各テンソルの形が、実効 mode で組んだネットと一致するか（--base_ch 等を変えた再開をここで止める。launch を書く前）。"""
    import torch
    from models.unet import build_net

    saved = torch.load(weight_path, map_location="cpu", weights_only=True)
    expect = build_net(mode).state_dict()
    bad = [f"{k}: 重み {tuple(saved[k].shape) if k in saved else '無し'} ≠ 設定 {tuple(v.shape)}" for k, v in expect.items() if k not in saved or tuple(saved[k].shape) != tuple(v.shape)]
    extra = [k for k in saved if k not in expect]
    if bad or extra:
        raise ConfigError(f"{weight_path} の形が実効設定（arch / base_ch / n_pool）と合いません: " + "; ".join(bad[:3] + [f"余分 {k}" for k in extra[:3]]) + ("" if len(bad) + len(extra) <= 3 else " ..."))


def prepare_resume(a, train_yaml, mode_yaml, machine_yaml, overrides, now):
    """再開の基準は**選んだ checkpoint の state.pth に入っている実効設定**（その重みを作った設定）。失敗した起動の launch_resume_*.yaml は基準にならない。
    古い run（state に config が無い）だけ最新 launch を基準にする。"""
    import torch  # state.pth を読む

    probe = (dict(train_yaml), dict(mode_yaml), dict(machine_yaml))
    apply_overrides(overrides, sections(*probe))
    run_dir = Path(probe[2]["stage2_checkpoints_dir"]) / a.resume
    if not run_dir.is_dir():
        raise ConfigError(f"再開する run が見つかりません: {run_dir}")
    state_path = ckpt_path(run_dir, a.resume_tag, "state.pth")
    weight_path = ckpt_path(run_dir, a.resume_tag, "net_G.pth")
    if not state_path.is_file():
        raise ConfigError(f"再開用 state がありません: {state_path}（保存済み tag: {saved_tags(run_dir)}）")
    if not weight_path.is_file():
        raise ConfigError(f"再開用の重みがありません: {weight_path}")
    st = torch.load(state_path, map_location="cpu", weights_only=True)
    if isinstance(st.get("config"), dict) and all(k in st["config"] for k in ("train", "mode", "machine")):
        saved, base = {k: dict(st["config"][k]) for k in ("train", "mode", "machine")}, state_path
    else:
        base = latest_launch(run_dir)
        if not base.is_file():
            raise ConfigError(f"{state_path} に設定が無く、{run_dir / LAUNCH_FILE} もありません")
        print(f"[run_train] 注意: {state_path.name} に実効設定が無い古い run なので、最新の launch（{base.name}）を基準にする")
        saved = load_yaml(base)
    try:
        for section, cur, schema in (("train", train_yaml, TRAIN), ("mode", mode_yaml, MODE), ("machine", machine_yaml, MACHINE)):
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
        mode = validate("mode(run)", saved["mode"], MODE)
        machine = validate_shared("machine(run)", saved["machine"], MACHINE)
    except (KeyError, TypeError) as e:
        raise ConfigError(f"{base} の形式が不正です: {e}")
    for section, cur, sv in (("train", train_yaml, train), ("mode", mode_yaml, mode), ("machine", machine_yaml, machine)):
        diff = {k: (sv.get(k), v) for k, v in cur.items() if sv.get(k) != v}
        if diff:
            print(f"[run_train] 注意: 今の {section} 設定は checkpoint の実効設定と異なる（checkpoint の値を使う。変えるなら --flag で上書き）: {diff}")
    applied = apply_overrides(overrides, sections(train, mode, machine))
    check_train_values(train, mode, machine)
    _check_split(train, machine)
    if Path(machine["stage2_checkpoints_dir"]) / a.resume != run_dir:
        raise ConfigError(f"再開時に stage2_checkpoints_dir を変えることはできません: {machine['stage2_checkpoints_dir']} != {run_dir.parent}")
    check_weights_match(weight_path, mode)  # launch を書く前に、重みと設定の食い違いを止める
    epoch_count = st["epoch"] + 1 if st["epoch_done"] else st["epoch"]  # epoch 末保存なら次の epoch から、途中保存ならその epoch を頭から（残り batch だけの厳密な再開ではない）
    if epoch_count > train["optim.n_epochs"]:
        raise ConfigError(f"run は既に epoch {st['epoch']} まで終わっています（n_epochs {train['optim.n_epochs']}）。延ばすなら --n_epochs を上書き")
    print(f"[run_train] resume {a.resume} from tag '{a.resume_tag}' (base: {base.name if base != state_path else 'state.pth の config'}): saved epoch {st['epoch']} (done={st['epoch_done']}), total_iters {st['total_iters']} → epoch_count {epoch_count}, n_epochs {train['optim.n_epochs']}")
    return a.resume, run_dir, epoch_count, state_path, f"launch_resume_{now.strftime('%Y%m%d_%H%M%S')}.yaml", applied, base, train, mode, machine


def main():
    p = argparse.ArgumentParser(description="Stage 2 学習の起動器")
    p.add_argument("--machine", required=True, help="configs/machines.yaml のエントリ名（PC1, mac, ...）")
    p.add_argument("--train", required=True, help="学習パラメータ YAML")
    p.add_argument("--mode", required=True, help="モード切替 YAML")
    p.add_argument("--machines", required=True, help="マシン定義 YAML")
    p.add_argument("--resume", default=None, metavar="RUN", help="再開する run 名（yyyy_mmdd_HHMM）")
    p.add_argument("--resume_tag", default="latest", help="再開に使う checkpoint の tag（latest | best | epoch 番号）")
    p.add_argument("overrides", nargs=argparse.REMAINDER, help="'--' の後に上書き（schema のフラグのみ。bool は --flag true|false、list は --flag a,b,c）")
    a = p.parse_args()
    overrides = a.overrides[1:] if a.overrides and a.overrides[0] == "--" else a.overrides

    jst = datetime.timezone(datetime.timedelta(hours=9), name="JST")
    now = datetime.datetime.now(jst)
    try:
        train = validate("train", load_yaml(a.train), TRAIN)
        mode = validate("mode", load_yaml(a.mode), MODE)
        machines = load_yaml(a.machines)
        if not isinstance(machines, dict) or a.machine not in machines:
            raise ConfigError(f"machines.yaml にエントリ '{a.machine}' がありません。候補: {list(machines) if isinstance(machines, dict) else '(不正な形式)'}")
        machine = validate_shared(f"machines.{a.machine}", machines[a.machine], MACHINE)
        if a.resume:
            name, run_dir, epoch_count, state_path, launch_name, applied, base, train, mode, machine = prepare_resume(a, train, mode, machine, overrides, now)
        else:
            name, run_dir, epoch_count, state_path, launch_name, applied, base, train, mode, machine = prepare_new(train, mode, machine, overrides, now)
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
            {"timestamp": now.isoformat(timespec="seconds"), "machine_name": a.machine,
             "train": train, "mode": mode, "machine": machine,  # 実効値（yaml + 上書き、平坦キー）。再開はこれを読む
             "overrides": list(overrides), "applied": applied,
             "resume": a.resume, "resume_tag": a.resume_tag if a.resume else None, "base_launch": str(base) if base else None,
             "argv": argv},
            f, allow_unicode=True, sort_keys=False,
        )
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
