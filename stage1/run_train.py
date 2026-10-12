"""run_train.py — Stage 1 学習の起動器（Docker 内で動く）。

設計: docs/plans/20260905_config-design-plan.md、修正: docs/plans/20260907_review-fix-list.md（F-02 / F-03 / F-07 / F-10 / F-11）
  1. configs/train.yaml, configs/mode.yaml, ../configs/machines.yaml を読み、configs/schema.py で検証する
     （必須キー欠落・未知キー・型違いはエラー。既定値は無い）
  1'. mode.yaml の task（sh の --task で上書き可）が sr なら configs/sr.yaml（--sr）も読む [2026-10-12]:
     SR のキー（split_file / b_exclude_cases）を検証し、`train:` セクションで train.yaml を差し替え（優先順位 train.yaml → sr.yaml → --flag）、
     A = eid1024_dir / B = pcd1024_dir の存在と eid1024_dir/manifest.yaml（補間の方式）を確認して launch の sr に記録する。degrade では sr.yaml を読まない
  2. sh から渡された上書き引数を **平坦 dict に反映**してから（後勝ち。schema に無いフラグはエラー）、値域・相互条件を検証する
  3. schema の順に train.py（junyanz 本家）の引数列を**全パラメータ明示**で組む（実効値から。argv に生の上書きは足さない）
  4. 実験名 --name を起動時刻 yyyy_mmdd_HHMM（JST 固定。コンテナは UTC のため）から生成する。同名の run が既にあればエラー（F-11）
  5. 解決済み設定（実効値）を <checkpoints_dir>/<run>/launch.yaml に保存してから train.py を exec する。
     推論（run_infer.py）と再開はこの実効値を読むので、上書きした G の構造や HU 正規化が引き継がれる

使い方（通常は ../train_stage1.sh 経由）:
  python run_train.py --machine PC1 --train configs/train.yaml --mode configs/mode.yaml --machines ../configs/machines.yaml -- --n_epochs 50
再開:
  python run_train.py --machine PC1 ... --resume 2026_0905_1234 [--resume_tag latest|best|30] -- [--n_epochs 400] [--lr 0.0001]
  → **選んだ checkpoint の state.pth に入っている実効設定**（その重みを作った設定。古い run は最新の launch）を基準に、今回の上書きを反映し、
    重みディレクトリ（<run>/latest | best | weights/epoch_NNN）の state.pth（optimizer / RNG / 進捗）と net_G/D.pth を復元して同じディレクトリに続きを保存する。
    設定は launch_resume_<日時秒>.yaml に保存され、次の再開・推論の基準になる。
    residual / task / d_pack は checkpoint から変えられない（重みの意味・形が変わる）。d_pack は net_D.pth の 1 層目の入力チャネルとも照合する（2026-10-12）
"""

import argparse
import datetime
import os
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from configs.schema import LEGACY_MODE_DEFAULTS, MACHINE, MODE, SR, SR_UPSAMPLE_KEYS, TRAIN, ConfigError, apply_overrides, check_split, check_sr_values, check_values, to_argv, validate, validate_sr  # noqa: E402
from util.run_paths import LAUNCH_FILE, ckpt_path, latest_launch, run_name, saved_tags  # noqa: E402

# Stage 1 で固定の引数（チューニング対象ではないので設定ファイルには置かない）
FIXED_ARGV = ["--model", "fidelity_gan", "--dataset_mode", "ct"]
RESUME_ONLY_FLAGS = ("--continue_train", "--epoch_count")  # 再開の内部状態。sh からは上書きできない（run_train が決める）
REPO_ROOT = HERE.parent  # sr.yaml の split_file の相対パスの基準（コンテナでは /workspace）
MANIFEST = "manifest.yaml"  # eid1024_dir の manifest（bash start2.sh dataset が書く。dataset.scale / interp / input_size）


def peek_task(overrides, default):
    """上書き列から --task の値だけ先に読む（sr.yaml を読むかどうかを決めるため。正式な反映は apply_overrides）。"""
    task = default
    for i, t in enumerate(overrides):
        if t == "--task" and i + 1 < len(overrides):
            task = overrides[i + 1]
        elif t.startswith("--task="):
            task = t.split("=", 1)[1]
    return task


def load_sr(path, train):
    """configs/sr.yaml を読み、SR のキーを検証して、`train:` の差し替えを train（平坦 dict）に反映する。戻り値 (sr_flat, overlay_flat)。"""
    sr, overlay = validate_sr(load_yaml(path))
    for k, v in overlay.items():
        train[k] = v
    return sr, overlay


def read_upsample(eid1024_dir):
    """eid1024_dir/manifest.yaml（bash start2.sh dataset が書く）の dataset.scale / interp / input_size。無い・不正ならエラー（推測で補間しない）。"""
    p = Path(eid1024_dir) / MANIFEST
    if not p.is_file():
        raise ConfigError(f"{p} がありません。eid1024_dir は bash start2.sh dataset --input_dir <eid_dir> --eidlike1024_dir <eid1024_dir> で作ったもの（manifest.yaml 付き）を指定してください")
    m = load_yaml(p)
    try:
        d = m["dataset"]
        up = {"scale": int(d["scale"]), "interp": str(d["interp"]), "input_size": int(d["input_size"])}
    except (KeyError, TypeError, ValueError) as e:
        raise ConfigError(f"{p} の dataset.{{scale, interp, input_size}} が読めません: {e}")
    up["from"] = str(p)
    return up


def check_sr_dirs(sr, machine):
    """sr: A = eid1024_dir / B = pcd1024_dir の存在、split_file の中身、manifest。sr["upsample"] を埋める。"""
    for k in ("eid1024_dir", "pcd1024_dir"):
        if not Path(machine[k]).is_dir():
            raise ConfigError(f"task: sr には machines.yaml の {k} が必要です（無い）: {machine[k]}")
    check_split(load_yaml(sr["split_file"]), f"split_file {sr['split_file']}")
    sr["upsample"] = read_upsample(machine["eid1024_dir"])


def sr_argv(sr):
    """sr の実効値 → train.py の引数列（SR のフラグ + manifest の方式）。"""
    up = sr["upsample"]
    return to_argv(sr, SR) + ["--sr_scale", str(up["scale"]), "--sr_interp", up["interp"], "--sr_input_size", str(up["input_size"])]


def check_d_pack_weights(weight_path, train, mode):
    """net_D.pth の 1 層目（state_dict で最初の 4 次元 weight）の入力チャネルが実効 mode の output_nc × d_pack と一致するか（launch を書く前に止める。2026-10-12）。
    strict な load_state_dict でも落ちるが、train.py の起動後（launch_resume を書いた後）になるのでここで先に照合する。"""
    import torch  # コンテナ内でのみ使う

    sd = torch.load(weight_path, map_location="cpu", weights_only=True)
    first = next(((k, v) for k, v in sd.items() if k.endswith(".weight") and getattr(v, "dim", lambda: 0)() == 4), None)
    if first is None:
        raise ConfigError(f"{weight_path} に畳み込みの weight がありません（形式が不正）")
    expect = train["network.output_nc"] * mode["d_pack"]
    if first[1].shape[1] != expect:
        raise ConfigError(f"{weight_path} の D の 1 層目 {first[0]} の入力チャネル {first[1].shape[1]} が実効設定 output_nc × d_pack = {train['network.output_nc']} × {mode['d_pack']} = {expect} と合いません（d_pack は resume で変えられない）")


def start_tensorboard(run_dir, port):
    """この run の tb/ だけを logdir にして TensorBoard を起動する（start.sh が SPICA_TB_PORT を渡したときだけ）。
    **同じポート**で動いている前の run の TensorBoard や `bash start.sh tb`（全 run 表示）は止める。別ポート（Stage 2 の stage2_tb_port）のものは触らない（2026-09-08）。ログは <run>/tensorboard.log。"""
    import shutil
    import signal
    import subprocess

    exe = shutil.which("tensorboard") or (str(Path(sys.executable).parent / "tensorboard") if (Path(sys.executable).parent / "tensorboard").exists() else None)  # venv の bin が PATH に無くても同じ環境のものを使う
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
    print(f"[run_train] TensorBoard: http://localhost:{port} → {tb_dir}（この run だけ。全 run は bash start.sh tb）", flush=True)


def load_yaml(path):
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"設定ファイルが見つかりません: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_argv(name, train, mode, machine, sr, extra):
    """実効値（平坦 dict）から train.py の引数列を組む。task: sr のときは --dir_A / --dir_B に eid1024_dir / pcd1024_dir を入れる（machine の記録は変えない）。"""
    argv_machine = dict(machine)
    sr_part = []
    if mode["task"] == "sr":
        argv_machine["pcd_dir"], argv_machine["eid_dir"] = machine["eid1024_dir"], machine["pcd1024_dir"]  # --dir_A = EID1024, --dir_B = PCD1024
        sr_part = sr_argv(sr)
    return ["--name", name] + FIXED_ARGV + to_argv(train, TRAIN) + to_argv(mode, MODE) + to_argv(argv_machine, MACHINE) + sr_part + list(extra)


SR_PROBE = {"split_file": "", "b_exclude_cases": []}  # task が degrade のときも sr のフラグを構文として受け、finish_sr で「sr のときだけ」と明示的に止めるためのダミー


def sections(train, mode, machine, sr=None):
    secs = {"train": (train, TRAIN), "mode": (mode, MODE), "machine": (machine, MACHINE), "sr": (sr if sr is not None else dict(SR_PROBE), SR)}
    return secs


def finish_sr(train, mode, machine, sr, applied):
    """上書き反映後の task に応じて sr を確定する: sr なら存在・manifest を検査（sr は必須）、degrade なら None。--task の上書きで sr.yaml を読まずに sr になった事故を止める。"""
    if mode["task"] == "sr":
        if sr is None:
            raise ConfigError("task が sr なのに sr.yaml が読まれていません（mode.yaml の task か --task sr を確認）")
        check_sr_values(sr, REPO_ROOT)
        check_sr_dirs(sr, machine)
        return sr
    used = [k.flag for k in SR.values() if k.flag in applied]
    if used:
        raise ConfigError(f"sr.yaml のフラグ {used} は task: sr のときだけ使えます（今は task: {mode['task']}）")
    return None


def prepare_new(a, train, mode, machine, sr, overrides, now):
    """新規 run: yaml の値（+ sr.yaml の差し替え）に上書きを反映 → 検証 → run 名と保存先。"""
    applied = apply_overrides(overrides, sections(train, mode, machine, sr))
    if any(f in applied for f in RESUME_ONLY_FLAGS):
        raise ConfigError(f"{' / '.join(RESUME_ONLY_FLAGS)} は再開の内部フラグです。再開は --resume <run> で行ってください")
    if train["log.continue_train"] or train["log.epoch_count"] != 1:
        raise ConfigError("新規 run では log.continue_train = false, log.epoch_count = 1 にしてください（再開は --resume <run>）")
    check_values(train, mode, machine)
    sr = finish_sr(train, mode, machine, sr, applied)
    name = run_name(now)
    out_dir = Path(machine["checkpoints_dir"]) / name
    if out_dir.exists():
        raise ConfigError(f"run ディレクトリが既に存在します: {out_dir}（同一分の再起動。1 分待つか、不要なら消してから起動）")
    return name, out_dir, [], LAUNCH_FILE, applied, None, sr


def prepare_resume(a, train_yaml, mode_yaml, machine_yaml, sr_yaml, overrides, now):
    """再開: 選んだ checkpoint の実効設定を基準に、今回の上書きを反映 → 検証 → 再開フラグ。task（degrade | sr）は checkpoint のものを引き継ぐ（変更不可）。"""
    import torch  # コンテナ内でのみ使う（state.pth の進捗を読む）

    # run の場所: 今の machines.yaml の checkpoints_dir に、今回の上書き（--checkpoints_dir があれば）を反映して探す
    probe = (dict(train_yaml), dict(mode_yaml), dict(machine_yaml), dict(sr_yaml) if sr_yaml is not None else None)  # sr のフラグも構文として通す（sections がダミーを足す。実効値は checkpoint 基準）
    apply_overrides(overrides, sections(*probe))
    run_dir = Path(probe[2]["checkpoints_dir"]) / a.resume
    if not run_dir.is_dir():
        raise ConfigError(f"再開する run が見つかりません: {run_dir}")
    state_path = ckpt_path(run_dir, a.resume_tag, "state.pth")
    if not state_path.is_file():
        raise ConfigError(f"再開用 state がありません: {state_path}（保存済み tag: {saved_tags(run_dir)}）")
    st = torch.load(state_path, map_location="cpu", weights_only=True)
    # 再開の基準は**選んだ checkpoint の state.pth に入っている実効設定**（その重みを作った設定）。失敗した起動の launch_resume_*.yaml は基準にならない（2026-09-09）。
    # 古い run（state に config が無い）だけ最新 launch を基準にする
    if isinstance(st.get("config"), dict) and all(k in st["config"] for k in ("train", "mode", "machine")):
        saved, base = {k: dict(st["config"][k]) for k in ("train", "mode", "machine")}, state_path
        if isinstance(st["config"].get("sr"), dict):
            saved["sr"] = dict(st["config"]["sr"])
    else:
        base = latest_launch(run_dir)
        if not base.is_file():
            raise ConfigError(f"{state_path} に設定が無く、{run_dir / LAUNCH_FILE} もありません")
        print(f"[run_train] 注意: {state_path.name} に実効設定が無い古い run なので、最新の launch（{base.name}）を基準にする")
        saved = load_yaml(base)
        if isinstance(saved, dict) and not isinstance(saved.get("sr"), dict):
            saved.pop("sr", None)
    try:
        # run 作成後に schema に追加されたキー（例: optim.seed, log.preview_bits）は run の launch に無い。今の yaml の値で補い、警告する
        for section, cur, schema in (("train", train_yaml, TRAIN), ("mode", mode_yaml, MODE), ("machine", machine_yaml, MACHINE)):
            missing = [k for k in schema if k not in saved[section]]
            legacy = [k for k in missing if section == "mode" and k in LEGACY_MODE_DEFAULTS]  # 導入前の run = その機構なしで学習された重み。今の yaml で補わず導入前の意味に
            if legacy:
                print(f"[run_train] 注意: run の mode に無いキー（導入前の run）は導入前の値で扱う: " + ", ".join(f"{k}={LEGACY_MODE_DEFAULTS[k]!r}" for k in legacy))
                for k in legacy:
                    saved[section][k] = LEGACY_MODE_DEFAULTS[k]
                missing = [k for k in missing if k not in legacy]
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
        machine = validate("machine(run)", saved["machine"], MACHINE)
        sr = None
        if mode["task"] == "sr":
            if "sr" not in saved:
                raise ConfigError(f"{base} は task: sr なのに sr セクションがありません")
            sr_saved = dict(saved["sr"])
            upsample = sr_saved.pop("upsample", None)
            sr_saved.pop("train_overlay", None)
            sr = validate("sr(run)", sr_saved, SR)
            if not isinstance(upsample, dict) or any(k not in upsample for k in SR_UPSAMPLE_KEYS):
                raise ConfigError(f"{base} の sr.upsample が不正です（scale / interp / input_size）")
            sr["upsample"] = upsample
    except (KeyError, TypeError) as e:
        raise ConfigError(f"{base} の形式が不正です: {e}")
    for section, cur, sv in (("train", train_yaml, train), ("mode", mode_yaml, mode), ("machine", machine_yaml, machine)):
        diff = {k: (sv.get(k), v) for k, v in cur.items() if sv.get(k) != v and k not in ("log.continue_train", "log.epoch_count")}
        if diff:
            print(f"[run_train] 注意: 今の {section} 設定は checkpoint の実効設定と異なる（checkpoint の値を使う。変えるなら --flag で上書き）: {diff}")
    saved_residual, saved_task, saved_d_pack = mode["residual"], mode["task"], mode["d_pack"]
    applied = apply_overrides(overrides, sections(train, mode, machine, sr))
    if sr is None:
        used = [k.flag for k in SR.values() if k.flag in applied]
        if used:
            raise ConfigError(f"sr.yaml のフラグ {used} は task: sr の run にだけ使えます（この checkpoint は task: {saved_task}）")
    if any(f in applied for f in RESUME_ONLY_FLAGS):
        raise ConfigError(f"{' / '.join(RESUME_ONLY_FLAGS)} は再開の内部フラグです（run_train が決めます）")
    if mode["residual"] != saved_residual:
        raise ConfigError(f"再開時に residual を変えることはできません（重みの意味が変わる）: checkpoint は residual={saved_residual}")
    if mode["task"] != saved_task:
        raise ConfigError(f"再開時に task を変えることはできません（データと G の意味が変わる）: checkpoint は task={saved_task}")
    if mode["d_pack"] != saved_d_pack:
        raise ConfigError(f"再開時に d_pack を変えることはできません（D の 1 層目の形が変わる）: checkpoint は d_pack={saved_d_pack}")
    if Path(machine["checkpoints_dir"]) / a.resume != run_dir:
        raise ConfigError(f"再開時に checkpoints_dir を変えることはできません: {machine['checkpoints_dir']} != {run_dir.parent}")
    for kind in ("net_G.pth", "net_D.pth"):
        if not ckpt_path(run_dir, a.resume_tag, kind).is_file():
            raise ConfigError(f"再開用の重みがありません: {ckpt_path(run_dir, a.resume_tag, kind)}")
    check_d_pack_weights(ckpt_path(run_dir, a.resume_tag, "net_D.pth"), train, mode)  # 旧 run（d_pack 無し = 1）の net_D.pth は 1 ch なので整合する
    epoch_count = st["epoch"] + 1 if st["epoch_done"] else st["epoch"]  # epoch 末保存なら次の epoch から、途中保存ならその epoch を頭から
    train["log.continue_train"] = True
    train["log.epoch_count"] = epoch_count
    check_values(train, mode, machine)
    if sr is not None:
        saved_up = sr["upsample"]
        check_sr_values(sr, REPO_ROOT)
        check_sr_dirs(sr, machine)  # 今のディスクの manifest を読み直す
        if any(sr["upsample"][k] != saved_up[k] for k in SR_UPSAMPLE_KEYS):
            raise ConfigError(f"eid1024_dir の manifest の方式が checkpoint と違います: checkpoint {saved_up} vs 今 {sr['upsample']}")
    extra = ["--epoch", str(a.resume_tag), "--resume_state", str(state_path)]
    print(f"[run_train] resume {a.resume} from tag '{a.resume_tag}' (base: {'state.pth の config' if base == state_path else base.name}): saved epoch {st['epoch']} (done={st['epoch_done']}), total_iters {st['total_iters']} → epoch_count {epoch_count}, n_epochs {train['optim.n_epochs']}, task {mode['task']}")
    return a.resume, run_dir, extra, f"launch_resume_{now.strftime('%Y%m%d_%H%M%S')}.yaml", applied, base, train, mode, machine, sr


def main():
    p = argparse.ArgumentParser(description="Stage 1 (FE-GAN) 学習の起動器")
    p.add_argument("--machine", required=True, help="configs/machines.yaml のエントリ名（PC1, mac, ...）")
    p.add_argument("--train", required=True, help="学習パラメータ YAML")
    p.add_argument("--mode", required=True, help="モード切替 YAML")
    p.add_argument("--sr", required=True, help="task: sr 用の YAML（configs/sr.yaml）。task が degrade のときは読まない")
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
        # task: sr なら sr.yaml を読んで train.yaml を差し替える（--flag より前。新規 run のとき。再開は checkpoint の sr を基準にする）
        sr, overlay = (load_sr(a.sr, train) if peek_task(overrides, mode["task"]) == "sr" else (None, {}))
        if a.resume:
            name, out_dir, extra, launch_name, applied, base, train, mode, machine, sr = prepare_resume(a, train, mode, machine, sr, overrides, now)
        else:
            name, out_dir, extra, launch_name, applied, base, sr = prepare_new(a, train, mode, machine, sr, overrides, now)
    except ConfigError as e:
        print(f"[run_train] 設定エラー: {e}", file=sys.stderr)
        sys.exit(2)
    sr_record = None
    if sr is not None:
        sr_record = dict(sr)
        sr_record["train_overlay"] = overlay if not a.resume else "(resume: checkpoint の train が基準)"  # sr.yaml が train.yaml のどのキーを差し替えたか（記録用）

    if machine["gpu_gen"] != 0:
        extra = list(extra) + ["--require_cuda"]  # F-16: GPU マシンの定義で CUDA が無ければ train.py が止める（黙って CPU に落ちない）
    extra = list(extra) + ["--launch_path", str(out_dir / launch_name)]  # この起動の実効設定を state.pth に写すため（再開の基準。2026-09-09）
    argv = build_argv(name, train, mode, machine, sr, extra)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / launch_name, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {"timestamp": now.isoformat(timespec="seconds"), "machine_name": a.machine,
             "train": train, "mode": mode, "machine": machine,  # 実効値（yaml + 上書き）。推論・再開はこれを読む
             "sr": sr_record,  # task: sr のとき（split_file / b_exclude_cases / upsample = manifest の方式 / train_overlay）。degrade では null
             "overrides": list(overrides), "applied": applied,
             "resume": a.resume, "resume_tag": a.resume_tag if a.resume else None, "base_launch": str(base) if base else None,
             "argv": argv},
            f, allow_unicode=True, sort_keys=False,
        )

    cmd = [sys.executable, str(HERE / "train.py")] + argv
    print(f"[run_train] run {name} → {out_dir / launch_name}（全引数はこのファイルの argv）", flush=True)
    if os.environ.get("SPICA_TB_PORT"):  # start.sh 経由のとき。手動起動や scratch では起動しない
        try:
            start_tensorboard(out_dir, int(os.environ["SPICA_TB_PORT"]))
        except ConfigError as e:
            print(f"[run_train] 設定エラー: {e}", file=sys.stderr)
            sys.exit(2)
    os.chdir(HERE)
    os.execv(sys.executable, cmd)


if __name__ == "__main__":
    main()
