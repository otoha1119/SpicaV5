"""run_train.py — Stage 1 学習の起動器（Docker 内で動く）。

設計: docs/plans/20260905_config-design-plan.md、修正: docs/plans/20260907_review-fix-list.md（F-02 / F-03 / F-07 / F-10 / F-11）
  1. configs/train.yaml, configs/mode.yaml, ../configs/machines.yaml を読み、configs/schema.py で検証する
     （必須キー欠落・未知キー・型違いはエラー。既定値は無い）
  1'. mode.yaml の task（sh の --task で上書き可）が sr なら configs/sr.yaml（--sr）も読む [2026-10-12]:
     SR のキー（split_file / b_exclude_cases）を検証し、`train:` / `mode:` セクションで train.yaml / mode.yaml を差し替え（優先順位 yaml → sr.yaml → --flag）、
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
    residual / task / d_pack / noise_input は checkpoint から変えられない（重みの意味・形が変わる）。d_pack は net_D.pth の 1 層目の入力チャネル、
    noise_input は net_G.pth の 1 層目の入力チャネル（input_nc + 1）とも照合する（2026-10-12）

2 モデル同時学習（bash start.sh multi、2026-10-13、docs/plans/20261012_multi-model-training-plan.md）:
  python run_train.py --machine PC1 --train configs/train.yaml --multi configs/modes/n_only.yaml configs/modes/n_pacgan.yaml --sr ... --machines ... -- [--flag value]
  → mode ファイルごとに「モデルの仕様」（train / mode / machine / sr の実効値）を組み、run 名は同じ時刻 + '-' + mode ファイル名（2026_1013_0130-n_only / -n_pacgan）。
    mode ファイルの `train:` でそのモデルだけ PER_MODEL_TRAIN_KEYS（λ、lr、β、G/D の ch と初期化）を差し替えられる。共有キーを書くとエラー。--flag は全モデルに同じ値。
    各 run は単独と同じ中身（launch.yaml / latest best weights / loss_log.txt / tb/ / output_images/）+ group.yaml（組の run 名、mode ファイル、起動時刻、共有設定のハッシュ）。
    train.py には先頭モデルの argv + --co_launch <2 つ目の launch.yaml> を渡し、1 プロセス・DataLoader 1 つで 2 モデルを同じ batch で回す。受け付けるのは 2 つまで（MULTI_MAX_MODELS）
  組の再開: --resume <runA> <runB> [--resume_tag tag]（各 run の state.pth の実効設定を基準に、共有キーの一致と進捗の一致を再検査。group.yaml の相手と違えば警告して続行）。
  --resume <run> だけなら単独（組の片方だけでも可。従来どおり）
"""

import argparse
import datetime
import hashlib
import os
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from configs.schema import LEGACY_MODE_DEFAULTS, MACHINE, MODE, MULTI_MAX_MODELS, PER_MODEL_TRAIN_KEYS, SHARED_MODE_KEYS, SR, SR_UPSAMPLE_KEYS, TRAIN, ConfigError, apply_overrides, check_shared, check_split, check_sr_values, check_values, to_argv, validate, validate_mode_file, validate_sr  # noqa: E402
from util.run_paths import GROUP_FILE, LAUNCH_FILE, ckpt_path, latest_launch, run_name, saved_tags  # noqa: E402

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


def load_sr(path, train, mode):
    """configs/sr.yaml を読み、SR のキーを検証して、`train:` / `mode:` の差し替えを train / mode（平坦 dict）に反映する。戻り値 (sr_flat, {"train": overlay, "mode": overlay})。"""
    sr, train_overlay, mode_overlay = validate_sr(load_yaml(path))
    for k, v in train_overlay.items():
        train[k] = v
    for k, v in mode_overlay.items():
        mode[k] = v
    return sr, {"train": train_overlay, "mode": mode_overlay}


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


def check_noise_input_weights(weight_path, train, mode):
    """net_G.pth の 1 層目（state_dict で最初の 4 次元 weight）の入力チャネルが実効 mode の input_nc + (1 if noise_input else 0) と一致するか（2026-10-12、段階 5-5）。
    check_d_pack_weights と同じ流儀: strict な load_state_dict でも落ちるが train.py の起動後になるので、launch を書く前にここで照合する。
    resume のほか、推論・crop（run_infer.py）も旧 run を読むときにこれを呼ぶ。"""
    import torch  # コンテナ内でのみ使う

    sd = torch.load(weight_path, map_location="cpu", weights_only=True)
    first = next(((k, v) for k, v in sd.items() if k.endswith(".weight") and getattr(v, "dim", lambda: 0)() == 4), None)
    if first is None:
        raise ConfigError(f"{weight_path} に畳み込みの weight がありません（形式が不正）")
    expect = train["network.input_nc"] + (1 if mode["noise_input"] else 0)
    if first[1].shape[1] != expect:
        raise ConfigError(f"{weight_path} の G の 1 層目 {first[0]} の入力チャネル {first[1].shape[1]} が実効設定 input_nc + (noise_input ? 1 : 0) = {train['network.input_nc']} + {1 if mode['noise_input'] else 0} = {expect} と合いません（noise_input は resume で変えられない）")


def start_tensorboard(run_dirs, port):
    """この run の tb/ だけを logdir にして TensorBoard を起動する（start.sh が SPICA_TB_PORT を渡したときだけ）。
    組（2 モデル同時学習、2026-10-13）では --logdir_spec <nameA>:<runA>/tb,<nameB>:<runB>/tb で 2 run を同時表示する。
    **同じポート**で動いている前の run の TensorBoard や `bash start.sh tb`（全 run 表示）は止める。別ポート（Stage 2 の stage2_tb_port）のものは触らない（2026-09-08）。ログは <先頭 run>/tensorboard.log。"""
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
    tb_dirs = [Path(d) / "tb" for d in run_dirs]
    for t in tb_dirs:
        t.mkdir(parents=True, exist_ok=True)
    if len(tb_dirs) == 1:
        logdir_args = ["--logdir", str(tb_dirs[0])]
    else:
        logdir_args = ["--logdir_spec", ",".join(f"{Path(d).name}:{t}" for d, t in zip(run_dirs, tb_dirs))]
    log = open(Path(run_dirs[0]) / "tensorboard.log", "ab")
    subprocess.Popen([exe] + logdir_args + ["--port", str(port), "--bind_all", "--reload_interval", "5"],
                     stdout=log, stderr=log, start_new_session=True)
    print(f"[run_train] TensorBoard: http://localhost:{port} → {' , '.join(str(t) for t in tb_dirs)}（{'この組だけ' if len(tb_dirs) > 1 else 'この run だけ'}。全 run は bash start.sh tb）", flush=True)


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


def prepare_new(a, train_yaml, mode, train_overlay, machine_yaml, overrides, now, suffix, mode_file):
    """新規 run の 1 モデル分: yaml の値（+ mode ファイルの train: の差し替え + sr.yaml の差し替え）に上書きを反映 → 検証 → run 名と保存先。
    戻り値はモデルの仕様 dict（name / out_dir / train / mode / machine / sr / extra / launch_name / applied / base / mode_file / train_overlay / sr_overlay）。"""
    train, machine = dict(train_yaml), dict(machine_yaml)
    for k, v in train_overlay.items():  # multi の mode ファイルの train:（PER_MODEL_TRAIN_KEYS だけ。validate_mode_file 済み）。優先順位 train.yaml → mode ファイルの train: → sr.yaml の train: → --flag
        train[k] = v
    sr, sr_overlay = (load_sr(a.sr, train, mode) if peek_task(overrides, mode["task"]) == "sr" else (None, {}))  # task: sr なら sr.yaml を読んで train / mode を差し替える（--flag より前）
    applied = apply_overrides(overrides, sections(train, mode, machine, sr))
    if any(f in applied for f in RESUME_ONLY_FLAGS):
        raise ConfigError(f"{' / '.join(RESUME_ONLY_FLAGS)} は再開の内部フラグです。再開は --resume <run> で行ってください")
    if train["log.continue_train"] or train["log.epoch_count"] != 1:
        raise ConfigError("新規 run では log.continue_train = false, log.epoch_count = 1 にしてください（再開は --resume <run>）")
    check_values(train, mode, machine)
    sr = finish_sr(train, mode, machine, sr, applied)
    try:
        name = run_name(now, suffix)
    except ValueError as e:
        raise ConfigError(str(e))
    out_dir = Path(machine["checkpoints_dir"]) / name
    if out_dir.exists():
        raise ConfigError(f"run ディレクトリが既に存在します: {out_dir}（同一分の再起動。1 分待つか、不要なら消してから起動）")
    return {"name": name, "out_dir": out_dir, "extra": [], "launch_name": LAUNCH_FILE, "applied": applied, "base": None,
            "train": train, "mode": mode, "machine": machine, "sr": sr, "mode_file": mode_file, "train_overlay": dict(train_overlay), "sr_overlay": sr_overlay,
            "resume": None, "resume_tag": None, "progress": None}


def prepare_resume(a, run, train_yaml, mode_yaml, machine_yaml, sr_yaml, overrides, now):
    """再開の 1 run 分: 選んだ checkpoint の実効設定を基準に、今回の上書きを反映 → 検証 → 再開フラグ。task（degrade | sr）は checkpoint のものを引き継ぐ（変更不可）。
    戻り値は prepare_new と同じ形の仕様 dict（progress に state.pth の epoch / epoch_done / total_iters）。"""
    import torch  # コンテナ内でのみ使う（state.pth の進捗を読む）

    # run の場所: 今の machines.yaml の checkpoints_dir に、今回の上書き（--checkpoints_dir があれば）を反映して探す
    probe = (dict(train_yaml), dict(mode_yaml), dict(machine_yaml), dict(sr_yaml) if sr_yaml is not None else None)  # sr のフラグも構文として通す（sections がダミーを足す。実効値は checkpoint 基準）
    apply_overrides(overrides, sections(*probe))
    run_dir = Path(probe[2]["checkpoints_dir"]) / run
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
            sr_saved.pop("mode_overlay", None)
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
    saved_residual, saved_task, saved_d_pack, saved_noise = mode["residual"], mode["task"], mode["d_pack"], mode["noise_input"]
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
    if mode["noise_input"] != saved_noise:
        raise ConfigError(f"再開時に noise_input を変えることはできません（G の 1 層目の形が変わる）: checkpoint は noise_input={saved_noise}")
    if Path(machine["checkpoints_dir"]) / run != run_dir:
        raise ConfigError(f"再開時に checkpoints_dir を変えることはできません: {machine['checkpoints_dir']} != {run_dir.parent}")
    for kind in ("net_G.pth", "net_D.pth"):
        if not ckpt_path(run_dir, a.resume_tag, kind).is_file():
            raise ConfigError(f"再開用の重みがありません: {ckpt_path(run_dir, a.resume_tag, kind)}")
    check_d_pack_weights(ckpt_path(run_dir, a.resume_tag, "net_D.pth"), train, mode)  # 旧 run（d_pack 無し = 1）の net_D.pth は 1 ch なので整合する
    check_noise_input_weights(ckpt_path(run_dir, a.resume_tag, "net_G.pth"), train, mode)  # 旧 run（noise_input 無し = false）の net_G.pth は input_nc ch なので整合する
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
    print(f"[run_train] resume {run} from tag '{a.resume_tag}' (base: {'state.pth の config' if base == state_path else base.name}): saved epoch {st['epoch']} (done={st['epoch_done']}), total_iters {st['total_iters']} → epoch_count {epoch_count}, n_epochs {train['optim.n_epochs']}, task {mode['task']}")
    return {"name": run, "out_dir": run_dir, "extra": extra, "launch_name": f"launch_resume_{now.strftime('%Y%m%d_%H%M%S')}.yaml", "applied": applied, "base": base,
            "train": train, "mode": mode, "machine": machine, "sr": sr, "mode_file": None, "train_overlay": "(resume: checkpoint の train が基準)", "sr_overlay": None,
            "resume": run, "resume_tag": a.resume_tag, "progress": {"epoch": int(st["epoch"]), "epoch_done": bool(st["epoch_done"]), "total_iters": int(st["total_iters"])}}


def shared_hash(sp):
    """組で共有する実効値（train の共有キー + mode の共有キー + machine）の sha1（group.yaml の記録用。一致の検査は check_shared）。"""
    shared = {"train": {k: sp["train"][k] for k in TRAIN if k not in PER_MODEL_TRAIN_KEYS},
              "mode": {k: sp["mode"][k] for k in SHARED_MODE_KEYS},
              "machine": {k: sp["machine"][k] for k in MACHINE}}
    return hashlib.sha1(yaml.safe_dump(shared, sort_keys=True, allow_unicode=True).encode("utf-8")).hexdigest()


def check_group(specs, a):
    """組（2 モデル）の検査: 共有キーの一致（check_shared）、同一内容の警告、resume なら進捗の一致と group.yaml の相手の照合（違えば警告して続行）。"""
    check_shared([{"name": sp["name"], "train": sp["train"], "mode": sp["mode"], "machine": sp["machine"]} for sp in specs])
    if all(sp["train"] == specs[0]["train"] and sp["mode"] == specs[0]["mode"] for sp in specs[1:]):
        print("[run_train] 注意: 組の mode ファイルが同一内容です（mode / train の実効値が全部同じ。比較にならないが続行する。seed も同じなので同じ run が 2 つできる）")
    if a.resume:
        prog = [sp["progress"] for sp in specs]
        if any(p != prog[0] for p in prog[1:]):
            raise ConfigError("組の各 run の checkpoint の進捗（epoch / epoch_done / total_iters）が一致しません: " + "; ".join(f"{sp['name']}={sp['progress']}" for sp in specs) + "（同じ時点の tag を選ぶ）")
        names = [sp["name"] for sp in specs]
        for sp in specs:
            g = sp["out_dir"] / GROUP_FILE
            if not g.is_file():
                print(f"[run_train] 注意: {sp['name']} に {GROUP_FILE} が無い（単独で作られた run）。組として再開する")
                continue
            rec = load_yaml(g)
            runs = rec.get("runs") if isinstance(rec, dict) else None
            if runs != names:
                print(f"[run_train] 注意: {sp['name']} の {GROUP_FILE} の組 {runs} と今回の組 {names} が違う。共有キーは一致しているので続行する")


def main():
    p = argparse.ArgumentParser(description="Stage 1 (FE-GAN) 学習の起動器")
    p.add_argument("--machine", required=True, help="configs/machines.yaml のエントリ名（PC1, mac, ...）")
    p.add_argument("--train", required=True, help="学習パラメータ YAML")
    p.add_argument("--mode", default=None, help="モード切替 YAML（単独実行・再開のとき。--multi と排他）")
    p.add_argument("--multi", nargs="+", default=None, metavar="MODE_YAML", help=f"2 モデル同時学習: mode ファイル（configs/modes/<name>.yaml）を 2 つ。run 名は <時刻>-<name>。受け付けるのは {MULTI_MAX_MODELS} つまで（2026-10-13）")
    p.add_argument("--sr", required=True, help="task: sr 用の YAML（configs/sr.yaml）。task が degrade のときは読まない")
    p.add_argument("--machines", required=True, help="マシン定義 YAML")
    p.add_argument("--resume", nargs="+", default=None, metavar="RUN", help="再開する run 名（yyyy_mmdd_HHMM[-<name>]）。2 つ並べると組として再開（各 run の state.pth の実効設定を基準に、共有キーと進捗の一致を再検査）。同じディレクトリに続きを保存する")
    p.add_argument("--resume_tag", default="latest", help="再開に使う checkpoint の tag（latest | best | epoch 番号）")
    p.add_argument("overrides", nargs=argparse.REMAINDER, help="'--' の後に train.py の引数（schema にあるフラグのみ。bool は --flag true|false も可）")
    a = p.parse_args()
    overrides = a.overrides[1:] if a.overrides and a.overrides[0] == "--" else a.overrides

    # コンテナ内は UTC なので、実験名の時刻は JST (+9) 固定で生成する（ホストの時計と一致させる）
    jst = datetime.timezone(datetime.timedelta(hours=9), name="JST")
    now = datetime.datetime.now(jst)
    try:
        if (a.mode is None) == (a.multi is None):
            raise ConfigError("--mode（単独）か --multi <modeA> <modeB>（2 モデル同時学習）のどちらか一方を指定してください")
        if a.multi is not None:
            if a.resume:
                raise ConfigError("組の再開は --resume <runA> <runB> だけで指定します（mode は各 run の checkpoint から。--multi は新規用）")
            if len(a.multi) == 1:
                raise ConfigError(f"multi は mode ファイルを 2 つ指定します（1 つなら単独: bash start.sh [--flag]）: {a.multi}")
            if len(a.multi) > MULTI_MAX_MODELS:
                raise ConfigError(f"multi で受け付けるのは {MULTI_MAX_MODELS} つまでです（ユーザー決定 2026-10-13）: {len(a.multi)} 個 {a.multi}")
        if a.resume and len(a.resume) > MULTI_MAX_MODELS:
            raise ConfigError(f"組の再開で受け付けるのは {MULTI_MAX_MODELS} run までです: {a.resume}")
        if a.resume and len(set(a.resume)) != len(a.resume):
            raise ConfigError(f"同じ run を 2 回指定しています: {a.resume}")
        train_yaml = validate("train", load_yaml(a.train), TRAIN)
        machines = load_yaml(a.machines)
        if not isinstance(machines, dict) or a.machine not in machines:
            raise ConfigError(f"machines.yaml にエントリ '{a.machine}' がありません。候補: {list(machines) if isinstance(machines, dict) else '(不正な形式)'}")
        machine_yaml = validate(f"machines.{a.machine}", machines[a.machine], MACHINE)
        if a.resume:
            mode_yaml = validate("mode", load_yaml(a.mode), MODE)  # 再開は checkpoint の設定が基準。今の yaml は「run 作成後に追加されたキー」の補完にだけ使う
            sr_yaml = None
            if peek_task(overrides, mode_yaml["task"]) == "sr":  # sr のフラグを構文として通すため（実効値は checkpoint）
                sr_yaml = validate_sr(load_yaml(a.sr))[0]
            specs = [prepare_resume(a, run, train_yaml, mode_yaml, machine_yaml, sr_yaml, overrides, now) for run in a.resume]
        elif a.multi is not None:
            stems = [Path(f).stem for f in a.multi]
            if len(set(stems)) != len(stems):
                raise ConfigError(f"mode ファイル名（拡張子なし）が重複しています（run 名の接尾辞になるので別名にする）: {stems}")
            specs = []
            for f, stem in zip(a.multi, stems):
                mode, train_overlay = validate_mode_file(load_yaml(f), f"mode({Path(f).name})")
                specs.append(prepare_new(a, train_yaml, mode, train_overlay, machine_yaml, overrides, now, stem, str(Path(f).resolve())))
        else:
            mode = validate("mode", load_yaml(a.mode), MODE)
            specs = [prepare_new(a, train_yaml, mode, {}, machine_yaml, overrides, now, None, str(Path(a.mode).resolve()))]
        if len(specs) > 1:
            check_group(specs, a)
    except ConfigError as e:
        print(f"[run_train] 設定エラー: {e}", file=sys.stderr)
        sys.exit(2)

    names = [sp["name"] for sp in specs]
    group = None
    if len(specs) > 1:
        group = {"runs": names, "mode_files": {sp["name"]: sp["mode_file"] for sp in specs}, "timestamp": now.isoformat(timespec="seconds"),
                 "shared_hash": shared_hash(specs[0]), "resume": a.resume, "resume_tag": a.resume_tag if a.resume else None}
    for i, sp in enumerate(specs):
        sr_record = None
        if sp["sr"] is not None:
            sr_record = dict(sp["sr"])
            sr_record["train_overlay"] = sp["sr_overlay"]["train"] if not a.resume else "(resume: checkpoint の train が基準)"  # sr.yaml が train.yaml のどのキーを差し替えたか（記録用）
            sr_record["mode_overlay"] = sp["sr_overlay"]["mode"] if not a.resume else "(resume: checkpoint の mode が基準)"  # 同じく mode.yaml（noise_input: false。2026-10-12）
        extra = list(sp["extra"])
        if sp["machine"]["gpu_gen"] != 0:
            extra += ["--require_cuda"]  # F-16: GPU マシンの定義で CUDA が無ければ train.py が止める（黙って CPU に落ちない）
        extra += ["--launch_path", str(sp["out_dir"] / sp["launch_name"])]  # この起動の実効設定を state.pth に写すため（再開の基準。2026-09-09）
        sp["argv"] = build_argv(sp["name"], sp["train"], sp["mode"], sp["machine"], sp["sr"], extra)
        sp["out_dir"].mkdir(parents=True, exist_ok=True)
        with open(sp["out_dir"] / sp["launch_name"], "w", encoding="utf-8") as f:
            yaml.safe_dump(
                {"timestamp": now.isoformat(timespec="seconds"), "machine_name": a.machine,
                 "train": sp["train"], "mode": sp["mode"], "machine": sp["machine"],  # 実効値（yaml + 上書き）。推論・再開はこれを読む
                 "sr": sr_record,  # task: sr のとき（split_file / b_exclude_cases / upsample = manifest の方式 / train_overlay）。degrade では null
                 "overrides": list(overrides), "applied": sp["applied"],
                 "resume": sp["resume"], "resume_tag": sp["resume_tag"], "base_launch": str(sp["base"]) if sp["base"] else None,
                 "mode_file": sp["mode_file"], "train_overlay": sp["train_overlay"],  # multi の mode ファイルとその train: の差し替え（2026-10-13。単独は mode.yaml と {}）
                 "group": ({"runs": names, "index": i} if group else None),  # 組の run 名と自分の位置（index 0 の argv に --co_launch が付く）
                 "argv": sp["argv"]},
                f, allow_unicode=True, sort_keys=False,
            )
        if group and not a.resume:
            with open(sp["out_dir"] / GROUP_FILE, "w", encoding="utf-8") as f:
                yaml.safe_dump(group, f, allow_unicode=True, sort_keys=False)

    cmd = [sys.executable, str(HERE / "train.py")] + specs[0]["argv"]
    if len(specs) > 1:
        cmd += ["--co_launch"] + [str(sp["out_dir"] / sp["launch_name"]) for sp in specs[1:]]  # 2 つ目以降は launch.yaml の argv から train.py が opt を組む
    for sp in specs:
        print(f"[run_train] run {sp['name']} → {sp['out_dir'] / sp['launch_name']}（全引数はこのファイルの argv）", flush=True)
    if len(specs) > 1:
        print(f"[run_train] multi: {' + '.join(names)}（1 プロセス・DataLoader 1 つで同じ batch を順に。group.yaml に組を記録）", flush=True)
    if os.environ.get("SPICA_TB_PORT"):  # start.sh 経由のとき。手動起動や scratch では起動しない
        try:
            start_tensorboard([sp["out_dir"] for sp in specs], int(os.environ["SPICA_TB_PORT"]))
        except ConfigError as e:
            print(f"[run_train] 設定エラー: {e}", file=sys.stderr)
            sys.exit(2)
    os.chdir(HERE)
    os.execv(sys.executable, cmd)


if __name__ == "__main__":
    main()
