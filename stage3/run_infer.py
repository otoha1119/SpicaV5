"""stage3/run_infer.py — Stage 3 推論（フォルダ丸ごと）の起動器（Docker 内で動く）。stage2/run_infer.py から 2026-10-13 に複製し、ネットの構成を ddim.yml から取るようにした。

  1. configs/infer.yaml と ../configs/machines.yaml を configs/schema.py で検証する（既定値なし）
  2. 重みディレクトリ（--weight_dir）に model.pth / ema.pth があることを確認し、run（launch.yaml がある所）の最新 launch から ddim（ネットと拡散の設定）と HU 正規化を取る
  3. 512 入力の補間方式と学習画像の一辺は run の dataset_info.yaml（train.py が書いたもの）から取る（学習と同じ）
  4. 表示設定（log.display_hu_*・preview_bits）は現在の configs/train.yaml から取る（表示は重みに紐づかない）
  5. sh からの上書きを反映し、解決済み設定を <repo>/output/<run>_<重み>_<入力名>_<時刻>/infer.yaml に保存してから inference_dir.py を exec する
"""

import argparse
import datetime
import os
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from configs.schema import DDIM, INFER, MACHINE, TRAIN, ConfigError, apply_overrides, check_ddim_values, check_infer_values, to_argv, validate, validate_shared  # noqa: E402
from data.ct_io import INTERP_FLAGS  # noqa: E402
from functions.denoising import overlapping_grid_indices  # noqa: E402
from util.run_paths import CKPT_EMA, CKPT_MODEL, DATASET_INFO_FILE, LAUNCH_FILE, find_run_dir, infer_output_dir, latest_launch  # noqa: E402

PATH_FLAGS = ("--weight_dir", "--input_dir")
HU_KEYS = ("data.hu_offset", "data.hu_min", "data.hu_max")
DISPLAY_KEYS = ("log.display_hu_min", "log.display_hu_max", "log.preview_bits")


def load_yaml(path):
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"設定ファイルが見つかりません: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def split_overrides(tokens):
    paths, rest, i = {}, [], 0
    while i < len(tokens):
        t = tokens[i]
        if t in PATH_FLAGS:
            if i + 1 >= len(tokens) or tokens[i + 1].startswith("--"):
                raise ConfigError(f"{t} には値が必要です")
            paths[t] = tokens[i + 1]
            i += 2
            continue
        rest.append(t)
        i += 1
    return paths, rest


def run_config(launch, path):
    """run の launch から ddim（schema で検証）と HU 正規化を取る。"""
    if not isinstance(launch.get("ddim"), dict) or not isinstance(launch.get("train"), dict):
        raise ConfigError(f"{path} に ddim / train セクションがありません")
    ddim = validate("ddim(run)", launch["ddim"], DDIM)
    check_ddim_values(ddim)
    missing = [k for k in HU_KEYS if k not in launch["train"]]
    if missing:
        raise ConfigError(f"{path} の train に {missing} がありません")
    return ddim, {k: launch["train"][k] for k in HU_KEYS}


def upsample_config(run_dir):
    p = Path(run_dir) / DATASET_INFO_FILE
    if not p.is_file():
        raise ConfigError(f"{p} がありません（train.py が起動時に書く）")
    info = load_yaml(p)
    try:
        scale, interp = int(info["eid_upsample"]["scale"]), str(info["eid_upsample"]["interp"])
        h, w = (int(x) for x in info["image_hw"])
    except (KeyError, TypeError, ValueError) as e:
        raise ConfigError(f"{p} の eid_upsample.scale / interp / image_hw を読めません: {e}")
    if scale < 2 or interp not in INTERP_FLAGS:
        raise ConfigError(f"{p} の補間の方式が不正です: scale {scale}, interp {interp!r}")
    if h != w or h % scale:
        raise ConfigError(f"{p} の image_hw {[h, w]} が正方形で scale {scale} の倍数ではありません")
    return scale, interp, h


def main():
    p = argparse.ArgumentParser(description="Stage 3（条件付き DDIM）推論の起動器")
    p.add_argument("--machine", required=True)
    p.add_argument("--infer", required=True, help="configs/infer.yaml")
    p.add_argument("--train", required=True, help="現在の configs/train.yaml（表示の設定だけ使う）")
    p.add_argument("--machines", required=True)
    p.add_argument("--weight_dir", required=True, help="重みディレクトリ（model.pth / ema.pth がある所: <run>/latest | <run>/best | <run>/weights/iter_NNNNNNN）")
    p.add_argument("--input_dir", required=True, help="処理する PNG 群のフォルダ（直下に症例フォルダ。一辺 1024 か 512）")
    p.add_argument("--out_root", default=None, help="出力の根（通常は省略 = <repo>/output/。scratch 実行用）")
    p.add_argument("overrides", nargs=argparse.REMAINDER)
    a = p.parse_args()
    overrides = a.overrides[1:] if a.overrides and a.overrides[0] == "--" else a.overrides

    try:
        infer = validate("infer", load_yaml(a.infer), INFER)
        machines = load_yaml(a.machines)
        if not isinstance(machines, dict) or a.machine not in machines:
            raise ConfigError(f"machines.yaml にエントリ '{a.machine}' がありません。候補: {list(machines) if isinstance(machines, dict) else '(不正な形式)'}")
        machine = validate_shared(f"machines.{a.machine}", machines[a.machine], MACHINE)
        path_over, rest = split_overrides(overrides)
        applied = apply_overrides(rest, {"infer": (infer, INFER), "machine": (machine, MACHINE)})
        weight_dir = Path(path_over.get("--weight_dir", a.weight_dir))
        input_dir = Path(path_over.get("--input_dir", a.input_dir))
        for k in (CKPT_MODEL, CKPT_EMA):
            if not (weight_dir / k).is_file():
                raise ConfigError(f"重みディレクトリに {k} がありません: {weight_dir}（<run>/latest | <run>/best | <run>/weights/iter_NNNNNNN を指定。infer_stage3.sh の WEIGHT_DIR か --weight_dir）")
        run_dir = find_run_dir(weight_dir)
        if run_dir is None:
            raise ConfigError(f"重みディレクトリの上に {LAUNCH_FILE} を持つ run が見つかりません: {weight_dir}")
        launch_path = latest_launch(run_dir)
        ddim, hu = run_config(load_yaml(launch_path), launch_path)
        check_infer_values(infer, machine, ddim["diffusion.num_diffusion_timesteps"])
        if infer["use_ema"] and not ddim["model.ema"]:
            raise ConfigError("run は EMA なし（ddim.yml model.ema: false）なので --use_ema false にする")
        scale, interp, image_size = upsample_config(run_dir)
        try:
            overlapping_grid_indices(image_size, image_size, ddim["data.image_size"], infer["tile_stride"])
        except ValueError as e:
            raise ConfigError(f"infer.yaml tile_stride: {e}")
        if not input_dir.is_dir():
            raise ConfigError(f"入力フォルダがありません: {input_dir}（infer_stage3.sh の INPUT_DIR か --input_dir）")
        missing_cases = [c for c in infer["cases"] if not (input_dir / c).is_dir()]
        if missing_cases:
            raise ConfigError(f"infer.yaml cases の症例フォルダが入力にありません: {missing_cases}（{input_dir}）")
        if infer["teacher"] and not Path(machine["pcd1024_dir"]).is_dir():
            raise ConfigError(f"teacher=true ですが pcd1024_dir がありません: {machine['pcd1024_dir']}（実 EID は --teacher false）")
        if infer["save_panel"]:
            if infer["input"] == "eidlike" and infer["eid_slice"] and not (Path(machine["eid_dir"]) / infer["eid_slice"]).is_file():
                raise ConfigError(f"infer.yaml の eid_slice がありません: {Path(machine['eid_dir']) / infer['eid_slice']}（不要なら --eid_slice \"\"）")
            if infer["input"] == "eid" and infer["pcd_slice"]:
                for root in (machine["eidlike1024_dir"], machine["pcd1024_dir"]):
                    if not (Path(root) / infer["pcd_slice"]).is_file():
                        raise ConfigError(f"infer.yaml の pcd_slice がありません: {Path(root) / infer['pcd_slice']}（不要なら --pcd_slice \"\"）")
        device = "cpu" if machine["gpu_gen"] == 0 else "cuda"
        cur_train = validate("train", load_yaml(a.train), TRAIN)
        display = {k: cur_train[k] for k in DISPLAY_KEYS}
        if display["log.display_hu_min"] >= display["log.display_hu_max"] or display["log.preview_bits"] not in (8, 16):
            raise ConfigError(f"{a.train} の log.display_hu_min < display_hu_max、preview_bits は 8 | 16")
    except ConfigError as e:
        print(f"[run_infer] 設定エラー: {e}", file=sys.stderr)
        sys.exit(2)

    jst = datetime.timezone(datetime.timedelta(hours=9), name="JST")
    now = datetime.datetime.now(jst)
    out_dir = infer_output_dir(run_dir, weight_dir, input_dir, now, root=a.out_root)
    if out_dir.exists():
        print(f"[run_infer] 設定エラー: 出力先が既に存在します（同一秒の再実行）: {out_dir}", file=sys.stderr)
        sys.exit(2)
    out_dir.mkdir(parents=True)
    argv = ["--weight_dir", str(weight_dir), "--input_dir", str(input_dir), "--out_dir", str(out_dir), "--device", device, "--launch", str(launch_path),
            "--index_cache_dir", str(Path(machine["stage3_checkpoints_dir"]) / ".case_index"), "--pcd1024_dir", str(machine["pcd1024_dir"]),
            "--eid_dir", str(machine["eid_dir"]), "--eidlike1024_dir", str(machine["eidlike1024_dir"]),
            "--hu_offset", str(hu["data.hu_offset"]), "--hu_min", str(hu["data.hu_min"]), "--hu_max", str(hu["data.hu_max"]),
            "--scale", str(scale), "--interp", interp, "--image_size", str(image_size),
            "--display_hu_min", str(display["log.display_hu_min"]), "--display_hu_max", str(display["log.display_hu_max"]), "--preview_bits", str(display["log.preview_bits"])]
    argv += to_argv(infer, INFER)

    with open(out_dir / "infer.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {"timestamp": now.isoformat(timespec="seconds"), "machine_name": a.machine, "device": device, "run_dir": str(run_dir), "launch": str(launch_path),
             "weight_dir": str(weight_dir), "input_dir": str(input_dir), "infer": infer, "machine": machine, "ddim": ddim, "hu": hu,
             "upsample": {"scale": scale, "interp": interp, "image_size": image_size, "from": str(Path(run_dir) / DATASET_INFO_FILE)},
             "display": display, "overrides": list(overrides), "applied": applied, "argv": argv},
            f, allow_unicode=True, sort_keys=False,
        )
    print(f"[run_infer] {weight_dir} × {input_dir} → {out_dir}（全引数は infer.yaml の argv）", flush=True)
    os.chdir(HERE)
    os.execv(sys.executable, [sys.executable, str(HERE / "inference_dir.py")] + argv)


if __name__ == "__main__":
    main()
