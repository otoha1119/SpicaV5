"""stage2/run_infer.py — Stage 2 推論（フォルダ丸ごと）の起動器（Docker 内で動く）。Stage 1 の run_infer.py と同じ流儀。[SpicaV5 新規 2026-09-10]

  1. configs/infer.yaml（方式）と ../configs/machines.yaml（gpu_gen / pcd1024_dir / stage2_checkpoints_dir）を configs/schema.py で検証する（既定値なし）
  2. 重みディレクトリ（--weight_dir、infer_stage2.sh の変数）に net_G.pth があることを確認し、そこから run（launch.yaml がある所）を探して
     **最新の launch の実効設定**から U-Net の構成（mode: arch / base_ch / n_pool / final_act / residual）と HU 正規化（train: data.hu_*）を取る（学習時と同じネットを組む）
  3. 512 入力を補間する方式（scale / interp）と学習画像の大きさは run の dataset_info.yaml（train.py が起動時に eidlike1024_dir/manifest.yaml から写したもの）から取る
     （学習入力と同じ関数 data/ct_io.upsample・同じ方式 = 学習中の実 EID テストと同じ結果）
  4. 表示設定（log.display_hu_min/max・preview_bits）は**現在の configs/train.yaml**（--train）から取る（表示は重みに紐づかない。Stage 1 と同じ）
  5. デバイスは machines.yaml の gpu_gen（0 → cpu、それ以外 → cuda）
  6. sh からの上書き（INFER / MACHINE のフラグ、--weight_dir / --input_dir）を反映し、解決済み設定を
     <repo>/output/<run>_<重みディレクトリ名>_<入力フォルダ名>_<実行時刻>/infer.yaml に保存してから inference_dir.py を exec する（実行ごとに別ディレクトリ）

使い方（通常は ../infer_stage2.sh 経由）:
  python run_infer.py --machine PC2 --infer configs/infer.yaml --train configs/train.yaml --machines ../configs/machines.yaml \
      --weight_dir /workspace/stage2/checkpoints/2026_0909_1200/best --input_dir /workspace/DataSet/EIDlike1024_v1 -- --max_slices 1 --save_panel true
"""

import argparse
import datetime
import os
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from configs.schema import INFER, MACHINE, MODE, TRAIN, ConfigError, apply_overrides, check_infer_values, to_argv, validate, validate_shared  # noqa: E402
from data.ct_io import INTERP_FLAGS  # noqa: E402
from util.run_paths import DATASET_INFO_FILE, LAUNCH_FILE, find_run_dir, infer_output_dir, latest_launch  # noqa: E402

PATH_FLAGS = ("--weight_dir", "--input_dir")  # sh の変数を --flag で上書きできる（start2.sh の引数が最優先）
HU_KEYS = ("data.hu_offset", "data.hu_min", "data.hu_max")  # run の launch.yaml の train から（学習時の正規化窓）
DISPLAY_KEYS = ("log.display_hu_min", "log.display_hu_max", "log.preview_bits")  # 現在の configs/train.yaml から（表示は重みに紐づかない）


def load_yaml(path):
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"設定ファイルが見つかりません: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def split_overrides(tokens):
    """上書き列から sh 変数系のフラグ（--weight_dir / --input_dir）を取り出し、残り（INFER / MACHINE のフラグ）を返す。"""
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


def net_config(launch, path):
    """run の launch から U-Net の構成（mode 全部。schema で検証）と HU 正規化（train.data.hu_*）を取る。無ければエラー（古い run）。"""
    if not isinstance(launch.get("train"), dict) or not isinstance(launch.get("mode"), dict):
        raise ConfigError(f"{path} に train / mode セクションがありません")
    mode = validate("mode(run)", launch["mode"], MODE)
    missing = [k for k in HU_KEYS if k not in launch["train"]]
    if missing:
        raise ConfigError(f"{path} の train に {missing} がありません（古い run?）")
    hu = {k: launch["train"][k] for k in HU_KEYS}
    if any(not isinstance(v, int) or isinstance(v, bool) for v in hu.values()):
        raise ConfigError(f"{path} の {HU_KEYS} は int: {hu}")
    return mode, hu


def upsample_config(run_dir):
    """run の dataset_info.yaml から 512 入力の補間方式（学習が manifest.yaml から写した scale / interp）と学習画像の一辺を取る。"""
    p = Path(run_dir) / DATASET_INFO_FILE
    if not p.is_file():
        raise ConfigError(f"{p} がありません（train.py が起動時に書く。古い run?）")
    info = load_yaml(p)
    try:
        scale, interp = int(info["eid_upsample"]["scale"]), str(info["eid_upsample"]["interp"])
        h, w = (int(x) for x in info["image_hw"])
    except (KeyError, TypeError, ValueError) as e:
        raise ConfigError(f"{p} の eid_upsample.scale / eid_upsample.interp / image_hw を読めません（2026-09-09 より前の run には無い）: {e}")
    if scale < 2 or interp not in INTERP_FLAGS:
        raise ConfigError(f"{p} の補間の方式が不正です: scale {scale}, interp {interp!r}")
    if h != w or h % scale:
        raise ConfigError(f"{p} の image_hw {[h, w]} が正方形で scale {scale} の倍数ではありません")
    return scale, interp, h


def main():
    p = argparse.ArgumentParser(description="Stage 2（U-Net）推論の起動器")
    p.add_argument("--machine", required=True, help="configs/machines.yaml のエントリ名（gpu_gen / pcd1024_dir / stage2_checkpoints_dir を使う）")
    p.add_argument("--infer", required=True, help="推論方式 YAML（configs/infer.yaml）")
    p.add_argument("--train", required=True, help="現在の学習設定 YAML（configs/train.yaml）。表示の設定（log.display_hu_*, preview_bits）だけ使う")
    p.add_argument("--machines", required=True, help="マシン定義 YAML")
    p.add_argument("--weight_dir", required=True, help="重みディレクトリ（net_G.pth がある所。<run>/latest | <run>/best | <run>/weights/epoch_NNN）")
    p.add_argument("--input_dir", required=True, help="処理する PNG 群のフォルダ（直下に症例フォルダ。一辺 1024 か 512）")
    p.add_argument("--out_root", default=None, help="出力の根。省略時はリポジトリ直下の output/（通常は省略。scratch 実行用）")
    p.add_argument("overrides", nargs=argparse.REMAINDER, help="'--' の後に上書き（INFER / MACHINE schema のフラグ、--weight_dir / --input_dir）")
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
        check_infer_values(infer, machine)
        weight_dir = Path(path_over.get("--weight_dir", a.weight_dir))
        input_dir = Path(path_over.get("--input_dir", a.input_dir))
        if not (weight_dir / "net_G.pth").is_file():
            raise ConfigError(f"重みディレクトリに net_G.pth がありません: {weight_dir}（<run>/latest | <run>/best | <run>/weights/epoch_NNN を指定。infer_stage2.sh の WEIGHT_DIR か --weight_dir）")
        run_dir = find_run_dir(weight_dir)
        if run_dir is None:
            raise ConfigError(f"重みディレクトリの上に {LAUNCH_FILE} を持つ run が見つかりません: {weight_dir}")
        if not input_dir.is_dir():
            raise ConfigError(f"入力フォルダがありません: {input_dir}（infer_stage2.sh の INPUT_DIR か --input_dir）")
        if infer["teacher"] and not Path(machine["pcd1024_dir"]).is_dir():
            raise ConfigError(f"teacher=true ですが machines.yaml の pcd1024_dir がありません: {machine['pcd1024_dir']}（実 EID など教師の無い入力は --teacher false）")
        device = "cpu" if machine["gpu_gen"] == 0 else "cuda"
        launch_path = latest_launch(run_dir)
        mode, hu = net_config(load_yaml(launch_path), launch_path)
        scale, interp, image_size = upsample_config(run_dir)
        cur_train = validate("train", load_yaml(a.train), TRAIN)
        display = {k: cur_train[k] for k in DISPLAY_KEYS}
        if display["log.display_hu_min"] >= display["log.display_hu_max"]:
            raise ConfigError(f"{a.train} の log.display_hu_min < display_hu_max")
        if display["log.preview_bits"] not in (8, 16):
            raise ConfigError(f"{a.train} の log.preview_bits は 8 | 16")
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
    argv = ["--weight_dir", str(weight_dir), "--input_dir", str(input_dir), "--out_dir", str(out_dir), "--device", device,
            "--index_cache_dir", str(Path(machine["stage2_checkpoints_dir"]) / ".case_index"), "--pcd1024_dir", str(machine["pcd1024_dir"]),
            "--arch", mode["arch"], "--base_ch", str(mode["base_ch"]), "--n_pool", str(mode["n_pool"]), "--final_act", mode["final_act"]]
    if mode["residual"]:
        argv.append("--residual")
    argv += ["--hu_offset", str(hu["data.hu_offset"]), "--hu_min", str(hu["data.hu_min"]), "--hu_max", str(hu["data.hu_max"]),
             "--scale", str(scale), "--interp", interp, "--image_size", str(image_size),
             "--display_hu_min", str(display["log.display_hu_min"]), "--display_hu_max", str(display["log.display_hu_max"]), "--preview_bits", str(display["log.preview_bits"])]
    argv += to_argv(infer, INFER)  # 上書きは infer dict に反映済み

    with open(out_dir / "infer.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {"timestamp": now.isoformat(timespec="seconds"), "machine_name": a.machine, "device": device, "run_dir": str(run_dir), "launch": str(launch_path),
             "weight_dir": str(weight_dir), "input_dir": str(input_dir), "infer": infer, "machine": machine, "net": mode, "hu": hu,
             "upsample": {"scale": scale, "interp": interp, "image_size": image_size, "from": str(Path(run_dir) / DATASET_INFO_FILE)},
             "display": display, "overrides": list(overrides), "applied": applied, "argv": argv},
            f, allow_unicode=True, sort_keys=False,
        )

    cmd = [sys.executable, str(HERE / "inference_dir.py")] + argv
    print(f"[run_infer] {weight_dir} × {input_dir} → {out_dir}（全引数は infer.yaml の argv）", flush=True)
    os.chdir(HERE)
    os.execv(sys.executable, cmd)


if __name__ == "__main__":
    main()
