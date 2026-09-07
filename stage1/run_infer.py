"""run_infer.py — Stage 1 推論（症例丸ごと）の起動器（Docker 内で動く）。[SpicaV5 新規]

設計: docs/plans/20260907_inference-plan.md
  1. configs/infer.yaml（方式）と ../configs/machines.yaml（checkpoints_dir・gpu_gen）を configs/schema.py で検証する（既定値なし）
  2. 重みディレクトリ（--weight_dir、infer_stage1.sh の変数）に net_G.pth があることを確認し、そこから run（launch.yaml がある所）を探して
     **最新の launch（launch_resume_*.yaml があればそれ）の実効設定**から G の構成（netG / ngf / input_nc / output_nc / norm / final_norm_act / hu_*）を取る
     （学習時と同じ G を組むため。今の train.yaml / mode.yaml は見ない。実効値 = yaml + 学習時の上書き。F-02 / F-10）
  3. デバイスは machines.yaml の gpu_gen から決める（0 → cpu、それ以外 → cuda）
  4. 出力形式（--output_format png | dicom | both）と元 DICOM ルート（--dicom_dir、dicom / both のとき必須）も infer_stage1.sh の変数から受ける
  5. sh からの上書き（infer.yaml のキーのフラグ、および --weight_dir / --input_dir / --output_format / --dicom_dir）を反映し、解決済み設定を
     <run>/infer/<重みディレクトリ名>/<入力フォルダ名>/<実行時刻>/infer.yaml に保存してから inference_dir.py を exec する（実行ごとに別ディレクトリ。F-13）

使い方（通常は ../infer_stage1.sh 経由）:
  python run_infer.py --machine PC1 --infer configs/infer.yaml --machines ../configs/machines.yaml \
      --weight_dir /workspace/stage1/checkpoints/2026_0907_1742/weights/epoch_030 --input_dir /workspace/DataSet/PCD512_v2 \
      --output_format png --dicom_dir "" -- --mode full --max_slices 4
"""

import argparse
import datetime
import os
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from configs.schema import INFER, MACHINE, ConfigError, apply_overrides, check_infer_values, to_argv, validate  # noqa: E402
from util.run_paths import LAUNCH_FILE, find_run_dir, infer_dir, latest_launch  # noqa: E402

# launch.yaml の train / mode セクション（平坦キー）→ inference_dir.py のフラグ
G_TRAIN_KEYS = {"network.ngf": "--ngf", "network.input_nc": "--input_nc", "network.output_nc": "--output_nc", "network.norm": "--norm",
                "data.hu_offset": "--hu_offset", "data.hu_min": "--hu_min", "data.hu_max": "--hu_max"}
G_MODE_KEYS = {"netG": "--netG"}
G_MODE_BOOL_KEYS = {"final_norm_act": "--final_norm_act"}
PATH_FLAGS = ("--weight_dir", "--input_dir", "--output_format", "--dicom_dir")  # sh の変数を --flag で上書きできる（start.sh の引数が最優先）
OUTPUT_FORMATS = ("png", "dicom", "both")


def load_yaml(path):
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"設定ファイルが見つかりません: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def split_overrides(tokens):
    """上書き列から sh 変数系のフラグ（--weight_dir / --input_dir / --output_format / --dicom_dir）を取り出し、残り（INFER のフラグ）を返す。
    残りは apply_overrides で infer dict に反映する（schema に無いフラグはそこでエラー）。"""
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


def generator_argv(launch):
    """run の launch.yaml から G の構成を取り出してフラグ列にする。無いキーはエラー（古い run）。"""
    train, mode = launch.get("train"), launch.get("mode")
    if not isinstance(train, dict) or not isinstance(mode, dict):
        raise ConfigError("launch.yaml に train / mode セクションがありません")
    argv, cfg = [], {}
    for k, flag in G_TRAIN_KEYS.items():
        if k not in train:
            raise ConfigError(f"launch.yaml の train に {k} がありません（古い run?）")
        argv += [flag, str(train[k])]
        cfg[k] = train[k]
    for k, flag in G_MODE_KEYS.items():
        if k not in mode:
            raise ConfigError(f"launch.yaml の mode に {k} がありません（古い run?）")
        argv += [flag, str(mode[k])]
        cfg[k] = mode[k]
    for k, flag in G_MODE_BOOL_KEYS.items():
        if k not in mode:
            raise ConfigError(f"launch.yaml の mode に {k} がありません（古い run?）")
        if mode[k]:
            argv.append(flag)
        cfg[k] = mode[k]
    return argv, cfg


def main():
    p = argparse.ArgumentParser(description="Stage 1 (FE-GAN) 推論の起動器")
    p.add_argument("--machine", required=True, help="configs/machines.yaml のエントリ名（gpu_gen でデバイスを決める）")
    p.add_argument("--infer", required=True, help="推論方式 YAML（configs/infer.yaml）")
    p.add_argument("--machines", required=True, help="マシン定義 YAML")
    p.add_argument("--weight_dir", required=True, help="重みディレクトリ（net_G.pth がある所。<run>/latest | <run>/best | <run>/weights/epoch_NNN）")
    p.add_argument("--input_dir", required=True, help="処理する PNG 群のフォルダ")
    p.add_argument("--output_format", required=True, help="png | dicom | both（infer_stage1.sh の OUTPUT_FORMAT）")
    p.add_argument("--dicom_dir", required=True, help="元 DICOM ルート（dicom / both のとき必須。png のときは空文字でよい）")
    p.add_argument("overrides", nargs=argparse.REMAINDER, help="'--' の後に inference_dir.py の引数（INFER schema のフラグ、--weight_dir / --input_dir / --output_format / --dicom_dir）")
    a = p.parse_args()
    overrides = a.overrides[1:] if a.overrides and a.overrides[0] == "--" else a.overrides

    try:
        infer = validate("infer", load_yaml(a.infer), INFER)
        machines = load_yaml(a.machines)
        if not isinstance(machines, dict) or a.machine not in machines:
            raise ConfigError(f"machines.yaml にエントリ '{a.machine}' がありません。候補: {list(machines) if isinstance(machines, dict) else '(不正な形式)'}")
        machine = validate(f"machines.{a.machine}", machines[a.machine], MACHINE)
        path_over, overrides = split_overrides(overrides)
        applied = apply_overrides(overrides, {"infer": (infer, INFER)})  # 上書きを実効値に反映（F-02 と同じ方式）
        check_infer_values(infer)  # F-07: stride ≤ size など
        weight_dir = Path(path_over.get("--weight_dir", a.weight_dir))
        input_dir = Path(path_over.get("--input_dir", a.input_dir))
        output_format = path_over.get("--output_format", a.output_format)
        dicom_dir = path_over.get("--dicom_dir", a.dicom_dir)
        if output_format not in OUTPUT_FORMATS:
            raise ConfigError(f"OUTPUT_FORMAT は png | dicom | both のどれか: {output_format!r}")
        if output_format != "png":
            if not dicom_dir or not Path(dicom_dir).is_dir():
                raise ConfigError(f"OUTPUT_FORMAT={output_format} には元 DICOM ルート（infer_stage1.sh の DICOM_DIR）が必要です: {dicom_dir!r}")
            dicom_dir = str(Path(dicom_dir))
        if not (weight_dir / "net_G.pth").is_file():
            raise ConfigError(f"重みディレクトリに net_G.pth がありません: {weight_dir}（<run>/latest | <run>/best | <run>/weights/epoch_NNN を指定）")
        run_dir = find_run_dir(weight_dir)
        if run_dir is None:
            raise ConfigError(f"重みディレクトリの上に {LAUNCH_FILE} を持つ run が見つかりません: {weight_dir}")
        if not input_dir.is_dir():
            raise ConfigError(f"入力フォルダがありません: {input_dir}")
        device = "cpu" if machine["gpu_gen"] == 0 else "cuda"
        launch_path = latest_launch(run_dir)  # 最新の実効設定（F-10）
        g_argv, g_cfg = generator_argv(load_yaml(launch_path))
    except ConfigError as e:
        print(f"[run_infer] 設定エラー: {e}", file=sys.stderr)
        sys.exit(2)

    jst = datetime.timezone(datetime.timedelta(hours=9), name="JST")
    now = datetime.datetime.now(jst)
    out_dir = infer_dir(run_dir, weight_dir, input_dir, now)  # 実行時刻つき（F-13）
    if out_dir.exists():
        print(f"[run_infer] 設定エラー: 出力先が既に存在します（同一秒の再実行）: {out_dir}", file=sys.stderr)
        sys.exit(2)
    out_dir.mkdir(parents=True)
    argv = (["--weight_dir", str(weight_dir), "--input_dir", str(input_dir), "--out_dir", str(out_dir), "--device", device,
             "--index_cache_dir", str(Path(machine["checkpoints_dir"]) / ".case_index"),
             "--output_format", output_format, "--dicom_dir", dicom_dir]
            + g_argv + to_argv(infer, INFER))  # 上書きは infer dict に反映済み

    with open(out_dir / "infer.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {"timestamp": now.isoformat(timespec="seconds"), "machine_name": a.machine, "device": device, "run_dir": str(run_dir),
             "launch": str(launch_path), "weight_dir": str(weight_dir), "input_dir": str(input_dir), "output_format": output_format, "dicom_dir": dicom_dir,
             "infer": infer, "generator": g_cfg, "overrides": list(overrides), "applied": applied, "argv": argv},
            f, allow_unicode=True, sort_keys=False,
        )

    cmd = [sys.executable, str(HERE / "inference_dir.py")] + argv
    print("[run_infer] " + " ".join(cmd), flush=True)
    os.chdir(HERE)
    os.execv(sys.executable, cmd)


if __name__ == "__main__":
    main()
