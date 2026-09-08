"""run_crop.py — パッチ切り出し（bash start.sh crop）の起動器（Docker 内で動く）。run_infer.py と同じ流儀。

  1. configs/crop.yaml（CROP schema + cases）と ../configs/machines.yaml を schema.py で検証する（既定値は無い）
  2. 重みディレクトリ（--weight_dir、crop_stage1.sh の変数）から run を探し、最新の launch の実効設定から G の構成・HU 正規化・表示設定
     （display_hu_min/max、preview_bits、diff_range_hu）を取る（学習時の TensorBoard / preview と同じ見た目になる）
  3. '--' の後の上書き（--weight_dir、CROP のフラグ --device / --patch / --panel_scale）を反映する
  4. 出力先 <run>/infer/<重みディレクトリ名>/crop/<実行時刻>/ を作り、実効設定を crop.yaml に保存して crop_patches.py を exec する
学習中に走らせてよい（start.sh は起動済みコンテナに exec するだけ。device cpu なら VRAM を使わない。出力は学習の書き込み先と別）。

使い方（通常は ../crop_stage1.sh 経由）:
  python run_crop.py --machine PC1 --crop configs/crop.yaml --machines ../configs/machines.yaml --weight_dir <run>/best -- [--device cuda] [--patch 72]
"""

import argparse
import datetime
import os
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from configs.schema import CROP, MACHINE, ConfigError, apply_overrides, check_crop_cases, check_crop_values, to_argv, validate  # noqa: E402
from run_infer import generator_argv, load_yaml  # noqa: E402
from util.run_paths import LAUNCH_FILE, crop_dir, find_run_dir, latest_launch  # noqa: E402

DISPLAY_TRAIN_KEYS = {"log.display_hu_min": "--display_hu_min", "log.display_hu_max": "--display_hu_max", "log.preview_bits": "--preview_bits"}  # 表示は学習時と同じ
PATH_FLAGS = ("--weight_dir",)


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


def main():
    p = argparse.ArgumentParser(description="Stage 1 (FE-GAN) パッチ切り出しの起動器")
    p.add_argument("--machine", required=True, help="configs/machines.yaml のエントリ名（pcd_dir / eid_dir / gpu_gen を使う）")
    p.add_argument("--crop", required=True, help="切り出し設定 YAML（configs/crop.yaml）")
    p.add_argument("--machines", required=True, help="マシン定義 YAML")
    p.add_argument("--weight_dir", required=True, help="重みディレクトリ（net_G.pth がある所）")
    p.add_argument("overrides", nargs=argparse.REMAINDER, help="'--' の後に crop_patches.py の上書き（--weight_dir、CROP schema のフラグ）")
    a = p.parse_args()
    overrides = a.overrides[1:] if a.overrides and a.overrides[0] == "--" else a.overrides

    try:
        data = load_yaml(a.crop)
        if not isinstance(data, dict):
            raise ConfigError(f"[crop] トップレベルが dict ではありません: {a.crop}")
        cases = check_crop_cases(data.pop("cases", None))
        crop = validate("crop", data, CROP)
        machines = load_yaml(a.machines)
        if not isinstance(machines, dict) or a.machine not in machines:
            raise ConfigError(f"machines.yaml にエントリ '{a.machine}' がありません。候補: {list(machines) if isinstance(machines, dict) else '(不正な形式)'}")
        machine = validate(f"machines.{a.machine}", machines[a.machine], MACHINE)
        path_over, rest = split_overrides(overrides)
        applied = apply_overrides(rest, {"crop": (crop, CROP)})
        check_crop_values(crop)
        if crop["device"] == "cuda" and machine["gpu_gen"] == 0:
            raise ConfigError(f"device=cuda ですが machines.yaml の '{a.machine}' は gpu_gen 0（CPU）です")
        weight_dir = Path(path_over.get("--weight_dir", a.weight_dir))
        if not (weight_dir / "net_G.pth").is_file():
            raise ConfigError(f"重みディレクトリに net_G.pth がありません: {weight_dir}（<run>/latest | <run>/best | <run>/weights/epoch_NNN を指定）")
        run_dir = find_run_dir(weight_dir)
        if run_dir is None:
            raise ConfigError(f"重みディレクトリの上に {LAUNCH_FILE} を持つ run が見つかりません: {weight_dir}")
        for k in ("pcd_dir", "eid_dir"):
            if not Path(machine[k]).is_dir():
                raise ConfigError(f"machines.yaml の {k} がありません: {machine[k]}")
        launch_path = latest_launch(run_dir)
        launch = load_yaml(launch_path)
        g_argv, g_cfg = generator_argv(launch)
        disp_argv, disp_cfg = [], {}
        for k, flag in DISPLAY_TRAIN_KEYS.items():
            if k not in launch["train"]:
                raise ConfigError(f"launch.yaml の train に {k} がありません（古い run?）")
            disp_argv += [flag, str(launch["train"][k])]
            disp_cfg[k] = launch["train"][k]
    except ConfigError as e:
        print(f"[run_crop] 設定エラー: {e}", file=sys.stderr)
        sys.exit(2)

    jst = datetime.timezone(datetime.timedelta(hours=9), name="JST")
    now = datetime.datetime.now(jst)
    out_dir = crop_dir(run_dir, weight_dir, now)
    if out_dir.exists():
        print(f"[run_crop] 設定エラー: 出力先が既に存在します（同一秒の再実行）: {out_dir}", file=sys.stderr)
        sys.exit(2)
    out_dir.mkdir(parents=True)
    argv = ["--weight_dir", str(weight_dir), "--pcd_dir", str(machine["pcd_dir"]), "--eid_dir", str(machine["eid_dir"]), "--out_dir", str(out_dir)]
    argv += g_argv + disp_argv + to_argv(crop, CROP)
    for c in cases:
        argv += ["--case", c["pcd"], str(c["pcd_x"]), str(c["pcd_y"]), c["eid"], str(c["eid_x"]), str(c["eid_y"])]

    with open(out_dir / "crop.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {"timestamp": now.isoformat(timespec="seconds"), "machine_name": a.machine, "run_dir": str(run_dir), "launch": str(launch_path),
             "weight_dir": str(weight_dir), "crop": crop, "cases": cases, "generator": g_cfg, "display": disp_cfg,
             "overrides": list(overrides), "applied": applied, "argv": argv},
            f, allow_unicode=True, sort_keys=False,
        )
    cmd = [sys.executable, str(HERE / "crop_patches.py")] + argv
    print(f"[run_crop] {weight_dir} → {out_dir}（全引数は crop.yaml の argv）", flush=True)
    os.execv(sys.executable, cmd)


if __name__ == "__main__":
    main()
