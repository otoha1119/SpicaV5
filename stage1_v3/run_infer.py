"""stage1_v3/run_infer.py — 推論起動器。

  1. configs/infer.yaml と ../configs/machines.yaml（使うキーだけ）を schema で検証（既定値なし）、sh の上書きを反映
  2. 重みディレクトリ（--weight_dir）に net_G.pth があることを確認し、run の最新 launch から G の構成・P の範囲・HU 正規化を、dataset_info.yaml から学習画像サイズを取る。
     表示設定（log.display_hu_min/max・preview_bits・diff_range_hu）は**現在の configs/train.yaml**
  3. 出力 <repo>/output/stage1_v3/<run>_<重み>_<入力名>_<時刻>/ に infer.yaml を保存して inference_dir.py を exec
"""

import argparse
import datetime
import os
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from configs.schema import INFER, LEGACY_TRAIN_DEFAULTS, MACHINE, TRAIN, ConfigError, apply_overrides, check_infer_values, to_argv, validate, validate_shared  # noqa: E402
from util.run_paths import STAGE, DATASET_INFO_FILE, INFER_WEIGHT, LAUNCH_FILE, checkpoints_root, find_run_dir, infer_output_dir, latest_launch  # noqa: E402

G_KEYS = {"network.netG": "--netG", "network.ngf": "--ngf", "network.norm": "--norm", "data.hu_offset": "--hu_offset", "data.hu_min": "--hu_min", "data.hu_max": "--hu_max"}
P_KEYS = {"degrade.enabled": "--p_enabled", "degrade.mode": "--p_mode",
          "degrade.bilateral_sigma_color_hu_min": "--p_bil_color_min", "degrade.bilateral_sigma_color_hu_max": "--p_bil_color_max",
          "degrade.bilateral_sigma_space_px_min": "--p_bil_space_min", "degrade.bilateral_sigma_space_px_max": "--p_bil_space_max", "degrade.blur_sigma_px_min": "--p_blur_min", "degrade.blur_sigma_px_max": "--p_blur_max",
          "degrade.noise_sigma_hu_min": "--p_noise_min", "degrade.noise_sigma_hu_max": "--p_noise_max", "degrade.noise_corr_px_min": "--p_corr_min",
          "degrade.noise_corr_px_max": "--p_corr_max", "degrade.noise_body_only": "--p_body_only", "degrade.body_threshold_hu": "--p_body_threshold_hu"}
DISPLAY_KEYS = {"log.display_hu_min": "--display_hu_min", "log.display_hu_max": "--display_hu_max", "log.preview_bits": "--preview_bits", "log.diff_range_hu": "--diff_range_hu"}
PATH_FLAGS = ("--weight_dir", "--input_dir")


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


def main():
    p = argparse.ArgumentParser(description=f"{STAGE} 推論の起動器")
    p.add_argument("--machine", required=True)
    p.add_argument("--infer", required=True)
    p.add_argument("--train", required=True, help="現在の configs/train.yaml（表示の設定だけ使う）")
    p.add_argument("--machines", required=True)
    p.add_argument("--weight_dir", required=True)
    p.add_argument("--input_dir", required=True)
    p.add_argument("--out_root", default=None, help="出力の根（scratch 実行用。通常は省略 = <repo>/output/）")
    p.add_argument("overrides", nargs=argparse.REMAINDER)
    a = p.parse_args()
    overrides = a.overrides[1:] if a.overrides and a.overrides[0] == "--" else a.overrides
    try:
        infer = validate("infer", load_yaml(a.infer), INFER)
        machines = load_yaml(a.machines)
        if not isinstance(machines, dict) or a.machine not in machines:
            raise ConfigError(f"machines.yaml にエントリ '{a.machine}' がありません。候補: {list(machines) if isinstance(machines, dict) else '(不正な形式)'}")
        machine = validate_shared(f"machines.{a.machine}", machines[a.machine], MACHINE)
        path_over, overrides = split_overrides(overrides)
        applied = apply_overrides(overrides, {"infer": (infer, INFER)})
        check_infer_values(infer, machine)
        weight_dir = Path(path_over.get("--weight_dir", a.weight_dir))
        input_dir = Path(path_over.get("--input_dir", a.input_dir))
        if not (weight_dir / INFER_WEIGHT).is_file():
            raise ConfigError(f"重みディレクトリに {INFER_WEIGHT} がありません: {weight_dir}（<run>/latest | <run>/best | <run>/weights/epoch_NNN）")
        run_dir = find_run_dir(weight_dir)
        if run_dir is None:
            raise ConfigError(f"重みディレクトリの上に {LAUNCH_FILE} を持つ run が見つかりません: {weight_dir}")
        if not input_dir.is_dir():
            raise ConfigError(f"入力フォルダがありません: {input_dir}")
        launch_path = latest_launch(run_dir)
        launch = load_yaml(launch_path)
        if launch.get("stage") != STAGE:
            raise ConfigError(f"{launch_path} は {STAGE} の run ではありません")
        for key, value in LEGACY_TRAIN_DEFAULTS.items():  # 旧 run（degrade.mode 以前）の launch も推論できるように
            launch["train"].setdefault(key, value)
        ltrain = validate("train(launch)", launch["train"], TRAIN)
        info_path = run_dir / DATASET_INFO_FILE
        info = load_yaml(info_path)
        hw = info.get("image_hw") if isinstance(info, dict) else None
        if not (isinstance(hw, list) and len(hw) == 2 and hw[0] == hw[1] and isinstance(hw[0], int)):
            raise ConfigError(f"{info_path} の image_hw が不正です: {hw}")
        if infer["apply_degrade"] and not ltrain["degrade.enabled"]:
            raise ConfigError("apply_degrade: true ですが、この run は degrade.enabled: false で学習しています（P の範囲が無い）。--apply_degrade false にしてください")
        g_argv = [x for k, flag in G_KEYS.items() for x in (flag, str(ltrain[k]))] + ["--input_size", str(hw[0])]
        p_argv = [x for k, flag in P_KEYS.items() for x in (flag, str(ltrain[k]))]
        cur = validate("train", load_yaml(a.train), TRAIN)
        d_argv = [x for k, flag in DISPLAY_KEYS.items() for x in (flag, str(cur[k]))]
        device = "cpu" if machine["gpu_gen"] == 0 else "cuda"
    except (ConfigError, KeyError, TypeError) as e:
        print(f"[run_infer] 設定エラー: {e}", file=sys.stderr)
        sys.exit(2)

    jst = datetime.timezone(datetime.timedelta(hours=9), name="JST")
    now = datetime.datetime.now(jst)
    out_dir = infer_output_dir(run_dir, weight_dir, input_dir, now, root=a.out_root)
    if out_dir.exists():
        print(f"[run_infer] 設定エラー: 出力先が既に存在します（同一秒の再実行）: {out_dir}", file=sys.stderr)
        sys.exit(2)
    out_dir.mkdir(parents=True)
    argv = (["--weight_dir", str(weight_dir), "--input_dir", str(input_dir), "--out_dir", str(out_dir), "--device", device,
             "--index_cache_dir", str(checkpoints_root(machine) / ".case_index")] + g_argv + p_argv + d_argv + to_argv(infer, INFER))
    with open(out_dir / "infer.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"stage": STAGE, "timestamp": now.isoformat(timespec="seconds"), "machine_name": a.machine, "device": device, "run_dir": str(run_dir),
                        "launch": str(launch_path), "weight_dir": str(weight_dir), "input_dir": str(input_dir), "infer": infer,
                        "generator": {k: ltrain[k] for k in G_KEYS}, "degrade": {k: ltrain[k] for k in P_KEYS}, "image_hw": hw, "display": {k: cur[k] for k in DISPLAY_KEYS},
                        "overrides": list(overrides), "applied": applied, "argv": argv}, f, allow_unicode=True, sort_keys=False)
    print(f"[run_infer] {weight_dir} × {input_dir} → {out_dir}", flush=True)
    os.chdir(HERE)
    os.execv(sys.executable, [sys.executable, str(HERE / "inference_dir.py")] + argv)


if __name__ == "__main__":
    main()
