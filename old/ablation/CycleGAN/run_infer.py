"""ablation/CycleGAN/run_infer.py — 推論の起動器（Docker 内で動く）。ablation/SRCycleGAN/run_infer.py の複製（2026-10-27。同解像度。512 入力は launch の upsample で補間）。

  1. configs/infer.yaml と ../../configs/machines.yaml（使うキーだけ）を schema で検証（既定値なし）、sh の上書きを反映
  2. 重みディレクトリ（--weight_dir）に net_G_A.pth があることを確認し、run（launch.yaml がある所）を探して
     **最新の launch の実効設定**から G_A の構成（network.netG / ngf / norm、data.hu_*）と、512 入力の補間方式（launch の upsample = eid1024_dir/manifest.yaml）を取る。
     表示設定（log.display_hu_min/max・preview_bits・diff_range_hu）は**現在の configs/train.yaml**
  3. デバイスは machines.yaml の gpu_gen（0 → cpu）。出力 <repo>/output/ablation/CycleGAN/<run>_<重み>_<入力名>_<時刻>/ に infer.yaml を保存して inference_dir.py を exec
"""

import argparse
import datetime
import os
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from configs.schema import INFER, MACHINE, TRAIN, ConfigError, apply_overrides, check_infer_values, to_argv, validate, validate_shared  # noqa: E402
from util.run_paths import ABLATION, INFER_WEIGHT, LAUNCH_FILE, checkpoints_root, find_run_dir, infer_output_dir, latest_launch  # noqa: E402

G_KEYS = {"network.netG": "--netG", "network.ngf": "--ngf", "network.norm": "--norm", "data.hu_offset": "--hu_offset", "data.hu_min": "--hu_min", "data.hu_max": "--hu_max"}
UPSAMPLE_KEYS = ("scale", "interp", "input_size")
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
    p = argparse.ArgumentParser(description=f"ablation {ABLATION} 推論の起動器")
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
        if launch.get("ablation") != ABLATION:
            raise ConfigError(f"{launch_path} は {ABLATION} の run ではありません（ablation: {launch.get('ablation')!r}）")
        ltrain = validate("train(launch)", launch["train"], TRAIN)
        up = launch.get("upsample")
        if not isinstance(up, dict) or any(k not in up for k in UPSAMPLE_KEYS):
            raise ConfigError(f"{launch_path} に upsample（scale / interp / input_size）がありません")
        g_argv = [x for k, flag in G_KEYS.items() for x in (flag, str(ltrain[k]))] + ["--up_scale", str(up["scale"]), "--up_interp", str(up["interp"]), "--up_input_size", str(up["input_size"])]
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
             "--index_cache_dir", str(checkpoints_root(machine) / ".case_index")] + g_argv + d_argv + to_argv(infer, INFER))
    with open(out_dir / "infer.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"ablation": ABLATION, "timestamp": now.isoformat(timespec="seconds"), "machine_name": a.machine, "device": device, "run_dir": str(run_dir),
                        "launch": str(launch_path), "weight_dir": str(weight_dir), "input_dir": str(input_dir), "infer": infer,
                        "generator": {k: ltrain[k] for k in G_KEYS}, "upsample": {k: up[k] for k in UPSAMPLE_KEYS}, "display": {k: cur[k] for k in DISPLAY_KEYS},
                        "overrides": list(overrides), "applied": applied, "argv": argv}, f, allow_unicode=True, sort_keys=False)
    print(f"[run_infer] {weight_dir} × {input_dir} → {out_dir}", flush=True)
    os.chdir(HERE)
    os.execv(sys.executable, [sys.executable, str(HERE / "inference_dir.py")] + argv)


if __name__ == "__main__":
    main()
