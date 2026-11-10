"""stage1/run_generate.py — 疑似 EID の作成の起動器（Docker 内で動く。ホストからは bash start.sh generate --run <run> [--flag value]）。

  python stage1/run_generate.py --machine PC1 --generate stage1/configs/generate.yaml --shared configs/shared.yaml --machines configs/machines.yaml -- --run 2026_1112_0030 --blur_px 2 --noise_hu 50

  generate.yaml / shared.yaml / machines.yaml を schema で検証 → 上書きを反映 → 値域と「σ_a ≥ 2·σ_b」（設計書 F-2）を検査 → generate.run を呼ぶ。
  ネットワークと拡散の設定は、重みの state.pth の実効設定から取る。実効値は <組>/manifest.yaml に残る。
  実 EID の描き直しは入力フォルダを替えるだけ: --input_dir /workspace/DataSet/EID_v5 --cases EID-049
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from configs.schema import GENERATE, MACHINE, SHARED, ConfigError, apply_overrides, check_generate_values, validate, validate_shared  # noqa: E402
from generate import run  # noqa: E402
from util.run_paths import CKPT_KINDS, LAUNCH_FILE, ckpt_dir, saved_tags  # noqa: E402


def load_yaml(path):
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"設定ファイルが見つかりません: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    p = argparse.ArgumentParser(description="疑似 EID の作成の起動器")
    p.add_argument("--machine", required=True)
    p.add_argument("--generate", required=True)
    p.add_argument("--shared", required=True)
    p.add_argument("--machines", required=True)
    p.add_argument("overrides", nargs=argparse.REMAINDER)
    a = p.parse_args()
    overrides = a.overrides[1:] if a.overrides and a.overrides[0] == "--" else a.overrides
    try:
        gen = validate("generate", load_yaml(a.generate), GENERATE)
        shared = validate_shared("shared", load_yaml(a.shared), SHARED)
        machines = load_yaml(a.machines)
        if not isinstance(machines, dict) or a.machine not in machines:
            raise ConfigError(f"machines.yaml にエントリ '{a.machine}' がありません")
        machine = validate_shared(f"machines.{a.machine}", machines[a.machine], MACHINE)
        apply_overrides(overrides, {"generate": (gen, GENERATE), "shared": (shared, SHARED), "machine": (machine, MACHINE)})
        check_generate_values(gen, shared, machine)
        run_dir = Path(machine["painter_checkpoints_dir"]) / gen["generate.run"]
        if not (run_dir / LAUNCH_FILE).is_file():
            raise ConfigError(f"run が見つかりません: {run_dir}")
        weight_dir = ckpt_dir(run_dir, gen["generate.tag"])
        missing = [k for k in CKPT_KINDS if not (weight_dir / k).is_file()]
        if missing:
            raise ConfigError(f"重みが揃っていません: {weight_dir} に {missing} が無い（保存済み tag: {saved_tags(run_dir)}）")
    except ConfigError as e:
        print(f"[generate] 設定エラー: {e}", file=sys.stderr)
        sys.exit(2)
    if machine["gpu_gen"] != 0 and not torch.cuda.is_available():
        raise RuntimeError("machines.yaml の gpu_gen ≠ 0 なのに CUDA が使えません")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    try:
        run(gen, shared, machine, weight_dir, run_dir.name, device)
    except RuntimeError as e:
        print(f"[generate] エラー: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
