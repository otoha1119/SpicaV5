"""stage1/run_bridge.py — 型 X の疑似 EID の作成の起動器（Docker 内で動く。ホストからは bash start.sh bridge --enc_run <run> --dec_run <run> [--flag value]）。

  python stage1/run_bridge.py --machine PC1 --bridge stage1/configs/bridge.yaml --shared configs/shared.yaml --machines configs/machines.yaml -- --enc_run <PCD の描き手> --dec_run <実 EID の描き手>

  bridge.yaml / shared.yaml / machines.yaml を schema で検証 → 上書きを反映 → 値域を検査 → bridge.run を呼ぶ。
  ネットワークと拡散の設定は、重みの state.pth の実効設定から取る。実効値と確認用の数字は <組>/manifest.yaml に残る。
  往復の確認（元の絵に戻るか）: --enc_run と --dec_run を同じ run にする。実 EID で試すなら --input_dir /workspace/DataSet/EID_v5 --cases EID-049
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bridge import run  # noqa: E402
from configs.schema import BRIDGE, MACHINE, SHARED, ConfigError, apply_overrides, check_bridge_values, validate, validate_shared  # noqa: E402
from util.run_paths import CKPT_KINDS, LAUNCH_FILE, ckpt_dir, saved_tags  # noqa: E402


def load_yaml(path):
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"設定ファイルが見つかりません: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def weight_dir_of(machine, run_name, tag):
    run_dir = Path(machine["painter_checkpoints_dir"]) / run_name
    if not (run_dir / LAUNCH_FILE).is_file():
        raise ConfigError(f"run が見つかりません: {run_dir}")
    wdir = ckpt_dir(run_dir, tag)
    missing = [k for k in CKPT_KINDS if not (wdir / k).is_file()]
    if missing:
        raise ConfigError(f"重みが揃っていません: {wdir} に {missing} が無い（保存済み tag: {saved_tags(run_dir)}）")
    return wdir


def main():
    p = argparse.ArgumentParser(description="型 X の疑似 EID の作成の起動器")
    p.add_argument("--machine", required=True)
    p.add_argument("--bridge", required=True)
    p.add_argument("--shared", required=True)
    p.add_argument("--machines", required=True)
    p.add_argument("overrides", nargs=argparse.REMAINDER)
    a = p.parse_args()
    overrides = a.overrides[1:] if a.overrides and a.overrides[0] == "--" else a.overrides
    try:
        br = validate("bridge", load_yaml(a.bridge), BRIDGE)
        shared = validate_shared("shared", load_yaml(a.shared), SHARED)
        machines = load_yaml(a.machines)
        if not isinstance(machines, dict) or a.machine not in machines:
            raise ConfigError(f"machines.yaml にエントリ '{a.machine}' がありません")
        machine = validate_shared(f"machines.{a.machine}", machines[a.machine], MACHINE)
        apply_overrides(overrides, {"bridge": (br, BRIDGE), "shared": (shared, SHARED), "machine": (machine, MACHINE)})
        check_bridge_values(br, shared, machine)
        enc_dir = weight_dir_of(machine, br["bridge.enc_run"], br["bridge.enc_tag"])
        dec_dir = weight_dir_of(machine, br["bridge.dec_run"], br["bridge.dec_tag"])
    except ConfigError as e:
        print(f"[bridge] 設定エラー: {e}", file=sys.stderr)
        sys.exit(2)
    if machine["gpu_gen"] != 0 and not torch.cuda.is_available():
        raise RuntimeError("machines.yaml の gpu_gen ≠ 0 なのに CUDA が使えません")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    try:
        run(br, shared, machine, enc_dir, dec_dir, device)
    except RuntimeError as e:
        print(f"[bridge] エラー: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
