"""stage2/run_infer.py — 推論の起動器（Docker 内で動く。ホストからは bash start2.sh infer --run <run> --input_dir <フォルダ> [--flag value]）。

  python stage2/run_infer.py --machine PC1 --infer stage2/configs/infer.yaml --machines configs/machines.yaml -- --run 2026_1113_0900 --input_dir /workspace/DataSet/EID_v5 --cases EID-049
  infer.yaml / machines.yaml を schema で検証 → 上書きを反映 → run と重みの有無を検査 → inference_dir.run。
  ネットワーク・HU の物差し・補間・明るさの広さは、重みの state.pth の実効設定から取る（V の重みディレクトリには U の写しが入っている）。
"""

import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from configs.schema import INFER, MACHINE, ConfigError, apply_overrides, check_infer_values, validate  # noqa: E402
from inference_dir import run  # noqa: E402
from util.launcher import load_machine, load_yaml  # noqa: E402
from util.run_paths import LAUNCH_FILE, ckpt_dir, kind_root, missing_files, saved_tags  # noqa: E402


def find_weights(machine, kind, run_name, tag):
    run_dir = kind_root(machine, kind) / run_name
    if not (run_dir / LAUNCH_FILE).is_file():
        raise ConfigError(f"run が見つかりません: {run_dir}")
    wdir = ckpt_dir(run_dir, tag)
    miss = missing_files(wdir, kind)
    if miss:
        raise ConfigError(f"重みが揃っていません: {wdir} に {miss} が無い（保存済み tag: {saved_tags(run_dir, kind)}）")
    return run_dir, wdir


def main():
    p = argparse.ArgumentParser(description="新 Stage 2 の推論の起動器")
    p.add_argument("--machine", required=True)
    p.add_argument("--infer", required=True)
    p.add_argument("--machines", required=True)
    p.add_argument("overrides", nargs=argparse.REMAINDER)
    a = p.parse_args()
    overrides = a.overrides[1:] if a.overrides and a.overrides[0] == "--" else a.overrides
    try:
        infer = validate("infer", load_yaml(a.infer), INFER)
        machine = load_machine(a.machines, a.machine)
        apply_overrides(overrides, {"infer": (infer, INFER), "machine": (machine, MACHINE)})
        check_infer_values(infer, machine)
        if not Path(infer["infer.input_dir"]).is_dir():
            raise ConfigError(f"入力フォルダがありません: {infer['infer.input_dir']}")
        run_dir, wdir = find_weights(machine, infer["infer.kind"], infer["infer.run"], infer["infer.tag"])
    except ConfigError as e:
        print(f"[infer] 設定エラー: {e}", file=sys.stderr)
        sys.exit(2)
    if machine["gpu_gen"] != 0 and not torch.cuda.is_available():
        raise RuntimeError("machines.yaml の gpu_gen ≠ 0 なのに CUDA が使えません")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    try:
        run(infer, wdir, run_dir.name, device)
    except RuntimeError as e:
        print(f"[infer] エラー: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
