"""stage2/run_train_v.py — V の学習の起動器（Docker 内で動く。ホストからは bash start2.sh v）。本体は util/launcher.py。

  python stage2/run_train_v.py --machine PC1 --data stage2/configs/data.yaml --cfg stage2/configs/v.yaml --shared configs/shared.yaml --machines configs/machines.yaml [--resume <run> [--resume_tag latest]] -- [--flag value ...]
  → 設定とディスクを検査 → runs/stage2/v/<run>/launch.yaml → python stage2/train_v.py --run_dir /workspace/runs/stage2/v/<run> --launch … を exec
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from util.launcher import main  # noqa: E402

if __name__ == "__main__":
    main("v")
