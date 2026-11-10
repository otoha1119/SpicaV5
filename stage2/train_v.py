"""stage2/train_v.py — V の学習ループ。**run_train_v.py 経由で起動する**（直接叩くと launch.yaml が無くて止まる）。本体は util/train_loop.py。

  python stage2/train_v.py --run_dir /workspace/runs/stage2/v/<run> --launch <run>/launch.yaml [--resume_state <重みディレクトリ>/state.pth] [--require_cuda]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from util.train_loop import main  # noqa: E402

if __name__ == "__main__":
    main("v")
