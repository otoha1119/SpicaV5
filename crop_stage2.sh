#!/bin/bash
# Stage 2 の固定プラーク領域だけを U-Net に通す。通常はホストから bash start2.sh crop。
set -euo pipefail

# ===== ここだけ書き換える =====
WEIGHT_DIR="/workspace/stage2/checkpoints/<run>/best"
# =============================

cd "$(dirname "$0")/stage2"
MACHINE="${SPICA_MACHINE:?SPICA_MACHINE が未設定です。bash start2.sh から起動してください}"
python run_crop.py \
    --machine "$MACHINE" \
    --crop configs/crop.yaml \
    --train configs/train.yaml \
    --machines ../configs/machines.yaml \
    --weight_dir "$WEIGHT_DIR" \
    -- "$@"
