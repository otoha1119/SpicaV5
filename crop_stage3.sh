#!/bin/bash
# Stage 3 の固定プラーク領域だけを DDIM でサンプリング。通常はホストから bash start3.sh crop。
set -euo pipefail

# ===== ここだけ書き換える =====
WEIGHT_DIR="/workspace/stage3/checkpoints/<run>/best"
# =============================

cd "$(dirname "$0")/stage3"
MACHINE="${SPICA_MACHINE:?SPICA_MACHINE が未設定です。bash start3.sh から起動してください}"
python run_crop.py \
    --machine "$MACHINE" \
    --crop configs/crop.yaml \
    --train configs/train.yaml \
    --machines ../configs/machines.yaml \
    --weight_dir "$WEIGHT_DIR" \
    -- "$@"
