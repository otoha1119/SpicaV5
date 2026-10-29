#!/bin/bash
# stage1_v2/infer.sh — Docker コンテナ内の推論入口。
#   入力は Stage 1 と同じ PCD512 の <case>/<slice>.png。
#   方式は configs/infer.yaml。出力は <repo>/output/stage1_v2/。
set -euo pipefail
WEIGHT_DIR="/workspace/stage1_v2/checkpoints/2026_1027_0000/latest"   # --weight_dir で上書き可
INPUT_DIR="/workspace/DataSet/PCD512_v2"
cd "$(dirname "$0")"
MACHINE="${SPICA_MACHINE:?SPICA_MACHINE が未設定です。bash start1_v2.sh から起動してください}"
python run_infer.py \
    --machine "$MACHINE" \
    --infer configs/infer.yaml \
    --train configs/train.yaml \
    --machines ../configs/machines.yaml \
    --weight_dir "$WEIGHT_DIR" \
    --input_dir "$INPUT_DIR" \
    -- "$@"
