#!/bin/bash
# stage1_v3/infer.sh — Docker コンテナ内の推論入口。
#   入力は <case>/<slice>.png（1ch uint16、512）。PCD なら apply_degrade: true（P → G）、実 EID なら --apply_degrade false（G だけ、恒等の確認）。
#   方式は configs/infer.yaml。出力は <repo>/output/stage1_v3/。
set -euo pipefail
WEIGHT_DIR="/workspace/stage1_v3/checkpoints/2026_1031_0000/latest"   # --weight_dir で上書き可
INPUT_DIR="/workspace/DataSet/PCD512_v2"                             # --input_dir で上書き可
cd "$(dirname "$0")"
MACHINE="${SPICA_MACHINE:?SPICA_MACHINE が未設定です。bash start1_v3.sh から起動してください}"
python run_infer.py \
    --machine "$MACHINE" \
    --infer configs/infer.yaml \
    --train configs/train.yaml \
    --machines ../configs/machines.yaml \
    --weight_dir "$WEIGHT_DIR" \
    --input_dir "$INPUT_DIR" \
    -- "$@"
