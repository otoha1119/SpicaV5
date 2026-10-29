#!/bin/bash
# stage1_v2/train.sh — Docker コンテナ内の学習入口。
# ホストから: bash start1_v2.sh <machine> [resume <run> [tag]] [--flag value]
set -euo pipefail
cd "$(dirname "$0")"
MACHINE="${SPICA_MACHINE:?SPICA_MACHINE が未設定です。bash start1_v2.sh から起動してください}"
RESUME_ARGS=()
if [ -n "${SPICA_RESUME:-}" ]; then
  RESUME_ARGS=(--resume "$SPICA_RESUME" --resume_tag "${SPICA_RESUME_TAG:-latest}")
fi
python run_train.py \
    --machine "$MACHINE" \
    --train configs/train.yaml \
    --machines ../configs/machines.yaml \
    "${RESUME_ARGS[@]}" \
    -- "$@"
