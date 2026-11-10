#!/bin/bash
# ablation/SRCycleGAN/train.sh — SR-CycleGAN 学習の入口（**Docker コンテナ内で実行**。通常はホストの bash start_ablation.sh <マシン> SRCycleGAN から呼ばれる）。
#   手で叩くとき（bash start_ablation.sh <マシン> SRCycleGAN shell で入ってから、/workspace で）: bash ablation/SRCycleGAN/train.sh [--flag value ...]
#   再開: 環境変数 SPICA_RESUME=<run>（と SPICA_RESUME_TAG=latest|best|<epoch>）。ホストからは bash start_ablation.sh <マシン> SRCycleGAN resume <run> [tag]
#   マシン名は環境変数 SPICA_MACHINE（start_ablation.sh が渡す）。設定は configs/train.yaml、保存先は machines.yaml の ablation_checkpoints_dir/SRCycleGAN/<run>/。
set -euo pipefail
cd "$(dirname "$0")"
MACHINE="${SPICA_MACHINE:?SPICA_MACHINE が未設定です。bash start_ablation.sh から起動するか、export SPICA_MACHINE=<machine> してください}"
RESUME_ARGS=()
if [ -n "${SPICA_RESUME:-}" ]; then
  RESUME_ARGS=(--resume "$SPICA_RESUME" --resume_tag "${SPICA_RESUME_TAG:-latest}")
fi
python run_train.py \
    --machine "$MACHINE" \
    --train configs/train.yaml \
    --machines ../../configs/machines.yaml \
    "${RESUME_ARGS[@]}" \
    -- "$@"
