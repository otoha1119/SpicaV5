#!/bin/bash
# stage1_v3/denoise.sh — v4 の Dn（PCD512 の自己教師あり denoiser、Neighbor2Neighbor 型）の学習。Docker コンテナ内の入口。
#   bash start1_v3.sh <machine> dn [--flag value]   フラグは configs/schema.py の DENOISE にあるものだけ。出力 stage1_v3/checkpoints/denoiser/<run>/ と denoiser/latest/
set -euo pipefail
cd "$(dirname "$0")"
MACHINE="${SPICA_MACHINE:?SPICA_MACHINE が未設定です。bash start1_v3.sh から起動してください}"
RUN_FILE_ARGS=()
if [ -n "${SPICA_DN_RUN_FILE:-}" ]; then RUN_FILE_ARGS=(--write_run_dir "$SPICA_DN_RUN_FILE"); fi
python train_denoiser.py --machine "$MACHINE" --denoise configs/denoise.yaml --machines ../configs/machines.yaml ${RUN_FILE_ARGS[@]+"${RUN_FILE_ARGS[@]}"} -- "$@"
