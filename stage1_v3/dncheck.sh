#!/bin/bash
# stage1_v3/dncheck.sh — v4 の Dn 単体の合否（決定 §6.4）。Docker コンテナ内の入口。
#   bash start1_v3.sh <machine> dncheck [--weights PATH] [--input_dir DIR] [--n_slices N] [--resid_sd_max_hu .. 等 check.* の上書き]
#   --weights 省略時は configs/train.yaml の pipeline.dn_weights。終了コード 0 = 全項目 PASS
set -euo pipefail
cd "$(dirname "$0")"
MACHINE="${SPICA_MACHINE:?SPICA_MACHINE が未設定です。bash start1_v3.sh から起動してください}"
python check_denoiser.py --machine "$MACHINE" --denoise configs/denoise.yaml --train configs/train.yaml --machines ../configs/machines.yaml -- "$@"
