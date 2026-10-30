#!/bin/bash
# stage1_v3/ponly.sh — P 単独（学習なし）で EID-like を生成する（推薦の最初の対照。レビュー I）。
#   bash start1_v3.sh <machine> ponly [--input_dir DIR] [--out_name NAME] [--max_slices N] [--seed N] [--blur_min .. --blur_max .. --noise_min .. --noise_max .. --corr_min .. --corr_max ..]
#   P の範囲は configs/train.yaml の degrade.*（--flag で上書き）。出力 <repo>/output/stage1_v3/ponly_<入力名>_<時刻>/full/ と manifest.yaml（スライスごとの σ_b, σ_n, c）。
set -euo pipefail
INPUT_DIR="/workspace/DataSet/PCD512_v2"
cd "$(dirname "$0")"
MACHINE="${SPICA_MACHINE:?SPICA_MACHINE が未設定です。bash start1_v3.sh から起動してください}"
python degrade_dir.py \
    --machine "$MACHINE" \
    --train configs/train.yaml \
    --machines ../configs/machines.yaml \
    --input_dir "$INPUT_DIR" \
    -- "$@"
