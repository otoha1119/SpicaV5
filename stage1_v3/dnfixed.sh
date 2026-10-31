#!/bin/bash
# stage1_v3/dnfixed.sh — 固定 Dn（BM3D + stats の実測 PSD。学習なし）で PCD512 を先に denoise する。Docker コンテナ内の入口。
#   bash start1_v3.sh <machine> dnfixed [--input_dir DIR] [--max_slices N] [--workers K] [--out_name NAME] [--psd_path P --bm3d_profile np|refilter 等 DENOISE の上書き]
#   出力 <repo>/output/stage1_v3/dnfixed_<入力名>_<時刻>/full/<case>/<slice>.png と manifest.yaml。train は --dn_source precomputed --dn_dir <出力> --dn_weights "" で読む
set -euo pipefail
cd "$(dirname "$0")"
MACHINE="${SPICA_MACHINE:?SPICA_MACHINE が未設定です。bash start1_v3.sh から起動してください}"
python denoise_dir.py --machine "$MACHINE" --denoise configs/denoise.yaml --machines ../configs/machines.yaml -- "$@"
