#!/bin/bash
# stage1_v3/stats.sh — R-S22 の計測（一様 ROI の局所 SD・NPS、体内 HU 分位）を複数フォルダで並べる。
#   bash start1_v3.sh <machine> stats --dirs /workspace/DataSet/EID_v5,/workspace/DataSet/PCD512_v2,/workspace/output/stage1_v3/ponly_.../full [--max_slices 200]
#   出力 <repo>/output/stage1_v3/stats_<時刻>/stats.yaml と summary.txt
set -euo pipefail
cd "$(dirname "$0")"
MACHINE="${SPICA_MACHINE:?SPICA_MACHINE が未設定です。bash start1_v3.sh から起動してください}"
python measure_dir.py --machine "$MACHINE" --train configs/train.yaml --machines ../configs/machines.yaml -- "$@"
