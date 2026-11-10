#!/bin/bash
# =============================================================================
# crop_stage1.sh — パッチ切り出し（コンテナ内で実行。ホストからは bash start.sh crop [--flag value ...]）
#   指定した PCD スライスを 512 のフル推論にかけ、左上 (x, y) から patch 四方を切り出して EID の代表パッチと並べる（スライド用）。
#   ケース（PCD / EID のスライス名と座標）・patch・拡大率・デバイスは stage1/configs/crop.yaml。重みディレクトリは下の WEIGHT_DIR（--weight_dir で上書き可）。
#   出力: <リポジトリ直下>/output/<run>_<重みディレクトリ名>_crop_<実行時刻>/case<N>_<pcd>_<eid>/（1_EID / 2_EID-like / 3_PCD / 4_R_color の個別 + panel.png）
#   学習中に走らせてよい（start.sh は起動済みコンテナに exec するだけ。crop.yaml の device が cpu なら VRAM も使わない）。
#   上書き例: bash start.sh crop --weight_dir /workspace/stage1/checkpoints/2026_0907_1742/weights/epoch_141 --device cuda --patch 96
# =============================================================================
set -euo pipefail

# ===== ここだけ書き換える =====
WEIGHT_DIR="/workspace/stage1/checkpoints/2026_0908_0041/weights/epoch_170"   # 重みディレクトリ（net_G.pth がある所。学習中なら latest が最新）
# =============================

cd "$(dirname "$0")/stage1"
MACHINE="${SPICA_MACHINE:?SPICA_MACHINE が未設定です。bash start.sh から起動するか、export SPICA_MACHINE=<machine> してください}"
python run_crop.py \
    --machine "$MACHINE" \
    --crop configs/crop.yaml \
    --train configs/train.yaml \
    --machines ../configs/machines.yaml \
    --weight_dir "$WEIGHT_DIR" \
    -- "$@"
