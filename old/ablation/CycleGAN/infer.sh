#!/bin/bash
# ablation/CycleGAN/infer.sh — 普通の CycleGAN 推論の入口（**Docker コンテナ内で実行**。ホストからは bash start_ablation.sh <マシン> CycleGAN infer [--flag]）。
#   重みディレクトリ（net_G_A.pth がある所）と入力フォルダ（EID1024 または EID512 の <case>/<slice>.png。512 は学習と同じ方式で補間）はここに書く（--weight_dir / --input_dir で上書き可）。
#   方式は configs/infer.yaml（--max_slices 4 --save_panel true などで上書き）。出力は <repo>/output/ablation/CycleGAN/<run>_<重み>_<入力名>_<時刻>/。
set -euo pipefail
WEIGHT_DIR="/workspace/ablation/checkpoints/CycleGAN/2026_1027_0000/latest"   # 重みディレクトリ（<run>/latest | <run>/best | <run>/weights/epoch_NNN）
INPUT_DIR="/workspace/DataSet/EID_v5"                                            # 入力（実 EID512 → 補間、または EID1024_v1 をそのまま）。test 症例だけにするなら症例フォルダを別に用意する
cd "$(dirname "$0")"
MACHINE="${SPICA_MACHINE:?SPICA_MACHINE が未設定です。bash start_ablation.sh から起動するか、export SPICA_MACHINE=<machine> してください}"
python run_infer.py \
    --machine "$MACHINE" \
    --infer configs/infer.yaml \
    --train configs/train.yaml \
    --machines ../../configs/machines.yaml \
    --weight_dir "$WEIGHT_DIR" \
    --input_dir "$INPUT_DIR" \
    -- "$@"
