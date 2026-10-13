#!/bin/bash
# =============================================================================
# Stage 3 NPS 評価の入口 — 取扱説明（決定 7 の評価 (ii)）
#   **Docker コンテナ内で実行する**。通常はホスト側の  bash start3.sh nps [--flag value ...]  から自動で呼ばれる。
#   1 回の推論の出力フォルダで、同じ ROI の NPS を 入力 / 出力 / 教師 の 3 つで比べる（stage3/eval_nps.py）。ROI は stage3/configs/nps.yaml に書く（空だとエラー）。
#
# ■ ここに書くもの
#   INFER_DIR   bash start3.sh infer の出力フォルダ（infer.yaml がある所）。--infer_dir で上書き可
#
# ■ 出力: <INFER_DIR>/nps_<時刻>/ に nps.txt（表と判定）、nps_radial.csv、nps.yaml、roi_<n>_<set>.png
# =============================================================================
set -euo pipefail

# ===== ここだけ書き換える =====
INFER_DIR="/workspace/output/<推論の出力フォルダ>"
# =============================

cd "$(dirname "$0")/stage3"
MACHINE="${SPICA_MACHINE:?SPICA_MACHINE が未設定です。bash start3.sh から起動するか、export SPICA_MACHINE=<machine> してください}"
python eval_nps.py \
    --machine "$MACHINE" \
    --nps configs/nps.yaml \
    --machines ../configs/machines.yaml \
    --infer_dir "$INFER_DIR" \
    -- "$@"
