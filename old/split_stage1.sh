#!/bin/bash
# =============================================================================
# Stage 1 の症例分割ファイル生成（task: sr 用）の入口 — **Docker コンテナ内で実行する**。通常はホストの  bash start.sh split  から呼ばれる。
#   EID（machines.yaml の eid_dir。症例名は EID1024_v1 と同じ）を val : test : train = 15 : 15 : 70 に分け、OUT に固定する。
#   FORCE_TEST の症例（EID-049 = Stage 2 の目視テスト・crop の代表）は必ず test に入れる。
#   既に OUT があれば止まる（分割は一度決めたら固定）。作り直すときは  bash start.sh split --overwrite。
#   手で叩くとき: bash start.sh <machine> shell → bash split_stage1.sh [--seed 1] [--dir /workspace/DataSet/EID_v5]
# =============================================================================
set -euo pipefail

OUT="../configs/splits/eid_v5.yaml"   # 出力（stage1/ からの相対。sr.yaml の split_file = configs/splits/eid_v5.yaml と一致させる）
VAL=0.15                              # val の比率
TEST=0.15                             # test の比率（残り 70 % が train）
SEED=0                                # shuffle の seed
FORCE_TEST="EID-049"                  # test に固定する症例（スペース区切りで複数可）

cd "$(dirname "$0")/stage1"
MACHINE="${SPICA_MACHINE:?SPICA_MACHINE が未設定です。bash start.sh から起動するか、export SPICA_MACHINE=<machine> してください}"
FORCE_ARGS=()
for c in $FORCE_TEST; do FORCE_ARGS+=(--force_test "$c"); done
python make_split.py \
    --machine "$MACHINE" \
    --machines ../configs/machines.yaml \
    --out "$OUT" \
    --val "$VAL" --test "$TEST" --seed "$SEED" \
    "${FORCE_ARGS[@]}" \
    "$@"
