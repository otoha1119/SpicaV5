#!/bin/bash
# =============================================================================
# Stage 3（条件付き DDIM、本家 ermongroup/ddim + 入力の channel 連結）学習の入口 — 取扱説明
#   **Docker コンテナ内で実行する**。通常はホスト側の  bash start3.sh  から自動で呼ばれる（train_stage2.sh と同じ流儀）。
#
# ■ 手で叩くとき（bash start3.sh <machine> shell で入ってから、/workspace で）
#   bash train_stage3.sh                       # configs の設定どおりに学習
#   bash train_stage3.sh --n_iters 300000      # 上書き（最優先）。stage3/configs/schema.py にあるフラグだけ受け付ける
#
# ■ 再開
#   環境変数 SPICA_RESUME=<run 名>（と任意で SPICA_RESUME_TAG=latest|best|<step>）があれば、その run の続きから学習する。
#   ホストからは  bash start3.sh resume 2026_1013_2130 [50000]  で同じ。
#
# ■ 処理
#   stage3/run_train.py → ddim.yml / train.yaml / machines.yaml を schema.py で検証 → 実効値を launch.yaml に保存 → train.py を exec。
#   保存先は machines.yaml の stage3_checkpoints_dir/<run>/（レイアウトは stage3/util/run_paths.py）。
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/stage3"
MACHINE="${SPICA_MACHINE:?SPICA_MACHINE が未設定です。bash start3.sh から起動するか、export SPICA_MACHINE=<machine> してください}"
RESUME_ARGS=()
if [ -n "${SPICA_RESUME:-}" ]; then
  RESUME_ARGS=(--resume "$SPICA_RESUME" --resume_tag "${SPICA_RESUME_TAG:-latest}")
fi
python run_train.py \
    --machine "$MACHINE" \
    --ddim configs/ddim.yml \
    --train configs/train.yaml \
    --machines ../configs/machines.yaml \
    ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"} \
    -- "$@"
