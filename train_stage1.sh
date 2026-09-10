#!/bin/bash
# =============================================================================
# Stage 1 (FE-GAN: Park et al. 2019) 学習の入口 — 取扱説明
#   **Docker コンテナ内で実行する**。通常はホスト側の  bash start.sh  から自動で呼ばれる。
#
# ■ 手で叩くとき（bash start.sh <machine> shell で入ってから、/workspace で）
#   bash train_stage1.sh                      # configs の設定どおりに学習
#   bash train_stage1.sh --n_epochs 50        # 上書き（最優先）。stage1/configs/schema.py にあるフラグだけ受け付ける
#
# ■ 再開
#   環境変数 SPICA_RESUME=<run 名 yyyy_mmdd_HHMM>（と任意で SPICA_RESUME_TAG=latest|best|<epoch>）があれば、その run の続きから学習する。
#   ホストからは  bash start.sh resume 2026_0905_1234 [30]  で同じ。
#
# ■ マシン名
#   環境変数 SPICA_MACHINE から取る（start.sh が compose 経由でコンテナに渡す）。未設定ならエラー（既定値なし）。
#   別のマシン定義で試すときは  export SPICA_MACHINE=<machine>  してから実行。
#
# ■ 処理
#   stage1/run_train.py に渡す → train.yaml / mode.yaml / machines.yaml を schema.py で検証 → 全パラメータ明示で train.py を exec。
#   実験名（run）は起動時刻 yyyy_mmdd_HHMM（JST）。保存先は machines.yaml の checkpoints_dir/<run>/（レイアウトは stage1/util/run_paths.py）。
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/stage1"
MACHINE="${SPICA_MACHINE:?SPICA_MACHINE が未設定です。bash start.sh から起動するか、export SPICA_MACHINE=<machine> してください}"
RESUME_ARGS=()
if [ -n "${SPICA_RESUME:-}" ]; then
  RESUME_ARGS=(--resume "$SPICA_RESUME" --resume_tag "${SPICA_RESUME_TAG:-latest}")
fi
python run_train.py \
    --machine "$MACHINE" \
    --train configs/train.yaml \
    --mode configs/mode.yaml \
    --machines ../configs/machines.yaml \
    "${RESUME_ARGS[@]}" \
    -- "$@"
