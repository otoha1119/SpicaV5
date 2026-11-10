#!/bin/bash
# =============================================================================
# Stage 2（U-Net + MSE、ILUMENATE 準拠）学習の入口 — 取扱説明
#   **Docker コンテナ内で実行する**。通常はホスト側の  bash start2.sh  から自動で呼ばれる。
#
# ■ 手で叩くとき（bash start2.sh <machine> shell で入ってから、/workspace で）
#   bash train_stage2.sh                      # configs の設定どおりに学習
#   bash train_stage2.sh --n_epochs 50        # 上書き（最優先）。stage2/configs/schema.py にあるフラグだけ受け付ける（list は --val_cases PCD-017,PCD-018）
#
# ■ 再開
#   環境変数 SPICA_RESUME=<run 名 yyyy_mmdd_HHMM>（と任意で SPICA_RESUME_TAG=latest|best|<epoch>）があれば、その run の続きから学習する。
#   ホストからは  bash start2.sh resume 2026_0908_2130 [30]  で同じ。
#
# ■ マシン名
#   環境変数 SPICA_MACHINE から取る（start2.sh が compose 経由でコンテナに渡す）。未設定ならエラー（既定値なし）。
#
# ■ 処理
#   stage2/run_train.py に渡す → train.yaml / mode.yaml / machines.yaml を schema.py で検証 → 実効値を launch.yaml に保存 → train.py を exec。
#   実験名（run）は起動時刻 yyyy_mmdd_HHMM（JST）。保存先は machines.yaml の stage2_checkpoints_dir/<run>/（レイアウトは stage2/util/run_paths.py）。
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/stage2"
MACHINE="${SPICA_MACHINE:?SPICA_MACHINE が未設定です。bash start2.sh から起動するか、export SPICA_MACHINE=<machine> してください}"
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
