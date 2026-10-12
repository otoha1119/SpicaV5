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
#   環境変数 SPICA_RESUME=<run 名 yyyy_mmdd_HHMM[-<name>]>（と任意で SPICA_RESUME_TAG=latest|best|<epoch>）があれば、その run の続きから学習する。
#   ホストからは  bash start.sh resume 2026_0905_1234 [30]  で同じ。run 名を空白区切りで 2 つ入れると組として再開（bash start.sh resume <runA> <runB> [tag]。2026-10-13）
#
# ■ 2 モデル同時学習（2026-10-13）
#   環境変数 SPICA_MULTI_MODES="<modeA.yaml> <modeB.yaml>"（空白区切り。リポジトリ直下からの相対パス可）があれば、configs/mode.yaml の代わりにその 2 つで
#   run_train.py --multi を呼ぶ（run 名 <時刻>-<ファイル名>、1 プロセスで 2 モデル）。ホストからは  bash start.sh multi stage1/configs/modes/n_only.yaml stage1/configs/modes/n_pacgan.yaml
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
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT/stage1"
MACHINE="${SPICA_MACHINE:?SPICA_MACHINE が未設定です。bash start.sh から起動するか、export SPICA_MACHINE=<machine> してください}"
RESUME_ARGS=()
if [ -n "${SPICA_RESUME:-}" ]; then
  # shellcheck disable=SC2206  # run 名は空白を含まない。2 つなら組の再開（2026-10-13）
  RESUME_ARGS=(--resume $SPICA_RESUME --resume_tag "${SPICA_RESUME_TAG:-latest}")
fi
MODE_ARGS=(--mode configs/mode.yaml)
if [ -n "${SPICA_MULTI_MODES:-}" ]; then  # 2 モデル同時学習: mode ファイルはリポジトリ直下からの相対パスで来るので、ここ（stage1/）から読めるよう絶対パスにする
  MODE_ARGS=(--multi)
  for m in $SPICA_MULTI_MODES; do
    case "$m" in /*) MODE_ARGS+=("$m") ;; *) MODE_ARGS+=("$ROOT/$m") ;; esac
  done
fi
python run_train.py \
    --machine "$MACHINE" \
    --train configs/train.yaml \
    "${MODE_ARGS[@]}" \
    --sr configs/sr.yaml \
    --machines ../configs/machines.yaml \
    "${RESUME_ARGS[@]}" \
    -- "$@"
