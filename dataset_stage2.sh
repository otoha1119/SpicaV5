#!/bin/bash
# =============================================================================
# Stage 2 データセット作成の入口 — 取扱説明
#   **Docker コンテナ内で実行する**。通常はホスト側の  bash start2.sh dataset [--flag value ...]  から自動で呼ばれる。
#   INPUT_DIR の <case>/<slice>.png（uint16 1ch、一辺 input_size = 512）を scale 倍（2 → 1024）に補間し、OUTPUT_DIR に同じ <case>/<slice>.png で書く。
#   値の規約は変えない（stored = HU + 1400 のまま）。PCD1024（別途変換）とはファイル名 PCD-nnn-sss で対応する。
#
# ■ ここに書くもの（下の 2 変数。start.sh の MACHINE と同じ流儀。パスはコンテナ内から見えるもの）
#   INPUT_DIR   変換元。Stage 1 の推論出力 <run>/infer/<重み>/<入力>/<時刻>/full（EID-like512）が本命。実 EID（EID_v5）を推論用に 1024 にするのにも使える。
#               直下に症例フォルダ（PCD-nnn/）、中に PNG。サブフォルダは再帰的に拾う。'.' 始まりは無視。PNG 以外は無視（枚数を表示）
#   OUTPUT_DIR  作成先。machines.yaml の container_data_root（DataSet のマウント）配下であること。**存在しないこと**（既にあればエラー。上書きしない）。
#               名前は由来が分かるもの（例 EIDlike1024_v1。実 EID の 1024 は EID1024_v1 のように区別する）
#
# ■ 方式  stage2/configs/dataset.yaml（scale / interp / input_size）。既定値は無い。
#   コマンドで上書き（最優先）: bash start2.sh dataset --interp bilinear --input_dir ... --output_dir ...
#   受け付けるのは stage2/configs/schema.py の DATASET にあるフラグと --input_dir / --output_dir だけ。無いフラグはエラー
#
# ■ 出力
#   OUTPUT_DIR/<case>/<slice>.png   uint16 1ch、一辺 input_size × scale
#   OUTPUT_DIR/manifest.yaml        由来（入力フォルダ、隣の infer.yaml があればその内容）、方式、症例ごとの枚数、実効設定
#   書き込みは OUTPUT_DIR.tmp に行い、全部終わってから OUTPUT_DIR に rename する（途中で止まれば .tmp が残る = 中断の痕跡。消してからやり直す）
#
# ■ 並列  machines.yaml の num_threads を worker 数にする（0 で逐次）。Stage 1 の学習中に走らせると CPU とディスクを取り合う
# =============================================================================
set -euo pipefail

# ===== ここだけ書き換える =====
INPUT_DIR="/workspace/stage1/checkpoints/2026_0907_2222/infer/epoch_141/PCD512_v2/2026_0908_120000/full"   # 変換元（実機で ls して確認）
OUTPUT_DIR="/workspace/DataSet/EIDlike1024_v1"                                                            # 作成先（存在しないこと）
# =============================

cd "$(dirname "$0")/stage2"
MACHINE="${SPICA_MACHINE:?SPICA_MACHINE が未設定です。bash start2.sh から起動するか、export SPICA_MACHINE=<machine> してください}"
python make_dataset.py \
    --machine "$MACHINE" \
    --config configs/dataset.yaml \
    --machines ../configs/machines.yaml \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    -- "$@"
