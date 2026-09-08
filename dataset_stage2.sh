#!/bin/bash
# =============================================================================
# Stage 2 データセット作成の入口 — 取扱説明
#   **Docker コンテナ内で実行する**。通常はホスト側の  bash start2.sh dataset [--flag value ...]  から自動で呼ばれる。
#   INPUT_DIR の <case>/<slice>.png（uint16 1ch、一辺 input_size = 512）を scale 倍（2 → 1024）に補間し、
#   machines.yaml の eidlike1024_dir に同じ <case>/<slice>.png で書く。値の規約は変えない（stored = HU + 1400 のまま）。
#   PCD1024（machines.yaml の pcd1024_dir、別途変換）とはファイル名 PCD-nnn-sss で対応する。
#
# ■ ここに書くもの（下の 1 変数。infer_stage1.sh の WEIGHT_DIR と同じ流儀 = 実行ごとに変わるもの。パスはコンテナ内から見えるもの）
#   INPUT_DIR   変換元。Stage 1 の推論出力 <run>/infer/<重み>/<入力>/<時刻>/full（EID-like512）が本命。
#               直下に症例フォルダ（PCD-nnn/）、中に PNG。サブフォルダは再帰的に拾う。'.' 始まりは無視。PNG 以外は無視（枚数を表示）
#
# ■ 出力先（machines.yaml。マシンごとのデータ配置なので Stage 1 の pcd_dir / eid_dir と同じ場所に書く）
#   eidlike1024_dir   作成先 = Stage 2 の学習入力（例 /workspace/DataSet/EIDlike1024_v1）。container_data_root 配下で、**存在しないこと**（既にあればエラー。上書きしない）。
#                     別バージョンを作るときは machines.yaml の値を変える（学習もその値を読む）か、--eidlike1024_dir で一時的に上書きする。
#                     実 EID（EID_v5）を推論用に 1024 にするときも --input_dir /workspace/DataSet/EID_v5 --eidlike1024_dir /workspace/DataSet/EID1024_v1 のように上書きで使える
#
# ■ 方式  stage2/configs/dataset.yaml（scale / interp / input_size）。既定値は無い。
#   コマンドで上書き（最優先）: bash start2.sh dataset --interp bilinear --input_dir ... --eidlike1024_dir ...
#   受け付けるのは stage2/configs/schema.py の DATASET / MACHINE にあるフラグと --input_dir だけ。無いフラグはエラー
#
# ■ 出力
#   <eidlike1024_dir>/<case>/<slice>.png   uint16 1ch、一辺 input_size × scale
#   <eidlike1024_dir>/manifest.yaml        由来（入力フォルダ、隣の infer.yaml があればその内容）、方式、症例ごとの枚数、実効設定
#   書き込みは <eidlike1024_dir>.tmp に行い、全部終わってから rename する（途中で止まれば .tmp が残る = 中断の痕跡。消してからやり直す）
#
# ■ 並列  machines.yaml の num_threads を worker 数にする（0 で逐次）。Stage 1 の学習中に走らせると CPU とディスクを取り合う
# =============================================================================
set -euo pipefail

# ===== ここだけ書き換える =====
INPUT_DIR="/workspace/stage1/checkpoints/2026_0907_2222/infer/epoch_141/PCD512_v2/2026_0908_120000/full"   # 変換元（実機で ls して確認）
# =============================

cd "$(dirname "$0")/stage2"
MACHINE="${SPICA_MACHINE:?SPICA_MACHINE が未設定です。bash start2.sh から起動するか、export SPICA_MACHINE=<machine> してください}"
python make_dataset.py \
    --machine "$MACHINE" \
    --config configs/dataset.yaml \
    --machines ../configs/machines.yaml \
    --input_dir "$INPUT_DIR" \
    -- "$@"
