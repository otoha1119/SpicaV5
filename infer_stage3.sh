#!/bin/bash
# =============================================================================
# Stage 3（条件付き DDIM）推論（フォルダ丸ごと）の入口 — 取扱説明
#   **Docker コンテナ内で実行する**。通常はホスト側の  bash start3.sh infer [--flag value ...]  から自動で呼ばれる（infer_stage2.sh と同じ流儀）。
#   1024 は 256 タイルに分け、毎ステップ重なりの ε を平均して全体で 1 回 DDIM 更新する（WeatherDiffusion 型）。所要時間は「タイル数 × sample_steps」に比例し、1 スライスずつ回す。
#   入力は 1024（EID-like1024）でも 512（実 EID_v5 など）でもよく、512 は run の dataset_info.yaml の方式（学習と同じ）で補間してから通す。
#
# ■ ここに書くもの（下の 2 変数。パスはコンテナ内から見えるもの）
#   WEIGHT_DIR   重みディレクトリ（model.pth / ema.pth / state.pth がある所）: <stage3_checkpoints_dir>/<run>/latest | best | weights/iter_NNNNNNN
#   INPUT_DIR    処理する PNG 群のフォルダ（直下に症例フォルダ）
#
# ■ コマンドで上書き（最優先。stage3/configs/schema.py の INFER / MACHINE のフラグと --weight_dir / --input_dir）
#   bash start3.sh infer --cases PCD-017,PCD-018 --slices_per_case 10                        # 決定 7 の 1 本目の評価（教師あり → metrics.txt）
#   bash start3.sh infer --input_dir /workspace/DataSet/EID_v5 --input eid --teacher false --cases EID-049 --max_slices 10
#   bash start3.sh infer --sample_steps 100 --tile_stride 64                                  # S と r を振る
#
# ■ 出力: <リポジトリ直下>/output/<run>_<重み>_<入力名>_<実行時刻>/（full/, full_input1024/, full_panel/, eid/ | ref/, metrics.txt, slices.txt, infer.yaml）
# =============================================================================
set -euo pipefail

# ===== ここだけ書き換える =====
WEIGHT_DIR="/workspace/stage3/checkpoints/<run>/latest"   # 重みディレクトリ。<run> を実際の run 名に
INPUT_DIR="/workspace/DataSet/EIDlike1024_v1"             # 処理する PNG 群のフォルダ（一辺 1024 か 512）
# =============================

cd "$(dirname "$0")/stage3"
MACHINE="${SPICA_MACHINE:?SPICA_MACHINE が未設定です。bash start3.sh から起動するか、export SPICA_MACHINE=<machine> してください}"
python run_infer.py \
    --machine "$MACHINE" \
    --infer configs/infer.yaml \
    --train configs/train.yaml \
    --machines ../configs/machines.yaml \
    --weight_dir "$WEIGHT_DIR" \
    --input_dir "$INPUT_DIR" \
    -- "$@"
