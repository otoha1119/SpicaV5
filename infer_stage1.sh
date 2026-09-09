#!/bin/bash
# =============================================================================
# Stage 1 (FE-GAN) 推論（症例丸ごと）の入口 — 取扱説明
#   **Docker コンテナ内で実行する**。通常はホスト側の  bash start.sh infer [--flag value ...]  から自動で呼ばれる。
#   学習末尾のテストではなく、学習済みの重みで PNG フォルダを丸ごと EID-like に変換する「ちゃんとした推論」。
#
# ■ ここに書くもの（下の 4 変数。start.sh の MACHINE と同じ流儀。パスはコンテナ内から見えるもの）
#   WEIGHT_DIR     重みディレクトリ = net_G.pth があるディレクトリ。次のどれか:
#                    <checkpoints_dir>/<run>/latest            最後に保存したもの
#                    <checkpoints_dir>/<run>/best              bash start.sh best <run> <epoch> で作ったもの
#                    <checkpoints_dir>/<run>/weights/epoch_NNN  epoch ごとの checkpoint
#   INPUT_DIR      処理する PNG 群のフォルダ（uint16 1ch、stored = HU + 1400。直下に症例フォルダ PCD-nnn/、中に PCD-nnn-sss.png）。
#                  machines.yaml の host_data_root → container_data_root のマウント配下（/workspace/DataSet/...）に置くこと
#   OUTPUT_FORMAT  png   : 16bit PNG（入力と同じ規約）
#                  dicom : 前処理を全部戻して DICOM（元 DICOM のヘッダを継承、HU を元の RescaleSlope/Intercept で格納値に戻す）
#                  both  : 両方
#   DICOM_DIR      元 DICOM のルート（dicom / both のとき必須。png のときは使わない）。変換時（SpicaV3 convert_pcd.py）と同じ構成であること:
#                  症例フォルダ名の数値部分 = PCD-nnn の nnn、フォルダ内の *.dcm を名前順に並べた sss 番目（1 始まり）= PCD-nnn-sss。
#                  書く直前に参照 DICOM と入力 PNG を全画素で照合する（ズレていればエラー）。症例フォルダは単一 Series であること
#
# ■ コマンドで上書き（最優先）
#   bash start.sh infer --weight_dir /workspace/stage1/checkpoints/2026_0907_1742/best --input_dir /workspace/DataSet/PCD512_v2
#   bash start.sh infer --output_format both --dicom_dir /workspace/DataSet/photonCT/PhotonCT512_original
#   bash start.sh infer --mode full --max_slices 4        # 方式の上書き。stage1/configs/schema.py の INFER にあるフラグだけ受け付ける
#
# ■ 出力
#   <リポジトリ直下>/output/<run>_<重みディレクトリ名>_<入力フォルダ名>_<実行時刻 yyyy_mmdd_HHMMSS>/   （実行ごとに別ディレクトリ。混ざらない。コンテナでは /workspace/output/）
#     {full,patch}/<case>/<slice>.png         EID-like（16bit PNG、入力と同じ規約）        … png / both
#     {full,patch}_dicom/<case>/<slice>.dcm   EID-like（DICOM、元ヘッダ継承・UID は新規）  … dicom / both
#     {full,patch}_R/<case>/<slice>.png       残差（16bit PNG、0 HU = 32768。PNG のみ）      … infer.yaml の save_residual
#     diff_stats.txt（mode = both のとき）、infer.yaml（解決済み設定）
#
# ■ デバイス
#   machines.yaml の gpu_gen（start.sh から渡る SPICA_MACHINE のエントリ）で決まる: 0 → cpu、それ以外 → cuda（使えなければエラー。cpu に落とさない）。
#   GPU の無いマシンで推論するときは gpu_gen: 0 のマシン定義で  bash start.sh <machine> infer  。
#
# ■ 処理
#   stage1/run_infer.py → infer.yaml / machines.yaml を schema.py で検証、WEIGHT_DIR の上の run の launch.yaml から G の構成を取り、
#   全引数明示で inference_dir.py を exec（full / patch / both）。
# =============================================================================
set -euo pipefail

# ===== ここだけ書き換える =====
WEIGHT_DIR="/workspace/weight/2026_0908_0041/epoch_300"   # 重みディレクトリ（net_G.pth がある所）
INPUT_DIR="/workspace/DataSet/PCD512_v2"                          # 処理する PNG 群のフォルダ
OUTPUT_FORMAT="png"                                               # png | dicom | both
DICOM_DIR="/workspace/DataSet/photonCT/PhotonCT512_original"      # 元 DICOM のルート（dicom / both のときだけ使う）。SpicaV3 convert_pcd.py の変換元と同じ配置。実機で ls して確認
# =============================

cd "$(dirname "$0")/stage1"
MACHINE="${SPICA_MACHINE:?SPICA_MACHINE が未設定です。bash start.sh から起動するか、export SPICA_MACHINE=<machine> してください}"
python run_infer.py \
    --machine "$MACHINE" \
    --infer configs/infer.yaml \
    --train configs/train.yaml \
    --machines ../configs/machines.yaml \
    --weight_dir "$WEIGHT_DIR" \
    --input_dir "$INPUT_DIR" \
    --output_format "$OUTPUT_FORMAT" \
    --dicom_dir "$DICOM_DIR" \
    -- "$@"
