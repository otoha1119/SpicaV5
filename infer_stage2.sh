#!/bin/bash
# =============================================================================
# Stage 2（U-Net）推論（フォルダ丸ごと）の入口 — 取扱説明
#   **Docker コンテナ内で実行する**。通常はホスト側の  bash start2.sh infer [--flag value ...]  から自動で呼ばれる。
#   学習済みの重みで PNG フォルダを丸ごと PCD-like1024 に変換する。入力は 1024（EID-like1024 = bash start2.sh dataset の出力）でも
#   512（実 EID = EID_v5、Stage 1 の推論出力 full/ など）でもよく、512 なら run の dataset_info.yaml に記録された方式（学習データ作成と同じ scale / interp）で
#   1024 に補間してから通す（1 枚目の大きさで判定。混在・他の大きさはエラー）。
#
# ■ ここに書くもの（下の 2 変数。infer_stage1.sh と同じ流儀。パスはコンテナ内から見えるもの）
#   WEIGHT_DIR   重みディレクトリ = net_G.pth があるディレクトリ: <stage2_checkpoints_dir>/<run>/latest | best | weights/epoch_NNN
#   INPUT_DIR    処理する PNG 群のフォルダ（uint16 1ch、stored = HU + 1400。直下に症例フォルダ、中に <slice>.png）。一辺 1024 か 512
#                教師（machines.yaml の pcd1024_dir/<case>/<slice>.png）は stage2/configs/infer.yaml の teacher=true のとき名前で読む（PCD のテスト症例向け）。
#                実 EID など教師の無い入力は --teacher false
#
# ■ コマンドで上書き（最優先）
#   bash start2.sh infer --weight_dir /workspace/stage2/checkpoints/2026_0909_1200/best --input_dir /workspace/DataSet/EIDlike1024_v1
#   bash start2.sh infer --max_slices 1 --save_panel true                                   # 1 枚だけ試す（stage2/configs/schema.py の INFER / MACHINE にあるフラグだけ受け付ける）
#   bash start2.sh infer --input_dir /workspace/DataSet/EID_v5 --input eid --teacher false   # 実 EID512（補間してから通す。教師なし。パネルは 4・5 列目に入り 1〜3 列目は pcd_slice）
#
# ■ 出力
#   <リポジトリ直下>/output/<run>_<重みディレクトリ名>_<入力フォルダ名>_<実行時刻 yyyy_mmdd_HHMMSS>/   （実行ごとに別ディレクトリ。コンテナでは /workspace/output/。Stage 1 と同じ根）
#     full/<case>/<slice>.png            PCD-like1024（16bit PNG、入力と同じ規約）
#     full_input1024/<case>/<slice>.png  512 を補間した 1024 入力（16bit）              … 512 入力 かつ infer.yaml の save_input1024
#     full_panel/<case>/<slice>.png      表示用パネル。学習の固定パネルと同じ 5 列 [EID-like1024 | PCD-like1024 | PCD1024 | 実 EID → PCD-like1024 | 実 EID1024] … infer.yaml の save_panel
#                                        input=eidlike なら 1〜3 列目が入力で 4・5 列目は eid_slice、input=eid なら 4・5 列目が入力で 1〜3 列目は pcd_slice（infer.yaml）
#     eid/ または ref/                   その相方の 16bit（eid_slice の入力・出力 / pcd_slice の入力・出力・教師）
#     metrics.txt                         出力 vs 教師の rmse / ssim / psnr（+ 入力そのままの参照値）… infer.yaml の teacher
#     infer.yaml                          解決済み設定
#
# ■ デバイス
#   machines.yaml の gpu_gen（start2.sh から渡る SPICA_MACHINE のエントリ）で決まる: 0 → cpu、それ以外 → cuda（使えなければエラー。cpu に落とさない）。
#   1024 フルの U-Net は 1 枚で数 GB の VRAM を使う。学習と同じ GPU で回すなら infer.yaml の batch_slices は 1 のまま。
#
# ■ 処理
#   stage2/run_infer.py → infer.yaml / machines.yaml を schema.py で検証、run の launch.yaml から U-Net の構成と HU 正規化、dataset_info.yaml から補間方式を取り、
#   全引数明示で inference_dir.py を exec。
# =============================================================================
set -euo pipefail

# ===== ここだけ書き換える =====
WEIGHT_DIR="/workspace/stage2/checkpoints/<run>/best"   # 重みディレクトリ（net_G.pth がある所）。<run> を実際の run 名に
INPUT_DIR="/workspace/DataSet/EIDlike1024_v1"           # 処理する PNG 群のフォルダ（一辺 1024 か 512）
# =============================

cd "$(dirname "$0")/stage2"
MACHINE="${SPICA_MACHINE:?SPICA_MACHINE が未設定です。bash start2.sh から起動するか、export SPICA_MACHINE=<machine> してください}"
python run_infer.py \
    --machine "$MACHINE" \
    --infer configs/infer.yaml \
    --train configs/train.yaml \
    --machines ../configs/machines.yaml \
    --weight_dir "$WEIGHT_DIR" \
    --input_dir "$INPUT_DIR" \
    -- "$@"
