#!/usr/bin/env bash
# chain_stage1to2.sh — コンテナ内: Stage 1 の学習 → 推論（best の重みで PCD512 全部を full） → Stage 2 のデータ作成（×2 補間） → Stage 2 の学習 を 1 本で（2026-11-03）。
#
# ホストからは:
#   bash start.sh [マシン] chain [--flag value ...]                       新規に Stage 1 を学習してから続ける（--flag は Stage 1 の学習へ。bash start.sh と同じ）
#   bash start.sh [マシン] chain resume <run> [latest|best|<epoch>] [...] 既存 run の続きから。n_epochs まで学習済みなら学習は即終わり、推論から始まる
#
# 流れ（どこかで失敗したらそこで止まる。set -e。手前の成果物は残る）:
#   1. train_stage1.sh "$@"（SPICA_RESUME があれば再開）。run ディレクトリは run_train.py が SPICA_RUN_RECORD に書く
#   2. stage1/run_infer.py: 重み = <run>/${CHAIN_WEIGHT}（best は best_eNNN に解決）、入力 = その run の launch の pcd_dir（--dir_A の上書きも反映）、--mode full、png
#   3. stage2/make_dataset.py: 入力 = 2 の full/、出力 = ${CHAIN_DATA_ROOT}/EIDlike1024_<run>_<重みディレクトリ名>（既にあれば止まる。上書きしない）
#   4. stage2/run_train.py: --eidlike1024_dir に 3 の出力。設定は stage2/configs/train.yaml / mode.yaml のまま + CHAIN_STAGE2_ARGS
#   TensorBoard: Stage 1 は start.sh が渡す SPICA_TB_PORT、Stage 2 は machines.yaml の stage2_tb_port（start2.sh と同じ）
# 容量の目安: PCD512 全部（約 2 万枚）の EID-like1024 は 30〜40 GB。推論の full/（512）も output/ に残る。
set -euo pipefail

# ===== ここだけ実験ごとに書き換える =====
CHAIN_WEIGHT="best"                       # Stage 1 の重み: best（採点 val/score の最良 = best_eNNN）| latest | <epoch 番号>
CHAIN_DATA_ROOT="/workspace/DataSet"      # Stage 2 の学習入力を作る場所（container_data_root 配下）
CHAIN_STAGE2_ARGS=()                      # Stage 2 の学習への上書き（例: (--n_epochs 50)）。空なら stage2/configs の値
# =========================================

ROOT="$(cd "$(dirname "$0")" && pwd)"
MACHINE="${SPICA_MACHINE:?SPICA_MACHINE が未設定です。bash start.sh から起動してください}"
REC_RUN="$(mktemp)"; REC_INFER="$(mktemp)"
trap 'rm -f "${REC_RUN}" "${REC_INFER}"' EXIT
say() { echo "[chain] $(date '+%F %T') $*"; }

# ---- 1. Stage 1 の学習 ----
say "1/4 Stage 1 の学習"
SPICA_RUN_RECORD="${REC_RUN}" bash "${ROOT}/train_stage1.sh" "$@"
RUN_DIR="$(cat "${REC_RUN}")"
[ -n "${RUN_DIR}" ] && [ -d "${RUN_DIR}" ] || { echo "[chain] Stage 1 の run ディレクトリが分かりません" >&2; exit 2; }
RUN="$(basename "${RUN_DIR}")"
say "Stage 1 の run: ${RUN_DIR}"

# ---- 2. 推論（full、PCD512 全部） ----
LAUNCH_INFO="$(cd "${ROOT}/stage1" && python -c "import sys, yaml; from util.run_paths import latest_launch; y = yaml.safe_load(open(latest_launch(sys.argv[1]), encoding='utf-8')); print(y['machine']['pcd_dir']); print(y['mode']['task'])" "${RUN_DIR}")"
PCD_DIR="$(echo "${LAUNCH_INFO}" | sed -n 1p)"; TASK="$(echo "${LAUNCH_INFO}" | sed -n 2p)"
[ "${TASK}" = "degrade" ] || { echo "[chain] task: ${TASK} の run は対象外（degrade のみ）" >&2; exit 2; }
WEIGHT_DIR="$(cd "${ROOT}/stage1" && python -c "import sys; from util.run_paths import ckpt_dir; print(ckpt_dir(sys.argv[1], sys.argv[2]))" "${RUN_DIR}" "${CHAIN_WEIGHT}")"
[ -f "${WEIGHT_DIR}/net_G.pth" ] || { echo "[chain] 重みがありません: ${WEIGHT_DIR}（CHAIN_WEIGHT=${CHAIN_WEIGHT}）" >&2; exit 2; }
say "2/4 推論: ${WEIGHT_DIR} × ${PCD_DIR}（full）"
(cd "${ROOT}/stage1" && SPICA_INFER_RECORD="${REC_INFER}" python run_infer.py \
    --machine "${MACHINE}" --infer configs/infer.yaml --train configs/train.yaml --machines ../configs/machines.yaml \
    --weight_dir "${WEIGHT_DIR}" --input_dir "${PCD_DIR}" --output_format png --dicom_dir "" -- --mode full)
INFER_DIR="$(cat "${REC_INFER}")"
[ -d "${INFER_DIR}/full" ] || { echo "[chain] 推論の full/ がありません: ${INFER_DIR}" >&2; exit 2; }

# ---- 3. Stage 2 のデータ作成 ----
DATA_DIR="${CHAIN_DATA_ROOT}/EIDlike1024_${RUN}_$(basename "${WEIGHT_DIR}")"
say "3/4 Stage 2 のデータ作成: ${INFER_DIR}/full → ${DATA_DIR}"
(cd "${ROOT}/stage2" && python make_dataset.py \
    --machine "${MACHINE}" --config configs/dataset.yaml --machines ../configs/machines.yaml \
    --input_dir "${INFER_DIR}/full" -- --eidlike1024_dir "${DATA_DIR}")

# ---- 4. Stage 2 の学習 ----
TB2="$(python -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))[sys.argv[2]]['stage2_tb_port'])" "${ROOT}/configs/machines.yaml" "${MACHINE}")"
say "4/4 Stage 2 の学習（入力 ${DATA_DIR}、TensorBoard :${TB2}）"
env -u SPICA_RESUME -u SPICA_RESUME_TAG SPICA_TB_PORT="${TB2}" bash "${ROOT}/train_stage2.sh" --eidlike1024_dir "${DATA_DIR}" ${CHAIN_STAGE2_ARGS[@]+"${CHAIN_STAGE2_ARGS[@]}"}
say "完了: Stage 1 ${RUN} → 推論 ${INFER_DIR} → データ ${DATA_DIR} → Stage 2（stage2_checkpoints_dir の最新 run）"
