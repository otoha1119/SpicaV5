#!/bin/bash
# stage1_v3/pipeline_v4.sh — v4 の通し実行（Docker コンテナ内）: (1) Dn の学習 → (2) Dn の合否（FAIL なら停止）→ (3) v4 の学習。
#   bash start1_v3.sh <machine> v4 [--flag value ...]
#   --flag は (3) の学習（configs/schema.py の TRAIN / MACHINE）にだけ渡す。Dn の設定は configs/denoise.yaml（変えるなら dn を単独で回す）。
#   (3) は --pipeline_mode v4 --dn_weights <(1) の latest/net_Dn.pth> を先頭に付ける（ユーザーの --flag が後勝ち）。
set -euo pipefail
cd "$(dirname "$0")"
MACHINE="${SPICA_MACHINE:?SPICA_MACHINE が未設定です。bash start1_v3.sh から起動してください}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "[v4] (1/3) Dn の学習（configs/denoise.yaml）"
SPICA_DN_RUN_FILE="$TMP/dn_run" bash denoise.sh
DN_RUN="$(cat "$TMP/dn_run")"
DN_W="$DN_RUN/latest/net_Dn.pth"
[ -f "$DN_W" ] || { echo "[v4] Dn の重みがありません: $DN_W" >&2; exit 1; }

echo "[v4] (2/3) Dn の合否（決定 §6.4）: $DN_W"
set +e
bash dncheck.sh --weights "$DN_W" 2>&1 | tee "$TMP/dncheck.log"; RC=$?   # pipefail なので dncheck の終了コード
set -e
DNCHECK_DIR="$(sed -n 's/^\[dncheck\] -> //p' "$TMP/dncheck.log" | tail -1)"
if [ "$RC" -ne 0 ]; then
  case "$RC" in
    1) echo "[v4] Dn が合否表を通りません（上の FAIL 行）。v4 の学習は始めません。denoise.yaml（cell_size / min_dist / epoch）を見直して dn を回し直すか、dncheck --weights で別の重みを選ぶ" >&2 ;;
    2) echo "[v4] dncheck の設定エラー（終了コード 2。上の [dncheck] 設定エラー 行）。denoise.yaml / machines.yaml を直す" >&2 ;;
    *) echo "[v4] dncheck が例外で止まりました（終了コード ${RC}。上の traceback）" >&2 ;;
  esac
  echo "[v4] dn run: $DN_RUN | dncheck: $DNCHECK_DIR" >&2
  exit 1
fi

echo "[v4] (3/3) v4 の学習: --pipeline_mode v4 --dn_weights $DN_W $*"
set +e
bash train.sh --pipeline_mode v4 --dn_weights "$DN_W" "$@" 2>&1 | tee "$TMP/train.log"; RC=$?
set -e
TRAIN_RUN="$(sed -n 's/^\[run_train\] stage1_v3 run \([^ ]*\) → .*/\1/p' "$TMP/train.log" | head -1)"
echo "[v4] 完了（学習の終了コード ${RC}）"
echo "[v4] dn run      : $DN_RUN"
echo "[v4] dncheck     : $DNCHECK_DIR"
echo "[v4] train run   : ${TRAIN_RUN:-(不明)}"
exit "$RC"
