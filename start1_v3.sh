#!/bin/bash
# 射影型 Stage 1（P + R + 恒等項） のホスト側入口。
#   bash start1_v3.sh [machine] [train|v4|dn|dncheck|dnfixed|infer|ponly|stats|resume|best|tb|build|shell|down] [...]
#   例: bash start1_v3.sh PC3                                   # 学習（P + R + 恒等項）
#       bash start1_v3.sh PC3 resume 2026_1031_1200 latest
#       bash start1_v3.sh PC3 infer --weight_dir /workspace/stage1_v3/checkpoints/<run>/best --input_dir /workspace/DataSet/PCD512_v2
#       bash start1_v3.sh PC3 infer --weight_dir ... --input_dir /workspace/DataSet/EID_v5 --apply_degrade false   # 実 EID で恒等の確認
#       bash start1_v3.sh PC3 ponly --input_dir /workspace/DataSet/PCD512_v2 --max_slices 50          # P 単独（学習なし）で EID-like を生成
#       bash start1_v3.sh PC3 stats --dirs /workspace/DataSet/EID_v5,/workspace/DataSet/PCD512_v2      # R-S22 の計測（noise_corr_d0 も）
#       bash start1_v3.sh PC3 v4                                 # v4 の通し: dn → dncheck → 学習（--pipeline_mode v4）。--flag は学習にだけ渡す
#       bash start1_v3.sh PC3 dn [--n_epochs 50]                 # Dn（PCD denoiser、configs/denoise.yaml）の学習だけ
#       bash start1_v3.sh PC3 dncheck [--weights PATH]           # Dn の合否（決定 §6.4）。省略時は train.yaml の pipeline.dn_weights。--method bm3d なら固定 denoiser
#       bash start1_v3.sh PC3 dnfixed [--input_dir DIR] [--max_slices N] [--workers K] [--psd_path P]   # 固定 Dn（BM3D + stats の PSD）でデータを先に denoise（学習なし）
#                                                                # → bash start1_v3.sh PC3 --dn_source precomputed --dn_dir <dnfixed の出力> --dn_weights ""
set -euo pipefail

if [ -f /.dockerenv ]; then
  echo "[start1_v3] ホスト側で実行してください。コンテナ内では bash stage1_v3/train.sh を使います" >&2
  exit 2
fi

MACHINE="PC3"
ACTION="train"
BUILD=""
RESUME=""
RESUME_TAG="latest"
BEST_RUN=""
BEST_EPOCH=""
MACHINE_ARG=""

usage() {
  echo "usage: bash start1_v3.sh [machine] [train|v4|dn|dncheck|dnfixed|infer|ponly|stats|tb|build|shell|down] [--flag value ...]"
  echo "       bash start1_v3.sh [machine] resume <run> [latest|best|epoch] [--flag value ...]"
  echo "       bash start1_v3.sh [machine] best <run> <epoch>"
}

while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    train|v4|dn|dncheck|dnfixed|infer|ponly|stats|tb|shell|down) ACTION="$1"; shift ;;
    build) ACTION="build"; BUILD="--build"; shift ;;
    resume)
      ACTION="train"; shift
      [ $# -gt 0 ] && [[ "$1" != --* ]] || { echo "[start1_v3] resume には run 名が必要です" >&2; exit 2; }
      RESUME="$1"; shift
      if [ $# -gt 0 ] && [[ "$1" =~ ^([0-9]+|latest|best)$ ]]; then RESUME_TAG="$1"; shift; fi ;;
    best)
      ACTION="best"; shift
      [ $# -ge 2 ] && [[ "$1" != --* ]] && [[ "$2" =~ ^[0-9]+$ ]] || { echo "[start1_v3] best には run 名と epoch 番号が必要です" >&2; exit 2; }
      BEST_RUN="$1"; BEST_EPOCH="$2"; shift 2 ;;
    --*) break ;;
    *)
      [ -z "$MACHINE_ARG" ] || { echo "[start1_v3] マシン名が複数あります: $MACHINE_ARG, $1" >&2; exit 2; }
      MACHINE_ARG="$1"; shift ;;
  esac
done
if [ -n "$MACHINE_ARG" ]; then MACHINE="$MACHINE_ARG"; fi
case "$ACTION" in build|shell|down|tb|best) [ $# -eq 0 ] || { echo "[start1_v3] $ACTION に --flag は指定できません" >&2; exit 2; } ;; esac

ROOT="$(cd "$(dirname "$0")" && pwd)"
MACHINES="$ROOT/configs/machines.yaml"
PY=""
for c in python3 python; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c "import yaml" 2>/dev/null; then PY="$c"; break; fi
done
[ -n "$PY" ] || { echo "[start1_v3] python3 と pyyaml が必要です" >&2; exit 1; }

IFS=$'\t' read -r GPU_GEN HOST_DATA_ROOT CONTAINER_DATA_ROOT TB_PORT TB_PORT_STAGE2 TB_PORT_ABLATION < <("$PY" - "$MACHINES" "$MACHINE" <<'PYEOF'
import sys, yaml
path, name = sys.argv[1], sys.argv[2]
m = yaml.safe_load(open(path, encoding="utf-8"))
if not isinstance(m, dict) or name not in m:
    sys.exit(f"[start1_v3] machines.yaml に '{name}' がありません。候補: {list(m) if isinstance(m, dict) else []}")
e = m[name]
keys = ("gpu_gen", "host_data_root", "container_data_root", "tb_port", "stage2_tb_port", "ablation_tb_port")  # compose が 3 ポートを公開するため全部要る
missing = [k for k in keys if k not in e]
if missing:
    sys.exit(f"[start1_v3] machines.{name} にキーがありません: {missing}")
print("\t".join(str(e[k]) for k in keys))
PYEOF
)

if grep -qi microsoft /proc/version 2>/dev/null && [[ "$HOST_DATA_ROOT" =~ ^[A-Za-z]:[/\\] ]]; then
  command -v wslpath >/dev/null 2>&1 || { echo "[start1_v3] WSL では host_data_root を /mnt/<drive>/... にしてください" >&2; exit 1; }
  HOST_DATA_ROOT="$(wslpath -u "$HOST_DATA_ROOT")"
fi
[ -d "$HOST_DATA_ROOT" ] || { echo "[start1_v3] host_data_root がありません: $HOST_DATA_ROOT" >&2; exit 1; }

case "$GPU_GEN" in
  30|40) GEN="gen30" ;;
  50) GEN="gen50" ;;
  0) GEN="cpu" ;;
  *) echo "[start1_v3] gpu_gen=$GPU_GEN は未対応です" >&2; exit 1 ;;
esac

COMPOSE=(docker compose -f "$ROOT/docker/compose.$GEN.yaml" -p spicav5)
export HOST_DATA_ROOT CONTAINER_DATA_ROOT TB_PORT TB_PORT_STAGE2 TB_PORT_ABLATION SPICA_MACHINE="$MACHINE"
CHECKPOINTS_DIR="/workspace/stage1_v3/checkpoints"
echo "[start1_v3] machine=$MACHINE gpu_gen=$GPU_GEN action=$ACTION${RESUME:+ resume=$RESUME tag=$RESUME_TAG}"

if [ "$ACTION" = "down" ]; then
  "${COMPOSE[@]}" down
  exit 0
fi

container_running() { [ "$(docker inspect -f '{{.State.Running}}' spicav5 2>/dev/null)" = "true" ]; }
if container_running; then
  [ "$ACTION" != "build" ] || { echo "[start1_v3] コンテナが起動中です。down 後に build してください" >&2; exit 2; }
  ACTIVE_IMAGE="$(docker inspect -f '{{.Config.Image}}' spicav5)"
  ACTIVE_MACHINE="$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' spicav5 | sed -n 's/^SPICA_MACHINE=//p' | head -1)"
  [ "$ACTIVE_IMAGE" = "spicav5-$GEN" ] || { echo "[start1_v3] 起動中の image=$ACTIVE_IMAGE、指定は spicav5-$GEN です。処理終了後に down してください" >&2; exit 2; }
  [ "$ACTIVE_MACHINE" = "$MACHINE" ] || { echo "[start1_v3] 起動中の machine=$ACTIVE_MACHINE、指定は $MACHINE です。マウントが違うため処理終了後に down してください" >&2; exit 2; }
else
  "${COMPOSE[@]}" up -d $BUILD
  "${COMPOSE[@]}" exec -T spicav5 python -c "import torch; print('[start1_v3] torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"
fi
if [ "$ACTION" = "build" ]; then exit 0; fi

WINPTY=""; EXEC_TTY=()
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*)
    export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' MSYS2_ENV_CONV_EXCL='*'
    if command -v winpty >/dev/null 2>&1; then WINPTY="winpty"; else EXEC_TTY=(-T); fi ;;
esac
exec_it() {
  $WINPTY "${COMPOSE[@]}" exec ${EXEC_TTY[@]+"${EXEC_TTY[@]}"} -e SPICA_MACHINE="$MACHINE" "$@"
}

tb_kill() {
  "${COMPOSE[@]}" exec -T spicav5 bash -c 'for p in /proc/[0-9]*; do [ "$p" = "/proc/$$" ] && continue; tr "\0" " " < "$p/cmdline" 2>/dev/null | grep -q "tensorboard --logdir.* --port '"$TB_PORT"' " && kill "${p#/proc/}" 2>/dev/null; done; exit 0'
}
start_tb() {
  tb_kill
  "${COMPOSE[@]}" exec -d spicav5 bash -c "tensorboard --logdir '$CHECKPOINTS_DIR' --port '$TB_PORT' --bind_all > /workspace/tb_server_stage1_v3.log 2>&1"
  echo "[start1_v3] TensorBoard: http://localhost:$TB_PORT"
}

case "$ACTION" in
  shell) exec_it spicav5 bash ;;
  tb) start_tb ;;
  infer) exec_it spicav5 bash stage1_v3/infer.sh "$@" ;;
  ponly) exec_it spicav5 bash stage1_v3/ponly.sh "$@" ;;
  stats) exec_it spicav5 bash stage1_v3/stats.sh "$@" ;;
  dn)
    tb_kill
    exec_it -e SPICA_TB_PORT="$TB_PORT" spicav5 bash stage1_v3/denoise.sh "$@" ;;
  dncheck) exec_it spicav5 bash stage1_v3/dncheck.sh "$@" ;;
  dnfixed) exec_it spicav5 bash stage1_v3/dnfixed.sh "$@" ;;
  v4)
    tb_kill
    exec_it -e SPICA_TB_PORT="$TB_PORT" spicav5 bash stage1_v3/pipeline_v4.sh "$@" ;;
  best) exec_it spicav5 python stage1_v3/mark_best.py --machine "$MACHINE" --machines configs/machines.yaml --run "$BEST_RUN" --epoch "$BEST_EPOCH" ;;
  train)
    tb_kill
    if [ -n "$RESUME" ]; then
      exec_it -e SPICA_TB_PORT="$TB_PORT" -e SPICA_RESUME="$RESUME" -e SPICA_RESUME_TAG="$RESUME_TAG" spicav5 bash stage1_v3/train.sh "$@"
    else
      exec_it -e SPICA_TB_PORT="$TB_PORT" spicav5 bash stage1_v3/train.sh "$@"
    fi ;;
esac
