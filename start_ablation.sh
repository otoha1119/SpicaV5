#!/bin/bash
# =============================================================================
# SpicaV5 アブレーション起動スクリプト（start_ablation.sh）— 取扱説明（2026-10-27 新設。start_ablation.sh を複製して「アブレーション名」を足したもの）
#   ablation/<名前>/ に自己完結で置いた比較実験（今は SRCycleGAN だけ）の発火点。Stage 1 / 2 と同じ Docker コンテナ（spicav5）を使い、学習中に打っても作り直さない。
#   OS 共通。リポジトリ直下でホストのシェルから実行する。コンテナ内では動かない（/.dockerenv があれば即エラー）。
#
# ■ 引数の規則
#   bash start_ablation.sh [マシン名] <アブレーション名> [infer|build|shell|down|tb] [resume <run> [tag]] [best <run> <epoch>] [--flag value ...]
#   ・--flag より前の単語を読む。順不同。infer / build / shell / down / tb / resume / best はサブコマンド。
#     それ以外の単語は、ablation/ 直下にそのディレクトリ（train.sh 入り）があれば「アブレーション名」、無ければ「マシン名」（sh 内の MACHINE より優先）。
#     アブレーション名は shell / down / build 以外では必須（無ければ候補を出してエラー）。マシン名を 2 つ渡すとエラー。
#   ・サブコマンド無し = 学習（ablation/<名前>/train.sh → run_train.py → train.py）。設定は ablation/<名前>/configs/train.yaml、
#     保存先は machines.yaml の ablation_checkpoints_dir/<名前>/<run>（run = 起動時刻 yyyy_mmdd_HHMM）。--flag で上書き（その ablation の schema のフラグだけ）。
#     学習開始時に起動器がその run の tb/ だけを logdir に TensorBoard を起動する（machines.yaml の **ablation_tb_port**。Stage 1 / 2 とは別ポート）。
#   ・resume <run> [latest|best|<epoch>] : その run の checkpoint から続きを学習。
#   ・best <run> <epoch> : weights/epoch_NNN/ を best/ にコピー（ablation/<名前>/mark_best.py）。
#   ・tb : そのアブレーションの全 run を並べた TensorBoard（logdir = ablation_checkpoints_dir/<名前>）。
#   ・infer : 推論（ablation/<名前>/infer.sh。重みと入力は sh の WEIGHT_DIR / INPUT_DIR、--weight_dir / --input_dir で上書き可、方式は configs/infer.yaml）。
#   ・build / shell / down : start.sh と同じ（コンテナは共用。down は Stage 1 / 2 も止める）。
#   ・コンテナが起動済みなら、どのアクションでも up を呼ばず exec だけ。build は起動済みなら拒否。compose を変えたら中の処理が終わってから down → 起動し直す。
#
# ■ アブレーションを増やすとき
#   ablation/<新しい名前>/ に train.sh / infer.sh / mark_best.py（と configs/ 以下）を同じ契約で置けば、このファイルは変えずに bash start_ablation.sh <マシン> <新しい名前> で動く。
#   一覧と目的は ablation/README.md に 1 行足す。
#
# ■ 例
#   bash start_ablation.sh PC3 SRCycleGAN                       # 学習（論文の設定: batch 4、lr 1e-5、100 + 100 epoch）
#   bash start_ablation.sh PC3 SRCycleGAN --n_epochs 50 --n_epochs_decay 50
#   bash start_ablation.sh PC3 SRCycleGAN resume 2026_1027_1200 # latest から再開
#   bash start_ablation.sh PC3 SRCycleGAN infer --max_slices 4 --save_panel true
#   bash start_ablation.sh PC3 SRCycleGAN best 2026_1027_1200 120
#   bash start_ablation.sh PC3 SRCycleGAN tb
#   bash start_ablation.sh PC3 shell
# =============================================================================
set -euo pipefail

# --- ホスト専用。コンテナ内で叩かれたら docker が無いので即エラー ---
if [ -f /.dockerenv ]; then
  echo "[start_ablation] start_ablation.sh はホスト側で実行するスクリプトです（コンテナ内には docker がありません）。コンテナ内で使うのは bash ablation/<名前>/train.sh / infer.sh です" >&2
  exit 2
fi

# --- Windows（Git Bash / MSYS2）対策（start.sh F-04 / F-05 と同じ） ---
WINPTY=""; EXEC_TTY=()
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*)
    export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' MSYS2_ENV_CONV_EXCL='*'
    if command -v winpty >/dev/null 2>&1; then WINPTY="winpty"; else EXEC_TTY=(-T); fi ;;
esac
exec_it() {  # 対話 exec。Windows では winpty か -T を付ける。引数は docker compose exec に渡すもの（-e ... サービス コマンド）
  $WINPTY "${COMPOSE[@]}" exec ${EXEC_TTY[@]+"${EXEC_TTY[@]}"} -e SPICA_MACHINE="$MACHINE" "$@"  # 今回のマシン名を毎回渡す（コンテナ作成時の値に頼らない。2026-09-09）
}

# ===== ここだけマシンごとに書き換える =====
MACHINE="PC2"          # configs/machines.yaml のエントリ名（PC1 / PC2 / mac / ...）
# =========================================

ROOT="$(cd "$(dirname "$0")" && pwd)"
MACHINES="$ROOT/configs/machines.yaml"

# --- 引数: --flag より前の単語を読む。build / shell / down / tb / train / infer / resume / best はサブコマンド、ablation/ にあるディレクトリ名はアブレーション名、それ以外はマシン名 ---
ACTION=train; BUILD=""; MACHINE_ARG=""; ABLATION=""; RESUME=""; RESUME_TAG="latest"; BEST_RUN=""; BEST_EPOCH=""
while [ $# -gt 0 ]; do
  case "$1" in
    build)   ACTION=build; BUILD="--build"; shift ;;
    shell)   ACTION=shell; shift ;;
    down)    ACTION=down; shift ;;
    tb)      ACTION=tb; shift ;;
    train)   ACTION=train; shift ;;
    infer)   ACTION=infer; shift ;;
    resume)
      shift
      [ $# -gt 0 ] && [[ "$1" != --* ]] || { echo "[start_ablation] resume には run 名が必要です: bash start_ablation.sh resume yyyy_mmdd_HHMM [latest|best|<epoch>]" >&2; exit 2; }
      RESUME="$1"; shift
      if [ $# -gt 0 ] && [[ "$1" =~ ^([0-9]+|latest|best)$ ]]; then RESUME_TAG="$1"; shift; fi ;;
    best)
      shift
      [ $# -ge 2 ] && [[ "$1" != --* ]] && [[ "$2" =~ ^[0-9]+$ ]] || { echo "[start_ablation] best には run 名と epoch 番号が必要です: bash start_ablation.sh best yyyy_mmdd_HHMM <epoch>" >&2; exit 2; }
      ACTION=best; BEST_RUN="$1"; BEST_EPOCH="$2"; shift 2 ;;
    --*)     break ;;
    *)
      if [ -d "$ROOT/ablation/$1" ] && [ -f "$ROOT/ablation/$1/train.sh" ]; then
        if [ -n "$ABLATION" ]; then echo "[start_ablation] アブレーション名が 2 つ指定されています: $ABLATION, $1" >&2; exit 2; fi
        ABLATION="$1"; shift; continue
      fi
      if [ -n "$MACHINE_ARG" ]; then echo "[start_ablation] マシン名が 2 つ指定されています: $MACHINE_ARG, $1（ablation/ にそのディレクトリも無い）" >&2; exit 2; fi
      MACHINE_ARG="$1"; shift ;;
  esac
done
if [ -n "$MACHINE_ARG" ]; then
  echo "[start_ablation] マシン名をコマンド引数で上書き: $MACHINE -> $MACHINE_ARG"
  MACHINE="$MACHINE_ARG"
fi
case "$ACTION" in
  build|shell|down|tb|best)
    if [ $# -gt 0 ]; then echo "[start_ablation] $ACTION に --flag は付けられません: $*" >&2; exit 2; fi ;;
esac
case "$ACTION" in
  shell|down|build) ;;
  *)
    if [ -z "$ABLATION" ]; then
      echo "[start_ablation] アブレーション名を指定してください（ablation/ 直下のディレクトリ名）。候補: $(ls -d "$ROOT"/ablation/*/ 2>/dev/null | xargs -n1 basename | grep -v '^checkpoints$' | tr '\n' ' ')" >&2; exit 2
    fi ;;
esac

# --- ホストの python（machines.yaml を読む） ---
PY=""
for c in python3 python; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c "import yaml" 2>/dev/null; then PY="$c"; break; fi
done
[ -n "$PY" ] || { echo "[start_ablation] python3 と pyyaml がホストに必要です (pip install pyyaml)" >&2; exit 1; }

# --- machines.yaml から gpu_gen / host_data_root / container_data_root / tb_port を取る（欠落はエラー） ---
IFS=$'\t' read -r GPU_GEN HOST_DATA_ROOT CONTAINER_DATA_ROOT TB_PORT TB_PORT_STAGE2 ABL_CKPT TB_PORT_ABLATION < <("$PY" - "$MACHINES" "$MACHINE" <<'PYEOF'
import sys, yaml
path, name = sys.argv[1], sys.argv[2]
m = yaml.safe_load(open(path, encoding="utf-8"))
if not isinstance(m, dict) or name not in m:
    sys.exit(f"[start_ablation] {path} にエントリ '{name}' がありません。候補: {list(m) if isinstance(m, dict) else '(不正な形式)'}  → start_ablation.sh の MACHINE かコマンド引数のマシン名を直してください")
e = m[name]
missing = [k for k in ("gpu_gen", "host_data_root", "container_data_root", "tb_port", "stage2_tb_port", "ablation_checkpoints_dir", "ablation_tb_port") if k not in e]
if missing:
    sys.exit(f"[start_ablation] machines.yaml の '{name}' にキーがありません: {missing}")
print("\t".join(str(e[k]) for k in ("gpu_gen", "host_data_root", "container_data_root", "tb_port", "stage2_tb_port", "ablation_checkpoints_dir", "ablation_tb_port")))  # tb_port / stage2_tb_port は compose が 3 ポートとも公開するため
PYEOF
)

# --- host_data_root の正規化（start.sh と同じ）: WSL の bash から起動したときは Windows 表記を /mnt/<drive> に変換する ---
if grep -qi microsoft /proc/version 2>/dev/null && [[ "$HOST_DATA_ROOT" =~ ^[A-Za-z]:[/\\] ]]; then
  command -v wslpath >/dev/null 2>&1 || { echo "[start_ablation] WSL ですが wslpath が見つかりません。configs/machines.yaml の host_data_root を /mnt/<drive>/... で書いてください" >&2; exit 1; }
  HOST_DATA_ROOT_WIN="$HOST_DATA_ROOT"
  HOST_DATA_ROOT="$(wslpath -u "$HOST_DATA_ROOT_WIN")"
  echo "[start_ablation] WSL: host_data_root を変換 $HOST_DATA_ROOT_WIN -> $HOST_DATA_ROOT"
fi
[ -d "$HOST_DATA_ROOT" ] || { echo "[start_ablation] host_data_root がホストに存在しません: $HOST_DATA_ROOT  → configs/machines.yaml の '$MACHINE' を確認" >&2; exit 1; }

case "$GPU_GEN" in
  30|40) GEN=gen30 ;;
  50)    GEN=gen50 ;;
  0)     GEN=cpu ;;
  *) echo "[start_ablation] gpu_gen=$GPU_GEN は未対応 (30 / 40 / 50 / 0)" >&2; exit 1 ;;
esac
COMPOSE=("docker" "compose" "-f" "$ROOT/docker/compose.$GEN.yaml" "-p" "spicav5")
export HOST_DATA_ROOT CONTAINER_DATA_ROOT TB_PORT TB_PORT_STAGE2 TB_PORT_ABLATION SPICA_MACHINE="$MACHINE"
CHECKPOINTS_DIR="$ABL_CKPT/$ABLATION"   # この ablation の全 run（tb の logdir）
echo "[start_ablation] ablation=${ABLATION:-(none)} machine=$MACHINE gpu_gen=$GPU_GEN -> $GEN | mount $HOST_DATA_ROOT -> $CONTAINER_DATA_ROOT | action=$ACTION ${BUILD:+(rebuild)}${RESUME:+ resume=$RESUME tag=$RESUME_TAG}"

if [ "$ACTION" = down ]; then
  "${COMPOSE[@]}" down
  exit 0
fi

# 起動の規則（start.sh と同じ。コンテナは Stage 1 / Stage 2 で共用なので、中で動いている処理を止めないことを最優先にする）
#   ・コンテナ spicav5 が起動済みなら、**どのアクションでも up を呼ばず exec だけ行う**。compose は設定が変わっていると up -d でコンテナを作り直し、中の学習を殺すため
#   ・build は起動済みなら拒否する。コンテナを止めるのは down だけ
container_running() { [ "$(docker inspect -f '{{.State.Running}}' spicav5 2>/dev/null)" = "true" ]; }
check_container() {  # 起動済みコンテナが今回の指定（マシン名・イメージ・データマウント）と一致するか。違えば止める（exec だけでは変えられない）
  local env_machine image mount_src host_real
  env_machine=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' spicav5 2>/dev/null | sed -n 's/^SPICA_MACHINE=//p' | head -1)
  image=$(docker inspect -f '{{.Config.Image}}' spicav5 2>/dev/null)
  mount_src=$(docker inspect -f "{{range .Mounts}}{{if eq .Destination \"$CONTAINER_DATA_ROOT\"}}{{.Source}}{{end}}{{end}}" spicav5 2>/dev/null)
  if [ "$env_machine" != "$MACHINE" ]; then
    echo "[start_ablation] 起動中のコンテナ spicav5 はマシン '$env_machine' で作られています（今回の指定は '$MACHINE'）。マウントも作成時のままなので、中の処理が終わってから bash start_ablation.sh down → 起動し直してください" >&2
    exit 2
  fi
  if [ "$image" != "spicav5-$GEN" ]; then
    echo "[start_ablation] 起動中のコンテナ spicav5 のイメージは '$image' です（今回の指定 gpu_gen=$GPU_GEN → spicav5-$GEN）。中の処理が終わってから bash start_ablation.sh down → 起動し直してください" >&2
    exit 2
  fi
  host_real=$(cd "$HOST_DATA_ROOT" 2>/dev/null && pwd -P)
  if [ -n "$mount_src" ] && [ "$mount_src" != "$HOST_DATA_ROOT" ] && [ "$mount_src" != "$host_real" ]; then
    echo "[start_ablation] 注意: 起動中のコンテナのデータマウント元は '$mount_src' で、今回の host_data_root '$HOST_DATA_ROOT' と表記が違います（同じ場所なら問題ない。違う場所なら down → 起動し直す）"
  fi
}
port_published() { [ -n "$(docker port spicav5 "$1" 2>/dev/null)" ]; }
if container_running; then
  if [ "$ACTION" = build ]; then
    echo "[start_ablation] コンテナ spicav5 は起動済みです（Stage 1 / Stage 2 の学習中かもしれません）。build はコンテナを作り直すので、中の処理が終わってから bash start_ablation.sh down → bash start_ablation.sh build の順で実行してください" >&2
    exit 2
  fi
  check_container
  echo "[start_ablation] コンテナ spicav5 は起動済み（中の処理は触らない。マシン名・イメージは今回の指定と一致）。up は呼ばず exec だけ行う"
  if ! port_published "$TB_PORT_ABLATION"; then
    echo "[start_ablation] 注意: 起動中のコンテナはポート $TB_PORT_ABLATION を公開していません（compose の設定を変えた後に down していない）。処理は動くが TensorBoard はホストから見えない。中の処理が終わってから bash start_ablation.sh down → 起動し直してください"
  fi
else
  "${COMPOSE[@]}" up -d $BUILD
  "${COMPOSE[@]}" exec -T spicav5 python -c "import torch; print('[start_ablation] torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"
fi
if [ "$ACTION" = build ]; then
  echo "[start_ablation] ビルド完了。コンテナ spicav5 は起動したまま（学習: bash start_ablation.sh <マシン> <名前> / 停止: bash start_ablation.sh down）"
  exit 0
fi

# --- TensorBoard（start.sh と同じ仕組み。学習時は起動器 stage2/run_train.py が run 単位で起動する） ---
open_browser() {
  local url="$1"
  case "$(uname -s)" in
    Darwin) open "$url" ;;
    MINGW*|MSYS*|CYGWIN*) cmd //c start "" "$url" ;;
    Linux) if grep -qi microsoft /proc/version 2>/dev/null; then cmd.exe /c start "" "$url" 2>/dev/null; else xdg-open "$url" 2>/dev/null; fi ;;
    *) return 1 ;;
  esac
}
tb_running_all() {  # コンテナ内で **アブレーションのポートかつ全 run の logdir** の TensorBoard が動いているか（run 単位のものは別物として扱う。2026-09-09）
  "${COMPOSE[@]}" exec -T spicav5 bash -c 'for p in /proc/[0-9]*; do [ "$p" = "/proc/$$" ] && continue; tr "\0" " " < "$p/cmdline" 2>/dev/null | grep -qF -- "tensorboard --logdir '"$CHECKPOINTS_DIR"' --port '"$TB_PORT_ABLATION"' " && exit 0; done; exit 1'
}
tb_kill() {  # コンテナ内の **アブレーションのポート（ablation_tb_port）** の TensorBoard だけ止める（Stage 1 / 2 のポートのものは触らない）
  "${COMPOSE[@]}" exec -T spicav5 bash -c 'for p in /proc/[0-9]*; do [ "$p" = "/proc/$$" ] && continue; tr "\0" " " < "$p/cmdline" 2>/dev/null | grep -q "tensorboard --logdir.* --port '"$TB_PORT_ABLATION"' " && kill "${p#/proc/}" 2>/dev/null; done; exit 0'
}
start_tensorboard() {  # 全 run 表示（logdir = ablation_checkpoints_dir/<名前>、port = ablation_tb_port）。start.sh と同じく、起動済みなら起動し直さない
  local url="http://localhost:$TB_PORT_ABLATION"
  if ! tb_running_all; then
    tb_kill  # run 単位の TensorBoard（学習が起動したもの）が同じポートにいれば止めて、全 run 表示に切り替える
    echo "[start_ablation] TensorBoard を起動: logdir=$CHECKPOINTS_DIR port=${TB_PORT_ABLATION}（ログ: /workspace/tb_server_ablation.log）"
    "${COMPOSE[@]}" exec -d spicav5 bash -c "tensorboard --logdir '$CHECKPOINTS_DIR' --port '$TB_PORT_ABLATION' --bind_all > /workspace/tb_server_ablation.log 2>&1"
  else
    echo "[start_ablation] TensorBoard（全 run）は起動済み ($url)"
  fi
  if command -v curl >/dev/null 2>&1; then
    for _ in $(seq 1 20); do curl -s -o /dev/null "$url" && break; sleep 1; done
  else
    sleep 5
  fi
  echo "[start_ablation] TensorBoard: $url"
  open_browser "$url" || echo "[start_ablation] ブラウザを自動で開けませんでした。$url を手で開いてください"
}
open_when_ready() {  # バックグラウンド: 起動器が起動する run 単位の TensorBoard の応答を待ってブラウザを開く（最大 120 秒）
  local url="http://localhost:$TB_PORT_ABLATION"
  for _ in $(seq 1 120); do
    if curl -s -o /dev/null "$url"; then
      echo "[start_ablation] TensorBoard: $url（この run だけ。全 run は bash start_ablation.sh tb）"
      open_browser "$url" || echo "[start_ablation] ブラウザを自動で開けませんでした。$url を手で開いてください"
      return
    fi
    sleep 1
  done
  echo "[start_ablation] TensorBoard の応答がありません: $url（<run>/tensorboard.log を確認）"
}

case "$ACTION" in
  tb)      start_tensorboard ;;
  shell)   exec_it spicav5 bash ;;
  infer)   exec_it spicav5 bash "ablation/$ABLATION/infer.sh" "$@" ;;
  best)    exec_it spicav5 python "ablation/$ABLATION/mark_best.py" --machine "$MACHINE" --machines configs/machines.yaml --run "$BEST_RUN" --epoch "$BEST_EPOCH" ;;
  train)
    tb_kill
    if command -v curl >/dev/null 2>&1; then open_when_ready & fi
    if [ -n "$RESUME" ]; then
      exec_it -e SPICA_TB_PORT="$TB_PORT_ABLATION" -e SPICA_RESUME="$RESUME" -e SPICA_RESUME_TAG="$RESUME_TAG" spicav5 bash "ablation/$ABLATION/train.sh" "$@"
    else
      exec_it -e SPICA_TB_PORT="$TB_PORT_ABLATION" spicav5 bash "ablation/$ABLATION/train.sh" "$@"
    fi ;;
esac
