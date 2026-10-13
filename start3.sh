#!/bin/bash
# =============================================================================
# SpicaV5 Stage 3 起動スクリプト（start3.sh）— 取扱説明
#   Stage 3（条件付き DDIM: EID-like1024 → PCD-like1024。本家 ermongroup/ddim を stage3/ に vendoring し、入力の channel 連結だけ足したもの）の発火点。
#   決定: docs/decisions/20261013_pivot-stage2-diffusion.md（決定 7 が 1 本目の条件）。変更履歴: stage3/UPSTREAM.md
#   start2.sh を 2026-10-13 に複製。同じ Docker コンテナ（spicav5、docker/compose.*.yaml）を使い、Stage 1 / 2 の学習中に打ってもコンテナを作り直さない。
#   OS 共通。リポジトリ直下でホストのシェルから実行する。コンテナ内では動かない（/.dockerenv があれば即エラー）。
#
# ■ 引数の規則（start2.sh と同じ）
#   bash start3.sh [マシン名] [infer|nps|build|shell|down|tb] [resume <run> [tag]] [best <run> <step>] [--flag value ...]
#   ・サブコマンド無し = 学習（train_stage3.sh → stage3/run_train.py → train.py）。設定は stage3/configs/ddim.yml（本家 church.yml を写したもの）と train.yaml（SpicaV5 側）、
#     保存先は machines.yaml の stage3_checkpoints_dir/<run>（run = 起動時刻 yyyy_mmdd_HHMM）。--flag で上書き（stage3/configs/schema.py のフラグだけ）。
#     学習開始時にその run の TensorBoard を machines.yaml の stage3_tb_port で起動する（Stage 1 の tb_port、Stage 2 の stage2_tb_port とは別）。
#   ・resume <run> [latest|best|<step>] : その run の checkpoint から続きを学習（step 番号は weights/iter_NNNNNNN の番号）。--n_iters 等の上書き可
#   ・best <run> <step> : weights/iter_NNNNNNN/ を best/ にコピー（手動。Stage 3 は best を自動では書かない）
#   ・tb    : Stage 3 の全 run を並べた TensorBoard（stage3_tb_port）
#   ・infer : 推論（infer_stage3.sh → stage3/run_infer.py → inference_dir.py）。重み・入力は infer_stage3.sh の WEIGHT_DIR / INPUT_DIR、方式は stage3/configs/infer.yaml。
#             1024 は 256 タイルに分けて毎ステップ重なりの ε を平均する（WeatherDiffusion 型）。所要時間はタイル数 × sample_steps に比例
#   ・nps   : NPS 評価（nps_stage3.sh → stage3/eval_nps.py）。対象は推論の出力フォルダ（nps_stage3.sh の INFER_DIR）、ROI は stage3/configs/nps.yaml
#   ・データ作成は Stage 2 と同じデータ（eidlike1024_dir）を使うので Stage 3 には無い（必要なら bash start2.sh dataset）
#   ・build / shell / down : start.sh / start2.sh と同じ（コンテナは共用なので down は Stage 1 / 2 も止める）
#   ・コンテナが起動済みなら、どのアクションでも up を呼ばず exec だけ行う。build は起動済みなら拒否。止めるのは down だけ
#
# ■ 例
#   bash start3.sh                                   # 学習（sh 内の MACHINE）
#   bash start3.sh PC3 --batch_size 8 --n_iters 300000
#   bash start3.sh resume 2026_1013_2130             # latest から再開
#   bash start3.sh resume 2026_1013_2130 50000 --n_iters 2000000
#   bash start3.sh infer --cases PCD-017,PCD-018 --slices_per_case 10            # 決定 7 の 1 本目の評価（val 2 症例 × 10 枚、教師あり → metrics.txt）
#   bash start3.sh infer --input_dir /workspace/DataSet/EID_v5 --input eid --teacher false --cases EID-049 --max_slices 10   # 実 EID
#   bash start3.sh infer --sample_steps 10           # S を振る（10 / 20 / 50 / 100）
#   bash start3.sh nps --infer_dir /workspace/output/<推論の出力フォルダ>
#
# ■ 設定ファイル（既定値は無い。無いキーはエラー）
#   stage3/configs/ddim.yml     本家 church.yml（data / model / diffusion / training / optim）。本家からの変更は冒頭のコメントと UPSTREAM.md
#   stage3/configs/train.yaml   seed、症例分割・HU 窓・patch の切り方、監視（頻度・学習中のサンプリング設定）
#   stage3/configs/infer.yaml   推論（S / eta / skip_type / seed / タイル stride・batch / 入力・教師・パネル）
#   stage3/configs/nps.yaml     NPS 評価（ROI の表）
#   stage3/configs/schema.py    唯一の正
#   configs/machines.yaml       Stage 3 は gpu_gen / host_data_root / container_data_root / num_threads / pcd1024_dir / eid_dir / eidlike1024_dir / stage3_checkpoints_dir / stage3_tb_port を使う
#                               （tb_port / stage2_tb_port は compose が全 Stage のポートを公開するために読む）
#
# ■ run ディレクトリ（正は stage3/util/run_paths.py）
#   <stage3_checkpoints_dir>/<run>/  launch.yaml, dataset_info.yaml, loss_log.txt, latest/ best/ weights/iter_NNNNNNN/（model.pth + ema.pth + state.pth）, output_images/, tb/
# =============================================================================
set -euo pipefail

# --- ホスト専用。コンテナ内で叩かれたら docker が無いので即エラー ---
if [ -f /.dockerenv ]; then
  echo "[start3] start3.sh はホスト側で実行するスクリプトです（コンテナ内には docker がありません）。コンテナ内で使うのは bash train_stage3.sh / bash infer_stage3.sh / bash nps_stage3.sh です" >&2
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

# --- 引数: --flag より前の単語を読む。nps / build / shell / down / train / infer / tb はサブコマンド、それ以外の単語はマシン名 ---
ACTION=train; BUILD=""; MACHINE_ARG=""; RESUME=""; RESUME_TAG="latest"; BEST_RUN=""; BEST_EPOCH=""
while [ $# -gt 0 ]; do
  case "$1" in
    nps)     ACTION=nps; shift ;;
    build)   ACTION=build; BUILD="--build"; shift ;;
    shell)   ACTION=shell; shift ;;
    down)    ACTION=down; shift ;;
    tb)      ACTION=tb; shift ;;
    train)   ACTION=train; shift ;;
    infer)   ACTION=infer; shift ;;
    resume)
      shift
      [ $# -gt 0 ] && [[ "$1" != --* ]] || { echo "[start3] resume には run 名が必要です: bash start3.sh resume yyyy_mmdd_HHMM [latest|best|<step>]" >&2; exit 2; }
      RESUME="$1"; shift
      if [ $# -gt 0 ] && [[ "$1" =~ ^([0-9]+|latest|best)$ ]]; then RESUME_TAG="$1"; shift; fi ;;
    best)
      shift
      [ $# -ge 2 ] && [[ "$1" != --* ]] && [[ "$2" =~ ^[0-9]+$ ]] || { echo "[start3] best には run 名と step 番号が必要です: bash start3.sh best yyyy_mmdd_HHMM <step>" >&2; exit 2; }
      ACTION=best; BEST_RUN="$1"; BEST_EPOCH="$2"; shift 2 ;;
    --*)     break ;;
    *)
      if [ -n "$MACHINE_ARG" ]; then echo "[start3] マシン名が 2 つ指定されています: $MACHINE_ARG, $1" >&2; exit 2; fi
      MACHINE_ARG="$1"; shift ;;
  esac
done
if [ -n "$MACHINE_ARG" ]; then
  echo "[start3] マシン名をコマンド引数で上書き: $MACHINE -> $MACHINE_ARG"
  MACHINE="$MACHINE_ARG"
fi
case "$ACTION" in
  build|shell|down|tb|best)
    if [ $# -gt 0 ]; then echo "[start3] $ACTION に --flag は付けられません: $*" >&2; exit 2; fi ;;
esac

# --- ホストの python（machines.yaml を読む） ---
PY=""
for c in python3 python; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c "import yaml" 2>/dev/null; then PY="$c"; break; fi
done
[ -n "$PY" ] || { echo "[start3] python3 と pyyaml がホストに必要です (pip install pyyaml)" >&2; exit 1; }

# --- machines.yaml から gpu_gen / host_data_root / container_data_root / 各 Stage の TensorBoard ポート / stage3_checkpoints_dir を取る（欠落はエラー） ---
IFS=$'\t' read -r GPU_GEN HOST_DATA_ROOT CONTAINER_DATA_ROOT TB_PORT TB_PORT_STAGE2 TB_PORT_STAGE3 CHECKPOINTS_DIR < <("$PY" - "$MACHINES" "$MACHINE" <<'PYEOF'
import sys, yaml
path, name = sys.argv[1], sys.argv[2]
m = yaml.safe_load(open(path, encoding="utf-8"))
if not isinstance(m, dict) or name not in m:
    sys.exit(f"[start3] {path} にエントリ '{name}' がありません。候補: {list(m) if isinstance(m, dict) else '(不正な形式)'}  → start3.sh の MACHINE かコマンド引数のマシン名を直してください")
e = m[name]
keys = ("gpu_gen", "host_data_root", "container_data_root", "tb_port", "stage2_tb_port", "stage3_tb_port", "stage3_checkpoints_dir")
missing = [k for k in keys if k not in e]
if missing:
    sys.exit(f"[start3] machines.yaml の '{name}' にキーがありません: {missing}")
print("\t".join(str(e[k]) for k in keys))  # tb_port / stage2_tb_port は compose が全 Stage のポートを公開するため
PYEOF
)

# --- host_data_root の正規化（start.sh と同じ）: WSL の bash から起動したときは Windows 表記を /mnt/<drive> に変換する ---
if grep -qi microsoft /proc/version 2>/dev/null && [[ "$HOST_DATA_ROOT" =~ ^[A-Za-z]:[/\\] ]]; then
  command -v wslpath >/dev/null 2>&1 || { echo "[start3] WSL ですが wslpath が見つかりません。configs/machines.yaml の host_data_root を /mnt/<drive>/... で書いてください" >&2; exit 1; }
  HOST_DATA_ROOT_WIN="$HOST_DATA_ROOT"
  HOST_DATA_ROOT="$(wslpath -u "$HOST_DATA_ROOT_WIN")"
  echo "[start3] WSL: host_data_root を変換 $HOST_DATA_ROOT_WIN -> $HOST_DATA_ROOT"
fi
[ -d "$HOST_DATA_ROOT" ] || { echo "[start3] host_data_root がホストに存在しません: $HOST_DATA_ROOT  → configs/machines.yaml の '$MACHINE' を確認" >&2; exit 1; }

case "$GPU_GEN" in
  30|40) GEN=gen30 ;;
  50)    GEN=gen50 ;;
  0)     GEN=cpu ;;
  *) echo "[start3] gpu_gen=$GPU_GEN は未対応 (30 / 40 / 50 / 0)" >&2; exit 1 ;;
esac
COMPOSE=("docker" "compose" "-f" "$ROOT/docker/compose.$GEN.yaml" "-p" "spicav5")
export HOST_DATA_ROOT CONTAINER_DATA_ROOT TB_PORT TB_PORT_STAGE2 TB_PORT_STAGE3 SPICA_MACHINE="$MACHINE"
echo "[start3] machine=$MACHINE gpu_gen=$GPU_GEN -> $GEN | mount $HOST_DATA_ROOT -> $CONTAINER_DATA_ROOT | action=$ACTION ${BUILD:+(rebuild)}${RESUME:+ resume=$RESUME tag=$RESUME_TAG}"

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
    echo "[start3] 起動中のコンテナ spicav5 はマシン '$env_machine' で作られています（今回の指定は '$MACHINE'）。マウントも作成時のままなので、中の処理が終わってから bash start3.sh down → 起動し直してください" >&2
    exit 2
  fi
  if [ "$image" != "spicav5-$GEN" ]; then
    echo "[start3] 起動中のコンテナ spicav5 のイメージは '$image' です（今回の指定 gpu_gen=$GPU_GEN → spicav5-$GEN）。中の処理が終わってから bash start3.sh down → 起動し直してください" >&2
    exit 2
  fi
  host_real=$(cd "$HOST_DATA_ROOT" 2>/dev/null && pwd -P)
  if [ -n "$mount_src" ] && [ "$mount_src" != "$HOST_DATA_ROOT" ] && [ "$mount_src" != "$host_real" ]; then
    echo "[start3] 注意: 起動中のコンテナのデータマウント元は '$mount_src' で、今回の host_data_root '$HOST_DATA_ROOT' と表記が違います（同じ場所なら問題ない。違う場所なら down → 起動し直す）"
  fi
}
port_published() { [ -n "$(docker port spicav5 "$1" 2>/dev/null)" ]; }
if container_running; then
  if [ "$ACTION" = build ]; then
    echo "[start3] コンテナ spicav5 は起動済みです（Stage 1 / Stage 2 の学習中かもしれません）。build はコンテナを作り直すので、中の処理が終わってから bash start3.sh down → bash start3.sh build の順で実行してください" >&2
    exit 2
  fi
  check_container
  echo "[start3] コンテナ spicav5 は起動済み（中の処理は触らない。マシン名・イメージは今回の指定と一致）。up は呼ばず exec だけ行う"
  if ! port_published "$TB_PORT_STAGE3"; then
    echo "[start3] 注意: 起動中のコンテナはポート $TB_PORT_STAGE3 を公開していません（compose の設定を変えた後に down していない）。処理は動くが TensorBoard はホストから見えない。中の処理が終わってから bash start3.sh down → 起動し直してください"
  fi
else
  "${COMPOSE[@]}" up -d $BUILD
  "${COMPOSE[@]}" exec -T spicav5 python -c "import torch; print('[start3] torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"
fi
if [ "$ACTION" = build ]; then
  echo "[start3] ビルド完了。コンテナ spicav5 は起動したまま（学習: bash start3.sh / 推論: bash start3.sh infer / 停止: bash start3.sh down）"
  exit 0
fi

# --- TensorBoard（start.sh と同じ仕組み。学習時は起動器 stage3/run_train.py が run 単位で起動する） ---
open_browser() {
  local url="$1"
  case "$(uname -s)" in
    Darwin) open "$url" ;;
    MINGW*|MSYS*|CYGWIN*) cmd //c start "" "$url" ;;
    Linux) if grep -qi microsoft /proc/version 2>/dev/null; then cmd.exe /c start "" "$url" 2>/dev/null; else xdg-open "$url" 2>/dev/null; fi ;;
    *) return 1 ;;
  esac
}
tb_running_all() {  # コンテナ内で **Stage 3 のポートかつ全 run の logdir** の TensorBoard が動いているか（run 単位のものは別物として扱う。2026-09-09）
  "${COMPOSE[@]}" exec -T spicav5 bash -c 'for p in /proc/[0-9]*; do [ "$p" = "/proc/$$" ] && continue; tr "\0" " " < "$p/cmdline" 2>/dev/null | grep -qF -- "tensorboard --logdir '"$CHECKPOINTS_DIR"' --port '"$TB_PORT_STAGE3"' " && exit 0; done; exit 1'
}
tb_kill() {  # コンテナ内の **Stage 3 のポート（stage3_tb_port）** の TensorBoard だけ止める（Stage 1 / 2 のポートのものは触らない）
  "${COMPOSE[@]}" exec -T spicav5 bash -c 'for p in /proc/[0-9]*; do [ "$p" = "/proc/$$" ] && continue; tr "\0" " " < "$p/cmdline" 2>/dev/null | grep -q "tensorboard --logdir.* --port '"$TB_PORT_STAGE3"' " && kill "${p#/proc/}" 2>/dev/null; done; exit 0'
}
start_tensorboard() {  # 全 run 表示（logdir = stage3_checkpoints_dir、port = stage3_tb_port）。start.sh と同じく、起動済みなら起動し直さない
  local url="http://localhost:$TB_PORT_STAGE3"
  if ! tb_running_all; then
    tb_kill  # run 単位の TensorBoard（学習が起動したもの）が同じポートにいれば止めて、全 run 表示に切り替える
    echo "[start3] TensorBoard を起動: logdir=$CHECKPOINTS_DIR port=${TB_PORT_STAGE3}（ログ: /workspace/tb_server_stage3.log）"
    "${COMPOSE[@]}" exec -d spicav5 bash -c "tensorboard --logdir '$CHECKPOINTS_DIR' --port '$TB_PORT_STAGE3' --bind_all > /workspace/tb_server_stage3.log 2>&1"
  else
    echo "[start3] TensorBoard（全 run）は起動済み ($url)"
  fi
  if command -v curl >/dev/null 2>&1; then
    for _ in $(seq 1 20); do curl -s -o /dev/null "$url" && break; sleep 1; done
  else
    sleep 5
  fi
  echo "[start3] TensorBoard: $url"
  open_browser "$url" || echo "[start3] ブラウザを自動で開けませんでした。$url を手で開いてください"
}
open_when_ready() {  # バックグラウンド: 起動器が起動する run 単位の TensorBoard の応答を待ってブラウザを開く（最大 120 秒）
  local url="http://localhost:$TB_PORT_STAGE3"
  for _ in $(seq 1 120); do
    if curl -s -o /dev/null "$url"; then
      echo "[start3] TensorBoard: $url（この run だけ。全 run は bash start3.sh tb）"
      open_browser "$url" || echo "[start3] ブラウザを自動で開けませんでした。$url を手で開いてください"
      return
    fi
    sleep 1
  done
  echo "[start3] TensorBoard の応答がありません: $url（<run>/tensorboard.log を確認）"
}

case "$ACTION" in
  tb)      start_tensorboard ;;
  shell)   exec_it spicav5 bash ;;
  infer)   exec_it spicav5 bash infer_stage3.sh "$@" ;;
  nps)     exec_it spicav5 bash nps_stage3.sh "$@" ;;
  best)    exec_it spicav5 python stage3/mark_best.py --machine "$MACHINE" --machines configs/machines.yaml --run "$BEST_RUN" --step "$BEST_EPOCH" ;;
  train)
    tb_kill
    if command -v curl >/dev/null 2>&1; then open_when_ready & fi
    if [ -n "$RESUME" ]; then
      exec_it -e SPICA_TB_PORT="$TB_PORT_STAGE3" -e SPICA_RESUME="$RESUME" -e SPICA_RESUME_TAG="$RESUME_TAG" spicav5 bash train_stage3.sh "$@"
    else
      exec_it -e SPICA_TB_PORT="$TB_PORT_STAGE3" spicav5 bash train_stage3.sh "$@"
    fi ;;
esac
