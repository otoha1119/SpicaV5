#!/bin/bash
# =============================================================================
# SpicaV5 起動スクリプト — 新しい Stage 1（描き手 P: 実 EID だけで学ぶ拡散モデル。コードは stage1/、説明は stage1/README.md）
#   ホストのシェルから実行する（mac / Linux はターミナル、Windows は Git Bash か WSL。bash 経由）。コンテナ内では動かない。
#   作りは old/start.sh（旧 Stage 1 の起動口）と同じ: 同じ compose（docker/compose.<gen>.yaml）・同じコンテナ spicav5 を使う。
#   2026-11-11 以前の実装は bash old/start.sh / old/start2.sh / old/start_ablation.sh（説明は old/README.md）。
#
# ■ 引数
#   bash start.sh [マシン名] [build|shell|down|tb|generate] [resume <run> [tag]] [best <run> <tag>] [--flag value ...]
#   ・サブコマンド無し = 学習。--flag より前の単語を読む（順不同）。サブコマンド以外の単語はマシン名（sh 内の MACHINE より優先）。
#   ・resume の直後は run 名（yyyy_mmdd_HHMM）、その次が latest / best / step 番号なら重みの tag（省略時 latest）。
#   ・best の直後は run 名と tag（weights/ にある step 番号、または latest）。目で選んだ重みを best/ に写す（自動では選ばない）。
#   ・generate は疑似 EID の作成（PCD512 → <pseudo_eid_dir>/<組>/k00…/<症例>/<スライス>.png + manifest.yaml）。run は --run で渡す。
#   ・--flag 以降は上書きとしてそのまま渡す。受け付けるのは stage1/configs/schema.py にあるフラグだけ（無いフラグはエラー）。
#     実効値は run の launch.yaml（generate は <組>/manifest.yaml）に残る。再開の基準は選んだ重みの state.pth の中の実効設定。
#   ・コンテナ spicav5 が起動済みなら、どのアクションでも up を呼ばず exec だけ行う（中の処理を止めない）。build は起動済みなら拒否。止めるのは down だけ。
#
# ■ 例
#   bash start.sh PC1                                   # 学習（run = 起動時刻 yyyy_mmdd_HHMM、置き場 runs/stage1/<run>/）
#   bash start.sh PC1 --max_steps 2000 --compile false  # 上書きして学習
#   bash start.sh PC1 resume 2026_1112_0030             # その run の latest から再開
#   bash start.sh PC1 resume 2026_1112_0030 150000 --max_steps 300000 --max_hours 48
#   bash start.sh PC1 generate --run 2026_1112_0030 --tag latest --blur_px 2 --noise_hu 50 --k 2
#   bash start.sh PC1 --domain pcd --init_run 2026_1111_0517 --max_steps 4000   # 型 X: PCD の描き手（実 EID の描き手の重みから PCD512 で学ぶ）
#   bash start.sh PC1 bridge --enc_run <PCD の描き手> --dec_run <実 EID の描き手>  # 型 X: PCD → ノイズ → 疑似 EID
#   bash start.sh PC1 y [--max_steps 6000]                # 型 Y: 消し手 A を学ぶ（stage1/configs/y.yaml）。再開は bash start.sh PC1 y resume <run>
#   bash start.sh PC1 ygen --run <型 Y の run> --s_hu 100 # 型 Y: PCD → A → E で描き直し → 疑似 EID
#   bash start.sh PC1 best 2026_1112_0030 150000        # weights/step_0150000/ を best/ に写す
#   bash start.sh PC1 tb                                # 全 run を並べた TensorBoard
#   bash start.sh PC1 build | shell | down
#
# ■ 設定（既定値は無い。無いキー・未知のキー・型違いはエラー）
#   stage1/configs/train.yaml / model.yaml / generate.yaml、configs/shared.yaml（両 Stage 共有）、configs/machines.yaml（マシン定義）。唯一の正は stage1/configs/schema.py
#
# ■ TensorBoard
#   学習を始めると、起動器（stage1/run_train.py）がその run の tb/ だけを見る TensorBoard を machines.yaml の tb_port で起動する（旧 Stage 1 と同じポート。同時には回さない）。
#   compose は tb_port / stage2_tb_port / ablation_tb_port の 3 本を公開する（旧コードと共用のコンテナなので、ここでも 3 つとも読む）。
#
# ■ ホスト要件 / Windows
#   docker compose v2、python3 + pyyaml。Git Bash では MSYS のパス変換を止め、対話 exec は winpty（無ければ -T）。WSL では host_data_root を wslpath で変換する（old/start.sh と同じ）。
# =============================================================================
set -euo pipefail

# --- ホスト専用。コンテナ内で叩かれたら docker が無いので即エラー（WSL 判定より先に見る。コンテナは WSL2 のカーネルを共有するので /proc/version だけでは区別できない） ---
if [ -f /.dockerenv ]; then
  echo "[start] start.sh はホスト側で実行するスクリプトです（コンテナ内には docker がありません）。exit でコンテナを抜けて、ホストの PowerShell / ターミナルで bash start.sh <machine> ... を実行してください。コンテナ内で使うのは python stage1/run_train.py などです" >&2
  exit 2
fi

# --- Windows（Git Bash / MSYS2）対策（F-04 / F-05） ---
#   ・MSYS2 ランタイムは native の docker.exe に渡す引数・環境変数の中の /workspace/... を C:/Program Files/Git/... に書き換える。
#     compose の volume 先（CONTAINER_DATA_ROOT）や推論の --weight_dir が壊れるので変換を止める（ホスト側の C:/DataSet は POSIX 形式でないので影響なし）
#   ・mintty から docker exec に擬似 TTY を割り当てると "the input device is not a TTY" で止まるので、対話 exec は winpty を通す（無ければ -T）
WINPTY=""; EXEC_TTY=()
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*)
    export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' MSYS2_ENV_CONV_EXCL='*'
    if command -v winpty >/dev/null 2>&1; then WINPTY="winpty"; else EXEC_TTY=(-T); fi ;;
esac
exec_it() {  # 対話 exec（学習 / 推論 / shell / best）。Windows では winpty か -T を付ける。引数は docker compose exec に渡すもの（-e ... サービス コマンド）
  $WINPTY "${COMPOSE[@]}" exec ${EXEC_TTY[@]+"${EXEC_TTY[@]}"} -e SPICA_MACHINE="$MACHINE" "$@"  # 今回のマシン名を毎回渡す（コンテナ作成時の値に頼らない。2026-09-09）
}

# ===== ここだけマシンごとに書き換える =====
MACHINE="PC1"          # configs/machines.yaml のエントリ名（PC1 / mac / ...）
# =========================================

ROOT="$(cd "$(dirname "$0")" && pwd)"   # リポジトリ直下
MACHINES="$ROOT/configs/machines.yaml"

# --- 引数: --flag より前の単語を読む。build / shell / down / tb / generate / resume / best はサブコマンド、それ以外の単語はマシン名（sh 内の MACHINE より優先）。
#     build はビルドだけで終わる（学習しない）。--flag 以降は上書き引数として stage1/run_train.py / run_generate.py へそのまま渡す ---
ACTION=train; BUILD=""; MACHINE_ARG=""; RESUME=""; RESUME_TAG="latest"; BEST_RUN=""; BEST_TAG=""
while [ $# -gt 0 ]; do
  case "$1" in
    build) ACTION=build; BUILD="--build"; shift ;;
    shell) ACTION=shell; shift ;;
    down)  ACTION=down; shift ;;
    tb)    ACTION=tb; shift ;;
    generate) ACTION=generate; shift ;;
    bridge) ACTION=bridge; shift ;;
    y) ACTION=y; shift ;;
    ygen) ACTION=ygen; shift ;;
    resume)
      shift
      [ $# -gt 0 ] && [[ "$1" != --* ]] || { echo "[start] resume には run 名が必要です: bash start.sh resume yyyy_mmdd_HHMM [latest|best|<step>]" >&2; exit 2; }
      RESUME="$1"; shift
      if [ $# -gt 0 ] && [[ "$1" =~ ^([0-9]+|latest|best)$ ]]; then RESUME_TAG="$1"; shift; fi ;;
    best)
      shift
      [ $# -ge 2 ] && [[ "$1" != --* ]] && [[ "$2" =~ ^([0-9]+|latest)$ ]] || { echo "[start] best には run 名と tag（step 番号か latest）が必要です: bash start.sh best yyyy_mmdd_HHMM <step>" >&2; exit 2; }
      ACTION=best; BEST_RUN="$1"; BEST_TAG="$2"; shift 2 ;;
    --*)   break ;;
    *)
      if [ -n "$MACHINE_ARG" ]; then echo "[start] マシン名が 2 つ指定されています: $MACHINE_ARG, $1" >&2; exit 2; fi
      MACHINE_ARG="$1"; shift ;;
  esac
done
if [ -n "$MACHINE_ARG" ]; then
  echo "[start] マシン名をコマンド引数で上書き: $MACHINE -> $MACHINE_ARG"
  MACHINE="$MACHINE_ARG"
fi
if [ "$ACTION" = build ] && [ $# -gt 0 ]; then
  echo "[start] build はビルドだけで学習しないので --flag は付けられません: $*  （学習は bash start.sh [--flag ...]）" >&2; exit 2
fi

# --- ホストの python（machines.yaml を読む） ---
PY=""
for c in python3 python; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c "import yaml" 2>/dev/null; then PY="$c"; break; fi
done
[ -n "$PY" ] || { echo "[start] python3 と pyyaml がホストに必要です (pip install pyyaml)" >&2; exit 1; }

# --- machines.yaml から gpu_gen / host_data_root / container_data_root を取る（欠落はエラー） ---
IFS=$'\t' read -r GPU_GEN HOST_DATA_ROOT CONTAINER_DATA_ROOT TB_PORT CHECKPOINTS_DIR TB_PORT_STAGE2 TB_PORT_ABLATION < <("$PY" - "$MACHINES" "$MACHINE" <<'PYEOF'
import sys, yaml
path, name = sys.argv[1], sys.argv[2]
m = yaml.safe_load(open(path, encoding="utf-8"))
if not isinstance(m, dict) or name not in m:
    sys.exit(f"[start] {path} にエントリ '{name}' がありません。候補: {list(m) if isinstance(m, dict) else '(不正な形式)'}  → start.sh の MACHINE かコマンド引数のマシン名を直してください")
e = m[name]
missing = [k for k in ("gpu_gen", "host_data_root", "container_data_root", "tb_port", "painter_checkpoints_dir", "stage2_tb_port", "ablation_tb_port") if k not in e]
if missing:
    sys.exit(f"[start] machines.yaml の '{name}' にキーがありません: {missing}")
print("\t".join(str(e[k]) for k in ("gpu_gen", "host_data_root", "container_data_root", "tb_port", "painter_checkpoints_dir", "stage2_tb_port", "ablation_tb_port")))  # タブ区切り（パスの空白対応。F-24）。stage2_tb_port は compose が両ポートを公開するため
PYEOF
)

# --- host_data_root の正規化: WSL の bash から起動した場合、Windows 表記（D:/...）は Linux 側 docker に渡せないので /mnt/d/... に変換する。
#     Git Bash（MINGW）は D:/ のままで Docker Desktop が解釈するので変換しない ---
if grep -qi microsoft /proc/version 2>/dev/null && [[ "$HOST_DATA_ROOT" =~ ^[A-Za-z]:[/\\] ]]; then
  command -v wslpath >/dev/null 2>&1 || { echo "[start] WSL ですが wslpath が見つかりません。configs/machines.yaml の host_data_root を /mnt/<drive>/... で書いてください" >&2; exit 1; }
  HOST_DATA_ROOT_WIN="$HOST_DATA_ROOT"
  HOST_DATA_ROOT="$(wslpath -u "$HOST_DATA_ROOT_WIN")"
  echo "[start] WSL: host_data_root を変換 $HOST_DATA_ROOT_WIN -> $HOST_DATA_ROOT"
fi
[ -d "$HOST_DATA_ROOT" ] || { echo "[start] host_data_root がホストに存在しません: $HOST_DATA_ROOT  → configs/machines.yaml の '$MACHINE' を確認（compose は無いパスを空ディレクトリとして作ってしまうので起動前に止める）" >&2; exit 1; }

case "$GPU_GEN" in
  30|40) GEN=gen30 ;;
  50)    GEN=gen50 ;;
  0)     GEN=cpu ;;
  *) echo "[start] gpu_gen=$GPU_GEN は未対応 (30 / 40 / 50 / 0)" >&2; exit 1 ;;
esac
COMPOSE=("docker" "compose" "-f" "$ROOT/docker/compose.$GEN.yaml" "-p" "spicav5")
export HOST_DATA_ROOT CONTAINER_DATA_ROOT TB_PORT TB_PORT_STAGE2 TB_PORT_ABLATION SPICA_MACHINE="$MACHINE"
echo "[start] machine=$MACHINE gpu_gen=$GPU_GEN -> $GEN | mount $HOST_DATA_ROOT -> $CONTAINER_DATA_ROOT | action=$ACTION ${BUILD:+(rebuild)}${RESUME:+ resume=$RESUME tag=$RESUME_TAG}${BEST_RUN:+ run=$BEST_RUN tag=$BEST_TAG}"

if [ "$ACTION" = down ]; then
  "${COMPOSE[@]}" down
  exit 0
fi

# 起動の規則（2026-09-08 改訂。コンテナは Stage 1 / Stage 2 で共用なので、中で動いている処理を止めないことを最優先にする）
#   ・コンテナ spicav5 が起動済みなら、**どのアクションでも up を呼ばず exec だけ行う**（学習 / 再開も同じ）。compose は設定（ポート・マウント・イメージ）が
#     変わっていると up -d でコンテナを作り直し、中の学習を殺すため。設定を変えたときは、学習が終わってから bash start.sh down → 起動し直す
#   ・build は起動済みなら拒否する（イメージの再ビルドはコンテナの作り直しを伴う）
#   ・コンテナを止めるのは down だけ
container_running() { [ "$(docker inspect -f '{{.State.Running}}' spicav5 2>/dev/null)" = "true" ]; }
check_container() {  # 起動済みコンテナが今回の指定（マシン名・イメージ・データマウント）と一致するか。違えば止める（exec だけでは変えられない）
  local env_machine image mount_src host_real
  env_machine=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' spicav5 2>/dev/null | sed -n 's/^SPICA_MACHINE=//p' | head -1)
  image=$(docker inspect -f '{{.Config.Image}}' spicav5 2>/dev/null)
  mount_src=$(docker inspect -f "{{range .Mounts}}{{if eq .Destination \"$CONTAINER_DATA_ROOT\"}}{{.Source}}{{end}}{{end}}" spicav5 2>/dev/null)
  if [ "$env_machine" != "$MACHINE" ]; then
    echo "[start] 起動中のコンテナ spicav5 はマシン '$env_machine' で作られています（今回の指定は '$MACHINE'）。マウントも作成時のままなので、中の処理が終わってから bash start.sh down → 起動し直してください" >&2
    exit 2
  fi
  if [ "$image" != "spicav5-$GEN" ]; then
    echo "[start] 起動中のコンテナ spicav5 のイメージは '$image' です（今回の指定 gpu_gen=$GPU_GEN → spicav5-$GEN）。中の処理が終わってから bash start.sh down → 起動し直してください" >&2
    exit 2
  fi
  host_real=$(cd "$HOST_DATA_ROOT" 2>/dev/null && pwd -P)
  if [ -n "$mount_src" ] && [ "$mount_src" != "$HOST_DATA_ROOT" ] && [ "$mount_src" != "$host_real" ]; then
    echo "[start] 注意: 起動中のコンテナのデータマウント元は '$mount_src' で、今回の host_data_root '$HOST_DATA_ROOT' と表記が違います（同じ場所なら問題ない。違う場所なら down → 起動し直す）"
  fi
}
port_published() { [ -n "$(docker port spicav5 "$1" 2>/dev/null)" ]; }  # 起動中のコンテナがそのポートをホストに公開しているか
if container_running; then
  if [ "$ACTION" = build ]; then
    echo "[start] コンテナ spicav5 は起動済みです（学習中かもしれません）。build はコンテナを作り直すので、中の処理が終わってから bash start.sh down → bash start.sh build の順で実行してください" >&2
    exit 2
  fi
  check_container
  echo "[start] コンテナ spicav5 は起動済み（中の処理は触らない。マシン名・イメージは今回の指定と一致）。up は呼ばず exec だけ行う"
  if ! port_published "$TB_PORT"; then
    echo "[start] 注意: 起動中のコンテナはポート $TB_PORT を公開していません（compose の設定を変えた後に down していない）。処理は動くが TensorBoard はホストから見えない。中の処理が終わってから bash start.sh down → 起動し直してください"
  fi
else
  "${COMPOSE[@]}" up -d $BUILD
  "${COMPOSE[@]}" exec -T spicav5 python -c "import torch; print('[start] torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"
fi
if [ "$ACTION" = build ]; then
  echo "[start] ビルド完了。コンテナ spicav5 は起動したまま（学習: bash start.sh / 停止: bash start.sh down）"
  exit 0
fi

# --- TensorBoard: コンテナ内でバックグラウンド起動（二重起動しない）→ ホストのブラウザを開く ---
open_browser() {  # ホスト OS ごとにブラウザを開く。開けなくても学習は止めない
  local url="$1"
  case "$(uname -s)" in
    Darwin) open "$url" ;;
    MINGW*|MSYS*|CYGWIN*) cmd //c start "" "$url" ;;
    Linux) if grep -qi microsoft /proc/version 2>/dev/null; then cmd.exe /c start "" "$url" 2>/dev/null; else xdg-open "$url" 2>/dev/null; fi ;;
    *) return 1 ;;
  esac
}
tb_running_all() {  # コンテナ内で **このポート（tb_port）かつ全 run の logdir** の TensorBoard が動いているか（run 単位のものは別物として扱う。2026-09-09）
  "${COMPOSE[@]}" exec -T spicav5 bash -c 'for p in /proc/[0-9]*; do [ "$p" = "/proc/$$" ] && continue; tr "\0" " " < "$p/cmdline" 2>/dev/null | grep -qF -- "tensorboard --logdir '"$CHECKPOINTS_DIR"' --port '"$TB_PORT"' " && exit 0; done; exit 1'
}
start_tensorboard() {
  local url="http://localhost:$TB_PORT"
  if ! tb_running_all; then
    tb_kill  # run 単位の TensorBoard（学習が起動したもの）が同じポートにいれば止めて、全 run 表示に切り替える
    echo "[start] TensorBoard を起動: logdir=$CHECKPOINTS_DIR port=${TB_PORT}（ログ: /workspace/runs/tb_server_stage1.log）"
    "${COMPOSE[@]}" exec -d spicav5 bash -c "tensorboard --logdir '$CHECKPOINTS_DIR' --port '$TB_PORT' --bind_all > /workspace/runs/tb_server_stage1.log 2>&1"
  else
    echo "[start] TensorBoard（全 run）は起動済み ($url)"
  fi
  # 応答を待ってからブラウザを開く（最大 20 秒）。curl が無ければ 5 秒待つだけ
  if command -v curl >/dev/null 2>&1; then
    for _ in $(seq 1 20); do curl -s -o /dev/null "$url" && break; sleep 1; done
  else
    sleep 5
  fi
  echo "[start] TensorBoard: $url"
  open_browser "$url" || echo "[start] ブラウザを自動で開けませんでした。$url を手で開いてください"
}

tb_kill() {  # コンテナ内の **このポート（tb_port）** の TensorBoard を止める（学習側が run 単位のものを起動し直す。Stage 2 のものは止めない）
  "${COMPOSE[@]}" exec -T spicav5 bash -c 'for p in /proc/[0-9]*; do [ "$p" = "/proc/$$" ] && continue; tr "\0" " " < "$p/cmdline" 2>/dev/null | grep -q "tensorboard --logdir.* --port '"$TB_PORT"' " && kill "${p#/proc/}" 2>/dev/null; done; exit 0'
}
open_when_ready() {  # バックグラウンド: 学習側（run_train.py）が起動する run 単位の TensorBoard の応答を待ってブラウザを開く（最大 120 秒）
  local url="http://localhost:$TB_PORT"
  for _ in $(seq 1 120); do
    if curl -s -o /dev/null "$url"; then
      echo "[start] TensorBoard: $url（この run だけ。全 run は bash start.sh tb）"
      open_browser "$url" || echo "[start] ブラウザを自動で開けませんでした。$url を手で開いてください"
      return
    fi
    sleep 1
  done
  echo "[start] TensorBoard の応答がありません: $url（<run>/tensorboard.log を確認）"
}

CFG=(--machine "$MACHINE" --shared configs/shared.yaml --machines configs/machines.yaml)   # コンテナ内の作業ディレクトリは /workspace（リポジトリ直下）
case "$ACTION" in
  tb)    "${COMPOSE[@]}" exec -T spicav5 mkdir -p "$CHECKPOINTS_DIR"; start_tensorboard ;;
  shell) exec_it spicav5 bash ;;
  generate) exec_it spicav5 python stage1/run_generate.py "${CFG[@]}" --generate stage1/configs/generate.yaml -- "$@" ;;
  bridge) exec_it spicav5 python stage1/run_bridge.py "${CFG[@]}" --bridge stage1/configs/bridge.yaml -- "$@" ;;
  y)
    Y_ARGS=("${CFG[@]}" --config stage1/configs/y.yaml)
    if [ -n "$RESUME" ]; then Y_ARGS+=(--resume "$RESUME"); fi
    exec_it spicav5 python stage1/train_y.py "${Y_ARGS[@]}" -- "$@" ;;
  ygen) exec_it spicav5 python stage1/generate_y.py "${CFG[@]}" --config stage1/configs/y_generate.yaml -- "$@" ;;
  best)
    [ $# -eq 0 ] || { echo "[start] best に --flag は付けられません: $*" >&2; exit 2; }
    exec_it spicav5 python stage1/mark_best.py --machine "$MACHINE" --machines configs/machines.yaml --run "$BEST_RUN" --tag "$BEST_TAG" ;;
  train)
    tb_kill
    if command -v curl >/dev/null 2>&1; then open_when_ready & fi
    TRAIN_ARGS=("${CFG[@]}" --train stage1/configs/train.yaml --model stage1/configs/model.yaml)
    if [ -n "$RESUME" ]; then TRAIN_ARGS+=(--resume "$RESUME" --resume_tag "$RESUME_TAG"); fi
    # 起動器が設定を検証 → runs/stage1/<run>/launch.yaml を保存 → python stage1/train.py --run_dir /workspace/runs/stage1/<run> ... を exec する
    exec_it -e SPICA_TB_PORT="$TB_PORT" spicav5 python stage1/run_train.py "${TRAIN_ARGS[@]}" -- "$@" ;;
esac
