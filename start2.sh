#!/bin/bash
# =============================================================================
# SpicaV5 起動スクリプト — 新しい Stage 2（U = 形を決める回帰 + V = 鋭さとノイズの粒を作るフロー。コードは stage2/、説明は stage2/README.md）
#   ホストのシェルから実行する（mac / Linux はターミナル、Windows は Git Bash か WSL。bash 経由）。コンテナ内では動かない。
#   作りは直下の start.sh（新 Stage 1 の起動口）と同じ: 同じ compose（docker/compose.<gen>.yaml）・同じコンテナ spicav5 を使う。
#   2026-11-11 以前の Stage 2 は bash old/start2.sh（説明は old/README.md）。
#
# ■ 引数
#   bash start2.sh [マシン名] [u|v|infer|check|tb|shell|down|build] [resume <u|v> <run> [tag]] [best <u|v> <run> <tag>] [--flag value ...]
#   ・u = U の学習、v = V の学習（--u_run <U の run> が必須）。--flag より前の単語を読む（順不同）。サブコマンド以外の単語はマシン名（sh 内の MACHINE より優先）。
#   ・resume の直後は種類（u | v）と run 名（yyyy_mmdd_HHMM）、その次が latest / best / step 番号なら重みの tag（省略時 latest）。
#   ・best の直後は種類と run 名と tag（weights/ にある step 番号、または latest）。目で選んだ重みを best/ に写す（学習が上限まで進むと最後の EMA が best になる。自動では選ばない）。
#   ・infer はフォルダ丸ごとの推論（--run と --input_dir で渡す）。check は確認の道具（虫眼鏡の数字・乱数を変えた出力・隣り合うスライス。--kind と --run で渡す）。
#   ・--flag 以降は上書きとしてそのまま渡す。受け付けるのは stage2/configs/schema.py にあるフラグだけ（無いフラグはエラー）。
#     実効値は run の launch.yaml（infer は出力フォルダの infer.yaml、check は numbers.yaml）に残る。再開の基準は選んだ重みの state.pth の中の実効設定。
#   ・コンテナ spicav5 が起動済みなら、どのアクションでも up を呼ばず exec だけ行う（中の処理 = Stage 1 の学習など を止めない）。build は起動済みなら拒否。止めるのは down だけ。
#
# ■ 例（疑似 EID の組 = bash start.sh generate が作ったフォルダ名）
#   bash start2.sh PC1 u --pseudo_eid_set 2026_1112_0030_step0250000_b2.0_c050_a16      # U の学習（置き場 runs/stage2/u/<run>/）
#   bash start2.sh PC1 v --pseudo_eid_set 2026_1112_0030_step0250000_b2.0_c050_a16 --u_run 2026_1113_0900   # V の学習（U の best を固定。置き場 runs/stage2/v/<run>/）
#   bash start2.sh PC1 resume u 2026_1113_0900                 # その run の latest から再開
#   bash start2.sh PC1 resume v 2026_1113_1000 latest --max_steps 200000 --max_hours 24
#   bash start2.sh PC1 infer --run 2026_1113_1000 --input_dir /workspace/DataSet/EID_v5 --cases EID-049          # 実 EID512 → 出力 1024（outputs/stage2/…）
#   bash start2.sh PC1 check --kind u --run 2026_1113_0900 --what magnifier                                      # U だけの段階の確認（設計書 G-2）
#   bash start2.sh PC1 check --kind v --run 2026_1113_1000 --what magnifier,seeds,flicker                        # U + V の確認（G-3・G-4）
#   bash start2.sh PC1 best v 2026_1113_1000 100000
#   bash start2.sh PC1 tb | build | shell | down
#
# ■ 設定（既定値は無い。無いキー・未知のキー・型違いはエラー）
#   stage2/configs/data.yaml / u.yaml / v.yaml / infer.yaml / check.yaml、configs/shared.yaml（両 Stage 共有。明るさを平均する広さ）、configs/machines.yaml。唯一の正は stage2/configs/schema.py
#
# ■ TensorBoard
#   学習を始めると、起動器がその run の tb/ だけを見る TensorBoard を machines.yaml の stage2_tb_port（6007）で起動する（Stage 1 の tb_port 6006 とは別。Stage 1 の TensorBoard は止めない）。
#   bash start2.sh tb は runs/stage2/ 全体（u/ と v/ の全 run）を並べる。
#
# ■ ホスト要件 / Windows
#   docker compose v2、python3 + pyyaml。Git Bash では MSYS のパス変換を止め、対話 exec は winpty（無ければ -T）。WSL では host_data_root を wslpath で変換する（start.sh と同じ）。
# =============================================================================
set -euo pipefail

# --- ホスト専用。コンテナ内で叩かれたら docker が無いので即エラー（WSL 判定より先に見る。コンテナは WSL2 のカーネルを共有するので /proc/version だけでは区別できない） ---
if [ -f /.dockerenv ]; then
  echo "[start2] start2.sh はホスト側で実行するスクリプトです（コンテナ内には docker がありません）。exit でコンテナを抜けて、ホストの PowerShell / ターミナルで bash start2.sh <machine> ... を実行してください。コンテナ内で使うのは python stage2/run_train_u.py などです" >&2
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


# --- 引数: --flag より前の単語を読む。u / v / infer / check / tb / shell / down / build / resume / best はサブコマンド、それ以外の単語はマシン名（sh 内の MACHINE より優先）。
#     build はビルドだけで終わる。--flag 以降は上書き引数として stage2/run_train_u.py / run_train_v.py / run_infer.py / run_check.py へそのまま渡す ---
ACTION=""; BUILD=""; MACHINE_ARG=""; KIND=""; RESUME=""; RESUME_TAG="latest"; BEST_RUN=""; BEST_TAG=""
while [ $# -gt 0 ]; do
  case "$1" in
    build) ACTION=build; BUILD="--build"; shift ;;
    shell) ACTION=shell; shift ;;
    down)  ACTION=down; shift ;;
    tb)    ACTION=tb; shift ;;
    infer) ACTION=infer; shift ;;
    check) ACTION=check; shift ;;
    u|v)   ACTION=train; KIND="$1"; shift ;;
    resume)
      shift
      [ $# -ge 2 ] && [[ "$1" =~ ^(u|v)$ ]] && [[ "$2" != --* ]] || { echo "[start2] resume には種類（u | v）と run 名が必要です: bash start2.sh resume u yyyy_mmdd_HHMM [latest|best|<step>]" >&2; exit 2; }
      ACTION=train; KIND="$1"; RESUME="$2"; shift 2
      if [ $# -gt 0 ] && [[ "$1" =~ ^([0-9]+|latest|best)$ ]]; then RESUME_TAG="$1"; shift; fi ;;
    best)
      shift
      [ $# -ge 3 ] && [[ "$1" =~ ^(u|v)$ ]] && [[ "$2" != --* ]] && [[ "$3" =~ ^([0-9]+|latest)$ ]] || { echo "[start2] best には種類（u | v）と run 名と tag（step 番号か latest）が必要です: bash start2.sh best v yyyy_mmdd_HHMM <step>" >&2; exit 2; }
      ACTION=best; KIND="$1"; BEST_RUN="$2"; BEST_TAG="$3"; shift 3 ;;
    --*)   break ;;
    *)
      if [ -n "$MACHINE_ARG" ]; then echo "[start2] マシン名が 2 つ指定されています: $MACHINE_ARG, $1" >&2; exit 2; fi
      MACHINE_ARG="$1"; shift ;;
  esac
done
if [ -z "$ACTION" ]; then
  echo "[start2] 何をするかを指定してください: bash start2.sh [マシン名] u | v | infer | check | tb | shell | down | build | resume <u|v> <run> [tag] | best <u|v> <run> <tag>  [--flag value ...]" >&2; exit 2
fi
if [ -n "$MACHINE_ARG" ]; then
  echo "[start2] マシン名をコマンド引数で上書き: $MACHINE -> $MACHINE_ARG"
  MACHINE="$MACHINE_ARG"
fi
if [ "$ACTION" = build ] && [ $# -gt 0 ]; then
  echo "[start2] build はビルドだけで学習しないので --flag は付けられません: $*" >&2; exit 2
fi

# --- ホストの python（machines.yaml を読む） ---
PY=""
for c in python3 python; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c "import yaml" 2>/dev/null; then PY="$c"; break; fi
done
[ -n "$PY" ] || { echo "[start2] python3 と pyyaml がホストに必要です (pip install pyyaml)" >&2; exit 1; }

# --- machines.yaml から gpu_gen / host_data_root / container_data_root を取る（欠落はエラー） ---
IFS=$'\t' read -r GPU_GEN HOST_DATA_ROOT CONTAINER_DATA_ROOT TB_PORT CHECKPOINTS_DIR TB_PORT_STAGE2 TB_PORT_ABLATION < <("$PY" - "$MACHINES" "$MACHINE" <<'PYEOF'
import sys, yaml
path, name = sys.argv[1], sys.argv[2]
m = yaml.safe_load(open(path, encoding="utf-8"))
if not isinstance(m, dict) or name not in m:
    sys.exit(f"[start2] {path} にエントリ '{name}' がありません。候補: {list(m) if isinstance(m, dict) else '(不正な形式)'}  → start2.sh の MACHINE かコマンド引数のマシン名を直してください")
e = m[name]
missing = [k for k in ("gpu_gen", "host_data_root", "container_data_root", "tb_port", "flow_checkpoints_dir", "stage2_tb_port", "ablation_tb_port") if k not in e]
if missing:
    sys.exit(f"[start2] machines.yaml の '{name}' にキーがありません: {missing}")
print("\t".join(str(e[k]) for k in ("gpu_gen", "host_data_root", "container_data_root", "tb_port", "flow_checkpoints_dir", "stage2_tb_port", "ablation_tb_port")))  # タブ区切り（パスの空白対応。F-24）。stage2_tb_port は compose が両ポートを公開するため
PYEOF
)

# --- host_data_root の正規化: WSL の bash から起動した場合、Windows 表記（D:/...）は Linux 側 docker に渡せないので /mnt/d/... に変換する。
#     Git Bash（MINGW）は D:/ のままで Docker Desktop が解釈するので変換しない ---
if grep -qi microsoft /proc/version 2>/dev/null && [[ "$HOST_DATA_ROOT" =~ ^[A-Za-z]:[/\\] ]]; then
  command -v wslpath >/dev/null 2>&1 || { echo "[start2] WSL ですが wslpath が見つかりません。configs/machines.yaml の host_data_root を /mnt/<drive>/... で書いてください" >&2; exit 1; }
  HOST_DATA_ROOT_WIN="$HOST_DATA_ROOT"
  HOST_DATA_ROOT="$(wslpath -u "$HOST_DATA_ROOT_WIN")"
  echo "[start2] WSL: host_data_root を変換 $HOST_DATA_ROOT_WIN -> $HOST_DATA_ROOT"
fi
[ -d "$HOST_DATA_ROOT" ] || { echo "[start2] host_data_root がホストに存在しません: $HOST_DATA_ROOT  → configs/machines.yaml の '$MACHINE' を確認（compose は無いパスを空ディレクトリとして作ってしまうので起動前に止める）" >&2; exit 1; }

case "$GPU_GEN" in
  30|40) GEN=gen30 ;;
  50)    GEN=gen50 ;;
  0)     GEN=cpu ;;
  *) echo "[start2] gpu_gen=$GPU_GEN は未対応 (30 / 40 / 50 / 0)" >&2; exit 1 ;;
esac
COMPOSE=("docker" "compose" "-f" "$ROOT/docker/compose.$GEN.yaml" "-p" "spicav5")
source "$ROOT/docker/extra_mounts.sh"   # machines.yaml の host_runs_root / host_outputs_root（PC5 だけパス、ほかは none = 今までどおり）。COMPOSE に -f を足し check_extra_mounts を定義する（2026-11-12）
export HOST_DATA_ROOT CONTAINER_DATA_ROOT TB_PORT TB_PORT_STAGE2 TB_PORT_ABLATION SPICA_MACHINE="$MACHINE"
echo "[start2] machine=$MACHINE gpu_gen=$GPU_GEN -> $GEN | mount $HOST_DATA_ROOT -> $CONTAINER_DATA_ROOT | action=$ACTION ${BUILD:+(rebuild)}${KIND:+ kind=$KIND}${RESUME:+ resume=$RESUME tag=$RESUME_TAG}${BEST_RUN:+ run=$BEST_RUN tag=$BEST_TAG}"
MY_TB_PORT="$TB_PORT_STAGE2"   # この起動口が使う TensorBoard のポート（Stage 2 専用。Stage 1 の tb_port の TensorBoard には触らない）

if [ "$ACTION" = down ]; then
  "${COMPOSE[@]}" down
  exit 0
fi

# 起動の規則（2026-09-08 改訂。コンテナは Stage 1 / Stage 2 で共用なので、中で動いている処理を止めないことを最優先にする）
#   ・コンテナ spicav5 が起動済みなら、**どのアクションでも up を呼ばず exec だけ行う**（学習 / 再開も同じ）。compose は設定（ポート・マウント・イメージ）が
#     変わっていると up -d でコンテナを作り直し、中の学習を殺すため。設定を変えたときは、学習が終わってから bash start2.sh down → 起動し直す
#   ・build は起動済みなら拒否する（イメージの再ビルドはコンテナの作り直しを伴う）
#   ・コンテナを止めるのは down だけ
container_running() { [ "$(docker inspect -f '{{.State.Running}}' spicav5 2>/dev/null)" = "true" ]; }
check_container() {  # 起動済みコンテナが今回の指定（マシン名・イメージ・データマウント）と一致するか。違えば止める（exec だけでは変えられない）
  local env_machine image mount_src host_real
  env_machine=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' spicav5 2>/dev/null | sed -n 's/^SPICA_MACHINE=//p' | head -1)
  image=$(docker inspect -f '{{.Config.Image}}' spicav5 2>/dev/null)
  mount_src=$(docker inspect -f "{{range .Mounts}}{{if eq .Destination \"$CONTAINER_DATA_ROOT\"}}{{.Source}}{{end}}{{end}}" spicav5 2>/dev/null)
  if [ "$env_machine" != "$MACHINE" ]; then
    echo "[start2] 起動中のコンテナ spicav5 はマシン '$env_machine' で作られています（今回の指定は '$MACHINE'）。マウントも作成時のままなので、中の処理が終わってから bash start2.sh down → 起動し直してください" >&2
    exit 2
  fi
  if [ "$image" != "spicav5-$GEN" ]; then
    echo "[start2] 起動中のコンテナ spicav5 のイメージは '$image' です（今回の指定 gpu_gen=$GPU_GEN → spicav5-$GEN）。中の処理が終わってから bash start2.sh down → 起動し直してください" >&2
    exit 2
  fi
  host_real=$(cd "$HOST_DATA_ROOT" 2>/dev/null && pwd -P)
  if [ -n "$mount_src" ] && [ "$mount_src" != "$HOST_DATA_ROOT" ] && [ "$mount_src" != "$host_real" ]; then
    echo "[start2] 注意: 起動中のコンテナのデータマウント元は '$mount_src' で、今回の host_data_root '$HOST_DATA_ROOT' と表記が違います（同じ場所なら問題ない。違う場所なら down → 起動し直す）"
  fi
}
port_published() { [ -n "$(docker port spicav5 "$1" 2>/dev/null)" ]; }  # 起動中のコンテナがそのポートをホストに公開しているか
if container_running; then
  if [ "$ACTION" = build ]; then
    echo "[start2] コンテナ spicav5 は起動済みです（学習中かもしれません）。build はコンテナを作り直すので、中の処理が終わってから bash start2.sh down → bash start2.sh build の順で実行してください" >&2
    exit 2
  fi
  check_container
  check_extra_mounts   # runs/・outputs/ をリポジトリの外に置くマシン（PC5）で、マウントが今回の指定と違えば止める（docker/extra_mounts.sh）
  echo "[start2] コンテナ spicav5 は起動済み（中の処理は触らない。マシン名・イメージは今回の指定と一致）。up は呼ばず exec だけ行う"
  if ! port_published "$MY_TB_PORT"; then
    echo "[start2] 注意: 起動中のコンテナはポート $MY_TB_PORT を公開していません（compose の設定を変えた後に down していない）。処理は動くが TensorBoard はホストから見えない。中の処理が終わってから bash start2.sh down → 起動し直してください"
  fi
else
  "${COMPOSE[@]}" up -d $BUILD
  "${COMPOSE[@]}" exec -T spicav5 python -c "import torch; print('[start2] torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"
fi
if [ "$ACTION" = build ]; then
  echo "[start2] ビルド完了。コンテナ spicav5 は起動したまま（学習: bash start2.sh / 停止: bash start2.sh down）"
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
  "${COMPOSE[@]}" exec -T spicav5 bash -c 'for p in /proc/[0-9]*; do [ "$p" = "/proc/$$" ] && continue; tr "\0" " " < "$p/cmdline" 2>/dev/null | grep -qF -- "tensorboard --logdir '"$CHECKPOINTS_DIR"' --port '"$MY_TB_PORT"' " && exit 0; done; exit 1'
}
start_tensorboard() {
  local url="http://localhost:$MY_TB_PORT"
  if ! tb_running_all; then
    tb_kill  # run 単位の TensorBoard（学習が起動したもの）が同じポートにいれば止めて、全 run 表示に切り替える
    echo "[start2] TensorBoard を起動: logdir=$CHECKPOINTS_DIR port=${MY_TB_PORT}（ログ: /workspace/runs/tb_server_stage2.log）"
    "${COMPOSE[@]}" exec -d spicav5 bash -c "tensorboard --logdir '$CHECKPOINTS_DIR' --port '$MY_TB_PORT' --bind_all > /workspace/runs/tb_server_stage2.log 2>&1"
  else
    echo "[start2] TensorBoard（全 run）は起動済み ($url)"
  fi
  # 応答を待ってからブラウザを開く（最大 20 秒）。curl が無ければ 5 秒待つだけ
  if command -v curl >/dev/null 2>&1; then
    for _ in $(seq 1 20); do curl -s -o /dev/null "$url" && break; sleep 1; done
  else
    sleep 5
  fi
  echo "[start2] TensorBoard: $url"
  open_browser "$url" || echo "[start2] ブラウザを自動で開けませんでした。$url を手で開いてください"
}

tb_kill() {  # コンテナ内の **このポート（tb_port）** の TensorBoard を止める（学習側が run 単位のものを起動し直す。Stage 1 のものは止めない）
  "${COMPOSE[@]}" exec -T spicav5 bash -c 'for p in /proc/[0-9]*; do [ "$p" = "/proc/$$" ] && continue; tr "\0" " " < "$p/cmdline" 2>/dev/null | grep -q "tensorboard --logdir.* --port '"$MY_TB_PORT"' " && kill "${p#/proc/}" 2>/dev/null; done; exit 0'
}
open_when_ready() {  # バックグラウンド: 学習側（stage2 の起動器）が起動する run 単位の TensorBoard の応答を待ってブラウザを開く（最大 120 秒）
  local url="http://localhost:$MY_TB_PORT"
  for _ in $(seq 1 120); do
    if curl -s -o /dev/null "$url"; then
      echo "[start2] TensorBoard: $url（この run だけ。全 run は bash start2.sh tb）"
      open_browser "$url" || echo "[start2] ブラウザを自動で開けませんでした。$url を手で開いてください"
      return
    fi
    sleep 1
  done
  echo "[start2] TensorBoard の応答がありません: $url（<run>/tensorboard.log を確認）"
}

CFG=(--machine "$MACHINE" --machines configs/machines.yaml)   # コンテナ内の作業ディレクトリは /workspace（リポジトリ直下）
case "$ACTION" in
  tb)    "${COMPOSE[@]}" exec -T spicav5 mkdir -p "$CHECKPOINTS_DIR"; start_tensorboard ;;
  shell) exec_it spicav5 bash ;;
  infer) exec_it spicav5 python stage2/run_infer.py "${CFG[@]}" --infer stage2/configs/infer.yaml -- "$@" ;;
  check) exec_it spicav5 python stage2/run_check.py "${CFG[@]}" --check stage2/configs/check.yaml -- "$@" ;;
  best)
    [ $# -eq 0 ] || { echo "[start2] best に --flag は付けられません: $*" >&2; exit 2; }
    exec_it spicav5 python stage2/mark_best.py "${CFG[@]}" --kind "$KIND" --run "$BEST_RUN" --tag "$BEST_TAG" ;;
  train)
    tb_kill
    if command -v curl >/dev/null 2>&1; then open_when_ready & fi
    TRAIN_ARGS=("${CFG[@]}" --data stage2/configs/data.yaml --cfg "stage2/configs/$KIND.yaml" --shared configs/shared.yaml)
    if [ -n "$RESUME" ]; then TRAIN_ARGS+=(--resume "$RESUME" --resume_tag "$RESUME_TAG"); fi
    # 起動器が設定とディスクを検査 → runs/stage2/<u|v>/<run>/launch.yaml を保存 → python stage2/train_<u|v>.py --run_dir /workspace/runs/stage2/<u|v>/<run> ... を exec する
    exec_it -e SPICA_TB_PORT="$MY_TB_PORT" spicav5 python "stage2/run_train_$KIND.py" "${TRAIN_ARGS[@]}" -- "$@" ;;
esac
