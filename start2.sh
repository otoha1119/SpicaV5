#!/bin/bash
# =============================================================================
# SpicaV5 Stage 2 起動スクリプト（start2.sh）— 取扱説明
#   Stage 2（EID-like1024 → PCD1024 の教師あり U-Net、ILUMENATE = Koons et al., Med Phys 2025 準拠）の発火点。
#   Stage 1 の start.sh と同じ流儀・同じ Docker コンテナ（spicav5、docker/compose.*.yaml）を使う。Stage 1 の学習中に打ってもコンテナを作り直さない。
#   OS 共通。mac / Linux はターミナル、Windows は Git Bash か WSL。リポジトリ直下（このファイルがある場所）でホストのシェルから実行する。
#   コンテナ内では動かない（/.dockerenv があれば即エラー）。コンテナ内で使うのは dataset_stage2.sh の方。
#
# ■ 何をするか
#   1. configs/machines.yaml からマシン定義（GPU 世代・データのパス）を読む（start.sh と同じ）
#   2. GPU 世代に合う compose を選び、コンテナが無ければ起動する（イメージが無ければビルド）
#   3. サブコマンドに応じてコンテナ内のスクリプトを exec する
#
# ■ 引数の規則
#   bash start2.sh [マシン名] [dataset|build|shell|down|tb] [resume <run> [tag]] [best <run> <epoch>] [--flag value ...]
#   ・--flag より前の単語を読む。順不同。dataset / build / shell / down / tb / resume / best / infer はサブコマンド、それ以外の単語は「マシン名」とみなし、
#     sh 内の MACHINE より優先する。マシン名を 2 つ渡すとエラー。configs/machines.yaml に無い名前もエラー（候補を表示）。
#   ・サブコマンド無し = 学習（train_stage2.sh → stage2/run_train.py → train.py）。設定は stage2/configs/train.yaml（数値・症例分割）と mode.yaml（方式）、
#     保存先は machines.yaml の stage2_checkpoints_dir/<run>（run = 起動時刻 yyyy_mmdd_HHMM、同一分の衝突はエラー）。--flag で上書き（schema のフラグだけ。
#     list は --val_cases PCD-017,PCD-018）。上書きは実効値として <run>/launch.yaml に保存され、再開はそれを読む。
#     学習開始時に起動器がその run の tb/ だけを logdir に TensorBoard を起動し、ホストのブラウザを開く（machines.yaml の **stage2_tb_port**。Stage 1 の tb_port とは別ポートで、
#     起動・停止は自分のポートのものだけ。同じマシンで両 Stage を同時に学習しても互いの TensorBoard を止めない）。
#   ・resume <run> [latest|best|<epoch>] : その run の checkpoint から続きを学習（optimizer / RNG / 進捗を復元。run 起動時の設定を使う。--n_epochs 等の上書き可）。
#   ・best <run> <epoch> : weights/epoch_NNN/ を best/ にコピーして best.txt に記録する（手動の上書き）。通常は学習中に val の train.yaml log.best_metric
#               （rmse 最小 | ssim / psnr 最大）が更新されるたびに best/ が自動で書かれるので、目視で別の epoch にしたいときだけ使う（stage2/mark_best.py）。
#   ・tb : Stage 2 の全 run を並べた TensorBoard を起動してブラウザを開く（logdir = stage2_checkpoints_dir）。
#   ・dataset : Stage 2 の学習データ作成。INPUT_DIR の <case>/<slice>.png（uint16、一辺 input_size = 512）を scale 倍（2 → 1024）に補間し、
#               machines.yaml の eidlike1024_dir に同じ <case>/<slice>.png で書く（値の規約 stored = HU + 1400 はそのまま）。dataset_stage2.sh → stage2/make_dataset.py。
#               変換元は dataset_stage2.sh の INPUT_DIR（実行ごとに変わるもの）、出力先 = 学習入力は machines.yaml の eidlike1024_dir（マシンごとのデータ配置。
#               教師 PCD1024 は pcd1024_dir）、方式（scale / interp / input_size）は stage2/configs/dataset.yaml。
#               --flag で上書き（stage2/configs/schema.py の DATASET / MACHINE にあるフラグと --input_dir だけ。無いフラグはエラー）。
#               出力先は container_data_root（DataSet のマウント）配下であること。既にあればエラー（上書きしない）。
#   ・build   : イメージを（再）ビルドしてコンテナ起動・torch/cuda 確認まで。学習はしない（start.sh build と同じ）。
#   ・shell / down : コンテナに入る / 停止・削除（start.sh と同じ。コンテナは Stage 1 と共用なので down は Stage 1 も止める）。
#   ・infer : 未実装（別フェーズ）。
#   ・コンテナが起動済みなら、どのアクションでも up を呼ばず exec だけ行う（Stage 1 の学習中でも安全）。build は起動済みなら拒否。止めるのは down だけ。
#     compose の設定を変えたときは、中の処理が終わってから down → 起動し直す（起動済みのままだと反映されず、注意が出る）。
#
# ■ 例
#   bash start2.sh                                # 学習（sh 内の MACHINE）。TensorBoard が開く
#   bash start2.sh PC1 --n_epochs 50 --batch_size 32   # 上書きして学習
#   bash start2.sh --val_cases PCD-017,PCD-018 --test_cases PCD-019,PCD-020 --train_cases PCD-001,PCD-002,...   # 分割の一時変更（恒久的には train.yaml）
#   bash start2.sh resume 2026_0908_2130          # その run の latest から再開
#   bash start2.sh resume 2026_0908_2130 30 --n_epochs 200   # epoch 30 の checkpoint から、epoch 数を延ばして再開
#   bash start2.sh best 2026_0908_2130 30         # epoch 30 を best にする（手動。自動更新は学習中に val で行われる）
#   bash start2.sh tb                             # Stage 2 の全 run を並べた TensorBoard（stage2_tb_port）
#   bash start2.sh dataset                        # dataset_stage2.sh の INPUT_DIR → machines.yaml の eidlike1024_dir、stage2/configs/dataset.yaml の方式
#   bash start2.sh PC1 dataset                    # マシン名を指定（sh 内の MACHINE より優先）
#   bash start2.sh dataset --input_dir /workspace/stage1/checkpoints/2026_0907_2222/infer/epoch_141/PCD512_v2/2026_0908_120000/full
#   bash start2.sh dataset --input_dir /workspace/DataSet/EID_v5 --eidlike1024_dir /workspace/DataSet/EID1024_v1     # 実 EID（推論用）も同じ道具で 1024 にする（出力先を一時上書き）
#   bash start2.sh dataset --interp bilinear      # 方式の上書き
#   bash start2.sh PC1 build                      # イメージをビルド（本番機の初回。start.sh build と同じ）
#   bash start2.sh shell                          # コンテナに入る（手動: bash dataset_stage2.sh [--flag ...]）
#
# ■ 設定ファイル（既定値は無い。無いキーはエラー）
#   stage2/configs/train.yaml     学習パラメータ（optim / data / log。症例分割 train_cases / val_cases / test_cases もここ）
#   stage2/configs/mode.yaml      方式（arch / base_ch / n_pool / init_type / loss / final_act / residual / serial_batches）
#   stage2/configs/dataset.yaml   データ作成の方式（scale / interp / input_size）
#   stage2/configs/schema.py      Stage 2 の設定ファイルの唯一の正（必須キー・型・フラグ）
#   configs/machines.yaml         マシン定義（Stage 共通。Stage 2 は gpu_gen / host_data_root / container_data_root / num_threads / pcd1024_dir / eidlike1024_dir / stage2_checkpoints_dir / stage2_tb_port を使う。
#                                 tb_port は compose が Stage 1 のポートも公開するために読むだけ）
#
# ■ run ディレクトリ（正は stage2/util/run_paths.py）
#   <stage2_checkpoints_dir>/<run>/  launch.yaml（実効設定。再開で launch_resume_*.yaml が増える）, dataset_info.yaml（症例分割とペア枚数）, loss_log.txt,
#     latest/ best/（net_G.pth + state.pth。best は val の log.best_metric 更新で自動、bash start2.sh best で手動）, best.txt, weights/epoch_NNN/（save_epoch_freq ごと）, output_images/, tb/
#
# ■ ホスト要件・Windows・WSL・改行コード: start.sh と同じ（docker compose v2、python3 + pyyaml。MSYS のパス変換停止、winpty、wslpath 変換、LF 固定）
# =============================================================================
set -euo pipefail

# --- ホスト専用。コンテナ内で叩かれたら docker が無いので即エラー ---
if [ -f /.dockerenv ]; then
  echo "[start2] start2.sh はホスト側で実行するスクリプトです（コンテナ内には docker がありません）。コンテナ内で使うのは bash train_stage2.sh / bash dataset_stage2.sh です" >&2
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

# --- 引数: --flag より前の単語を読む。dataset / build / shell / down / train / infer はサブコマンド、それ以外の単語はマシン名 ---
ACTION=train; BUILD=""; MACHINE_ARG=""; RESUME=""; RESUME_TAG="latest"; BEST_RUN=""; BEST_EPOCH=""
while [ $# -gt 0 ]; do
  case "$1" in
    dataset) ACTION=dataset; shift ;;
    build)   ACTION=build; BUILD="--build"; shift ;;
    shell)   ACTION=shell; shift ;;
    down)    ACTION=down; shift ;;
    tb)      ACTION=tb; shift ;;
    train)   ACTION=train; shift ;;
    infer)   ACTION=infer; shift ;;
    resume)
      shift
      [ $# -gt 0 ] && [[ "$1" != --* ]] || { echo "[start2] resume には run 名が必要です: bash start2.sh resume yyyy_mmdd_HHMM [latest|best|<epoch>]" >&2; exit 2; }
      RESUME="$1"; shift
      if [ $# -gt 0 ] && [[ "$1" =~ ^([0-9]+|latest|best)$ ]]; then RESUME_TAG="$1"; shift; fi ;;
    best)
      shift
      [ $# -ge 2 ] && [[ "$1" != --* ]] && [[ "$2" =~ ^[0-9]+$ ]] || { echo "[start2] best には run 名と epoch 番号が必要です: bash start2.sh best yyyy_mmdd_HHMM <epoch>" >&2; exit 2; }
      ACTION=best; BEST_RUN="$1"; BEST_EPOCH="$2"; shift 2 ;;
    --*)     break ;;
    *)
      if [ -n "$MACHINE_ARG" ]; then echo "[start2] マシン名が 2 つ指定されています: $MACHINE_ARG, $1" >&2; exit 2; fi
      MACHINE_ARG="$1"; shift ;;
  esac
done
if [ -n "$MACHINE_ARG" ]; then
  echo "[start2] マシン名をコマンド引数で上書き: $MACHINE -> $MACHINE_ARG"
  MACHINE="$MACHINE_ARG"
fi
if [ "$ACTION" = infer ]; then
  echo "[start2] Stage 2 の infer は未実装です（別フェーズ）。今使えるのは: bash start2.sh [train] | resume | dataset | tb | build | shell | down" >&2; exit 2
fi
case "$ACTION" in
  build|shell|down|tb|best)
    if [ $# -gt 0 ]; then echo "[start2] $ACTION に --flag は付けられません: $*" >&2; exit 2; fi ;;
esac

# --- ホストの python（machines.yaml を読む） ---
PY=""
for c in python3 python; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c "import yaml" 2>/dev/null; then PY="$c"; break; fi
done
[ -n "$PY" ] || { echo "[start2] python3 と pyyaml がホストに必要です (pip install pyyaml)" >&2; exit 1; }

# --- machines.yaml から gpu_gen / host_data_root / container_data_root / tb_port を取る（欠落はエラー） ---
IFS=$'\t' read -r GPU_GEN HOST_DATA_ROOT CONTAINER_DATA_ROOT TB_PORT CHECKPOINTS_DIR TB_PORT_STAGE2 < <("$PY" - "$MACHINES" "$MACHINE" <<'PYEOF'
import sys, yaml
path, name = sys.argv[1], sys.argv[2]
m = yaml.safe_load(open(path, encoding="utf-8"))
if not isinstance(m, dict) or name not in m:
    sys.exit(f"[start2] {path} にエントリ '{name}' がありません。候補: {list(m) if isinstance(m, dict) else '(不正な形式)'}  → start2.sh の MACHINE かコマンド引数のマシン名を直してください")
e = m[name]
missing = [k for k in ("gpu_gen", "host_data_root", "container_data_root", "tb_port", "stage2_checkpoints_dir", "stage2_tb_port") if k not in e]
if missing:
    sys.exit(f"[start2] machines.yaml の '{name}' にキーがありません: {missing}")
print("\t".join(str(e[k]) for k in ("gpu_gen", "host_data_root", "container_data_root", "tb_port", "stage2_checkpoints_dir", "stage2_tb_port")))  # tb_port は compose が Stage 1 のポートも公開するため
PYEOF
)

# --- host_data_root の正規化（start.sh と同じ）: WSL の bash から起動したときは Windows 表記を /mnt/<drive> に変換する ---
if grep -qi microsoft /proc/version 2>/dev/null && [[ "$HOST_DATA_ROOT" =~ ^[A-Za-z]:[/\\] ]]; then
  command -v wslpath >/dev/null 2>&1 || { echo "[start2] WSL ですが wslpath が見つかりません。configs/machines.yaml の host_data_root を /mnt/<drive>/... で書いてください" >&2; exit 1; }
  HOST_DATA_ROOT_WIN="$HOST_DATA_ROOT"
  HOST_DATA_ROOT="$(wslpath -u "$HOST_DATA_ROOT_WIN")"
  echo "[start2] WSL: host_data_root を変換 $HOST_DATA_ROOT_WIN -> $HOST_DATA_ROOT"
fi
[ -d "$HOST_DATA_ROOT" ] || { echo "[start2] host_data_root がホストに存在しません: $HOST_DATA_ROOT  → configs/machines.yaml の '$MACHINE' を確認" >&2; exit 1; }

case "$GPU_GEN" in
  30|40) GEN=gen30 ;;
  50)    GEN=gen50 ;;
  0)     GEN=cpu ;;
  *) echo "[start2] gpu_gen=$GPU_GEN は未対応 (30 / 40 / 50 / 0)" >&2; exit 1 ;;
esac
COMPOSE=("docker" "compose" "-f" "$ROOT/docker/compose.$GEN.yaml" "-p" "spicav5")
export HOST_DATA_ROOT CONTAINER_DATA_ROOT TB_PORT TB_PORT_STAGE2 SPICA_MACHINE="$MACHINE"
echo "[start2] machine=$MACHINE gpu_gen=$GPU_GEN -> $GEN | mount $HOST_DATA_ROOT -> $CONTAINER_DATA_ROOT | action=$ACTION ${BUILD:+(rebuild)}${RESUME:+ resume=$RESUME tag=$RESUME_TAG}"

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
port_published() { [ -n "$(docker port spicav5 "$1" 2>/dev/null)" ]; }
if container_running; then
  if [ "$ACTION" = build ]; then
    echo "[start2] コンテナ spicav5 は起動済みです（Stage 1 / Stage 2 の学習中かもしれません）。build はコンテナを作り直すので、中の処理が終わってから bash start2.sh down → bash start2.sh build の順で実行してください" >&2
    exit 2
  fi
  check_container
  echo "[start2] コンテナ spicav5 は起動済み（中の処理は触らない。マシン名・イメージは今回の指定と一致）。up は呼ばず exec だけ行う"
  if ! port_published "$TB_PORT_STAGE2"; then
    echo "[start2] 注意: 起動中のコンテナはポート $TB_PORT_STAGE2 を公開していません（compose の設定を変えた後に down していない）。処理は動くが TensorBoard はホストから見えない。中の処理が終わってから bash start2.sh down → 起動し直してください"
  fi
else
  "${COMPOSE[@]}" up -d $BUILD
  "${COMPOSE[@]}" exec -T spicav5 python -c "import torch; print('[start2] torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"
fi
if [ "$ACTION" = build ]; then
  echo "[start2] ビルド完了。コンテナ spicav5 は起動したまま（学習: bash start2.sh / データ作成: bash start2.sh dataset / 停止: bash start2.sh down）"
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
tb_running_all() {  # コンテナ内で **Stage 2 のポートかつ全 run の logdir** の TensorBoard が動いているか（run 単位のものは別物として扱う。2026-09-09）
  "${COMPOSE[@]}" exec -T spicav5 bash -c 'for p in /proc/[0-9]*; do [ "$p" = "/proc/$$" ] && continue; tr "\0" " " < "$p/cmdline" 2>/dev/null | grep -qF -- "tensorboard --logdir '"$CHECKPOINTS_DIR"' --port '"$TB_PORT_STAGE2"' " && exit 0; done; exit 1'
}
tb_kill() {  # コンテナ内の **Stage 2 のポート（stage2_tb_port）** の TensorBoard だけ止める（Stage 1 の tb_port のものは触らない）
  "${COMPOSE[@]}" exec -T spicav5 bash -c 'for p in /proc/[0-9]*; do [ "$p" = "/proc/$$" ] && continue; tr "\0" " " < "$p/cmdline" 2>/dev/null | grep -q "tensorboard --logdir.* --port '"$TB_PORT_STAGE2"' " && kill "${p#/proc/}" 2>/dev/null; done; exit 0'
}
start_tensorboard() {  # 全 run 表示（logdir = stage2_checkpoints_dir、port = stage2_tb_port）。start.sh と同じく、起動済みなら起動し直さない
  local url="http://localhost:$TB_PORT_STAGE2"
  if ! tb_running_all; then
    tb_kill  # run 単位の TensorBoard（学習が起動したもの）が同じポートにいれば止めて、全 run 表示に切り替える
    echo "[start2] TensorBoard を起動: logdir=$CHECKPOINTS_DIR port=${TB_PORT_STAGE2}（ログ: /workspace/tb_server_stage2.log）"
    "${COMPOSE[@]}" exec -d spicav5 bash -c "tensorboard --logdir '$CHECKPOINTS_DIR' --port '$TB_PORT_STAGE2' --bind_all > /workspace/tb_server_stage2.log 2>&1"
  else
    echo "[start2] TensorBoard（全 run）は起動済み ($url)"
  fi
  if command -v curl >/dev/null 2>&1; then
    for _ in $(seq 1 20); do curl -s -o /dev/null "$url" && break; sleep 1; done
  else
    sleep 5
  fi
  echo "[start2] TensorBoard: $url"
  open_browser "$url" || echo "[start2] ブラウザを自動で開けませんでした。$url を手で開いてください"
}
open_when_ready() {  # バックグラウンド: 起動器が起動する run 単位の TensorBoard の応答を待ってブラウザを開く（最大 120 秒）
  local url="http://localhost:$TB_PORT_STAGE2"
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

case "$ACTION" in
  tb)      start_tensorboard ;;
  shell)   exec_it spicav5 bash ;;
  dataset) exec_it spicav5 bash dataset_stage2.sh "$@" ;;
  best)    exec_it spicav5 python stage2/mark_best.py --machine "$MACHINE" --machines configs/machines.yaml --run "$BEST_RUN" --epoch "$BEST_EPOCH" ;;
  train)
    tb_kill
    if command -v curl >/dev/null 2>&1; then open_when_ready & fi
    if [ -n "$RESUME" ]; then
      exec_it -e SPICA_TB_PORT="$TB_PORT_STAGE2" -e SPICA_RESUME="$RESUME" -e SPICA_RESUME_TAG="$RESUME_TAG" spicav5 bash train_stage2.sh "$@"
    else
      exec_it -e SPICA_TB_PORT="$TB_PORT_STAGE2" spicav5 bash train_stage2.sh "$@"
    fi ;;
esac
