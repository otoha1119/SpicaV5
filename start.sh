#!/bin/bash
# =============================================================================
# SpicaV5 起動スクリプト — 取扱説明
#   OS 共通。mac / Linux はターミナル、Windows は Git Bash か WSL。すべて bash 経由（実行権限は不要）。
#   リポジトリ直下（このファイルがある場所）で、ホストのシェルから実行する。コンテナ内（bash start.sh shell で入った先）では動かない
#   （docker が無い。/.dockerenv があれば即エラーで止める）。コンテナ内で使うのは train_stage1.sh / infer_stage1.sh の方。
#
# ■ 何をするか
#   1. configs/machines.yaml からマシン定義（GPU 世代・データのパス）を読む
#   2. GPU 世代に合う compose を選び、コンテナを起動する（イメージが無ければビルド）
#   3. コンテナ内で torch / cuda の状態を 1 行表示する
#   4. コンテナ内で train_stage1.sh を実行して学習を始める（shell / down のときは除く）
#
# ■ 引数の規則
#   bash start.sh [マシン名] [build|shell|down|tb|infer] [resume <run> [tag]] [best <run> <epoch>] [--flag value ...]
#   ・--flag より前の単語を読む。順不同。
#   ・build / shell / down / tb / infer / resume / best はサブコマンド。それ以外の単語は「マシン名」とみなし、sh 内の MACHINE より優先する。
#   ・build は「イメージを（再）ビルドしてコンテナを起動し、torch / cuda の確認を表示して終わる」だけ。学習は始めない（本番機の初期セットアップ用）。
#   ・resume の直後は run 名（yyyy_mmdd_HHMM）、その次が latest / best / epoch 番号なら checkpoint の tag（省略時 latest）。
#   ・best の直後は run 名と epoch 番号（両方必須）。best/ は best.tmp/ に揃えてから差し替える。
#   ・infer は引数を取らない。重みディレクトリ・入力フォルダ・出力形式・元 DICOM ルートは infer_stage1.sh の変数
#     （WEIGHT_DIR / INPUT_DIR / OUTPUT_FORMAT / DICOM_DIR）に書く（同名の --flag で上書き可）。
#   ・マシン名を 2 つ渡すとエラー。configs/machines.yaml に無い名前もエラー（候補を表示）。
#   ・--flag 以降は上書き引数としてそのまま train_stage1.sh / infer_stage1.sh へ渡す（sh の引数が最優先）。
#     受け付けるのは stage1/configs/schema.py にあるフラグだけ（学習は TRAIN / MODE / MACHINE、推論は INFER）。無いフラグはエラー。
#     bool は --flag（= true）か --flag true|false。上書きは実効値として launch.yaml に保存され、再開・推論に引き継がれる（F-02）。
#   ・同じ分に 2 回起動すると 2 回目はエラー（run 名の衝突。F-11）。
#
# ■ 例
#   bash start.sh                            # sh 内の MACHINE で起動 → 学習
#   bash start.sh PC2                        # マシン名を指定（sh 内の MACHINE より優先）
#   bash start.sh build                      # イメージをビルド（既にあれば再ビルド）→ コンテナ起動 → torch/cuda 確認 → 終了。学習はしない
#   bash start.sh --n_epochs 50              # 学習パラメータを上書きして学習
#   bash start.sh PC2 build                  # マシン指定でビルドだけ（本番機で最初にやる）
#   bash start.sh resume 2026_0905_1234      # その run の latest から学習を再開（同じディレクトリに続きを保存）
#   bash start.sh resume 2026_0905_1234 30 --n_epochs 400   # epoch 30 の checkpoint（weights/30_*）から、epoch 数を延ばして再開
#   bash start.sh best 2026_0905_1234 30     # epoch 30 を best にする（weights/epoch_030/ → best/、best.txt に記録。目視で選ぶ）
#   bash start.sh infer                      # 推論。infer_stage1.sh の WEIGHT_DIR / INPUT_DIR / OUTPUT_FORMAT / DICOM_DIR、stage1/configs/infer.yaml の方式
#   bash start.sh infer --mode full --max_slices 4                        # 方式を上書きして推論
#   bash start.sh infer --output_format both --dicom_dir /workspace/DataSet/PhotonCT512_original   # PNG と DICOM の両方を書く
#   bash start.sh mac infer --weight_dir /workspace/stage1/checkpoints/2026_0905_1234/best --input_dir /workspace/DataSet/PCD512_v2
#                                            # GPU 無しのマシン定義（gpu_gen 0 → cpu）で、重みと入力を引数で指定して推論
#   bash start.sh tb                         # TensorBoard だけ起動してブラウザを開く（学習はしない）
#   bash start.sh mac shell                  # 学習せず、mac のコンテナの bash に入る（exit で抜けてもコンテナは残る）
#   bash start.sh mac down                   # mac のコンテナを停止・削除（イメージは残る）
#
# ■ マシン定義（configs/machines.yaml）
#   ・MACHINE（下）または引数のマシン名でエントリを選ぶ。
#   ・gpu_gen → compose: 30 / 40 → docker/compose.gen30.yaml, 50 → docker/compose.gen50.yaml, 0 → docker/compose.cpu.yaml
#   ・host_data_root → container_data_root をそのまま rw マウント（配下丸ごと、SpicaV3 と同じ）。
#     Windows 機は host_data_root を D:/DataSet のようにホスト表記で書く。WSL の bash から起動した場合は docker が Linux 側なので
#     start.sh が wslpath で /mnt/d/DataSet に変換してから渡す（Git Bash は D:/ のままで Docker Desktop が解釈する）。
#     host_data_root がホストに無ければ起動前にエラーで止める（compose は無いパスを空ディレクトリとして作ってしまい、学習が pcd_dir 無しで落ちるまで気付けないため）。
#   ・pcd_dir / eid_dir / checkpoints_dir はコンテナ内から見えるパスで書く。
#   ・実験の保存先は checkpoints_dir/<run>（run = 起動時刻 yyyy_mmdd_HHMM、JST）。レイアウト（正は stage1/util/run_paths.py）:
#       launch.yaml / train_opt.txt / loss_log.txt
#       latest/net_G.pth, net_D.pth, state.pth        直下の重みディレクトリは latest と best だけ
#       best/net_G.pth,   net_D.pth, state.pth        bash start.sh best <run> <epoch> で作る（best.txt に epoch を記録）
#       weights/epoch_NNN/net_G.pth, net_D.pth, state.pth   save_epoch_freq ごと（+ 学習終了時の最終 epoch。保存は .tmp → rename で原子的）
#       output_images/samples/   学習中の 128 patch グリッド（8bit、TensorBoard と同じ表示用）
#       output_images/epoch_NNN/ checkpoint ごとのフル 512（<slice>_pcd / _eidlike / _R.png、16bit、入力と同じ規約）
#       infer/<重みディレクトリ名>/<入力フォルダ名>/<実行時刻>/   bash start.sh infer の出力（実行ごとに別ディレクトリ。full/ patch/ = 16bit PNG、*_dicom/ = DICOM、*_R/ = 残差 PNG、diff_stats.txt、infer.yaml）
#       tb/                      TensorBoard
#
# ■ 設定ファイル（既定値は無い。無いキーはエラー）
#   stage1/configs/train.yaml   学習パラメータ（数値。各行に論文の出典）
#   stage1/configs/mode.yaml    モード切替（sampling / gan_mode / netG / netD / ...）
#   stage1/configs/infer.yaml   推論の方式（mode full|patch|both、patch の size/stride/blend、...）
#   configs/machines.yaml       マシン定義（Stage 共通）
#   stage1/configs/schema.py    上記 4 ファイルの唯一の正（必須キー・型・フラグ）
#
# ■ ホスト要件
#   docker compose v2、python3 + pyyaml（machines.yaml を読むため）。無ければエラーで止まる（フォールバックなし）。
#
# ■ Windows（Git Bash）
#   ・MSYS2 のパス自動変換（/workspace/... → C:/Program Files/Git/...）を MSYS_NO_PATHCONV=1 等で止める（F-04）。
#   ・対話 exec は winpty を通す。winpty が無ければ -T（擬似 TTY なし。tqdm は動く）（F-05）。mintty 以外（Windows Terminal / cmd）なら不要だが害はない。
#   ・WSL から動かす場合は machines.yaml の host_data_root を /mnt/c/DataSet の形にする。
#   ・改行コードは .gitattributes で LF 固定。古い clone が CRLF なら README の手順で取り直す。
#
# ■ TensorBoard（学習開始時に自動）
#   ・学習（train / resume）のときは、コンテナ内で TensorBoard をバックグラウンド起動し（logdir = machines.yaml の checkpoints_dir、
#     port = tb_port、ログ /workspace/tb_server.log）、応答を待ってからホストのブラウザで http://localhost:<tb_port> を開く。
#   ・既に起動していれば起動せずブラウザだけ開く。コンテナを down すると止まる。ポートが塞がっていたら machines.yaml の tb_port を変える。
#   ・ブラウザは mac = open、Windows Git Bash = cmd //c start、Linux = xdg-open、WSL = cmd.exe。開けなくても学習は続く。
#   ・曲線は checkpoints_dir 以下の全 run（yyyy_mmdd_HHMM）が並ぶ。画像は images/current / images/fixed（8bit 表示、同じものが <run>/output_images/samples/）と images/full。
#
# ■ 再開（resume）の仕組み
#   ・学習中は重み（net_G.pth / net_D.pth）に加えて state.pth（optimizer / RNG / epoch / iteration 数）を同じ重みディレクトリに保存する。
#     保存タイミングは本家と同じ（save_latest_freq 枚ごとに latest/、save_epoch_freq epoch ごとに weights/epoch_NNN/）。
#   ・tag は latest | best | <epoch>。best は bash start.sh best <run> <epoch> で手動指定（判定指標は未実装）。
#   ・resume は run の launch.yaml に記録された起動時の設定を使う（今の yaml は無視。差があれば表示）。
#     sh の --flag 上書きだけは末尾に付くので、n_epochs を延ばすなどに使える。
#   ・途中保存（epoch 未完）の latest から再開すると、その epoch を頭からやり直す（ケース 1 はランダム抽選なので害はない）。
#
# ■ よくある操作
#   ・イメージを作り直す: bash start.sh down → bash start.sh build（→ 学習は bash start.sh）
#   ・本番機の初回: bash start.sh build で "[start] torch ... | cuda available: True" が出れば準備完了
#   ・コンテナ内で手動学習: bash start.sh shell → bash train_stage1.sh [--flag value ...]
#   ・コンテナ内で手動推論: bash start.sh shell → bash infer_stage1.sh [--flag value ...]
#
# ■ 推論（infer）= 学習済みの重みで PNG フォルダを丸ごと EID-like に変換する
#   ・重みディレクトリ・入力フォルダ・出力形式・元 DICOM ルートは infer_stage1.sh の WEIGHT_DIR / INPUT_DIR / OUTPUT_FORMAT / DICOM_DIR（同名の --flag で上書き可）。
#   ・G の構成は重みディレクトリの上の run の launch.yaml から（学習時と同じ G）。方式は stage1/configs/infer.yaml。
#   ・デバイスは machines.yaml の gpu_gen（0 → cpu、それ以外 → cuda。cuda が使えなければエラー）。GPU 無しで回すなら gpu_gen 0 のマシン定義を使う。
#   ・full = 512 を一発 / patch = 128 patch を stride 刻みで切って重なり平均 / both = 両方保存 + |full − patch| を diff_stats.txt に。
#   ・出力形式 OUTPUT_FORMAT: png = 16bit PNG（入力と同じ規約）/ dicom = 前処理を戻して DICOM（DICOM_DIR の元ヘッダを継承、HU を元の RescaleSlope/Intercept で格納値に。UID は新規）/ both。
#     DICOM は入力 PNG 名 PCD-nnn-sss と元 DICOM ルート（症例フォルダの数値部分 = nnn、.dcm の名前順 sss 番目）の対応で書く。対応が取れなければエラー。
#   ・GPU に乗っているか: 起動時の "[start] torch ... | cuda available: True/False" を見る
# =============================================================================
set -euo pipefail

# --- ホスト専用。コンテナ内で叩かれたら docker が無いので即エラー（WSL 判定より先に見る。コンテナは WSL2 のカーネルを共有するので /proc/version だけでは区別できない） ---
if [ -f /.dockerenv ]; then
  echo "[start] start.sh はホスト側で実行するスクリプトです（コンテナ内には docker がありません）。exit でコンテナを抜けて、ホストの PowerShell / ターミナルで bash start.sh <machine> ... を実行してください。コンテナ内で使うのは bash train_stage1.sh / bash infer_stage1.sh です" >&2
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
  $WINPTY "${COMPOSE[@]}" exec ${EXEC_TTY[@]+"${EXEC_TTY[@]}"} "$@"
}

# ===== ここだけマシンごとに書き換える =====
MACHINE="PC1"          # configs/machines.yaml のエントリ名（PC1 / mac / ...）
# =========================================

ROOT="$(cd "$(dirname "$0")" && pwd)"
MACHINES="$ROOT/configs/machines.yaml"

# --- 引数: --flag より前の単語を読む。build / shell / down / tb / resume / infer / best はサブコマンド、それ以外の単語はマシン名（sh 内の MACHINE より優先）。
#     build はビルドだけで終わる（学習しない）。
#     --flag 以降は上書き引数として train_stage1.sh / infer_stage1.sh へそのまま渡す ---
ACTION=train; BUILD=""; MACHINE_ARG=""; RESUME=""; RESUME_TAG="latest"; BEST_RUN=""; BEST_EPOCH=""
while [ $# -gt 0 ]; do
  case "$1" in
    build) ACTION=build; BUILD="--build"; shift ;;
    shell) ACTION=shell; shift ;;
    down)  ACTION=down; shift ;;
    tb)    ACTION=tb; shift ;;
    resume)
      shift
      [ $# -gt 0 ] && [[ "$1" != --* ]] || { echo "[start] resume には run 名が必要です: bash start.sh resume yyyy_mmdd_HHMM [latest|best|<epoch>]" >&2; exit 2; }
      RESUME="$1"; shift
      if [ $# -gt 0 ] && [[ "$1" =~ ^([0-9]+|latest|best)$ ]]; then RESUME_TAG="$1"; shift; fi ;;
    infer) ACTION=infer; shift ;;
    best)
      shift
      [ $# -ge 2 ] && [[ "$1" != --* ]] && [[ "$2" =~ ^[0-9]+$ ]] || { echo "[start] best には run 名と epoch 番号が必要です: bash start.sh best yyyy_mmdd_HHMM <epoch>" >&2; exit 2; }
      ACTION=best; BEST_RUN="$1"; BEST_EPOCH="$2"; shift 2 ;;
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
IFS=$'\t' read -r GPU_GEN HOST_DATA_ROOT CONTAINER_DATA_ROOT TB_PORT CHECKPOINTS_DIR < <("$PY" - "$MACHINES" "$MACHINE" <<'PYEOF'
import sys, yaml
path, name = sys.argv[1], sys.argv[2]
m = yaml.safe_load(open(path, encoding="utf-8"))
if not isinstance(m, dict) or name not in m:
    sys.exit(f"[start] {path} にエントリ '{name}' がありません。候補: {list(m) if isinstance(m, dict) else '(不正な形式)'}  → start.sh の MACHINE かコマンド引数のマシン名を直してください")
e = m[name]
missing = [k for k in ("gpu_gen", "host_data_root", "container_data_root", "tb_port", "checkpoints_dir") if k not in e]
if missing:
    sys.exit(f"[start] machines.yaml の '{name}' にキーがありません: {missing}")
print("\t".join(str(e[k]) for k in ("gpu_gen", "host_data_root", "container_data_root", "tb_port", "checkpoints_dir")))  # タブ区切り（パスの空白対応。F-24）
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
export HOST_DATA_ROOT CONTAINER_DATA_ROOT TB_PORT SPICA_MACHINE="$MACHINE"
echo "[start] machine=$MACHINE gpu_gen=$GPU_GEN -> $GEN | mount $HOST_DATA_ROOT -> $CONTAINER_DATA_ROOT | action=$ACTION ${BUILD:+(rebuild)}${RESUME:+ resume=$RESUME tag=$RESUME_TAG}${BEST_RUN:+ run=$BEST_RUN epoch=$BEST_EPOCH}"

if [ "$ACTION" = down ]; then
  "${COMPOSE[@]}" down
  exit 0
fi

# 起動（up -d はイメージが無ければビルドする。build 指定時は --build で強制再ビルド）
"${COMPOSE[@]}" up -d $BUILD
"${COMPOSE[@]}" exec -T spicav5 python -c "import torch; print('[start] torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"
if [ "$ACTION" = build ]; then
  echo "[start] ビルド完了。コンテナ spicav5 は起動したまま（学習: bash start.sh / 推論: bash start.sh infer / 停止: bash start.sh down）"
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
tb_running() {  # コンテナ内で TensorBoard が動いているか（pgrep が無いイメージでも動くよう /proc を走査）
  "${COMPOSE[@]}" exec -T spicav5 bash -c 'for p in /proc/[0-9]*; do [ "$p" = "/proc/$$" ] && continue; tr "\0" " " < "$p/cmdline" 2>/dev/null | grep -q "tensorboard --logdir" && exit 0; done; exit 1'
}
start_tensorboard() {
  local url="http://localhost:$TB_PORT"
  if ! tb_running; then
    echo "[start] TensorBoard を起動: logdir=$CHECKPOINTS_DIR port=${TB_PORT}（ログ: /workspace/tb_server.log）"
    "${COMPOSE[@]}" exec -d spicav5 bash -c "tensorboard --logdir '$CHECKPOINTS_DIR' --port '$TB_PORT' --bind_all > /workspace/tb_server.log 2>&1"
  else
    echo "[start] TensorBoard は起動済み ($url)"
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

case "$ACTION" in
  tb)    start_tensorboard ;;
  shell) exec_it spicav5 bash ;;
  infer) exec_it spicav5 bash infer_stage1.sh "$@" ;;
  best)  exec_it spicav5 python stage1/mark_best.py --machine "$MACHINE" --machines configs/machines.yaml --run "$BEST_RUN" --epoch "$BEST_EPOCH" ;;
  train)
    start_tensorboard
    if [ -n "$RESUME" ]; then
      exec_it -e SPICA_RESUME="$RESUME" -e SPICA_RESUME_TAG="$RESUME_TAG" spicav5 bash train_stage1.sh "$@"
    else
      exec_it spicav5 bash train_stage1.sh "$@"
    fi ;;
esac
