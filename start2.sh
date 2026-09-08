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
#   bash start2.sh [マシン名] [dataset|build|shell|down] [--flag value ...]
#   ・--flag より前の単語を読む。順不同。dataset / build / shell / down / train / infer はサブコマンド、それ以外の単語は「マシン名」とみなし、
#     sh 内の MACHINE より優先する。マシン名を 2 つ渡すとエラー。configs/machines.yaml に無い名前もエラー（候補を表示）。
#   ・dataset : Stage 2 の学習データ作成。INPUT_DIR の <case>/<slice>.png（uint16、一辺 input_size = 512）を scale 倍（2 → 1024）に補間し、
#               machines.yaml の eidlike1024_dir に同じ <case>/<slice>.png で書く（値の規約 stored = HU + 1400 はそのまま）。dataset_stage2.sh → stage2/make_dataset.py。
#               変換元は dataset_stage2.sh の INPUT_DIR（実行ごとに変わるもの）、出力先 = 学習入力は machines.yaml の eidlike1024_dir（マシンごとのデータ配置。
#               教師 PCD1024 は pcd1024_dir）、方式（scale / interp / input_size）は stage2/configs/dataset.yaml。
#               --flag で上書き（stage2/configs/schema.py の DATASET / MACHINE にあるフラグと --input_dir だけ。無いフラグはエラー）。
#               出力先は container_data_root（DataSet のマウント）配下であること。既にあればエラー（上書きしない）。
#   ・build   : イメージを（再）ビルドしてコンテナ起動・torch/cuda 確認まで。学習はしない（start.sh build と同じ）。
#   ・shell / down : コンテナに入る / 停止・削除（start.sh と同じ。コンテナは Stage 1 と共用なので down は Stage 1 も止める）。
#   ・train / infer : 未実装（次のフェーズ）。サブコマンド無しは train 扱いなので、今は dataset を明示すること。
#   ・dataset / shell は、コンテナが起動済みなら up を呼ばず exec だけ行う（Stage 1 の学習中でも安全）。
#
# ■ 例
#   bash start2.sh dataset                        # dataset_stage2.sh の INPUT_DIR → machines.yaml の eidlike1024_dir、stage2/configs/dataset.yaml の方式
#   bash start2.sh PC1 dataset                    # マシン名を指定（sh 内の MACHINE より優先）
#   bash start2.sh dataset --input_dir /workspace/stage1/checkpoints/2026_0907_2222/infer/epoch_141/PCD512_v2/2026_0908_120000/full
#   bash start2.sh dataset --input_dir /workspace/DataSet/EID_v5 --eidlike1024_dir /workspace/DataSet/EID1024_v1     # 実 EID（推論用）も同じ道具で 1024 にする（出力先を一時上書き）
#   bash start2.sh dataset --interp bilinear      # 方式の上書き
#   bash start2.sh PC1 build                      # イメージをビルド（本番機の初回。start.sh build と同じ）
#   bash start2.sh shell                          # コンテナに入る（手動: bash dataset_stage2.sh [--flag ...]）
#
# ■ 設定ファイル（既定値は無い。無いキーはエラー）
#   stage2/configs/dataset.yaml   データ作成の方式（scale / interp / input_size）
#   stage2/configs/schema.py      Stage 2 の設定ファイルの唯一の正（必須キー・型・フラグ）
#   configs/machines.yaml         マシン定義（Stage 共通。Stage 2 は gpu_gen / host_data_root / container_data_root / num_threads / tb_port / pcd1024_dir / eidlike1024_dir を使う）
#
# ■ ホスト要件・Windows・WSL・改行コード: start.sh と同じ（docker compose v2、python3 + pyyaml。MSYS のパス変換停止、winpty、wslpath 変換、LF 固定）
# =============================================================================
set -euo pipefail

# --- ホスト専用。コンテナ内で叩かれたら docker が無いので即エラー ---
if [ -f /.dockerenv ]; then
  echo "[start2] start2.sh はホスト側で実行するスクリプトです（コンテナ内には docker がありません）。コンテナ内で使うのは bash dataset_stage2.sh です" >&2
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
  $WINPTY "${COMPOSE[@]}" exec ${EXEC_TTY[@]+"${EXEC_TTY[@]}"} "$@"
}

# ===== ここだけマシンごとに書き換える =====
MACHINE="PC1"          # configs/machines.yaml のエントリ名（PC1 / PC2 / mac / ...）
# =========================================

ROOT="$(cd "$(dirname "$0")" && pwd)"
MACHINES="$ROOT/configs/machines.yaml"

# --- 引数: --flag より前の単語を読む。dataset / build / shell / down / train / infer はサブコマンド、それ以外の単語はマシン名 ---
ACTION=train; BUILD=""; MACHINE_ARG=""
while [ $# -gt 0 ]; do
  case "$1" in
    dataset) ACTION=dataset; shift ;;
    build)   ACTION=build; BUILD="--build"; shift ;;
    shell)   ACTION=shell; shift ;;
    down)    ACTION=down; shift ;;
    train)   ACTION=train; shift ;;
    infer)   ACTION=infer; shift ;;
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
case "$ACTION" in
  train|infer)
    echo "[start2] Stage 2 の $ACTION は未実装です（次のフェーズ）。今使えるのは: bash start2.sh dataset | build | shell | down" >&2; exit 2 ;;
esac
if [ "$ACTION" = build ] && [ $# -gt 0 ]; then
  echo "[start2] build はビルドだけなので --flag は付けられません: $*" >&2; exit 2
fi
if [ "$ACTION" != dataset ] && [ $# -gt 0 ]; then
  echo "[start2] $ACTION に --flag は付けられません: $*" >&2; exit 2
fi

# --- ホストの python（machines.yaml を読む） ---
PY=""
for c in python3 python; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c "import yaml" 2>/dev/null; then PY="$c"; break; fi
done
[ -n "$PY" ] || { echo "[start2] python3 と pyyaml がホストに必要です (pip install pyyaml)" >&2; exit 1; }

# --- machines.yaml から gpu_gen / host_data_root / container_data_root / tb_port を取る（欠落はエラー） ---
IFS=$'\t' read -r GPU_GEN HOST_DATA_ROOT CONTAINER_DATA_ROOT TB_PORT < <("$PY" - "$MACHINES" "$MACHINE" <<'PYEOF'
import sys, yaml
path, name = sys.argv[1], sys.argv[2]
m = yaml.safe_load(open(path, encoding="utf-8"))
if not isinstance(m, dict) or name not in m:
    sys.exit(f"[start2] {path} にエントリ '{name}' がありません。候補: {list(m) if isinstance(m, dict) else '(不正な形式)'}  → start2.sh の MACHINE かコマンド引数のマシン名を直してください")
e = m[name]
missing = [k for k in ("gpu_gen", "host_data_root", "container_data_root", "tb_port") if k not in e]
if missing:
    sys.exit(f"[start2] machines.yaml の '{name}' にキーがありません: {missing}")
print("\t".join(str(e[k]) for k in ("gpu_gen", "host_data_root", "container_data_root", "tb_port")))
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
export HOST_DATA_ROOT CONTAINER_DATA_ROOT TB_PORT SPICA_MACHINE="$MACHINE"
echo "[start2] machine=$MACHINE gpu_gen=$GPU_GEN -> $GEN | mount $HOST_DATA_ROOT -> $CONTAINER_DATA_ROOT | action=$ACTION ${BUILD:+(rebuild)}"

if [ "$ACTION" = down ]; then
  "${COMPOSE[@]}" down
  exit 0
fi

# 起動。dataset / shell はコンテナが起動済みなら up を呼ばず exec だけ行う（Stage 1 の学習中に compose がコンテナを作り直さないように）
container_running() { [ "$(docker inspect -f '{{.State.Running}}' spicav5 2>/dev/null)" = "true" ]; }
case "$ACTION" in
  dataset|shell)
    if container_running; then
      echo "[start2] コンテナ spicav5 は起動済み（Stage 1 の学習中なら触らない）。up は呼ばず exec だけ行う"
    else
      "${COMPOSE[@]}" up -d
      "${COMPOSE[@]}" exec -T spicav5 python -c "import torch; print('[start2] torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"
    fi ;;
  *)
    "${COMPOSE[@]}" up -d $BUILD
    "${COMPOSE[@]}" exec -T spicav5 python -c "import torch; print('[start2] torch', torch.__version__, '| cuda available:', torch.cuda.is_available())" ;;
esac
if [ "$ACTION" = build ]; then
  echo "[start2] ビルド完了。コンテナ spicav5 は起動したまま（データ作成: bash start2.sh dataset / 停止: bash start2.sh down）"
  exit 0
fi

case "$ACTION" in
  shell)   exec_it spicav5 bash ;;
  dataset) exec_it spicav5 bash dataset_stage2.sh "$@" ;;
esac
