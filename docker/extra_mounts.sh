# docker/extra_mounts.sh — runs/ と outputs/ をリポジトリの外に置くマシンのための追加マウント（2026-11-12、PC5 用）。
# 起動 sh（start.sh / start2.sh / old/start.sh / old/start2.sh / old/start_ablation.sh）が COMPOSE を作った直後に source する。
#
#   入力（呼び出し側で定義済みのもの）: PY（pyyaml 入りの python）、MACHINES、MACHINE、ROOT、COMPOSE（配列）
#   machines.yaml の各マシンに host_runs_root / host_outputs_root を必ず書く（既定値なし）:
#     none         → 追加マウントしない。runs/・outputs/ はリポジトリの中のまま（PC5 以外はこれ。今までと同じ動き）
#     ホストのパス → そのフォルダをコンテナの /workspace/runs・/workspace/outputs にマウントする。
#                    Windows 表記（D:/...）は WSL から起動したとき wslpath で /mnt/d/... に変換する（host_data_root と同じ）。
#                    フォルダが無ければ起動前に止める（compose は無いパスを空ディレクトリとして作ってしまうため）
#   出力: COMPOSE に -f docker/compose.mount_<runs|outputs>.yaml を足し、HOST_RUNS_ROOT / HOST_OUTPUTS_ROOT を export する。
#   関数 check_extra_mounts: 起動中のコンテナのマウントが今回の指定と一致するかを調べ、違えば止める
#     （違うまま exec すると、学習結果が今回の指定と別の場所に書かれる。直すには中の処理が終わってから down → 起動し直す）

_XM_TAG="[$(basename "$0" .sh)]"

IFS=$'\t' read -r _XM_RUNS _XM_OUTPUTS < <("$PY" - "$MACHINES" "$MACHINE" <<'PYEOF'
import sys, yaml
path, name = sys.argv[1], sys.argv[2]
m = yaml.safe_load(open(path, encoding="utf-8"))
if not isinstance(m, dict) or name not in m:
    sys.exit(f"{path} にエントリ '{name}' がありません")
e = m[name]
missing = [k for k in ("host_runs_root", "host_outputs_root") if k not in e or e[k] in (None, "")]
if missing:
    sys.exit(f"machines.yaml の '{name}' にキーがありません: {missing}（追加マウントしないマシンは none と書く）")
print(f"{e['host_runs_root']}\t{e['host_outputs_root']}")
PYEOF
) || { echo "$_XM_TAG machines.yaml の host_runs_root / host_outputs_root を読めませんでした（上のエラーを参照）" >&2; exit 1; }

_xm_resolve() {  # $1 = キー名、$2 = machines.yaml の値 → ホストのパスを echo（none はそのまま）
  local key="$1" v="$2"
  if [ "$v" = none ]; then echo none; return 0; fi
  if grep -qi microsoft /proc/version 2>/dev/null && [[ "$v" =~ ^[A-Za-z]:[/\\] ]]; then
    command -v wslpath >/dev/null 2>&1 || { echo "$_XM_TAG WSL ですが wslpath が見つかりません。configs/machines.yaml の $key を /mnt/<drive>/... で書いてください" >&2; return 1; }
    v="$(wslpath -u "$v")"
  fi
  [ -d "$v" ] || { echo "$_XM_TAG $key のフォルダがホストにありません: $v  → 先に作ってください（compose は無いパスを空ディレクトリとして作ってしまうので起動前に止める）" >&2; return 1; }
  echo "$v"
}
HOST_RUNS_ROOT="$(_xm_resolve host_runs_root "$_XM_RUNS")" || exit 1
HOST_OUTPUTS_ROOT="$(_xm_resolve host_outputs_root "$_XM_OUTPUTS")" || exit 1
export HOST_RUNS_ROOT HOST_OUTPUTS_ROOT
if [ "$HOST_RUNS_ROOT" != none ]; then
  COMPOSE+=("-f" "$ROOT/docker/compose.mount_runs.yaml")
  echo "$_XM_TAG runs/ はリポジトリの外: $HOST_RUNS_ROOT -> /workspace/runs"
fi
if [ "$HOST_OUTPUTS_ROOT" != none ]; then
  COMPOSE+=("-f" "$ROOT/docker/compose.mount_outputs.yaml")
  echo "$_XM_TAG outputs/ はリポジトリの外: $HOST_OUTPUTS_ROOT -> /workspace/outputs"
fi

check_extra_mounts() {  # 起動中のコンテナ spicav5 の /workspace/runs・/workspace/outputs のマウントが今回の指定と一致するか。違えば止める
  local pair dest want src real
  for pair in "/workspace/runs|$HOST_RUNS_ROOT" "/workspace/outputs|$HOST_OUTPUTS_ROOT"; do
    dest="${pair%%|*}"; want="${pair#*|}"
    src=$(docker inspect -f "{{range .Mounts}}{{if eq .Destination \"$dest\"}}{{.Source}}{{end}}{{end}}" spicav5 2>/dev/null)
    if [ "$want" = none ]; then
      [ -z "$src" ] && continue
    elif [ -n "$src" ]; then
      real=$(cd "$want" 2>/dev/null && pwd -P)
      if [ "$src" = "$want" ] || [ "$src" = "$real" ]; then continue; fi
    fi
    echo "$_XM_TAG 起動中のコンテナ spicav5 の $dest のマウント元（${src:-なし = リポジトリの中}）が、今回の指定（${want}）と違います。このまま動かすと結果が別の場所に書かれるので止めます。中の処理が終わってから bash $0 $MACHINE down → 起動し直してください" >&2
    exit 2
  done
}
