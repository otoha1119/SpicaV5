"""train-watch の exporter（コンテナ spicav5 の中で動かす）。

学習ブランチに watch/ が無くても動くように、ホストから標準入力で流し込んで実行する:
    docker exec -i spicav5 python - --machine PC1 --active_min 30 < watch/exporter.py

/workspace/configs/machines.yaml の <machine> のうち名前が checkpoints_dir で終わるキーをすべて run の置き場とみなし、
launch.yaml を持つディレクトリ（深さ 2 まで。ablation は <名前>/<run>）を run とする。
学習プロセスが生きている（/proc の cmdline に run 名を含み、tensorboard ではない）か、event が直近 active_min 分以内に
更新された run だけを JSON にして標準出力に出す。

TensorBoard の event は scalar だけを読む。TFRecord を自前で読み、ファイルごとの読み終わり位置と値をキャッシュ（--cache）に
持って差分だけ読む（画像の多い巨大な event を毎回全部読まないため）。
"""
import argparse
import json
import math
import os
import re
import struct
import sys
import time
from pathlib import Path

import yaml
from tensorboard.compat.proto import event_pb2

MAX_SERIES = 100   # 間引いた系列の最大点数
LAUNCH_MAX = 20000


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine", required=True)
    ap.add_argument("--machines_yaml", required=True)
    ap.add_argument("--active_min", type=float, required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--extra_root", action="append", default=[], help="machines.yaml に無い run 置き場（コンテナ内のパス）")
    return ap.parse_args()


def run_roots(machines_yaml, machine, extra):
    m = yaml.safe_load(Path(machines_yaml).read_text(encoding="utf-8"))
    if machine not in m:
        sys.exit(f"[exporter] machines.yaml に {machine} がありません")
    roots = {k: Path(v) for k, v in m[machine].items() if k.endswith("checkpoints_dir") and v}
    for i, r in enumerate(extra):
        roots[f"extra{i}"] = Path(r)
    return roots


def find_runs(roots):
    runs = []
    for key, root in roots.items():
        if not root.is_dir():
            continue
        for launch in list(root.glob("*/launch.yaml")) + list(root.glob("*/*/launch.yaml")):
            run_dir = launch.parent
            rel = run_dir.relative_to(root).as_posix()
            runs.append((key, rel, run_dir))
    return runs


def process_table():
    procs = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit() or int(p.name) == os.getpid():
            continue
        try:
            cmd = (p / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace").strip()
        except OSError:
            continue
        if cmd:
            procs.append((int(p.name), cmd))
    return procs


def is_alive(run_dir, procs):
    """run 名が語として現れ、run の親ディレクトリのパスも含む train*.py の python プロセスがあるか（pc_agent の PS_SNIPPET と同じ規則）。"""
    pat = re.compile(r"(?<![0-9A-Za-z_.\-])" + re.escape(run_dir.name) + r"(?![0-9A-Za-z_.\-])")
    parent = str(run_dir.parent)
    return any("python" in cmd and "tensorboard" not in cmd and re.search(r"train[0-9A-Za-z_]*\.py", cmd)
               and pat.search(cmd) and parent in cmd for _, cmd in procs)


# ---------------------------------------------------------------- event の差分読み

def read_records(path, offset):
    """offset から完全なレコードだけを読む。書きかけの末尾は次回に回す。"""
    out = []
    with open(path, "rb") as f:
        f.seek(offset)
        while True:
            head = f.read(12)
            if len(head) < 12:
                break
            (length,) = struct.unpack("<Q", head[:8])
            data = f.read(length)
            tail = f.read(4)
            if len(data) < length or len(tail) < 4:
                break
            out.append(data)
            offset = f.tell()
    return out, offset


def scalar_of(value):
    if value.HasField("simple_value"):
        return float(value.simple_value)
    if value.HasField("tensor") and value.metadata.plugin_data.plugin_name == "scalars":
        t = value.tensor
        if t.float_val:
            return float(t.float_val[0])
        if t.double_val:
            return float(t.double_val[0])
        if t.tensor_content:
            fmt = {1: "<f", 2: "<d"}.get(t.dtype)
            if fmt:
                return float(struct.unpack(fmt, t.tensor_content[:struct.calcsize(fmt)])[0])
    return None


def update_file(path, entry):
    """entry = {offset, size, tags: {tag: [[step, wall, value], ...]}}。ファイルが縮んでいたら最初から読み直す。"""
    size = path.stat().st_size
    if size < entry.get("offset", 0):
        entry.clear()
    entry.setdefault("offset", 0)
    entry.setdefault("tags", {})
    recs, entry["offset"] = read_records(path, entry["offset"])
    for data in recs:
        ev = event_pb2.Event.FromString(data)
        if not ev.HasField("summary"):
            continue
        for v in ev.summary.value:
            x = scalar_of(v)
            if x is None:
                continue
            entry["tags"].setdefault(v.tag, []).append([int(ev.step), float(ev.wall_time), x])
    entry["size"] = size


def jnum(x):
    """JSON に NaN / Inf を書かない（文字列にする）。"""
    if x is None:
        return None
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return round(x, 6)


def summarize(points):
    # 書いた順（wall_time）に並べ、step が巻き戻ったら（resume）その step 以降の古い値を捨てる = 今の学習の系列だけを残す
    seq = []
    for s, w, v in sorted(points, key=lambda p: p[1]):
        while seq and seq[-1][0] >= s:
            seq.pop()
        seq.append((s, w, v))
    last_wall = seq[-1][1]
    pts = [(s, v) for s, _, v in seq]
    vals = [v for _, v in pts]
    fin = [(s, v) for s, v in pts if math.isfinite(v)]
    last5 = [v for _, v in fin[-5:]]
    slope = None
    tail = fin[-20:]
    if len(tail) >= 3 and tail[-1][0] != tail[0][0]:
        n = len(tail)
        ms = sum(s for s, _ in tail) / n
        mv = sum(v for _, v in tail) / n
        den = sum((s - ms) ** 2 for s, _ in tail)
        slope = sum((s - ms) * (v - mv) for s, v in tail) / den * 1000 if den else None   # 1000 step あたり
    idx = sorted(set([round(i * (len(pts) - 1) / (MAX_SERIES - 1)) for i in range(MAX_SERIES)])) if len(pts) > MAX_SERIES else range(len(pts))
    return {
        "n": len(pts),
        "last_step": pts[-1][0],
        "last_wall": round(last_wall, 3),   # 最終値を書いた時刻（resume で同じ step にまた NaN が出たのを区別する）
        "last": jnum(vals[-1]),
        "min": jnum(min(v for _, v in fin)) if fin else None,
        "max": jnum(max(v for _, v in fin)) if fin else None,
        "mean_last5": jnum(sum(last5) / len(last5)) if last5 else None,
        "slope_per_1k_steps": jnum(slope),
        "nonfinite": sum(1 for v in vals if not math.isfinite(v)),
        "series": [[pts[i][0], jnum(pts[i][1])] for i in idx],
    }


def main():
    a = parse_args()
    cache_path = Path(a.cache)
    cache = json.loads(cache_path.read_text()) if cache_path.is_file() else {}
    procs = process_table()
    now = time.time()
    out = {"machine": a.machine, "runs": []}
    for key, rel, run_dir in find_runs(run_roots(a.machines_yaml, a.machine, a.extra_root)):
        events = sorted(run_dir.rglob("events.out.tfevents.*"))
        newest = max([e.stat().st_mtime for e in events], default=(run_dir / "launch.yaml").stat().st_mtime)
        alive = is_alive(run_dir, procs)
        if not alive and now - newest > a.active_min * 60:
            continue
        tags = {}
        for e in events:
            entry = cache.setdefault(str(e), {})
            update_file(e, entry)
            for tag, pts in entry["tags"].items():
                tags.setdefault(tag, []).extend(pts)
        summary = {t: summarize(p) for t, p in sorted(tags.items())}
        launch = (run_dir / "launch.yaml").read_text(encoding="utf-8", errors="replace")
        out["runs"].append({
            "root_key": key,
            "run": rel,
            "run_dir": str(run_dir),
            "alive": alive,
            "updated": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(newest)),
            "hard_stop": [f"{t}: {s['last']}" for t, s in summary.items() if s["nonfinite"] and isinstance(s["last"], str)],
            "tags": summary,
            "launch_yaml": launch[:LAUNCH_MAX],
        })
    tmp = cache_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache))
    os.replace(tmp, cache_path)
    json.dump(out, sys.stdout, ensure_ascii=False)


if __name__ == "__main__":
    main()
