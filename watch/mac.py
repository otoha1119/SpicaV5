"""train-watch の Mac 側の道具（Claude Code の /watch-training スキルから使う。stdlib のみ）。

    python3 watch/mac.py setup                    # data worktree の確認と Discord のチャンネル作成（初回だけ）
    python3 watch/mac.py wait --timeout 1800      # watch-data に更新が来るまで待つ（run_in_background で使う）
    python3 watch/mac.py sync                     # pull → 新しい応答・status の要約、未投稿の応答を Discord に中継
    python3 watch/mac.py status                   # いま手元にある status の要約（pull しない）
    python3 watch/mac.py request --machine PC1 --kind resume --summary "..." --command "bash start.sh PC1 resume <run> --lr 1e-4" ...
    python3 watch/mac.py log "..." [--file note.md] / call "..."
    python3 watch/mac.py install-skill            # watch/skill を ~/.claude/skills/watch-training に symlink

設定は watch/mac_config.json（例: mac_config.example.json）。
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (HERE, SHARED_CONFIG, DataRepo, WatchError, discord_api, discord_post, discord_token, git,  # noqa: E402
                    load_json, need, now_iso, now_stamp, one_line, read_doc, shared_config, write_doc)

MAC_CONFIG = HERE / "mac_config.json"
KINDS = ["stop", "run", "resume", "free"]
SKILL_LINK = Path.home() / ".claude" / "skills" / "watch-training"
SUMMARY_TAG_HINTS = ("loss", "val/", "diag/", "sigma", "fid", "D_", "G_")


def repo():
    cfg = load_json(MAC_CONFIG)
    return DataRepo(need(cfg, "data_dir", str(MAC_CONFIG)), need(shared_config(), "data_branch", str(SHARED_CONFIG)))


# ---------------------------------------------------------------- setup

def setup():
    r = repo()
    print(f"[watch] data リポジトリ OK: {r.path}（{r.branch}、remote {'あり' if r.remote_sha() else 'まだ push されていない'}）")
    cfg = shared_config()
    d = need(cfg, "discord", "watch/config.json")
    tok = discord_token()
    guild = discord_api("GET", f"/guilds/{need(d, 'guild_id', 'discord')}", tok)
    chans = discord_api("GET", f"/guilds/{d['guild_id']}/channels", tok)
    cat = next((c for c in chans if c["type"] == 4 and c["name"] == d["category"]), None)
    if cat is None:
        cat = discord_api("POST", f"/guilds/{d['guild_id']}/channels", tok, {"name": d["category"], "type": 4})
        print(f"[watch] カテゴリ {d['category']} を作りました")
    for kind in ("log", "call"):
        name = d[f"{kind}_channel_name"]
        ch = next((c for c in chans if c["type"] == 0 and c["name"] == name and c.get("parent_id") == cat["id"]), None)
        if ch is None:
            topic = "train-watch: Claude と Codex のやり取り" if kind == "log" else "train-watch: 研究者の判断が必要なときの呼び出し"
            ch = discord_api("POST", f"/guilds/{d['guild_id']}/channels", tok,
                             {"name": name, "type": 0, "parent_id": cat["id"], "topic": topic})
            print(f"[watch] チャンネル #{name} を作りました")
        d[f"{kind}_channel_id"] = ch["id"]
    d["call_user_id"] = d.get("call_user_id") or guild["owner_id"]
    SHARED_CONFIG.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[watch] {SHARED_CONFIG} にチャンネル id を書きました（commit して学習 PC と共有する）")


# ---------------------------------------------------------------- wait / sync / status

def wait(timeout, poll):
    r = repo()
    base = r.head()
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            sha = r.remote_sha()
        except WatchError as e:
            print(f"[watch] ls-remote 失敗（続行）: {one_line(str(e))}", flush=True)
            sha = base
        if sha and sha != base:
            print(f"[watch] 更新あり（{(base or '')[:8]} → {sha[:8]}）。sync してください")
            return
        time.sleep(poll)
    print(f"[watch] {timeout} 秒、更新なし（定期確認として status を見てください）")


def fmt_tag(name, t):
    return f"{name}={t['last']}（avg5 {t['mean_last5']}, 傾き/1k {t['slope_per_1k_steps']}, step {t['last_step']}）"


def run_summary(f):
    s = json.loads(f.read_text(encoding="utf-8"))
    tags = s.get("tags", {})
    pick = [k for k in tags if any(h in k for h in SUMMARY_TAG_HINTS)][:12]
    lines = [f"- {f.parent.name}/{f.stem}（run_dir {s['run_dir']}）: {'学習中' if s['alive'] else '停止'}、更新 {s['updated']}、tag {len(tags)} 個"
             + (f"、⚠ NaN/Inf: {', '.join(s['hard_stop'])}" if s.get("hard_stop") else "")]
    lines += [f"    {fmt_tag(k, tags[k])}" for k in pick]
    return "\n".join(lines)


def status(r=None):
    r = r or repo()
    files = sorted((r.path / "status").glob("*/*.json"))
    beats = [f for f in files if f.name == "_agent.json"]
    for b in beats:
        j = json.loads(b.read_text(encoding="utf-8"))
        print(f"[{b.parent.name}] エージェント最終応答 {j['at']}、コンテナ {j['container']}、active {j['active_runs']}")
    for f in files:
        if f.name != "_agent.json":
            print(run_summary(f))
    print(f"（詳細: {r.path / 'status'}/<machine>/<run>.json の tags.*.series）")


def sync():
    r = repo()
    r.pull()
    st = r.read_state("mac")
    last = st.get("last_sha")
    head = r.head()
    if head is None:
        print("[watch] data リポジトリはまだ空です")
        return
    if last == head:
        print("[watch] 新しい更新なし")
    changed = git(r.path, "diff", "--name-only", f"{last}..{head}").stdout.split() if last else \
        git(r.path, "ls-files").stdout.split()
    relayed = set(st.get("relayed", []))
    for name in sorted(n for n in changed if n.startswith("responses/") and n.endswith(".md")):
        f = r.path / name
        if not f.is_file():
            continue
        fm, body = read_doc(f)
        print(f"\n## 応答 {fm['id']}（{fm['machine']}、{fm['kind']}、{fm['result']}）\n{fm['summary']}\n  → {f}")
        head_line = f"[{fm['machine']}] {fm['summary']}"
        if fm.get("discord_log_posted") != "true" and f"{fm['id']}:log" not in relayed:   # PC が投稿できなかった分だけ中継
            discord_post("log", head_line)
            relayed.add(f"{fm['id']}:log")
        if fm.get("discord_call_posted") != "true" and f"{fm['id']}:call" not in relayed:
            discord_post("call", head_line + "（判断が必要）", mention=True)
            relayed.add(f"{fm['id']}:call")
    stat = [n for n in changed if n.startswith("status/") and not n.endswith("_agent.json")]
    if stat:
        print("\n## 更新された status")
        for n in sorted(stat):
            if (r.path / n).is_file():
                print(run_summary(r.path / n))
    st.update({"last_sha": head, "relayed": sorted(relayed)[-500:], "synced_at": now_iso()})
    r.write_state("mac", st)


# ---------------------------------------------------------------- request / log / call

def request(a):
    if a.kind in ("run", "resume") and not a.command:
        raise WatchError(f"kind {a.kind} には --command が要ります")
    if a.kind == "stop" and not a.run:
        raise WatchError("kind stop には --run が要ります")
    if a.kind in ("run", "resume") and not (a.branch and a.commit):
        raise WatchError(f"kind {a.kind} には --branch と --commit が要ります（Codex が checkout して合わせる）")
    if a.size not in ("small", "large", "none"):
        raise WatchError("--size は small | large | none")
    r = repo()
    r.pull()
    rid = f"{now_stamp()}-{a.machine}-{a.kind}"
    body = Path(a.body_file).read_text(encoding="utf-8") if a.body_file else (a.body or "")
    fm = {"id": rid, "machine": a.machine, "kind": a.kind, "size": a.size, "summary": a.summary, "run": a.run or "",
          "stop_first": a.stop_first or "", "branch": a.branch or "", "commit": a.commit or "", "command": a.command or "",
          "at": now_iso()}
    files = []
    if a.note_file:
        note = r.path / "notes" / f"{rid}.md"
        note.parent.mkdir(parents=True, exist_ok=True)
        note.write_text(Path(a.note_file).read_text(encoding="utf-8"), encoding="utf-8")
        fm["note"] = f"notes/{rid}.md"
        files.append(note)
    write_doc(r.path / "requests" / f"{rid}.md", fm, f"# {a.summary}\n\n{body}")
    r.commit_push(f"watch: request {rid}")
    discord_post("log", f"[Mac→{a.machine}] {a.kind}: {a.summary}", files=files)
    print(f"[watch] 依頼 {rid} を push しました")


def install_skill():
    src = HERE / "skill"
    SKILL_LINK.parent.mkdir(parents=True, exist_ok=True)
    if SKILL_LINK.is_symlink() or SKILL_LINK.exists():
        if SKILL_LINK.is_symlink() and SKILL_LINK.resolve() == src.resolve():
            print(f"[watch] 既に入っています: {SKILL_LINK}")
            return
        raise WatchError(f"{SKILL_LINK} が既にあります（別物。確認してから消してください）")
    SKILL_LINK.symlink_to(src)
    print(f"[watch] {SKILL_LINK} → {src}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup")
    s = sub.add_parser("wait")
    s.add_argument("--timeout", type=int, required=True)
    s.add_argument("--poll", type=int, default=60)
    sub.add_parser("sync")
    sub.add_parser("status")
    s = sub.add_parser("request")
    s.add_argument("--machine", required=True)
    s.add_argument("--kind", required=True, choices=KINDS)
    s.add_argument("--summary", required=True, help="Discord に載せる一行")
    s.add_argument("--size", required=True, help="small | large | none（変更の大きさ）")
    s.add_argument("--run")
    s.add_argument("--stop_first", help="先に止める run")
    s.add_argument("--branch")
    s.add_argument("--commit")
    s.add_argument("--command", help="学習 PC で launch するコマンド（bash start*.sh ...）")
    s.add_argument("--body", help="依頼の本文（目的・根拠・起動後に見る点）")
    s.add_argument("--body_file")
    s.add_argument("--note_file", help="改良内容の md（notes/ に保存し Discord に添付）")
    for k in ("log", "call"):
        s = sub.add_parser(k)
        s.add_argument("text")
        s.add_argument("--file", action="append", default=[])
    sub.add_parser("install-skill")
    a = ap.parse_args()
    try:
        if a.cmd == "setup":
            setup()
        elif a.cmd == "wait":
            wait(a.timeout, a.poll)
        elif a.cmd == "sync":
            sync()
        elif a.cmd == "status":
            status()
        elif a.cmd == "request":
            request(a)
        elif a.cmd in ("log", "call"):
            discord_post(a.cmd, ("[Mac] " if a.cmd == "log" else "") + a.text, files=a.file, mention=(a.cmd == "call"))
        elif a.cmd == "install-skill":
            install_skill()
    except WatchError as e:
        sys.exit(f"[watch] {e}")


if __name__ == "__main__":
    main()
