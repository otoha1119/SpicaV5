"""train-watch の Mac 側の道具（Claude Code の /watch-training スキルから使う。stdlib のみ）。

    python3 watch/mac.py setup                    # data worktree の確認と Discord のチャンネル作成（初回だけ）
    python3 watch/mac.py wait --timeout 180       # watch-data に更新が来るまで待つ（run_in_background で使う。timeout は状況を見て毎回選ぶ）
    python3 watch/mac.py sync                     # pull → 新しい応答・status の要約、未投稿の応答を Discord に中継
    python3 watch/mac.py status                   # いま手元にある status の要約（pull しない）
    python3 watch/mac.py request --machine PC1 --kind resume --summary "..." --command "bash start.sh PC1 resume <run> --lr 1e-4" ...
    python3 watch/mac.py request --machine PC1 --kind job --summary "..." --script eval.py --command 'python _job/eval.py --out "$SPICA_JOB_OUT"' ...
    python3 watch/mac.py fetch-blobs <job id> --to <dir>   # job の _out/blobs/（重みなど）を取り出し、使い捨てブランチを消す
    python3 watch/mac.py log "..." [--file note.md] / call "..."
    python3 watch/mac.py setup-chat               # #watch-chat（研究者と会話するチャンネル）を作る（初回だけ）
    python3 watch/mac.py chat-inbox               # #watch-chat の研究者の未読メッセージを表示して既読にする
    python3 watch/mac.py chat "..." [--file x.md] # #watch-chat に返事する（.md / .txt だけ添付可）
    python3 watch/mac.py install-skill            # watch/skill を ~/.claude/skills/watch-training に symlink

設定は watch/mac_config.json（例: mac_config.example.json）。
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (DISCORD_MAX, HERE, SHARED_CONFIG, DataRepo, WatchError, discord_api, discord_post, discord_token,  # noqa: E402
                    git, load_json, need, now_iso, now_stamp, one_line, read_doc, shared_config, write_doc)

MAC_CONFIG = HERE / "mac_config.json"
KINDS = ["stop", "run", "resume", "free", "job"]
SKILL_LINK = Path.home() / ".claude" / "skills" / "watch-training"
SUMMARY_TAG_HINTS = ("loss", "val/", "diag/", "sigma", "fid", "D_", "G_")
CHAT_POLL_S = 10   # wait が #watch-chat を見る間隔（Discord への問い合わせは軽い。git の ls-remote は --poll 秒ごとのまま）


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
    tok = discord_token()
    chats = [chat_config()]
    try:   # #watch-ideas（議論用）は設定があれば一緒に見る
        chats.append(chat_config("ideas"))
    except WatchError:
        pass
    for chat in chats:
        try:   # 初回はここで既読位置を決める（過去のメッセージで起きない）。Discord が落ちていても git の監視は続ける
            chat_state(r, tok, chat)
        except (WatchError, OSError) as e:
            print(f"[watch] Discord の確認に失敗（git の監視は続行）: {one_line(str(e))}", flush=True)
    base = r.head()
    t0 = time.time()
    last_git = 0.0
    while time.time() - t0 < timeout:
        sha = base
        if time.time() - last_git >= poll:   # git（ls-remote）は poll 秒ごと、Discord は毎周（CHAT_POLL_S 秒ごと）
            last_git = time.time()
            try:
                sha = r.remote_sha()
            except WatchError as e:
                print(f"[watch] ls-remote 失敗（続行）: {one_line(str(e))}", flush=True)
        if sha and sha != base:
            # status/ だけの更新（数値の定期送信）では起きない。応答・自動停止・依頼など status/ 以外が来たら起きる。
            # 数値は timeout ごとの定期確認で見る（毎回起きると夜通しトークンを使い続けるため）
            git(r.path, "fetch", "-q", "origin", r.branch, timeout=180)
            paths = git(r.path, "diff", "--name-only", f"{base}..{sha}").stdout.split() if base else ["?"]
            if any(not x.startswith("status/") for x in paths):
                print(f"[watch] 更新あり（{(base or '')[:8]} → {sha[:8]}）。sync してください")
                return
        # git とは別に、#watch-chat / #watch-ideas に研究者の新しいメッセージが来たら起きる（status/ だけの git 更新で起きない規則はそのまま）
        for chat in chats:
            try:
                msgs, _ = chat_fetch(r, tok, chat)
            except (WatchError, OSError) as e:   # URLError・timeout は OSError。Discord の一時的な障害で監視を止めない
                print(f"[watch] Discord の確認に失敗（続行）: {one_line(str(e))}", flush=True)
                msgs = []
            if msgs:
                chat_check_content(msgs)
                print(f"[watch] Discord {chat['name']} に新しいメッセージ（{len(msgs)} 件）。{chat['kind']}-inbox で読んでください")
                return
        time.sleep(CHAT_POLL_S)
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
        if f.name == "_jobs.json":
            for rid, j in json.loads(f.read_text(encoding="utf-8"))["jobs"].items():
                print(f"[{f.parent.name}] job {rid}（{j['started']} 開始）: {j['summary']}\n    " + "\n    ".join(j["log_tail"][-5:]))
        elif f.name != "_agent.json":
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
        res = r.path / "results" / fm["id"]
        if res.is_dir():
            print(f"  結果: {', '.join(x.relative_to(res).as_posix() for x in sorted(res.rglob('*')) if x.is_file())}（{res}）")
        head_line = f"[{fm['machine']}] {fm['summary']}"
        if fm.get("discord_log_posted") != "true" and f"{fm['id']}:log" not in relayed:   # PC が投稿できなかった分だけ中継
            discord_post("log", head_line)
            relayed.add(f"{fm['id']}:log")
        if fm.get("discord_call_posted") != "true" and f"{fm['id']}:call" not in relayed:
            discord_post("call", head_line + "（判断が必要）", mention=True)
            relayed.add(f"{fm['id']}:call")
    stat = [n for n in changed if n.startswith("status/") and not n.endswith(("_agent.json", "_jobs.json"))]
    if stat:
        print("\n## 更新された status")
        for n in sorted(stat):
            if (r.path / n).is_file():
                print(run_summary(r.path / n))
    st.update({"last_sha": head, "relayed": sorted(relayed)[-500:], "synced_at": now_iso()})
    r.write_state("mac", st)


# ---------------------------------------------------------------- chat（#watch-chat。研究者が外出先から Discord でホストと会話する）

CHAT_KINDS = {"chat": "#watch-chat", "ideas": "#watch-ideas"}   # 研究者と話すチャンネル。ideas = 手法・方針の議論用（2026-11-09）


def chat_config(kind="chat"):
    d = need(shared_config(), "discord", str(SHARED_CONFIG))
    return {"kind": kind, "name": CHAT_KINDS[kind],
            "channel": need(d, f"{kind}_channel_id", f"{SHARED_CONFIG} の discord（mac.py setup-chat / setup-ideas で作る）"),
            "user": need(d, "call_user_id", f"{SHARED_CONFIG} の discord")}


def chat_state(r, tok, chat):
    """既読位置（最後に見たメッセージ id）。無ければ今の最新を既読にする（過去の会話を新着として扱わない）。"""
    st = r.read_state(chat.get("kind", "chat"))
    if st.get("channel") != chat["channel"]:
        latest = discord_api("GET", f"/channels/{chat['channel']}/messages?limit=1", tok)
        st = {"channel": chat["channel"], "last_seen": latest[0]["id"] if latest else "0", "at": now_iso()}
        r.write_state(chat.get("kind", "chat"), st)
        print(f"[watch] {chat.get('name', '#watch-chat')} の既読位置を初期化しました（{st['last_seen']}）", flush=True)
    return st


def chat_fetch(r, tok, chat):
    """既読位置より後のメッセージを古い順に全部取る。返すのは（研究者本人が書いた bot 以外のもの, 見た中で最大の id）。"""
    after = chat_state(r, tok, chat)["last_seen"]
    seen, top = [], after
    while True:
        page = discord_api("GET", f"/channels/{chat['channel']}/messages?after={top}&limit=100", tok)
        if not page:
            break
        page.sort(key=lambda m: int(m["id"]))
        seen += page
        top = page[-1]["id"]
        if len(page) < 100:
            break
    mine = [m for m in seen if m["author"]["id"] == chat["user"] and not m["author"].get("bot")]
    return mine, top


def chat_check_content(msgs):
    # Message Content Intent が無効だと、bot へのメンションが無いメッセージの本文・添付が空で返る。黙って読み飛ばさない
    if any(not m.get("content") and not m.get("attachments") and not m.get("sticker_items") for m in msgs):
        raise WatchError("#watch-chat のメッセージの本文が空で届きました。Discord Developer Portal → bot（auto-study）→ Bot → "
                         "Privileged Gateway Intents の Message Content Intent を有効にしてください")


def chat_inbox(kind="chat"):
    r = repo()
    tok = discord_token()
    chat = chat_config(kind)
    msgs, top = chat_fetch(r, tok, chat)
    chat_check_content(msgs)   # 本文が読めないうちは既読にしない
    if not msgs:
        print(f"[watch] {chat['name']} に未読はありません")
    for m in msgs:
        at = m["timestamp"][:19].replace("T", " ")
        files = [a["filename"] for a in m.get("attachments", [])]
        print(f"\n## {at} UTC（{m['id']}）\n{m.get('content', '')}" + (f"\n  添付（開いていない）: {', '.join(files)}" if files else ""))
    st = r.read_state(kind)
    st.update({"last_seen": top, "at": now_iso()})
    r.write_state(kind, st)


def chat_send(text, files, kind="chat"):
    """#watch-chat（kind=ideas なら #watch-ideas）に返事する。長い本文は行で分けて最大 3 通、それを超えるなら全文を md にして添付する。"""
    chunks, cur = [], ""
    for line in text.splitlines(keepends=True):
        while len(line) > DISCORD_MAX:   # 1 行が長すぎるときはそのまま切る
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(line[:DISCORD_MAX])
            line = line[DISCORD_MAX:]
        if len(cur) + len(line) > DISCORD_MAX:
            chunks.append(cur)
            cur = ""
        cur += line
    if cur.strip():
        chunks.append(cur)
    if not chunks and not files:
        raise WatchError("chat: 本文も添付もありません")
    files = list(files)
    if len(chunks) > 3:
        full = repo().local / "chat" / f"{now_stamp()}_reply.md"
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(text, encoding="utf-8")
        chunks = [one_line(text, 200) + "\n（長いので全文を md で添付します）"]
        files.append(str(full))
    for i, c in enumerate(chunks or [""]):
        discord_post(kind, c, files=files if i == len(chunks or [""]) - 1 else None)


def setup_ideas():
    """#watch-ideas（手法・方針の議論用）を作って config に入れる。setup_chat と同じ作り。"""
    cfg = shared_config()
    d = need(cfg, "discord", str(SHARED_CONFIG))
    if d.get("ideas_channel_id"):
        print(f"[watch] #watch-ideas は設定済みです（{d['ideas_channel_id']}）")
        return
    tok = discord_token()
    name = "watch-ideas"
    chans = discord_api("GET", f"/guilds/{need(d, 'guild_id', 'discord')}/channels", tok)
    cat = next((c for c in chans if c["type"] == 4 and c["name"] == d.get("category")), None)
    ch = next((c for c in chans if c["type"] == 0 and c["name"] == name), None)
    if ch is None:
        ch = discord_api("POST", f"/guilds/{d['guild_id']}/channels", tok,
                         {"name": name, "type": 0, "parent_id": cat["id"] if cat else None,
                          "topic": "手法・方針の議論（Mac の Claude と研究者）。監視の報告は #watch-log / #watch-call、運用の会話は #watch-chat"})
    d["ideas_channel_name"], d["ideas_channel_id"] = name, ch["id"]
    SHARED_CONFIG.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[watch] #watch-ideas を用意しました（{ch['id']}）。watch/config.json を commit してください")


def setup_chat():
    cfg = shared_config()
    d = need(cfg, "discord", str(SHARED_CONFIG))
    if d.get("chat_channel_id"):
        print(f"[watch] #watch-chat は設定済みです（{d['chat_channel_id']}）")
        return
    tok = discord_token()
    name = "watch-chat"
    chans = discord_api("GET", f"/guilds/{need(d, 'guild_id', 'discord')}/channels", tok)
    cat = next((c for c in chans if c["type"] == 4 and c["name"] == need(d, "category", "discord")), None)
    if cat is None:
        raise WatchError(f"カテゴリ {d['category']} がありません（先に mac.py setup）")
    if any(c["type"] == 0 and c["name"] == name and c.get("parent_id") == cat["id"] for c in chans):
        raise WatchError(f"#{name} が既にありますが config.json に id がありません。自分で作っていないチャンネルは使わないので、確認して消してから実行してください")
    ch = discord_api("POST", f"/guilds/{d['guild_id']}/channels", tok,
                     {"name": name, "type": 0, "parent_id": cat["id"], "topic": "train-watch: 研究者とホストの Claude の会話"})
    d["chat_channel_name"], d["chat_channel_id"] = name, ch["id"]
    SHARED_CONFIG.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[watch] チャンネル #{name} を作り、{SHARED_CONFIG} に id を書きました（commit して共有する）")


# ---------------------------------------------------------------- request / log / call

def request(a):
    if a.kind in ("run", "resume") and not a.command:
        raise WatchError(f"kind {a.kind} には --command が要ります")
    if a.kind == "job" and not a.command:
        raise WatchError("kind job には --command が要ります（コンテナ内で job の worktree を cwd にして bash -c で走る）")
    if a.script and a.kind != "job":
        raise WatchError("--script は kind job だけです")
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
    if a.script:   # job に添付するスクリプト。PC で job の worktree の _job/ に置かれる
        d = r.path / "jobs" / rid
        d.mkdir(parents=True, exist_ok=True)
        for sp in a.script:
            sp = Path(sp)
            if not sp.is_file():
                raise WatchError(f"--script のファイルがありません: {sp}")
            (d / sp.name).write_bytes(sp.read_bytes())
        fm["scripts"] = ", ".join(Path(x).name for x in a.script)
    write_doc(r.path / "requests" / f"{rid}.md", fm, f"# {a.summary}\n\n{body}")
    r.commit_push(f"watch: request {rid}")
    discord_post("log", f"[Mac→{a.machine}] {a.kind}: {a.summary}", files=files)
    print(f"[watch] 依頼 {rid} を push しました")


def fetch_blobs(rid, to):
    r = repo()
    ref = f"watch-blob/{rid}"
    p = git(r.path, "fetch", "-q", "origin", f"refs/heads/{ref}", check=False, timeout=1800)
    if p.returncode != 0:
        raise WatchError(f"{ref} がありません（job の応答の「使い捨てブランチ」を確認）:\n{p.stderr.strip()}")
    sha = git(r.path, "rev-parse", "FETCH_HEAD").stdout.strip()
    out = Path(to)
    out.mkdir(parents=True, exist_ok=True)
    for ent in git(r.path, "ls-tree", "-z", sha).stdout.split("\0"):
        if ent:
            obj, name = ent.split()[2], ent.split("\t", 1)[1]
            data = subprocess.run(["git", "-C", str(r.path), "cat-file", "blob", obj], capture_output=True, check=True).stdout
            name = name.strip()   # 古い pc_agent は Windows の改行変換で名前の末尾に \r が付いていた
            (out / name).write_bytes(data)
            print(f"[watch] {out / name}（{len(data) / 1e6:.1f} MB）")
    git(r.path, "push", "-q", "origin", "--delete", ref, check=False, timeout=180)


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
    s.add_argument("--command", help="run / resume: 学習 PC で launch するコマンド（bash start*.sh ...）。job: コンテナ内で走るコマンド")
    s.add_argument("--script", action="append", default=[], help="job に添付するファイル（PC で _job/ に置かれる。複数可）")
    s.add_argument("--body", help="依頼の本文（目的・根拠・起動後に見る点）")
    s.add_argument("--body_file")
    s.add_argument("--note_file", help="改良内容の md（notes/ に保存し Discord に添付）")
    for k in ("log", "call"):
        s = sub.add_parser(k)
        s.add_argument("text")
        s.add_argument("--file", action="append", default=[])
    sub.add_parser("setup-chat")
    sub.add_parser("chat-inbox")
    sub.add_parser("ideas-inbox")
    sub.add_parser("setup-ideas")
    s = sub.add_parser("ideas")
    s.add_argument("text", nargs="?", default="")
    s.add_argument("--file", action="append", default=[])
    s = sub.add_parser("chat")
    s.add_argument("text")
    s.add_argument("--file", action="append", default=[])
    s = sub.add_parser("fetch-blobs")
    s.add_argument("rid")
    s.add_argument("--to", required=True)
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
        elif a.cmd == "setup-chat":
            setup_chat()
        elif a.cmd == "ideas-inbox":
            chat_inbox("ideas")
        elif a.cmd == "setup-ideas":
            setup_ideas()
        elif a.cmd == "ideas":
            chat_send(a.text, a.file, "ideas")
        elif a.cmd == "chat-inbox":
            chat_inbox()
        elif a.cmd == "chat":
            chat_send(a.text, a.file)
        elif a.cmd == "fetch-blobs":
            fetch_blobs(a.rid, a.to)
        elif a.cmd == "install-skill":
            install_skill()
    except WatchError as e:
        sys.exit(f"[watch] {e}")


if __name__ == "__main__":
    main()
