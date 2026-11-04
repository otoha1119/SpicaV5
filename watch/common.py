"""train-watch の共通部品（Mac / 学習 PC の両方から使う。stdlib のみ）。

- data リポジトリ（orphan ブランチ watch-data の worktree）の pull / commit / push
- 依頼・応答ファイル（frontmatter 付き Markdown）の読み書き
- Discord REST への投稿（添付ファイル付き）

設定の既定値は持たない。必要なキーが無ければ WatchError で止める。
"""
import json
import mimetypes
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHARED_CONFIG = HERE / "config.json"
SECRETS_FILE = Path.home() / ".config" / "autostudy" / "secrets.env"   # auto-study と共用の bot トークン
DISCORD_API = "https://discord.com/api/v10"
DISCORD_MAX = 1900   # 本文の上限（Discord は 2000 文字。余裕を残して切る）


class WatchError(Exception):
    pass


# ---------------------------------------------------------------- 設定

def load_json(path):
    path = Path(path)
    if not path.is_file():
        raise WatchError(f"設定ファイルがありません: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise WatchError(f"設定ファイルの JSON が壊れています: {path}: {e}")


def need(cfg, key, where):
    if key not in cfg or cfg[key] in (None, ""):
        raise WatchError(f"{where} に {key} がありません")
    return cfg[key]


def shared_config():
    return load_json(SHARED_CONFIG)


def now_stamp():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def now_iso():
    return datetime.now().astimezone().isoformat(timespec="seconds")


# ---------------------------------------------------------------- git（data リポジトリ）

def git(repo, *args, check=True, timeout=120):
    p = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=timeout)
    if check and p.returncode != 0:
        raise WatchError(f"git {' '.join(args)} が失敗しました（{repo}）:\n{p.stderr.strip()}")
    return p


class DataRepo:
    """watch-data ブランチの worktree。書き手ごとにディレクトリを分けるので（status/ responses/ は PC、requests/ notes/ は Mac）衝突しない。"""

    def __init__(self, path, branch):
        self.path = Path(path)
        self.branch = branch
        if not (self.path / ".git").exists():
            raise WatchError(f"data リポジトリがありません（watch-data の clone / worktree を作ってください）: {self.path}")
        cur = git(self.path, "symbolic-ref", "--short", "HEAD").stdout.strip()
        if cur != branch:
            raise WatchError(f"data リポジトリのブランチが {cur} です（{branch} であること）: {self.path}")
        self.local = self.path / ".local"   # 端末ごとの状態（.gitignore 済み）
        self.local.mkdir(exist_ok=True)

    def remote_sha(self):
        out = git(self.path, "ls-remote", "origin", f"refs/heads/{self.branch}", timeout=60).stdout.split()
        return out[0] if out else None

    def head(self):
        p = git(self.path, "rev-parse", "HEAD", check=False)
        return p.stdout.strip() if p.returncode == 0 else None

    def pull(self):
        if self.remote_sha() is None:
            return   # まだ誰も push していない
        git(self.path, "pull", "--rebase", "--autostash", "-q", "origin", self.branch, timeout=180)

    def commit_push(self, message, tries=4):
        """変更があれば commit して push。push が競合したら pull --rebase して再試行。push したら True。"""
        git(self.path, "add", "-A")
        if git(self.path, "diff", "--cached", "--quiet", check=False).returncode == 0:
            return False
        git(self.path, "commit", "-q", "-m", message)
        last = ""
        for i in range(tries):
            p = git(self.path, "push", "-q", "-u", "origin", self.branch, check=False, timeout=180)
            if p.returncode == 0:
                return True
            last = p.stderr.strip()
            time.sleep(3 * (i + 1))
            self.pull()
        raise WatchError(f"push が {tries} 回失敗しました（commit は手元に残っています）:\n{last}")

    def read_state(self, name):
        f = self.local / f"{name}.json"
        return json.loads(f.read_text(encoding="utf-8")) if f.is_file() else {}

    def write_state(self, name, data):
        f = self.local / f"{name}.json"
        tmp = f.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, f)


# ---------------------------------------------------------------- frontmatter 付き Markdown

_FM = re.compile(r"\A---\n(.*?)\n---\n?(.*)\Z", re.S)


def read_doc(path):
    text = Path(path).read_text(encoding="utf-8")
    m = _FM.match(text)
    if not m:
        raise WatchError(f"frontmatter がありません: {path}")
    fm = {}
    for line in m.group(1).splitlines():
        if not line.strip():
            continue
        k, sep, v = line.partition(":")
        if not sep:
            raise WatchError(f"frontmatter の行が読めません: {path}: {line}")
        fm[k.strip()] = json.loads(v.strip()) if v.strip().startswith('"') else v.strip()
    return fm, m.group(2)


def write_doc(path, fm, body):
    lines = ["---"]
    for k, v in fm.items():
        v = "" if v is None else str(v)
        if "\n" in v or v != v.strip() or ":" in v or v.startswith('"'):
            v = json.dumps(v, ensure_ascii=False)   # 改行やコロンを含む値（command など）は JSON 文字列で
        lines.append(f"{k}: {v}")
    lines.append("---")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(lines) + "\n" + body.rstrip() + "\n", encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------- Discord

def discord_token(required=True):
    tok = os.environ.get("DISCORD_BOT_TOKEN")
    if not tok and SECRETS_FILE.is_file():
        for line in SECRETS_FILE.read_text(encoding="utf-8").splitlines():
            k, _, v = line.partition("=")
            if k.strip() == "DISCORD_BOT_TOKEN" and v.strip():
                tok = v.strip().strip('"').strip("'")
    if not tok and required:
        raise WatchError(f"Discord の bot トークンがありません（環境変数 DISCORD_BOT_TOKEN か {SECRETS_FILE}）")
    return tok


def discord_api(method, path, token, payload=None, files=None):
    """Discord REST。files = [(ファイル名, bytes)] があれば multipart。429 は retry_after だけ待って 1 回だけやり直す。"""
    url = DISCORD_API + path
    for attempt in range(2):
        headers = {"Authorization": f"Bot {token}", "User-Agent": "spica-watch (https://github.com/otoha1119, 1.0)"}
        if files:
            boundary = uuid.uuid4().hex
            parts = [f"--{boundary}\r\nContent-Disposition: form-data; name=\"payload_json\"\r\n"
                     f"Content-Type: application/json\r\n\r\n{json.dumps(payload or {})}\r\n".encode()]
            for i, (name, data) in enumerate(files):
                ctype = mimetypes.guess_type(name)[0] or "text/plain"
                parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"files[{i}]\"; filename=\"{name}\"\r\n"
                             f"Content-Type: {ctype}\r\n\r\n".encode() + data + b"\r\n")
            parts.append(f"--{boundary}--\r\n".encode())
            body = b"".join(parts)
            headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
        else:
            body = json.dumps(payload).encode() if payload is not None else None
            if body is not None:
                headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=body, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read()
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")
            if e.code == 429 and attempt == 0:
                time.sleep(float(json.loads(detail).get("retry_after", 2)) + 0.5)
                continue
            raise WatchError(f"Discord {method} {path} が {e.code} で失敗しました: {detail[:300]}")
    raise WatchError(f"Discord {method} {path} が rate limit で失敗しました")


def discord_post(kind, text, files=None, mention=False, token=None):
    """kind = 'log' | 'call'。call は呼び出し先のユーザーにメンションする。files = [Path]。"""
    cfg = need(shared_config(), "discord", "watch/config.json")
    channel = need(cfg, f"{kind}_channel_id", "watch/config.json の discord")
    token = token or discord_token()
    if mention:
        text = f"<@{need(cfg, 'call_user_id', 'watch/config.json の discord')}> " + text
    if len(text) > DISCORD_MAX:
        text = text[:DISCORD_MAX] + " …（以下略）"
    payload = {"content": text, "allowed_mentions": {"parse": ["users"]}}
    for f in files or []:   # 送ってよいのは Markdown とテキストだけ（医用画像とその派生画像を送らないため）
        if Path(f).suffix.lower() not in (".md", ".txt"):
            raise WatchError(f"Discord に添付できるのは .md / .txt だけです: {f}")
    att = [(Path(f).name, Path(f).read_bytes()) for f in (files or [])]
    return discord_api("POST", f"/channels/{channel}/messages", token, payload, att or None)


def one_line(text, limit=300):
    for line in (text or "").splitlines():
        if line.strip():
            s = line.strip()
            return s if len(s) <= limit else s[:limit] + "…"
    return ""
