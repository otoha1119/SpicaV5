"""train-watch の学習 PC 側エージェント（ホストで動かす。python3 + stdlib）。

    python watch/pc_agent.py loop                 # 常駐: 数値の書き出し・NaN の即停止・依頼の処理
    python watch/pc_agent.py running              # コンテナ内で学習中のプロセスを表示
    python watch/pc_agent.py stop <run>           # その run の学習プロセスを止める
    python watch/pc_agent.py launch -- bash start.sh PC1 resume <run> --lr 1e-4   # 学習を裏で起動（Codex はこれを使う）
    python watch/pc_agent.py export               # 書き出しを 1 回だけ（確認用。push しない）

設定は watch/pc_config.json（例: pc_config.example.json）。既定値は持たない。
"""
import argparse
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (HERE, DataRepo, WatchError, discord_post, discord_token, load_json, need, now_iso,  # noqa: E402
                    now_stamp, one_line, read_doc, shared_config, write_doc)

PC_CONFIG = HERE / "pc_config.json"
REQUEST_POLL_S = 60   # 依頼の確認間隔（git の pull だけなので軽い）。数値の書き出しは pc_config の interval_s ごと
KEYS = ["machine", "repo_dir", "data_dir", "container", "interval_s", "active_min", "codex_cmd", "codex_timeout_s",
        "discord", "launch_shell", "git_bash"]
RESULT_RE = re.compile(r"^RESULT:\s*(ok|failed|needs_user)\s*$", re.M)

# コンテナ内で学習プロセスを列挙する（tensorboard・推論を除き、train*.py を動かしている python）。
# 引数（run 名・run の親ディレクトリ）があれば、run 名が語として現れ、かつ親ディレクトリのパスも含むものだけ
# （同じ分に起動した別 Stage の同名 run を巻き込まないため。stage1 は --name と --checkpoints_dir、stage2 / ablation は --run_dir に出る）
PS_SNIPPET = r"""
import os, re, sys
name, parent = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else (None, None)
pat = re.compile(r'(?<![0-9A-Za-z_.\-])' + re.escape(name) + r'(?![0-9A-Za-z_.\-])') if name else None
for p in os.listdir('/proc'):
    if not p.isdigit() or int(p) == os.getpid():
        continue
    try:
        cmd = open('/proc/%s/cmdline' % p, 'rb').read().replace(b'\0', b' ').decode('utf-8', 'replace').strip()
    except OSError:
        continue
    if 'python' not in cmd or 'tensorboard' in cmd or 'pc_agent' in cmd or not re.search(r'train[0-9A-Za-z_]*\.py', cmd):
        continue
    if pat is None or (pat.search(cmd) and parent in cmd):
        print(p, cmd)
"""

# コンテナ内でシグナルを送る（slim イメージには kill / ps の実行ファイルが無いので python で送る）
KILL_SNIPPET = r"""
import os, signal, sys
sig = getattr(signal, sys.argv[1])
for p in sys.argv[2:]:
    try:
        os.kill(int(p), sig)
    except ProcessLookupError:
        pass
"""


def load_cfg():
    cfg = load_json(PC_CONFIG)
    for k in KEYS:
        if k not in cfg:
            raise WatchError(f"{PC_CONFIG} に {k} がありません（pc_config.example.json を参照）")
    return cfg


def docker_py(cfg, snippet, *args, timeout=120, stdin_text=None):
    """コンテナ内の python に snippet を流し込んで実行する。"""
    p = subprocess.run(["docker", "exec", "-i", cfg["container"], "python", "-", *args],
                       input=stdin_text if stdin_text is not None else snippet, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=timeout)
    if p.returncode != 0:
        raise WatchError(f"コンテナ内の python が失敗しました（{cfg['container']}）:\n{p.stderr.strip()[-2000:]}")
    return p.stdout


def container_running(cfg):
    p = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", cfg["container"]], capture_output=True, text=True)
    return p.returncode == 0 and p.stdout.strip() == "true"


# ---------------------------------------------------------------- 書き出し

def export(cfg):
    if not container_running(cfg):
        return {"machine": cfg["machine"], "runs": [], "container": "stopped"}
    src = (HERE / "exporter.py").read_text(encoding="utf-8")
    args = ["--machine", cfg["machine"], "--machines_yaml", "/workspace/configs/machines.yaml",
            "--active_min", str(cfg["active_min"]), "--cache", "/tmp/spica_watch_exporter_cache.json"]
    for r in cfg.get("extra_roots", []):
        args += ["--extra_root", r]
    out = docker_py(cfg, None, *args, timeout=600, stdin_text=src)
    return json.loads(out)


def status_label(run):
    return f"{run['root_key'].replace('_checkpoints_dir', '').replace('checkpoints_dir', 'stage1')}__{run['run'].replace('/', '__')}"


def write_status(repo, cfg, data):
    """run ごとに status/<machine>/<label>.json。中身が変わったときだけ書く。書いた run のラベルを返す。"""
    d = repo.path / "status" / cfg["machine"]
    d.mkdir(parents=True, exist_ok=True)
    changed = []
    for run in data["runs"]:
        label = status_label(run)
        f = d / f"{label}.json"
        old = json.loads(f.read_text(encoding="utf-8")) if f.is_file() else None
        if old is not None:
            old.pop("exported_at", None)
        if old == run:
            continue
        f.write_text(json.dumps({**run, "exported_at": now_iso()}, ensure_ascii=False, indent=1), encoding="utf-8")
        changed.append(label)
    seen = {status_label(r) for r in data["runs"]}
    for f in d.glob("*.json"):   # exporter に出なくなった run（コンテナ停止・30 分以上更新なし）は停止扱いにする
        if f.name == "_agent.json" or f.stem in seen:
            continue
        old = json.loads(f.read_text(encoding="utf-8"))
        if old.get("alive"):
            old.update({"alive": False, "missing_since": now_iso()})
            f.write_text(json.dumps(old, ensure_ascii=False, indent=1), encoding="utf-8")
            changed.append(f.stem)
    beat = d / "_agent.json"   # PC が生きている印。30 分ごとに更新（毎回だと commit が増えるため）
    old = json.loads(beat.read_text(encoding="utf-8")) if beat.is_file() else {}
    if changed or time.time() - old.get("epoch", 0) > 1800:
        beat.write_text(json.dumps({"epoch": int(time.time()), "at": now_iso(), "container": data.get("container", "running"),
                                    "active_runs": [status_label(r) for r in data["runs"]]}, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    return changed


# ---------------------------------------------------------------- 停止・起動

def running(cfg, run_dir=None):
    if not container_running(cfg):
        return []
    args = [PurePosixPath(run_dir).name, str(PurePosixPath(run_dir).parent)] if run_dir else []
    out = docker_py(cfg, PS_SNIPPET, *args)
    return [line.split(" ", 1) for line in out.splitlines() if line.strip()]


def resolve_run(repo_path, cfg, run):
    """stop の対象をコンテナ内の run ディレクトリに解決する。絶対パスならそのまま、run 名なら status から一意に探す。"""
    if run.startswith("/"):
        return run
    hits = []
    for f in (Path(repo_path) / "status" / cfg["machine"]).glob("*.json"):
        if f.name == "_agent.json":
            continue
        s = json.loads(f.read_text(encoding="utf-8"))
        if run in (s.get("run"), PurePosixPath(s.get("run_dir", "")).name, f.stem):
            hits.append(s["run_dir"])
    if len(set(hits)) != 1:
        raise WatchError(f"run {run} を一意に決められません（候補 {sorted(set(hits))}）。status の run_dir（コンテナ内の絶対パス）で指定してください")
    return hits[0]


def stop(cfg, run_dir):
    """その run の学習プロセスを SIGTERM → 30 秒待って残れば SIGKILL。止めたプロセス数を返す。"""
    if not run_dir.startswith("/") or not re.fullmatch(r"[0-9A-Za-z_.\-]+", PurePosixPath(run_dir).name):
        raise WatchError(f"run ディレクトリが不正です: {run_dir}")
    procs = running(cfg, run_dir)
    if not procs:
        return 0
    pids = [p for p, _ in procs]
    docker_py(cfg, KILL_SNIPPET, "SIGTERM", *pids)
    for _ in range(30):
        time.sleep(1)
        if not running(cfg, run_dir):
            return len(pids)
    left = [p for p, _ in running(cfg, run_dir)]
    docker_py(cfg, KILL_SNIPPET, "SIGKILL", *left)
    time.sleep(3)
    if running(cfg, run_dir):
        raise WatchError(f"{run_dir} の学習プロセスが止まりません: {running(cfg, run_dir)}")
    return len(pids)


def launch(cfg, command):
    """学習を裏で起動する。Windows は新しいウィンドウ（start.sh が端末を要るため）、それ以外は nohup。
    Windows のシェルは pc_config の launch_shell: wsl（研究者がふだん VS Code のターミナルで打つ `bash` = WSL と同じ）| git_bash。"""
    if not command:
        raise WatchError("launch に起動コマンドがありません（launch -- bash start.sh ...）")
    if len(command) < 2 or command[0] != "bash" or not Path(command[1]).name.startswith("start"):
        raise WatchError(f"launch できるのは bash start*.sh だけです: {shlex.join(command)}")
    repo = Path(cfg["repo_dir"])
    line = shlex.join(command)
    if platform.system() == "Windows":
        shell = need(cfg, "launch_shell", str(PC_CONFIG))
        tail = f"{line}; echo; echo '[watch] 終了コード' $?; exec bash"
        if shell == "wsl":
            wsl = shutil.which("wsl")
            if wsl is None:
                raise WatchError("wsl.exe が PATH にありません（launch_shell: wsl）")
            subprocess.Popen([wsl, "--cd", str(repo), "--", "bash", "-c", tail], creationflags=subprocess.CREATE_NEW_CONSOLE)
        elif shell == "git_bash":
            gb = Path(need(cfg, "git_bash", str(PC_CONFIG)))
            if not gb.is_file():
                raise WatchError(f"git-bash.exe がありません: {gb}")
            subprocess.Popen([str(gb), "-c", f"cd {shlex.quote(repo.as_posix())} && {tail}"], cwd=str(repo),
                             creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        else:
            raise WatchError(f"launch_shell は wsl | git_bash です: {shell!r}（{PC_CONFIG}）")
        return f"新しいウィンドウ（{shell}）で起動: {line}"
    logdir = Path(cfg["data_dir"]) / ".local" / "launch"
    logdir.mkdir(parents=True, exist_ok=True)
    log = logdir / f"{now_stamp()}.log"
    with open(log, "ab") as f:
        subprocess.Popen(["nohup", *command], cwd=str(repo), stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         start_new_session=True)
    return f"裏で起動: {line}（ログ {log}）"


# ---------------------------------------------------------------- 依頼の処理

def post(cfg, kind, text, files=None):
    """Discord に投稿。pc_config の discord が false なら投稿せず False（Mac が中継する）。"""
    if not cfg["discord"]:
        return False
    try:
        discord_post(kind, text, files=files, mention=(kind == "call"))
        return True
    except WatchError as e:
        print(f"[watch] Discord 投稿に失敗（Mac 側で中継される）: {e}", flush=True)
        return False


def respond(repo, cfg, rid, fm, body, result, summary):
    """応答を書く。Discord の #log / #call（needs_user のとき）に投稿できたかを別々に記録し、できなかった分は Mac が中継する。"""
    head = f"[{cfg['machine']}] {summary}"
    log_ok = post(cfg, "log", head)
    call_ok = post(cfg, "call", head + "（判断が必要）") if result == "needs_user" else True
    write_doc(repo.path / "responses" / f"{rid}.md",
              {"id": rid, "machine": cfg["machine"], "kind": fm.get("kind", ""), "run": fm.get("run", ""), "result": result,
               "summary": summary, "at": now_iso(), "discord_log_posted": "true" if log_ok else "false",
               "discord_call_posted": "true" if call_ok else "false"}, body)


def codex(cfg, rid, req_text):
    agent = f"{shlex.quote(sys.executable)} {shlex.quote(str(HERE / 'pc_agent.py'))}"
    prompt = (HERE / "codex_prompt.md").read_text(encoding="utf-8").replace("<pc_agent>", agent).replace("<学習リポジトリ>", cfg["repo_dir"])
    ctx = (f"\n\n# この PC の情報\n- machine: {cfg['machine']}\n- 学習リポジトリ: {cfg['repo_dir']}\n"
           f"- pc_agent: {sys.executable} {HERE / 'pc_agent.py'}\n- コンテナ: {cfg['container']}\n\n# 依頼（requests/{rid}.md）\n\n{req_text}\n")
    out = Path(cfg["data_dir"]) / ".local" / "codex" / f"{rid}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [c.replace("{repo_dir}", cfg["repo_dir"]).replace("{out}", str(out)) for c in cfg["codex_cmd"]]
    exe = shutil.which(cmd[0])  # Windows の npm は codex.cmd を入れる。subprocess は裸の名前から .cmd を探さないので PATH から解決する
    if exe is None:
        raise WatchError(f"{cmd[0]} が PATH にありません（pc_config.json の codex_cmd を確認）")
    cmd[0] = exe
    try:
        p = subprocess.run(cmd, input=prompt + ctx, capture_output=True, text=True, encoding="utf-8", errors="replace",
                           timeout=cfg["codex_timeout_s"])
    except subprocess.TimeoutExpired:
        return "failed", f"Codex が {cfg['codex_timeout_s']} 秒で終わりませんでした（{rid}）", ""
    msg = out.read_text(encoding="utf-8") if out.is_file() else ""
    if p.returncode != 0 or not msg.strip():
        return "failed", f"Codex が失敗しました（終了コード {p.returncode}）", (msg + "\n\n```\n" + p.stderr[-3000:] + "\n```")
    m = RESULT_RE.search(msg)
    return (m.group(1) if m else "failed"), one_line(msg) or "（要約なし）", msg


def handle_requests(repo, cfg):
    started = repo.read_state("started")
    for f in sorted((repo.path / "requests").glob("*.md")):
        fm, body = read_doc(f)
        rid = fm.get("id", f.stem)
        if fm.get("machine") != cfg["machine"] or (repo.path / "responses" / f"{rid}.md").exists():
            continue
        if rid in started:   # 処理中に落ちた。やり直さず失敗として返す（二重起動を避ける）
            respond(repo, cfg, rid, fm, "エージェントが処理中に再起動しました。状態を確認してから依頼し直してください。", "failed",
                    f"{rid} は処理中にエージェントが再起動したため中断扱い")
            repo.commit_push(f"watch: {cfg['machine']} response {rid}")
            continue
        started[rid] = now_iso()
        repo.write_state("started", started)
        print(f"[watch] 依頼 {rid}（{fm.get('kind')}）を処理します", flush=True)
        try:
            if fm.get("kind") == "stop":
                n = stop(cfg, resolve_run(repo.path, cfg, need(fm, "run", f.name)))
                result, summary, detail = "ok", (f"{fm['run']} を止めました（{n} プロセス）" if n else f"{fm['run']} は動いていませんでした"), body
            else:
                result, summary, detail = codex(cfg, rid, f.read_text(encoding="utf-8"))
        except Exception as e:   # 依頼 1 件の失敗でループを止めない
            result, summary, detail = "failed", f"{rid} の処理でエラー: {one_line(str(e))}", traceback.format_exc()
        respond(repo, cfg, rid, fm, detail, result, summary)
        repo.commit_push(f"watch: {cfg['machine']} response {rid}")


def hard_stops(repo, cfg, data):
    """最新値が NaN / Inf の run は即停止（判断は要らない。直して回し直すのは Mac の Claude）。"""
    done = repo.read_state("hard_stopped")
    for run in data["runs"]:
        if not run["alive"] or not run["hard_stop"]:
            continue
        label = status_label(run)
        step = max((t["last_step"] for t in run["tags"].values()), default=0)
        # 同じ NaN を二度扱わない印 = NaN になった tag の最終値の書き込み時刻。resume して同じ step でまた NaN になれば時刻が違うので止める
        # （resume 直後で新しい値がまだ無いときは前回の NaN と同じ時刻なので、起動したばかりの run を止めない）
        mark = max(t["last_wall"] for t in run["tags"].values() if isinstance(t["last"], str))
        if done.get(label) == mark:
            continue
        n = stop(cfg, run["run_dir"])
        rid = f"{now_stamp()}-{cfg['machine']}-autostop-{label}"
        reasons = ", ".join(run["hard_stop"][:5])
        respond(repo, cfg, rid, {"kind": "autostop", "run": run["run"]},
                f"最新値が NaN / Inf になったので止めました（step {step}）。\n\n" + "\n".join(f"- {r}" for r in run["hard_stop"]),
                "ok", f"🛑 {label} を即停止（{reasons}、{n} プロセス）")
        done[label] = mark
        repo.write_state("hard_stopped", done)


def loop(cfg):
    repo = DataRepo(cfg["data_dir"], need(shared_config(), "data_branch", "watch/config.json"))
    if cfg["discord"]:
        discord_token()   # 投稿する設定ならトークンが無いと始めない
    print(f"[watch] {cfg['machine']} のエージェントを開始（依頼の確認 {REQUEST_POLL_S} 秒ごと、数値の送信 {cfg['interval_s']} 秒ごと）", flush=True)
    last_export = 0.0
    while True:
        try:
            repo.pull()
            if time.time() - last_export >= cfg["interval_s"]:
                last_export = time.time()   # 書き出しが失敗しても毎分やり直さない
                data = export(cfg)
                changed = write_status(repo, cfg, data)
                repo.commit_push(f"watch: {cfg['machine']} status ({len(changed)} run)")
                hard_stops(repo, cfg, data)
                repo.commit_push(f"watch: {cfg['machine']} autostop")
            handle_requests(repo, cfg)
        except Exception:
            print(f"[watch] {now_iso()} ループでエラー（続行）:\n{traceback.format_exc()}", flush=True)
        time.sleep(REQUEST_POLL_S)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("loop")
    sub.add_parser("export")
    sub.add_parser("running")
    s = sub.add_parser("stop")
    s.add_argument("run", help="run 名（status から一意に決まるとき）か、コンテナ内の run ディレクトリ")
    s = sub.add_parser("launch")
    s.add_argument("command", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    try:
        cfg = load_cfg()
        if a.cmd == "loop":
            loop(cfg)
        elif a.cmd == "export":
            print(json.dumps(export(cfg), ensure_ascii=False, indent=1)[:20000])
        elif a.cmd == "running":
            for pid, cmd in running(cfg):
                print(pid, cmd)
        elif a.cmd == "stop":
            print(f"{stop(cfg, resolve_run(cfg['data_dir'], cfg, a.run))} プロセスを止めました")
        elif a.cmd == "launch":
            cmd = a.command[1:] if a.command[:1] == ["--"] else a.command
            print(launch(cfg, cmd))
    except WatchError as e:
        sys.exit(f"[watch] {e}")


if __name__ == "__main__":
    main()
