# train-watch — 学習の監視と自律改良（auto-study の簡易版）

学習 PC の数値を git で Mac に運び、Mac の Claude（`/watch-training` スキル）が判断して、学習 PC の Codex に「止める・直す・回し直す」を依頼する。
やり取りは Discord（otoham サーバーのカテゴリ `spica-watch`）に短く残す。2026-11-04 新設。

```
学習 PC                                   origin（watch-data ブランチ）           Mac
pc_agent.py loop ── status/<PC>/<run>.json ──▶                  ──▶ mac.py wait / sync（Claude）
  ・exporter.py をコンテナに流し込み TB scalar を要約                       判断 → 止める / work_dir で直して push
  ・NaN / Inf は即停止（autostop）                                      → mac.py request
  ・requests/ を処理（stop は自分、それ以外は Codex） ◀── requests/<id>.md, notes/<id>.md ──
  └ responses/<id>.md ─────────────────────────▶                   ──▶ sync で確認・Discord に中継
Discord: #watch-log（やり取りの履歴・改良内容の md）、#watch-call（研究者の判断が必要なときの呼び出し）
```

- コード: このディレクトリ（ブランチ `016-f_main-train_watch`）。学習ブランチとは独立に、別の checkout から動かす
- データ: orphan ブランチ `watch-data`（コードは置かない。書き手ごとにディレクトリが分かれているので衝突しない）
- 送るのは数値と Markdown だけ。**医用画像とその派生画像は送らない**
- 自律範囲: 停止は承認なし。パラメータと、それに伴うバグの修正は同じブランチで直してよい。新しい損失・構造などの大きな変更は、新しいブランチを切るなら可。今のブランチへの大きな変更と Pivot は研究者の承認（#watch-call）。詳細は `skill/SKILL.md` と `codex_prompt.md`

## 学習 PC のセットアップ（Windows、Git Bash）

1. 必要なもの: Git for Windows（Git Bash）、Python 3（start.sh と同じもの）、Docker Desktop（既存）、**Codex CLI**（`npm i -g @openai/codex` → `codex login`）
2. コードとデータの checkout（学習用の SpicaV5 とは別のフォルダ）:
   ```bash
   cd ~/Documents
   git clone -b 016-f_main-train_watch https://github.com/otoha1119/SpicaV5.git SpicaV5-watch
   git clone -b watch-data --single-branch https://github.com/otoha1119/SpicaV5.git SpicaV5-watch-data
   ```
3. `SpicaV5-watch/watch/pc_config.example.json` を `pc_config.json` にコピーして書き換える（`machine` / `repo_dir` = 学習に使う SpicaV5 の checkout / `data_dir` / `launch_shell`（`wsl` = ふだん VS Code のターミナルで `bash start.sh` を打つのと同じ WSL、`git_bash`）/ `git_bash`）
4. Discord に直接書くなら（推奨）、Windows のユーザー環境変数 `DISCORD_BOT_TOKEN` に auto-study の bot トークンを入れる（Mac の `~/.config/autostudy/secrets.env` と同じ値）。
   入れない場合は `"discord": false` にすると Mac が sync のときに中継する（Mac が止まっていると通知が遅れる）
5. 起動（Git Bash で。閉じると止まる）:
   ```bash
   cd ~/Documents/SpicaV5-watch && python watch/pc_agent.py loop
   ```
   確認だけなら `python watch/pc_agent.py export`（push しない）、`python watch/pc_agent.py running`

## Mac のセットアップ（初回だけ。済み）

```bash
git worktree add ../SpicaV5-watch -b 016-f_main-train_watch main      # コード
git worktree add --orphan -b watch-data ../SpicaV5-watch-data          # データ
cp watch/mac_config.example.json watch/mac_config.json
python3 watch/mac.py setup            # Discord のカテゴリ・チャンネルを作り config.json に id を書く
python3 watch/mac.py install-skill    # ~/.claude/skills/watch-training
```
使うときは Claude Code で `/watch-training`。

## 未検証（2026-11-04 時点。Mac 上の偽コンテナと偽 Codex で往復だけ確認）

- Windows での `launch`（`git-bash.exe -c` で新しいウィンドウを開き、その中で winpty 付きの start.sh が動くか）
- Windows での pc_agent 全般（パス表記、`docker exec -i` に標準入力でスクリプトを流す動作、長時間の常駐）
- 本物の spicav5 コンテナでの `stop`（run 名で学習プロセスを見つけて SIGTERM。保存中に止めた場合も latest から resume できるか）
- Codex の権限: 既定は `--dangerously-bypass-approvals-and-sandbox`（docker と git push に要るため）。Windows のサンドボックス事情で別の指定が要るかもしれない
- Codex の実際の振る舞い（codex_prompt.md の規則を守るか、3 分以内の起動確認、報告の形）
- 大きな event ファイル（画像入り）での exporter の速度とキャッシュ（`/tmp/spica_watch_exporter_cache.json`、コンテナを作り直すと最初から読み直す）
- 実 run の TB tag 名での `mac.py` の要約（tag 名にヒント文字列を含むものだけ表示。全部は JSON に入っている）
