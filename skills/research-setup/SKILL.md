---
name: research-setup
description: Install or update the research-skills set (sota-survey, experiment-cycle, research-report) plus external companion skills into the current project. Idempotent — safe to re-run for updates. Explicit invocation only.
disable-model-invocation: true
---

# research-setup

research-skills（sota-survey / experiment-cycle / research-report）と、それらが利用する外部スキルを対象プロジェクトへ導入・更新するためのインストーラ。`/research-setup` による明示呼び出し専用（自動発火しない）。

全ステップは**再実行しても壊れない**（冪等）。初回導入と、スキル更新の配り直しは同じ手順・同じコマンドで行う。以下の1〜8を、番号の順に、各ステップの「検証」がすべて満たされたことを確認しながら進める。途中のステップで検証に失敗した場合は、次のステップに進む前にユーザーに状況を報告する。

## 手順

### 1. 前提確認

**実行：**
- カレントディレクトリが対象プロジェクトのルートであることを確認する。
  - `git rev-parse --is-inside-work-tree` がgit管理下であることを示すか確認する。
  - `git rev-parse --show-toplevel` の結果とカレントディレクトリの絶対パスが一致することを確認する。一致しない場合（サブディレクトリで実行された場合）は、続行前にユーザーに「プロジェクトルートで実行し直すか、このまま進めるか」を確認する。
- 配布元ディレクトリを特定する。このSKILL.mdを読み込んだファイルパスの親ディレクトリが、このスキルのbase directory（`.../skills/research-setup/`）である。そのさらに親ディレクトリ（`.../skills/`）が配布元となる。

**検証：** 配布元直下に `sota-survey/SKILL.md`・`experiment-cycle/SKILL.md`・`research-report/SKILL.md` の3ファイルが存在することを確認する。1つでも欠けていれば配布元が壊れているとみなし、ここで中断してユーザーに報告する（以降のステップは実行しない）。配布元の特定に失敗した場合（親ディレクトリに3スキルのディレクトリが見つからない場合）も同様に中断し、README.mdの導入方法（リポジトリへのシンボリックリンク経由で呼び出すこと）をユーザーに案内する。

**冪等性：** 確認のみで状態を変更しないため、常に安全に再実行できる。

### 2. スキル配置

**実行：**
```bash
mkdir -p .claude/skills
cp -R <配布元>/sota-survey <配布元>/experiment-cycle <配布元>/research-report .claude/skills/
```
（`<配布元>` はステップ1で特定したディレクトリ。）

**検証：** `ls .claude/skills/*/SKILL.md` を実行し、`.claude/skills/sota-survey/SKILL.md`・`.claude/skills/experiment-cycle/SKILL.md`・`.claude/skills/research-report/SKILL.md` の3つが出力に含まれることを確認する。

**冪等性：** `.claude/skills/sota-survey` 等が既に存在する場合、`cp -R` は配布元にある同名ファイルを上書きする（＝更新）が、配布元に存在しないファイルの削除は行わない（マージ動作）。ディレクトリごと消してから作り直す必要はないが、スキル更新で不要になったファイルが `.claude/skills/` 側に残る場合があるため、その旨を認識しておく。

### 3. 外部スキル導入

**実行：** 以下の3コマンドを、順に実行する。1つのコマンドが失敗しても、そのエラーを報告した上で次のコマンドへ進む（全体を中断しない）。

```bash
npx skills@latest add mattpocock/skills --skill research --skill grilling --skill grill-me -a claude-code -y
npx skills@latest add K-Dense-AI/scientific-agent-skills --skill paper-lookup --skill literature-review --skill citation-management --skill scientific-writing --skill peer-review -a claude-code -y
npx skills@latest add lllllllama/rigorpilot-skills --skill ai-research-reproduction --skill env-and-assets-bootstrap --skill safe-debug -a claude-code -y
```

導入対象は次の11スキル：`research`・`grilling`・`grill-me`（mattpocock/skills）、`paper-lookup`・`literature-review`・`citation-management`・`scientific-writing`・`peer-review`（K-Dense-AI/scientific-agent-skills）、`ai-research-reproduction`・`env-and-assets-bootstrap`・`safe-debug`（lllllllama/rigorpilot-skills）。

導入後、これら各スキルについて `.claude/skills/<name>` または `.agents/skills/<name>` のいずれかに実体（`SKILL.md` を含むディレクトリ、またはそこへのシンボリックリンク）があり、かつ `.claude/skills/` 側から到達可能かを確認する。**`.agents/skills/<name>` にしか実体が無く、`.claude/skills/<name>` にシンボリックリンクも存在しない場合**は、次のコマンドでリンクを作成する（配置先が認識されない事故への対策）。

```bash
[ -e .claude/skills/<name> ] || ln -s ../../.agents/skills/<name> .claude/skills/<name>
```

**検証：** 11スキルそれぞれについて、`.claude/skills/<name>/SKILL.md`（シンボリックリンク経由を含む）が読めることを確認する。読めないものがあれば、そのスキル名と原因（npxコマンドの失敗／配置先不一致など）を記録し、後述ステップ8の一覧表に反映する。

**冪等性：** `npx skills@latest add ... -y` は同一スキル・同一バージョンであれば再実行しても内容を更新するだけで、更新の配り直しにそのまま使える。シンボリックリンク作成は `[ -e ... ] ||` で既存リンクがあれば何もしないため、再実行しても失敗しない。

### 4. superpowers有効化

**実行：** `.claude/settings.json` を読む。
- ファイルが存在し、JSONとして読み込めた場合：`enabledPlugins` オブジェクト内に `"superpowers@claude-plugins-official": true` が無ければ、既存のキー（`enabledPlugins` 内外を問わず）をすべて保持したまま、このキーを追加して書き戻す。既に `true` であれば何もしない。
- ファイルが存在しない場合：`{"enabledPlugins": {"superpowers@claude-plugins-official": true}}` の内容で新規作成する。
- ファイルが存在するがJSONとしてパースできない場合：**書き込まずに中断**し、ファイルパスとパースエラーの内容をユーザーに報告して指示を仰ぐ（上書きで新規作成すると既存の権限設定・フック設定を破壊するため）。

**検証：** 書き込み後の `.claude/settings.json` を再度読み、（a）有効なJSONとしてパースできること、（b）`enabledPlugins["superpowers@claude-plugins-official"]` が `true` であること、（c）このキー以外の既存キーが変更前と変わっていないこと、の3点を確認する。

**冪等性：** 既にキーが `true` になっている場合は書き込みを行わない（内容が変わらないため再実行しても差分が出ない）。

### 5. 記録ディレクトリ

**実行：**
```bash
mkdir -p docs/surveys docs/experiments docs/decisions docs/reports
```

**検証：** `ls -d docs/surveys docs/experiments docs/decisions docs/reports` の4つすべてが存在することを確認する。

**冪等性：** `mkdir -p` は対象がすでに存在してもエラーにならず、既存ファイルを破壊しない。

### 6. 設定生成

**実行：** `docs/research-config.md` の存在を確認する。
- 存在しない場合：このスキルのbase directory基準の `templates/research-config.md` を `docs/research-config.md` としてコピーする。その後、「主要評価指標」「現行ベースライン（`<EXP-ID>`）」をユーザーに質問して該当項目を埋める。「データセット規約」も分かる範囲で埋め、不明な項目は空欄のまま残してよい。
- 存在する場合：**一切変更しない**（プロジェクト側の上書き設定を尊重する）。検証のため、変更前にハッシュ（または内容コピー）を取得しておく。

**検証：** `test -f docs/research-config.md` でファイルの存在を確認する。新規作成した場合は、少なくとも「主要評価指標」と「現行ベースライン」の行がテンプレートのプレースホルダのままではなく、ユーザーとのやり取りに基づく値（または明示的な「未定」等）に置き換わっていることを確認する。既存だった場合は、実行前後でファイルの内容（ハッシュ等）が一致することを確認する。

**冪等性：** 既存ファイルには触れないため、再実行しても内容は変わらない。

### 7. CLAUDE.md追記

**実行：** このスキルのbase directory基準の `templates/claude-md-snippet.md` を読む。プロジェクトルートの `CLAUDE.md` を確認する。
- `CLAUDE.md` が存在しない場合：新規作成し、`templates/claude-md-snippet.md` の内容をそのまま書き込む。
- `CLAUDE.md` が存在し、`<!-- research-skills:start -->` 〜 `<!-- research-skills:end -->` のマーカー区間が既にある場合：その区間全体（開始マーカー行から終了マーカー行まで、両マーカー行を含む）を `templates/claude-md-snippet.md` の内容で**丸ごと置換**する。区間外の既存内容には一切手を加えない。
- `CLAUDE.md` は存在するがマーカー区間が無い場合：ファイル末尾に空行を挟んで `templates/claude-md-snippet.md` の内容を追加する。

**検証：** 書き込み後の `CLAUDE.md` に対して `grep -c` 等で `<!-- research-skills:start -->` と `<!-- research-skills:end -->` がそれぞれ**ちょうど1回ずつ**出現することを確認する（2回以上あれば追記が二重化しているので誤り）。さらに、マーカー区間の内容が `templates/claude-md-snippet.md` の内容と一致することを確認する。

**冪等性：** 既存マーカー区間を区間ごと置換する（差分がない場合は結果的に無変化）ため、同じ内容で何度再実行してもマーカーが増殖しない。

### 8. 最終検証

**実行：** ステップ2・3で導入した全スキル（`sota-survey`・`experiment-cycle`・`research-report` と、ステップ3で導入対象とした11の外部スキル）について、`.claude/skills/<name>/SKILL.md` のfrontmatterにある `name:` 行を実際に読み、内容が読めることを確認する。

その結果を「スキル名／配置先／状態（成功・失敗・要リンク作成 等）」の一覧表としてユーザーに提示する。あわせて、「スキルの認識は次のメッセージ以降で反映される場合がある」ことをユーザーに伝える。

**検証：** 一覧表に、ステップ2・3で対象とした全スキル（コア3種＋外部11種、計14種）が過不足なく記載されていることを確認する。

**冪等性：** 読み取りと報告のみで状態を変更しない。
