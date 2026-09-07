# research-skills

汎用ML研究運用スキル群。SOTA調査 → 実験計画 → 実行 → 評価 → 記録 → Tune/Pivot判断 → 報告、という研究サイクル全体をClaude Codeのスキルとして規律化する。

SpicaV3のCLAUDE.mdにある研究運用ルール——**推測で埋めない・良い結果も悪い結果も記録する・事実と解釈を分ける**——を、毎回意識しなくても実行される仕組みに落とし込んだもの。特定プロジェクトに依存しない汎用・独立型。

設計の全容は [docs/spec.md](docs/spec.md)、実装計画は [docs/superpowers/plans/2026-08-19-research-skills.md](docs/superpowers/plans/2026-08-19-research-skills.md) を参照。

## 収録スキル（4本柱）

| スキル | 役割 | 呼び出し |
|---|---|---|
| `sota-survey` | 多角的・定量的なSOTA/手法調査 | 自動発火 + `/sota-survey` |
| `experiment-cycle` | 事前登録つき実験ループ、Tune/Pivot判断（6転換ゲート） | 自動発火 + `/experiment-cycle` |
| `research-report` | 実験ログ・判断記録からの報告生成 | 自動発火 + `/research-report` |
| `research-setup` | 対象プロジェクトへの導入・更新（冪等） | `/research-setup` のみ |

原則：「規律・ワークフロー系は自動発火、破壊的操作（インストール）は明示専用」。

### sota-survey（調査）

「SOTAを調べて」「代替手法ある？」で発火。問題を定式化してから、**古典 / デファクト / SOTA研究 / 問題再定義の4系統＋異分野越境**で候補を発散させ、次の3点セットで定量的に収束させる：

1. **生値テーブル**（精度・速度・実装コスト・データ要件・成熟度。データセット条件併記、検証不能値は「未確認」マーク）
2. **重み付きスコアリング**（重みはノート内で宣言して恣意性を可視化）
3. **Pareto所見**

推薦は2〜3件で必ず異なる原理を混ぜ、各推薦に**最小反証実験・タイムボックス・撤退ライン・移行/ロールバック方針**を付ける。最小反証実験はexperiment-cycleの事前登録形式と同一項目なので、採用したらそのまま実験ログへ転記できる。SOTA主張には出典＋確認日必須。棄却候補も理由付きで残す。検索は英語、証拠引用は英語原文のまま。

出力先：`docs/surveys/<YYYYMMDD>_<slug>.md`

### experiment-cycle（実験ループ）

「実験して」「学習回して」で発火。中核となる規律：

- **事前登録**：実行前に仮説・反証条件・Go/No-Go基準・タイムボックスを固定してから実験する
- **再現条件の完全記録**：commitハッシュ（dirty有無）・データセット版・seed・試行回数・環境・コピペ可能な再現コマンド
- **ネガティブ結果も義務記録**。確定した記録は書き換えず追記訂正のみ
- **凍結ベースラインとの公平比較**：ベースライン確立が先。比較条件を揃える
- **Tune / Pivot の2レーン制**：局所改善（Tune）を続けるか手法転換（Pivot）するかは、**6つの転換ゲート**で判断する
  1. 理論的/経験的上限が要件未満と判明
  2. 同系統実験のN連続棄却（デフォルト3。`docs/research-config.md` で上書き可）
  3. 定数でなく計算量オーダーの変更が必要
  4. 集計指標は停滞しつつ特定データスライスが失敗し続ける
  5. 複雑性・保守コストの増加が利得を上回る
  6. 手法の前提を壊す新要件
- **Pivotの最終決断だけは人間（あなた）の承認が必須**。それ以外の承認ゲートはない。承認後の転換先探索はsota-surveyに接続する

出力先：実験ログ `docs/experiments/EXP-<YYYYMMDD>-<seq>.md`、判断記録 `docs/decisions/<YYYYMMDD>_<slug>.md`

### research-report（報告）

「進捗まとめて」で発火。`docs/{experiments,decisions,surveys}/` の記録を読み、事実（記録の生値を転記）と解釈（報告時点の考察）を分離した報告書を生成する。ネガティブ結果を省略しない。記録にない数値は補完せず「記録なし」と書く。

出力先：`docs/reports/<YYYYMMDD>_<slug>.md`

### research-setup（インストーラ）

`/research-setup` 専用（自動発火しない）。対象プロジェクトで実行すると、8手順を検証しながら対話的に進める。**冪等**——再実行しても壊れず、スキル更新の配り直しにも同じコマンドを使う。

## セットアップ

### 1. インストーラを呼べるようにする（初回のみ）

このリポジトリの `skills/research-setup` へのシンボリックリンクをユーザーレベルの `~/.claude/skills/` に張る：

```bash
ln -s /Users/otoha/Documents/program/Claude/research-skills/skills/research-setup ~/.claude/skills/research-setup
```

### 2. 対象プロジェクトで実行

対象の研究プロジェクトをClaude Codeで開き：

```
/research-setup
```

### 3. インストーラがやること

| # | 内容 |
|---|---|
| 1 | 配布元（このリポジトリの `skills/`）の特定と検証 |
| 2 | 自作3スキル（sota-survey / experiment-cycle / research-report）を対象の `.claude/skills/` へコピー |
| 3 | 外部スキル11種の導入（下表）。`.agents/skills/` 止まりで認識されない事故への対策（シンボリックリンク補修）つき |
| 4 | superpowersプラグインの有効化（`.claude/settings.json`。壊れたJSONなら書き込まず中断） |
| 5 | 記録ディレクトリ `docs/{surveys,experiments,decisions,reports}/` の作成 |
| 6 | `docs/research-config.md` の生成（既存なら不変更）と、評価指標・ベースラインの対話設定 |
| 7 | CLAUDE.mdへ研究運用ルールを追記（マーカー区間 `<!-- research-skills:start/end -->` で冪等に置換） |
| 8 | 全スキルの認識検証と導入結果一覧の提示 |

### 導入される外部スキル

| 提供元 | スキル | 用途 |
|---|---|---|
| mattpocock/skills | `research` / `grilling` / `grill-me` | 一次情報の深掘り調査／計画の詰め（グリル） |
| K-Dense-AI/scientific-agent-skills | `paper-lookup` | 文献検索（arXiv・PubMed・Semantic Scholar等11 API横断） |
| 〃 | `literature-review` | 系統的文献レビュー |
| 〃 | `citation-management` | 引用検証・DOI→BibTeX |
| 〃 | `scientific-writing` | 引用追跡つき科学文書執筆 |
| 〃 | `peer-review` | 査読シミュレーション |
| lllllllama/rigorpilot-skills | `ai-research-reproduction` | 論文リポジトリの再現実装 |
| 〃 | `env-and-assets-bootstrap` | 環境・データセット・重みの準備 |
| 〃 | `safe-debug` | 診断してから直す安全デバッグ |

実験ループ本体（RigorPilotのrun-train等）は自作のexperiment-cycleと競合するため意図的に導入しない。wandb/skillsはTensorBoard運用継続のため見送り。

## 日常の使い方（研究サイクル）

```
新テーマ・頭打ち時
  「この問題の手法を調べて」 → sota-survey が docs/surveys/ に調査ノート
        ↓ 推薦を採用
  「この手法で実験して」 → experiment-cycle が事前登録 → 実行 → docs/experiments/ に記録
        ↓ 結果をもとに
  Tune継続（そのまま次の実験へ） or 頭打ち → 6ゲート判定 → あなたが承認したらPivot
        ↓ 節目で
  「今月の進捗まとめて」 → research-report が docs/reports/ に報告書
```

閾値などのプロジェクト固有調整は `docs/research-config.md` を直接編集する。

## 共通規約（全スキル共通）

- **実験ID**：`EXP-<YYYYMMDD>-<seq>`（例：EXP-20260819-01）
- **記録パス**：`docs/{surveys, experiments, decisions, reports}/` に固定
- **言語**：検索・論文読解は英語／証拠引用は英語原文のまま／記録・報告・要約は日本語（専門用語は英語のまま）
- **科学的規律**：ベースライン先行、成功基準の事前固定、ネガティブ結果も記録、確定記録は追記訂正のみ、SOTA主張には出典＋確認日

## 既知の問題（初回使用前に確認）

1. **【要修正】シンボリックリンク経由の配布元検出**：現状のresearch-setup手順1は、シンボリックリンク経由で呼ばれるとbase directoryが論理パスで渡されるため配布元の特定に失敗する（実測確認済み）。修正は手順1に「シンボリックリンクの場合は `realpath` で実体パスに解決してから親を辿る」の1文を追加するだけ。**修正が入るまでは、手順1で止まったらClaudeに「realpathで解決して」と伝えれば先へ進める**
2. `research-config.md` の「チューニングのタイムボックス」「データセット規約」はどのスキルもまだ読まない（前者は実験ごとの事前登録が実質カバー、後者は人間向けメモ）
3. 体裁の細部（括弧種の不統一等）が数件残っている。詳細はリポジトリ内 `.superpowers/sdd/2026-08-19-research-skills/progress.md` のparked項目を参照

## リポジトリ構成

```
research-skills/
├── README.md                  # 本ファイル
├── docs/
│   ├── spec.md                # 確定仕様（grillingセッションで合意した設計の全容）
│   └── superpowers/plans/     # 実装計画
└── skills/
    ├── sota-survey/           # SKILL.md + templates/survey_note.md
    ├── experiment-cycle/      # SKILL.md + templates/{exp_log,decision_note}.md
    ├── research-report/       # SKILL.md + templates/report.md
    └── research-setup/        # SKILL.md + templates/{research-config,claude-md-snippet}.md
```
