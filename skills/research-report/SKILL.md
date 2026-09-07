---
name: research-report
description: Generate research progress reports and summaries from experiment logs, decision records, and survey notes under docs/. Use when the user asks to summarize research progress, write a progress report, or compile results for a meeting or paper. Triggers include 進捗まとめ, 報告書, 進捗報告, research summary, report.
---

# research-report

実験ログ（`docs/experiments/`）・判断記録（`docs/decisions/`）・調査ノート（`docs/surveys/`）から進捗報告やまとめを生成するスキル。報告書は `docs/reports/<YYYYMMDD>_<slug>.md` に固定で残す。

## 手順

### ① 対象範囲の確認

報告の対象期間、または対象とする実験（実験ID・実験系統など）をユーザーに確認する。ユーザーから期間や対象がすでに指定されている場合は、それをそのまま使い、改めて聞き返さない。

対象が定まったら、`docs/experiments/`・`docs/decisions/`・`docs/surveys/` を確認し、対象範囲に該当する記録（`EXP-<YYYYMMDD>-<seq>.md`、判断記録、調査ノート）をすべて読む。

### ② 報告書の作成

このスキルのディレクトリを基準にした相対パス `templates/report.md` を読み、そのテンプレートの構成（サマリー→実施した実験→判断・方針転換→得られた知見→解釈と次の計画）に従って `docs/reports/<YYYYMMDD>_<slug>.md` を新規作成する。

- `<YYYYMMDD>` は報告書の作成日。
- 「実施した実験（事実）」の表は、対象の実験ログから仮説・結果・判定を転記する。結果は実験ログに記録された**生値**をそのまま転記し、要約・丸めによって数値を変えない。判定列には、実験ログの「判定」セクションに記録された**Go/No-Go**の結果と次のアクション（Tune継続・Pivot検討を提起・完了）を転記する。
- 「判断・方針転換（decisions/より）」には、対象範囲に該当する判断記録（Tune継続・Pivotなど）の内容を転記する。
- 「得られた知見（ネガティブ結果を含める）」には、仮説が棄却された、Go/No-Go基準を満たさなかったといった**ネガティブ結果**も、都合よく省略せず必ず含める。
- 「解釈と次の計画（事実と分離して書く）」には、報告時点での解釈・考察・今後の方針など、記録上の事実ではなく報告者の判断にあたる内容を書く。これらは「実施した実験」「判断・方針転換」「得られた知見」の各セクションには書かず、**事実と解釈を分離**したこのセクションに集約する。

### ③ 論文・学会向け整形への接続

対象プロジェクトに `scientific-writing`（原稿執筆用スキル）や `citation-management`（引用検証・BibTeX管理用スキル）が導入されている場合、論文原稿や学会発表向けに報告内容を整形する段階でそれらのスキルを使う。導入されていない場合は、このスキル単独で `docs/reports/` への報告書作成までを行う。

### ④ 推測で埋めない

記録（実験ログ・判断記録・調査ノート）に存在しない数値や主張を、報告書に補完してはならない。対象範囲に該当する記録が見つからない項目や、記録はあるが値が欠けている項目は、空欄にしたり推測で埋めたりせず「記録なし」と明記する。
