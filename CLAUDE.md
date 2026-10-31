<!-- Claude Code と Codex の共通ルール。AGENTS.md はこのファイルへの symlink。経緯・履歴は書かない -->
## 作業の進め方（トークン節約、2026-10-30）

- 独立したシェルコマンドは 1 回の呼び出しにまとめる。確認のためだけの小刻みな呼び出しはしない
- 大きな画像は読む前に縮小する。大きなファイルは必要な範囲だけ読む。ログは tail / grep で絞る
- コンテキストが 100k tokens を超えたら compact（Claude Code は `/context` で確認して `/compact`）、タスクが変わったら新しいセッションにする
- 経緯・履歴はこのファイルに書かず `docs/` に置く。このファイルは運用ルールと「どこを見るか」だけ

## 研究運用ルール（research-skills）

- 実験は experiment-cycle スキルの規律に従う：実行前に仮説・反証条件・Go/No-Go 基準・タイムボックスを固定し、`docs/experiments/EXP-<YYYYMMDD>-<seq>.md` に記録する。ネガティブ結果も記録し、確定した記録は追記訂正のみ
- 手法調査は sota-survey スキルで多角的・定量的に行い `docs/surveys/` に記録する。SOTA 主張には出典と確認日を付ける
- Tune/Pivot 判断は `docs/decisions/` に記録する。Pivot（手法転換）はユーザー承認がない限り実行しない
- 閾値等の上書き設定は `docs/research-config.md`
- 言語規約：検索・論文読解は英語、証拠引用は英語原文のまま、記録・報告は日本語（専門用語は英語のまま）
- **テストコードはリポジトリに置かない**。smoke 実行はリポジトリ外の scratch venv / 合成データで行う
- 論文（Park 2019）に記載が無い点は推測で埋めず、その場でユーザーに確認する。照合の正は `docs/reference/park2019_implementation_checklist.md`、変更履歴は `stage1/UPSTREAM.md`
- 研究内容はネットに公開しない（Artifact 等禁止）。図や資料は `docs/` に置く

## 現在の研究方針

- 要件の正本 `docs/reference/research_requirements.md`、転換記録 `docs/decisions/`、作業計画 `docs/plans/`。経緯の全文は `docs/reference/claude_md_history_20261030.md`（2026-10-30 までの旧 CLAUDE.md。git 履歴 a5eac05 にもある）
- Stage 1 = PCD512 → EID-like512 の非ペア劣化生成（Park et al., IEEE Access 2019 の fidelity GAN を junyanz/pytorch-CycleGAN-and-pix2pix の vendoring `stage1/` 上に実装）。`mode.yaml` の `residual` / `task: degrade | sr` で切替、resume で変更不可
- Stage 2 = EID-like1024 → PCD1024 の教師あり U-Net（ILUMENATE, Koons et al. 2025）。`stage2/`
- Stage 3 = DDIM（`stage3/`、`start3.sh`）。ablation = `ablation/<名前>/`（SRCycleGAN、CycleGAN）
- **Stage 1 v3（2026-10-30〜、ブランチ 011）** = 射影型劣化器（物理コア P + 学習残差 R + 実 EID 恒等項、`stage1_v3/`、`start1_v3.sh [マシン] [train|v4|dn|dncheck|infer|ponly|stats|resume|best|tb]`）。設計 `docs/surveys/20261030_stage1-projection-identity.md`、要件 R-S21〜28。着手順は P 単独（`ponly` + `stats`）→ P + R。**v4（2026-10-31〜、ブランチ 012、`pipeline.mode: v4`）** = x → 凍結 Dn（`dn` で Neighbor2Neighbor 事前学習、`dncheck` で合否）→ G → ⊛ k(z) → + ノイズ（σ ≥ σ_min）。設計 `docs/decisions/20261031_stage1-v3-noise-path-analysis.md` §3・§6。通しは `v4`
- 評価は SENTINEL-CARE（`SpicaV3/docs/working/stage1_evaluation_metric_design.md`）とプラーク評価（`eval/`）。worst-slice をヘッドラインにする

## 設定と起動の原則

- **既定値・フォールバック禁止**。設定の正は各 Stage の `configs/schema.py`。学習パラメータ `configs/train.yaml`、モード `configs/mode.yaml`、マシン定義は共通の `configs/machines.yaml`（キーを足すときは Stage 1 と Stage 2 の schema 両方に書く）。Stage 間で import しない
- 起動はすべてホストの sh から Docker 内で実行: `bash start.sh [マシン] [build|shell|down|tb|infer|crop|split|multi] [resume <run> [tag]] [best <run> <epoch>] [--flag value]`、Stage 2 は `start2.sh`（+ `dataset`）、Stage 3 は `start3.sh`、ablation は `start_ablation.sh [マシン] <名前> ...`。sh の引数が最優先で、実効値は `<run>/launch.yaml` に保存される
- run 名は起動時刻 `yyyy_mmdd_HHMM`。再開の基準は選んだ checkpoint の `state.pth` 内の実効設定
- コンテナは全 Stage 共用。起動済みならどのアクションも exec だけ、build は拒否、止めるのは down だけ。compose の変更は down → 起動し直しで反映
- run レイアウトの正は各 Stage の `util/run_paths.py`（`latest/ best/ weights/epoch_NNN/`、`output_images/`、`preview_*/`、`infer/<重み>/<入力>/<時刻>/`）。推論・パネルの仕様は `docs/plans/20260907_inference-plan.md` と各 `util/panel.py`

## データの約束

- データは 1ch 16bit PNG。`unaligned_dataset.py` は 8bit に潰すので `ct_dataset.py` を使う
- 評価専用症例（PCD-021〜030、EID-1021〜1030）は `configs/eval_only.yaml` の `cases` が唯一の定義。学習・検証・分割に使わない（推論・評価は可）。Stage 1 だけ `--use_eval_only_cases` で除外を切れる（マシン `PC1_new`）
- `task: sr` だけ症例分割（`configs/splits/eid_v5.yaml`、EID-049 を test）。degrade は全症例
- 表示の線形範囲は stored 0〜4400 = HU −1400〜3000（TB / preview / 推論・crop パネル / スライド図すべて）。残差カラーは `util/residual_color.py`
- `patches_per_slice` は全 Stage 8

## 実装の進め方

- 1 変更ずつ。各フェーズは論文の記述と図の照合レビューで確認する
- 変更後はユーザーが自分でコミットするので、`git status` / `git log` を見てから prefix + 日本語一行の推奨コミットメッセージを示す
- 比較対象の手法は Stage 1 / 2 から import せず `ablation/<名前>/` に自己完結で置き、`ablation/README.md` に 1 行足す
