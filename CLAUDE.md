<!-- research-skills:start -->
## 研究運用ルール（research-skills）

- 実験はexperiment-cycleスキルの規律に従う：実行前に仮説・反証条件・Go/No-Go基準・タイムボックスを固定し、`docs/experiments/EXP-<YYYYMMDD>-<seq>.md` に記録する。ネガティブ結果も必ず記録し、確定した記録は追記訂正のみとする
- 手法調査はsota-surveyスキルで多角的（古典/デファクト/SOTA/再定義/越境）かつ定量的に行い `docs/surveys/` に記録する。SOTA主張には出典と確認日を付ける
- Tune/Pivot判断は `docs/decisions/` に記録する。Pivot（手法転換）はユーザー承認がない限り実行しない
- 閾値等の上書き設定は `docs/research-config.md`
- 言語規約：検索・論文読解は英語、証拠引用は英語原文のまま、記録・報告は日本語（専門用語は英語のまま）
<!-- research-skills:end -->

## 進め方の規律（2026-11-11、ユーザー指示）

- **手で測った数字や単純な区切りを、設計の的にしない。** ノイズ量・ぼけ幅を測ってその値に合わせる処理を作る、HU のしきい値で組織を分ける、固定のぼかし量を決める、といった進め方を提案しない。臓器の模様とノイズは手では分けられず、分けられるのは学習だけ。測定は「結果の確認」にだけ使う
- **解決策は概念で考え、要件から導く。** 設計の部品（ネットワーク、損失、データの与え方）は 1 つずつ、要件のどの行に対応するか・なぜその形になるかの根拠を付ける。根拠があれば大きく作り直してよい（小刻みな追加に寄せない）。要件に対応しない部品は入れず、どの部品も対応していない要件は「抜け」として示す
- **要件から先回りしない。** 議論中の論点を詰め切る前に、要件案や選択肢を出さない
- 要件の再定義（2026-11-10 開始）の正本は `docs/reference/requirements_v3_manual.md`。旧システムの説明（`old/README.md`）と食い違う所は、こちらを優先する

## 現在の研究方針（2026-11-11）

- **案 E**（Pivot、ユーザー承認済み）: **Stage 1 = 実 EID だけで学んだ拡散モデルに描き直させる**（PCD を粗くした絵を手がかりに EID を描く。PCD は Stage 1 の学習に見せない。判別器と全画素 fidelity は使わない）。**Stage 2 = U-Net（形）+ フロー（鋭さとノイズの粒）**（フローには入力の EID を見せない。判別器は使わない）
- 決定記録: `docs/decisions/20261111_pivot-redraw-stage1-create-stage2.md`（末尾の追記で、Stage 1 の描き手の設計は案 F `docs/plans/20261111_solution_proposals/F_huang_base_vs_E.html` に更新されている。実装前にここを読む）
- 設計: `docs/plans/20261111_solution_proposals/E_final_proposal.html`
- 要件の正本: `docs/reference/requirements_v3_manual.md`
- `docs/` は git 管理外（ローカルだけ）。研究内容はネットに公開しない

## リポジトリの構成（2026-11-11 整理）

- `stage1/`（直下）: **新しい Stage 1（描き手 P）の実装**（ブランチ `028-f_main-redraw_create`）。起動は `bash start.sh [マシン名] [build|shell|down|tb|generate] [resume <run> [tag]] [best <run> <tag>] [--flag value]`、設定は `stage1/configs/{train,model,generate}.yaml` + `configs/shared.yaml`（正は `stage1/configs/schema.py`）、説明 `stage1/README.md`、式の出典と設計書からの差分 `stage1/UPSTREAM.md`。学習結果は `runs/stage1/<run>/`（`machines.yaml` の `painter_checkpoints_dir`）、疑似 EID は `pseudo_eid_dir`
- `stage2/`（直下）: 新しい Stage 2（未実装）。学習結果の置き場は `runs/stage2/`（`flow_checkpoints_dir`。コンテナ内 `/workspace/runs/...`、git 管理外）
- `old/`: 2026-11-11 以前の実装（旧 Stage 1 / Stage 2 / ablation と起動 sh）。説明は `old/README.md`。起動は `bash old/start.sh` / `bash old/start2.sh` / `bash old/start_ablation.sh`。旧コードの学習結果・出力も `old/` 配下（`old/stage1/checkpoints`・`old/stage2/checkpoints`・`old/ablation/checkpoints`・`old/output`。`configs/machines.yaml` の `*checkpoints_dir`）
- 共有: `configs/`（`machines.yaml` = マシン定義、`shared.yaml` = 両 Stage で共有する値、`eval_only.yaml` = 評価専用症例、`splits/`）、`docker/`（世代別 compose / Dockerfile。リポジトリは `/workspace` にマウント）、`create_dataset/`
- 監視: `watch/`（train-watch。説明は `watch/README.md`）
- 直下の `start.sh` は新しい Stage 1 の起動口（Stage 2 の起動口は別に作る）

## 実装の決まり

- **bf16 autocast と `torch.compile` を最初から入れる（研究者の指示 2026-11-11）**: 新しく学習コードを作るときは、最初から学習の forward に bf16 の autocast と `torch.compile` を入れる（旧 Stage 2 の `optim.precision: bf16` / `optim.compile: true` と同じ考え方。実装例 `old/stage2/models/regression_model.py`）。**損失・検証・推論は FP32**。どちらも設定で切れるようにする
- **既定値・フォールバック禁止**: 設定は yaml + schema（宣言したキーは必須、未知キー・型違いはエラー）。sh の `--flag` が最優先で、実効値を run の `launch.yaml` に保存する。再開の基準は checkpoint の `state.pth` 内の実効設定
- **テストコードはリポジトリに置かない**。リポジトリ外の scratch venv / 合成データでの smoke 実行は行う
- 全環境 Docker 内で実行する。コンテナ `spicav5` が起動済みなら up を呼ばず exec だけ（中の処理を止めない。止めるのは down だけ）
- 評価専用症例（`configs/eval_only.yaml`）は学習・検証・症例分割に使わない（推論・評価には通してよい）
- データは 1ch 16bit PNG、`stored = HU + 1400`。表示の正規化は HU −1400〜3000（stored 0〜4400）
- 記載が無い点は推測で埋めず、その場でユーザーに確認する
