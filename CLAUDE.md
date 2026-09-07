<!-- research-skills:start -->
## 研究運用ルール（research-skills）

- 実験はexperiment-cycleスキルの規律に従う：実行前に仮説・反証条件・Go/No-Go基準・タイムボックスを固定し、`docs/experiments/EXP-<YYYYMMDD>-<seq>.md` に記録する。ネガティブ結果も必ず記録し、確定した記録は追記訂正のみとする
- 手法調査はsota-surveyスキルで多角的（古典/デファクト/SOTA/再定義/越境）かつ定量的に行い `docs/surveys/` に記録する。SOTA主張には出典と確認日を付ける
- Tune/Pivot判断は `docs/decisions/` に記録する。Pivot（手法転換）はユーザー承認がない限り実行しない
- 閾値等の上書き設定は `docs/research-config.md`
- 言語規約：検索・論文読解は英語、証拠引用は英語原文のまま、記録・報告は日本語（専門用語は英語のまま）
<!-- research-skills:end -->

## 現在の研究方針（2026-09-05 改訂）

- 研究の主戦場は **Stage 1（PCD512 → EID-like512、非ペア劣化生成）**。Stage 2 は単純な教師あり SR で保留。転換記録: `docs/decisions/20260905_pivot-stage1-fidelity-gan.md`
- 要件の正本は `docs/reference/research_requirements.md`（R-SC1 改訂、R-S10〜14 が Stage 1 の設計要件）。上書き設定は `docs/research-config.md`
- 土台: Park et al., IEEE Access 2019（一方向 fidelity-embedded GAN、cycle なし）。実装基盤は **junyanz/pytorch-CycleGAN-and-pix2pix 本家を `stage1/` に vendoring** し学習フレームワークとして借りる。5 段階（本家そのまま → 論文 G/D → fidelity GAN → baseline → Residual 化）を一つずつ、独自機構は段階 5 以降
- 評価は SENTINEL-CARE（`SpicaV3/docs/working/stage1_evaluation_metric_design.md`）。構造はハード VETO、worst-slice をヘッドラインにする
- 定量ベースライン（`EXP-<date>-01`）が未発行。実験着手前に必ず確立する
- 実装の進め方（ユーザー指示 2026-09-05、2026-09-07 補足）: **テストコードはリポジトリに置かない**が、リポジトリ外の scratch venv / 合成データでの smoke 実行は行う。各フェーズは論文（Park 2019）の記述と図の照合レビューで確認する。論文に記載が無い点は推測で埋めず、その場でユーザーに確認する。照合の正は `docs/reference/park2019_implementation_checklist.md`、変更履歴は `stage1/UPSTREAM.md`
- 実装仕様 兼 作業計画: `docs/plans/20260905_stage1-fidelity-gan-implementation-spec.md`（v3）。データは 1ch 16bit PNG に変換済み。junyanz の `unaligned_dataset.py` は 16bit を 8bit に潰すので `ct_dataset.py` を自作する
- 設定と起動（2026-09-05 確定）: **既定値・フォールバック禁止**。学習パラメータは `stage1/configs/train.yaml`、モード切替は `stage1/configs/mode.yaml`、マシン定義は `configs/machines.yaml`（Stage 共通）。唯一の正は `stage1/configs/schema.py`。起動はホストで `bash start.sh [マシン名] [build|shell|down|tb|infer] [resume <run> [tag]] [best <run> <epoch>] [--flag value]`（マシン名は start.sh 内の `MACHINE` に書き、`bash start.sh PC2` のように引数で渡せば上書き。sh の引数が最優先）。実験名（run）は起動時刻 `yyyy_mmdd_HHMM`（`/` は使わない。同一分の衝突はエラー）。sh の上書きは実効値として `launch.yaml` に保存され、推論・再開はそれ（最新の `launch_resume_*.yaml`）を読む。全環境 Docker 内で実行: `bash start.sh`（起動 → 学習。イメージが無ければビルド）/ `bash start.sh build`（イメージを（再）ビルドしてコンテナ起動・torch/cuda 確認まで。学習はしない。本番機の初期セットアップ用）/ `bash start.sh resume <run> [latest|best|<epoch>]`（続きから）/ `bash start.sh best <run> <epoch>`（目視で選んだ epoch を best にする。判定指標は未実装）/ `bash start.sh infer [--mode full|patch|both ...]`（症例丸ごと推論。重みディレクトリ・入力フォルダ・出力形式 png|dicom|both・元 DICOM ルートは `infer_stage1.sh` の `WEIGHT_DIR` / `INPUT_DIR` / `OUTPUT_FORMAT` / `DICOM_DIR`、方式は `stage1/configs/infer.yaml`、デバイスは `machines.yaml` の `gpu_gen`、出力は `<run>/infer/<重みディレクトリ名>/<入力フォルダ名>/<実行時刻>/`）/ `bash start.sh tb`（TensorBoard を起動してブラウザを開く。train 時は自動）/ `bash start.sh shell` / `bash start.sh down`。世代別 `docker/compose.{gen30,gen50,cpu}.yaml`（gen40 は gen30 と共用）。計画: `docs/plans/20260905_docker-plan.md`
- run ディレクトリのレイアウト（正は `stage1/util/run_paths.py`、2026-09-07 第 2 版）: checkpoint は重みディレクトリ（`net_G.pth` / `net_D.pth` / `state.pth` の 3 ファイル 1 組）で、直下は `latest/` と `best/`（+ `best.txt`）だけ、epoch ごとは `weights/epoch_NNN/`。画像は `output_images/epoch_NNN/`（毎 epoch の checkpoint ごとのフル 512 の pcd / eidlike / R、**16bit**。学習中の 128 patch グリッドは TensorBoard だけで、ディスクには書かない）、推論は `infer/<重みディレクトリ名>/<入力フォルダ名>/<実行時刻>/{full,patch}/<case>/<slice>.png`（**16bit**）、`{full,patch}_dicom/…/<slice>.dcm`（元 DICOM のヘッダ継承。`stage1/util/dicom_io.py`）、`{full,patch}_R/`（残差 PNG、0 HU = 32768）。計画: `docs/plans/20260907_inference-plan.md`
