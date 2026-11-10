# SpicaV5

Photon-counting CT（PCD-CT）と従来型 CT（EID-CT）の画像を扱う研究コード。実 EID512 から PCD1024 相当の画像を得ることを目標に、2 段構成（Stage 1 / Stage 2）で学習する。

2026-11-11 に手法を全面的に作り直すことを決めた（案 E）。それまでの実装は `old/` にまとめてあり、リポジトリ直下は新しい Stage 1 / Stage 2 のための場所。

## 1. 現在の方針（2026-11-11、案 E）

- **Stage 1**: 実 EID だけで学んだ拡散モデルに描き直させる（PCD を粗くした絵を手がかりに EID を描く。PCD は Stage 1 の学習に見せない）
- **Stage 2**: U-Net（形）+ フロー（鋭さとノイズの粒）
- 決定記録 `docs/decisions/20261111_pivot-redraw-stage1-create-stage2.md`、設計 `docs/plans/20261111_solution_proposals/E_final_proposal.html`、要件の正本 `docs/reference/requirements_v3_manual.md`（`docs/` は git 管理外でローカルにだけある）

## 2. リポジトリ構成

| 場所 | 内容 |
|---|---|
| `stage1/` | **新しい Stage 1（描き手 P: 実 EID だけで学ぶ拡散モデル）**。ブランチ `028-f_main-redraw_create` で実装（設定 `stage1/configs/`、説明 `stage1/README.md`、式の出典 `stage1/UPSTREAM.md`） |
| `stage2/` | **新しい Stage 2（U = 形を決める回帰 + V = 鋭さとノイズの粒を作るフロー）**。同じブランチで実装（設定 `stage2/configs/`、説明 `stage2/README.md`、式の出典 `stage2/UPSTREAM.md`） |
| `runs/` | 新しい学習結果の置き場（`runs/stage1/`・`runs/stage2/`、コンテナ内 `/workspace/runs/...`。git 管理外） |
| `old/` | 2026-11-11 以前の実装: `old/stage1/`（FE-GAN、junyanz 基盤）、`old/stage2/`（U-Net 回帰）、`old/ablation/`（SRCycleGAN / CycleGAN）と起動 sh。説明は `old/README.md`、詳しい仕様は `old/README_main_20261111.md`。学習結果・推論出力も `old/` 配下（`old/stage1/checkpoints`・`old/stage2/checkpoints`・`old/ablation/checkpoints`・`old/output`、git 管理外） |
| `configs/` | 共有の設定: `machines.yaml`（マシン定義: GPU 世代、データのパス、学習結果の置き場、TensorBoard のポート）、`shared.yaml`（両 Stage で共有する値: 明るさを平均する広さ）、`eval_only.yaml`（評価専用症例）、`splits/`（症例分割） |
| `docker/` | 世代別の compose / Dockerfile / requirements（gen30 = RTX 30・40 系、gen50 = RTX 50 系、cpu = mac）。コンテナ名 `spicav5`、リポジトリは `/workspace` にマウント |
| `create_dataset/` | PCD1024 データの変換・照合スクリプト |
| `watch/` | 学習の監視（train-watch。`watch/README.md`） |
| `skills/` | research-skills |
| `start.sh` | 新しい Stage 1 の起動口（学習 / resume / generate / best / tb / build / shell / down） |
| `start2.sh` | 新しい Stage 2 の起動口（u / v / resume / infer / check / best / tb / build / shell / down） |

## 3. 起動

- 新しい Stage 1: `bash start.sh [マシン名] [build|shell|down|tb|generate] [resume <run> [tag]] [best <run> <tag>] [--flag value]`（サブコマンド無し = 学習。学習結果は `runs/stage1/<run>/`、疑似 EID は `machines.yaml` の `pseudo_eid_dir`。詳しくは `stage1/README.md`）
- 新しい Stage 2: `bash start2.sh [マシン名] [u|v|infer|check|tb|shell|down|build] [resume <u|v> <run> [tag]] [best <u|v> <run> <tag>] [--flag value]`（u = U の学習、v = V の学習。学習結果は `runs/stage2/u/<run>/`・`runs/stage2/v/<run>/`、推論の出力は `outputs/stage2/`。Stage 1 の疑似 EID の「組」を `--pseudo_eid_set` で渡す。詳しくは `stage2/README.md`）
- 旧版（リポジトリ直下から、ホストのシェルで）:
  - `bash old/start.sh [マシン名] [build|shell|down|tb|infer|crop|split|chain] [resume <run> [tag]] [best <run> <epoch>] [--flag value]`（旧 Stage 1）
  - `bash old/start2.sh [マシン名] [dataset|infer|tb|build|shell|down] ...`（旧 Stage 2）
  - `bash old/start_ablation.sh [マシン名] <名前> [infer|tb|shell|down|build] ...`（旧 ablation）
- 全環境 Docker 内で実行する。コンテナ `spicav5` が起動済みなら up を呼ばず exec だけ行う（中の処理を止めない。止めるのは down だけ）

## 4. 実装の決まり

### 4.1 bf16 autocast と `torch.compile`（研究者の指示 2026-11-11）

新しく学習コードを作るときは、**最初から** 学習の forward に bf16 の autocast と `torch.compile` を入れる。

- 旧 Stage 2 の `optim.precision: bf16` / `optim.compile: true` と同じ考え方（実装例 `old/stage2/models/regression_model.py`、設定の宣言 `old/stage2/configs/schema.py`）
- **損失・検証・推論は FP32**（forward の出力を FP32 に戻してから損失を計算する。検証・推論は compile しない同じ重みで通す）
- どちらも設定で切れるようにする（`fp32` / compile なしで従来どおり動くこと）
- `torch.compile` には C コンパイラが要る（gen30 / gen50 の Dockerfile は `build-essential` 入り。cpu の Dockerfile には無い）

### 4.2 そのほか

- **既定値・フォールバック禁止**: 設定は yaml + schema（宣言したキーは必須、未知キー・型違いはエラー）。sh の `--flag` が最優先。実効値を run の `launch.yaml` に保存し、再開は checkpoint の `state.pth` 内の実効設定を基準にする
- **テストコードはリポジトリに置かない**（リポジトリ外の scratch venv / 合成データでの smoke 実行は行う）
- checkpoint は原子的に保存する（`.tmp` に書いてから差し替え。Windows のバインドマウントでは名前変更が PermissionError になることがあるので、旧実装の `_swap_dirs` と同じ対策を入れる）
- 評価専用症例（`configs/eval_only.yaml`）は学習・検証・症例分割に使わない

## 5. データ規約

- uint16 1ch PNG、`stored = HU + 1400`（EID・PCD 共通）。ディレクトリは `<root>/<症例>/<slice>.png`
- 表示の正規化は HU −1400〜3000（stored 0〜4400）
- 患者由来の画像（DICOM・PNG）と学習結果はリポジトリに入れない（`.gitignore` で `DataSet/`・`runs/`・`old/*/checkpoints/`・`*.pth`・`*.dcm` 等を除外）
