# UPSTREAM

このディレクトリは DDIM 著者の公式コードを vendoring し、入力の channel 連結だけを足したもの。

- URL: https://github.com/ermongroup/ddim（MIT、`LICENSE`）
- commit: e8df66cf02e457719930748b5c26e37e3126dfdd（2024-07-26）
- 論文: Song, Meng, Ermon. "Denoising Diffusion Implicit Models", ICLR 2021（arXiv:2010.02502）
- 取得日: 2026-10-13
- 決定: `docs/decisions/20261013_pivot-stage2-diffusion.md`（決定 3・7）、調査: `docs/surveys/20261013_stage2-ddim.md`

条件付き化とタイル合成の書き方は次の公式コードに合わせた（コードの一部を写した。MIT、`LICENSE_WeatherDiffusion`）。

- URL: https://github.com/IGITUGraz/WeatherDiffusion（commit 5f78729、2023-02-23）
- 論文: Özdenizci & Legenstein, "Restoring Vision in Adverse Weather Conditions with Patch-Based Denoising Diffusion Models", TPAMI 2023
- WeatherDiffusion 自体が ermongroup/ddim を写して、条件の連結（`torch.cat([x_cond, x_t], dim=1)`）と重なりタイルの合成を足したもの

## 本家から写したファイル

| ファイル | 扱い |
|---|---|
| `models/diffusion.py` | **そのまま**（ε-net の U-Net。`assert x.shape[2] == x.shape[3] == self.resolution` も残す） |
| `models/ema.py` | **そのまま** |
| `functions/__init__.py` | **そのまま**（`get_optimizer`） |
| `functions/denoising.py` | 本家の関数（`compute_alpha` / `generalized_steps` / `ddpm_steps`）は**そのまま**で、末尾に SpicaV5 の関数を追記（下記） |
| `functions/losses.py` | 本家の `noise_estimation_loss` は**そのまま**で、末尾に条件付き版を追記（下記） |
| `runners/diffusion.py` | 本家を写して改変（下記）。`get_beta_schedule` と、本家 `main.py` の `dict2namespace` はそのまま |
| `configs/ddim.yml` | 本家 `configs/church.yml`（LSUN church 256）を写して改変（下記） |
| `runners/__init__.py`、`LICENSE` | そのまま |

使わないもの（写していない）: `datasets/`（CIFAR / CelebA / LSUN / FFHQ のローダ）、`functions/ckpt_util.py`（公開済み重みのダウンロード）、`main.py`（argparse。起動は `run_train.py` / `run_infer.py`）、他の `configs/*.yml`。
`models/__init__.py` は本家に無いが、別の `models` パッケージに取られないよう空ファイルを置いた。

## 本家からの変更履歴

### 2026-10-13 — 1 本目（決定 7: 本家そのまま + 3 点）

**1. batch**: `training.batch_size` 64 → 8（ユーザー決定）。

**2. 入力の channel 連結（唯一のアルゴリズム上の逸脱）**
- `configs/ddim.yml`: `data.channels` 3 → 1、`model.in_channels` 3 → 2、`model.out_ch` 3 → 1（mirthAI/Fast-DDPM と同じ形）
- `functions/losses.py` に `noise_estimation_loss_cond` を追記。本家の `noise_estimation_loss` との違いは `model(torch.cat([cond, x_t], 1), t)` だけ（x_t の作り方、`(C,H,W)` の和の batch 平均は同じ。WeatherDiffusion `models/ddm.py` と同じ形）
- `functions/denoising.py` に `generalized_steps_cond` を追記（WeatherDiffusion の `generalized_steps`。`'cuda'` 固定 → `x.device`）
- 条件（EID-like の同位置 patch）にはノイズを加えない。conditioning augmentation は入れない

**3. データ読み込み**
- LSUN 用のキー（`dataset` / `category` / `logit_transform` / `uniform_dequantization` / `gaussian_dequantization` / `rescaled` / `num_workers`）を外した
- 1ch 16bit PNG のペアを `data/pair_dataset.py`（stage2 から複製）で読む。HU 窓 [−1400, 4096] を [−1, 1] に線形写像（本家の `rescaled: true` と同じ値域）
- `data.random_flip` true → false（Stage 2 と同じく augmentation なし。キーは残し、true なら入力と教師を同じく左右反転する）

**タイル推論（推論・監視だけ。学習は本家どおり 256 patch）**
- `functions/denoising.py` に `generalized_steps_overlapping` と `overlapping_grid_indices` を追記（WeatherDiffusion）。1024 で 1 本の x_t を持ち、毎ステップ 256 のタイルを stride r で切り出して ε を予測し、画素ごとに平均してから全体で 1 回 DDIM 更新する
- WeatherDiffusion からの変更: `'cuda'` 固定 → `x.device`、`x_cond` に `data_transform` を掛けない（正規化済み）、タイルのバッチ数（本家 64 固定）を引数 `tile_batch` に、t をタイルのバッチと同じ長さに、batch 1 以外・被覆されない画素（`r > p` や `(H − p) % r ≠ 0`）はエラー

**`runners/diffusion.py`（本家 `Diffusion` を写して `CondDiffusion` に）**
- 本家と同じ: betas / logvar、Model を `DataParallel` で包む、Adam（`get_optimizer`）、EMA（`EMAHelper`）、1 step の中身（antithetic な t、loss → zero_grad → backward → grad_clip → step → EMA 更新）、サンプリングは EMA の重み、step 列（`skip_type` uniform | quad）、再開時に optimizer の eps を戻す
- 変えた: デバイスは呼び出し側が明示（`gpu_gen`。黙って cpu に落とさない）。保存は本家の `ckpt_{step}.pth`（1 ファイル）ではなく `model.pth` / `ema.pth` / `state.pth` の 1 組を原子的に（`util/run_paths.py`）。x_T は CPU の `torch.Generator`（seed とスライス名）で作ってからデバイスに送る
- 本家の FID / interpolation / sequence のサンプリングは持たない

**`configs/ddim.yml` のその他**
- `training.n_epochs` を外し `training.n_iters` で止める（本家 `train()` は `n_epochs` で止まり `n_iters` を見ない）。値は上限 1,000,000（本家 5,000,000）
- `training.validation_freq`（本家 2000。本家 `train()` は未使用）を、固定 val patch の ε の MSE を測る間隔として使う
- `sampling.*`（FID 用の `batch_size` / `last_only`）を外した。本家 `main.py` の `--timesteps` / `--eta` / `--skip_type` / `--seed` に当たる設定は `configs/train.yaml` の `log.*`（学習中の監視）と `configs/infer.yaml`（推論）
- `optim.grad_clip`: church.yml に無い（本家 `train()` は `config.optim.grad_clip` の AttributeError を try/except で握りつぶすので clip しない）→ `0.0` = clip しない、を明示

**SpicaV5 の部品（stage2 から複製。Stage 間で import しない）**
- `data/ct_io.py`、`util/metrics.py`、`util/panel.py` はそのまま
- `data/pair_dataset.py`、`util/run_paths.py`、`util/monitor.py`、`run_train.py`、`train.py`、`run_infer.py`、`inference_dir.py`、`mark_best.py` は stage2 を写して拡散モデルに合わせた（区切りは epoch ではなく本家と同じ step）
- `configs/schema.py`: 検証関数は stage2 から複製。int のリスト型（`ch_mult` / `attn_resolutions`）を追加
- `eval_nps.py`（新規）: 推論の出力で入力・出力・教師の NPS を同じ ROI で比べる（決定 7 の評価 (ii)）

### 2026-10-13 — 監視の区切り（ユーザー決定。学習の中身は変えない）
- `train.yaml data.samples_per_epoch`（patch 数。stage2 から写した 64000）を廃止し、`log.steps_per_epoch`（step 数、10000）に置き換えた。epoch は step から決まる（`step // steps_per_epoch + 1`）ので、epoch の途中から再開しても数え方は変わらない。廃止キーを持つ旧 run の再開は、そのキーを無視して今の `steps_per_epoch` を使う
- プログレスバーの単位を patch の枚数から step に変えた。1 本 = 1 epoch、左に `epoch e/E | step s/n_iters`
- 既定の間隔: `log.image_freq` 1000 → 500、`log.full_freq` 5000 → 10000、`log.save_weights_freq` 50000 → 10000 → **50000**（同日、ユーザー決定。100 万 step で 100 組 ≈ 170 GB → 20 組 ≈ 34 GB。`latest/` は 5,000 step ごとのまま）

### 2026-10-13 — 固定領域の画像（ユーザー指示。学習の中身は変えない）
- `train.yaml log.crop_pcd` / `log.crop_eid`（Stage 1 の `crop.yaml` の 4 か所を ×2 して 1024 の座標にしたもの。切り抜きは 72 → 144 画素）と `log.crop_freq`（500 step）で、学習中に固定領域を EMA の重みでサンプリングして TensorBoard `images/crop/<スライス名>` に出す
- **切り抜きの中心を保ち、周りを含めた今の patch の大きさ（256）の窓全体**を表示する（ユーザー指示: 「クロップの中心はそのままで周り領域を入れた今のパッチサイズに」）。窓は 1 枚のタイルで、合成（重なりの平均）は無い。窓が画像の端にかかるときは端に寄せる。x_T は領域ごとに固定（どの checkpoint でも同じ）
- PCD は [EID-like | PCD-like | PCD1024]、実 EID は [EID1024（×2 補間）| PCD-like]。PCD-002 は test の症例で、目視用にだけ読む（`full_slice` と同じ扱いを 2 枚目にも広げた）

### 動作確認（2026-10-13、合成データ・CPU・リポジトリ外の scratch venv、torch 2.x）
- 縮小設定（`ch 32`, `ch_mult [1, 2]`, `image_size 32`, 128 の合成画像, batch 4）で学習 6 step → `latest/` と `weights/iter_*` の保存、TB、固定 val patch と固定スライス・実 EID のタイル合成サンプル（16 タイル）まで通ることを確認
- `weights/iter_0000004` からの再開（`--n_iters 8`）が step 5 から続き、`launch_resume_*.yaml` を書くことを確認。`--ch 64` での再開は重みの形の不一致で、`n_iters` 到達済みの再開・未知フラグ・`r > p`・`T` を割り切れない S は起動前にエラーで止まることを確認
- 推論: 教師あり（`--cases` / `--slices_per_case` → `metrics.txt`、パネル 5 列）と、64 の実 EID 相当を ×2 補間して通す `--input eid`（η 0.5、r 16 = 49 タイル）が通ることを確認
- NPS: 白色雑音 σ = 10 で分散 100.3、radial NPS が平坦に約 100 HU²·px²（理論値 100）になることを確認
- 本番設定（`configs/ddim.yml`）の ε-net は 113.7M params。学習時の活性化は 1 サンプルあたり約 2.6 GiB（256、fp32、CPU で保存テンソルを数えた概算）で、batch 8 では活性化だけで約 21 GiB + 重み・EMA・Adam 1.7 GiB になる
