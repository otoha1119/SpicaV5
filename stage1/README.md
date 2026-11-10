# stage1 — 新しい Stage 1「描き手 P」（2026-11-11〜）

実 EID だけで学ぶ拡散モデル。学習では「実 EID を粗くしてノイズを足した絵（手がかり）」から元の実 EID を描く練習をし、作るときは PCD512 を同じやり方で粗くした手がかりを渡して疑似 EID を描かせる。PCD は学習に見せない。判別器と全画素 fidelity は使わない。

- 設計: `docs/plans/20261111_detail_design/stage1_painter_spec.md`、裁定 `RECONCILE.md`（`docs/` はローカルだけ）
- 式の出典・設計書からの差分: `UPSTREAM.md`

## 起動（ホストのシェルで、リポジトリ直下から）

```
bash start.sh PC1                                   # 学習。run = 起動時刻 yyyy_mmdd_HHMM
bash start.sh PC1 --max_steps 2000 --compile false  # 上書き（schema にあるフラグだけ）
bash start.sh PC1 resume <run> [latest|best|<step>] [--max_steps 300000 --max_hours 48]
bash start.sh PC1 generate --run <run> [--tag latest] --blur_px 2 --noise_hu 50 [--k 2 --cases PCD-002 --slice_step 4]
bash start.sh PC1 best <run> <step|latest>          # 目で選んだ重みを best/ に写す
bash start.sh PC1 tb | build | shell | down
```

コンテナ `spicav5`（旧コードと同じ compose）が起動済みなら exec だけ行う。コンテナの中では
`python stage1/run_train.py …` → 設定を検証 → `runs/stage1/<run>/launch.yaml` → `python stage1/train.py --run_dir /workspace/runs/stage1/<run> --launch …` を exec。

## 設定（既定値は無い。欠けたキー・未知のキー・型違いは起動前にエラー。正は `configs/schema.py`）

| ファイル | 中身 |
|---|---|
| `configs/train.yaml` | `optim.*`（lr、batch、EMA、bf16、compile、上限 step / 時間）、`data.*`（パッチ、HU の変換、分割）、`cue.*`（手がかりのくじの幅）、`diffusion.*`（EDM の σ）、`sample.*`（途中経過の描画）、`log.*`（間隔、途中経過のスライス = コンテナ内の絶対パス） |
| `configs/model.yaml` | U-Net（幅 96 / 192 / 288 / 384、ResBlock、埋め込み、Haar 1 段）。再開で変えられない |
| `configs/generate.yaml` | 疑似 EID の作成: `generate.*`（run、重み、入力、症例、K、seed、間引き）、`cue.*`（σ_b、σ_c）、`sample.*` |
| `../configs/shared.yaml` | 両 Stage 共有: `brightness.sigma_px_512`（明るさを平均する広さ σ_a）。上書き `--brightness_sigma_px_512` |
| `../configs/machines.yaml` | `eid_dir`、`pcd_dir`、`painter_checkpoints_dir`（= `/workspace/runs/stage1`）、`pseudo_eid_dir`、`num_threads`、`tb_port` |

再開の基準は、選んだ重みの `state.pth` の中の実効設定（`--flag` だけが上に乗る）。

## 出力

```
runs/stage1/<run>/
  launch.yaml, launch_resume_*.yaml   実効設定
  dataset_info.yaml                   分割、枚数、測った平均・標準偏差、パラメータ数、実測の速度
  loss_log.txt                        学習のログ（画面と同じ）
  latest/  best/  weights/step_NNNNNNN/   net.pth・ema.pth・state.pth の 3 つ 1 組（.tmp → 差し替えの原子的な保存）
  output_images/step_NNNNNNN/         途中経過の生の 16bit（stored = HU + 1400）
  preview/<名前>/step_NNNNNNN.png     途中経過のパネル（8bit）
  tb/                                 TensorBoard

<pseudo_eid_dir>/<run>_step<N>_b<σ_b>_c<σ_c>_a<σ_a>/     疑似 EID の 1 組
  manifest.yaml                       重み、度合い、σ_a、段数、η、seed、K、入力、git の commit、枚数
  k00/<症例>/<スライス>.png, k01/…    512、1ch 16bit、入力と同じファイル名
```

## 途中経過の見方

- 止まり方: 上限（`optim.max_steps` / `optim.max_hours`）で latest を保存して終了コード 0。SIGTERM は latest を保存して 143。NaN / Inf は保存せず 3
- 最初: scalar は 1, 2, 4, …, 64 step 目と以後 100 step ごと。21 step 目に `torch.compile` が始まる（数分。ログに 1 行出る）。200 step 目に val・パネル・latest の保存を 1 回通す。以後 val とパネルは 2,000 step ごと
- scalar（TensorBoard / 監視ツール）:
  - `loss/train`、`loss_bin/0〜5`（σ の帯: 境 2 / 6 / 20 / 60 / 200 / 2,000 / 40,000 HU）、`loss_cue/none・clean・blur_only・noise_only・both`
  - `val/loss`、`val/loss_bin/0〜5`（val の固定パッチ 96 個、EMA、FP32）、`diag/bf16_val_ratio/bin0〜5`（同じ入力の bf16 ÷ FP32。bin0 が 1.2 を超えるなら `--fp32_top_level true`）
  - `diag/sec_per_step`、`diag/imgs_per_sec`、`diag/data_time_frac`、`diag/lr`、`diag/grad_norm`、`diag/hours`
  - `diag/check_redraw_wide_mean_diff_hu`・`_fine_sd_ratio`・`_fine_corr_src`（実 EID の描き直し）、`diag/check_pseudo_wide_mean_diff_hu`・`_fine_corr_pcd`・`_fine_sd_ratio_vs_eid`（PCD からの疑似 EID）。度合いごとは末尾に `/b2_c0` など。確認用で合格線は無い
- 画像（TensorBoard の `preview/…` と `runs/stage1/<run>/preview/`）: `train_patch`（[元 | 手がかり | ノイズ入り | x̂0]）、`eid_redraw/<スライス>`（行 = 度合い、列 = [実 EID | 手がかり | 描き直し | 差]）、`pcd_pseudo/<スライス>`（[PCD | 手がかり | 疑似 EID | 実 EID（参照）| 差]）、`zoom/…`、`nocue_256`（手がかり無し）。表示は HU −1400〜3000、差は白 = 0・青 −・赤 +

## 精度と compile

U-Net 本体（`net.body`）だけ bf16 autocast + `torch.compile`。入口（Haar・stem）、出口（head・逆 Haar・x̂0 の合成）、埋め込み、損失、ぼかし、錨は FP32。val は FP32、途中経過の描画と generate は compile しない EMA（本体だけ `sample.precision`）。
切り方: `--precision fp32`、`--compile false`、`--compile_mode default`。
