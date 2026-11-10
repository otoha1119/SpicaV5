# stage2 — 新しい Stage 2「U（形）+ V（質感）」（2026-11-11〜）

入力の EID512 を bicubic ×2 した絵 `x↑` から PCD1024 相当を作る。2 つのネットワークを順に学習する。

- **U**（形を決める回帰）: 旧 Stage 2 の U-Net そのまま。`μ = f + G∗(x↑ − f)`（`f = U(x↑)`、G = 明るさを平均する Gaussian）。損失は `MSE(μ, X′)`、`X′ = X + G∗(x↑ − X)`
- **V**（鋭さとノイズの粒を作るフロー）: PMRF 型の rectified flow。出発点 `Z0 = μ + σs·ε`、到達点 `X′`。V に渡すのは `Zt`・時刻 `t`・乱数の強さ `σs` の 3 つだけ（EID も μ も別には渡さない）。判別器なし
- **明るさの固定**: 広い平均（`configs/shared.yaml` の `brightness.sigma_px_512` = 16 → 1024 で σ 32 画素）は入力のものをそのまま使う。U の出力・学習の正解・最終出力の 3 か所に同じ計算を入れる
- 設計: `docs/plans/20261111_detail_design/stage2_u_flow_spec.md`、裁定 `RECONCILE.md`（`docs/` はローカルだけ）。式の出典・設計書からの差分: `UPSTREAM.md`

## 起動（ホストのシェルで、リポジトリ直下から）

前提: Stage 1 の `bash start.sh PC1 generate …` で疑似 EID の「組」（`<pseudo_eid_dir>/<組>/k00, k01, …` と `manifest.yaml`）ができていること。組には train・val の 16 症例の全スライスと、表示用の PCD-002 が要る。

```
bash start2.sh PC1 u --pseudo_eid_set <組>                         # U の学習（run = 起動時刻 yyyy_mmdd_HHMM、置き場 runs/stage2/u/<run>/）
bash start2.sh PC1 check --kind u --run <U の run> --what magnifier  # U だけの段階の確認（設計書 G-2。不成立なら V を作らない）
bash start2.sh PC1 v --pseudo_eid_set <組> --u_run <U の run>       # V の学習（U の best を固定。置き場 runs/stage2/v/<run>/）
bash start2.sh PC1 infer --run <V の run> --input_dir /workspace/DataSet/EID_v5 --cases EID-049     # 実 EID512 → 1024（outputs/stage2/…）
bash start2.sh PC1 check --kind v --run <V の run> --what magnifier,seeds,flicker                    # 虫眼鏡の数字・乱数を変えた出力・隣り合うスライス
bash start2.sh PC1 resume u <run> [latest|best|<step>] [--max_steps 60000]
bash start2.sh PC1 resume v <run> [latest|best|<step>] [--max_steps 200000 --max_hours 24]
bash start2.sh PC1 best v <run> <step|latest>                      # 目で選んだ重みを best/ に写す
bash start2.sh PC1 tb | build | shell | down
```

コンテナ `spicav5`（Stage 1 と同じ compose）が起動済みなら exec だけ行う（Stage 1 の学習を止めない）。コンテナの中では
`python stage2/run_train_u.py …` → 設定とディスクを検査 → `runs/stage2/u/<run>/launch.yaml` → `python stage2/train_u.py --run_dir /workspace/runs/stage2/u/<run> --launch …` を exec（V も同じ形）。

起動前に止まる検査: 組と `manifest.yaml` があること／manifest の明るさの広さが `configs/shared.yaml` と同じこと（違えばエラー。`--allow_brightness_mismatch true` で明示したときだけ通す）／
症例分割（ディスク上の症例はどれかのリストに入っていること）／表示用のスライスがあること／表示・確認用の実 EID が `configs/splits/eid_v5.yaml` の train 側でないこと／V は U の run と重みがあり、U と data・明るさの広さが同じこと。

## 設定（既定値は無い。欠けたキー・未知のキー・型違いは起動前にエラー。正は `configs/schema.py`）

| ファイル | 中身 |
|---|---|
| `configs/data.yaml` | U・V 共通: 疑似 EID の組（`--pseudo_eid_set`）、HU の物差し、補間（bicubic ×2）、`patches_per_slice`、症例分割、表示の範囲 |
| `configs/u.yaml` | U: `net.*`（旧 U-Net）、`optim.*`（Adam 3e-4、batch 8、4,000 step × 10 epoch、EMA 0.999、bf16、compile）、`patch.size` 256、`log.*`、`mag.*` |
| `configs/v.yaml` | V: `u.run` / `u.tag`、`net.*`（Haar + ResBlock U-Net 64 / 128 / 256 / 256）、`flow.*`（σs 25〜400 HU、表示は 100 HU・Euler 25 回）、`optim.*`（Adam 1e-4、warmup 1,000、clip 1.0、batch 16、15 万 step / 16 時間、EMA 0.9999）、`patch.size` 256 + `patch.margin` 96、`log.*`、`mag.*` |
| `configs/infer.yaml` | 推論: 重み、入力フォルダ、σs、Euler の回数、seed、保存するもの |
| `configs/check.yaml` | 確認の道具: やること、σs の一覧、seed の数、切り抜き、連続スライスの系列 |
| `../configs/shared.yaml` | 両 Stage 共有: `brightness.sigma_px_512`（Stage 2 からは上書きできない） |
| `../configs/machines.yaml` | `pcd1024_dir`、`eid_dir`、`pseudo_eid_dir`、`flow_checkpoints_dir`（= `/workspace/runs/stage2`）、`num_threads`、`stage2_tb_port` |

再開の基準は、選んだ重みの `state.pth` の中の実効設定（`--flag` だけが上に乗る）。`net.*`、HU の物差し、補間、`flow.unit_hu`、使う U は再開で変えられない。

## 出力

```
runs/stage2/u/<run>/   runs/stage2/v/<run>/
  launch.yaml, launch_resume_*.yaml   実効設定（V は使った U の run・重みも）
  dataset_info.yaml                   組と manifest の写し、組の数、パラメータ数、実測の速度
  loss_log.txt                        学習のログ（画面と同じ）
  latest/  best/(+best.txt)  weights/step_NNNNNNN/
        U: net_U.pth・net_U_ema.pth・state.pth    V: net_V.pth・net_V_ema.pth・net_U.pth（使った U の EMA の写し）・state.pth
        best は、上限（max_steps / max_hours）で止まったときの最後の EMA（rule: last。指標では選ばない）
  output_images/<印>/                 epoch 末の生の 16bit（stored = HU + 1400）。<印> = epoch_NNN、最初の確認は first_step_NNNNNNN
  preview/<名前>/<印>.png             epoch 末のパネル（8bit、等倍）
  check/<印>/magnifier.md             epoch 末の虫眼鏡の数字。確認コマンドの出力は check/<重み>_<時刻>/
  tb/                                 TensorBoard

outputs/stage2/<run>_<重み>_<入力名>_<時刻>/     推論
  full/<症例>/<スライス>.png          最終出力（1024、1ch 16bit）。full_seed01/… は 2 つ目以降の種
  full_mu/  full_input1024/  full_panel/  infer.yaml
```

## 途中経過の見方

- 止まり方: 上限で latest と best を保存して終了コード 0。SIGTERM は latest を保存して 143。NaN / Inf は保存せず 3
- 最初: scalar は 1, 2, 4, …, 64 step 目と以後 100 step ごと。21 step 目に `torch.compile` が始まる（数分。ログに 1 行出る）。200 step 目に latest の保存と確認の縮小版を 1 回通す
- scalar（TensorBoard / 監視ツール。確認用の数字に合格線は無い）:
  - U: `loss/mse`、`diag/train_rmse`、`val/rmse`・`val/rmse_input`（**入力そのままの値に近いほど正直**。best の選択には使わない）、`diag/leak_noise_explained`（0 に近い = 正解のノイズを当てていない）、`diag/var_gap`、`diag/lock_target_shift_hu`（Stage 1 の明るさの錨のずれ）、`diag/lock_f_shift_hu_<組>`、`diag/lock_out_shift_hu_<組>`
  - V: `loss/flow`、`loss/flow_t0〜t3`、`val/flow`、`val/rmse_mu`・`val/rmse_out`・`val/rmse_input`、`diag/leak_corr_fine`、`diag/lock_corr_rms_hu_<組>`、`diag/lock_out_shift_hu_<組>`、`diag/fake_seed_sd_hu`・`diag/fake_seed_low_ratio`・`diag/fake_ref_low_ratio`
  - 共通: `diag/sec_per_step`、`diag/imgs_per_sec`、`diag/data_time_frac`、`diag/lr`、`diag/grad_norm`、`diag/hours`、`diag/check_sec`、`diag/mag/<数字>_<段>_<組>`（数字 = fine_sd / edge / band1〜5 / shift_hu、段 = in / mu / out / ref、組 = pseudo / imagecas / hospital / pcd）
- 画像（TensorBoard の `images/…` と `preview/`）: `fixed`（U: [疑似 EID↑ | μ | PCD1024 | 実 EID → μ | 実 EID↑]、V: [疑似 EID↑ | μ | U+V | PCD1024 | 実 EID → μ | 実 EID → U+V | 実 EID↑]）、`eid/<スライス>`（病院 EID: [EID↑ | → μ | → U+V]）、`random/0`（val の別の 1 枚）、`crop_pcd`・`crop_eid`（切り抜き 256 四方 ×2）、`seed_diff`（V: [seed a | seed b | 差 | ぼかした差]）、`train_patch`
- 見る順（設計書 G）: U の段階で「μ が疑似 EID でも実 EID でも同じようにのっぺりしているか」（`diag/mag/fine_sd_mu_pseudo` と `…_imagecas`、`val/rmse` が `val/rmse_input` に近いか）。V の段階で「実 EID の出力が、疑似 EID の出力と入力のどちらに近いか」と、乱数の差に線・点の形が出ていないか
- **病院 EID の出力を見て重み・設定を選ばない**（評価専用）

## 精度と compile

- U の学習: ネットの forward だけ bf16 autocast + `torch.compile`。明るさの固定・損失は FP32
- V の学習: 本体（`net.body`）だけ bf16 + compile。入口（Haar・最初の Conv）、出口（LN・Conv・逆 Haar）、埋め込み、損失は FP32。固定した U の順伝播は `optim.u_precision`（bf16）+ `optim.u_compile`
- val・epoch 末の確認・推論・確認コマンド: compile しない EMA、FP32（推論・確認は `--precision bf16` で V の本体だけ bf16 にできる）
- 切り方: `--precision fp32`、`--compile false`、`--compile_mode default`、V は加えて `--u_precision fp32`、`--u_compile false`
