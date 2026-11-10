# UPSTREAM — ablation/SRCycleGAN

## 移植元

- 論文: T. Zheng, H. Oda, Y. Hayashi, S. Nakamura, M. Mori, H. Takabatake, H. Natori, M. Oda, K. Mori, "SR-CycleGAN: super-resolution of clinical CT to micro-CT level with multi-modality super-resolution loss," *J. Med. Imaging* 9(2), 024003 (2022). DOI 10.1117/1.JMI.9.2.024003（PMC8983071）。会議版: SPIE Medical Imaging 2020, arXiv:1912.12838
- コード: 著者公開コード（junyanz pytorch-CycleGAN-and-pix2pix 派生）。手元の正は **SpicaV2 リポジトリの初回クローン commit `741924c`**（2025-08-12 "SR-CycleGANののクローンを行いました"）。SpicaV2 のそれ以降の改変（MMSR 損失の無効化、InstanceNorm 化、×2 化、識別子の変更）は使わない
- 取得日: 2026-10-27。計画: `docs/plans/20261027_stage1-sr-cyclegan-plan.md`

## 移植したもの（741924c → ここ）

| ここ | 741924c | 変更 |
|---|---|---|
| `models/networks.py` `Clinical2MicroGenerator` | `models/networks.py` 1066〜1136 行（論文 G1） | `upsampling_times` を `scale` の log2 から決める。`print` を削除。層の並び・`self.model` の index は同一（**scale 8 で state_dict のキー・形が 741924c と完全一致、同じ重みで出力 max 差 0.0 を確認**） |
| 同 `Micro2ClinicalGenerator` | 1478〜1544 行（論文 G2） | `downsampling_times` を `scale` から決める。同上（一致確認済み） |
| 同 `ResnetBlock` / `NLayerDiscriminator` / `GANLoss` / `get_norm_layer` / `init_weights` / linear scheduler | junyanz 本家（741924c 502 / 756 / 275 行など） | `NLayerDiscriminator` の "SpectralNormalization" の print を削除（本家と同一の構造。一致確認済み）。scheduler は関数に |
| `models/ssim.py` | `pytorch_ssim/__init__.py` | C1 / C2 を引数に（741924c は 0.01² / 0.03² 固定）。window を buffer に |
| `models/image_pool.py` | `util/image_pool.py` | 複製（`stage1/util/image_pool.py` 経由） |
| `models/sr_cyclegan_model.py` | `models/medical_cycle_gan_model.py` | junyanz BaseModel を外し、`forward / backward_D_basic / backward_D_A / backward_D_B / backward_G / optimize_parameters` の構成を保って移植。`AvgPool2d(8)` / `UpsamplingNearest2d(8)` → `scale`。損失の重み・形を設定キーに（下記）。`criterionCycle` で計算していた downsample / upsample 損失を論文どおり MSE に。保存・再開は Stage 1 `fidelity_gan_model.py` の流儀 |

複製したもの（SpicaV5 内）: `configs/schema.py` の検証関数（`stage2/configs/schema.py`）、`data/dataset.py` の I/O・`CaseIndex`・`RandomSampler`（`stage1/data/ct_dataset.py`。A / B 別 grid に改変）、`util/run_paths.py` / `util/monitor.py` / `util/panel.py` / `util/residual_color.py`（`stage1/util/`）、`run_train.py` / `train.py` / `run_infer.py` / `inference_dir.py` / `mark_best.py`（Stage 1 / 2 の流儀）。

## 論文とコードの差と採用値（ユーザー決定 2026-10-27: **論文の値**。コードの値は設定キーで選べる）

| 項目 | 論文 | コード 741924c | 採用（`configs/train.yaml`） |
|---|---|---|---|
| cycle loss | squared L2（JMI Eq. 1） | L1 | `loss.cycle_norm: l2` |
| λ1 / λ2（SSIM G1 / G2） | 1.0 / 1.0（JMI Table 1。SPIE 版 λ1 = 2.0） | 1.0 / 1.0 | 1.0 / 1.0 |
| λ3 / λ4（down / up） | 0.7 / 0.3（JMI Table 1） | 1.0 / 1.0 | 0.7 / 0.3 |
| λ5 / λ6 / λ7（GAN ×2 / cycle） | 1.0 | 1.0 | 1.0 |
| identity loss | 無し（JMI Eq. 8） | 0.2（idt_B はさらに ×0.1、bilinear ×8） | `lambda_identity: 0.0`（有効化時の f↑ は nearest に統一） |
| GAN 損失 | log 形（JMI Eq. 2） | lsgan | `gan_mode: vanilla`（BCEWithLogits） |
| 正規化層 | BN（JMI Sec. 2.3.1） | instance | `norm: batch` |
| SSIM 定数 | SPIE: C1 = 0.02, C2 = 0.06（JMI 未記載） | 0.01², 0.03² | 0.02 / 0.06（window 11、σ 1.5 はコード） |
| D | "CNN-based"（未記載） | basic 70×70 PatchGAN + ImagePool 50 | コード |
| 倍率 / patch | ×8、32 / 256 | 同 | **×2、128 / 256**（HR 側の画素数は論文と同じ。LR は ×2 の関係） |
| 学習 | batch 4、Adam lr 1e-5（100 epoch 一定 + 100 epoch 線形減衰）、2000 patch × 5 症例 = 10,000/epoch | 同 | 同（`samples_per_epoch 10000`、`patches_per_slice 8`） |
| 正規化 | モダリティごと min-max → [−1, 1] | 同 | HU 窓 [−1400, 4096] → [−1, 1]（Stage 1 と同じ。Tanh と一致） |
| データ | 肺マスク内の patch。コードは 4 回に 1 回 µCT を縮めた自己ペアを混ぜる（論文に無い） | — | マスク無し、自己ペア無し |
| 推論 | n×n patch → 8n×8n を非重複タイル | 同 | 512 を丸ごと 1 回で 1024（G1 は全畳み込み） |

## 本研究での設計（§計画書 3 / 4）

- A = `machines.yaml eid_dir`（実 EID512、`configs/splits/eid_v5.yaml` の train 症例。val はプレビュー、test は読まない）、B = `pcd1024_dir`（PCD1024、`b_exclude_cases []`）。起動時に B の一辺 = A の一辺 × scale を検査
- patch A 128 / B = A × scale = 256（B の patch は yaml に書かない）。grid は A / B 別。非ペア（症例・スライス・位置とも独立）
- checkpoint: `net_G_A.pth`（G1、推論用）/ `net_G_B.pth` / `net_D_A.pth` / `net_D_B.pth` / `state.pth`。レイアウト `util/run_paths.py`
- 表示: R = G1(z) − nearest↑(z)、`diag/d_in_HU` = mean|f↓(G1(z)) − z| の HU。TB に `images/cycle_A` / `cycle_B`（forward / backward cycle）
- 推論: `full` のみ（512 → 1024）。patch 推論・DICOM・crop は未対応（エラーで止める）
- `machines.yaml` に `ablation_checkpoints_dir` / `ablation_tb_port`（6008）を追加。`stage1/configs/schema.py` / `stage2/configs/schema.py` の MACHINE に flag None で宣言、`docker/compose.*.yaml` に 3 本目のポート、`start.sh` / `start2.sh` が `TB_PORT_ABLATION` を export（compose が `:?` で要求するため）

## smoke（scratch venv torch 2.14 CPU + 合成データ EID 64² × 6 症例 / PCD 128² × 3 症例、2026-10-27）

- networks: G1 128 → 256、G2 256 → 128、D(256) → 30²、G1 512 → 1024。SSIM 同一画像で 1.0。**scale 8 で G1 / G2 / D の state_dict が 741924c と一致、同じ重みで出力差 0.0**
- 学習 2 epoch（batch 4、patch 32 / 64、n_blocks 2、ngf 8）: 全損失が有限、`latest/` + `weights/epoch_00N/` に 5 ファイル、`output_images/epoch_00N/<slice>_{eid,pcdlike,R}.png`（64 / 128 / 128）、`preview_fixed_EID-006-003/`（00_eid、01_pcd、epoch_NNN_{pcdlike,R_color,panel}）、`preview_random/`、TB の scalar 18 本と images/{current,fixed,cycle_A,cycle_B,full/fixed,full/random}
- resume: 完了済み run の再開はエラー、`--scale 4` は「重みの形」でエラー、`--n_epochs_decay 2` で epoch 3 を継続（state.pth の config 基準、lr 復元）。`mark_best` で best/ + best.txt
- エラー系: `--gan_mode fgan_kl`、未知フラグ `--lambda_fid`、`--patch_size 30`（4 の倍数でない）
- 推論: 30 枚 64 → 128、`pcdlike == up(eid) + (R − 32768)` が厳密に成立、学習の epoch 末フル画像（latest）と推論出力が画素一致、`full_panel/`、128 入力は「一辺 64 のみ」でエラー
- `start_ablation.sh` の dry-run（偽 docker）: `mac SRCycleGAN --n_epochs 1` → `exec -e SPICA_TB_PORT=6008 ... bash ablation/SRCycleGAN/train.sh --n_epochs 1`、`infer` → `infer.sh`、アブレーション名無し・2 つ目のマシン名はエラー
- Stage 1 / Stage 2 / ablation の 3 つの schema で `machines.yaml` の全マシン（PC1〜4、mac）が通る
