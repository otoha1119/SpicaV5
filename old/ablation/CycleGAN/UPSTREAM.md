# UPSTREAM — ablation/CycleGAN

## 目的と出典

- **普通の CycleGAN**（Zhu, Park, Isola, Efros, "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks," ICCV 2017）で A = **EID1024**（EID_v5 を ×2 bicubic 補間、`machines.yaml eid1024_dir`、Stage 1 `task: sr` と同じ）→ B = **PCD1024** の同解像度・非ペア変換。
- 経緯（2026-10-27）: `ablation/SRCycleGAN`（Zheng et al. 2022 の MMSR 損失あり）は downsample / upsample / SSIM 項が出力を EID 側に引き戻す拘束（スタイル制限）になるため、それを持たない素の CycleGAN を並べる（ユーザー決定）。SRCycleGAN はアブレーションとして残す
- コード: junyanz/pytorch-CycleGAN-and-pix2pix 本家（`stage1/` に vendoring 済み commit 2a7afba）の `models/cycle_gan_model.py` / `networks.py` / `util/image_pool.py`。`ablation/SRCycleGAN/` の起動・データ・表示・推論の骨組みを複製し、ネットワークと損失を本家 CycleGAN に置き換えた

## 本家との対応

| ここ | 本家 | 変更 |
|---|---|---|
| `models/networks.py` | `ResnetGenerator` / `ResnetBlock` / `NLayerDiscriminator` / `GANLoss` / `get_norm_layer` / `init_weights` / linear scheduler | 必要分だけ複製。`build_G`（resnet_9blocks \| resnet_6blocks、1ch → 1ch、dropout なし）/ `build_D`（basic 70×70） |
| `models/cycle_gan_model.py` | `cycle_gan_model.py` | BaseModel を外し、forward / backward_D_basic / backward_D_A / backward_D_B / backward_G / optimize_parameters を同じ構成で移植。λ_A / λ_B / λ_identity / pool_size / gan_mode は `configs/train.yaml`。保存・再開は SRCycleGAN と同じ |
| `models/image_pool.py` | `util/image_pool.py` | 複製 |
| `data/dataset.py` | — | SRCycleGAN 版（A / B 別寸法）を同解像度に戻した。`upsample()`（Stage 1 `ct_dataset.upsample` と同じ演算）を推論の 512 入力用に持つ |
| `run_train.py` | — | `eid1024_dir/manifest.yaml`（scale / interp / input_size）を読んで launch の `upsample` に記録。再開で方式が変わっていればエラー |
| `run_infer.py` / `inference_dir.py` | — | 入力が 1024 ならそのまま、512 なら launch の方式で補間して `full_input/` に残す。他はエラー。R = G_A(z) − z |

## 採用値（`configs/train.yaml`、本家の既定 = 論文）

| 項目 | 値 | 出典 |
|---|---|---|
| G | resnet_9blocks、ngf 64、InstanceNorm、dropout なし | 本家既定（論文 Sec. 7.2） |
| D | 70×70 PatchGAN、ndf 64 | 本家 basic |
| GAN 損失 | lsgan | 本家既定（論文 Sec. 4） |
| cycle | L1、λ_A = λ_B = 10 | 本家（論文 λ = 10） |
| identity | 0.5 | 本家既定（論文の一般目的関数には無い。0 で無し） |
| ImagePool | 50 | 本家（論文 Sec. 4） |
| optim | Adam lr 2e-4 β(0.5, 0.999)、100 epoch 一定 + 100 epoch 線形減衰 | 本家（論文 Sec. 4） |
| batch | **4**（本家・論文は 1。InstanceNorm なので batch に依らない。SRCycleGAN と揃えた） | ユーザー判断 |
| patch | 256（A / B 共通。本家 crop_size 256、`task: sr` と同じ解剖スケール） | — |
| 1 epoch | 10,000 patch、patches_per_slice 8 | SRCycleGAN と揃えた |
| 分割 | `configs/splits/eid_v5.yaml`（EID-049 は test）、B は全 20 症例 | `task: sr` と同じ |
| 正規化 | HU 窓 [−1400, 4096] → [−1, 1]（Tanh と一致） | Stage 1 と同じ |

## smoke（scratch venv torch 2.14 CPU + 合成データ EID1024 相当 128² × 6 症例（64² を bicubic ×2、manifest 付き）/ PCD1024 相当 128² × 3 症例、2026-10-27）

- 学習 2 epoch（batch 4、patch 32、resnet_6blocks、ngf 8）: 損失 8 本 + G_total + d_in_HU が有限、`latest/` に 5 ファイル、`output_images/epoch_00N/<slice>_{eid,pcdlike,R}.png`（全部 128）、`preview_fixed_*/`、`preview_random/`、TB の images/{current,fixed,cycle_A,cycle_B,full/fixed,full/random}
- resume: `--n_epochs_decay 2` で epoch 3 を継続、`--netG resnet_9blocks` は「重みの形」でエラー。`--scale` は未知フラグでエラー、`--gan_mode fgan_kl` はエラー。`mark_best` で best/
- 推論: 1024（128）入力はそのまま、512（64）入力は manifest の方式で補間して `full_input/` に残し、補間結果が EID1024 ファイルと画素一致、同じスライスの出力が 1024 入力のときと一致。`pcdlike == eid + (R − 32768)` が厳密に成立、学習の epoch 末フル画像（latest）と推論出力が画素一致。他の大きさはエラー
