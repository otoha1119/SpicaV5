# Stage 1 v3 — 射影型劣化器（確率的物理コア P + 学習残差 R + 実 EID 恒等項）

**2026-10-31〜 の既定は v4**（`configs/train.yaml` の `pipeline.mode: v4`。下の「v4 の流れ」）。v3（従来）は `--pipeline_mode v3 --dn_weights ""` で動く。旧 run の resume / infer はそのまま（v3 として扱う）。

Stage 1 と同じ `PCD512 → EID-like512` の非ペア劣化生成を、次の構成で作り直したもの（設計: `docs/surveys/20261030_stage1-projection-identity.md` 推薦 1 と同日の敵対的レビュー、要件: `docs/decisions/20261030_stage1-redesign-requirements.md` R-S21〜28）。

- **P（物理コア、学習なし）**: ぼかし σ_b [px] + NPS 整形ノイズ σ_n [HU]・相関 c [px]。スライスごとに `configs/train.yaml` の `degrade.*` の範囲から抽選（R-S23）。`degrade.py`
- **G = in + R(in)**: Park 2019 の wavelet G（`stage1/` と同じ構造）。R は決定的、乱数入力なし。内容依存の成分（HU 写像・blooming）だけを学ぶ（R-S25）
- **D**: Park 2019 の PatchGAN（受容野 22、R-S11）。本物 = 実 EID y、偽物 = u + R(u)（u = P(x)）
- **損失**: `L_GAN + λ_fid ‖R(u)‖² + λ_id ‖R(y)‖²`。恒等項は実 EID y を G に入れたときの残差（上の経路。D には見せない）。PCD 経路と EID 経路は 1 バッチに連結して G に通す（BatchNorm 対策）
- cycle なし（R-S12）。Stage 1 は Stage 2 の学習データ生成専用で、実運用は Stage 2 に実 EID を直接入れる

データは `configs/machines.yaml` の `pcd_dir` / `eid_dir`、症例列挙・`eval_only.yaml`・sampling・128 patch・HU 正規化は Stage 1 `task: degrade` と同じ。症例 split なし。

## 起動（ホストの sh から Docker 内で）

```bash
bash start1_v3.sh <machine>                                   # 学習（P + R + 恒等項）
bash start1_v3.sh <machine> --degrade false                   # 対照: P なし（fidelity GAN + 恒等項）
bash start1_v3.sh <machine> --lambda_id 0                     # 対照: 恒等項なし（P + fidelity GAN）
bash start1_v3.sh <machine> resume <run> [latest|best|epoch] [--flag value]
bash start1_v3.sh <machine> best <run> <epoch>
bash start1_v3.sh <machine> infer --weight_dir /workspace/stage1_v3/checkpoints/<run>/best --input_dir /workspace/DataSet/PCD512_v2 [--n_draws 8 --save_residual true]
bash start1_v3.sh <machine> infer --weight_dir ... --input_dir /workspace/DataSet/EID_v5 --apply_degrade false   # 実 EID で恒等の確認（R(y) の RMS）
bash start1_v3.sh <machine> ponly --input_dir ... --degrade_mode gaussian|bilateral_blur [--blur_max 1.5 ...]  # 固定 P 単独（learned は重みが要るので infer --apply_degrade true）
bash start1_v3.sh <machine> stats --dirs /workspace/DataSet/EID_v5,/workspace/DataSet/PCD512_v2,<ponly の full>  # R-S22 の計測
bash start1_v3.sh <machine> tb
```

新規 run は既定で BF16 autocast と CUDA 上の `torch.compile(mode="max-autotune")` を使う（008 と同じ。重みと optimizer state は FP32、checkpoint 形式は同じ）。初回 iteration はコンパイルで長い。従来条件は `--precision fp32 --compile false`。compile には build-essential 入りの image が要るので、古い image のマシンは `down → build`。

sh の `--flag` は `configs/schema.py` にあるものだけ（学習は TRAIN と MACHINE、推論は INFER）。実効値は `<run>/launch.yaml`。再開で `network.*` と `degrade.enabled` は変えられない。

## 着手順（レビュー I）

0. `stats` で実 EID と PCD512 の局所 SD・NPS（f_peak / f_av）・エッジ勾配・HU 分位を測り、`degrade.*` の範囲をその実測を含むように決める（今の `train.yaml` は実測前の仮置き）
1. `ponly` で P だけの EID-like を作り、同じ `stats` で実 EID と並べる（R-S22 の校正）。Stage 2 プローブ（`bash start2.sh dataset` → U-Net）で実 EID 入力の出力を R-S27 の指標で比べる
2. 通らなければ P + R（学習）。判定は R-S22 の 3 軸、`diag/R_ratio`（RMS(R(u))/RMS(R(y))。≫ 1 かつ D 分離維持 = R がドメイン検出器化）、`draws/std_ratio`（< 1 = R が P のノイズを消す denoise）、`draws/corr_dR_du`、`images/full/identity`。事前登録 `docs/experiments/EXP-*`

## v4 の流れ（pipeline.mode v4、docs/decisions/20261031_stage1-v3-noise-path-analysis.md §3・§6、ブランチ 012）

v3 の行き詰まり（ぼかした PCD ノイズが「指紋」として残り、fidelity が σ を 0 に押す）を順序の入れ替えで潰す。

- 経路: `x → Dn（凍結 denoiser）→ x_c → G: x_c + R(x_c) → ⊛ k(z) → + σ(x_c) ⊙ (h(z) ⊛ ε)、σ ∈ [σ_min, σ_max] → u`。D は実 EID y と u。恒等経路 `y → G` は v3 と同じ（Dn も k もノイズも通さない）
- 損失: `L_GAN(u) + λ_fid ‖R(x_c)‖² + λ_id ‖R(y)‖² + λ_k ‖x_c ⊛ k − x_c‖² + λ_com COM²`。ε に依存する項は L_GAN だけ（C2）。σ の下限 `degrade.learned_sigma_min_hu`（実 EID の局所 SD 22.7、C3）
- Dn は PCD512 だけで Neighbor2Neighbor（Huang et al., CVPR 2021）を事前学習して凍結する（`models/denoiser.py`、`train_denoiser.py`、`configs/denoise.yaml`）。GAN の勾配は入れない（C1）。セル幅 `n2n.cell_size` と最小距離 `n2n.min_dist` は `stats` の `noise_corr_d0`（|ρ(d)| < 0.05 になる最小距離）から決める
- 合否（`check_denoiser.py`、§6.4 の表）: Dn(x) の一様 ROI 局所 SD ≤ 6 HU、edge_grad p90/p95/p99 ≥ 0.8 × x、残差 x − Dn(x) の edge_grad p99 ≤ 残差 SD の 3 倍。閾値は `configs/denoise.yaml` の `check.*`。終了コード 0 = 全項目 PASS
- 判定（v4 の学習中）: `val/fingerprint_sd_HU` = ε = 0 で生成した u の局所 SD の中央値（score に `/ val.target_fingerprint_sd_hu` で加算）、`draws/std_out_body_HU`（u の引き直し間 SD）、`draws/std_v_body_HU`（ε = 0、z だけ引き直し）、`draws/std_eps_body_HU`（ノイズだけ）、`diag/Dn_resid_rms_HU`。パネルは `[x | Dn(x) | G(x_c) | u | R(x_c)]`、`images/full/eps0` が指紋

```bash
bash start1_v3.sh <machine> stats --dirs /workspace/DataSet/PCD512_v2      # noise_rho_1..4 と noise_corr_d0 → denoise.yaml の n2n.min_dist / cell_size = 2·d₀
bash start1_v3.sh <machine> v4 [--flag value]                               # 通し: dn → dncheck（FAIL なら停止）→ 学習（--pipeline_mode v4 --dn_weights <dn の latest>）。--flag は学習にだけ渡す
bash start1_v3.sh <machine> dn [--n_epochs 50 --cell_size 4 --min_dist 2]   # Dn の学習だけ（configs/denoise.yaml、schema の DENOISE）。resume は無い（毎回新規）
bash start1_v3.sh <machine> dncheck [--weights PATH] [--input_dir DIR] [--n_slices N]   # Dn の合否。--weights 省略時は train.yaml の pipeline.dn_weights
bash start1_v3.sh <machine> --dn_weights /workspace/stage1_v3/checkpoints/denoiser/<run>/latest/net_Dn.pth   # 学習だけ（train.yaml が v4 なので dn_weights を選ぶ）
bash start1_v3.sh <machine> --pipeline_mode v3 --dn_weights ""              # 従来の v3
bash start1_v3.sh <machine> infer --weight_dir .../best --input_dir /workspace/DataSet/PCD512_v2 --n_draws 8   # v4: full/ = u、full_eps0/ = 指紋（infer.eps_zero）、full_std/、manifest.yaml に G / P / Dn の sha256
```

出力: Dn は `stage1_v3/checkpoints/denoiser/<run>/`（`latest/net_Dn.pth`、`weights/epoch_NNN/`、`tb/`、`output_images/epoch_NNN/<slice>_panel.png` = [x | Dn(x) | 残差]）と、最新 run の複製 `checkpoints/denoiser/latest/`（`train.yaml` の `pipeline.dn_weights` の既定）。v4 の run は Dn を `<run>/net_Dn.pth` と各重みディレクトリに複製する（resume / infer / best はそちらを優先）。`dncheck` は `output/stage1_v3/dncheck_<時刻>/summary.yaml`。
TensorBoard（`tb`）: Dn は `loss/rec, loss/reg, loss/total, n2n/gamma, lr, check/*`（§6.4 の指標を固定スライスで毎 epoch）。
設定キー: `pipeline.mode` / `pipeline.dn_weights`（resume で mode は変えられない）、`degrade.learned_sigma_min_hu`、`val.target_fingerprint_sd_hu`、`infer.yaml` の `eps_zero`、`configs/denoise.yaml`（`data.* / n2n.* / model.* / optim.* / log.* / check.*`、すべて schema.py の DENOISE）。

## 自動採点と best

epoch ごとに固定の PCD 32 枚（P → G）と実 EID 128 枚で R-S22 の指標（local_sd, nps_f_av, edge_grad_p90, edge_grad_p99, hu_p50, hu_p90）を測り、`score = Σ |median_gen − median_eid| / IQR_eid + RMS(R(y))_eval / 20 HU` が最良の epoch を `best/` に自動保存する（`val.auto_best`、`best.txt` に内訳、TB `val/*`、`val_log.txt`）。GAN は最終 epoch が最良とは限らない（1 本目で HU 水準が epoch ごとに揺れた）。手動の `best <run> <epoch>` で上書きできる。

D は本物と偽物を 1 バッチに連結して通す（2 本目から。1 本目は別 forward で、D の 1 層目の BN が偽物バッチの HU 水準差を消して R の HU 水準が揺れた）。

## 出力

checkpoint は `stage1_v3/checkpoints/<run>/`（レイアウトは `util/run_paths.py`）。推論は `output/stage1_v3/`。推論用の重みは `net_G.pth`。
epoch 末の preview: `[EID代表 | EID-like u+R(u) | P(x) | PCD | R(u)]`、恒等経路 `[y | y+R(y) | R(y)]`、固定スライスの n_draws 標準偏差画像（`*_std.png`、HU）。
TensorBoard: `loss/*`、`diag/*`（R_u_rms_HU, R_y_rms_HU, R_ratio, P_rms_HU, d_in_HU, lambda_id）、`draws/*`（std_out_body_HU, std_ratio, corr_dR_du, R_y_full_rms_HU）。

## 煙試験

リポジトリ外の scratch venv と合成データで行う（テストコードはリポジトリに置かない）。`SPICA_V3_CHECKPOINTS_ROOT` で checkpoint の置き場を scratch に向けられる（通常運用では設定しない）。
