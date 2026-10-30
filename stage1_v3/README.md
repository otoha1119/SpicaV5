# Stage 1 v3 — 射影型劣化器（確率的物理コア P + 学習残差 R + 実 EID 恒等項）

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
bash start1_v3.sh <machine> ponly --input_dir /workspace/DataSet/PCD512_v2 [--max_slices 50] [--blur_max 1.5 ...]  # P 単独（学習なし）
bash start1_v3.sh <machine> stats --dirs /workspace/DataSet/EID_v5,/workspace/DataSet/PCD512_v2,<ponly の full>  # R-S22 の計測
bash start1_v3.sh <machine> tb
```

新規 run は既定で BF16 autocast と CUDA 上の `torch.compile(mode="max-autotune")` を使う（008 と同じ。重みと optimizer state は FP32、checkpoint 形式は同じ）。初回 iteration はコンパイルで長い。従来条件は `--precision fp32 --compile false`。compile には build-essential 入りの image が要るので、古い image のマシンは `down → build`。

sh の `--flag` は `configs/schema.py` にあるものだけ（学習は TRAIN と MACHINE、推論は INFER）。実効値は `<run>/launch.yaml`。再開で `network.*` と `degrade.enabled` は変えられない。

## 着手順（レビュー I）

0. `stats` で実 EID と PCD512 の局所 SD・NPS（f_peak / f_av）・エッジ勾配・HU 分位を測り、`degrade.*` の範囲をその実測を含むように決める（今の `train.yaml` は実測前の仮置き）
1. `ponly` で P だけの EID-like を作り、同じ `stats` で実 EID と並べる（R-S22 の校正）。Stage 2 プローブ（`bash start2.sh dataset` → U-Net）で実 EID 入力の出力を R-S27 の指標で比べる
2. 通らなければ P + R（学習）。判定は R-S22 の 3 軸、`diag/R_ratio`（RMS(R(u))/RMS(R(y))。≫ 1 かつ D 分離維持 = R がドメイン検出器化）、`draws/std_ratio`（< 1 = R が P のノイズを消す denoise）、`draws/corr_dR_du`、`images/full/identity`。事前登録 `docs/experiments/EXP-*`

## 出力

checkpoint は `stage1_v3/checkpoints/<run>/`（レイアウトは `util/run_paths.py`）。推論は `output/stage1_v3/`。推論用の重みは `net_G.pth`。
epoch 末の preview: `[EID代表 | EID-like u+R(u) | P(x) | PCD | R(u)]`、恒等経路 `[y | y+R(y) | R(y)]`、固定スライスの n_draws 標準偏差画像（`*_std.png`、HU）。
TensorBoard: `loss/*`、`diag/*`（R_u_rms_HU, R_y_rms_HU, R_ratio, P_rms_HU, d_in_HU, lambda_id）、`draws/*`（std_out_body_HU, std_ratio, corr_dR_du, R_y_full_rms_HU）。

## 煙試験

リポジトリ外の scratch venv と合成データで行う（テストコードはリポジトリに置かない）。`SPICA_V3_CHECKPOINTS_ROOT` で checkpoint の置き場を scratch に向けられる（通常運用では設定しない）。
