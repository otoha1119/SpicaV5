# Upstream

標準 CycleGAN（Zhu et al., ICCV 2017）と
`junyanz/pytorch-CycleGAN-and-pix2pix` の構成を基にしています。

- A: PCD512
- B: EID512
- G_A: PCD512 → EID-like512（推論用）
- G_B: EID512 → PCD-like512
- Loss: LSGAN + forward/backward cycle + identity
- D_A / D_B: 70×70 PatchGAN

Stage 1 の fidelity loss は使用しません。データ処理だけを Stage 1
`task: degrade` と揃えています。

## 全帯域2段Wavelet ResNet（2026-10-30）

- `models/networks.py`に`wavelet_resnet_9blocks`を追加。元の`resnet_9blocks`は比較用に残す
- stride-2 Conv ×2 / ConvTranspose ×2を、固定・正規直交Haar DWT ×2 / IDWT ×2に置換
- 各DWTは特徴量をLL/LH/HL/HHの`4C × H/2 × W/2`へ完全分解し、4帯域をまとめて学習可能3×3 Convへ入力する。LLだけを深層へ送らず、高周波の生skipも作らない
- encoderは`64 --DWT→ 256 --Conv→ 128 --DWT→ 512 --Conv→ 256`、中央はResNet Block ×9。decoderはConvで4帯域を生成してIDWTする対称構造
- 採用設定は`wavelet_resnet_9blocks`、cycle λ_A / λ_B = 20。`--netG resnet_9blocks --lambda_A 10 --lambda_B 10`で旧baselineを再現できる
- 学習の既定をBF16 autocast + CUDA上の`torch.compile(mode="max-autotune")`へ変更。重み・Adam stateはFP32のまま保存する。導入前checkpointは指定なしならFP32 / compileなしとして再開し、`--precision bf16 --compile true`を明示した場合だけ途中切替する
- `torch.compile` / Tritonが実行時に生成するCUDA補助モジュールのビルドに必要なC/C++コンパイラを、gen30 / gen50 Dockerイメージへ`build-essential`として追加
