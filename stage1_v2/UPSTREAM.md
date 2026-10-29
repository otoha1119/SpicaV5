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
