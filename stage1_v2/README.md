# Stage 1 v2 — CycleGAN

Stage 1 と同じ `PCD512 → EID-like512` を標準 CycleGAN で学習します。
元の Stage 1 が一方向 GAN + fidelity loss なのに対し、v2 は
`G_A: PCD→EID` と `G_B: EID→PCD` の cycle consistency を使います。

データは `configs/machines.yaml` の既存 `pcd_dir` / `eid_dir` を使い、
症例列挙、`eval_only.yaml`、random / paper sampling、128 patch、HU正規化は
Stage 1 `task: degrade` と同じです。degrade と同じく症例 split はありません。

既定のGeneratorは、2段Haar DWT/IDWTでLL/LH/HL/HHを全てResNet bodyへ通す
`wavelet_resnet_9blocks`です。高周波の生skipは持ちません。cycle係数は両方向20です。
旧baselineは`--netG resnet_9blocks --lambda_A 10 --lambda_B 10`で実行できます。

新規runは既定でBF16 autocastと、CUDA上の4ネットワークに対する
`torch.compile(mode="max-autotune")`を使います。重みとAdam stateはFP32で保存するため、
checkpoint形式は従来と同じです。初回iterationはコンパイルのため通常より長くかかります。
成功したautotune候補一覧は表示せず、compilerの警告とエラーだけを端末に残します。
従来条件は`--precision fp32 --compile false`で実行できます。

```bash
bash start1_v2.sh <machine>
bash start1_v2.sh <machine> resume <run> [latest|best|epoch]
bash start1_v2.sh <machine> resume <run> latest --precision bf16 --compile true
bash start1_v2.sh <machine> best <run> <epoch>
bash start1_v2.sh <machine> infer --weight_dir /workspace/stage1_v2/checkpoints/<run>/best --input_dir /workspace/DataSet/PCD512_v2
bash start1_v2.sh <machine> tb
```

起動中のPythonプロセスにはコード変更が反映されません。旧runを高速設定へ切り替える場合は
checkpoint保存後に停止し、pull後、上記の明示指定付きresumeを使います。指定なしのresumeは
そのrunの条件を維持し、導入前のrunならFP32・compileなしで再開します。

checkpoint は `stage1_v2/checkpoints/<run>/`、推論結果は
`output/stage1_v2/` に保存します。推論用 generator は `net_G_A.pth` です。

epoch preview は Stage 1 と同じ `[EID代表 | EID-like | PCD | R]` です。
