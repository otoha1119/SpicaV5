# SpicaV5

Photon-counting CT（PCD-CT）の再構成画像から、従来型 CT（EID-CT）風の画像を **非ペア学習**で生成する研究コード。

- **Stage 1（本リポジトリの主体）**: PCD512 → EID-like512。別患者の PCD 画像群と EID 画像群から、PCD 固有の成分を除いて EID の分布に寄せる変換 G を学習する。
- **Stage 2（保留）**: EID-like512 → PCD1024 の超解像。Stage 1 の出力を教師データにする。

土台は Park, Baek, You, Choi, Seo, *"Unpaired image denoising using a generative adversarial network in X-ray CT"*, IEEE Access 2019（DOI 10.1109/access.2019.2934178。以下 **FE-GAN**）。GAN 損失に fidelity 項 λ‖G(z)−z‖² を埋め込んだ**一方向・cycle なし**の GAN で、学習フレームワークは [junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)（commit 2a7afba）を `stage1/` に vendoring して借りている。本家からの変更はすべて `stage1/UPSTREAM.md` に記録する。

---

## 1. 全体の流れ

```
元 DICOM（PCD-CT / EID-CT）
   │  前処理（SpicaV3 create_dataset/）: pixel × RescaleSlope + RescaleIntercept = HU → HU + 1400 → uint16 1ch PNG
   ▼
<データルート>/PCD512_v2/PCD-nnn/PCD-nnn-sss.png     ← A 側（論文の z）
<データルート>/EID_v5/EID-nnn/EID-nnn-sss.png        ← B 側（論文の x）
   │  bash start.sh            学習（128×128 patch、非ペア）
   ▼
<checkpoints_dir>/<run>/      重み・ログ・途中画像・checkpoint ごとのフル 512 画像
   │  bash start.sh infer      症例丸ごと推論（512 を一発 / patch 分割合成 / 両方）
   ▼
<run>/infer/<重み>/<入力>/    EID-like（16bit PNG、または元ヘッダを継承した DICOM）
```

すべて Docker コンテナ内で実行する（mac の CPU も含む）。ホスト側は `bash start.sh` を叩くだけ。

## 2. モデル仕様（FE-GAN を論文どおりに再現）

| 項目 | 仕様 |
|---|---|
| Generator | deep convolutional framelet U-Net（論文 Fig. 3）。ConvBlock = Conv3×3（zero pad、bias なし）→ BatchNorm → LeakyReLU(0.2)。3 段（32 / 64 / 128 ch）、bottleneck 128 → 256 → 128。段の間は固定 Haar DWT / IDWT（正規直交、学習しない）。高周波サブバンドは同じ段の IDWT へ直結し、エンコーダ特徴と concat。最終層は 3×3 conv（32 → 1、BN / 活性化なし）。1.36M パラメータ |
| Discriminator | [Conv4×4 stride 2 → BN → LeakyReLU(0.2)] × 3（32, 32, 128 ch）→ Conv1×1。受容野 22×22 の PatchGAN。83k パラメータ |
| 目的関数 | J(D,G) = E_x[D(x)] + E_z[log(1 − D(G(z)))] + λ E_z[‖G(z) − z‖²]、G* = argmin_G max_D J、λ = 10。D の出力は D := 1 − exp(v)（v は 1×1 conv の生出力）で実装し、最適 D = 1 − p_G / p_x を満たす。fidelity は画素平均 |
| 最適化 | Adam、lr 2e-4、β = (0.5, 0.999)、batch 40、300 epoch（減衰なし）、重み初期化 N(0, 0.01) |
| 入力 | 128×128 patch、1ch。1 epoch = 24,000 サンプル（論文の patch 集合と同数） |
| 推論 | G は全畳み込み + Haar 3 段なので H, W が 8 の倍数なら 512 を一発で通せる。論文の "patch-by-patch" に対応する patch 分割合成も持つ |

論文に記載が無い点（D の実装形、LeakyReLU の傾き、Haar の正規化、β1、減衰、flip の有無、最終層の BN、正規化範囲、推論 stride）は推測で埋めず、根拠つきで `docs/reference/park2019_implementation_checklist.md` §7 に置き、コードとの照合は `docs/reference/20260905_park2019_implementation_review.md` にまとめている。

## 3. データ規約

| 項目 | 規約 |
|---|---|
| ファイル | uint16 **1ch** PNG（8bit や 3ch は読み込み時にエラー） |
| 値 | `stored = HU + 1400`（HU −1400 → 0、水 0 HU → 1400、EID・PCD 共通） |
| 学習時の正規化 | `stored − 1400 → clip(−1400, 4096) → [0, 1] → [−1, 1]`。逆変換は最後に 1 回だけ丸め・クリップ |
| ディレクトリ | `<root>/<症例>/<slice>.png`。症例フォルダ直下にサブフォルダがあれば再帰的に拾う。`.` で始まるファイル・フォルダ（macOS の `._xxx.png` 等）は無視。列挙結果は `<checkpoints_dir>/.case_index/` にキャッシュされ、ディレクトリの mtime が変わらなければ再走査しない |
| 命名（PCD） | `PCD-nnn-sss.png`。nnn = 元 DICOM の症例フォルダ名の数値部分、sss = フォルダ内 `.dcm` を名前順に並べた 1 始まりの連番。DICOM 出力はこの規則で元ファイルに対応付ける |
| サンプリング | `random`（ケース 1: 症例一様 → スライス一様 → 位置一様、A / B 独立）/ `paper`（論文の固定 40 patch/画像、stride 8）/ `aligned`（ケース 2: 症例メタで位置合わせ。切替口だけ用意） |
| 残差 R | `R = G(z) − z` を uint16 で保存するときは `0 HU = 32768`。`eidlike = pcd + (R − 32768)` が厳密に成り立つ |

患者由来の画像（DICOM・PNG）はリポジトリに入れない。`.gitignore` で `DataSet/`・`stage1/checkpoints/`・`*.pth`・`*.dcm` 等を二重に除外している。

## 4. 設定ファイル（既定値・フォールバック禁止）

すべての値はどこかのファイルに**必ず**書く。必須キーの欠落・未知のキー・型違いは起動時にエラーで止まり、コード側も `set_defaults(None)` + 番兵検査で「設定ファイルを経由せずに動く」事故を防ぐ。唯一の正は `stage1/configs/schema.py`。

| ファイル | 内容 |
|---|---|
| `stage1/configs/train.yaml` | 学習パラメータ全部（optim / loss / network / data / log）。各行に論文の出典 |
| `stage1/configs/mode.yaml` | アルゴリズムの切替（sampling、gan_mode、netG、netD、final_norm_act、serial_batches） |
| `stage1/configs/infer.yaml` | 推論の方式（mode full \| patch \| both、patch の size / stride / blend / batch_size、max_slices、save_residual、DICOM の Rescale 値） |
| `configs/machines.yaml` | マシン定義（Stage 共通）: gpu_gen（30 / 40 / 50 / 0 = CPU）、ホスト側データルート → コンテナ側マウント先、pcd_dir / eid_dir / align_meta / checkpoints_dir / num_threads / tb_port |

優先順位は **sh のコマンド引数 > yaml**。`bash start.sh --n_epochs 50` のように schema にあるフラグだけ上書きできる（無いフラグはエラー）。

## 5. 起動（`bash start.sh`）

リポジトリ直下で実行。マシン名はファイル内の `MACHINE="PC1"` に書き、`bash start.sh PC2` のように引数で渡せば上書きされる。

| コマンド | 動き |
|---|---|
| `bash start.sh` | コンテナ起動（イメージが無ければビルド）→ 学習（その run だけの TensorBoard を起動し、ブラウザを開く） |
| `bash start.sh build` | イメージを（再）ビルド → コンテナ起動 → torch / cuda の確認表示で終了（学習しない。本番機の初期セットアップ用） |
| `bash start.sh resume <run> [latest\|best\|<epoch>]` | その run の checkpoint から続きを学習（optimizer / RNG / 進捗を復元。run 起動時の設定を使う） |
| `bash start.sh best <run> <epoch>` | `weights/epoch_NNN/` を `best/` にコピーして `best.txt` に記録 |
| `bash start.sh infer [--flag ...]` | 症例丸ごと推論（§8） |
| `bash start.sh tb` | 全 run を並べた TensorBoard を起動してブラウザを開く（比較用） |
| `bash start.sh shell` / `down` | コンテナに入る / 停止・削除 |

実験名（run）は起動時刻 `yyyy_mmdd_HHMM`（JST）。同じ分に 2 回起動すると 2 回目はエラー（run 名の衝突）。コマンドの上書きは実効値として `launch.yaml` に保存され、再開・推論に引き継がれる。再開は最新の `launch_resume_*.yaml` を基準にし、再開時の `--lr` 等の上書きも効く。

処理の経路: `start.sh`（ホスト）→ `docker compose`（`docker/compose.{gen30,gen50,cpu}.yaml`、gen40 は gen30 と共用）→ コンテナ内 `train_stage1.sh` / `infer_stage1.sh` → `stage1/run_train.py` / `run_infer.py`（yaml を検証し全引数明示で exec）→ `stage1/train.py` / `inference_dir.py`。

ホスト要件: docker compose v2、python3 + pyyaml（machines.yaml を読むため）。Windows は Git Bash か WSL（PowerShell で `bash` と打つと WSL の bash になる。Git Bash なら Windows パスを直接マウントし、WSL なら `wslpath` で変換する）。依存パッケージは `docker/requirements-*.txt`（torch は固定、他は上限つき）。初回ビルド後に `pip freeze > docker/lock-<gen>.txt` を取ると再現条件になる。

Windows のデータパス: `machines.yaml` の `host_data_root` は `D:/DataSet` のようにホスト表記で書く。PowerShell から `bash start.sh` と打つと通常は WSL の bash（`C:\Windows\System32\bash.exe`）が動き、docker も WSL 側の Linux CLI になる。この場合 `D:/...` はそのまま渡せない（`invalid volume specification: 'D:/DataSet/DataSet:/workspace/DataSet:rw'`）ので、`start.sh` が `wslpath` で `/mnt/d/DataSet` に変換して渡す（変換結果は `[start] WSL: host_data_root を変換 ...` に出る）。Git Bash からの起動なら `D:/` のままで Docker Desktop が解釈する。どちらの bash かは `Get-Command bash` で分かる。`host_data_root` がホストに無い場合は起動前にエラーで止める（compose は無いパスを空ディレクトリとして作ってしまうため）。

改行コード: `.gitattributes` で `.sh` / `.py` / `.yaml` などを LF に固定している（`.sh` はホストの Git Bash とコンテナの Linux bash の両方が読むため）。Windows で `.gitattributes` 追加前に clone した作業ツリーは CRLF になっていて `start.sh: set: pipefail\r: invalid option name`（表示は `: invalid option namet: pipefail` に崩れる）で止まるので、一度 LF で取り直す:

```
git pull
git config core.autocrlf false
git rm --cached -r .
git reset --hard
```

## 6. run ディレクトリ（正は `stage1/util/run_paths.py`）

```
<checkpoints_dir>/2026_0907_1742/
  launch.yaml                 解決済み設定（再開時は launch_resume_<日時>.yaml が増える）
  train_opt.txt, loss_log.txt
  latest/net_G.pth, net_D.pth, state.pth     直下の重みディレクトリは latest と best だけ
  best/…, best.txt                           bash start.sh best で作る（判定は目視。指標ができたら自動化）
  weights/epoch_NNN/net_G.pth, net_D.pth, state.pth   save_epoch_freq（既定 1 = 毎 epoch）ごと
  output_images/epoch_NNN/    checkpoint（毎 epoch）ごとのフル 512（<slice>_pcd / _eidlike / _R.png、16bit = HU が読める）
  output_images/preview_fixed_<slice>/epoch_NNN.png   固定スライス（train.yaml log.full_slice）の表示用パネル [PCD | EID-like | R]。epoch 順に並べて見比べる
  output_images/preview_random/epoch_NNN_<slice>.png  epoch ごとに別のランダムスライス（log.n_full_random 枚）の同じパネル（表示範囲 stored 0〜3500 を 16bit いっぱいに伸ばす）
  （学習中の 128 patch グリッドは TensorBoard だけ）
  infer/<重みディレクトリ名>/<入力フォルダ名>/<実行時刻>/   推論の出力（§8。実行ごとに別ディレクトリ）
  tb/                         TensorBoard
```

毎 epoch のフル 512 は「固定 1 枚（`train.yaml log.full_slice`、pcd_dir からの相対パス）+ ランダム `log.n_full_random` 枚（epoch ごとに別。`patch_seed` と epoch から決定的）」。`state.pth` には optimizer の状態・学習率・RNG（python / torch / numpy）・epoch・iteration 数が入る。scheduler は再開時に作り直す（`lr_policy` は `linear` のみ対応）。学習終了時は `save_epoch_freq` の倍数でなくても最終 epoch を保存する。保存は `<dir>.tmp` に書いてから rename するので、途中で止まっても重みディレクトリに新旧が混ざらない（`.tmp` / `.old` が残っていれば中断の痕跡）。乱数 seed は `train.yaml` の `optim.seed`（cudnn.benchmark は本家のままなので完全な決定性ではない）。単一 GPU のみ対応（DDP は起動時にエラー）。途中保存の `latest/`（epoch 未完）から再開すると、その epoch を頭からやり直すので更新が余分に入り「中断なし」と同じ学習にはならない。再現性を重視するなら epoch 末の checkpoint（`weights/epoch_NNN/`、または epoch 末に保存された `latest/`）から再開する。

## 7. 学習中の表示

- ターミナル: tqdm バー（画像枚数単位。1 step = batch_size 枚）。末尾に D / G_GAN / G_fid と `d_in`（G(z) − z の平均絶対値 [HU]）。
- TensorBoard（学習の起動器が **その run の `tb/` だけ**を logdir にして自動起動。前の run のものは止める。全 run を並べるときは `bash start.sh tb`。`machines.yaml` の `tb_port` で公開）: `loss/*`、`diag/*`（D_real、D_fake、d_in_HU）、`time/*`、`train/lr`、`images/current`、`images/fixed`（固定サンプル。128 patch グリッドは TB にだけ出す）、`images/full/<slice>`（checkpoint 時のフル 512）。横軸は総画像枚数。
- 表示は窓を掛けず HU −1400〜2100（stored 0〜3500）を線形に黒〜白へ、差分パネルは ±200 HU（`train.yaml` の `log:` で変更可）。TensorBoard は 8bit、`preview_<slice>/` は `log.preview_bits`（16 = 表示範囲を 0〜65535 に伸ばす / 8）。

## 8. 推論（`bash start.sh infer`）

学習済みの重みで PNG フォルダを丸ごと EID-like に変換する。指定は `infer_stage1.sh` 冒頭の 4 変数（同名の `--flag` で上書き可）:

```bash
WEIGHT_DIR="/workspace/stage1/checkpoints/2026_0907_1742/best"   # net_G.pth があるディレクトリ（latest/ best/ weights/epoch_NNN/）
INPUT_DIR="/workspace/DataSet/PCD512_v2"                          # 処理する PNG 群
OUTPUT_FORMAT="png"                                               # png | dicom | both
DICOM_DIR="/workspace/DataSet/PhotonCT512_original"               # 元 DICOM ルート（dicom / both のとき）
```

| 項目 | 仕様 |
|---|---|
| G の構成 | 重みディレクトリの上にある run の `launch.yaml` から取る（学習時と同じ G を組む） |
| デバイス | `machines.yaml` の `gpu_gen`（0 → cpu、それ以外 → cuda。cuda が使えなければエラー、cpu には落とさない） |
| mode `full` | 512 を一発で G に通す |
| mode `patch` | `patch.size` の patch を `patch.stride` 刻みの均等格子（端まで過不足なく被覆）で切り、`patch.batch_size` 枚ずつ G に通し、窓 `uniform` / `hann` で重み付き平均 |
| mode `both` | 両方を保存し、スライスごとの \|full − patch\| の mean / max [HU] を `diff_stats.txt` に記録 |
| 出力 PNG | `{full,patch}/<症例>/<slice>.png`（16bit、入力と同じ規約。そのまま次段の学習データになる）、`{full,patch}_R/`（残差、0 HU = 32768） |
| 出力 DICOM | `{full,patch}_dicom/<症例>/<slice>.dcm`。元 DICOM のヘッダを継承し画素だけ置換。HU → 格納値は `infer.yaml` の `dicom.rescale_slope / rescale_intercept`（参照 DICOM のタグと全枚照合、違えばエラー）。書く直前に「参照 DICOM から再現した stored 値 == 入力 PNG」を全画素で照合し、症例内の Series が 1 種であることも検査する（単一 Series・単一フレーム CT のみ）。`ImageType` は `DERIVED\SECONDARY`、`SeriesNumber` は元 + 1000、SOP / Series UID は新規、SeriesDescription に由来を記す |
| 出力先 | `<run>/infer/<重みディレクトリ名>/<入力フォルダ名>/<yyyy_mmdd_HHMMSS>/`。実行ごとに別ディレクトリなので、重みや `--max_slices` を変えた再実行が混ざらない。解決済み設定は `infer.yaml` |

BatchNorm は eval（running 統計）。checkpoint 時のフル画像と同じ経路なので、同じ重み・同じスライスなら結果は一致する。

## 9. リポジトリ構成

```
SpicaV5/
├── start.sh                 起動スクリプト（ホスト）。ヘッダーが取扱説明
├── train_stage1.sh          学習の入口（コンテナ内）
├── infer_stage1.sh          推論の入口（コンテナ内。WEIGHT_DIR / INPUT_DIR / OUTPUT_FORMAT / DICOM_DIR）
├── configs/machines.yaml    マシン定義（Stage 共通）
├── docker/                  Dockerfile / compose / requirements（gen30, gen50, cpu）
├── stage1/                  junyanz 本家の vendoring + Stage 1 の実装
│   ├── UPSTREAM.md          本家からの変更履歴（全部）
│   ├── configs/             schema.py, train.yaml, mode.yaml, infer.yaml
│   ├── models/              fidelity_gan_model.py, wavelet_generator.py, paper_discriminator.py（+ 本家）
│   ├── data/ct_dataset.py   16bit 1ch PNG の読み込み・正規化・サンプリング
│   ├── util/                monitor.py（表示）, run_paths.py（run レイアウト）, dicom_io.py（DICOM 出力）
│   ├── run_train.py / train.py / run_infer.py / inference_dir.py / mark_best.py
│   └── checkpoints/         run の出力（git 管理外）
├── docs/
│   ├── reference/           要件（research_requirements.md）、論文照合チェックリスト、実装レビュー、junyanz 監査
│   ├── decisions/           方針転換の記録
│   ├── plans/               実装仕様・各計画書（設定、Docker、データローダ、表示、推論）
│   ├── surveys/             手法調査
│   ├── experiments/         実験ログ（EXP-<YYYYMMDD>-<seq>.md、事前登録つき）
│   └── research-config.md   閾値などの上書き設定
└── CLAUDE.md                研究運用ルールと現在の方針
```

## 10. 関連プロジェクト

- **SpicaV3**: 前世代（F-LSeSim ベース）。DICOM → PNG の前処理スクリプト（`create_dataset/convert_pcd.py`, `convert_eid.py`）と評価指標の設計（SENTINEL-CARE）はこちらを参照する。
- **SpicaV2**: SR-CycleGAN。DICOM 書き出しの元になった実装がある。
