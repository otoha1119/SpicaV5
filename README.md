# SpicaV5

Photon-counting CT（PCD-CT）の再構成画像から、従来型 CT（EID-CT）風の画像を **非ペア学習**で生成する研究コード。

- **Stage 1（本リポジトリの主体）**: PCD512 → EID-like512。別患者の PCD 画像群と EID 画像群から、PCD 固有の成分を除いて EID の分布に寄せる変換 G を学習する。
- **Stage 2（着手 2026-09-08、§11）**: EID-like1024 → PCD1024 の教師あり同解像度回帰（U-Net + MSE、ILUMENATE = Koons et al., Med Phys 2025 準拠）。Stage 1 の出力を ×2 補間したものを入力、PCD1024 を教師にする。

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
| `configs/machines.yaml` | マシン定義（Stage 共通）: gpu_gen（30 / 40 / 50 / 0 = CPU）、ホスト側データルート → コンテナ側マウント先、pcd_dir / eid_dir / align_meta / checkpoints_dir / num_threads / tb_port、Stage 2 用の pcd1024_dir / eidlike1024_dir / stage2_checkpoints_dir / stage2_tb_port（§11） |

優先順位は **sh のコマンド引数 > yaml**。`bash start.sh --n_epochs 50` のように schema にあるフラグだけ上書きできる（無いフラグはエラー）。

## 5. 起動（`bash start.sh`）

リポジトリ直下で実行。マシン名はファイル内の `MACHINE="PC1"` に書き、`bash start.sh PC2` のように引数で渡せば上書きされる。

| コマンド | 動き |
|---|---|
| `bash start.sh` | コンテナ起動（イメージが無ければビルド。**起動済みなら up を呼ばず exec だけ**）→ 学習（その run だけの TensorBoard を起動し、ブラウザを開く） |
| `bash start.sh build` | イメージを（再）ビルド → コンテナ起動 → torch / cuda の確認表示で終了（学習しない。本番機の初期セットアップ用。**コンテナ起動済みなら拒否**。`down` してから） |
| `bash start.sh resume <run> [latest\|best\|<epoch>]` | その run の checkpoint から続きを学習（optimizer / RNG / 進捗を復元。その checkpoint の `state.pth` に入っている実効設定を使う） |
| `bash start.sh best <run> <epoch>` | `weights/epoch_NNN/` を `best/` にコピーして `best.txt` に記録 |
| `bash start.sh infer [--flag ...]` | 症例丸ごと推論（§8） |
| `bash start.sh crop [--flag ...]` | パッチ切り出し（§8）。指定 PCD スライスを 512 でフル推論 → 左上 (x, y) から 72 四方を切り出し、EID の代表パッチと並べる（スライド用）。学習中に打ってよい |
| `bash start.sh tb` | 全 run を並べた TensorBoard を起動してブラウザを開く（比較用） |
| `bash start.sh shell` / `down` | コンテナに入る / 停止・削除 |

実験名（run）は起動時刻 `yyyy_mmdd_HHMM`（JST）。同じ分に 2 回起動すると 2 回目はエラー（run 名の衝突）。コマンドの上書きは実効値として `launch.yaml` に保存され、推論に引き継がれる。**再開の基準は選んだ checkpoint の `state.pth` に入っている実効設定**（その重みを作った設定。2026-09-09 変更。失敗した起動の `launch_resume_*.yaml` は基準にならない。`state.pth` に設定が無い古い run だけ最新の launch を使う）。再開時の `--lr` 等の上書きも効き、`launch_resume_<日時>.yaml` に記録される。

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
  output_images/preview_fixed_<slice>/                固定スライス（train.yaml log.full_slice）の表示用（表示範囲 stored 0〜3500 を線形に、log.preview_bits の深度）
    00_pcd_<slice>.png, 01_eid_<eid_slice>.png        代表: PCD 入力と EID（log.eid_slice）。初回の checkpoint で 1 回だけ（名前順で先頭）
    epoch_NNN_eidlike.png, epoch_NNN_R_color.png      epoch ごとの EID-like（グレー）と R（カラー、下記）
    epoch_NNN_panel.png                               上を並べた [EID | EID-like | PCD | R + ゲージ]、各列の下にラベル。epoch 順に並べて見比べる
  output_images/preview_random/epoch_NNN_<slice>.png  epoch ごとに別のランダムスライス（log.n_full_random 枚）の同じパネル（パネルだけ）
  （学習中の 128 patch グリッドは TensorBoard だけ）
  （推論と crop の出力は run の下ではなく、リポジトリ直下の output/。§8）
  tb/                         TensorBoard
```

毎 epoch のフル 512 は「固定 1 枚（`train.yaml log.full_slice`、pcd_dir からの相対パス）+ ランダム `log.n_full_random` 枚（epoch ごとに別。`patch_seed` と epoch から決定的）」。`state.pth` には optimizer の状態・学習率・RNG（python / torch / numpy）・epoch・iteration 数が入る。scheduler は再開時に作り直す（`lr_policy` は `linear` のみ対応）。学習終了時は `save_epoch_freq` の倍数でなくても最終 epoch を保存する。保存は `<dir>.tmp` に書いてから rename するので、途中で止まっても重みディレクトリに新旧が混ざらない（`.tmp` / `.old` が残っていれば中断の痕跡）。乱数 seed は `train.yaml` の `optim.seed`（cudnn.benchmark は本家のままなので完全な決定性ではない）。単一 GPU のみ対応（DDP は起動時にエラー）。途中保存の `latest/`（epoch 未完）から再開すると、その epoch を頭からやり直すので更新が余分に入り「中断なし」と同じ学習にはならない。再現性を重視するなら epoch 末の checkpoint（`weights/epoch_NNN/`、または epoch 末に保存された `latest/`）から再開する。

## 7. 学習中の表示

- ターミナル: tqdm バー（画像枚数単位。1 step = batch_size 枚）。末尾に D / G_GAN / G_fid と `d_in`（G(z) − z の平均絶対値 [HU]）。
- TensorBoard（学習の起動器が **その run の `tb/` だけ**を logdir にして自動起動。前の run のものは止める。全 run を並べるときは `bash start.sh tb`。`machines.yaml` の `tb_port` で公開）: `loss/*`、`diag/*`（D_real、D_fake、d_in_HU）、`time/*`、`train/lr`、`images/current`、`images/fixed`（固定サンプル。128 patch グリッドは TB にだけ出す）、`images/full/<slice>`（checkpoint 時のフル 512）。横軸は総画像枚数。
- 表示は窓を掛けず HU −1400〜2100（stored 0〜3500）を線形に黒〜白へ。差分パネル（R = G(z) − z）はカラーで、白 = 0 HU（変化なし）、純青 = −300 HU（G が HU を下げた）、純赤 = +300 HU（上げた）、白から純色へ線形、範囲外は端の色で飽和（`train.yaml` の `log.diff_range_hu`、実装は `stage1/util/residual_color.py`）。パネル（`stage1/util/panel.py`）は白い余白、各列の下にラベル帯（`EID` / `EID-like G(z) epoch NNN` / `PCD z` / `R = G(z) - z [HU]`）、R の右に小さな縦ゲージ（+300 … 0 … −300 の目盛り）。文字は cv2 の組み込みフォント（商用フォントはリポジトリに入れられないため）。TensorBoard は 8bit、`preview_*/` は RGB 3ch で `log.preview_bits`（16 = 表示範囲を 0〜65535 に伸ばす / 8）。16bit の `_R.png` はグレーの生データのまま（CT 論文の差分図の標準。図にするときは窓をキャプションに書く）。

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
| 出力 PNG | `{full,patch}/<症例>/<slice>.png`（16bit、入力と同じ規約。そのまま次段の学習データになる）、`{full,patch}_R/`（残差、0 HU = 32768）、`{full,patch}_R_color/`（同じ残差の表示用カラー 8bit RGB、512×512 のまま。範囲は現在の `train.yaml` の `log.diff_range_hu`）、`R_colorbar_pm300HU.png`（凡例 1 枚） |

パッチ切り出し `bash start.sh crop`: ケース（PCD / EID のスライス名と左上座標のペア。既定 2 ケース）・`patch`（72）・`panel_scale`（パネルだけ最近傍で拡大、既定 4）・`device`（既定 cpu。学習中に VRAM を取り合わない）は `stage1/configs/crop.yaml`、重みディレクトリは `crop_stage1.sh` の `WEIGHT_DIR`（`--weight_dir` で上書き可）。表示（格納値 0〜3500 を線形、`preview_bits`、R の範囲）は現在の `train.yaml` の `log` から（run の `launch.yaml` ではないので、古い run でも今の見た目）。出力は `output/<run>_<重みディレクトリ名>_crop_<実行時刻>/case<N>_<pcd>_<eid>/` に `1_EID` / `2_EID-like` / `3_PCD` / `4_R_color`（72×72、等倍）と `panel.png`（`[EID | EID-like | PCD | R + ゲージ]`）。16bit の生データは書かない。`shell` / `infer` / `crop` / `best` / `tb` はコンテナが起動済みなら `up` を呼ばず `exec` だけなので、学習中に打ってもコンテナは作り直されない。
| 出力 DICOM | `{full,patch}_dicom/<症例>/<slice>.dcm`。元 DICOM のヘッダを継承し画素だけ置換。HU → 格納値は `infer.yaml` の `dicom.rescale_slope / rescale_intercept`（参照 DICOM のタグと全枚照合、違えばエラー）。書く直前に「参照 DICOM から再現した stored 値 == 入力 PNG」を全画素で照合し、症例内の Series が 1 種であることも検査する（単一 Series・単一フレーム CT のみ）。`ImageType` は `DERIVED\SECONDARY`、`SeriesNumber` は元 + 1000、SOP / Series UID は新規、SeriesDescription に由来を記す |
| 出力先 | **リポジトリ直下の `output/<run>_<重みディレクトリ名>_<入力フォルダ名>_<yyyy_mmdd_HHMMSS>/`**（2026-09-09 に run の下の 6 階層から移動。コンテナでは `/workspace/output/`、`.gitignore` 済み）。名前に run・重み・入力・時刻が全部入るので探しやすく、実行ごとに別ディレクトリなので再実行が混ざらない。解決済み設定は `infer.yaml`。Stage 2 のデータ作成はこの下の `full/` を指す |
| 既定 | `mode: full`、`save_residual: false`（Stage 2 用のデータ生成向け）。品質確認は `bash start.sh infer --mode both --save_residual true --max_slices 4` のように一時的に上書き |
| 速度 | 読み込み・デコードを `io.read_workers` 本で先読み、`batch_slices` 枚のスライスをまとめて G に通す（full はそのまま 1 回の forward、patch は全スライスの patch を束ねて `patch.batch_size` ずつ。BN は eval なので 1 枚ずつと同じ結果）、PNG 書き込みは `io.write_workers` 本で非同期。GPU は数 ms なので律速は PNG の読み書き（HDD + WSL 越しは特に）。終了時に「読み待ち / full / patch / 書き待ち」の内訳を表示するので、どこが律速か分かる。CPU 実行（mac）では `--batch_slices 1` が最速（まとめるほど遅い）。patch 方式は 2026-09-09 に patch ごとの Python ループを廃止（gather + `index_add_`）。それ以前は `patch.batch_size` を上げても速くならなかった（GPU も CPU も遊んだまま 1 本のスレッドが 1 patch ずつ発行していたため） |

BatchNorm は eval（running 統計）。checkpoint 時のフル画像と同じ経路なので、同じ重み・同じスライスなら結果は一致する。

## 9. リポジトリ構成

```
SpicaV5/
├── start.sh                 Stage 1 の起動スクリプト（ホスト）。ヘッダーが取扱説明
├── start2.sh                Stage 2 の起動スクリプト（ホスト）。学習 | resume | dataset | tb | build | shell | down（§11）
├── train_stage1.sh          学習の入口（コンテナ内）
├── infer_stage1.sh          推論の入口（コンテナ内。WEIGHT_DIR / INPUT_DIR / OUTPUT_FORMAT / DICOM_DIR）
├── dataset_stage2.sh        Stage 2 データ作成の入口（コンテナ内。INPUT_DIR。出力先は machines.yaml の eidlike1024_dir）
├── train_stage2.sh          Stage 2 学習の入口（コンテナ内）
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
├── stage2/                  Stage 2（U-Net、ILUMENATE 準拠）。junyanz は使わない
│   ├── configs/             schema.py（Stage 2 設定の唯一の正）, dataset.yaml
│   ├── make_dataset.py      512 → 1024 補間でデータセットを作る（manifest.yaml 付き）
│   ├── run_train.py / train.py   学習の起動器（launch.yaml を書いて exec）と学習ループ（自前）
│   ├── data/                ct_io.py（PNG 読み書き・正規化・列挙）, pair_dataset.py（ペアローダ・症例分割）
│   ├── models/              unet.py（ILUMENATE の U-Net）, regression_model.py（損失・Adam・checkpoint・resume）
│   ├── util/                run_paths.py（run レイアウト）, monitor.py（tqdm / TensorBoard scalar / loss_log）
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

## 11. Stage 2（EID-like1024 → PCD1024）

Stage 1 が作った EID-like512 を **前処理で ×2 補間して 1024 のファイルにし**、PCD1024（同一患者・同一スキャンの別再構成。`DataSet/PCD1024_v1`）を教師にした教師あり同解像度回帰。土台は ILUMENATE（Koons et al., *Med Phys* 2025;52(7):e17874。入力も教師も 1024 マトリクスで、U-Net + MSE だけ）。仕様と作業計画: `docs/plans/20260908_stage2-unet-implementation-spec.md`。

```
Stage 1 推論出力 <run>/infer/<重み>/<入力>/<時刻>/full/<case>/<slice>.png   （EID-like512）
   │  bash start2.sh dataset        ×2 補間（bicubic）→ 同じ <case>/<slice>.png + manifest.yaml
   ▼
<DataSet>/EIDlike1024_v1/           学習入力 = machines.yaml の eidlike1024_dir（実 EID の推論用は EID_v5 から同じ道具で EID1024_v1 を作る）
<DataSet>/PCD1024_v1/               教師 = machines.yaml の pcd1024_dir（ユーザー作業で変換済み。512 と先頭から 1 対 1、3 症例は末尾の枚数が違うので min まで使う）
   │  bash start2.sh                学習（未実装）
```

| コマンド | 動き |
|---|---|
| `bash start2.sh [マシン名] dataset [--flag ...]` | `dataset_stage2.sh` の `INPUT_DIR`（変換元。実行ごとに変わるので sh に書く）を `stage2/configs/dataset.yaml`（`scale` 2 / `interp` bicubic / `input_size` 512）で補間し、**`configs/machines.yaml` の `eidlike1024_dir`**（マシンごとのデータ配置。Stage 1 の `pcd_dir` と同じ場所）に書く。`--input_dir`、`--eidlike1024_dir` / `--pcd1024_dir`、`DATASET` のフラグだけ上書き可。出力先は `container_data_root` 配下のサブフォルダで、既にあればエラー（上書きしない）。`<出力先>.tmp` に書いて検算後に rename。コンテナが起動済みなら up を呼ばず exec だけ（Stage 1 の学習中でも可） |
| `bash start2.sh build` / `shell` / `down` | start.sh と同じ（コンテナは Stage 1 と共用） |
| `bash start2.sh [マシン名] [--flag ...]` | **学習**。`stage2/configs/train.yaml`（数値と症例分割 `train_cases` / `val_cases` / `test_cases`）と `mode.yaml`（arch / base_ch / n_pool / init_type / loss / final_act / residual）を `stage2/configs/schema.py` で検証し、実効値を `<stage2_checkpoints_dir>/<run>/launch.yaml` に保存して `stage2/train.py` に渡す（junyanz は使わない）。run 名は `yyyy_mmdd_HHMM`。その run の `tb/` で TensorBoard を起動しブラウザを開く（scalar は `print_freq` step ごと、学習中の 128 patch グリッド `images/current` / `images/fixed` は `image_freq` step ごと、フル 1024 のパネルは epoch 末）。上書きは schema のフラグだけ（list は `--val_cases PCD-017,PCD-018`） |
| `bash start2.sh resume <run> [latest\|best\|<epoch>]` | 続きから学習（`net_G.pth` + `state.pth` = optimizer / RNG / 進捗を復元。**基準はその checkpoint の `state.pth` に入っている実効設定**。`--n_epochs` 等の上書き可。重みの形が実効設定と合わなければ launch を書く前に止める）。途中保存の `latest/` から再開するとその epoch を頭からやり直す（残り batch だけの厳密な再開ではない）ので、再現性を重視するなら `weights/epoch_NNN/` から |
| `bash start2.sh tb` | Stage 2 の全 run を並べた TensorBoard（`stage2_tb_port`） |
| `bash start2.sh infer` | 未実装（別フェーズ） |

- 値の規約は Stage 1 と同じ（uint16 1ch、stored = HU + 1400）。補間は float32 → 四捨五入 → clip。cv2.resize の half-pixel 規約は 512 / 1024 の再構成格子の対応と一致する（実ペアで NCC のシフト探索が (0, 0)。計画書 §2.3）
- `manifest.yaml` に時刻・マシン・入出力・方式・症例ごとの枚数・上書き・隣の `infer.yaml`（Stage 1 の由来）を残す
- **学習の中身**（論文どおり。仕様 §4）: U-Net = Conv3×3(zero pad)+ReLU ×2 を 3 段（128 / 256 / 512 ch）、max pool 2 回、up-conv（ConvTranspose 2×2 s2）2 回、skip concat、BN なし、最終 Conv3×3 → 1ch（活性化なし。`final_act: tanh` で変種）、約 9.8M パラメータ。損失 MSE（正規化空間 `[−1,1]`、Stage 1 と同じ HU 窓）、Adam lr 0.001 β (0.9, 0.999) 一定、100 epoch、batch 16、1024 グリッド上の対応位置 128² patch を症例一様 → スライス一様 → 位置一様で 1 epoch 64,000 サンプル、augmentation なし
- **症例分割**: `train_cases`（PCD-003〜016）/ `val_cases`（017, 018。epoch 末の指標だけ）/ `test_cases`（001, 002, 019, 020。**一切読まない**。2026-09-09 改訂）。ディスク上の症例は必ずどれかのリストに入っていること（未割当は起動前にエラー）。固定プレビュー `log.full_slice` は test の症例でもよい（ユーザー決定 2026-09-09: holdout を目視用に毎 epoch 見る。読むのはその 1 枚だけ。**best の選択は `val/rmse_HU` で行い、この絵で選ばない**。どのリストにも無い症例はエラー）。ペアはファイル名で対応させ、症例ごとに両方にある名前だけ使う（PCD-006 / 011 / 015 の末尾差はここで吸収。枚数は `<run>/dataset_info.yaml`）
- **監視**: ターミナルは Stage 1 と同じ tqdm バー（画像枚数単位、末尾に loss と RMSE [HU]）だけ。TensorBoard は scalar（`loss/mse`、`train/rmse_HU`、`val/rmse_HU` と参照線 `val/rmse_input_HU` = 入力 − 教師、`time/*`、`train/lr`）と、epoch 末のフル 1024 パネル `images/full/fixed`（4 列）/ `images/full/random`（3 列）。128 patch のグリッドは出さない
- **epoch 末のフル 1024 画像**（Stage 1 と同じ 2 系統。固定 = `log.full_slice` **`PCD-002/PCD-002-236.png`**（test の症例）、ランダム = `log.n_full_random` 枚（**val だけ**、epoch ごとに別。学習に使ったスライスは記憶で良く見えるので汎化の目視にならない）、実 EID テスト = `log.eid_slice` **`EID-049/EID-049-079.png`**（`eid_dir` の 512 を `eidlike1024_dir/manifest.yaml` の方式で 1 枚だけ補間して U-Net に通す。EID_v5 全件の 1024 化は約 200 GB になるのでファイルにしない）。すべて `train.yaml` で変更でき `--flag` で上書き可）:
  - `output_images/epoch_NNN/<slice>_{eidlike,pcd1024,pcdlike}.png`, `<eid_slice>_{eid1024,pcdlike}.png` — 生 uint16（stored = HU + 1400、HU が読める）
  - `output_images/preview_fixed_<slice>/` — 表示用（stored 0〜3500 を線形に、`log.preview_bits` 16 / 8）: `00_eidlike1024_` / `01_pcd1024_` / `02_eid1024_`（代表、初回だけ）、`epoch_NNN_pcdlike.png`、`epoch_NNN_eid_pcdlike.png`、`epoch_NNN_panel.png` = **[EID-like1024 | PCD1024 | PCD-like1024 | 実 EID → PCD-like1024]**（`stage2/util/panel.py`。白余白・1px 枠・ラベル帯は Stage 1 と同じ）
  - `output_images/preview_random/epoch_NNN_<slice>.png` — ランダムスライスの 3 列パネルだけ
  - 容量: 1 epoch あたり生 16bit 約 16 MB + preview（16bit）約 50 MB。100 epoch で約 6.5 GB（重み約 12 GB とは別）
- **run ディレクトリ**（正は `stage2/util/run_paths.py`）: `launch.yaml`（+ `launch_resume_*.yaml`）、`dataset_info.yaml`、`loss_log.txt`、`latest/` `best/` `weights/epoch_NNN/`（各 `net_G.pth` + `state.pth`、`.tmp` → rename で原子的）、`output_images/`、`tb/`
- **Docker と TensorBoard の配線**: コンテナ（spicav5）・`docker/` のイメージと compose・`configs/machines.yaml` は Stage 1 と共用。`start2.sh` は `start.sh` と同じ手順（machines.yaml → compose 選択 → up / exec）。**両 sh とも、コンテナが起動済みならどのアクションでも `up` を呼ばず exec だけ行い、`build` は拒否する**（compose は設定が変わっていると `up -d` でコンテナを作り直し、中の学習を止めるため。止めるのは `down` だけ）。起動済みのコンテナは**作成時のマシン名（`SPICA_MACHINE`）・イメージ・データマウント元を今回の指定と照合**し、マシン名かイメージが違えば止める（マウントは exec では変えられない）。全 exec に今回のマシン名を `-e` で渡す。`bash start.sh tb` / `bash start2.sh tb` は、同じポートで run 単位の TensorBoard（学習が起動したもの）が動いていれば止めて全 run 表示に切り替える（2026-09-09）。TensorBoard は Stage 2 専用の `stage2_tb_port`（既定 6007）で、compose が `tb_port` と両方を公開し、各 Stage は自分のポートの TensorBoard だけを起動・停止する（同じマシンで両 Stage を同時に学習しても互いを止めない）。compose の設定（ポート等）を変えても起動済みのコンテナには反映されない（ポート未公開の注意が出る）。**中の処理が終わってから `down` → 起動し直す**
- Stage 2 の設定の正は `stage2/configs/schema.py`。`configs/machines.yaml` は Stage 共通で、Stage 2 は使うキー（gpu_gen / host_data_root / container_data_root / num_threads / tb_port / eid_dir / pcd1024_dir / eidlike1024_dir / stage2_checkpoints_dir / stage2_tb_port）だけ検査する。Stage 1 の schema は未知キーを拒むので、Stage 2 のキーは `stage1/configs/schema.py` の MACHINE にも宣言してある（Stage 1 は使わない）

