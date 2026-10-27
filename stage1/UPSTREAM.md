# UPSTREAM

このディレクトリは以下のリポジトリを vendoring したもの（`.git`・notebook・`imgs/` を除去）。

- URL: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
- commit: 2a7afba2895d52556dd5dfe07e8555ef657ced6f（2025-08-06 "fix bugs (when use wandb without torchrun)"）
- 取得日: 2026-09-05

## 用途

CycleGAN としては使わない。学習フレームワーク（`train.py` / `test.py` / `data/base_dataset.py` / `models/base_model.py` / `options/*` / `util/*`）だけを借り、
Park et al. 2019 (IEEE Access, DOI 10.1109/access.2019.2934178, arXiv:1903.06257) の fidelity-embedded GAN をその上に再現する。

## 本家からの変更履歴

（各フェーズで追記する）

### フェーズ 0.5 — CycleGAN 論文との照合（`docs/reference/junyanz_vs_cyclegan_paper_audit.md`）
- `models/cycle_gan_model.py`: `--lambda_identity` 既定 0.5 → 0.0（論文 Eq.(3) の一般目的関数に identity loss は無い。Monet→photo / flower enhancement のみ 0.5λ）

### フェーズ 1 — CycleGAN からの不要部位の削除
- `models/fidelity_gan_model.py` を新規作成（`cycle_gan_model.py` のコピーから G_B / D_B / cycle / identity / ImagePool を削除）。`cycle_gan_model.py` 自体は変更せず残す

### フェーズ 2 — Generator の改良
- `models/wavelet_generator.py` を新規作成（HaarDWT / HaarIDWT / ConvBlock / WaveletGenerator。Park 2019 Fig. 3）
- `models/networks.py`: `define_G` に `netG == "wavelet"` を追加（import 1 行 + 分岐 2 行）。既存分岐は変更なし

### フェーズ 3 — Discriminator の改良
- `models/paper_discriminator.py` を新規作成（PaperDiscriminator。Park 2019 Sec. II.C / Fig. 3）
- `models/networks.py`: `define_D` に `netD == "paper"` を追加（import 1 行 + 分岐 2 行）。既存分岐は変更なし

### フェーズ 4 — 損失関数の改良
- `models/fidelity_gan_model.py`: 損失を Park 2019 Eq.(6)/(12) に置換。D := 1 − exp(v)（v は netD 生出力、ユーザー承認 2026-09-05）、D は J_D を最大化（Eq.(12) の argmin は誤植として Theorem II.1 に従う）、fidelity λ‖G(z)−z‖² は画素平均、λ=10。本家 D 損失の ×0.5 を撤去。`--gan_mode fgan_kl` 既定、`--lambda_fid` 追加。lsgan / vanilla は ablation 用に残置


### フェーズ 6 — 学習条件の既定値と実行入口
- `models/fidelity_gan_model.py`: `modify_commandline_options` に論文の学習条件を `set_defaults` で組み込み（netG/netD/ngf/ndf/norm/input_nc/output_nc/crop_size/lr/batch_size/n_epochs/init_gain/lambda_fid）。未記載項目は beta1=0.5 / n_epochs_decay=0 / no_flip / no_dropout / pool_size=0
- `../train_stage1.sh` を新規作成（`--model fidelity_gan` で `stage1/train.py` を呼ぶ）

### フェーズ 5 — データ入力
- `data/ct_dataset.py` を新規作成（`--dataset_mode ct`）。16bit 1ch PNG 読込、`[−1,1]` 正規化（V3 と同一式）、サンプラー切替（random=ケース 1 / paper=論文の固定 40 patch / aligned=ケース 2 未実装）、テスト用フル画像モード。本家 `unaligned_dataset.py` は変更せず残す
- `models/fidelity_gan_model.py`: `preprocess` 既定 `crop` → `none`（切り出しをローダ側へ）
- `../train_stage1.sh`: `--dataset_mode ct --sampling random --samples_per_epoch 24000`

### 動作確認（2026-09-05、合成データ・CPU・リポジトリ外の一時 venv）
- `train.py --model fidelity_gan --dataset_mode ct` を 2 epoch × 4 iter 実行: データ読込 → G → D → 損失 → optimizer 更新 → checkpoint 保存（`{1,2,latest}_net_{G,D}.pth`、`loss_log.txt`）まで一本で通ることを確認。G_fid が 0.195 → 0.06–0.09 に減少（λ=10 の fidelity が効いている）
- 本家最新版は visdom（`--display_id`）を廃止し wandb に移行済み。`train_stage1.sh` から `--display_id 0` を削除
- 実行に必要な追加パッケージ: `wandb`（`util/visualizer.py` が無条件 import）、`dominate`、`opencv-python-headless`。`visdom` は不要。`setuptools<70`（SpicaV3 の requirements）は torch 2.x と衝突するので使わない

### 設定ファイル・起動器（2026-09-05、docs/plans/20260905_config-design-plan.md）
- `configs/schema.py`（新規）: train / mode / machines の必須キー・型・train.py フラグの唯一の正。`validate` / `to_argv` / `check_opt`
- `configs/train.yaml`、`configs/mode.yaml`（新規）、`../configs/machines.yaml`（新規、Stage 共通）
- `run_train.py`（新規）: 3 ファイルを検証 → 全パラメータ明示で `train.py` を exec。実験名は起動時刻 `yyyy/mm/dd/HHMM`（2026-09-07 に `yyyy_mmdd_HHMM` へ変更、下記）。解決済み設定を `<checkpoints_dir>/<name>/launch.yaml` に保存
- `../train_stage1.sh`: `bash train_stage1.sh <machine> [--flag value ...]` に変更（上書きは schema にあるフラグのみ）
- `models/fidelity_gan_model.py`: FE-GAN が使う引数の既定値を `set_defaults(None)` で潰し `check_opt` で検査（既定値廃止）。`--beta2`、`--final_norm_act` を追加
- `models/networks.py`: `define_G(..., final_norm_act=False)` を追加（wavelet 分岐でのみ使用）
- `data/ct_dataset.py`: `<dataroot>/trainA|trainB` 方式を廃止し `--dir_A` / `--dir_B` を直接受ける。全 option の既定値を None に（`check_opt`）。`--sampling aligned` は `--align_meta` の存在を検査
- 動作確認: 起動器経由 smoke（合成データ、CPU、1 epoch）、不正な上書きフラグ・存在しないマシン・`train.py` 直接起動がそれぞれエラーで止まることを確認
- 追加依存: `pyyaml`

### Docker（2026-09-05、docs/plans/20260905_docker-plan.md。リポジトリ直上 `docker/`、Stage 共通）
- `docker/Dockerfile.{gen30,gen50,cpu}`、`docker/requirements-{gen30,gen50,cpu}.txt`、`docker/compose.{gen30,gen50,cpu}.yaml`、`../start.sh`（新規）
- gen30 = SpicaV3 の 30 系構成（CUDA 12.0.1 + torch 2.5.1+cu118）を 40 系と共用。gen50 = Layer_checker_v2 の構成（CUDA 12.8.1 + torch 2.10.0+cu128）から動画系依存を除いたもの。cpu = V3 mac 構成
- compose は `${HOST_DATA_ROOT:?}` 等で環境変数が無ければエラー（既定値なし）。image 名は世代ごとに分離

### 学習の再開（resume、2026-09-07）
- `train.py`（本家）に **4 行**追加: `total_iters` の初期値を `model.resume_total_iters` から取る / epoch 開始時に `model.cur_epoch, model.epoch_done` を設定 / 各 iter で `model.total_iters` を更新 / epoch 末に `model.epoch_done = True`。すべて `[SpicaV5]` コメント付き、他モデルには影響しない
- `models/fidelity_gan_model.py`: `save_networks` を override して `<tag>_state.pth`（epoch / epoch_done / total_iters / optimizer_G・D の state / lr / RNG）を重みと同時に保存。`setup` を override して `--resume_state` があれば optimizer state と RNG を復元し、**scheduler は作り直す**（junyanz の LambdaLR は `--epoch_count` を起点に 0 から数える設計のため、state を復元すると二重計上で lr が 0 や負になる。検証で確認して修正）
- `run_train.py`: `--resume <run> [--resume_tag latest|<epoch>]`。run の `launch.yaml` の argv を使い（今の yaml との差は警告表示）、`--continue_train --epoch_count <保存 epoch(+1)> --epoch <tag> --resume_state <path>` を付ける。`launch_resume_<日時>.yaml` を追加保存
- `../train_stage1.sh`: 環境変数 `SPICA_RESUME` / `SPICA_RESUME_TAG` で受ける。`../start.sh`: `resume <run> [latest|<epoch>]` サブコマンド（`docker compose exec -e` で渡す）
- 検証（合成データ）: 2 epoch 学習 → latest から `--n_epochs 4` で再開 → epoch 3・4 が続き、Adam の step 数が 8→16 と連続、lr が 0.0002 に復帰。線形減衰（2+4 epoch）の tag 3 から再開 → lr 0.00012 → 0.00008 → 0.00004 → 0 と元のスケジュールの続き。存在しない run / tag はエラー

### 学習表示（2026-09-07、docs/plans/20260907_training-monitor-plan.md）
- `util/monitor.py`（新規）: `TrainMonitor` — tqdm バー（画像枚数単位）/ TensorBoard（loss, diag, time, images/current, images/fixed）/ loss_log.txt / `samples/` に uint16 グリッド。本家 `util/visualizer.py` は使わない（残置）
- `train.py`: main ループを書き換え（Visualizer → TrainMonitor、print → tqdm.write、resume フック）。冒頭に変更点を明記
- `data/ct_dataset.py`: `fixed_batch(n)`（監視用の固定サンプル）
- `models/fidelity_gan_model.py`: `--image_freq --n_images --display_hu_min --display_hu_max --diff_range_hu`（番兵 None）
- `configs/schema.py` / `configs/train.yaml`: `log:` を更新（`no_html` 削除）。`../configs/machines.yaml`: `tb_port`
- `../docker/requirements-*.txt`: tensorboard, tqdm。`../docker/compose.*.yaml`: TB ポート公開。`../start.sh`: TB 自動起動 + ブラウザ + `tb`
- Docker 検証（mac）: `bash start.sh` で TensorBoard がコンテナ内で起動（HTTP 200、run を検出）→ ブラウザ → 学習完走。直した点: bash 3.2 の `$VAR全角` 変数名バグ（`${VAR}` に統一）、`pgrep` 非搭載イメージ対策（`/proc` 走査）

### checkpoint 時のフル画像保存（2026-09-07）
- `util/monitor.py` `save_full_images`: `save_epoch_freq` ごとの checkpoint 保存と同時に、固定スライス（`CTDataset.fixed_full(n)`、`patch_seed` で決定的）をフル 512 で G に通し、`<run>/epoch_images/epoch_NNN/<slice>_{pcd,eidlike,R}.png` を uint16 で保存（pcd / eidlike = HU + hu_offset、R = 32768 + ΔHU）。TB にも `images/full/<slice>`。推論は full-image モード（G は全畳み込み + 3 段 Haar、512 は 8 の倍数なので一発）
- `configs/schema.py` / `train.yaml`: `log.n_full_images`。`fidelity_gan_model.py`: `--n_full_images`。`train.py`: 保存呼び出し 1 行
- 検証: `(eidlike − pcd)` と `(R − 32768)` の差が最大 1 HU（丸め）で整合

### run ディレクトリの再編 + 推論（フェーズ 7、2026-09-07、docs/plans/20260907_inference-plan.md）
- run 名を `yyyy/mm/dd/HHMM` から **`yyyy_mmdd_HHMM`** に変更（`/` は引数や TensorBoard で階層に化けるため。ユーザー指示）。`util/run_paths.py`（新規）をレイアウトの唯一の正にした:
  直下は `latest_*` / `best_*`（+ `best.txt`）だけ、epoch ごとの重みは `weights/<epoch>_{net_G,net_D,state}.pth`、画像は `output_images/samples/`（旧 `samples/`）と `output_images/epoch_NNN/`（旧 `epoch_images/`）、推論は `infer/<tag>/`
- `models/base_model.py`（本家）に **`ckpt_path(tag, kind)` を追加し、`save_networks` / `setup` / `load_networks` の 3 箇所のパス組み立てを置き換え**（`[SpicaV5]` コメント付き。latest / best は直下、それ以外は `weights/`。保存時に `weights/` を作る）
- `models/fidelity_gan_model.py`: `<tag>_state.pth` も同じ規則で保存。`util/monitor.py`: `samples_dir` / `epoch_images_dir` を `run_paths` から取り、uint16 化を `denormalize` / `residual_stored` に統一（`eidlike = pcd + (R − 32768)` が厳密に成立）
- `data/ct_dataset.py`: `R_ZERO = 32768`、`residual_stored(a_stored, g_stored)` を追加
- `run_train.py`: `run_name(now)`、resume の state / 重みの所在を `ckpt_path` / `saved_tags` で解決（tag は latest | best | <epoch>）。resume の設定エラーも `[run_train] 設定エラー:` で止まるように
- `mark_best.py`（新規）: `weights/<epoch>_*` → `best_*` にコピーし `best.txt` に記録（`bash start.sh best <run> <epoch>`。定量指標が無いので当面は目視で手動指定）
- **推論（フェーズ 7）**: `inference_dir.py`（新規、全引数必須）— G を `networks.define_G` で組み `<tag>_net_G.pth` を strict に読み `eval()`。`full`（512 を一発。H, W が 8 の倍数でなければエラー）/ `patch`（`patch_size` を `patch_stride` 刻みの**均等格子**で切る = nulmil `image_cut` の「過不足なく被覆」の考え方、`patch_batch_size` 枚ずつ G に通し、窓 `uniform | hann` で重み付き平均）/ `both`（両方保存 + |full − patch| の mean / max [HU] を `diff_stats.txt` に）。出力は入力の相対パスを保った `<out_dir>/{full,patch}/<case>/<slice>.png`（uint16、学習データと同じ規約）と `*_R/`（0 HU = 32768）
- `run_infer.py`（新規）: `configs/infer.yaml`（schema `INFER`）と `machines.yaml`（`infer_input_dir` を追加）を検証し、G の構成は run の `launch.yaml` から取り、解決済み設定を `<run>/infer/<tag>/infer.yaml` に保存して `inference_dir.py` を exec。`../infer_stage1.sh`（新規）、`../start.sh`: `infer <run> [tag] [--flag ...]` / `best <run> <epoch>` サブコマンド、tag に `best` を追加
- 検証（合成データ、scratch venv、CPU）: 2 epoch 学習 → 新レイアウトどおりに保存 → `weights/1` から resume（損失が再現）→ `best` = epoch 2 → `best` から resume → infer `both` / `full` / `patch`（hann）/ tag latest・best・3。`eidlike == pcd + (R − 32768)` が全画素で成立、`infer full (best)` と `output_images/epoch_002` の eidlike / R が完全一致（max diff 0）。エラー: 無い tag / INFER に無いフラグ / infer.yaml のキー欠落 / `infer_input_dir` の無い machines.yaml / 無い run はすべて設定エラーで停止
- |full − patch| の実測（学習 3 epoch の合成重み、1 枚。値そのものより傾向）: stride 8 uniform = mean 1.8 / max 40 HU、stride 128（重なりなし）= mean 1.2 / max 69 HU、**stride 8 hann = mean 0.015 / max 0.33 HU（full とほぼ一致）**。CPU（mac）で stride 8 は 43 s/枚、full は 0.26 s/枚

### run レイアウト第 2 版 + 推論の指定方法の変更（2026-09-07、docs/plans/20260907_inference-plan.md §7）
- checkpoint を**重みディレクトリ**にした: `latest/`、`best/`、`weights/epoch_NNN/` の中に `net_G.pth` / `net_D.pth` / `state.pth`（`util/run_paths.py` の `ckpt_dir` / `ckpt_path` / `saved_tags` / `find_run_dir` / `infer_dir`）。`base_model.py` の呼び出し側は変更なし（`ckpt_path` 経由）
- `util/monitor.py`: `output_images/samples/` を **8bit**（TB と同じ表示用グリッド）に変更。uint16 グリッドは廃止（`_grids` → `_grid`）
- 推論: `bash start.sh infer`（引数なし）→ `../infer_stage1.sh` の変数 `WEIGHT_DIR` / `INPUT_DIR`（`--weight_dir` / `--input_dir` で上書き可）→ `run_infer.py --weight_dir --input_dir`。run は重みディレクトリの上の `launch.yaml` から特定。出力は `<run>/infer/<重みディレクトリ名>/<入力フォルダ名>/`。`machines.yaml` の `infer_input_dir` は削除
- デバイスを `machines.yaml` の `gpu_gen` から明示（`--device cuda|cpu`。cuda 要求で使えなければ `RuntimeError`）
- `mark_best.py`: `weights/epoch_NNN/` → `best/` にディレクトリ単位でコピー

### 推論の出力形式 png | dicom | both と DICOM 出力（2026-09-07、docs/plans/20260907_inference-plan.md §7.1）
- 出力形式は `infer.yaml` ではなく **`../infer_stage1.sh` の変数 `OUTPUT_FORMAT`**（ユーザー指示）。`DICOM_DIR`（元 DICOM ルート）も同じく sh 変数。`run_infer.py` は `--output_format` / `--dicom_dir` を受け、dicom / both のとき DICOM_DIR の存在を検査。CLI 上書きは `--weight_dir / --input_dir / --output_format / --dicom_dir` + INFER schema のフラグ
- `util/dicom_io.py`（新規）: `DicomIndex`（convert_pcd.py と同じ規則で元 DICOM を索引化）、`parse_png_stem`、`hu_to_stored_like_ref`、`write_like_reference`（SpicaV2 `inference_single.py` の `save_like_reference` を移植）。`inference_dir.py` は dicom / both のとき `<out_dir>/<mode>_dicom/<case>/<slice>.dcm` を書く（R は PNG のみ）
- `../docker/requirements-{cpu,gen30,gen50}.txt` に `pydicom` を追加（イメージ再ビルドが必要）
- 検証（合成 DICOM → PNG → 推論 both）: 出力 DICOM の HU が PNG − 1400 と全画素一致、ヘッダ継承、UID 新規、エラー系（DICOM_DIR 無し、不正な形式名、0 始まりの名前、無い症例）はすべて明示エラー
- `configs/infer.yaml` に `dicom.rescale_slope: 1.0` / `dicom.rescale_intercept: -8192.0` を追加（schema `INFER`）。実 DICOM（Siemens NAEOTOM Alpha、1024 系列 1266 枚）で確認した値。`util/dicom_io.py` の `check_rescale` が参照 DICOM のタグと照合し、無い・違えばエラー（`getattr` の既定値は廃止）。出力 DICOM にも同じタグを明示

### `bash start.sh build` をビルド専用に（2026-09-07）
- 以前は「強制再ビルド → 学習」だったが、本番機でコンテナだけ作りたいという要望で **ビルド → コンテナ起動 → torch/cuda 確認 → 終了** に変更（学習しない）。`--flag` を付けるとエラー。学習は続けて `bash start.sh`

### レビュー修正バッチ 1（2026-09-07、docs/plans/20260907_review-fix-list.md F-01〜F-11）
- F-01 `models/fidelity_gan_model.py setup`: state.pth を `map_location="cpu"` で読む（GPU では RNG 用 Tensor が CUDA に載り `.numpy()` で落ちていた）。`keys.cpu().numpy()`
- F-02 / F-03 `run_train.py` 全面書き換え + `configs/schema.py apply_overrides`: sh の上書きを **平坦 dict に反映してから** argv・保存先・launch.yaml を作る（実効値を保存。推論と再開はそれを読む）。bool は `--flag true|false` も可。`--checkpoints_dir` の上書きも保存先に効く
- F-04 / F-05 `../start.sh`: Windows（MINGW / MSYS / CYGWIN）で `MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' MSYS2_ENV_CONV_EXCL='*'` を export（docker.exe へ渡す `/workspace/...` の自動変換を止める）、対話 exec は `winpty`（無ければ `-T`）を通す `exec_it` に統一。`shell` は `bash`
- F-06 `train.py`: ループ終了後、最終 epoch が `save_epoch_freq` の倍数でなければ `latest` と `weights/epoch_NNN` とフル画像を保存
- F-07 `configs/schema.py check_values / check_infer_values`: 値域・相互条件（`save_latest_freq` / `samples_per_epoch` が `batch_size` の倍数、`n_images ≥ 1`、`patch_size % 8`、`1 ≤ stride ≤ size`、`lr_policy == linear`、選択肢の集合、`gpu_gen`、`hu_min < hu_max` など）。`run_train.py` / `run_infer.py` が上書き反映後に呼ぶ
- F-08 `inference_dir.py infer_patch`: 合成後に被覆（`wsum > 0`）と有限性を検査
- F-09 `../docker/compose.gen30.yaml`: `runtime: nvidia` / `NVIDIA_VISIBLE_DEVICES` を外し、SpicaV3 の PC1 実績構成（`deploy` 予約のみ）に戻す
- F-10 `run_train.py` / `fidelity_gan_model.py`: 再開の基準を**最新の** `launch_resume_*.yaml`（`util/run_paths.py latest_launch`、秒まで付けた名前）にし、optimizer 復元後に opt の lr / betas / initial_lr を param_groups に再適用（`--lr` 等の上書きが最優先になる）。`--continue_train` / `--epoch_count` は上書き不可（run_train が決める）。再開時に `checkpoints_dir` を変えるのは拒否。`run_infer.py` も最新 launch を読む
- F-11 `run_train.py`: run ディレクトリが既に存在すれば `ConfigError`（同一分の衝突。以前の「上書き」を撤回）
- 検証（合成データ、CPU venv）: 上書き（`--ngf 48 --hu_max 3000 --checkpoints_dir …`）が launch.yaml の実効値に入り推論が同じ G / HU で動く、`--n_epochs 1 --save_epoch_freq 10` で最終保存、`resume --n_epochs 3 --lr 1e-4` → 上書きなし resume で n_epochs 3 / lr 1e-4 が引き継がれる、9 種の値域違反と bool / `=` / 負数の解析、同一分衝突、stride 256 の拒否と内部検査、`start.sh` の MINGW 分岐（winpty / -T / 環境変数）を fake で確認。**GPU の resume（F-01）は実機で要確認**

### レビュー修正バッチ 2（2026-09-07、docs/plans/20260907_review-fix-list.md F-12〜F-25）
- F-12 `fidelity_gan_model.py setup`: `lr_policy != linear` での再開は `NotImplementedError`（schema でも linear のみ）
- F-13 `util/run_paths.py infer_dir` / `run_infer.py`: 出力先を `<run>/infer/<重み>/<入力>/<yyyy_mmdd_HHMMSS>/` に（実行ごとに別ディレクトリ）
- F-14 `fidelity_gan_model.py save_networks`（本家の save_networks を使わず全面 override）: `<dir>.tmp/` に net_G / net_D / state を書いてから rename で差し替え（原子的）。`mark_best.py` も `best.tmp/` → rename。`run_paths.saved_tags` は `.tmp` / `.old` を無視。`--run` は run 名かパス
- F-15 `configs/schema.py` / `train.yaml` `optim.seed`、`fidelity_gan_model.py --seed`、`train.py`: 起動直後に python / numpy / torch を seed（resume はその後の setup で保存済み RNG に上書き）
- F-16 `run_train.py` / `fidelity_gan_model.py --require_cuda` / `train.py`: machines.yaml の `gpu_gen ≠ 0` なら CUDA が無いと停止（推論側と同じ）
- F-17 `data/ct_dataset.py fixed_batch / fixed_full`: n ≤ 0 は None
- F-18 `util/dicom_io.py` / `inference_dir.py`: 書く直前に `check_pixels`（参照 DICOM → convert_pcd と同じ演算で stored を再現し入力 PNG と全画素照合）、`check_series_unique`（症例内 Series 1 種・単一フレーム）、`ImageType = DERIVED\SECONDARY + 元の 3 値目以降`、`SeriesNumber = 元 + 1000`
- F-19 `data/ct_dataset.py`: `sampling=paper` の `patch_stride 1 → 8` の暗黙置換を廃止。schema が `paper → patch_stride 8` を要求
- F-20 `train.py` / `util/monitor.py`: `total_iters` とバーは実バッチ枚数で加算
- F-21 `data/ct_dataset.py write_png`: `cv2.imwrite` の戻り値と `cv2.error` を `IOError` に。`inference_dir.py` / `monitor.py` はこれを使う
- F-22 `train.py` / `monitor.py accumulate`: epoch 要約は全 step のサンプル重み付き平均
- F-23 `fidelity_gan_model.py`: `diag/D_fake` は `v.clamp(max=80)` で計算（−inf 表示の回避）
- F-24 `../start.sh`: machines.yaml の値をタブ区切りで受ける（パスの空白対応）
- F-25 `train.py`: `WORLD_SIZE > 1` なら `NotImplementedError`（単一 GPU のみ）
- 検証（合成データ、CPU venv）: 同じ seed で 2 run の損失が一致、`.tmp` / `.old` が残らない、best の 2 回差し替えと best からの resume、gpu_gen 30 + CUDA 無しで停止、paper + stride 1 は拒否 / stride 8 で格子が 8 刻み、DDP 環境変数で停止、DICOM の DERIVED / SeriesNumber / 画素照合 OK と、PNG すり替え・Series 混在でエラー、推論 2 回で別ディレクトリ、`write_png` の失敗検出、空白入りパスの起動

### データ列挙の高速化とキャッシュ（2026-09-07、PC2 の初回起動で数分止まった件）
- `data/ct_dataset.py CaseIndex`: `Path.rglob + is_file()`（ファイルごとに stat）を `os.scandir / os.walk` + 拡張子判定に変更。'.' 始まりのファイル・フォルダ（macOS の `._xxx.png`、`.DS_Store`）は無視
- 列挙結果を `<checkpoints_dir>/.case_index/<フォルダ名>_<root の hash>.json` にキャッシュ。次回は走査した全ディレクトリの mtime を照合し、一致すれば再走査しない（ファイルの追加・削除で mtime が変わるので自動で再走査）。学習（`CTDataset`）と推論（`inference_dir.py --index_cache_dir`、`run_infer.py` が渡す）で共通
- `util/dicom_io.py DicomIndex`: 隠しディレクトリを walk から除外
- 背景: V3 は `.bat` から Windows パスを直接マウントしていたが、V5 を PowerShell の `bash`（= WSL）から起動すると WSL の drvfs 越しになり、メタデータ操作が 1 件数 ms〜十数 ms かかる。列挙は Git Bash 起動（Windows パス直接マウント）か WSL の ext4 にデータを置くとさらに速い

### 起動時の表示を短く（2026-09-07）
- `options/base_options.py print_options`: 本家の `Options` ブロックの `print` をやめ、`train_opt.txt` に書くだけに（1 行 `options saved to …` を出す）。実効設定は `launch.yaml`
- `run_train.py` / `run_infer.py`: exec 前に全引数を 1 行で出していたのを、run 名と launch.yaml / infer.yaml の場所だけに（全引数はそのファイルの `argv`）
- `util/monitor.py`: tqdm の `mininterval` を 0.5 → 0.1、`miniters=1`（GPU で 1 step が速いと表示が飛んで見えたため）

### レビュー修正バッチ 3（2026-09-07、docs/plans/20260907_review-fix-list.md F-27 / F-29 / F-30 / F-31 / F-32）
- F-27 `../docker/requirements-{gen30,gen50,cpu}.txt`: `wandb` を外す（import するのは本家 `util/visualizer.py` と `test.py` だけで未使用）。torch 以外に上限つき範囲（numpy<3, opencv-python-headless<6, pyyaml<7, dominate<3, Pillow<13, tensorboard<3, tqdm<5, pydicom<4）。lock の取り方をコメントに
- F-29 `../CLAUDE.md`: 起動コマンドの記述を現行（tb / infer / resume / best）に
- F-30 `../infer_stage1.sh`: `DICOM_DIR` の初期値を SpicaV3 `convert_pcd.py` の変換元と同じ `DataSet/photonCT/PhotonCT512_original` に
- F-31 `../README.md`: 途中 latest からの再開の注意、Git Bash / WSL の違い、lock の取り方
- F-32 `docs/experiments/EXP-20260907-01.md`（定量ベースライン）/ `EXP-20260907-02.md`（FE-GAN 再現 + λ 校正）の事前登録の下書き（Go/No-Go の数値はユーザー確定待ち。docs/ は git 管理外）
- 未決（ユーザー判断）: F-26 `docs` の gitignore、F-28 junyanz `test.py` 等の削除

### 毎 epoch 保存と samples の廃止（2026-09-07、ユーザー指示）
- `configs/train.yaml log.save_epoch_freq`: 10 → 1（毎 epoch `weights/epoch_NNN/` とフル 512 を保存し、後から最良 epoch を選ぶ）
- `util/monitor.py`: 128 patch グリッドは TensorBoard にだけ出し、`output_images/samples/` への書き出しを廃止。`util/run_paths.py` の `samples_dir` / `SAMPLES_DIR` を削除

### TensorBoard を run 単位に（2026-09-07、ユーザー指示）
- `run_train.py start_tensorboard`: `SPICA_TB_PORT`（start.sh が渡す）があるとき、run ディレクトリ確定後に `<run>/tb` だけを logdir にして TensorBoard を起動（`--reload_interval 5`、ログ `<run>/tensorboard.log`）。`/proc` を走査して前の TensorBoard を止める（コンテナ = Linux 前提。mac の scratch では kill は効かない）
- `../start.sh`: train / resume は `tb_kill` → バックグラウンドで応答待ち → ブラウザ、`exec_it -e SPICA_TB_PORT`。`bash start.sh tb` は従来どおり checkpoints_dir 全体（比較用）

### 表示用パネル preview_<slice>/ と表示範囲 0〜3500（2026-09-08、ユーザー指示）
- `util/monitor.py save_full_images`: 固定スライスごとに `output_images/preview_<slice>/epoch_NNN.png`（[PCD | EID-like | R] 横並び。TB の images/full と同じ絵）を書く。`log.preview_bits`（schema / train.yaml / `--preview_bits`）で 16（表示範囲を 0..65535 に伸ばす。プレビューで正しく見え階調 65536 段）か 8。`util/run_paths.preview_dir`
- `train.yaml log.display_hu_max`: 1600 → 2100（stored 3500）。TB と preview に共通
- `run_train.py prepare_resume`: run 作成後に schema へ追加されたキー（seed, preview_bits 等）が run の launch に無いときは今の yaml の値で補い警告（旧 run の resume が止まらないように）

### フル 512 を「固定 1 枚 + epoch ごとのランダム」に（2026-09-08、ユーザー指示）
- `configs/schema.py` / `train.yaml`: `log.n_full_images` を廃止し、`log.full_slice`（固定スライス。pcd_dir からの相対パス、例 `PCD-002/PCD-002-215.png`。起動時に存在を検査）と `log.n_full_random`（epoch ごとに別のランダム枚数）に
- `data/ct_dataset.py`: `fixed_full(n)` → `full_slices(epoch)`（固定 + `patch_seed` と epoch から決定的なランダム。resume しても同じ epoch は同じスライス）
- `util/monitor.py`: 出力先を `preview_fixed_<slice>/epoch_NNN.png` と `preview_random/epoch_NNN_<slice>.png` に分け、TB のタグは `images/full/fixed` / `images/full/random`（固定タグでスライダーが効く）
- `run_train.py prepare_resume`: 旧 run の launch にある廃止キー（`n_full_images` 等）は警告して無視

### 差分（R = G(z) − z）の表示をカラー化（2026-09-08、ユーザー確定）
- 配色: 白 = 0 HU、純青 (0,0,255) = −`log.diff_range_hu`、純赤 (255,0,0) = +。白から純色へ線形、範囲外は端の色で飽和。範囲は 200 → **300** HU（`train.yaml` / plan）。
  試作は epoch_118 の実データで 発散グレー / 青白赤 / 青黒赤 / jet / turbo / |ΔHU| inferno / 両側型 / ±100〜1000 を比較し、CT 論文の差分図（グレー + 窓明記）と発散型（Moreland）の慣行を踏まえて白背景の純色に決めた
- `util/residual_color.py`（新規）: `residual_rgb01 / residual_rgb8`（ΔHU → RGB）、`colorbar_rgb01`（凡例）、`colorbar_filename`、`rgb_to_bgr`（cv2 用）。monitor と推論で共通
- `util/monitor.py`: `_grid`（images/current, images/fixed）と `save_full_images`（images/full/*, preview_fixed_*/ preview_random/）の差分列を RGB に。グリッドは (H,W,3)、`add_image(..., dataformats="HWC")`、区切りは中間グレー（`GRID_PAD = 0.5`。白 = 0 と混ざらないため）、最下段に凡例。preview は RGB 3ch（16bit は 0..65535 に伸ばす。cv2 は BGR なので `rgb_to_bgr`）。`epoch_NNN/<slice>_R.png`（16bit 生データ）は変更なし
- `inference_dir.py`: `--diff_range_hu`（必須）を追加。`--save_residual` のとき `<mode>_R_color/<case>/<slice>.png`（8bit RGB、512×512、凡例なし）と `<out_dir>/R_colorbar_pm<range>HU.png`（1 枚）を書く。色は保存した 16bit R と同じ差（`eid16 − pcd16`）から作るので `_R.png` から再計算した色と一致する
- `run_infer.py`: `R_COLOR_TRAIN_KEYS = {"log.diff_range_hu": "--diff_range_hu"}` を launch.yaml から渡す（無い古い run はエラー）
- `configs/schema.py`: `log.diff_range_hu` の説明を更新。docs: README / CLAUDE.md / run_paths.py / inference-plan / training-monitor-plan
- 検証（リポジトリ外 scratch venv + 合成データ、2026-09-08）: 1 epoch 学習 → TB の 4 タグが (H,W,3) uint8 で凡例の両端が純青/純赤、preview が 3ch uint16、`epoch_001/*_R.png` が 1ch uint16 で `eidlike = pcd + (R − 32768)` 厳密一致。`run_infer.py` 経由の推論で `full_R_color/` が `full_R/` から再計算した色と完全一致、`R_colorbar_pm300HU.png` 生成、`--diff_range_hu 300` が launch.yaml から渡ることを確認

### 固定スライスの表示用を個別保存 + パネルに EID の代表を追加（2026-09-08、ユーザー指示）
- `configs/schema.py` / `train.yaml` / `models/fidelity_gan_model.py`: `log.eid_slice`（`--eid_slice`。eid_dir からの相対パス、例 `EID-001/EID-001-089.png`）を追加。`data/ct_dataset.py` は起動時に dir_B の下の存在を検査し、`eid_reference()` で正規化テンソルを返す。`train.py` が `TrainMonitor(opt, dataset_size, fixed, eid_ref)` で渡す
- `util/monitor.py save_full_images`: パネルを [PCD | EID-like | R | EID] の 4 列に（TB の images/full/fixed|random と preview の両方）。固定スライスは `preview_fixed_<slice>/` に個別保存:
  `00_pcd_<slice>.png` と `01_eid_<eid_slice>.png`（代表。初回の checkpoint で 1 回だけ、名前順で先頭）、`epoch_NNN_eidlike.png`（グレー 1ch）、`epoch_NNN_R_color.png`（RGB、凡例なし 512×512）、`epoch_NNN_panel.png`（旧 `epoch_NNN.png`）。
  深度はすべて `log.preview_bits`。ランダムスライスは `preview_random/epoch_NNN_<slice>.png` のパネルだけ（従来どおり）
- 旧 run の resume は `run_train.py prepare_resume` が `log.eid_slice` を今の yaml から補って警告する（既存の仕組み）
- 検証（scratch venv + 合成データ、2 epoch）: `preview_fixed_*/` に 8 ファイル（代表 2 + epoch ごと 3 × 2）、`01_eid` が入力 EID の表示変換と一致、パネル右端列が代表 EID と一致、パネル幅 = 4 × 512 + 5 × 4、`R_color` が 16bit R から再計算した色と一致、TB の images/full/* がパネルと同形

### パネルの並び順・ラベル・ゲージ（2026-09-08、ユーザー確定）
- `util/panel.py`（新規）: `build_panel(cols, labels, range_hu)` — 並び順 [EID | EID-like | PCD | R]、白い余白、画像の周りに 1px の薄いグレー枠、各列の下に白地・黒文字のラベル帯（`full_labels(epoch)` = `EID` / `EID-like  G(z)   epoch N` / `PCD  z` / `R = G(z) - z   [HU]`）、R の右に小さな縦ゲージ（上 +range 赤、中央 0 白、下 −range 青、±range / ±range/2 / 0 の目盛りと HU）。下端のカラーバーは廃止。入出力は RGB float01（画像列の 16bit 階調を保つ）
- 文字は cv2 の Hershey（英数のみ）。Calibri / 游ゴシックは商用でリポジトリにもイメージにも入れられないため見送り（ユーザー了承）。TrueType にするなら Pillow（requirements に有り）で OFL フォントを同梱する案が残っている
- `util/monitor.py save_full_images`: パネルを `build_panel` に置き換え（TB の images/full/* と preview_fixed_*/epoch_NNN_panel.png、preview_random/ の全部）。個別の `epoch_NNN_R_color.png` は 512×512 のまま（ゲージ無し）。128 patch の `images/current` / `fixed` は従来どおり（下端に凡例）
- 検証（scratch venv + 合成データ、2 epoch）: パネル形状 (560, 2176, 3)、1 列目 = `01_eid`、3 列目 = `00_pcd` と画素一致、TB の images/full/* がパネルと同形。実データ（epoch_118）を `build_panel` で描いて目視確認

### パッチ切り出し `bash start.sh crop`（2026-09-08、ユーザー指示）
- 目的: 小さいパッチの拡大画像をスライドで見せる。指定した PCD スライスを 512 のフル推論にかけ、左上 (x, y) から patch（72）四方を切り出し、EID の代表パッチ（別スライスの指定座標）と並べる
- `configs/crop.yaml`（新規）: `patch` / `panel_scale` / `device`（CROP schema）+ `cases`（PCD / EID のスライス名と左上座標のリスト。`schema.check_crop_cases` で必須キー・型・負座標・スライス名の形を検査）。既定は PCD-002-236 (266,222) × EID-049-079 (327,188)、PCD-002-222 (298,373) × EID-001-093 (270,343)
- `run_crop.py`（新規、run_infer.py と同じ流儀）: crop.yaml と machines.yaml を検証、`--weight_dir` から run → 最新 launch の G 構成（`run_infer.generator_argv`）と表示設定（`log.display_hu_min/max`、`preview_bits`、`diff_range_hu`）を取り、`<run>/infer/<重み>/crop/<時刻>/` を作って `crop.yaml`（実効値）を保存し `crop_patches.py` を exec。`device=cuda` で gpu_gen 0 はエラー
- `crop_patches.py`（新規）: `inference_dir.build_generator / infer_full` で 512 を一発推論。先に全ケースのファイルと座標を検査してから書く（中途半端な出力を残さない）。保存は表示用のみ（16bit 生データ無し。ユーザー指示）: `case<N>_<pcd>_<eid>/1_EID_<eid>_x_y.png`, `2_EID-like_<pcd>_x_y.png`, `3_PCD_<pcd>_x_y.png`（グレー 1ch、等倍）, `4_R_color_<pcd>_x_y.png`（RGB）, `panel.png`（`util/panel.build_panel`、`panel_scale` 倍に最近傍拡大、ラベルの EID-like に重みディレクトリ名）
- `../crop_stage1.sh`（新規）: `WEIGHT_DIR` を書いて `run_crop.py` を呼ぶ。`../start.sh`: `crop` サブコマンド追加。**shell / infer / crop / best / tb はコンテナが起動済み（`docker inspect` の State.Running）なら `up -d` を呼ばず exec だけ**にした（学習中に打っても compose がコンテナを作り直して学習を殺さないように）。train / build / resume は従来どおり up -d
- `util/run_paths.py`: `crop_dir(run_dir, weight_dir, now)` とレイアウト記述
- 検証（scratch venv + 合成データ）: 2 ケースの出力 10 ファイル、切り出し位置が入力の表示変換と画素一致、パネル列が 4 倍拡大と一致、はみ出し・キー欠落・device=cuda on CPU 機・未知フラグの各エラー（はみ出しは書き出し前に止まり画像を残さない）。fake docker で start.sh の crop を dry-run し、起動済みなら up を呼ばないことを確認

### patch 推論の高速化（2026-09-09、ユーザー報告: PC3 で batch 256 にしても 2 枚/秒、VRAM 2.6/95.5 GB・CPU 14%・SSD 0%）
- 原因: `inference_dir.infer_patch` が patch ごとに Python で `out[...] += o[k] * window` と `wsum[...] += window` を発行していた（2,401 patch で約 5,000 回の小さな GPU 演算/枚）。GPU も CPU も遊んだまま 1 本のスレッドの発行が律速で、`patch.batch_size` を上げても回数は減らない
- 修正: 切り出しは平坦 index の gather 1 回（`flat[idx]`）、合成は `index_add_`（重なりは加算）で、バッチ 1 回につき G の呼び出し + 数カーネル。不均等格子（`grid_positions` の linspace）もそのまま扱える。`cudnn.benchmark = True`（cuda のとき。形が固定）、`no_grad` → `inference_mode`
- 数値: `index_add_` の重なりの加算順は CUDA では不定なので float の下位桁が変わりうる（uint16 化後に 1 違うことがある程度）。合成データで旧実装と max|old−new| < 1e-5（0.05 HU 未満）を確認（uniform / hann、512 の均等格子、504×504 stride 24 の不均等格子）
- さらに（ユーザー指摘: patch.batch_size は 1 枚の中の束ね方で、スライスをまたいでいなかった）`batch_slices` 枚のスライスの patch を全部まとめて 1 つの列にし、`patch.batch_size` ずつ G に通す（`infer_patch` は (K,1,H,W) を受ける。スライス k の index は k·H·W だけずらす）。スライス末尾の半端なバッチと GPU 同期が K 枚に 1 回になる
- CPU（mac）は 2,401 patch の畳み込み計算そのものが律速なので速度は変わらない（旧 42 s → 新 42 s/枚）。速度の効果は PC3（GPU）で測る。計算量は full の 150 倍あるので、GPU 使用率（タスクマネージャーは Cuda / Compute を見る）が高ければ計算律速で、その先は patch.stride 8 → 16（patch 数 1/4。full との差は diff_stats.txt で確認）か bf16（数値が変わるので要判断）。`patch.batch_size` は 96 GB なら 512〜1024 でよい

### 推論の出力先を output/ に、既定を Stage 2 用に、読み・計算・書きの並列化（2026-09-09、ユーザー指示）
- `util/run_paths.py`: `infer_output_dir` / `crop_output_dir` を追加し、出力を run の下（6 階層）から **`<repo>/output/<run>_<重み>_<入力名>_<時刻>/`**（crop は `<run>_<重み>_crop_<時刻>/`）に移動。`repo_root()` はこのファイルの 2 つ上（コンテナでは /workspace）。旧 `infer_dir` / `crop_dir` / `INFER_DIR` は削除。`run_infer.py` / `run_crop.py` に `--out_root`（scratch 実行用。通常は省略）
- `configs/infer.yaml`: 既定を `mode: full`、`save_residual: false` に（Stage 2 用のデータ生成向け。`mode: both` は 1 枚 2,401 回 G を呼ぶので 1.8 s/枚かかっていた）。`batch_slices`（8。当初 `full.batch_size` で入れ、同日に patch も束ねるよう改名）、`io.read_workers`（4）、`io.write_workers`（2）を追加（schema INFER + check_infer_values）
- `inference_dir.py`: 読み込み・デコードをスレッドで先読み（順序は deque で保持）、`batch_slices` 枚を 1 回の forward（形が同じときだけまとめる。BN は eval なので 1 枚ずつと同じ）、PNG 書き込みはスレッドで非同期（future を溜めすぎないよう drain、例外は表に出す）。DICOM は主スレッドで直列のまま。終了時に「読み待ち / full / patch / 書き待ち」の内訳を表示
- 他ブランチ（Stage 2）との衝突を避けるため `start.sh` / `CLAUDE.md` / `machines.yaml` / compose は触っていない。CLAUDE.md の出力先の記述（`<run>/infer/...`）はマージ後に直す
- 検証（scratch venv + 合成データ、mac CPU）: full の batch 8 と batch 1 の出力 PNG が 8 枚とも完全一致（最大差 0）、mode both / patch も従来どおり（|full − patch| mean 0.11 HU）、出力先の名前、`--save_residual true/false` の上書き、`full.batch_size 0` / `read_workers -1` / 未知フラグのエラー。CPU では batch 1 → 2 → 4 → 8 で 0.22 → 0.26 → 0.45 → 0.86 s/枚と遅くなる（GPU 向けの設定。CPU では 1）

### 推論・crop の表示設定は現在の train.yaml から（2026-09-08、ユーザー指摘）
- 問題: `run_infer.py` / `run_crop.py` が `diff_range_hu`（crop は display_hu_* / preview_bits も）を run の launch.yaml から取っていたため、200 / 1600 の時代に起動した run ではカラー範囲 ±200・表示 0〜3000 のまま出ていた
- 修正: 表示は重みに紐づく値ではないので、**現在の `configs/train.yaml` の log**（`--train`、TRAIN schema で検証）から取る。`run_infer.display_argv`（共通）。G の構成と HU 正規化は従来どおり launch.yaml。`infer_stage1.sh` / `crop_stage1.sh` に `--train configs/train.yaml` を追加。出力の infer.yaml / crop.yaml に `display` として記録
- 学習中の TB / preview は run の launch の値（古い run を resume するときに今の値にしたければ `bash start.sh resume <run> --diff_range_hu 300 --display_hu_max 2100`）
- `train.py`: `save_latest_freq` ごとの "saving the latest model" の表示を消した（保存は従来どおり。2026-09-08 ユーザー指示）

### Stage 2 用キー（2026-09-08）
- `configs/schema.py`: `MACHINE` に `pcd1024_dir` / `eidlike1024_dir`（flag None、Stage 1 は使わない）を追加。`../configs/machines.yaml` は Stage 共通で Stage 2（`stage2/`、`start2.sh`）がこの 2 キーを読むため、未知キーで Stage 1 の検証が落ちないように宣言だけ置く。旧 run の resume は既存の「run 作成後に追加されたキーは今の yaml の値で補う」処理で通る
- 同日: `stage2_checkpoints_dir` も同様に追加（Stage 2 の学習フェーズ）
- 同日: `stage2_tb_port` を追加。`../start.sh` は stage2_tb_port も読んで `TB_PORT_STAGE2` を export（compose が両ポートを公開する）、`tb_running` / `tb_kill` は自分のポート（tb_port）の TensorBoard だけを対象に。`run_train.py start_tensorboard` も同じポートのプロセスだけ止める。`../docker/compose.*.yaml` に Stage 2 のポート公開を追加（ポート設定の変更は次の up でコンテナが作り直されるので学習中に up しない）
- 同日: `../start.sh` の起動規則を改訂 — コンテナ起動済みなら**どのアクションでも** up を呼ばず exec だけ（学習 / 再開も）、build は起動済みなら拒否、ポート未公開なら注意。中で動く処理（Stage 1 / 2 の学習）を止めないため（ユーザー要望）

### 再開の基準・コンテナ照合・TensorBoard 切替（2026-09-09、Stage 2 レビューの指摘を Stage 1 にも適用）
- `models/fidelity_gan_model.py`: `--launch_path` を追加し、`state.pth` に保存時の実効設定（launch の train / mode / machine）を `config` として入れる
- `run_train.py`: 再開の基準を「最新の launch_resume_*.yaml」から「選んだ checkpoint の state.pth の config」に変更（失敗した起動の設定を次回に引き継がない）。config の無い古い run は従来どおり最新 launch。argv に `--launch_path` を付ける
- `../start.sh`: 起動済みコンテナのマシン名（SPICA_MACHINE）・イメージ・データマウントを今回の指定と照合し、違えば止める。全 exec に `-e SPICA_MACHINE` を渡す。`tb` は「同じポートで全 run の logdir」が動いているときだけ起動済み扱いにし、run 単位のものは止めて起動し直す

### G の Residual 化 `mode.yaml residual`（2026-09-11、段階 5-1、ユーザー承認。ブランチ 002-f_main-create_stage1.5）
- 目的: 論文どおりの G（z → G(z) を直接出力）に対し、`G(z) = z + F(z)` の parameterization に切り替えられるようにする。F は生の generator（wavelet 等、線形出力）で、その出力がそのまま残差 R。損失は不変（fidelity `‖G(z) − z‖² = ‖F(z)‖²`）。論文からの逸脱（独自機構）なので `mode.yaml` で切替、`false` で論文どおり
- `models/networks.py`: `ResidualGenerator(inner, input_nc, output_nc)`（`forward = x + inner(x)`、`input_nc == output_nc` を検査）。`define_G(..., residual)` が `true` のとき包む。計画書 5-1 は「model の forward で足す」だったが、**G のモジュール自体に入れた**: 学習 / `util/monitor.py`（TB グリッド・epoch 末フル画像）/ `inference_dir.py` `infer_full` `infer_patch` / `crop_patches.py` が `netG(x)` を呼ぶだけで揃うため。state_dict のキーに `inner.` が付くので、非 residual の重みを residual 構成に strict load すると明示的に落ちる
- `models/fidelity_gan_model.py`: `--residual`（store_true）を `define_G` に渡す。forward / 損失は無変更。表示の R = `fake_B − real_A` は residual のとき F の生出力そのもの
- `configs/schema.py`: MODE に `residual`（bool）。`LEGACY_FALSE_MODE_KEYS = ("residual",)`: 導入前の run の launch / state.pth に無いキーは「その機構なしで学習された重み」なので **false として扱う**（「run 作成後に追加されたキーは今の yaml の値で補う」規則の例外。今の yaml の true で補うと旧 run の重みの意味が変わる）
- `configs/mode.yaml`: `residual: true`（素の `bash start.sh` は residual。旧方式は `bash start.sh --residual false`）
- `run_infer.py generator_argv`（`run_crop.py` も共用）: `G_MODE_BOOL_KEYS` に `residual`。launch に無ければ注意を出して false。`inference_dir.py` / `crop_patches.py` に `--residual` を追加し `build_generator` → `define_G` に渡す
- `run_train.py prepare_resume`: checkpoint の config に `residual` が無ければ false（注意表示）。**再開で `residual` を変えるのは禁止**（ConfigError。重みの意味が変わる）
- 入れていないもの（1 変更ずつ）: 最終 conv のゼロ初期化、和への soft clamp（値域逸脱で D が勝つ兆候が出てから）、G 構造の変更

### 一発超解像モード `mode.yaml task: sr`（2026-10-12、ブランチ 004-f_main-direct_SR-stage1、ユーザー決定）
- 目的: 「Stage 1 がうまければ一発 SR でもいいのでは」への答えを実験で出す ablation。Park 2019 の一方向 fidelity GAN を**逆向き**（A = EID1024 = EID_v5 を ×2 補間、B = PCD1024_v1）に使う。G / D / 損失は不変、変わるのは「どのデータを A / B に入れるか」と表示名だけ。degrade（従来）は不変。仕様: `docs/plans/20261012_stage1-sr-mode-plan.md`
- `configs/schema.py`: MODE に `task`、新 `SR`（`split_file` / `b_exclude_cases`）+ `validate_sr`（sr.yaml の `train:` は TRAIN のキーだけ差し替え可）+ `check_sr_values` + `check_split`、list 型（`--flag a,b`、空は `""`）、`LEGACY_FALSE_MODE_KEYS` → `LEGACY_MODE_DEFAULTS`（residual False / task degrade）、MACHINE に `eid1024_dir`（`pcd1024_dir` と共にフラグ付き）。`../stage2/configs/schema.py` にも `eid1024_dir` を宣言
- `configs/sr.yaml`（新）: task sr のときだけ読む。`split_file`（`configs/splits/eid_v5.yaml`）、`b_exclude_cases`（`[]` = 外さない。ユーザー決定 2026-10-12: PCD は比較用途のみ）、`train:`（patch 256、full_slice EID-049-079、eid_slice PCD-002-236）。補間の方式は書かず `eid1024_dir/manifest.yaml` を正にする
- `run_train.py`: `--sr`。`peek_task` → sr.yaml → 差し替え → 上書き（train.yaml → sr.yaml → --flag）→ `finish_sr`（ディレクトリ・split・manifest）。argv の `--dir_A` / `--dir_B` を eid1024_dir / pcd1024_dir に、`--split_file --b_exclude_cases --sr_scale --sr_interp --sr_input_size` を追加。launch.yaml に `sr:`（upsample = manifest の方式、train_overlay）。再開は state.pth の `config.sr` 基準、task 変更不可、manifest 変更はエラー
- `data/ct_dataset.py`: `CaseIndex(cases=, exclude=)`、sr の分岐（A = train、val は `full_slices` のランダム専用、test は読まない、未割当はエラー、B は除外、A の大きさ = input_size × scale と A = B の大きさを検査）、`upsample()`（Stage 2 `data/ct_io.upsample` の複製。Stage 間で import しない）。`--task --eid1024_dir --pcd1024_dir --split_file --b_exclude_cases --sr_scale --sr_interp --sr_input_size`
- `util/task_names.py`（新）→ `util/monitor.py`（ファイル名 `<slice>_eid / _pcdlike / _R`、`00_eid_ / 01_pcd_ / epoch_NNN_pcdlike`、パネル [PCD | PCD-like | EID | R]）、`util/panel.full_labels(epoch, task)`、`models/fidelity_gan_model._launch_config` に `sr`
- `run_infer.py generator_argv`: `--task`（旧 run は degrade）+ sr なら `--sr_*`。`inference_dir.py`: sr で 512 入力を manifest の方式で補間（`<mode>_input/` に 16bit で保存）、1024 はそのまま、他はエラー。DICOM は未対応（エラー）。`run_crop.py` / `crop_patches.py`: task で参照ディレクトリと入力モダリティ（sr: eid が入力、pcd が参照、座標は 1024 画素）を切替
- `make_split.py`（新）+ `../split_stage1.sh`（新）+ `../start.sh split`: EID の症例を val : test : train = 15 : 15 : 70、`--force_test EID-049`、seed 固定、既存ファイルは上書きしない
- **`data.patches_per_slice`（同日、PC1 の 1 本目でデータ律速を確認して追加）**: PC1（3090、SSD）で GPU 使用率が 60% の櫛形、SSD 55%、バーが不等速。EID1024 は 66k 枚 × 2 MB = 130 GB で RAM に収まらず、patch ごとに別スライスを SSD 読み + PNG デコードしていたため。`RandomSampler(k, seed)` が引いたスライス（A / B の組）を k サンプル使い回す（位置は毎回独立。通常は worker ごとのカウンタ、`--serial_batches` は index // k で決定的、`fixed_batch` は負のグループ番号で i ごと）。k = 1 は従来と同じ乱数列（単体で確認）。`sampling=paper` では 1 のみ（check_values）。PC1 で大幅高速化を確認（ユーザー報告）したため、同日ユーザー指示で **train.yaml の既定を 8**（degrade も）にし、Stage 2 の `pair_dataset.py` にも同型を入れた。1 にすれば 2026-10-12 以前の乱数列
- 検証（scratch venv + 合成データ 64 / 128 px、CPU）: sr の学習 2 epoch（launch / state の sr、ファイル名、val からのランダム）、resume（checkpoint の task 引き継ぎ）、task 変更・degrade での sr フラグ・manifest 無し・除外症例無しのエラー、推論（512 入力の補間 = EID1024 と画素一致、512 / 1024 入力の出力一致、`pcdlike = input + (R − 32768)`、patch、DICOM 拒否）、crop（sr）、degrade の学習・推論・crop の回帰、task 無しの旧 launch の推論

### D の packing `mode.yaml d_pack`（PacGAN、2026-10-12、段階 5-6 ①、ブランチ 005-f_main-add_randomZ_stage1）
- 目的: 乱数画像 n（段階 5-5、未実装）を入れたとき「毎回同じノイズ量」のような**集団としての幅の無さ**に D が気付けるようにする保険。出典: Zhang, Xu, Li, Jiang, "PacGAN: The power of two samples in generative adversarial networks," NeurIPS 2018。論文（Park 2019）からの逸脱なので `mode.yaml` で切替、`d_pack: 1` で論文どおり（既定）
- `configs/schema.py`: MODE に `d_pack`（int、`D_PACKS = (1, 2)`）。`LEGACY_MODE_DEFAULTS` に `d_pack: 1`（導入前の run の launch / state.pth に無ければ 1 = その機構なしで学習された重み）。`check_values`: `d_pack=2` では `optim.batch_size` 偶数、`batch_size/2 ≥ data.patches_per_slice`（連続する k サンプルは同じスライスなので、B/2 離れた組が同じスライスにならないように）、`sampling=paper + serial_batches` では `batch_size/2 ≥ data.patches_per_image`。フォールバックせず起動時に止める
- `configs/mode.yaml`: `d_pack: 1`
- `models/fidelity_gan_model.py`: `--d_pack`（int、既定 None = 番兵）。`define_D(output_nc × d_pack, ...)`（`PaperDiscriminator(input_nc=2)` = 1 層目の Conv4×4 が 2 ch 入力。他の層・受容野 22 は不変）。`_pack(x)`: `(B,C,H,W) → (B/d_pack, C·d_pack, H, W)`（`torch.cat(x.chunk(d_pack, 0), 1)` = サンプル i と i + B/2 の組。real_B は real_B どうし、fake_B は fake_B どうし。**同じスライスの 2 通りは束ねない**）。`backward_D` / `backward_G` は D に `_pack` した入力を渡すだけで、損失式（fgan_kl / lsgan / vanilla）と `loss_D` / `D_real` / `D_fake` / `G_GAN` の名前・意味は不変（組に対する値）。G の GAN 損失も packed した fake 対に対して取るので勾配は 2 枚とも G に戻る。fidelity は packing と無関係に 1 枚ずつ。`d_pack=1` は `_pack` が素通しで従来と同じ計算
- `run_train.py prepare_resume`: `d_pack` は checkpoint から変更不可（residual / task と同じ ConfigError）。加えて `check_d_pack_weights`: `net_D.pth` の 1 層目（最初の 4 次元 weight）の入力チャネルが `output_nc × d_pack` と一致するかを launch_resume を書く前に照合（Stage 1 には Stage 2 の `check_weights_match` に当たる全テンソル照合は無く、strict load は train.py 起動後に落ちるため、D の 1 層目だけ先に見る）
- 推論（`run_infer.py` / `inference_dir.py`）・crop（`run_crop.py` / `crop_patches.py`）は G だけを組む（`generator_argv` は G のキーだけ読む）ので `d_pack` に依存せず無変更。`task: sr` も同じ model / D なので同じフラグが効く（`bash start.sh --task sr --d_pack 2`）
- 入れていないもの: 乱数画像 n（段階 5-5）、MSGAN 等の他の保険、`d_pack ≥ 3`、ImagePool（本家 pool_size 0 のまま）
- 検証（scratch venv + 合成 64 px、CPU、`run_train.py` 経由）: `d_pack 1` 新規 1 epoch（net_D `model.0.weight` (32,1,4,4)）、`d_pack 2` 新規 1 epoch（(32,2,4,4)、TB `loss/D` 等が出る、launch / state に `d_pack: 2`）→ resume 1 epoch、`d_pack 2` で batch 奇数 / `batch_size/2 < patches_per_slice` / `d_pack 3` の起動時エラー、`d_pack 1` の run を `--d_pack 2` で resume → エラー、`d_pack` の無い旧 run の resume → 「導入前の値 1」で通る、state の d_pack を 2 に書き換えた 1 ch の net_D → 形の照合エラー（launch_resume を書かない）、`task: sr` + `d_pack 2` の学習、`d_pack 2` の run の推論・crop


### G の乱数画像 n の入力 `mode.yaml noise_input`（段階 5-5、2026-10-12、ブランチ 005-f_main-add_randomZ_stage1、ユーザー採用）
- 目的: 劣化の多様性。決定的な G には作れない「構造と無相関の粒」を 1 枚ごとに作る手段を G に与える（64 次元ベクトルではなく入力と同じ大きさの乱数画像。理由は仕様 段階 5-5）。論文（Park 2019）に無い独自機構なので `mode.yaml` で切替、**既定 `noise_input: true`**（ユーザー指示: 素の `bash start.sh` = n あり・PacGAN なし。`d_pack` は 1 のまま）。`false` で論文どおり（`bash start.sh --noise_input false`）。損失は不変（fidelity ‖G(z) − z‖² は n を含む出力に対して取る）
- `models/networks.py`: `NoiseInputGenerator(inner, sigma_hu_min, sigma_hu_max, hu_min, hu_max)` — `forward(x, n=None)` = `inner(cat[x, n])`。n を渡さなければサンプルごとに sigma_hu を log-uniform [min, max] から引き `n = randn(B,1,H,W) × sigma_hu / ((hu_max − hu_min)/2)`（グローバル RNG、x のデバイス。学習は毎回ランダム）。渡せばそれを使う（再現用）。`define_G(..., noise_input, sigma_hu=(min,max), hu=(hu_min,hu_max))` が F の 1 層目を `input_nc + 1` ch で組んで包む。**residual との包みの順は `ResidualGenerator(NoiseInputGenerator(F))`** = `z + F(cat[z, n])`（足すのは z だけ。`ResidualGenerator.forward(x, **kw)` が n を素通し。state_dict は `inner.inner.`）。`noise_wrapper(net)` で包みの中の NoiseInputGenerator を取る（DDP / compile / Residual を剥がす）。呼ぶ側は `netG(x)` のままで動く（学習・TB の patch グリッド）
- 再現用 API（`NoiseInputGenerator.slice_noise(name, hw, device, sigma_mode, sigma_hu_fixed, draw)`）: seed = `zlib.crc32(スライス名の stem)`（Python の `hash()` はプロセスごとに変わるので不可）。CPU の `torch.Generator(seed + draw)` で先頭の `rand` 1 個を sigma 用の u（sigma は draw 0 のものを全 draw で使う）、続く `randn` を n に。sigma_mode `random` = u から log-uniform、`fixed` = 指定値（乱数列は同じなので違いは振幅だけ）。cuda / cpu で同じ n
- `configs/schema.py`: MODE `noise_input`（bool）、`LEGACY_MODE_DEFAULTS` に `noise_input: False`（導入前の run は n なしで学習された重み）。TRAIN `noise.sigma_hu_min / max`（float、HU。`noise_input: false` でも必須で検査は同じ。`check_values`: 0 < min ≤ max）。`task: sr` + `noise_input: true` は `check_values` でエラー（フォールバックしない）。INFER `noise.sigma_mode`（random | fixed）/ `noise.sigma_hu_fixed`（> 0）/ `noise.n_draws`（≥ 1）。sr.yaml に **`mode:` セクション**（MODE のキーだけの差し替え。`validate_sr` → `(sr, train_overlay, mode_overlay)`、`_validate_overlay` で train: と共通化）
- `configs/train.yaml`: `noise: sigma_hu_min 15.0 / sigma_hu_max 60.0`（仮決め、`docs/decisions/20261012`。中央 30 HU は未測定）。`configs/mode.yaml`: `noise_input: true`。`configs/sr.yaml`: `mode: noise_input: false`（sr では n を使わない）。`configs/infer.yaml`: `noise: sigma_mode random / sigma_hu_fixed 30.0 / n_draws 1`
- `models/fidelity_gan_model.py`: `--noise_input`（store_true）、`--sigma_hu_min / max`（float、番兵 None）→ `define_G`。forward / 損失は無変更
- `run_train.py`: `load_sr(path, train, mode)` が `mode:` も差し替え（優先順位 mode.yaml → sr.yaml → --flag。launch の `sr.mode_overlay` に記録）。`prepare_resume`: `noise_input` は checkpoint から変更不可（ConfigError）+ `check_noise_input_weights`（`net_G.pth` の 1 層目の入力チャネル = `input_nc + (1 if noise_input)` を launch_resume を書く前に照合。`check_d_pack_weights` と同型）
- `util/monitor.py`: `save_full_images` は `noise_wrapper` があれば**スライス名の seed**で n を作り `netG(x, n=n)`（固定スライスは毎 epoch 同じ n、ランダムはそのスライス名の seed = 推論の `sigma_mode random` と同じ n）。要約行に seed / sigma を出す。TB `diag/sigma_hu`（直近 batch の平均）。128 patch グリッド（`log_images`）は変更なし（n は毎回ランダム）
- `run_infer.py generator_argv(launch, weight_dir)`: `G_MODE_BOOL_KEYS` に `noise_input`、true なら launch の `noise.sigma_hu_min / max` を `--sigma_hu_min / max` で渡す。`weight_dir` を渡すと `check_noise_input_weights` で照合（crop も）。**n の無い run に `n_draws ≠ 1` や `sigma_mode fixed` を指定するとエラー**（無視しない）
- `inference_dir.py`: `add_generator_args(p)`（G のフラグを crop と共通化）、`build_generator` に noise。`infer_full(net, x, n)` / `infer_patch(..., n)`: **1 スライスにつき H×W 全体の n を 1 枚作り、full はそのまま、patch は x と同じ index で n の対応領域を切り出す**（patch ごとに引き直すと重なりの平均で粒が薄まる。full と patch は同じ n なので `both` の diff_stats は n に依らず比較できる）。`--noise_n_draws k ≥ 2`: draw 0..k−1（seed + draw）で k 回推論し、**平均を通常の出力**（`<mode>/`）に、画素ごとの母標準偏差 [HU] を `<mode>_std/<case>/<slice>.png`（uint16、stored = round(std_HU)、オフセット無し）と `std_stats.txt`（mode / slice / sigma_hu / mean_std / max_std）に。出力ディレクトリに **`noise.yaml`**（`noise_input`、`sigma_hu_min / max`、`sigma_mode`、`sigma_hu_fixed`、`n_draws`、`seed_rule`、`slices: {<case>/<slice>.png: {seed, sigma_hu}}`。n の無い run は `noise_input: false` だけ）を常に書く
- `crop_patches.py`: `add_generator_args`、noise run ならスライス名の seed と sigma（random）で `infer_full(net, x, n=n)`、ログに seed / sigma
- 入れていないもの（1 変更ずつ）: BN と n の相互作用の対策、MSGAN 等の保険（段階 5-6 ②〜④）、学習中の 128 patch グリッドの固定 n、Stage 2 の前処理（スライスごとの n で EID-like を 1024 化。Stage 2 §9）、DICOM 出力への n の記録
- 検証（scratch venv + 合成 64 px、CPU、`run_train.py` / `run_infer.py` / `run_crop.py` 経由。テストコードはリポジトリに置かない）: `noise_input: true` 新規 1 epoch（net_G `inner.inner.enc.0.0.conv.weight` (32,2,3,3)、TB `loss/*` + `diag/sigma_hu`、preview / epoch 画像、要約行に seed / sigma）→ resume 1 epoch（固定スライスの seed / sigma が同じ）、`--noise_input false` 新規（(32,1,3,3)）→ `--noise_input true` で resume → エラー、state の noise_input を true に書き換えた 1 ch の net_G → 形の照合エラー（launch_resume を書かない）、`noise_input` を消した旧 run の resume → 「導入前の値 False」で通る、`--task sr --noise_input true` → エラー / `--task sr` だけ → sr.yaml の mode: で false になり学習・推論が通る、sigma 範囲（min > max、min 0）と infer.yaml の noise.* の値域エラー、推論 `--mode both`（`noise.yaml` に seed / sigma、同じ推論を 2 回で 19 PNG 全部一致、|full − patch| 0.36 HU）、`--noise_n_draws 16`（`full_std/` `patch_std/` `std_stats.txt`。1 epoch の init_gain 0.01 の重みでは std ≈ 0.005 HU = n がほぼ無視されている状態で、学習の浅さとして想定内）、`sigma_mode fixed` 15 / 60（noise.yaml に反映。出力の粒は 1 epoch では変わらず。仕組みは init_gain 0.1 の未学習 G で 16 draws の std が sigma 15 / 30 / 60 に比例することを別途確認）、n の無い run に `n_draws 2` / `fixed` → エラー、n の無い run・sr run の推論と crop（回帰）、noise run の crop（ログに seed / sigma）

### 2 モデル同時学習 `bash start.sh multi <modeA.yaml> <modeB.yaml>`（2026-10-13、ブランチ 005-f_main-add_randomZ_stage1、docs/plans/20261012_multi-model-training-plan.md、ユーザー決定 §5: 2 つ限定・mode 以外の違いも可・Stage 2 には入れない）
- 目的: 「n だけ」vs「n + PacGAN」のように違いが 1 点のモデル 2 つを**同じデータ列**で並べて回す（読み込みが律速なら SSD の負担を 1 本分に）。1 プロセス・DataLoader 1 つで 1 batch をモデル A → B の順に `set_input` + `optimize_parameters`。単独実行は長さ 1 のリストとして**同じループ**（分岐を 2 本にしない）
- `util/run_paths.py`: run 名の規則を `yyyy_mmdd_HHMM[-<name>]` に（`RUN_NAME_RE`、`RUN_SUFFIX_RE` = 英数と `_`。`-` は区切りなので接尾辞に含めない）。`run_name(now, suffix=None)`。`GROUP_FILE = "group.yaml"`。`find_run_dir` / `ckpt_path` / `latest_launch` は名前に依存しないので推論・crop・best・単独 resume はそのまま `-` 付きの run で動く
- `configs/schema.py`: `PER_MODEL_TRAIN_KEYS`（`optim.lr / beta1 / beta2`、`loss.lambda_fid`、`network.ngf / ndf / norm / init_type / init_gain`）= mode ファイルの `train:` でモデルごとに変えてよい train のキー。それ以外（`data.*`、`optim.batch_size / n_epochs / n_epochs_decay / lr_policy / seed`、`network.input_nc / output_nc`、`noise.*`、`log.*`）は組で共有（書くと起動時エラー）。`SHARED_MODE_KEYS = (sampling, serial_batches, task)` は sampler / dataset を決めるので組で同一必須。`MULTI_MAX_MODELS = 2`。`validate_mode_file(data, where)` → `(mode_flat, train_overlay_flat)`（MODE 全キー必須 + 任意 `train:`、`_validate_overlay` を sr.yaml と共用）、`check_shared(specs)`（共有キーの一致。新規は構成上一致、resume で効く）
- `configs/modes/n_only.yaml`（= 今の mode.yaml: noise_input true / d_pack 1）、`configs/modes/n_pacgan.yaml`（d_pack 2。`train:` の書き方の例をコメントに）。`configs/mode.yaml` は単独実行の既定として不変
- `run_train.py`: モデルの仕様（train / mode / machine / sr の実効値、run 名、extra、launch 名）を dict のリストで持つ。`--mode`（単独）と `--multi <A> <B>`（排他。1 つ / 3 つ以上 / `--resume` との併用 / 同名 stem / `-` を含む stem はエラー）、`--resume <runA> [<runB>]`（nargs="+"。2 つで組の再開: 各 run の `state.pth` の実効設定を基準に `prepare_resume` → `check_group` = 共有キーの一致 + 進捗（epoch / epoch_done / total_iters）の一致 + `group.yaml` の相手と違えば警告して続行。同一内容の 2 つは警告して続行）。優先順位 train.yaml → mode ファイルの `train:` → sr.yaml の `train:` → `--flag`（全モデルに同じ値）。各 run に従来形式の `launch.yaml`（+ `mode_file` / `train_overlay` / `group: {runs, index}`）、新規の組には `group.yaml`（`runs`、`mode_files`、`timestamp`、`shared_hash` = 共有 train キー + SHARED_MODE_KEYS + machine の sha1）。train.py には先頭モデルの argv + `--co_launch <B の launch(_resume).yaml>` を渡す。`start_tensorboard(run_dirs, port)`: 1 つなら従来の `--logdir <run>/tb`、組なら `--logdir_spec <nameA>:<runA>/tb,<nameB>:<runB>/tb`（start.sh の `tb_kill` / `tb_running_all` の grep は `--logdir.*` なので両方に効く）
- `train.py`: `parse_opts()`（先頭は sys.argv、`--co_launch` の launch の `argv` を `TrainOptions().parse(argv)` で）。model / monitor をリストで回す。`check_finite`: 損失が NaN / inf なら `RuntimeError` で全部止める（各 run の `latest/` は直前の保存のまま。**単独実行にも効く**ようになった = 同じループの副作用）。resume 時は各モデルの `resume_total_iters` が一致しなければエラー。epoch 末の `full_slices(epoch)` は 1 回だけ呼んで両モデルに渡す
- `options/base_options.py`: `gather_options(argv=None)` / `parse(argv=None)`（None なら本家どおり sys.argv）
- `models/fidelity_gan_model.py`: `self.rng = torch.Generator("cpu").manual_seed(opt.seed)`（isTrain のみ）= **モデル側の乱数源**（重みの初期化、乱数画像 n と sigma）。`noise_wrapper(netG).set_generator(self.rng)`。`state.pth` の `rng` に `"model": self.rng.get_state()` を追加、resume で `set_state`（無い旧 state は seed から作り直し、注意を表示）。`--co_launch`（nargs="*"、内部フラグ）。**seed はモデルの index に依らず共有 `optim.seed` そのもの**: 狙い「A 単独の乱数列 = A+B の A」は index 0 だけでなく B にも成り立ち、組の 2 モデルは同じ初期重み（同じ構造なら）・同じ n の列から出発するので違いが mode / λ に絞れる。データ側（patch 抽出・DataLoader）はグローバル RNG（`train.py` の `torch.manual_seed`）のまま
- `models/base_model.py setup`: `init_net(..., generator=getattr(self, "rng", None))`。`models/networks.py`: `init_weights` / `init_net` に `generator=`（`nn.init.normal_ / xavier_normal_ / kaiming_normal_ / orthogonal_` の `generator=` 引数。torch ≥ 2.1。generator ありのときは CPU で初期化してからデバイスへ）。`NoiseInputGenerator.set_generator(g)` / `random_noise` は generator で引いて x のデバイスへ送る（未設定なら RuntimeError。グローバル RNG へのフォールバックはしない。推論・preview は `slice_noise` の n を渡すので通らない）
- `util/monitor.py`: `label`（組のときの mode ファイル名）、`begin_epoch()`、`step()` は postfix（label があれば `<label>:D` 等）を返し、train.py が全モデル分をまとめて先頭 monitor の `set_postfix` へ。要約行は `[<label>] epoch ...`。TB / loss_log.txt / 画像は run ごと
- `../start.sh`: サブコマンド `multi`（直後の `*.yaml` を取る。ちょうど 2 つ・存在・学習専用・resume と排他をコンテナ起動前に検査）、`resume <runA> [<runB>] [tag]`（2 つ目が run 名の正規表現に合えば組。`SPICA_RESUME="runA runB"` を空白区切りで渡す）。`SPICA_MULTI_MODES` で `train_stage1.sh` へ。ヘルプと例を追記。`../train_stage1.sh`: `SPICA_MULTI_MODES` があれば `--multi`（相対パスはリポジトリ直下基準で絶対化）、`SPICA_RESUME` は空白区切りで `--resume` に展開
- **導入前との損失列の不一致（想定内、以後の基準）**: 導入前はモデル側（init、n）とデータ側（DataLoader の base seed、sampler の順列）が 1 本のグローバル RNG を交互に消費していた。分離するとデータ側の最初の draw が init の後ではなく `manual_seed` 直後になるので、同じ seed でも patch の順と n が変わる（合成データ 1 epoch で確認: 全 step の損失が別の値。再実行どうしは完全一致）。cuDNN の非決定性は対象外
- 検証（scratch venv + 合成 64 px、CPU、`run_train.py` 経由。テストコードはリポジトリに置かない）: (a) 単独 1 epoch の損失列は導入前と**不一致**（上記の理由。新コードの再実行どうしは一致）/ (b) `n_only` + `n_pacgan` 1 epoch: run 2 つ（`<時刻>-n_only` / `-n_pacgan`）、`group.yaml`、各 `loss_log.txt` / `tb/`、tqdm に `n_only:D … n_pacgan:d_in` / (c) **A 単独 = 組の A、B 単独（`--d_pack 2`）= 組の B が損失列として完全一致** / (d) 組の resume 1 epoch（`launch_resume_*.yaml` が各 run に）、片方だけの単独 resume、進捗がずれた 2 run の組 → エラー / (e) `train:` に `optim.batch_size` / `data.patch_size` → エラー、3 ファイル・1 ファイル・同名 stem・`-` 入り stem・`--mode` と `--multi` 併用 → エラー、同一内容 → 警告して続行（2 run の損失列が一致）、`train:` の λ / lr がその run の launch だけに効く、`--lambda_fid` は両 run に / (f) B の `train: optim.lr: 1.0e+30` で step 2 に NaN → 両方 `RuntimeError` で停止、両 `latest/` が残り、組の resume（`--lr` を戻す）が起動して B が再び NaN（重みが既に発散しているので想定どおり）、A 単独の resume は 2 epoch 完走 / (g) `-` 付き run 名で infer（full）、crop、best（`best.txt`）、`--resume_tag best` が通る / `start.sh` の引数解析は docker を呼ばない scratch コピーで `/bin/bash` 3.2 と確認（multi 2 つ・3 つ・1 つ・無いファイル・resume 2 run + tag・best）

## 2026-10-13: TB images/fixed を crop.yaml の cases の位置で切る（`train.yaml log.fixed_patches: crop | random`、`log.crop_file`）
- ユーザー指示「固定の patch は crop.yaml で見ている場所にしてほしい」（それまで Stage 1 の fixed は patch_seed からのランダムで位置指定が無かった。Stage 2 の fixed は full_slice から、crop.yaml は `bash start.sh crop` 専用）
- `configs/schema.py`: TRAIN に `log.fixed_patches`（crop | random）と `log.crop_file`（stage1/ 基準の相対パスか絶対パス）。`check_values` で値域と、crop なら crop_file 必須。`log.n_images` の説明を「current の枚数。fixed は random のときだけ」に
- `configs/train.yaml`: `fixed_patches: crop`、`crop_file: configs/crop.yaml`（既定を crop に。random で導入前の挙動）
- `data/ct_dataset.py`: `--fixed_patches` / `--crop_file`、`_crop_fixed_items()`（`check_crop_cases` で検証 → crop の左上 + crop patch の中心に学習 patch の中心を合わせ、はみ出す分は内側にずらす。degrade: A = dir_A の pcd / B = dir_B の eid、sr: A = dir_A の eid / B = dir_B の pcd。ファイル無し・crop patch のはみ出し・学習 patch > 画像 は起動時 ConfigError）。`fixed_batch(n)` は crop なら cases の数、random なら従来どおり
- 旧 run の resume は「run 作成後に追加されたキーは今の yaml の値で補う」規則で通る（挙動が変わるのは TB の fixed 行だけ）。multi は dataset を共有するので両モデル同じ固定 patch
- 検証（scratch venv + 合成 64 px、CPU、`run_train.py` 経由。テストコードはリポジトリに置かない）: crop 2 cases で 1 epoch → `images/fixed` が 2 行（高さ 2 patch 分）/ 無いスライス・はみ出す座標 → 起動時 ConfigError（メッセージに case 番号と側）/ `--fixed_patches foo` → 値域エラー / random → 従来どおり / multi 2 モデルでも通る


### 表示範囲の上限 HU 2100 → 3000（2026-10-28、ユーザー指示）

- `log.display_hu_max`: 2100 → 3000（stored 0〜3500 → 0〜4400）。下限 −1400 は据え置き。TB の 8bit 表示・preview パネル・推論 / crop のパネルに共通で、Stage 1 / 2 / 3 と ablation（`ablation/*/configs/train.yaml`）を全部揃える。学習・推論の正規化窓（`data.hu_min` / `hu_max` = −1400 / 4096）と 16bit の生出力は変えない
- 推論・crop は現在の `train.yaml` の表示設定を読むので、古い run の重みでも 3000 になる。既存 run を resume した学習中の TB / preview は `state.pth` の実効値（2100）のまま。揃えたいときは `--display_hu_max 3000` を付ける
