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

### 推論・crop の表示設定は現在の train.yaml から（2026-09-08、ユーザー指摘）
- 問題: `run_infer.py` / `run_crop.py` が `diff_range_hu`（crop は display_hu_* / preview_bits も）を run の launch.yaml から取っていたため、200 / 1600 の時代に起動した run ではカラー範囲 ±200・表示 0〜3000 のまま出ていた
- 修正: 表示は重みに紐づく値ではないので、**現在の `configs/train.yaml` の log**（`--train`、TRAIN schema で検証）から取る。`run_infer.display_argv`（共通）。G の構成と HU 正規化は従来どおり launch.yaml。`infer_stage1.sh` / `crop_stage1.sh` に `--train configs/train.yaml` を追加。出力の infer.yaml / crop.yaml に `display` として記録
- 学習中の TB / preview は run の launch の値（古い run を resume するときに今の値にしたければ `bash start.sh resume <run> --diff_range_hu 300 --display_hu_max 2100`）
- `train.py`: `save_latest_freq` ごとの "saving the latest model" の表示を消した（保存は従来どおり。2026-09-08 ユーザー指示）

### Stage 2 用キー（2026-09-08）
- `configs/schema.py`: `MACHINE` に `pcd1024_dir` / `eidlike1024_dir`（flag None、Stage 1 は使わない）を追加。`../configs/machines.yaml` は Stage 共通で Stage 2（`stage2/`、`start2.sh`）がこの 2 キーを読むため、未知キーで Stage 1 の検証が落ちないように宣言だけ置く。旧 run の resume は既存の「run 作成後に追加されたキーは今の yaml の値で補う」処理で通る
