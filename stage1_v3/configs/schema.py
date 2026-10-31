"""stage1_v3/configs/schema.py — 射影型 Stage 1（確率的物理コア P + 学習残差 R + 実 EID 恒等項）の設定スキーマ。

原則は Stage 1 / 2 / stage1_v2 と同じ: 既定値（フォールバック）は無い。ここに宣言したキーは**必ず**設定ファイルに存在しなければならず、未知のキー・型違いはエラー。
sh からの上書き（--flag value）は schema にあるフラグだけ受け付け、実効値を launch.yaml に残す。このディレクトリだけで完結させる（Stage 間で import しない）。

machines.yaml は Stage 共通ファイルなので、使う既存キーだけを検査する（validate_shared）。

各キーの宣言: Key(type, flag, doc)
  type : int | float | str | bool | list（str のリスト）。float は int の値も受け付ける。bool は YAML の true / false のみ
  flag : 上書き用のフラグ名。None は上書き不可
  doc  : 意味と出典。論文 = Park, Baek, You, Choi, Seo, "Unpaired Image Denoising Using a GAN in X-Ray CT," IEEE Access 2019（fidelity GAN、G / D / λ の出典）。
         設計 = docs/surveys/20261030_stage1-projection-identity.md 推薦 1 と、その敵対的レビュー（2026-10-30）による修正

【構成】（docs/surveys/20261030_stage1-projection-identity.html）
  PCD 経路: u = P(x; σ, c, b)（学習なし。スライスごとに σ・c・b を範囲から抽選）、ŷ = u + R(u)。D は実 EID y と ŷ を見分ける
  EID 経路: y + R(y)。恒等項 L_id = ‖R(y)‖²（D には見せない）
  損失    : L_GAN + λ_fid ‖R(u)‖² + λ_id ‖R(y)‖²。R は決定的（乱数入力なし）。両経路は 1 バッチに連結して G を通す（BatchNorm のドメイン分割対策）
【v4】（pipeline.mode: v4、docs/decisions/20261031_stage1-v3-noise-path-analysis.md §3・§6、2026-10-31）
  PCD 経路: x → Dn（凍結 denoiser、configs/denoise.yaml で事前学習）→ x_c → G: x_c + R(x_c) → ⊛ k(z) → + σ(x_c) ⊙ (h(z) ⊛ ε)、σ ≥ σ_min → u。D は y と u
  EID 経路: v3 と同じ（Dn も k もノイズも通さない）
  損失    : L_GAN(u) + λ_fid ‖R(x_c)‖² + λ_id ‖R(y)‖² + λ_k ‖x_c ⊛ k − x_c‖² + λ_com COM²。ε に依存する項は L_GAN だけ（C2）
  Dn の学習は DENOISE（configs/denoise.yaml、train_denoiser.py）。合否は check.*（check_denoiser.py、§6.4）
"""

from collections import namedtuple

Key = namedtuple("Key", "type flag doc")


# ---------------------------------------------------------------------------
# configs/train.yaml
# ---------------------------------------------------------------------------
TRAIN = {
    # pipeline（v3 = 従来、v4 = Dn → G → k(z) → +ノイズ。docs/decisions/20261031_stage1-v3-noise-path-analysis.md §3）
    "pipeline.mode":         Key(str,   "--pipeline_mode",   "v3（u = P_θ(x) → G、従来）| v4（x → Dn → G → ⊛ k(z) → + σ ⊙ (h ⊛ ε)。乱数ノイズを最後に足し、後ろに学習ネットを置かない）。重みの形と経路を決めるので resume で変えられない"),
    "pipeline.dn_source":    Key(str,   "--dn_source",       "v4 の Dn の与え方: weights（学習型 Dn の net_Dn.pth を凍結して学習中に掛ける。n2n / blindspot）| precomputed（bash start1_v3.sh <machine> dnfixed で先に denoise したデータ pipeline.dn_dir から x_c を読む。BM3D など学習ステップ内で回せない Dn 用）。経路を決めるので resume で変えられない。v3 では weights を書く（使わない）"),
    "pipeline.dn_weights":   Key(str,   "--dn_weights",      "v4 + dn_source weights: 凍結 denoiser の重み net_Dn.pth（コンテナ内パス。bash start1_v3.sh <machine> dn の出力 checkpoints/denoiser/latest/net_Dn.pth）。存在必須。v3 と dn_source precomputed では空文字 \"\" を書く（使わない）"),
    "pipeline.dn_dir":       Key(str,   "--dn_dir",          "v4 + dn_source precomputed: dnfixed の出力ディレクトリ（full/<case>/<slice>.png = Dn(x) と manifest.yaml を持つ。コンテナ内パス）。PCD の全スライスが同じ相対パスで要る。それ以外では空文字 \"\""),
    # optim（論文 Sec. III: lr 0.0002、batch 40、300 epoch。β・decay は未記載）
    "optim.lr_G":            Key(float, "--lr_G",            "G（残差 R）の学習率。論文 0.0002（G / D 共通）"),
    "optim.lr_D":            Key(float, "--lr_D",            "D の学習率。論文 0.0002"),
    "optim.beta1":           Key(float, "--beta1",           "Adam β1。論文未記載（junyanz 慣行 0.5）"),
    "optim.beta2":           Key(float, "--beta2",           "Adam β2。論文未記載（0.999）"),
    "optim.batch_size":      Key(int,   "--batch_size",      "PCD 側の patch 数。EID 側（恒等項）も同数を同じ step で通す。論文 40"),
    "optim.n_epochs":        Key(int,   "--n_epochs",        "lr 一定の epoch 数。論文 300"),
    "optim.n_epochs_decay":  Key(int,   "--n_epochs_decay",  "lr を 0 へ線形減衰させる epoch 数。論文未記載（0 = 減衰なし）"),
    "optim.seed":            Key(int,   "--seed",            "python / numpy / torch の乱数 seed。cudnn.benchmark は有効なので完全決定ではない"),
    "optim.precision":       Key(str,   "--precision",       "演算精度 fp32 | bf16。bf16 は CUDA の autocast（重み・optimizer state・損失の計算は FP32 のまま）。stage1_v2（008）の高速化と同じ"),
    "optim.compile":         Key(bool,  "--compile",         "CUDA で G / D を torch.compile(mode='max-autotune') する（CPU では無効）。初回 iteration はコンパイルで遅い。Docker image に build-essential が要る（docker/Dockerfile.gen30/gen50）"),
    # loss
    "loss.gan_mode":         Key(str,   "--gan_mode",        "fgan_kl（論文 Eq.6: D := 1 − exp(v)、stage1 と同じ）| lsgan | vanilla（ablation）"),
    "loss.lambda_fid":       Key(float, "--lambda_fid",      "fidelity λ_fid · mean(R(u)²)（正規化単位 [−1,1]。論文 λ = 10）。R を小さく保つ力。1 draw（レビュー B: 2 draw 内積は denoise に盲目なので不採用）"),
    "loss.lambda_id":        Key(float, "--lambda_id",       "恒等項 λ_id · mean(R(y)²)（実 EID y を G に入れたときの残差。設計 L_id）。正規化単位では RMS 20 HU ≈ MSE 5e-5 なので λ_fid より 2 桁以上大きい値が要る（レビュー F）。0 で無し（= P + fidelity GAN）"),
    "loss.lambda_id_decay_epochs": Key(int, "--lambda_id_decay_epochs", "λ_id を初期値から lambda_id_final まで線形に下げる epoch 数（Krishnan 2025 型の減衰。0 = 一定）"),
    "loss.lambda_id_final":  Key(float, "--lambda_id_final", "減衰後の λ_id（lambda_id_decay_epochs > 0 のときだけ使う）"),
    "loss.lambda_k_fid":     Key(float, "--lambda_k_fid",    "learned: カーネル側の fidelity λ_k · mean((x ⊛ k − x)²)（k を δ から遠ざけすぎない。ノイズ側には掛けない）。0 で無し"),
    "loss.lambda_k_com":     Key(float, "--lambda_k_com",    "learned: カーネル重心の正則化 λ_com · E[COM²]（px²。ぼかしが平行移動にならないように）"),
    # degrade = 物理コア P（学習なし。R-S23: σ・鮮鋭度はスライスごとに範囲から抽選。範囲は R-S22 の実測（bash start1_v3.sh <machine> stats）を含むまで広げる）
    "degrade.enabled":       Key(bool,  "--degrade",         "false で P を掛けない（u = x。従来の fidelity GAN + 恒等項の対照）"),
    "degrade.mode":          Key(str,   "--degrade_mode",    "gaussian（一様ぼかし + ノイズ。1〜2 本目）| bilateral_blur（エッジ保存の平滑化 → 弱いぼかし → ノイズ）| learned（学習型の確率的劣化 P_θ: u = x ⊛ k(z) + σ(x) ⊙ (h(z) ⊛ ε)。models/degrade_net.py。2026-10-30 の転換、docs/decisions/20261030_stage1-v3-learned-degradation.md）"),
    "degrade.learned_z_dim": Key(int,   "--learned_z_dim",   "learned: カーネル生成の乱数 z の次元"),
    "degrade.learned_kernel_size": Key(int, "--learned_kernel_size", "learned: ぼかしカーネル k(z) の大きさ（奇数、px）"),
    "degrade.learned_noise_kernel_size": Key(int, "--learned_noise_kernel_size", "learned: ノイズ整形フィルタ h(z) の大きさ（奇数、px。NPS の形）"),
    "degrade.learned_amp_ch": Key(int,  "--learned_amp_ch",  "learned: 振幅ネット σ(x) のチャネル数"),
    "degrade.learned_sigma_max_hu": Key(float, "--learned_sigma_max_hu", "learned: ノイズ振幅 σ(x) の上限 [HU]（sigmoid × これ。実 EID の局所 SD 22 HU の数倍）"),
    "degrade.learned_sigma_min_hu": Key(float, "--learned_sigma_min_hu", "v4 のみ: ノイズ振幅の下限 [HU]。σ(x) = σ_min + (σ_max − σ_min)·sigmoid(·)（C3: 乱数ノイズだけで実 EID の量に達していれば、決定的な残りノイズは量の超過として D が罰する）。実 EID の一様 ROI 局所 SD 中央値 22.7（EXP-20261030-01 の stats）。v3 では無視（σ = σ_max·sigmoid のまま）"),
    "degrade.bilateral_sigma_color_hu_min": Key(float, "--bil_color_min", "bilateral の sigma_color [HU] の下限（mode bilateral_blur のみ。PCD のノイズ SD ≈ 93 HU の 2〜3 倍が目安）"),
    "degrade.bilateral_sigma_color_hu_max": Key(float, "--bil_color_max", "同 上限"),
    "degrade.bilateral_sigma_space_px_min": Key(float, "--bil_space_min", "bilateral の sigma_space [px] の下限（mode bilateral_blur のみ）"),
    "degrade.bilateral_sigma_space_px_max": Key(float, "--bil_space_max", "同 上限"),
    "degrade.blur_sigma_px_min": Key(float, "--blur_min",    "ぼかし（MTF 低下）の Gaussian σ [px] の下限。0 で無ぼかしを含む"),
    "degrade.blur_sigma_px_max": Key(float, "--blur_max",    "同 上限。PCD512 → EID512 の鮮鋭度差（edge 10–90 % 幅の比）から決める"),
    "degrade.noise_sigma_hu_min": Key(float, "--noise_min",  "加算ノイズの標準偏差 [HU] の下限（ぼかし後の画像に足す。局所 SD の目標 = 実 EID の一様 ROI の SD − PCD 側の SD の二乗差）"),
    "degrade.noise_sigma_hu_max": Key(float, "--noise_max",  "同 上限"),
    "degrade.noise_corr_px_min": Key(float, "--corr_min",    "ノイズの空間相関（NPS の形）: 白色ノイズに掛ける Gaussian σ [px] の下限。0 = 白色。大きいほど低周波寄り（NPS ピークが低周波へ）"),
    "degrade.noise_corr_px_max": Key(float, "--corr_max",    "同 上限"),
    "degrade.noise_body_only": Key(bool, "--noise_body_only", "true なら体外（空気、HU < body_threshold_hu）にはノイズを足さない（実 EID の空気はほぼ −1024 で平坦なため）。false で全面"),
    "degrade.body_threshold_hu": Key(int, "--body_threshold_hu", "体内判定の HU 閾値（noise_body_only 用。ぼかした画像で判定）"),
    # network（論文 Fig. 3。stage1/ と同じ構造を複製）
    "network.netG":          Key(str,   "--netG",            "wavelet（論文 Fig.3 の deep convolutional framelet、stage1/models/wavelet_generator.py の複製）| resnet_9blocks | resnet_6blocks（ablation）"),
    "network.netD":          Key(str,   "--netD",            "paper（論文 Fig.3、受容野 22×22、R-S11）| basic（70×70 PatchGAN、ablation）"),
    "network.ngf":           Key(int,   "--ngf",             "G の基本チャネル数。論文 Fig.3 は 32"),
    "network.ndf":           Key(int,   "--ndf",             "D の 1 層目チャネル数。論文 Fig.3 は 32"),
    "network.norm":          Key(str,   "--norm",            "resnet / basic の正規化層 batch | instance。wavelet / paper は論文どおり BatchNorm 固定で無視"),
    "network.init_type":     Key(str,   "--init_type",       "重み初期化 normal | xavier | kaiming | orthogonal（論文 \"Gaussian distribution with a mean of 0\"）"),
    "network.init_gain":     Key(float, "--init_gain",       "初期化の標準偏差 / gain（論文 0.01）"),
    # data（stage1 degrade と同じ列挙・sampling・HU 正規化）
    "data.sampling":         Key(str,   "--sampling",        "random | paper。stage1 と同じ"),
    "data.use_eval_only_cases": Key(bool, "--use_eval_only_cases", "false なら configs/eval_only.yaml の評価専用症例を A / B とも学習から除外"),
    "data.patch_size":       Key(int,   "--patch_size",      "論文 128"),
    "data.patch_stride":     Key(int,   "--patch_stride",    "random は 1、paper は 8"),
    "data.patches_per_image": Key(int,  "--patches_per_image", "paper sampling で各画像から固定抽出する patch 数（論文 40）"),
    "data.samples_per_epoch": Key(int,  "--samples_per_epoch", "random sampling の epoch 長（PCD patch 数）。stage1 と同じ 24,000"),
    "data.patches_per_slice": Key(int,  "--patches_per_slice", "1 スライスから連続して切る patch 数（全 Stage 8）。P の σ・c・b はこの単位（= スライス）で抽選する"),
    "data.patch_seed":       Key(int,   "--patch_seed",      "fixed_batch / full_slices と serial_batches の決定的サンプリング用 seed"),
    "data.serial_batches":   Key(bool,  "--serial_batches",  "true で sampler を index 決定的に（再現用）"),
    "data.hu_offset":        Key(int,   "--hu_offset",       "stored = HU + hu_offset（1400）"),
    "data.hu_min":           Key(int,   "--hu_min",          "正規化窓の下限 HU（−1400 → −1）"),
    "data.hu_max":           Key(int,   "--hu_max",          "正規化窓の上限 HU（4096 → +1）"),
    # log
    "log.print_freq":        Key(int,   "--print_freq",      "損失・TensorBoard scalar・loss_log.txt の更新間隔（step）"),
    "log.image_freq":        Key(int,   "--image_freq",      "TensorBoard の patch グリッドの間隔（step）"),
    "log.n_images":          Key(int,   "--n_images",        "patch グリッドの行数"),
    "log.full_slice":        Key(str,   "--full_slice",      "epoch 末に通す固定 PCD512（pcd_dir 相対）"),
    "log.eid_slice":         Key(str,   "--eid_slice",       "パネル左端の代表 EID512（eid_dir 相対）。恒等経路のフル画像にも使う"),
    "log.n_full_random":     Key(int,   "--n_full_random",   "PCD 全症例から epoch ごとに選ぶランダムスライス数"),
    "log.n_draws":           Key(int,   "--n_draws",         "epoch 末の固定スライスで P を何回引き直して標準偏差画像（R-S21 の判定）を作るか。1 で作らない"),
    "log.display_hu_min":    Key(int,   "--display_hu_min",  "表示の線形範囲の下限 HU（−1400）"),
    "log.display_hu_max":    Key(int,   "--display_hu_max",  "表示の線形範囲の上限 HU（3000）"),
    "log.preview_bits":      Key(int,   "--preview_bits",    "output_images/preview_*/ のビット深度 16 | 8"),
    "log.diff_range_hu":     Key(int,   "--diff_range_hu",   "R のカラー表示の ±範囲 HU（白 = 0、青 = −、赤 = +）"),
    "log.tb_full_size":      Key(int,   "--tb_full_size",    "TensorBoard のフルパネル列サイズ"),
    "log.save_epoch_freq":   Key(int,   "--save_epoch_freq", "checkpoint（weights/epoch_NNN/）とフル画像を保存する epoch 間隔"),
    "log.save_latest_freq":  Key(int,   "--save_latest_freq", "latest/ を保存する間隔（画像枚数。batch_size の倍数）"),
    # val = epoch ごとの自動採点（util/validator.py）。R-S22 の 3 軸と恒等残差で best/ を自動更新する
    "val.n_pcd_slices":      Key(int,   "--val_pcd_slices",  "採点用の固定 PCD スライス数（学習症例から乱択、P を 1 回掛けて G に通す）。0 で採点しない"),
    "val.n_eid_slices":      Key(int,   "--val_eid_slices",  "参照の固定 実 EID スライス数（中央値と四分位を 1 回計算。恒等残差の eval RMS にも使う）"),
    "val.auto_best":         Key(bool,  "--auto_best",       "true なら採点（score）が最良のとき best/ を自動更新（best.txt に score を記録）。手動の bash start1_v3.sh best は上書きできる"),
    "val.target_ry_rms_hu":  Key(float, "--target_ry_rms_hu", "恒等残差 RMS(R(y))（eval mode）の目標 [HU]。score の正規化に使う（EXP-20261030-01 の H1 は 20）"),
    "val.target_fingerprint_sd_hu": Key(float, "--target_fingerprint_sd_hu", "v4 のみ: 指紋 SD（ε = 0 で生成した u の一様 ROI 局所 SD の中央値 [HU]）の目標。score に fingerprint_sd / これ を足す（§3.3: 実 EID の 1/4 = 6）。v3 では使わない（旧 run の移行値 0.0）"),
}

# ---------------------------------------------------------------------------
# configs/machines.yaml（リポジトリ直下、Stage 共通）のうち stage1_v3 が使うキー
# ---------------------------------------------------------------------------
MACHINE = {
    "gpu_gen":                 Key(int, None,                        "GPU 世代 30 | 40 | 50 | 0(CPU)"),
    "host_data_root":          Key(str, None,                        "Docker 起動時にマウントするホスト側のデータルート"),
    "container_data_root":     Key(str, None,                        "マウント先（コンテナ内）"),
    "num_threads":             Key(int, "--num_threads",             "DataLoader の worker 数"),
    "pcd_dir":                 Key(str, "--pcd_dir",                 "A = PCD512（stage1 degrade と同じ）"),
    "eid_dir":                 Key(str, "--eid_dir",                 "B = 実 EID512（stage1 degrade と同じ）"),
    "tb_port":                 Key(int, None,                        "stage1 と共通の TensorBoard ポート（stage1 と stage1_v3 は同時に学習しない前提。stage1_v2 と同じ扱い）"),
}

# ---------------------------------------------------------------------------
# configs/infer.yaml — 推論の方式（bash start1_v3.sh <マシン> infer）
# ---------------------------------------------------------------------------
INFER = {
    "apply_degrade":    Key(bool, "--apply_degrade", "入力に P を掛けてから G に通す（PCD 入力 = Stage 2 の学習データ生成）。false で G だけ（実 EID 入力 = 恒等の確認）"),
    "n_draws":          Key(int,  "--n_draws",       "P を何回引き直すか。>1 なら full_std/（引き直し間の標準偏差 [HU]、R-S21 の判定）も書く。apply_degrade=false なら 1 固定"),
    "max_slices":       Key(int,  "--max_slices",    "推論するスライス数の上限。0 で全部"),
    "batch_slices":     Key(int,  "--batch_slices",  "同時に G に通す 512 スライス数"),
    "save_residual":    Key(bool, "--save_residual", "R = 出力 − 入力 と、P の寄与 u − x を保存"),
    "save_panel":       Key(bool, "--save_panel",    "[出力 | P(x) | 入力 | R] パネルを保存"),
    "eps_zero":         Key(bool, "--eps_zero",      "v4 の run のみ: ε = 0（z は draw 0 と同じ）で生成した u を full_eps0/ にも書く（指紋 = 決定的に残るノイズ。§3.3）。apply_degrade=false や v3 の run では無視（恒等の確認 infer --apply_degrade false はこの値に関係なく通る）"),
    "io.read_workers":  Key(int,  "--read_workers",  "PNG の読み込みを先読みするスレッド数（0 = 直列）"),
    "io.write_workers": Key(int,  "--write_workers", "PNG の書き込みを非同期にするスレッド数（0 = 直列）"),
}

# ---------------------------------------------------------------------------
# configs/denoise.yaml — v4 の Dn（PCD512 の denoiser）の学習・合否（train_denoiser.py / check_denoiser.py / denoise_dir.py）。dn.method で 3 方式を切り替える:
#   n2n       : Neighbor2Neighbor（Huang et al., CVPR 2021, arXiv 2101.02824）。CT 向けの変更はセル幅 s と 2 画素の最小距離 d₀ をノイズ相関長
#               （bash start1_v3.sh <machine> stats の noise_corr_d0）から決めることだけ（決定 §6.2）。初回（cell 4 / min_dist 2）は実 PCD で残留 SD 44 HU・edge_ratio p90 0.68 で不合格:
#               学習は 4 倍 sub-sample 画像（ノイズが白く見える）で、推論はフル解像度（ノイズが 1〜2 px の塊）なので、相関成分が残る
#   blindspot : 構造化 blind-spot（Noise2Void: Krull et al., CVPR 2019, arXiv 1811.10980 のマスク方式を、Structured Noise2Void: Broaddus et al., ISBI 2020,
#               arXiv 2001.01689 の円盤盲点に拡張）。フル解像度で学習・推論するので上の不一致が無い。本命
#   bm3d      : BM3D（PyPI bm3d、Mäkinen / Azzari / Foi）に stats で測ったノイズ PSD を渡す固定 denoiser（学習なし、対照）。学習ステップ内では回せないので
#               bash start1_v3.sh <machine> dnfixed でデータを先に denoise し、train は pipeline.dn_source precomputed で読む
# ---------------------------------------------------------------------------
DENOISE = {
    "dn.method":             Key(str,   "--method",          "n2n | blindspot | bm3d（上の説明）。n2n.* は n2n だけ、blindspot.* は blindspot だけ、bm3d.* は bm3d だけが使う（他方式では値は無視するがキーは要る）"),
    "data.input_dir":        Key(str,   "--input_dir",       "PCD512 のフォルダ（コンテナ内パス。<case>/<slice>.png）。実 EID には掛けないので PCD だけ"),
    "data.use_eval_only_cases": Key(bool, "--use_eval_only_cases", "false なら configs/eval_only.yaml の評価専用症例を学習から除外（train.yaml と同じ規則）"),
    "data.patch_size":       Key(int,   "--patch_size",      "patch の一辺（§6.2 は 256）。cell_size で割り切れ、割った後が 2^n_levels の倍数であること"),
    "data.patches_per_slice": Key(int,  "--patches_per_slice", "1 スライスから連続して切る patch 数（全 Stage 8）"),
    "data.samples_per_epoch": Key(int,  "--samples_per_epoch", "1 epoch の patch 数（batch_size の倍数）"),
    "data.hu_offset":        Key(int,   "--hu_offset",       "stored = HU + hu_offset（train.yaml と同じ 1400）"),
    "data.hu_min":           Key(int,   "--hu_min",          "正規化窓の下限 HU（train.yaml と同じ。Dn は v4 の G と同じ正規化単位で動く）"),
    "data.hu_max":           Key(int,   "--hu_max",          "正規化窓の上限 HU（train.yaml と同じ）"),
    "data.patch_seed":       Key(int,   "--patch_seed",      "preview / check 用の固定スライス選択と sub-sample の乱数 seed"),
    "n2n.cell_size":         Key(int,   "--cell_size",       "n2n のみ: Neighbor2Neighbor のセル幅 s [px]（偶数）。s ≥ 2·d₀。PCD の見込み 4〜6（§6.1）"),
    "n2n.min_dist":          Key(int,   "--min_dist",        "n2n のみ: セル内で選ぶ 2 画素の最小 Chebyshev 距離 d₀ [px]（1 ≤ d₀ < s）。|ρ(d)| < 0.05 になる最小距離（stats の noise_corr_d0）"),
    "n2n.gamma_max":         Key(float, "--gamma_max",       "n2n のみ: 正則化項の重み γ の最終値（論文既定 2）。blindspot には γ が無い（無視）"),
    "n2n.gamma_ramp_epochs": Key(int,   "--gamma_ramp_epochs", "n2n のみ: γ を 0 から gamma_max へ線形に上げる epoch 数（§6.2: 全体の 1/4）。0 なら最初から gamma_max"),
    "blindspot.radius":      Key(int,   "--bs_radius",       "blindspot のみ: 盲点の半径 r [px]。中心からユークリッド距離 ≤ r の円盤を隠す（Broaddus 2020 の structured blind spot）。stats の noise_corr_d0 − 1 以上（PCD 実測 d₀ = 6 → r = 3 で ρ ≤ 0.09 の殻まで隠す）"),
    "blindspot.stride":      Key(int,   "--bs_stride",       "blindspot のみ: 中心の格子間隔 [px]（画像ごとに一様乱数の全体オフセット）。≥ 2r + 2 で円盤が重ならない。中心密度 = 1/stride²（N2V の 0.5〜2 % 相当 = 8〜12）"),
    "blindspot.fill":        Key(str,   "--bs_fill",         "blindspot のみ: 隠した画素の埋め方。neighbor = 同じ中心の環 r < d ≤ 2r+1 から画素ごとに乱択した値（隣の円盤に入る位置は候補から外す。N2V の random-neighbour 方式）。今は neighbor だけ"),
    "blindspot.loss":        Key(str,   "--bs_loss",         "blindspot のみ: 中心画素だけで取る損失 l2 | l1（N2V は L2）"),
    "bm3d.psd_path":         Key(str,   "--psd_path",        "bm3d のみ: stats が書くノイズ PSD の .npy（output/stage1_v3/stats_<時刻>/noise_psd_<入力名>.npy。画像と同じ H×W、bm3d の規約 = E|FFT2(noise)|²、DC が [0,0]）。空文字は不可（dnfixed / dncheck --method bm3d で存在必須）"),
    "bm3d.profile":          Key(str,   "--bm3d_profile",    "bm3d のみ: bm3d.bm3d(z, sigma_psd, profile) にそのまま渡す np | refilter"),
    "bm3d.sigma_scale":      Key(float, "--sigma_scale",     "bm3d のみ: 実測 PSD を sigma_scale² 倍して渡す（ノイズ振幅を sigma_scale 倍と見なす）。1.0 = 実測どおり。> 1 で強めに除去（残留 SD ↓、細い構造を削るおそれ → dncheck の resid_edge_energy で見る）"),
    "model.ch":              Key(int,   "--ch",              "U-Net の基本チャネル数（論文 48）"),
    "model.n_levels":        Key(int,   "--n_levels",        "U-Net の down / up の段数（2 倍ずつ。論文は 3 段相当）"),
    "optim.lr":              Key(float, "--lr",              "Adam の学習率（論文 3e-4）"),
    "optim.lr_half_epoch":   Key(int,   "--lr_half_epoch",   "この epoch から（この epoch を含む）lr を 1/2 にする（§6.2: 半分で 1/2。epoch は 1 始まり）。0 で一定"),
    "optim.beta1":           Key(float, "--beta1",           "Adam β1（論文は Adam 既定 0.9）"),
    "optim.beta2":           Key(float, "--beta2",           "Adam β2（0.999）"),
    "optim.n_epochs":        Key(int,   "--n_epochs",        "epoch 数（§6.2: 約 50）"),
    "optim.batch_size":      Key(int,   "--batch_size",      "batch の patch 数（論文 16）"),
    "optim.seed":            Key(int,   "--seed",            "python / numpy / torch の乱数 seed"),
    "optim.precision":       Key(str,   "--precision",       "fp32 | bf16（CUDA autocast。ProjectionModel と同じ扱い）"),
    "optim.compile":         Key(bool,  "--compile",         "CUDA で torch.compile(mode='max-autotune')（CPU では無効）"),
    "log.print_freq":        Key(int,   "--print_freq",      "TensorBoard scalar と loss_log.txt の間隔（step）"),
    "log.save_epoch_freq":   Key(int,   "--save_epoch_freq", "weights/epoch_NNN/net_Dn.pth を残す epoch 間隔（latest/ は毎 epoch）"),
    "log.n_preview":         Key(int,   "--n_preview",       "epoch ごとの preview と check/* に使う固定スライス数（症例列挙から等間隔）"),
    "log.display_hu_min":    Key(int,   "--display_hu_min",  "表示の線形範囲の下限 HU（−1400）"),
    "log.display_hu_max":    Key(int,   "--display_hu_max",  "表示の線形範囲の上限 HU（3000）"),
    "log.diff_range_hu":     Key(int,   "--diff_range_hu",   "残差 x − Dn(x) のカラー表示の ±範囲 HU（PCD のノイズ SD ≈ 93 HU）"),
    "check.n_slices":        Key(int,   "--n_slices",        "check_denoiser.py（dncheck）で測るスライス数（症例列挙から等間隔）"),
    "check.body_threshold_hu": Key(int, "--body_threshold_hu", "一様 ROI・エッジを測る体内判定の HU 閾値（train.yaml degrade.body_threshold_hu と同じ −500）"),
    "check.resid_sd_max_hu": Key(float, "--resid_sd_max_hu", "合格: Dn(x) の一様 ROI 局所 SD の中央値 ≤ これ [HU]（§6.4: 実 EID 22.7 の 1/4 = 6）"),
    "check.edge_ref_p90_hu": Key(float, "--edge_ref_p90_hu", "合格: Dn(x) の体内 edge_grad p90 の中央値 ≥ これ [HU/px]（§6.4 改訂: 実 EID の中央値。stats の EID_v5 の edge_grad_p90。x との比は x がノイズ勾配で膨らむので使わない）"),
    "check.edge_ref_p95_hu": Key(float, "--edge_ref_p95_hu", "合格: Dn(x) の体内 edge_grad p95 の中央値 ≥ これ [HU/px]（実 EID の中央値）"),
    "check.edge_ref_p99_hu": Key(float, "--edge_ref_p99_hu", "合格: Dn(x) の体内 edge_grad p99 の中央値 ≥ これ [HU/px]（実 EID の中央値）"),
    "check.resid_edge_energy_max": Key(float, "--resid_edge_energy_max", "合格: 残差 x − Dn(x) の縁への集中度（Dn(x) の体内 |∇| 上位 5 % での RMS ÷ 縁から 2 px 以上離れた体内の RMS）の中央値 ≤ これ（§6.4 改訂: 1.2。ノイズだけ除いていれば 1、縁を鈍らせると > 1）"),
}

PIPELINE_MODES = ("v3", "v4")
DN_SOURCES = ("weights", "precomputed")
DN_METHODS = ("n2n", "blindspot", "bm3d")
BLINDSPOT_FILLS = ("neighbor",)
BLINDSPOT_LOSSES = ("l2", "l1")
BM3D_PROFILES = ("np", "refilter")
GAN_MODES = ("fgan_kl", "lsgan", "vanilla")
DEGRADE_MODES = ("gaussian", "bilateral_blur", "learned")
PRECISIONS = ("fp32", "bf16")
NETGS = ("wavelet", "resnet_9blocks", "resnet_6blocks")
NETDS = ("paper", "basic")
NORMS = ("instance", "batch")
INIT_TYPES = ("normal", "xavier", "kaiming", "orthogonal")
GPU_GENS = (0, 30, 40, 50)
SAMPLINGS = ("random", "paper")

# この 2 項目が追加される前の run（2026-10-30 の初回 run）を resume するためだけの移行値。新規 run の値は train.yaml にあり、設定欠落の暗黙補完には使わない
LEGACY_TRAIN_DEFAULTS = {"optim.precision": "fp32", "optim.compile": False,
                         "degrade.mode": "gaussian", "degrade.bilateral_sigma_color_hu_min": 0.0, "degrade.bilateral_sigma_color_hu_max": 0.0,
                         "degrade.bilateral_sigma_space_px_min": 0.0, "degrade.bilateral_sigma_space_px_max": 0.0,
                         "degrade.learned_z_dim": 16, "degrade.learned_kernel_size": 21, "degrade.learned_noise_kernel_size": 9, "degrade.learned_amp_ch": 16, "degrade.learned_sigma_max_hu": 100.0,
                         "loss.lambda_k_fid": 0.0, "loss.lambda_k_com": 0.0,
                         "pipeline.mode": "v3", "pipeline.dn_weights": "", "degrade.learned_sigma_min_hu": 0.0, "val.target_fingerprint_sd_hu": 0.0,
                         "pipeline.dn_source": "weights", "pipeline.dn_dir": ""}

# 重みの形と P の有無を決めるキー。再開で変えられない
SHAPE_KEYS = ("pipeline.mode", "pipeline.dn_source", "network.netG", "network.netD", "network.ngf", "network.ndf", "network.norm", "degrade.enabled", "degrade.mode",
              "degrade.learned_z_dim", "degrade.learned_kernel_size", "degrade.learned_noise_kernel_size", "degrade.learned_amp_ch")


# ---------------------------------------------------------------------------
# 検証・変換（stage1_v2/configs/schema.py と同じ規則）
# ---------------------------------------------------------------------------
class ConfigError(Exception):
    pass


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def _type_ok(value, typ):
    if typ is bool:
        return isinstance(value, bool)
    if typ is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if typ is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if typ is str:
        return isinstance(value, str)
    if typ is list:
        return isinstance(value, list) and all(isinstance(x, str) and x for x in value)
    raise TypeError(typ)


def _coerce(value, typ):
    if typ is float:
        return float(value)
    if typ is list:
        return list(value)
    return value


def validate(name, data, schema):
    if not isinstance(data, dict):
        raise ConfigError(f"[{name}] トップレベルが dict ではありません: {type(data).__name__}")
    flat = flatten(data)
    missing = [k for k in schema if k not in flat]
    unknown = [k for k in flat if k not in schema]
    bad = [f"{k}: 期待 {schema[k].type.__name__}, 実際 {type(flat[k]).__name__} ({flat[k]!r})" for k in flat if k in schema and not _type_ok(flat[k], schema[k].type)]
    problems = []
    if missing:
        problems.append("必須キーがありません: " + ", ".join(missing))
    if unknown:
        problems.append("未知のキーがあります（typo?）: " + ", ".join(unknown))
    if bad:
        problems.append("型が違います: " + "; ".join(bad))
    if problems:
        raise ConfigError(f"[{name}] " + " / ".join(problems))
    return {k: _coerce(flat[k], schema[k].type) for k in schema}


def validate_shared(name, data, schema):
    if not isinstance(data, dict):
        raise ConfigError(f"[{name}] トップレベルが dict ではありません: {type(data).__name__}")
    flat = flatten(data)
    missing = [k for k in schema if k not in flat]
    bad = [f"{k}: 期待 {schema[k].type.__name__}, 実際 {type(flat[k]).__name__} ({flat[k]!r})" for k in schema if k in flat and not _type_ok(flat[k], schema[k].type)]
    problems = []
    if missing:
        problems.append("必須キーがありません: " + ", ".join(missing))
    if bad:
        problems.append("型が違います: " + "; ".join(bad))
    if problems:
        raise ConfigError(f"[{name}] " + " / ".join(problems))
    return {k: _coerce(flat[k], schema[k].type) for k in schema}


def to_argv(flat, schema):
    argv = []
    for k, key in schema.items():
        if key.flag is None:
            continue
        v = flat[k]
        if key.type is bool:
            if v:
                argv.append(key.flag)
        elif key.type is list:
            argv += [key.flag, ",".join(v)]
        else:
            argv += [key.flag, str(v)]
    return argv


def _parse_bool(val, flag):
    v = str(val).strip().lower()
    if v in ("true", "1", "yes"):
        return True
    if v in ("false", "0", "no"):
        return False
    raise ConfigError(f"{flag} は true / false で指定してください: {val!r}")


def _convert(val, typ, flag):
    try:
        if typ is int:
            if isinstance(val, str) and ("." in val or "e" in val.lower()):
                raise ValueError
            return int(val)
        if typ is float:
            return float(val)
        if typ is str:
            return str(val)
        if typ is bool:
            return _parse_bool(val, flag)
        if typ is list:
            if str(val).strip() == "":
                return []
            items = [x.strip() for x in str(val).split(",")]
            if any(not x for x in items):
                raise ValueError
            return items
    except ValueError:
        raise ConfigError(f"{flag} の値を {typ.__name__} に変換できません: {val!r}")
    raise TypeError(typ)


def apply_overrides(tokens, sections):
    """sh からの上書き列を平坦 dict に反映する（後勝ち）。schema に無いフラグ・値の無いフラグ・型違いは ConfigError。戻り値は適用した {flag: 値}。"""
    index = {}
    for sec, (flat, schema) in sections.items():
        for key_name, key in schema.items():
            if key.flag:
                index[key.flag] = (sec, key_name, key)
    applied = {}
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if not t.startswith("--"):
            raise ConfigError(f"上書き引数の形式が不正です（--flag value の並びにしてください）: {t!r}")
        if "=" in t:
            flag, val = t.split("=", 1)
            has_val = True
        else:
            flag, val, has_val = t, None, False
        if flag not in index:
            raise ConfigError("上書きできないフラグです（schema.py に無い）: " + flag)
        sec, key_name, key = index[flag]
        flat = sections[sec][0]
        if key.type is bool:
            if not has_val and i + 1 < len(tokens) and tokens[i + 1].lower() in ("true", "false", "1", "0", "yes", "no"):
                val = tokens[i + 1]
                i += 1
            v = True if val is None else _parse_bool(val, flag)
        else:
            if not has_val:
                if i + 1 >= len(tokens) or tokens[i + 1].startswith("--"):
                    raise ConfigError(f"{flag} には値が必要です")
                val = tokens[i + 1]
                i += 1
            v = _convert(val, key.type, flag)
        flat[key_name] = v
        applied[flag] = v
        i += 1
    return applied


# ---------------------------------------------------------------------------
# 値域・相互条件の検証（型検査の後、実効値に対して）
# ---------------------------------------------------------------------------
def _problems_to_error(where, problems):
    if problems:
        raise ConfigError(f"[{where}] " + " / ".join(problems))


def _check_machine(machine, P):
    if machine["gpu_gen"] not in GPU_GENS: P.append(f"gpu_gen は {GPU_GENS}")
    if machine["num_threads"] < 0: P.append("num_threads ≥ 0")
    if not (1 <= machine["tb_port"] <= 65535): P.append("tb_port は 1..65535")


def check_degrade_values(train, P):
    d = {k.split(".", 1)[1]: v for k, v in train.items() if k.startswith("degrade.")}
    if d["mode"] not in DEGRADE_MODES: P.append(f"degrade.mode は {DEGRADE_MODES}")
    for lo, hi in (("blur_sigma_px_min", "blur_sigma_px_max"), ("noise_sigma_hu_min", "noise_sigma_hu_max"), ("noise_corr_px_min", "noise_corr_px_max"),
                   ("bilateral_sigma_color_hu_min", "bilateral_sigma_color_hu_max"), ("bilateral_sigma_space_px_min", "bilateral_sigma_space_px_max")):
        if d[lo] < 0: P.append(f"degrade.{lo} ≥ 0")
        if d[hi] < d[lo]: P.append(f"degrade.{lo} ≤ degrade.{hi}")
    if d["enabled"] and d["mode"] == "gaussian" and d["blur_sigma_px_max"] == 0 and d["noise_sigma_hu_max"] == 0:
        P.append("degrade.enabled なのにぼかしもノイズも上限 0（P が恒等になる。--degrade false にするか範囲を指定）")
    if d["mode"] == "learned":
        if d["learned_z_dim"] < 1: P.append("degrade.learned_z_dim ≥ 1")
        for k in ("learned_kernel_size", "learned_noise_kernel_size"):
            if d[k] < 3 or d[k] % 2 == 0: P.append(f"degrade.{k} は 3 以上の奇数")
        if d["learned_amp_ch"] < 1: P.append("degrade.learned_amp_ch ≥ 1")
        if d["learned_sigma_max_hu"] <= 0: P.append("degrade.learned_sigma_max_hu は正")
    if d["enabled"] and d["mode"] == "bilateral_blur" and (d["bilateral_sigma_color_hu_max"] <= 0 or d["bilateral_sigma_space_px_max"] <= 0):
        P.append("degrade.mode bilateral_blur では bilateral_sigma_color_hu_max / sigma_space_px_max を正に")


def check_pipeline_values(train, P):
    mode = train["pipeline.mode"]
    if mode not in PIPELINE_MODES:
        P.append(f"pipeline.mode は {PIPELINE_MODES}")
        return
    if train["pipeline.dn_source"] not in DN_SOURCES:
        P.append(f"pipeline.dn_source は {DN_SOURCES}")
        return
    if mode == "v4":
        if not train["degrade.enabled"] or train["degrade.mode"] != "learned": P.append("pipeline.mode v4 は degrade.enabled true かつ degrade.mode learned（k(z)・σ・h(z) を使う）")
        src = train["pipeline.dn_source"]
        if src == "weights":
            if not train["pipeline.dn_weights"]: P.append("pipeline.mode v4 + dn_source weights は pipeline.dn_weights（net_Dn.pth のパス）が必要（bash start1_v3.sh <machine> dn で作る）")
            if train["pipeline.dn_dir"] != "": P.append('dn_source weights では pipeline.dn_dir は空文字 "" にする')
        elif src == "precomputed":
            if not train["pipeline.dn_dir"]: P.append("pipeline.mode v4 + dn_source precomputed は pipeline.dn_dir（dnfixed の出力）が必要（bash start1_v3.sh <machine> dnfixed で作る）")
            if train["pipeline.dn_weights"] != "": P.append('dn_source precomputed では pipeline.dn_weights は空文字 "" にする（Dn は学習中に掛けない）')
        smin, smax = train["degrade.learned_sigma_min_hu"], train["degrade.learned_sigma_max_hu"]
        if smin < 0: P.append("degrade.learned_sigma_min_hu ≥ 0")
        if not smin < smax: P.append(f"degrade.learned_sigma_min_hu ({smin}) < degrade.learned_sigma_max_hu ({smax})")
        if train["val.target_fingerprint_sd_hu"] <= 0: P.append("v4 では val.target_fingerprint_sd_hu は正")
    else:
        if train["pipeline.dn_weights"] != "": P.append('pipeline.mode v3 では pipeline.dn_weights は空文字 "" にする（Dn を使わない）')
        if train["pipeline.dn_dir"] != "": P.append('pipeline.mode v3 では pipeline.dn_dir は空文字 "" にする（Dn を使わない）')
        if train["pipeline.dn_source"] != "weights": P.append('pipeline.mode v3 では pipeline.dn_source は weights（Dn を使わないので既定の綴りだけ許す）')
        if train["val.target_fingerprint_sd_hu"] < 0: P.append("val.target_fingerprint_sd_hu ≥ 0")


def check_train_values(train, machine):
    P = []
    check_pipeline_values(train, P)
    bs = train["optim.batch_size"]
    for k in ("lr_G", "lr_D"):
        if train[f"optim.{k}"] <= 0: P.append(f"optim.{k} は正")
    if not (0.0 <= train["optim.beta1"] < 1.0 and 0.0 <= train["optim.beta2"] < 1.0): P.append("optim.beta1 / beta2 は [0, 1)")
    if bs < 1: P.append("optim.batch_size ≥ 1")
    if train["optim.n_epochs"] < 1: P.append("optim.n_epochs ≥ 1")
    if train["optim.n_epochs_decay"] < 0: P.append("optim.n_epochs_decay ≥ 0")
    if train["optim.seed"] < 0: P.append("optim.seed ≥ 0")
    if train["optim.precision"] not in PRECISIONS: P.append(f"optim.precision は {PRECISIONS}")
    if train["loss.gan_mode"] not in GAN_MODES: P.append(f"loss.gan_mode は {GAN_MODES}")
    for k in ("lambda_fid", "lambda_id", "lambda_id_final", "lambda_k_fid", "lambda_k_com"):
        if train[f"loss.{k}"] < 0: P.append(f"loss.{k} ≥ 0")
    if train["loss.lambda_id_decay_epochs"] < 0: P.append("loss.lambda_id_decay_epochs ≥ 0")
    check_degrade_values(train, P)
    if train["network.netG"] not in NETGS: P.append(f"network.netG は {NETGS}")
    if train["network.netD"] not in NETDS: P.append(f"network.netD は {NETDS}")
    if train["network.ngf"] < 1 or train["network.ndf"] < 1: P.append("network.ngf / ndf ≥ 1")
    if train["network.norm"] not in NORMS: P.append(f"network.norm は {NORMS}")
    if train["network.init_type"] not in INIT_TYPES: P.append(f"network.init_type は {INIT_TYPES}")
    if train["network.init_gain"] <= 0: P.append("network.init_gain は正")
    if train["data.sampling"] not in SAMPLINGS: P.append(f"data.sampling は {SAMPLINGS}")
    ps = train["data.patch_size"]
    if ps < 8 or ps % 8: P.append("data.patch_size は 8 の倍数（wavelet G の 3 段 Haar）")
    if train["data.patch_stride"] < 1: P.append("data.patch_stride ≥ 1")
    if train["data.patches_per_image"] < 1: P.append("data.patches_per_image ≥ 1")
    if train["data.sampling"] == "paper" and train["data.patch_stride"] != 8: P.append("paper sampling の patch_stride は 8")
    if train["data.sampling"] == "paper" and train["data.patches_per_slice"] != 1: P.append("paper sampling では patches_per_slice = 1")
    spe = train["data.samples_per_epoch"]
    if spe < bs or spe % bs: P.append(f"data.samples_per_epoch ({spe}) は optim.batch_size ({bs}) の倍数で batch_size 以上")
    if train["data.patches_per_slice"] < 1: P.append("data.patches_per_slice ≥ 1")
    if train["data.hu_min"] >= train["data.hu_max"]: P.append("data.hu_min < data.hu_max")
    if train["log.print_freq"] < 1 or train["log.image_freq"] < 1: P.append("log.print_freq / image_freq ≥ 1")
    if train["log.n_images"] < 1: P.append("log.n_images ≥ 1")
    if not train["log.full_slice"]: P.append("log.full_slice を指定")
    if not train["log.eid_slice"]: P.append("log.eid_slice を指定")
    if train["log.n_full_random"] < 0: P.append("log.n_full_random ≥ 0")
    if train["log.n_draws"] < 1: P.append("log.n_draws ≥ 1")
    if train["log.display_hu_min"] >= train["log.display_hu_max"]: P.append("log.display_hu_min < display_hu_max")
    if train["log.preview_bits"] not in (8, 16): P.append("log.preview_bits は 8 | 16")
    if train["log.diff_range_hu"] <= 0: P.append("log.diff_range_hu は正")
    if train["log.tb_full_size"] < 64: P.append("log.tb_full_size ≥ 64")
    if train["log.save_epoch_freq"] < 1: P.append("log.save_epoch_freq ≥ 1")
    if train["val.n_pcd_slices"] < 0 or train["val.n_eid_slices"] < 0: P.append("val.n_pcd_slices / n_eid_slices ≥ 0")
    if train["val.n_pcd_slices"] > 0 and train["val.n_eid_slices"] < 8: P.append("採点するなら val.n_eid_slices ≥ 8（四分位を取るため）")
    if train["val.target_ry_rms_hu"] <= 0: P.append("val.target_ry_rms_hu は正")
    slf = train["log.save_latest_freq"]
    if slf < bs or slf % bs: P.append(f"log.save_latest_freq ({slf}) は optim.batch_size ({bs}) の倍数")
    _check_machine(machine, P)
    _problems_to_error("train values", P)


def check_infer_values(infer, machine):
    P = []
    if infer["n_draws"] < 1: P.append("n_draws ≥ 1")
    if not infer["apply_degrade"] and infer["n_draws"] != 1: P.append("apply_degrade=false のとき n_draws は 1（P を掛けないので引き直しは無意味）")
    if infer["max_slices"] < 0: P.append("max_slices ≥ 0（0 で全部）")
    if infer["batch_slices"] < 1: P.append("batch_slices ≥ 1")
    if infer["io.read_workers"] < 0 or infer["io.write_workers"] < 0: P.append("io.read_workers / write_workers ≥ 0")
    _check_machine(machine, P)
    _problems_to_error("infer values", P)


def check_denoise_values(dn, machine):
    P = []
    if not dn["data.input_dir"]: P.append("data.input_dir を指定")
    method = dn["dn.method"]
    if method not in DN_METHODS: P.append(f"dn.method は {DN_METHODS}")
    ps, s, d0 = dn["data.patch_size"], dn["n2n.cell_size"], dn["n2n.min_dist"]
    if s < 2 or s % 2: P.append("n2n.cell_size は 2 以上の偶数")
    if d0 < 1 or (s >= 2 and d0 >= s): P.append("n2n.min_dist は 1 ≤ d₀ < cell_size")
    if ps < 8: P.append("data.patch_size ≥ 8")
    if dn["model.n_levels"] < 1: P.append("model.n_levels ≥ 1")
    if dn["model.ch"] < 1: P.append("model.ch ≥ 1")
    m = 2 ** dn["model.n_levels"]
    if method == "n2n":  # sub-sample 後の一辺が U-Net の down / up に合うこと
        if s >= 2 and ps % s: P.append(f"data.patch_size ({ps}) は n2n.cell_size ({s}) の倍数（sub-sample で H/s × W/s にする）")
        if s >= 2 and dn["model.n_levels"] >= 1 and (ps // s) % m: P.append(f"sub-sample 後の一辺 patch_size / cell_size = {ps // s} は 2^n_levels = {m} の倍数（U-Net の down / up）")
    r, st = dn["blindspot.radius"], dn["blindspot.stride"]
    if not 1 <= r <= 10: P.append("blindspot.radius は 1〜10（対角の格子点の円盤から置換値を取らない条件 2.8·r + 2.8 > 3·r + 1 が r ≤ 10 で成り立つ）")
    if st < 2 * r + 2: P.append(f"blindspot.stride ({st}) ≥ 2·radius + 2 = {2 * r + 2}（円盤が重ならない）")
    if dn["blindspot.fill"] not in BLINDSPOT_FILLS: P.append(f"blindspot.fill は {BLINDSPOT_FILLS}")
    if dn["blindspot.loss"] not in BLINDSPOT_LOSSES: P.append(f"blindspot.loss は {BLINDSPOT_LOSSES}")
    if method == "blindspot":  # フル解像度で U-Net に通す
        if ps % m: P.append(f"blindspot では data.patch_size ({ps}) は 2^n_levels = {m} の倍数（フル解像度で U-Net に通す）")
        if ps < 2 * (2 * r + 1) + st: P.append(f"blindspot では data.patch_size ({ps}) ≥ 2·(2r+1) + stride = {2 * (2 * r + 1) + st}（環が画像内に収まる中心が 1 つは要る）")
    if dn["bm3d.profile"] not in BM3D_PROFILES: P.append(f"bm3d.profile は {BM3D_PROFILES}")
    if not (dn["bm3d.sigma_scale"] > 0): P.append("bm3d.sigma_scale は正（1.0 = 実測 PSD どおり）")
    if method == "bm3d" and not dn["bm3d.psd_path"]: P.append("dn.method bm3d では bm3d.psd_path（stats の noise_psd_<入力名>.npy）を指定（空文字は不可）")
    if dn["data.patches_per_slice"] < 1: P.append("data.patches_per_slice ≥ 1")
    bs = dn["optim.batch_size"]
    if bs < 1: P.append("optim.batch_size ≥ 1")
    spe = dn["data.samples_per_epoch"]
    if spe < bs or spe % bs: P.append(f"data.samples_per_epoch ({spe}) は optim.batch_size ({bs}) の倍数で batch_size 以上")
    if dn["data.hu_min"] >= dn["data.hu_max"]: P.append("data.hu_min < data.hu_max")
    if dn["n2n.gamma_max"] < 0: P.append("n2n.gamma_max ≥ 0")
    if dn["n2n.gamma_ramp_epochs"] < 0: P.append("n2n.gamma_ramp_epochs ≥ 0")
    if dn["optim.lr"] <= 0: P.append("optim.lr は正")
    if dn["optim.lr_half_epoch"] < 0: P.append("optim.lr_half_epoch ≥ 0")
    if not (0.0 <= dn["optim.beta1"] < 1.0 and 0.0 <= dn["optim.beta2"] < 1.0): P.append("optim.beta1 / beta2 は [0, 1)")
    if dn["optim.n_epochs"] < 1: P.append("optim.n_epochs ≥ 1")
    if dn["optim.seed"] < 0: P.append("optim.seed ≥ 0")
    if dn["optim.precision"] not in PRECISIONS: P.append(f"optim.precision は {PRECISIONS}")
    if dn["log.print_freq"] < 1: P.append("log.print_freq ≥ 1")
    if dn["log.save_epoch_freq"] < 1: P.append("log.save_epoch_freq ≥ 1")
    if dn["log.n_preview"] < 1: P.append("log.n_preview ≥ 1（check/* と preview に要る）")
    if dn["log.display_hu_min"] >= dn["log.display_hu_max"]: P.append("log.display_hu_min < display_hu_max")
    if dn["log.diff_range_hu"] <= 0: P.append("log.diff_range_hu は正")
    if dn["check.n_slices"] < 1: P.append("check.n_slices ≥ 1")
    if dn["check.resid_sd_max_hu"] <= 0: P.append("check.resid_sd_max_hu は正")
    for q in ("p90", "p95", "p99"):
        if dn[f"check.edge_ref_{q}_hu"] <= 0: P.append(f"check.edge_ref_{q}_hu は正")
    if not (dn["check.edge_ref_p90_hu"] <= dn["check.edge_ref_p95_hu"] <= dn["check.edge_ref_p99_hu"]): P.append("check.edge_ref_p90_hu ≤ p95 ≤ p99")
    if dn["check.resid_edge_energy_max"] < 1.0: P.append("check.resid_edge_energy_max ≥ 1（ノイズだけ除いた残差で 1）")
    _check_machine(machine, P)
    _problems_to_error("denoise values", P)
