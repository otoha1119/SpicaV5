"""stage2/configs/schema.py — Stage 2 の設定ファイル（dataset.yaml / train.yaml / mode.yaml / infer.yaml）と、Stage 2 が読む machines.yaml のキーの唯一の正。

原則は Stage 1（stage1/configs/schema.py）と同じ: 既定値（フォールバック）は無い。ここに宣言したキーは**必ず**設定ファイルに存在しなければならず、
未知のキー・型違いはエラー。sh からの上書き（--flag value）は schema にあるフラグだけ受け付け、実効値を manifest / launch.yaml に残す。

検証・上書きの関数（validate / apply_overrides / flatten）は stage1/configs/schema.py と同じものを 2026-09-08 に複製した
（Stage 間で import し合わない。両方の `configs` パッケージが名前衝突するため）。Stage 2 で足したもの: 型 `list`（str のリスト。上書きはカンマ区切り）。

machines.yaml は Stage 共通ファイルなので、Stage 2 は「自分が使うキー」（MACHINE）の存在と型だけ検査し、Stage 1 だけのキー
（pcd_dir / eid_dir / align_meta / checkpoints_dir ...）は無視する（validate_shared）。Stage 1 側は未知キーを拒むので、Stage 2 専用のキー
（pcd1024_dir / eidlike1024_dir / stage2_checkpoints_dir / stage2_tb_port）は stage1/configs/schema.py の MACHINE にも flag None で宣言してある。キーを足すときは両方に書く。

Stage 2 の train.py は junyanz ではなく自前なので、起動器（run_train.py）は argv ではなく **解決済みの launch.yaml を train.py に渡す**。
flag は「sh からの上書き」の名前としてだけ使う（Stage 1 の flag = train.py の引数名、とは役割が違う）。

各キーの宣言: Key(type, flag, doc)
  type : int | float | str | bool | list（str のリスト）。float は int の値も受け付ける。bool は YAML の true / false のみ
  flag : 上書き用のフラグ名（--flag value。list は --flag a,b,c）。None は上書き不可
  doc  : 意味と出典（論文 = ILUMENATE, Koons et al., Med Phys 2025;52(7):e17874）
"""

from collections import namedtuple
from pathlib import Path

Key = namedtuple("Key", "type flag doc")


# ---------------------------------------------------------------------------
# stage2/configs/dataset.yaml — データセット作成（make_dataset.py）の方式
# ---------------------------------------------------------------------------
DATASET = {
    "scale":      Key(int, "--scale",      "拡大率。512 → 1024 は 2。2 以上"),
    "interp":     Key(str, "--interp",     "補間: nearest | bilinear | bicubic | area | lanczos（cv2.resize。half-pixel 規約）。学習と推論で同じ値にする"),
    "input_size": Key(int, "--input_size", "入力の一辺（画素）。違えばエラー。512"),
}

# ---------------------------------------------------------------------------
# stage2/configs/train.yaml — 学習パラメータ（数値・リスト）
# ---------------------------------------------------------------------------
TRAIN = {
    # optim
    "optim.lr":           Key(float, "--lr",           '論文 "Adam optimizer with a learning rate of 0.001"'),
    "optim.beta1":        Key(float, "--beta1",        "Adam β1。論文は Keras の Adam を lr 以外既定で使用 → 0.9（Stage 1 の 0.5 は GAN 慣行なので採らない）"),
    "optim.beta2":        Key(float, "--beta2",        "Adam β2。Keras 既定 0.999"),
    "optim.batch_size":   Key(int,   "--batch_size",   "論文未記載。ユーザー決定 2026-09-08: まず 16、様子を見て変える"),
    "optim.n_epochs":     Key(int,   "--n_epochs",     '論文 "Training consisted of 100 epochs"。lr は一定（減衰の記載なし）'),
    "optim.seed":         Key(int,   "--seed",         "python / numpy / torch の乱数 seed（再現条件）。cudnn.benchmark は Stage 1 と同じく有効なので完全決定ではない"),
    # data
    "data.patch_size":        Key(int,  "--patch_size",        '論文 "Patches of size 128 × 128 pixels"（1024 グリッド上）。ユーザー決定 128。4 の倍数（pool 2 回）'),
    "data.samples_per_epoch": Key(int,  "--samples_per_epoch", "1 epoch のサンプル数（random sampling: 症例一様 → スライス一様 → 位置一様、入力と教師は同じ位置）。論文の 1 epoch ≈ 61,101 組に合わせて 64,000"),
    "data.patch_seed":        Key(int,  "--patch_seed",        "serial_batches=true のときの決定的サンプリング用 seed"),
    "data.patches_per_slice": Key(int,  "--patches_per_slice", "1 回読んだペア（入力 1024 + 教師 1024）から連続して切る patch 数。1 = patch ごとに別ペア（2026-10-12 以前）。8 で読み出し・デコードが 1/8（Stage 1 と同じ仕組み。位置は patch ごとに独立。2026-10-12）"),
    "data.hu_offset":         Key(int,  "--hu_offset",         "stored = HU + hu_offset（1400）。Stage 1 と同じ"),
    "data.hu_min":            Key(int,  "--hu_min",            "正規化窓の下限 HU（−1400）→ −1。Stage 1 と同じ [−1,1]（final_act=tanh でも使える範囲）"),
    "data.hu_max":            Key(int,  "--hu_max",            "正規化窓の上限 HU（4096）→ +1"),
    "data.train_cases":       Key(list, "--train_cases",       "学習に使う症例名のリスト（ユーザー決定 2026-09-08: PCD-001〜014）。eidlike1024_dir と pcd1024_dir の両方に存在すること"),
    "data.val_cases":         Key(list, "--val_cases",         "検証（epoch 末の指標）に使う症例名（PCD-015, 016）。学習には使わない。両方に存在すること"),
    "data.test_cases":        Key(list, "--test_cases",        "最終評価用に**最初から避けておく**症例名（PCD-017〜020）。学習・検証で一切読まない。ディスクに無くてもよい"),
    # log
    "log.print_freq":              Key(int, "--print_freq",              "バー末尾の損失・TensorBoard scalar・loss_log.txt の更新間隔（step = batch_size 枚）"),
    "log.image_freq":              Key(int, "--image_freq",              "TensorBoard に学習中の 128 patch グリッド（images/current, images/fixed）を出す間隔（step）。Stage 1 と同じ仕組み（2026-09-09）"),
    "log.n_images":                Key(int, "--n_images",                "その patch グリッドの行数（current = いまのバッチ先頭 n 枚、fixed = log.full_slice から切った固定 n 枚）"),
    "log.save_epoch_freq":         Key(int, "--save_epoch_freq",         "重み（weights/epoch_NNN/）を保存する epoch 間隔。1 = 毎 epoch"),
    "log.save_latest_freq":        Key(int, "--save_latest_freq",        "latest/ を保存する間隔（画像枚数。batch_size の倍数）"),
    "log.val_max_slices_per_case": Key(int, "--val_max_slices_per_case", "epoch 末の検証で val 症例ごとに使うフル 1024 スライスの上限（等間隔に間引く。0 で全部）"),
    "log.best_metric":             Key(str, "--best_metric",             "best/ を自動更新する val の指標: rmse（最小を更新したら）| ssim | psnr（最大を更新したら）。毎 epoch の val の後に判定し、best/ と best.txt を書く。手動は bash start2.sh best <run> <epoch>（2026-09-09）"),
    "log.full_slice":              Key(str, "--full_slice",              "epoch 末にフル 1024 で書き出す固定スライス（eidlike1024_dir / pcd1024_dir からの相対パス。症例は train / val / test のどれかに入っていること。test の PCD-002/PCD-002-236.png、ユーザー決定 2026-09-09。best は val の指標で選ぶ）→ output_images/epoch_NNN/ と preview_fixed_<slice>/"),
    "log.eid_slice":               Key(str, "--eid_slice",               "実 EID のテストスライス（eid_dir = EID_v5 からの相対パス。EID-049/EID-049-079.png）。毎 epoch 512 → ×scale 補間（eidlike1024_dir の manifest.yaml の方式）→ U-Net に通し、固定パネルの 4 列目に出す"),
    "log.preview_bits":            Key(int, "--preview_bits",            "output_images/preview_*/（表示用）のビット深度。16 = 表示範囲を 0..65535 に伸ばす（Stage 1 と同じ）| 8"),
    "log.tb_full_size":            Key(int, "--tb_full_size",            "TensorBoard の images/full/* に出すフル画像の一辺（1024 は大きすぎて拡大が効かないので縮小して出す。512。ディスクの preview_*/ は縮小しない。2026-09-09）"),
    "log.n_full_random":           Key(int, "--n_full_random",           "epoch 末に書き出すランダムスライスの枚数（epoch ごとに別。val の症例だけから。0 で無し）"),
    "log.display_hu_min":          Key(int, "--display_hu_min",          "TensorBoard 画像の線形表示範囲の下限 HU（この値以下を黒）。Stage 1 と同じ −1400"),
    "log.display_hu_max":          Key(int, "--display_hu_max",          "同上の上限 HU（この値以上を白）。Stage 1 と同じ 2100（stored 3500）"),
    # adv — 実 EID 枝（mode.yaml real_eid_branch: true のときだけ使う。false なら読まない。2026-10-13。docs/plans/20260908_stage2-unet-implementation-spec.md §9）
    "adv.lambda_gan":             Key(float, "--lambda_gan",             "U-Net に戻す GAN 損失の重み（U-Net の損失 = mse + lambda_gan · gan_G）。MSE が主役になる小さい値。0.001 で崩壊（PC4 2026-10-13）→ 0.0001。TB diag/gan_ratio = lambda_gan · gan_G / mse は損失の比で勾配の比ではない（diag/grad_norm_mse / gan を見る）"),
    "adv.gan_mode":               Key(str,   "--gan_mode",               "GAN 損失: lsgan（最小二乗）| vanilla（BCE with logits）| hinge（D: max(0,1−D(real)) + max(0,1+D(fake))、G: −D(fake)。G の勾配が D の出力の大きさに比例しない。2026-10-13 追加）。D の最終層に活性化は無い"),
    "adv.warmup_epochs":          Key(int,   "--warmup_epochs",          "この epoch 数の間（通算 epoch ≤ warmup_epochs）は U-Net を MSE だけで更新し GAN 損失を戻さない。D はその間も学習する（本物 = 教師、偽物 = 実 EID 枝の出力）。resume 後も通算（state の epoch）で数える。TB diag/gan_active。2026-10-13（崩壊対策）"),
    "adv.lr_after_warmup":        Key(float, "--lr_after_warmup",        "ウォームアップ終了後（GAN が有効な epoch）の U-Net の lr（SRGAN 系の慣行 1e-4）。0 = 変えない（optim.lr のまま）。resume 時も通算 epoch で判定して適用。2026-10-13"),
    "adv.clip_grad_norm":         Key(float, "--clip_grad_norm",         "U-Net の optimizer.step 前の clip_grad_norm_ の max_norm。0 で無効。クリップ前のノルムは常に TB diag/grad_norm_G。2026-10-13"),
    "adv.clip_grad_norm_D":       Key(float, "--clip_grad_norm_D",       "D 側の clip_grad_norm_ の max_norm。0 で無効。クリップ前のノルムは TB diag/grad_norm_D。2026-10-13"),
    "adv.lr_D":                   Key(float, "--lr_D",                   "D の Adam の学習率（仮 1e-4。U-Net の 1e-3 より小さく）"),
    "adv.beta1_D":                Key(float, "--beta1_D",                "D の Adam β1（GAN 慣行 0.5）"),
    "adv.beta2_D":                Key(float, "--beta2_D",                "D の Adam β2（0.999）"),
    "adv.batch_eid":              Key(int,   "--batch_eid",              "1 step に U-Net → D に通す実 EID 小片の数（仮 = optim.batch_size）"),
    "adv.crop_eid":               Key(int,   "--crop_eid",               "実 EID512 上で切る小片の一辺（画素）。×scale 補間して U-Net に入れる（64 → 128 = 学習 patch と同じ大きさ）。crop_eid × scale は 2^n_pool の倍数"),
    "adv.patches_per_slice_eid":  Key(int,   "--patches_per_slice_eid",  "1 回読んだ実 EID512 から連続して切る小片の数（data.patches_per_slice と同じ仕組み。位置は小片ごとに独立）"),
    "adv.eid_split_file":         Key(str,   "--eid_split_file",         "実 EID（eid_dir）の症例分割 YAML（train / val / test の症例名リスト。stage1/make_split.py が書く形、bash start.sh split で生成。リポジトリ直下からの相対パスか絶対パス）。"
                                                                         "実 EID 枝は **train の症例だけ**から小片を引く。val / test は読まない（例外は監視用 log.eid_slice の 1 枚。その症例は train に入っていてはならない）。eid_dir 直下の症例は全部どれかのリストにあること"),
}

# ---------------------------------------------------------------------------
# stage2/configs/mode.yaml — アルゴリズムのモード切替
# ---------------------------------------------------------------------------
MODE = {
    "arch":           Key(str,  "--arch",           "unet_ilumenate（論文 Fig. 2: Conv3×3+ReLU ×2 を 3 段、max pool 2 回、up-conv 2 回、skip concat、BN なし）"),
    "base_ch":        Key(int,  "--base_ch",        "最初の段のチャネル数。論文 128（段ごとに 2 倍: 128 / 256 / 512）"),
    "n_pool":         Key(int,  "--n_pool",         "max pool の回数。論文 2"),
    "init_type":      Key(str,  "--init_type",      "重み初期化: xavier_uniform（Keras 既定 Glorot uniform + bias 0 に合わせる）| torch（PyTorch 既定のまま）"),
    "loss":           Key(str,  "--loss",           "mse（論文 Eq. 1）| l1（ablation）。正規化空間 [−1,1] で計算"),
    "final_act":      Key(str,  "--final_act",      "最終層の活性化: linear（論文、回帰）| tanh（出力を [−1,1] に縛る変種）"),
    "residual":       Key(bool, "--residual",       "true で 出力 = 入力 + net(入力)（変種。論文は false）"),
    "serial_batches": Key(bool, "--serial_batches", "true で sampler を index 決定的に（再現用）"),
    "real_eid_branch": Key(bool, "--real_eid_branch", "true で実 EID 枝を付ける（2026-10-13。§9）: 実 EID512 の小片 → ×scale → U-Net → D（PatchGAN、models/discriminator.py）が対応の無い PCD1024 の小片（ペア枝の教師）と見分け、"
                                                      "その GAN 損失 × train.yaml adv.lambda_gan を U-Net に戻す。false で MSE だけ（Stage 1 評価用プローブ）。net_D.pth の有無が変わるので resume で変更不可"),
}

# mode.yaml に後から追加したキーの「導入前の run」での意味（Stage 1 の LEGACY_MODE_DEFAULTS と同じ仕組み。2026-10-13）。
# resume / 推論で run の launch（state.pth の config）にキーが無ければ、今の mode.yaml で補わず導入前の意味（その機構なしで学習された重み）にする
LEGACY_MODE_DEFAULTS = {"real_eid_branch": False}

# ---------------------------------------------------------------------------
# configs/machines.yaml（リポジトリ直上、Stage 共通）のうち Stage 2 が使うキー。選択したマシンのエントリに対して検証（他のキーは無視）
# ---------------------------------------------------------------------------
MACHINE = {
    "gpu_gen":               Key(int, None,                      "GPU 世代 30 | 40 | 50 | 0(CPU)。compose の選択（start2.sh）と学習デバイスに使う"),
    "host_data_root":        Key(str, None,                      "Docker 起動時にマウントするホスト側のデータルート（start2.sh が使う）"),
    "container_data_root":   Key(str, None,                      "マウント先（コンテナ内）。dataset の出力先はこの配下に限る"),
    "num_threads":           Key(int, "--num_threads",           "make_dataset.py の worker 数 / 学習の DataLoader worker 数（0 で逐次・メインプロセス）"),
    "stage2_tb_port":        Key(int, None,                      "Stage 2 の TensorBoard のポート（Stage 1 の tb_port とは別。start2.sh が compose に両方渡し、Stage 2 はこちらで起動する）"),
    "pcd1024_dir":           Key(str, "--pcd1024_dir",           "Stage 2 の教師 PCD1024（<case>/<slice>.png、1ch uint16、512 と同じ命名）"),
    "eid_dir":               Key(str, "--eid_dir",               "実 EID512（ImageCAS、EID_v5。Stage 1 の x 側と同じキー）。学習では log.eid_slice の 1 枚だけ読む"),
    "eidlike1024_dir":       Key(str, "--eidlike1024_dir",       "Stage 2 の学習入力。bash start2.sh dataset の出力先（EID-like512 を ×2 補間）で、学習はここを読む。container_data_root 配下"),
    "eid1024_dir":           Key(str, None,                      "実 EID の 1024（Stage 1 の task: sr が使う）。Stage 2 は使わない（machines.yaml は Stage 共通なので宣言だけ。2026-10-12）"),
    "stage2_checkpoints_dir": Key(str, "--stage2_checkpoints_dir", "Stage 2 の run（yyyy_mmdd_HHMM）の保存先ルート。レイアウトは stage2/util/run_paths.py"),
}

# ---------------------------------------------------------------------------
# stage2/configs/infer.yaml — 推論の方式（bash start2.sh infer → run_infer.py → inference_dir.py。フラグは inference_dir.py のもの）
# 重みディレクトリ・入力フォルダは infer_stage2.sh の変数、デバイスは machines.yaml の gpu_gen（0 → cpu）。U-Net の構成と HU 正規化は run の launch.yaml、
# 512 入力の補間方式は run の dataset_info.yaml（train.py が eidlike1024_dir/manifest.yaml から写したもの）から取る
# ---------------------------------------------------------------------------
INFER = {
    "input":            Key(str,  "--input",          "入力の種類: eidlike（PCD 症例の EID-like。1024 か 512。パネルの 1〜3 列目 = 入力 / 出力 / 教師）| eid（実 EID512 = EID_v5。パネルの 4・5 列目 = 出力 / 入力。教師は無いので teacher は false）"),
    "max_slices":       Key(int,  "--max_slices",     "推論するスライス数の上限。0 で全部（1 枚だけ試すなら 1）"),
    "batch_slices":     Key(int,  "--batch_slices",   "同時に U-Net に通すスライス数（1024² × 128ch の活性化は 1 枚で数 GB。学習と GPU を共有する 3070 8GB は 1、96GB なら 4〜8）"),
    "teacher":          Key(bool, "--teacher",        "true で machines.yaml の pcd1024_dir から同じ <case>/<slice>.png を教師として読み、指標（rmse / ssim / psnr、学習の val と同じ）を metrics.txt に書き、パネルの 3 列目に並べる。全スライスに教師が無ければ開始前にエラー。実 EID など教師の無い入力は false"),
    "save_input1024":   Key(bool, "--save_input1024", "512 入力を補間したとき、その 1024 入力も full_input1024/ に 16bit で残すか（1024 入力のときは何もしない）"),
    "save_panel":       Key(bool, "--save_panel",     "true で表示用パネル [EID-like1024 | PCD-like1024 | PCD1024 | 実 EID → PCD-like1024 | 実 EID1024]（学習の固定パネルと同じ 5 列・正規化表示。util/panel.py）を full_panel/ に保存。input=eidlike なら 1〜3 列目が入力（教師は teacher）で 4・5 列目は eid_slice、input=eid なら 4・5 列目が入力で 1〜3 列目は pcd_slice"),
    "eid_slice":        Key(str,  "--eid_slice",      "input=eidlike のとき、パネルの 4・5 列目に並べる実 EID のスライス（machines.yaml の eid_dir からの相対パス。train.yaml log.eid_slice と同じ形。512 なら run の方式で補間して 1 回だけ通し、全スライスに同じものを並べる）。空文字で無し。save_panel のときだけ読む"),
    "pcd_slice":        Key(str,  "--pcd_slice",      "input=eid のとき、パネルの 1〜3 列目に並べる参照の PCD 症例スライス（eidlike1024_dir と pcd1024_dir からの相対パス。train.yaml log.full_slice と同じ形。1 回だけ通し、全スライスに同じものを並べる）。空文字で無し。save_panel のときだけ読む"),
    "io.read_workers":  Key(int,  "--read_workers",   "PNG の読み込み・デコード（512 なら補間も）を先読みするスレッド数（0 = 直列）"),
    "io.write_workers": Key(int,  "--write_workers",  "PNG の書き込みを非同期にするスレッド数（0 = 直列）"),
}

# ---------------------------------------------------------------------------
# stage2/configs/crop.yaml — Stage 3 と同じ固定プラーク領域の局所推論
# pcd_cases / eid_cases は run_crop.py が別途検証する。
# ---------------------------------------------------------------------------
CROP = {
    "crop_size":    Key(int, "--crop_size",    "保存するプラーク領域の一辺。Stage 3 の log.crop_size と同じ 144"),
    "panel_scale":  Key(int, "--panel_scale",  "表示パネルだけを最近傍で拡大する倍率"),
    "device":       Key(str, "--device",       "cpu | cuda"),
    "pcd_index":    Key(int, "--pcd_index",    "pcd_cases の何番目を出すか（1始まり。01の番号にも使う）"),
    "eid_index":    Key(int, "--eid_index",    "eid_cases の何番目を出すか（1始まり。番号は pcd_cases の件数を足す）"),
    "diff_max_hu":  Key(int, "--diff_max_hu",  "絶対HU差ヒートマップの赤になる上限。200"),
    "hist_hu_min":  Key(int, "--hist_hu_min",  "HU分布図の横軸下限"),
    "hist_hu_max":  Key(int, "--hist_hu_max",  "HU分布図の横軸上限"),
    "hist_bin_hu":  Key(int, "--hist_bin_hu",  "HU分布図のbin幅（HU）"),
    "hist_y_max_pct": Key(float, "--hist_y_max_pct", "HU分布図の縦軸上限（1 binあたりの画素割合%）"),
}

CROP_CASE_KEYS = {"slice": str, "x": int, "y": int}


def check_crop_cases(name, cases):
    if not isinstance(cases, list):
        raise ConfigError(f"[crop] {name} は list にしてください")
    out, problems = [], []
    for i, c in enumerate(cases, 1):
        if not isinstance(c, dict):
            problems.append(f"{name}[{i}] は dict")
            continue
        missing = [k for k in CROP_CASE_KEYS if k not in c]
        unknown = [k for k in c if k not in CROP_CASE_KEYS]
        bad = [k for k, t in CROP_CASE_KEYS.items() if k in c and (not isinstance(c[k], t) or (t is int and isinstance(c[k], bool)))]
        if missing: problems.append(f"{name}[{i}] キー不足 {missing}")
        if unknown: problems.append(f"{name}[{i}] 未知キー {unknown}")
        if bad: problems.append(f"{name}[{i}] 型違い {bad}")
        if not missing and not unknown and not bad:
            if c["x"] < 0 or c["y"] < 0: problems.append(f"{name}[{i}] x / y は 0 以上: {c}")
            else: out.append(dict(c))
    _problems_to_error("crop cases", problems)
    return out


INTERPS = ("nearest", "bilinear", "bicubic", "area", "lanczos")
INFER_INPUTS = ("eidlike", "eid")
BEST_METRICS = ("rmse", "ssim", "psnr")   # log.best_metric。rmse は小さいほど良い、ssim / psnr は大きいほど良い
GPU_GENS = (0, 30, 40, 50)
ARCHS = ("unet_ilumenate",)
INIT_TYPES = ("xavier_uniform", "torch")
LOSSES = ("mse", "l1")
FINAL_ACTS = ("linear", "tanh")
GAN_MODES = ("lsgan", "vanilla", "hinge")   # adv.gan_mode（実 EID 枝）


# ---------------------------------------------------------------------------
# 検証・変換（stage1/configs/schema.py と同じ + list 型）
# ---------------------------------------------------------------------------
class ConfigError(Exception):
    pass


def flatten(d, prefix=""):
    """入れ子 dict を 'a.b.c' キーの平坦 dict にする（list は葉）。"""
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
        return isinstance(value, list) and all(isinstance(x, str) for x in value)
    raise TypeError(typ)


def _coerce(value, typ):
    if typ is float:
        return float(value)
    if typ is list:
        return list(value)
    return value


def validate(name, data, schema):
    """平坦化した設定を schema と突き合わせる。欠落・未知・型違いをまとめて ConfigError にする。戻り値は平坦 dict。"""
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


def to_argv(flat, schema):
    """検証済みの平坦 dict を inference_dir.py の引数列にする（Stage 1 と同じ。bool は true のときだけフラグを付ける）。"""
    argv = []
    for k, key in schema.items():
        if key.flag is None:
            continue
        v = flat[k]
        if key.type is bool:
            if v:
                argv.append(key.flag)
        else:
            argv += [key.flag, str(v)]
    return argv


def validate_shared(name, data, schema):
    """Stage 共通ファイル（machines.yaml）用: schema のキーの存在と型だけ検査し、他のキーは無視する。戻り値は schema のキーだけの平坦 dict。"""
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
            items = [x.strip() for x in str(val).split(",")]
            if any(not x for x in items):
                raise ValueError
            return items
    except ValueError:
        raise ConfigError(f"{flag} の値を {typ.__name__} に変換できません: {val!r}" + ("（list はカンマ区切り。空要素は不可）" if typ is list else ""))
    raise TypeError(typ)


def apply_overrides(tokens, sections):
    """sh からの上書き列を平坦 dict に反映する（後勝ち）。
    sections: {"train": (flat_dict, TRAIN), "mode": (flat_dict, MODE), ...}。flat_dict はその場で書き換わる。
    受ける形: --flag value / --flag=value。bool は --flag（= true）または --flag true|false / --flag=false。list は --flag a,b,c。
    schema に無いフラグ・値の無いフラグ・余った語・型違いは ConfigError。戻り値は適用した {flag: 値}（記録用）。"""
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
# 値域・相互条件の検証。型検査（validate）の後、実効値（上書き反映後）に対して行う
# ---------------------------------------------------------------------------
def _problems_to_error(where, problems):
    if problems:
        raise ConfigError(f"[{where}] " + " / ".join(problems))


def _check_machine(machine, P):
    if machine["gpu_gen"] not in GPU_GENS: P.append(f"gpu_gen は {GPU_GENS}")
    if machine["num_threads"] < 0: P.append("num_threads ≥ 0")
    if not (1 <= machine["stage2_tb_port"] <= 65535): P.append("stage2_tb_port は 1..65535")


def check_dataset_values(dataset, machine):
    P = []
    if dataset["scale"] < 2: P.append("scale ≥ 2（1 はコピーになるので作らない）")
    if dataset["interp"] not in INTERPS: P.append(f"interp は {INTERPS}")
    if dataset["input_size"] < 1: P.append("input_size ≥ 1")
    _check_machine(machine, P)
    _problems_to_error("dataset values", P)


def check_infer_values(infer, machine):
    P = []
    if infer["input"] not in INFER_INPUTS: P.append(f"input は {INFER_INPUTS}")
    if infer["input"] == "eid" and infer["teacher"]: P.append("input=eid（実 EID）に教師は無いので teacher は false にする（--teacher false）")
    if infer["max_slices"] < 0: P.append("max_slices ≥ 0（0 で全部）")
    if infer["batch_slices"] < 1: P.append("batch_slices ≥ 1")
    if infer["io.read_workers"] < 0 or infer["io.write_workers"] < 0: P.append("io.read_workers / write_workers ≥ 0（0 = 直列）")
    _check_machine(machine, P)
    _problems_to_error("infer values", P)


def check_crop_values(crop, machine, image_size, n_pool, n_pcd, n_eid):
    P = []
    if crop["crop_size"] < 1 or crop["crop_size"] > image_size: P.append(f"crop_size は 1..{image_size}")
    if crop["panel_scale"] < 1: P.append("panel_scale >= 1")
    if crop["device"] not in ("cpu", "cuda"): P.append("device は cpu | cuda")
    if crop["device"] == "cuda" and machine["gpu_gen"] == 0: P.append("device=cuda だが machine は gpu_gen 0")
    if not (1 <= crop["pcd_index"] <= n_pcd): P.append(f"pcd_index は 1..{n_pcd}")
    if not (1 <= crop["eid_index"] <= n_eid): P.append(f"eid_index は 1..{n_eid}")
    if crop["diff_max_hu"] < 1: P.append("diff_max_hu >= 1")
    if crop["hist_hu_min"] >= crop["hist_hu_max"]: P.append("hist_hu_min < hist_hu_max")
    if crop["hist_bin_hu"] < 1: P.append("hist_bin_hu >= 1")
    elif (crop["hist_hu_max"] - crop["hist_hu_min"]) % crop["hist_bin_hu"]: P.append("hist_hu_max - hist_hu_min は hist_bin_hu の倍数")
    if crop["hist_y_max_pct"] <= 0: P.append("hist_y_max_pct > 0")
    _check_machine(machine, P)
    _problems_to_error("crop values", P)


def check_train_values(train, mode, machine):
    """train / mode / machine の実効値の値域と相互条件（学習）。症例リストの重複・ディスクとの照合は data/pair_dataset.py が行う。"""
    P = []
    bs = train["optim.batch_size"]
    if train["optim.lr"] <= 0: P.append("optim.lr は正")
    if not (0.0 <= train["optim.beta1"] < 1.0 and 0.0 <= train["optim.beta2"] < 1.0): P.append("optim.beta1 / beta2 は [0, 1)")
    if bs < 1: P.append("optim.batch_size ≥ 1")
    if train["optim.n_epochs"] < 1: P.append("optim.n_epochs ≥ 1")
    if train["optim.seed"] < 0: P.append("optim.seed ≥ 0")
    ps, m = train["data.patch_size"], 2 ** mode["n_pool"]
    if ps < m or ps % m: P.append(f"data.patch_size は {m}（= 2^n_pool）の倍数")
    spe = train["data.samples_per_epoch"]
    if spe < bs or spe % bs: P.append(f"data.samples_per_epoch ({spe}) は optim.batch_size ({bs}) の倍数で batch_size 以上")
    if train["data.patch_seed"] < 0: P.append("data.patch_seed ≥ 0")
    if train["data.patches_per_slice"] < 1: P.append("data.patches_per_slice ≥ 1")
    if train["data.hu_min"] >= train["data.hu_max"]: P.append("data.hu_min < data.hu_max")
    for k in ("data.train_cases", "data.val_cases"):
        if not train[k]: P.append(f"{k} は 1 症例以上")
    lists = {k: train[k] for k in ("data.train_cases", "data.val_cases", "data.test_cases")}
    for k, v in lists.items():
        if len(set(v)) != len(v): P.append(f"{k} に重複があります: {v}")
    names = list(lists)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            both = sorted(set(lists[names[i]]) & set(lists[names[j]]))
            if both: P.append(f"{names[i]} と {names[j]} に同じ症例があります: {both}")
    if train["log.print_freq"] < 1: P.append("log.print_freq ≥ 1（step）")
    if train["log.image_freq"] < 1: P.append("log.image_freq ≥ 1（step）")
    if train["log.n_images"] < 1: P.append("log.n_images ≥ 1")
    if train["log.save_epoch_freq"] < 1: P.append("log.save_epoch_freq ≥ 1")
    slf = train["log.save_latest_freq"]
    if slf < bs or slf % bs: P.append(f"log.save_latest_freq ({slf}) は optim.batch_size ({bs}) の倍数（画像枚数単位。倍数でないと latest が保存されない）")
    if train["log.val_max_slices_per_case"] < 0: P.append("log.val_max_slices_per_case ≥ 0（0 で全部）")
    if train["log.best_metric"] not in BEST_METRICS: P.append(f"log.best_metric は {BEST_METRICS}")
    if not train["log.full_slice"]: P.append("log.full_slice（固定スライスの相対パス）を指定")
    if not train["log.eid_slice"]: P.append("log.eid_slice（実 EID テストスライスの相対パス）を指定")
    if train["log.preview_bits"] not in (8, 16): P.append("log.preview_bits は 8 | 16")
    if train["log.tb_full_size"] < 64: P.append("log.tb_full_size ≥ 64")
    if train["log.n_full_random"] < 0: P.append("log.n_full_random ≥ 0")
    if train["log.display_hu_min"] >= train["log.display_hu_max"]: P.append("log.display_hu_min < display_hu_max")
    if mode["arch"] not in ARCHS: P.append(f"arch は {ARCHS}")
    if mode["base_ch"] < 1: P.append("base_ch ≥ 1")
    if mode["n_pool"] < 1: P.append("n_pool ≥ 1")
    if mode["init_type"] not in INIT_TYPES: P.append(f"init_type は {INIT_TYPES}")
    if mode["loss"] not in LOSSES: P.append(f"loss は {LOSSES}")
    if mode["final_act"] not in FINAL_ACTS: P.append(f"final_act は {FINAL_ACTS}")
    if mode["real_eid_branch"]:  # 実 EID 枝（2026-10-13）。false のときは adv.* を使わないので値域も見ない
        if train["adv.lambda_gan"] <= 0: P.append("adv.lambda_gan は正（実 EID 枝を使わないなら mode.yaml real_eid_branch: false）")
        if train["adv.gan_mode"] not in GAN_MODES: P.append(f"adv.gan_mode は {GAN_MODES}")
        if train["adv.lr_D"] <= 0: P.append("adv.lr_D は正")
        if train["adv.warmup_epochs"] < 0: P.append("adv.warmup_epochs ≥ 0")
        if train["adv.lr_after_warmup"] < 0: P.append("adv.lr_after_warmup ≥ 0（0 = 変えない）")
        if train["adv.clip_grad_norm"] < 0 or train["adv.clip_grad_norm_D"] < 0: P.append("adv.clip_grad_norm / clip_grad_norm_D ≥ 0（0 = 無効）")
        if not (0.0 <= train["adv.beta1_D"] < 1.0 and 0.0 <= train["adv.beta2_D"] < 1.0): P.append("adv.beta1_D / beta2_D は [0, 1)")
        if train["adv.batch_eid"] < 1: P.append("adv.batch_eid ≥ 1")
        if train["adv.crop_eid"] < 1: P.append("adv.crop_eid ≥ 1（crop_eid × scale が 2^n_pool の倍数かは manifest の scale を読んでから data/eid_dataset.py が検査）")
        if train["adv.patches_per_slice_eid"] < 1: P.append("adv.patches_per_slice_eid ≥ 1")
        if not train["adv.eid_split_file"]: P.append("adv.eid_split_file（実 EID の症例分割 YAML）を指定。bash start.sh split で作る")
    _check_machine(machine, P)
    _problems_to_error("train values", P)


# ---------------------------------------------------------------------------
# 実 EID の症例分割ファイル（adv.eid_split_file。Stage 1 の sr の split_file と同じ形・同じ規則。check_split は stage1/configs/schema.py から 2026-10-13 に複製）
# ---------------------------------------------------------------------------
SPLIT_KEYS = ("train", "val", "test")


def resolve_eid_split_file(train, root):
    """adv.eid_split_file を root（リポジトリ直下）基準で解決して存在を検査し、実効値を絶対パスに置き換える（launch.yaml にはそのパスが残る。Stage 1 の check_sr_values と同じ）。"""
    sf = Path(train["adv.eid_split_file"])
    sf = sf if sf.is_absolute() else Path(root) / sf
    if not sf.is_file():
        raise ConfigError(f"adv.eid_split_file がありません: {sf}（bash start.sh split で作る。実 EID 枝を使わないなら mode.yaml real_eid_branch: false）")
    train["adv.eid_split_file"] = str(sf)
    return sf


def check_split(split, where):
    """split_file の中身: train / val / test の文字列リスト（空リスト可）。重複・他のキー・型違いはエラー。戻り値 {kind: [case, ...]}。"""
    if not isinstance(split, dict):
        raise ConfigError(f"[{where}] トップレベルが dict ではありません")
    P = []
    missing = [k for k in SPLIT_KEYS if k not in split]
    if missing:
        P.append("キーがありません: " + ", ".join(missing))
    out = {}
    for k in SPLIT_KEYS:
        v = split.get(k)
        if k in split and not (isinstance(v, list) and all(isinstance(x, str) and x for x in v)):
            P.append(f"{k} は症例名（文字列）のリスト")
        else:
            out[k] = list(v or [])
    if not P:
        seen = {}
        for k in SPLIT_KEYS:
            for c in out[k]:
                if c in seen:
                    P.append(f"症例 {c} が {seen[c]} と {k} の両方にあります")
                seen[c] = k
        if len(set(out["train"])) != len(out["train"]) or len(set(out["val"])) != len(out["val"]) or len(set(out["test"])) != len(out["test"]):
            P.append("同じリストの中に重複があります")
        if not out["train"]:
            P.append("train が空です")
    _problems_to_error(where, P)
    return out


def load_eid_split(path):
    """adv.eid_split_file（絶対パス）を読んで check_split で検査する。"""
    import yaml
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"adv.eid_split_file がありません: {path}（bash start.sh split で作る）")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return check_split(data, f"adv.eid_split_file {path}")
