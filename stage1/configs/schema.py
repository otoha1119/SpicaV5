"""stage1/configs/schema.py — 新 Stage 1（描き手 P）の設定ファイル（train.yaml / model.yaml / generate.yaml）、共有設定（configs/shared.yaml）、
Stage 1 が読む configs/machines.yaml のキーの唯一の正。

原則（旧コードと同じ）: 既定値（フォールバック）は無い。ここに宣言したキーは**必ず**設定ファイルに存在しなければならず、未知のキー・型違いはエラー。
sh からの上書き（--flag value）は schema にあるフラグだけ受け付け、実効値を launch.yaml / manifest.yaml に残す。
検証・上書きの関数（validate / apply_overrides / flatten）は old/stage2/configs/schema.py から複製した（Stage 間で import しない）。足したもの: 型 ilist（int のリスト）。

設計の正: docs/plans/20261111_detail_design/stage1_painter_spec.md（以下「設計書」）と RECONCILE.md（食い違いの裁定）。

各キーの宣言: Key(type, flag, doc)
  type : int | float | str | bool | list（str のリスト）| ilist（int のリスト）。float は int の値も受け付ける。bool は YAML の true / false のみ
  flag : 上書き用のフラグ名（--flag value。list / ilist は --flag a,b,c。空リストは --flag ""）。None は上書き不可
"""

from collections import namedtuple

Key = namedtuple("Key", "type flag doc")
ilist = type("ilist", (), {})  # int のリストを表す型の印

# ---------------------------------------------------------------------------
# stage1/configs/train.yaml — 学習（run_train.py → train.py）
# ---------------------------------------------------------------------------
TRAIN = {
    # optim（設計書 G-1）
    "optim.lr":                  Key(float, "--lr",                  "AdamW の学習率。2e-4。発散したら 1e-4"),
    "optim.beta1":               Key(float, "--beta1",               "AdamW β1（0.9）"),
    "optim.beta2":               Key(float, "--beta2",               "AdamW β2（0.999）"),
    "optim.weight_decay":        Key(float, "--weight_decay",        "AdamW の weight decay（0）"),
    "optim.warmup_steps":        Key(int,   "--warmup_steps",        "学習率を 0 から直線で上げる step 数（2,000）。その後は一定。0 で無し"),
    "optim.batch_size":          Key(int,   "--batch_size",          "batch（パッチ数）。32"),
    "optim.grad_clip":           Key(float, "--grad_clip",           "勾配ノルムのクリップ（1.0）"),
    "optim.ema_decay":           Key(float, "--ema_decay",           "EMA の減衰の上限（0.9999）。序盤は min(ema_decay, (1 + n) / (10 + n))。描画・val は常に EMA"),
    "optim.precision":           Key(str,   "--precision",           "学習の U-Net 本体の演算精度 fp32 | bf16（CUDA の autocast）。入口（Haar・stem）・出口（head・逆 Haar・x̂0 の合成）・損失・埋め込みは常に FP32（設計書 C-4）"),
    "optim.compile":             Key(bool,  "--compile",             "CUDA で学習の U-Net 本体を torch.compile する（CPU では無効）。val・途中経過の描画は compile しない EMA"),
    "optim.compile_mode":        Key(str,   "--compile_mode",        "torch.compile の mode: max-autotune（設計書）| reduce-overhead | default。PC で max-autotune が落ちたら default に下げる"),
    "optim.compile_after_steps": Key(int,   "--compile_after_steps", "この起動の最初の N step は compile せずに回す（最初の scalar をすぐ出すため。compile は N + 1 step 目に始まり、初回は数分かかる）。0 で最初から"),
    "optim.channels_last":       Key(bool,  "--channels_last",       "U-Net を channels_last にする"),
    "optim.fp32_top_level":      Key(bool,  "--fp32_top_level",      "true で最上段（幅 base_ch の段）の ResBlock を FP32 で計算する（bf16 で弱い σ の帯の val が悪いとき。設計書 C-4）"),
    "optim.max_steps":           Key(int,   "--max_steps",           "学習の上限 step（250,000）。max_hours と早いほうで止まり latest を保存して正常終了する"),
    "optim.max_hours":           Key(float, "--max_hours",           "学習の上限時間（36。再開をまたいだ累計）"),
    "optim.seed":                Key(int,   "--seed",                "python / numpy / torch の乱数 seed"),
    # data（設計書 B）
    "data.patch_size":        Key(int,   "--patch_size",        "学習パッチの一辺（画素）。256。16 の倍数（Haar 1 段 + stride 2 が 3 回）"),
    "data.margin":            Key(int,   "--margin",            "パッチの周りに付けて切る余白（画素）。32。スライスを reflect で広げて patch_size + 2·margin を切り、ぼかしてから中央を使う。3·cue.blur_max_px 以上"),
    "data.patches_per_slice": Key(int,   "--patches_per_slice", "1 回読んだスライスから連続して切るパッチ数（4）。位置は毎回独立"),
    "data.hu_offset":         Key(int,   None,                  "stored = HU + hu_offset（1400）"),
    "data.hu_min":            Key(int,   None,                  "HU の切り下限（−1400）"),
    "data.hu_max":            Key(int,   None,                  "HU の切り上限（4096）"),
    "data.hu_scale":          Key(float, None,                  "x = HU ÷ hu_scale（1000）。線形 1 本"),
    "data.split_file":        Key(str,   "--split_file",        "実 EID の症例分割（リポジトリ直下からの相対パス。configs/splits/eid_v5.yaml）。train だけで学習、val は損失の監視、test は読まない"),
    "data.unlisted_cases":    Key(str,   "--unlisted_cases",    "分割ファイルに無い症例（分割を作った後に増えた症例。評価専用は別に除外済み）の扱い: train（学習に入れる。数を表示して記録）| error"),
    "data.domain":            Key(str,   "--domain",            "学習する絵: eid（実 EID。machines の eid_dir と分割ファイル）| pcd（PCD512。machines の pcd_dir。型 X の「PCD の描き手」。分割ファイルは使わず data.pcd_val_cases を val にする）"),
    "data.pcd_val_cases":     Key(list,  "--pcd_val_cases",     "data.domain: pcd のとき val に回す症例（学習に使わない。途中経過の PCD スライスの症例はここに入れる）。eid のときは使わない"),
    "data.stats_patches":     Key(int,   "--stats_patches",     "学習の最初にこの数のパッチで x の平均・標準偏差を測って表示・記録する（確認用。設定には使わない）。0 で測らない"),
    # init（型 X: PCD の描き手を、実 EID の描き手の重みから始める）
    "init.run":            Key(str,   "--init_run",        "この run の重み（EMA）を net と EMA の両方に読んでから学習を始める。空文字 = 最初から。再開のときは使わない（重みは再開元のもの）"),
    "init.tag":            Key(str,   "--init_tag",        "init.run の重みの tag: latest | best | step 番号"),
    # cue（設計書 E-2）
    "cue.drop_prob":       Key(float, "--cue_drop_prob",   "手がかりを隠す確率（0.10）"),
    "cue.blur_zero_prob":  Key(float, "--blur_zero_prob",  "σ_b = 0（ぼかさない）の確率（0.15）"),
    "cue.blur_min_px":     Key(float, "--blur_min_px",     "σ_b の下端（0 でないとき。0.5 画素）。[min, max] を対数一様"),
    "cue.blur_max_px":     Key(float, "--blur_max_px",     "σ_b の上端（8 画素）。埋め込みの正規化にも使うので再開で変えられない"),
    "cue.noise_zero_prob": Key(float, "--noise_zero_prob", "σ_c = 0（ノイズ無し）の確率（0.15）"),
    "cue.noise_min_hu":    Key(float, "--noise_min_hu",    "σ_c の下端（0 でないとき。5 HU）"),
    "cue.noise_max_hu":    Key(float, "--noise_max_hu",    "σ_c の上端（400 HU）"),
    # diffusion（設計書 D-1。EDM: Karras et al. 2022。出典は stage1/UPSTREAM.md）
    "diffusion.sigma_min":      Key(float, None, "σ の下端（x の単位。0.002 = 2 HU）"),
    "diffusion.sigma_max":      Key(float, None, "σ の上端（40 = 40,000 HU）"),
    "diffusion.sigma_data":     Key(float, None, "EDM の σ_data（0.5）"),
    "diffusion.sigma_sampling": Key(str,   None, "学習時の σ の引き方: loguniform（ln σ を [ln σ_min, ln σ_max] で一様）"),
    # sample（学習中の途中経過の描画。設計書 F-1。本番の描画は generate.yaml）
    "sample.steps":     Key(int,   "--sample_steps",     "途中経過の描画の段数（40）"),
    "sample.eta":       Key(float, "--sample_eta",       "Euler ancestral の η（1.0。0 で決定的）"),
    "sample.rho":       Key(float, "--sample_rho",       "Karras の配分 ρ（7）"),
    "sample.precision": Key(str,   "--sample_precision", "途中経過の描画の U-Net 本体の精度 fp32 | bf16（設計書 F-1: ネットの中は bf16、ほかは FP32）"),
    # log（設計書 G-2・G-3）
    "log.print_steps":         Key(int,   "--print_steps",         "scalar・loss_log.txt の間隔（step）。100。最初は 1, 2, 4, …, 64 step 目にも出す"),
    "log.steps_per_epoch":     Key(int,   "--steps_per_epoch",     "表示の区切り（10,000 step = 1 epoch 相当）"),
    "log.train_patch_steps":   Key(int,   "--train_patch_steps",   "TensorBoard preview/train_patch の間隔（step）。500"),
    "log.n_train_patches":     Key(int,   "--n_train_patches",     "その行数（4）"),
    "log.val_steps":           Key(int,   "--val_steps",           "val の損失の間隔（step）。2,000"),
    "log.val_patches":         Key(int,   "--val_patches",         "val の固定パッチ数（96。位置・σ・乱数・度合いを固定）"),
    "log.val_seed":            Key(int,   "--val_seed",            "val の固定パッチと乱数の seed"),
    "log.preview_steps":       Key(int,   "--preview_steps",       "途中経過のパネル（描き直し・疑似 EID・手がかり無し）と diag/check_* の間隔（step）。2,000"),
    "log.preview_first_step":  Key(int,   "--preview_first_step",  "この起動の最初にこの step 数で 1 回だけ val・パネル・latest の保存を通す（配線の確認。0 で無し）"),
    "log.save_latest_steps":   Key(int,   "--save_latest_steps",   "latest/ の保存間隔（step）。5,000"),
    "log.save_weights_steps":  Key(int,   "--save_weights_steps",  "weights/step_NNNNNNN/ の保存間隔（step）。25,000"),
    "log.eid_slices":          Key(list,  "--eid_slices",          "途中経過（描き直し）に使う実 EID のスライス。**コンテナ内の絶対パス**の一覧（病院 EID は eid_dir の外にあるため）。1 つ目が基準（diag/check_redraw_*、疑似 EID の横に並べる参照、拡大）。学習に使う症例（train 側）は置けない。無いファイルは起動前にエラー"),
    "log.pcd_slices":          Key(list,  "--pcd_slices",          "途中経過（疑似 EID）に使う PCD512 のスライス。コンテナ内の絶対パスの一覧。1 つ目が基準（diag/check_pseudo_*、拡大）。学習には使わない"),
    "log.preview_cues":        Key(list,  "--preview_cues",        "途中経過の度合い。'σ_b（画素）:σ_c（HU）' のリスト"),
    "log.preview_seed":        Key(int,   "--preview_seed",        "途中経過の乱数の種（毎回同じ描き始め）"),
    "log.nocue_n":             Key(int,   "--nocue_n",             "手がかり無しで描く patch_size 四方の枚数（4）。0 で無し"),
    "log.zoom_size":           Key(int,   "--zoom_size",           "拡大する範囲の一辺（72 画素）"),
    "log.zoom_scale":          Key(int,   "--zoom_scale",          "拡大率（4、最近傍）"),
    "log.eid_zoom_x":          Key(int,   "--eid_zoom_x",          "eid_slices の 1 つ目の拡大の左上 x（327。旧 crop.yaml の case 1）。2 つ目以降は拡大を出さない"),
    "log.eid_zoom_y":          Key(int,   "--eid_zoom_y",          "同 y（188）"),
    "log.pcd_zoom_x":          Key(int,   "--pcd_zoom_x",          "pcd_slices の 1 つ目の拡大の左上 x（266）"),
    "log.pcd_zoom_y":          Key(int,   "--pcd_zoom_y",          "同 y（222）"),
    "log.diff_range_hu":       Key(float, "--diff_range_hu",       "差の列の色の幅（±HU。白 = 0、青 −、赤 +）"),
    "log.display_hu_min":      Key(int,   "--display_hu_min",      "表示の下限 HU（−1400）"),
    "log.display_hu_max":      Key(int,   "--display_hu_max",      "表示の上限 HU（3000）"),
    "log.check_highpass_px":   Key(float, "--check_highpass_px",   "diag/check_* の「細かい成分」= 絵 − Gaussian(絵, この画素)。監視用の固定値（2）で、設計の的ではない"),
}

# ---------------------------------------------------------------------------
# stage1/configs/model.yaml — ネットワーク（設計書 C-1・C-2）。再開で変えられない
# ---------------------------------------------------------------------------
MODEL = {
    "base_ch":          Key(int,   "--base_ch",          "最上段の幅（96）"),
    "ch_mult":          Key(ilist, "--ch_mult",          "段ごとの幅の倍率（[1, 2, 3, 4] → 96 / 192 / 288 / 384）"),
    "blocks_per_level": Key(int,   "--blocks_per_level", "下りの各段の ResBlock 数（2）。上りは +1（skip を連結）"),
    "mid_blocks":       Key(int,   "--mid_blocks",       "底の ResBlock 数（2）"),
    "groups":           Key(int,   "--groups",           "GroupNorm の群の数（32）。各段の幅を割り切ること"),
    "emb_dim":          Key(int,   "--emb_dim",          "度合いの埋め込みの次元（384）"),
    "emb_freqs":        Key(int,   "--emb_freqs",        "スカラーを sin / cos にする周波数の数（32 → 64 次元）"),
    "emb_freq_min":     Key(float, "--emb_freq_min",     "その最小の周波数（0.25）。対数で等間隔"),
    "emb_freq_max":     Key(float, "--emb_freq_max",     "同最大（64）"),
    "fold":             Key(str,   "--fold",             "入口と出口の畳み方: haar1（Haar 1 段。512×512×1 ↔ 256×256×4）"),
}

# ---------------------------------------------------------------------------
# configs/shared.yaml — 両 Stage で共有する値（RECONCILE #1）。Stage 1 が使うキーだけ検査する
# ---------------------------------------------------------------------------
SHARED = {
    "brightness.sigma_px_512": Key(float, "--brightness_sigma_px_512", "明るさを平均する広さ σ_a（Gaussian の σ、512 の画素。16）。錨に使う。1024 では 2 倍"),
}

# ---------------------------------------------------------------------------
# configs/machines.yaml のうち Stage 1 が使うキー（他のキーは無視）
# ---------------------------------------------------------------------------
MACHINE = {
    "gpu_gen":                 Key(int, None,            "GPU 世代 30 | 40 | 50 | 0(CPU)。0 以外は CUDA 必須"),
    "container_data_root":     Key(str, None,            "データのマウント先（コンテナ内）"),
    "num_threads":             Key(int, "--num_threads", "DataLoader の worker 数（0 でメインプロセス）"),
    "tb_port":                 Key(int, None,            "TensorBoard のポート（旧 Stage 1 と共用。同時には回さない）"),
    "eid_dir":                 Key(str, "--eid_dir",     "実 EID512（ImageCAS、EID_v5）。P の学習データ"),
    "pcd_dir":                 Key(str, "--pcd_dir",     "PCD512。学習には使わない（途中経過の 1 枚と疑似 EID の元）"),
    "painter_checkpoints_dir": Key(str, None,            "新 Stage 1 の run の置き場（/workspace/runs/stage1）。監視ツールは語尾 checkpoints_dir で拾う"),
    "pseudo_eid_dir":          Key(str, "--pseudo_eid_dir", "疑似 EID の置き場（/workspace/DataSet/PseudoEID512）。この下に <組>/k00/<症例>/<スライス>.png"),
}

# ---------------------------------------------------------------------------
# stage1/configs/generate.yaml — 疑似 EID の作成・実 EID の描き直し（run_generate.py）。設計書 F
# ---------------------------------------------------------------------------
GENERATE = {
    "generate.run":          Key(str,   "--run",          "使う run 名（painter_checkpoints_dir の下）。yaml は空にして --run で渡す"),
    "generate.tag":          Key(str,   "--tag",          "重み: latest | best | step 番号"),
    "generate.input_dir":    Key(str,   "--input_dir",    "入力の PNG フォルダ（<症例>/<スライス>.png、512）。空文字なら machines.yaml の pcd_dir"),
    "generate.cases":        Key(list,  "--cases",        "この症例だけ作る（[] で全部。評価専用も除外しない）"),
    "generate.k":            Key(int,   "--k",            "1 枚につき何通り作るか（k00, k01, …）"),
    "generate.seed":         Key(int,   "--gen_seed",     "base_seed。画像ごとの種 = sha256('<seed>/<症例>/<スライス名>/<k>')"),
    "generate.slice_step":   Key(int,   "--slice_step",   "症例ごとに何枚おきに作るか（1 = 全部、2 = 1 枚おき）"),
    "generate.max_slices":   Key(int,   "--max_slices",   "症例ごとの上限枚数（0 で全部。試しに数枚だけ作るとき）"),
    "generate.name_suffix":  Key(str,   "--name_suffix",  "組の名前の末尾に付ける文字列（同じ度合いで別の組を作るとき。空文字で無し）"),
    "cue.blur_px":           Key(float, "--blur_px",      "手がかりの粗さ σ_b（画素。0 = ぼかさない）"),
    "cue.noise_hu":          Key(float, "--noise_hu",     "手がかりのノイズ σ_c（HU。0 = 足さない）"),
    "cue.use":               Key(bool,  "--use_cue",      "false で手がかり無し（診断用。錨も掛けない。512 では配置がでたらめになる）"),
    "sample.steps":          Key(int,   "--steps",        "段数（40）"),
    "sample.eta":            Key(float, "--eta",          "Euler ancestral の η（1.0）"),
    "sample.rho":            Key(float, "--rho",          "Karras の配分 ρ（7）"),
    "sample.batch":          Key(int,   "--batch",        "まとめて通す枚数（8）"),
    "sample.precision":      Key(str,   "--precision",    "U-Net 本体の精度 fp32 | bf16"),
    "sample.compile":        Key(bool,  "--compile",      "CUDA で U-Net 本体を torch.compile する（batch を固定し、端数は詰め物をして捨てる）"),
    "sample.compile_mode":   Key(str,   "--compile_mode", "max-autotune | reduce-overhead | default"),
}

# ---------------------------------------------------------------------------
# stage1/configs/bridge.yaml — 型 X の疑似 EID の作成（run_bridge.py）。2 つの描き手を同じノイズでつなぐ
# ---------------------------------------------------------------------------
BRIDGE = {
    "bridge.enc_run":      Key(str,   "--enc_run",      "行き（絵 → ノイズ）に使う run。PCD → 疑似 EID なら PCD の描き手。yaml は空にして --enc_run で渡す"),
    "bridge.enc_tag":      Key(str,   "--enc_tag",      "その重み: latest | best | step 番号"),
    "bridge.dec_run":      Key(str,   "--dec_run",      "帰り（ノイズ → 絵）に使う run。PCD → 疑似 EID なら実 EID の描き手。enc と同じにすると往復の確認（元の絵に戻るはず）"),
    "bridge.dec_tag":      Key(str,   "--dec_tag",      "その重み: latest | best | step 番号"),
    "bridge.input_dir":    Key(str,   "--input_dir",    "入力の PNG フォルダ（<症例>/<スライス>.png、512）。空文字なら machines.yaml の pcd_dir"),
    "bridge.cases":        Key(list,  "--cases",        "この症例だけ作る（[] で全部）"),
    "bridge.slices":       Key(list,  "--slices",       "このファイル名だけ作る（[] で絞らない。例 PCD-002-236.png）"),
    "bridge.slice_step":   Key(int,   "--slice_step",   "症例ごとに何枚おきに作るか（1 = 全部）"),
    "bridge.max_slices":   Key(int,   "--max_slices",   "症例ごとの上限枚数（0 で全部）"),
    "bridge.name_suffix":  Key(str,   "--name_suffix",  "組の名前の末尾に付ける文字列（空文字で無し）"),
    "cue.use":             Key(bool,  "--use_cue",      "粗くした絵を条件に渡す（行きと帰りで同じもの）。false = 条件なし"),
    "cue.blur_px":         Key(float, "--blur_px",      "条件の絵の粗さ σ_b（画素）。型 X では細かい形はノイズが運ぶので、粗い側に倒す"),
    "cue.noise_hu":        Key(float, "--noise_hu",     "条件の絵に足すノイズ σ_c（HU）。0 = 足さない（足すと乱数が入る。画像ごとの種で固定）"),
    "anchor.decode":       Key(bool,  "--anchor_decode", "帰りの各段で、広い平均を入力に合わせ直す（明るさの錨。σ_a は shared の brightness.sigma_px_512）"),
    "ode.steps":           Key(int,   "--steps",        "行き・帰りそれぞれの段数"),
    "ode.heun":            Key(bool,  "--heun",         "true = Heun（2 次。1 段に 2 回通す）、false = Euler"),
    "ode.rho":             Key(float, "--rho",          "Karras の配分 ρ（7）"),
    "ode.decode_eta":      Key(float, "--decode_eta",   "帰りの各段で新しい乱数を混ぜる度合い（Euler ancestral の η）。0 = 乱数なし（行きのノイズ z がそのまま絵を決める）"),
    "ode.batch":           Key(int,   "--batch",        "まとめて通す枚数"),
    "ode.precision":       Key(str,   "--precision",    "U-Net 本体の精度 fp32 | bf16（往復の誤差を見て選ぶ）"),
    "bridge.seed":         Key(int,   "--gen_seed",     "cue.noise_hu > 0 のときの乱数の種"),
}

# ---------------------------------------------------------------------------
# stage1/configs/y.yaml — 型 Y の学習（train_y.py）。消し手 A（PCD を入力）と疑似側の物差し F。実 EID の描き手 E は固定
# ---------------------------------------------------------------------------
Y_TRAIN = {
    "init.e_run":          Key(str,   "--e_run",          "実 EID の描き手 E の run（固定。F もこの EMA の重みから始める）"),
    "init.e_tag":          Key(str,   "--e_tag",          "その重み: latest | best | step 番号"),
    "optim.lr_a":          Key(float, "--lr_a",           "消し手 A の AdamW の学習率"),
    "optim.lr_f":          Key(float, "--lr_f",           "物差し F の AdamW の学習率"),
    "optim.beta1":         Key(float, "--beta1",          "AdamW β1"),
    "optim.beta2":         Key(float, "--beta2",          "AdamW β2"),
    "optim.batch_size":    Key(int,   "--batch_size",     "batch（PCD のパッチ数）"),
    "optim.grad_clip":     Key(float, "--grad_clip",      "勾配ノルムのクリップ（A・F それぞれ）"),
    "optim.f_steps":       Key(int,   "--f_steps",        "A を 1 回更新するあいだに F を何回更新するか（DMD は 1、DMD2 は 5）"),
    "optim.precision":     Key(str,   "--precision",      "E・F の U-Net 本体の精度 fp32 | bf16（A・損失・ぼかしは常に FP32）"),
    "optim.compile":       Key(bool,  "--compile",        "F の学習の U-Net 本体を torch.compile（E の推論は compile しない）"),
    "optim.compile_mode":  Key(str,   "--compile_mode",   "max-autotune | reduce-overhead | default"),
    "optim.compile_after_steps": Key(int, "--compile_after_steps", "最初の N step は compile せずに回す"),
    "optim.max_steps":     Key(int,   "--max_steps",      "上限 step（A の更新回数）"),
    "optim.max_hours":     Key(float, "--max_hours",      "上限時間（再開をまたいだ累計）"),
    "optim.seed":          Key(int,   "--seed",           "乱数 seed"),
    "data.patch_size":     Key(int,   "--patch_size",     "PCD のパッチの一辺（E の学習と同じ 256）"),
    "data.margin":         Key(int,   "--margin",         "パッチの余白（reflect で広げて切る。A の近傍と手がかりのぼかしが端で欠けないように）"),
    "data.patches_per_slice": Key(int, "--patches_per_slice", "1 スライスから切るパッチ数"),
    "data.pcd_val_cases":  Key(list,  "--pcd_val_cases",  "学習に使わない PCD の症例（途中経過のスライスはここから）"),
    "eraser.kind":         Key(str,   "--eraser_kind",    "free = 近傍の重みを自由に選ぶ（足して 1）| isotropic = 画素ごとに広さだけ選べる向きのないぼかし（左右上下対称）"),
    "eraser.radius":       Key(int,   "--radius",         "A が平均できる近傍の半径（上限。ぼかし量ではない）"),
    "eraser.width":        Key(int,   "--eraser_width",   "A の CNN の幅"),
    "eraser.depth":        Key(int,   "--eraser_depth",   "A の CNN の ResBlock の数（dilation 1, 2, 4, 8 の繰り返し）"),
    "eraser.center_bias":  Key(float, "--center_bias",    "初期値で中央の画素に与える logit（大きいほど最初は恒等に近い）"),
    "dmd.sigma_lo_hu":     Key(float, "--sigma_lo_hu",    "見比べのノイズの下端（HU）。この強さのノイズをかぶせた上でだけ E と F を比べる"),
    "dmd.sigma_hi_hu":     Key(float, "--sigma_hi_hu",    "同上端（HU）。[lo, hi] を対数一様"),
    "dmd.cue_blur_px":     Key(float, "--cue_blur_px",    "E・F に渡す手がかり = A の出力をこの広さでぼかした絵（明るさの配置をそろえて比べる。E の学習の上端以下）"),
    "log.print_steps":     Key(int,   "--print_steps",    "scalar の間隔"),
    "log.preview_steps":   Key(int,   "--preview_steps",  "途中経過（val のスライス全体を A に通す + 描き直し、確認の数字）の間隔"),
    "log.save_latest_steps": Key(int, "--save_latest_steps", "latest/ の保存間隔"),
    "log.save_weights_steps": Key(int, "--save_weights_steps", "weights/step_NNNNNNN/ の保存間隔"),
    "log.pcd_slices":      Key(list,  "--pcd_slices",     "途中経過に使う PCD のスライス（コンテナ内の絶対パス。val の症例）"),
    "log.preview_s_hu":    Key(list,  "--preview_s_hu",   "途中経過の描き直しの出発点 s（HU）の並び（文字列のリスト）"),
    "log.preview_steps_redraw": Key(int, "--preview_redraw_steps", "途中経過の描き直しの段数"),
}

# ---------------------------------------------------------------------------
# stage1/configs/y_generate.yaml — 型 Y の疑似 EID の作成（generate_y.py）
# ---------------------------------------------------------------------------
Y_GENERATE = {
    "generate.run":        Key(str,   "--run",            "型 Y の run（A を読む。E は run の launch の init.e_run）"),
    "generate.tag":        Key(str,   "--tag",            "A の重み: latest | step 番号"),
    "generate.input_dir":  Key(str,   "--input_dir",      "入力（<症例>/<スライス>.png）。空文字なら machines の pcd_dir"),
    "generate.cases":      Key(list,  "--cases",          "[] で全部"),
    "generate.slices":     Key(list,  "--slices",         "[] で絞らない"),
    "generate.slice_step": Key(int,   "--slice_step",     "何枚おき"),
    "generate.max_slices": Key(int,   "--max_slices",     "症例ごとの上限（0 で全部）"),
    "generate.k":          Key(int,   "--k",              "1 枚につき何通り（描き直しの乱数を変える）"),
    "generate.seed":       Key(int,   "--gen_seed",       "画像ごとの種の base"),
    "generate.name_suffix": Key(str,  "--name_suffix",    "組の名前の末尾"),
    "redraw.s_hu":         Key(float, "--s_hu",           "描き直しの出発点 s（HU）。0 = 描き直さず A の出力をそのまま書く"),
    "redraw.steps":        Key(int,   "--steps",          "描き直しの段数（s から 0 まで、Euler ancestral η = 1）"),
    "redraw.cue_blur_px":  Key(float, "--cue_blur_px",    "描き直しで E に渡す手がかり = A の出力をこの広さでぼかした絵"),
    "redraw.anchor":       Key(bool,  "--anchor",         "各段で広い平均を入力の PCD に合わせ直す（σ_a は shared）"),
    "redraw.batch":        Key(int,   "--batch",          "まとめて通す枚数"),
    "redraw.precision":    Key(str,   "--precision",      "fp32 | bf16"),
}

PRECISIONS = ("fp32", "bf16")
COMPILE_MODES = ("max-autotune", "reduce-overhead", "default")
GPU_GENS = (0, 30, 40, 50)
FOLDS = ("haar1",)
SIGMA_SAMPLINGS = ("loguniform",)
UNLISTED = ("train", "error")
DOMAINS = ("eid", "pcd")
# 再開で変えられない train のキー（重みの意味が変わる）。MODEL は全部変えられない
FROZEN_ON_RESUME = ("data.hu_offset", "data.hu_min", "data.hu_max", "data.hu_scale", "cue.blur_max_px",
                    "diffusion.sigma_min", "diffusion.sigma_max", "diffusion.sigma_data", "diffusion.sigma_sampling")


# ---------------------------------------------------------------------------
# 検証・変換（old/stage2/configs/schema.py と同じ + ilist 型）
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


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _type_ok(value, typ):
    if typ is bool:
        return isinstance(value, bool)
    if typ is int:
        return _is_int(value)
    if typ is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if typ is str:
        return isinstance(value, str)
    if typ is list:
        return isinstance(value, list) and all(isinstance(x, str) for x in value)
    if typ is ilist:
        return isinstance(value, list) and all(_is_int(x) for x in value)
    raise TypeError(typ)


def _coerce(value, typ):
    if typ is float:
        return float(value)
    if typ in (list, ilist):
        return list(value)
    return value


def _check(name, data, schema, allow_unknown):
    if not isinstance(data, dict):
        raise ConfigError(f"[{name}] トップレベルが dict ではありません: {type(data).__name__}")
    flat = flatten(data)
    missing = [k for k in schema if k not in flat]
    unknown = [] if allow_unknown else [k for k in flat if k not in schema]
    bad = [f"{k}: 期待 {schema[k].type.__name__}, 実際 {type(flat[k]).__name__} ({flat[k]!r})" for k in schema if k in flat and not _type_ok(flat[k], schema[k].type)]
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


def validate(name, data, schema):
    """平坦化した設定を schema と突き合わせる。欠落・未知・型違いをまとめて ConfigError にする。戻り値は平坦 dict。"""
    return _check(name, data, schema, allow_unknown=False)


def validate_shared(name, data, schema):
    """共有ファイル（machines.yaml / shared.yaml）用: schema のキーの存在と型だけ検査し、他のキーは無視する。"""
    return _check(name, data, schema, allow_unknown=True)


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
        if typ in (list, ilist):
            if str(val).strip() == "":
                return []
            items = [x.strip() for x in str(val).split(",")]
            if any(not x for x in items):
                raise ValueError
            return [int(x) for x in items] if typ is ilist else items
    except ValueError:
        raise ConfigError(f"{flag} の値を {typ.__name__} に変換できません: {val!r}")
    raise TypeError(typ)


def apply_overrides(tokens, sections):
    """sh からの上書き列を平坦 dict に反映する（後勝ち）。
    sections: {"train": (flat_dict, TRAIN), ...}。flat_dict はその場で書き換わる。
    受ける形: --flag value / --flag=value。bool は --flag（= true）または --flag true|false。list は --flag a,b,c。
    schema に無いフラグ・値の無いフラグ・余った語・型違いは ConfigError。戻り値は適用した {flag: 値}（記録用）。"""
    index = {}
    for sec, (_flat, schema) in sections.items():
        for key_name, key in schema.items():
            if key.flag:
                if key.flag in index:
                    raise RuntimeError(f"schema のフラグが重複しています: {key.flag}")
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
# 値域・相互条件の検証。型検査の後、実効値（上書き反映後）に対して行う
# ---------------------------------------------------------------------------
def _raise(where, problems):
    if problems:
        raise ConfigError(f"[{where}] " + " / ".join(problems))


def parse_cue(text):
    """'σ_b:σ_c' → (σ_b 画素, σ_c HU)。"""
    try:
        b, c = text.split(":")
        b, c = float(b), float(c)
    except ValueError:
        raise ConfigError(f"度合いは 'σ_b（画素）:σ_c（HU）' の形で書く: {text!r}")
    if b < 0 or c < 0:
        raise ConfigError(f"度合いは 0 以上: {text!r}")
    return b, c


def check_anchor(sigma_a, blur_px, where):
    """設計書 F-2 の起動時の検査: σ_a がその回に使う σ_b の 2 倍より小さければエラー（錨が手がかりより細かい情報を元の絵から運ばないように）。"""
    if sigma_a <= 0:
        raise ConfigError(f"[{where}] brightness.sigma_px_512 は正")
    if sigma_a < 2.0 * blur_px:
        raise ConfigError(f"[{where}] 明るさを平均する広さ σ_a（{sigma_a}）が、使う手がかりの粗さ σ_b（{blur_px}）の 2 倍より小さい")


def _check_machine(machine, P):
    if machine["gpu_gen"] not in GPU_GENS: P.append(f"gpu_gen は {GPU_GENS}")
    if machine["num_threads"] < 0: P.append("num_threads ≥ 0")
    if not (1 <= machine["tb_port"] <= 65535): P.append("tb_port は 1..65535")


def _check_model(model, P):
    if model["base_ch"] < 1: P.append("base_ch ≥ 1")
    if len(model["ch_mult"]) < 2 or any(m < 1 for m in model["ch_mult"]): P.append("ch_mult は 2 段以上・各 1 以上")
    if model["blocks_per_level"] < 1 or model["mid_blocks"] < 1: P.append("blocks_per_level / mid_blocks ≥ 1")
    if model["groups"] < 1 or any((model["base_ch"] * m) % model["groups"] for m in model["ch_mult"]): P.append("groups は各段の幅（base_ch × ch_mult）を割り切ること")
    if model["emb_dim"] < 1 or model["emb_freqs"] < 1: P.append("emb_dim / emb_freqs ≥ 1")
    if not (0 < model["emb_freq_min"] < model["emb_freq_max"]): P.append("0 < emb_freq_min < emb_freq_max")
    if model["fold"] not in FOLDS: P.append(f"fold は {FOLDS}")


def size_multiple(model):
    """入力の一辺が満たす倍数（Haar 1 段の 2 × stride 2 が（段数 − 1）回）。"""
    return 2 * 2 ** (len(model["ch_mult"]) - 1)


def check_train_values(train, model, shared, machine):
    P = []
    t = train
    if t["optim.lr"] <= 0: P.append("optim.lr は正")
    if not (0.0 <= t["optim.beta1"] < 1.0 and 0.0 <= t["optim.beta2"] < 1.0): P.append("optim.beta1 / beta2 は [0, 1)")
    if t["optim.weight_decay"] < 0: P.append("optim.weight_decay ≥ 0")
    if t["optim.warmup_steps"] < 0: P.append("optim.warmup_steps ≥ 0")
    if t["optim.batch_size"] < 1: P.append("optim.batch_size ≥ 1")
    if t["optim.grad_clip"] <= 0: P.append("optim.grad_clip は正")
    if not (0.0 <= t["optim.ema_decay"] < 1.0): P.append("optim.ema_decay は [0, 1)")
    if t["optim.precision"] not in PRECISIONS: P.append(f"optim.precision は {PRECISIONS}")
    if t["optim.compile_mode"] not in COMPILE_MODES: P.append(f"optim.compile_mode は {COMPILE_MODES}")
    if t["optim.compile_after_steps"] < 0: P.append("optim.compile_after_steps ≥ 0")
    if t["optim.max_steps"] < 1: P.append("optim.max_steps ≥ 1")
    if t["optim.max_hours"] <= 0: P.append("optim.max_hours は正")
    if t["optim.seed"] < 0: P.append("optim.seed ≥ 0")
    m = size_multiple(model)
    if t["data.patch_size"] < m or t["data.patch_size"] % m: P.append(f"data.patch_size は {m} の倍数")
    if t["data.margin"] < 0 or t["data.margin"] < 3 * t["cue.blur_max_px"]: P.append("data.margin は 3 × cue.blur_max_px 以上（ぼかしがパッチの端で欠けないように）")
    if t["data.patches_per_slice"] < 1: P.append("data.patches_per_slice ≥ 1")
    if t["data.hu_min"] >= t["data.hu_max"]: P.append("data.hu_min < data.hu_max")
    if t["data.hu_scale"] <= 0: P.append("data.hu_scale は正")
    if not t["data.split_file"]: P.append("data.split_file を指定")
    if t["data.unlisted_cases"] not in UNLISTED: P.append(f"data.unlisted_cases は {UNLISTED}")
    if t["data.stats_patches"] < 0: P.append("data.stats_patches ≥ 0")
    if t["data.domain"] not in DOMAINS: P.append(f"data.domain は {DOMAINS}")
    if t["data.domain"] == "pcd" and not t["data.pcd_val_cases"]: P.append("data.domain: pcd のときは data.pcd_val_cases に 1 症例以上")
    if not (t["init.tag"] in ("latest", "best") or t["init.tag"].isdigit()): P.append("init.tag は latest | best | step 番号")
    for k in ("cue.drop_prob", "cue.blur_zero_prob", "cue.noise_zero_prob"):
        if not (0.0 <= t[k] <= 1.0): P.append(f"{k} は [0, 1]")
    if not (0 < t["cue.blur_min_px"] <= t["cue.blur_max_px"]): P.append("0 < cue.blur_min_px ≤ cue.blur_max_px")
    if not (0 < t["cue.noise_min_hu"] <= t["cue.noise_max_hu"]): P.append("0 < cue.noise_min_hu ≤ cue.noise_max_hu")
    if not (0 < t["diffusion.sigma_min"] < t["diffusion.sigma_max"]): P.append("0 < diffusion.sigma_min < sigma_max")
    if t["diffusion.sigma_data"] <= 0: P.append("diffusion.sigma_data は正")
    if t["diffusion.sigma_sampling"] not in SIGMA_SAMPLINGS: P.append(f"diffusion.sigma_sampling は {SIGMA_SAMPLINGS}")
    if t["sample.steps"] < 1: P.append("sample.steps ≥ 1")
    if t["sample.eta"] < 0: P.append("sample.eta ≥ 0")
    if t["sample.rho"] <= 0: P.append("sample.rho は正")
    if t["sample.precision"] not in PRECISIONS: P.append(f"sample.precision は {PRECISIONS}")
    for k in ("log.print_steps", "log.steps_per_epoch", "log.train_patch_steps", "log.n_train_patches", "log.val_steps", "log.val_patches",
              "log.preview_steps", "log.save_latest_steps", "log.save_weights_steps", "log.zoom_size", "log.zoom_scale"):
        if t[k] < 1: P.append(f"{k} ≥ 1")
    if t["log.preview_first_step"] < 0: P.append("log.preview_first_step ≥ 0")
    if t["log.nocue_n"] < 0: P.append("log.nocue_n ≥ 0")
    for k in ("log.eid_slices", "log.pcd_slices"):
        if not t[k] or any(not s.startswith("/") for s in t[k]): P.append(f"{k} は 1 つ以上、コンテナ内の絶対パスで書く")
    if not t["log.preview_cues"]: P.append("log.preview_cues は 1 つ以上")
    if t["log.diff_range_hu"] <= 0: P.append("log.diff_range_hu は正")
    if t["log.display_hu_min"] >= t["log.display_hu_max"]: P.append("log.display_hu_min < display_hu_max")
    if t["log.check_highpass_px"] <= 0: P.append("log.check_highpass_px は正")
    _check_model(model, P)
    _check_machine(machine, P)
    _raise("train values", P)
    for c in t["log.preview_cues"]:
        b, _c = parse_cue(c)
        check_anchor(shared["brightness.sigma_px_512"], b, f"log.preview_cues {c}")


def check_bridge_values(br, shared, machine):
    P = []
    tag_ok = lambda t: t in ("latest", "best") or t.isdigit()
    if not br["bridge.enc_run"] or not br["bridge.dec_run"]: P.append("bridge.enc_run / bridge.dec_run（run 名）を --enc_run / --dec_run で指定")
    if not tag_ok(br["bridge.enc_tag"]) or not tag_ok(br["bridge.dec_tag"]): P.append("bridge.enc_tag / dec_tag は latest | best | step 番号")
    if br["bridge.slice_step"] < 1: P.append("bridge.slice_step ≥ 1")
    if br["bridge.max_slices"] < 0: P.append("bridge.max_slices ≥ 0（0 で全部）")
    if br["bridge.seed"] < 0: P.append("bridge.seed ≥ 0")
    if br["cue.blur_px"] < 0 or br["cue.noise_hu"] < 0: P.append("cue.blur_px / cue.noise_hu ≥ 0")
    if br["ode.steps"] < 2: P.append("ode.steps ≥ 2")
    if br["ode.rho"] <= 0: P.append("ode.rho は正")
    if br["ode.decode_eta"] < 0: P.append("ode.decode_eta ≥ 0")
    if br["ode.batch"] < 1: P.append("ode.batch ≥ 1")
    if br["ode.precision"] not in PRECISIONS: P.append(f"ode.precision は {PRECISIONS}")
    _check_machine(machine, P)
    _raise("bridge values", P)
    if br["cue.use"] and br["anchor.decode"]:
        check_anchor(shared["brightness.sigma_px_512"], br["cue.blur_px"], "bridge")


def check_generate_values(gen, shared, machine):
    P = []
    if not gen["generate.run"]: P.append("generate.run（run 名）を --run で指定")
    if not (gen["generate.tag"] in ("latest", "best") or gen["generate.tag"].isdigit()): P.append("generate.tag は latest | best | step 番号")
    if gen["generate.k"] < 1: P.append("generate.k ≥ 1")
    if gen["generate.seed"] < 0: P.append("generate.seed ≥ 0")
    if gen["generate.slice_step"] < 1: P.append("generate.slice_step ≥ 1")
    if gen["generate.max_slices"] < 0: P.append("generate.max_slices ≥ 0（0 で全部）")
    if gen["cue.blur_px"] < 0 or gen["cue.noise_hu"] < 0: P.append("cue.blur_px / cue.noise_hu ≥ 0")
    if gen["sample.steps"] < 1: P.append("sample.steps ≥ 1")
    if gen["sample.eta"] < 0: P.append("sample.eta ≥ 0")
    if gen["sample.rho"] <= 0: P.append("sample.rho は正")
    if gen["sample.batch"] < 1: P.append("sample.batch ≥ 1")
    if gen["sample.precision"] not in PRECISIONS: P.append(f"sample.precision は {PRECISIONS}")
    if gen["sample.compile_mode"] not in COMPILE_MODES: P.append(f"sample.compile_mode は {COMPILE_MODES}")
    _check_machine(machine, P)
    _raise("generate values", P)
    if gen["cue.use"]:
        check_anchor(shared["brightness.sigma_px_512"], gen["cue.blur_px"], "generate")
