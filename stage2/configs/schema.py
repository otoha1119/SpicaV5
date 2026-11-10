"""stage2/configs/schema.py — 新 Stage 2（U = 形を決める回帰、V = 質感を作るフロー）の設定ファイル（data.yaml / u.yaml / v.yaml / infer.yaml / check.yaml）、
共有設定（configs/shared.yaml）、Stage 2 が読む configs/machines.yaml のキーの唯一の正。

原則（Stage 1 と同じ）: 既定値（フォールバック）は無い。ここに宣言したキーは**必ず**設定ファイルに存在しなければならず、未知のキー・型違いはエラー。
sh からの上書き（--flag value）は schema にあるフラグだけ受け付け、実効値を run の launch.yaml（推論・確認は出力フォルダの yaml）に残す。
検証・上書きの関数（validate / apply_overrides / flatten）は stage1/configs/schema.py から複製した（Stage 間で import しない）。

設計の正: docs/plans/20261111_detail_design/stage2_u_flow_spec.md（以下「設計書」）と RECONCILE.md（食い違いの裁定。こちらが優先）。

各キーの宣言: Key(type, flag, doc)
  type : int | float | str | bool | list（str のリスト）| ilist（int のリスト）。float は int の値も受け付ける。bool は YAML の true / false のみ
  flag : 上書き用のフラグ名（--flag value。list / ilist は --flag a,b,c。空リストは --flag ""）。None は上書き不可
フラグ名は「同時に読む組」（data + u + shared + machine、data + v + shared + machine、infer + machine、check + machine）の中で重複しないこと。
"""

from collections import namedtuple

Key = namedtuple("Key", "type flag doc")
ilist = type("ilist", (), {})  # int のリストを表す型の印

# ---------------------------------------------------------------------------
# stage2/configs/data.yaml — U と V の学習で共通のデータの決まり（設計書 E・F-1、RECONCILE #3・#9・#10・#12）
# ---------------------------------------------------------------------------
DATA = {
    "data.pseudo_eid_set":            Key(str,   "--pseudo_eid_set",            "疑似 EID の「組」のフォルダ名（machines.yaml の pseudo_eid_dir の下。bash start.sh generate が作る <run>_step<N>_b…_c…_a…）。yaml は空にして --pseudo_eid_set で渡す"),
    "data.allow_brightness_mismatch": Key(bool,  "--allow_brightness_mismatch", "疑似 EID の manifest.yaml の明るさの広さ（brightness_sigma_px_512）が configs/shared.yaml と違っても通す（RECONCILE #9。明示したときだけ）"),
    "data.hu_offset":                 Key(int,   None,                          "stored = HU + hu_offset（1400）"),
    "data.hu_min":                    Key(int,   None,                          "n = −1 にあたる HU（−1400）。切らない（線形の物差しの端）"),
    "data.hu_max":                    Key(int,   None,                          "n = +1 にあたる HU（4096）"),
    "data.upsample_scale":            Key(int,   None,                          "入力の補間の倍率（2。疑似 EID512・実 EID512 → 1024。その場で計算しファイル化しない。設計書 E-2）"),
    "data.upsample_interp":           Key(str,   None,                          "補間の方式（bicubic。OpenCV）"),
    "data.patches_per_slice":         Key(int,   "--patches_per_slice",         "1 回読んだスライスの組（入力 + 正解）から連続して切るパッチ数（8）。位置は毎回独立。同じスライスの 8 個は同じ 1 通り（k）"),
    "data.train_cases":               Key(list,  "--train_cases",               "学習に使う症例（PCD-003〜016）"),
    "data.val_cases":                 Key(list,  "--val_cases",                 "epoch 末の数字と表示に使う症例（PCD-017, 018）。学習には使わない"),
    "data.test_cases":                Key(list,  "--test_cases",                "学習・val で読まない症例（PCD-001, 002, 019, 020）。例外は log.pcd_slice の 1 枚（目視用）"),
    "data.eid_split_file":            Key(str,   "--eid_split_file",            "実 EID の症例分割（configs/splits/eid_v5.yaml。リポジトリ直下からの相対パス）。Stage 1 の P が train 側で学んでいるので、表示・確認に使う実 EID が train 側でないことを起動前に検査する（RECONCILE #12）"),
    "display.hu_min":                 Key(int,   "--display_hu_min",            "表示の下限 HU（−1400）"),
    "display.hu_max":                 Key(int,   "--display_hu_max",            "表示の上限 HU（3000）"),
    "display.diff_range_hu":          Key(float, "--diff_range_hu",             "差の絵の色の幅（±HU。白 = 0、青 −、赤 +）。乱数の差の列（300）"),
    "display.diff_blur_range_hu":     Key(float, "--diff_blur_range_hu",        "ぼかした差の絵の色の幅（±HU。50）"),
    "display.tb_scale":               Key(int,   "--tb_scale",                  "TensorBoard に出すフルのパネルを 1/N に縮める（2。ディスクの preview は等倍）"),
    "display.body_hu":                Key(float, "--body_hu",                   "「体内」の印 = 入力の広い平均 > この HU（−500）。**測るためだけ**に使い、ネットワークには入れない（設計書 G）"),
}

# ---------------------------------------------------------------------------
# 学習中の確認（U・V 共通のキー。設計書 F-2・F-3・G-1）。U と V の両方の schema に同じものを入れる
# ---------------------------------------------------------------------------
_LOG = {
    "log.print_steps":         Key(int,   "--print_steps",         "scalar・loss_log.txt の間隔（step）。100。最初は 1, 2, 4, …, 64 step 目にも出す"),
    "log.image_steps":         Key(int,   "--image_steps",         "TensorBoard images/train_patch の間隔（step）。500"),
    "log.n_images":            Key(int,   "--n_images",            "その行数"),
    "log.save_latest_steps":   Key(int,   "--save_latest_steps",   "latest/ の保存間隔（step）"),
    "log.save_weights_epochs": Key(int,   "--save_weights_epochs", "weights/step_NNNNNNN/ を残す間隔（epoch）。1"),
    "log.first_check_step":    Key(int,   "--first_check_step",    "この起動の最初にこの step 数で 1 回だけ、latest の保存と epoch 末の確認（log.first_check_slices 枚に絞った縮小版）を通す（配線の確認。0 で無し）"),
    "log.first_check_slices":  Key(int,   "--first_check_slices",  "その縮小版で、症例ごとに使う枚数（2）"),
    "log.val_slices_per_case": Key(int,   "--val_slices_per_case", "epoch 末の val で、val 症例ごとに使うフル 1024 のスライス数（等間隔。k00 固定）"),
    "log.pcd_slice":           Key(str,   "--pcd_slice",           "固定パネルの疑似 EID / PCD のスライス（<組>/k00 と pcd1024_dir の両方からの相対パス。PCD-002/PCD-002-236.png。test の症例でも可 = 目視用）"),
    "log.eid_slices":          Key(list,  "--eid_slices",          "並べて見る実 EID512 のスライス。**コンテナ内の絶対パス**の一覧。1 つ目が基準（固定パネルの右側・切り抜き）、2 つ目以降は 1 枚ずつ別のパネル（病院 EID など）。P の学習に使った症例（分割の train 側）は置けない"),
    "log.n_random":            Key(int,   "--n_random",            "epoch ごとに別の val のスライスを何枚パネルにするか（1）"),
    "log.crop_size":           Key(int,   "--crop_size",           "虫眼鏡の切り抜きの一辺（1024 の画素。256）"),
    "log.crop_scale":          Key(int,   "--crop_scale",          "その拡大率（2、最近傍）"),
    "log.pcd_crop_center":     Key(ilist, "--pcd_crop_center",     "log.pcd_slice の切り抜きの中心 [y, x]（1024 の画素）"),
    "log.eid_crop_center":     Key(ilist, "--eid_crop_center",     "log.eid_slices の 1 つ目の切り抜きの中心 [y, x]（補間後の 1024 の画素）。2 つ目以降は位置を決めていないので切り抜かない"),
    "log.preview_seed":        Key(int,   "--preview_seed",        "表示の乱数の種（毎 epoch 同じ乱数。V だけが使う。乱数の差は seed と seed + 1）"),
    "mag.slices_per_case":     Key(int,   "--mag_slices_per_case", "虫眼鏡の数字（設計書 G-1）に使う枚数（症例ごと、等間隔）"),
    "mag.eid_cases":           Key(list,  "--mag_eid_cases",       "imagecas の組: eid_dir の下の症例名。分割の train 側は置けない（RECONCILE #12）。[] で出さない"),
    "mag.hospital_dirs":       Key(list,  "--mag_hospital_dirs",   "hospital の組: 病院 EID512 の症例フォルダ（コンテナ内の絶対パス）。評価専用 = 見るだけ。[] で出さない（そのマシンに無いときは --mag_hospital_dirs \"\"）"),
    "mag.tile":                Key(int,   "--mag_tile",            "帯域パワーを測るタイルの一辺（128）"),
}

_OPTIM_COMMON = {
    "optim.lr":                  Key(float, "--lr",                  "Adam の学習率"),
    "optim.beta1":               Key(float, "--beta1",               "Adam β1（0.9）"),
    "optim.beta2":               Key(float, "--beta2",               "Adam β2（0.999）"),
    "optim.batch_size":          Key(int,   "--batch_size",          "batch（パッチ数）"),
    "optim.max_steps":           Key(int,   "--max_steps",           "学習の上限 step。max_hours と早いほうで止まる。上限で止まったときは latest と best（rule: last）を保存して正常終了"),
    "optim.steps_per_epoch":     Key(int,   "--steps_per_epoch",     "1 epoch の step 数（epoch 末に val・確認・パネル・重みの保存）"),
    "optim.max_hours":           Key(float, "--max_hours",           "学習の上限時間（再開をまたいだ累計）"),
    "optim.ema_decay":           Key(float, "--ema_decay",           "EMA の減衰の上限。序盤は min(ema_decay, (1 + n) / (10 + n))。val・画像・推論・V の学習に使うのは EMA の重み"),
    "optim.precision":           Key(str,   "--precision",           "学習の forward の演算精度 fp32 | bf16（CUDA の autocast）。損失・明るさの固定・val・推論は常に FP32"),
    "optim.compile":             Key(bool,  "--compile",             "CUDA で学習の forward を torch.compile する（CPU では無効）。val・画像は compile しない EMA"),
    "optim.compile_mode":        Key(str,   "--compile_mode",        "torch.compile の mode: max-autotune | reduce-overhead | default。PC で max-autotune が落ちたら default に下げる"),
    "optim.compile_after_steps": Key(int,   "--compile_after_steps", "この起動の最初の N step は compile せずに回す（最初の scalar をすぐ出すため。compile は N + 1 step 目に始まり、初回は数分かかる）。0 で最初から"),
    "optim.channels_last":       Key(bool,  "--channels_last",       "学習するネットを channels_last にする"),
    "optim.seed":                Key(int,   "--seed",                "python / numpy / torch の乱数 seed"),
    "patch.size":                Key(int,   "--patch_size",          "学習パッチの一辺（1024 の画素）"),
}

# ---------------------------------------------------------------------------
# stage2/configs/u.yaml — U（設計書 B。RECONCILE #11: パッチ 256・batch 8）
# ---------------------------------------------------------------------------
U = {
    "net.arch":      Key(str,  None, "unet_ilumenate（旧 Stage 2 の U-Net をそのまま。設計書 B-1）"),
    "net.base_ch":   Key(int,  "--base_ch", "最上段の幅（128）"),
    "net.n_pool":    Key(int,  "--n_pool",  "max pool の回数（2）"),
    "net.init_type": Key(str,  None, "xavier_uniform | torch"),
    "net.final_act": Key(str,  None, "linear | tanh（設計書 A-3: 値は線形のまま = linear）"),
    "net.residual":  Key(bool, None, "true で f = 入力 + net(入力)（設計書 B-1 は false）"),
    **_OPTIM_COMMON,
    **_LOG,
}

# ---------------------------------------------------------------------------
# stage2/configs/v.yaml — V（設計書 C。PMRF: Ohayon et al. 2025。出典は stage2/UPSTREAM.md）
# ---------------------------------------------------------------------------
V = {
    "u.run":                 Key(str,   "--u_run", "固定して使う U の run 名（flow_checkpoints_dir/u/ の下）。yaml は空にして --u_run で渡す。再開では V の重みに同梱した net_U.pth を使う"),
    "u.tag":                 Key(str,   "--u_tag", "U の重み: best | latest | step 番号（best = 最終 epoch の EMA）"),
    "net.base_ch":           Key(int,   "--base_ch",          "最上段の幅（64）"),
    "net.ch_mult":           Key(ilist, "--ch_mult",          "段ごとの幅の倍率（[1, 2, 4, 4] → 64 / 128 / 256 / 256。畳んだ後の解像度の 1・1/2・1/4・1/8）"),
    "net.blocks_per_level":  Key(int,   "--blocks_per_level", "下り・上りの各段の ResBlock 数（2）"),
    "net.mid_blocks":        Key(int,   "--mid_blocks",       "中央の ResBlock 数（2）"),
    "net.emb_dim":           Key(int,   "--emb_dim",          "時刻・強さの埋め込みの次元（256）"),
    "net.emb_freq_dim":      Key(int,   "--emb_freq_dim",     "t と σs をそれぞれ正弦波にする次元（64）"),
    "flow.unit_hu":          Key(float, None,                 "V の中の値の単位（HU ÷ 100。設計書 E-1）。再開で変えられない"),
    "flow.sigma_s_min_hu":   Key(float, "--sigma_s_min_hu",   "学習でくじ引きする乱数の強さ σs の下端（25 HU）。[min, max] を対数一様。min = max なら PMRF 論文どおりの定数"),
    "flow.sigma_s_max_hu":   Key(float, "--sigma_s_max_hu",   "同上端（400 HU）"),
    "flow.sigma_s_hu":       Key(float, "--sigma_s_hu",       "学習中の表示・確認に使う σs（100 HU）。推論は infer.yaml"),
    "flow.steps":            Key(int,   "--flow_steps",       "学習中の表示・確認の Euler の回数 K（25）"),
    "flow.solver":           Key(str,   None,                 "euler（PMRF Algorithm 1）"),
    "flow.t_sampling":       Key(str,   "--t_sampling",       "学習時の t の引き方: uniform（設計書。サンプルごとに U[0, 1]）| stratified_uniform（PMRF の公式実装の既定。batch を等分した区間から 1 つずつ）"),
    "optim.weight_decay":    Key(float, "--weight_decay",     "weight decay（0）"),
    "optim.warmup_steps":    Key(int,   "--warmup_steps",     "学習率を 0 から直線で上げる step 数（1,000）。その後は一定"),
    "optim.grad_clip":       Key(float, "--grad_clip",        "勾配ノルムのクリップ（1.0）"),
    "optim.u_precision":     Key(str,   "--u_precision",      "学習中の固定した U の順伝播の精度 bf16 | fp32（設計書 C-5。fp32 だと 1 step が約 2 倍）。表示・確認・推論の U は常に FP32"),
    "optim.u_compile":       Key(bool,  "--u_compile",        "学習中の固定した U を torch.compile する"),
    "patch.margin":          Key(int,   "--margin",           "パッチの周りに付けて切る余白（1024 の画素）。U と明るさの固定を (size + 2·margin) 四方で計算して中央を V に渡す。明るさの広さ（1024 の σ）の 3 倍以上（RECONCILE #11）"),
    "log.val_patches":       Key(int,   "--val_patches",      "val/flow の固定パッチ数（16。位置・t・ε・σs を固定）"),
    "log.val_seed":          Key(int,   "--val_seed",         "その seed"),
    **_OPTIM_COMMON,
    **_LOG,
}

# ---------------------------------------------------------------------------
# configs/shared.yaml — 両 Stage で共有する値（RECONCILE #1・#9）。Stage 2 が使うキーだけ検査する。Stage 2 からは上書きできない
# （疑似 EID と違う広さで回すときは data.allow_brightness_mismatch を明示する）
# ---------------------------------------------------------------------------
SHARED = {
    "brightness.sigma_px_512": Key(float, None, "明るさを平均する広さ（Gaussian の σ、512 の画素。16）。Stage 2 は upsample_scale 倍（1024 で 32）して使う"),
}

# ---------------------------------------------------------------------------
# configs/machines.yaml のうち Stage 2 が使うキー（他のキーは無視）
# ---------------------------------------------------------------------------
MACHINE = {
    "gpu_gen":              Key(int, None,               "GPU 世代 30 | 40 | 50 | 0(CPU)。0 以外は CUDA 必須"),
    "container_data_root":  Key(str, None,               "データのマウント先（コンテナ内）"),
    "num_threads":          Key(int, "--num_threads",    "DataLoader の worker 数（0 でメインプロセス）"),
    "stage2_tb_port":       Key(int, None,               "Stage 2 の TensorBoard のポート（6007）"),
    "eid_dir":              Key(str, "--eid_dir",        "実 EID512（ImageCAS、EID_v5）。確認（imagecas の組）にだけ使う"),
    "pcd1024_dir":          Key(str, "--pcd1024_dir",    "正解の PCD1024（PCD1024_v1）"),
    "flow_checkpoints_dir": Key(str, None,               "新 Stage 2 の run の置き場（/workspace/runs/stage2）。この下に u/<run>/、v/<run>/。監視ツールは語尾 checkpoints_dir で拾う"),
    "pseudo_eid_dir":       Key(str, "--pseudo_eid_dir", "疑似 EID の置き場（/workspace/DataSet/PseudoEID512）。この下に <組>/k00, k01, …/<症例>/<スライス>.png と <組>/manifest.yaml"),
}

# ---------------------------------------------------------------------------
# stage2/configs/infer.yaml — フォルダ丸ごとの推論（run_infer.py）
# ---------------------------------------------------------------------------
INFER = {
    "infer.kind":           Key(str,   "--kind",           "どちらの run の重みか: u | v（v の重みには U の写しが同梱されているので、V の重みディレクトリだけで完結する）"),
    "infer.run":            Key(str,   "--run",            "run 名（flow_checkpoints_dir/<kind>/ の下）。yaml は空にして --run で渡す"),
    "infer.tag":            Key(str,   "--tag",            "重み: best | latest | step 番号"),
    "infer.mode":           Key(str,   "--mode",           "u（U の出力 μ だけ）| uv（U + V の最終出力。kind: v が必要）"),
    "infer.input_dir":      Key(str,   "--input_dir",      "入力の PNG フォルダ（<症例>/<スライス>.png か、直下に PNG）。コンテナ内の絶対パス"),
    "infer.upsample":       Key(bool,  "--upsample",       "true = 入力は 512 で、学習と同じ補間（run の data.upsample_*）をその場で掛ける。false = 既に 1024（補間済み）"),
    "infer.cases":          Key(list,  "--cases",          "この症例だけ（[] で全部。評価専用も除外しない）"),
    "infer.slice_step":     Key(int,   "--slice_step",     "症例ごとに何枚おきか（1 = 全部）"),
    "infer.max_slices":     Key(int,   "--max_slices",     "症例ごとの上限（0 = 全部）"),
    "infer.sigma_s_hu":     Key(float, "--sigma_s_hu",     "乱数の強さ σs（HU。100）。学習した幅（run の flow.sigma_s_min_hu〜max_hu）の外はエラー"),
    "infer.steps":          Key(int,   "--steps",          "Euler の回数 K（25）"),
    "infer.seed":           Key(int,   "--seed",           "乱数の種。スライスごとの種 = sha256('<seed>/<症例>/<スライス名>')"),
    "infer.n_seeds":        Key(int,   "--n_seeds",        "seed, seed + 1, … の何通りを出すか（1。2 以上で full_seed<NN>/ に分かれる）"),
    "infer.precision":      Key(str,   "--precision",      "V の本体の精度 fp32 | bf16（U・入口・出口・明るさは常に FP32。設計の既定は fp32）"),
    "infer.save_mu":        Key(bool,  "--save_mu",        "U の出力 μ を full_mu/ に残す"),
    "infer.save_input1024": Key(bool,  "--save_input1024", "補間した入力を full_input1024/ に残す"),
    "infer.save_panel":     Key(bool,  "--save_panel",     "[入力↑ | μ | U+V] のパネルを full_panel/ に残す"),
    "infer.out_root":       Key(str,   "--out_root",       "出力の根（/workspace/outputs/stage2）。この下に <run>_<重み>_<入力名>_<時刻>/"),
}

# ---------------------------------------------------------------------------
# stage2/configs/check.yaml — 確認の道具（run_check.py。設計書 G）
# ---------------------------------------------------------------------------
CHECK = {
    "check.kind":            Key(str,   "--kind",            "u | v"),
    "check.run":             Key(str,   "--run",             "run 名。--run で渡す"),
    "check.tag":             Key(str,   "--tag",             "重み: best | latest | step 番号"),
    "check.what":            Key(list,  "--what",            "やる確認: magnifier（虫眼鏡の数字。G-1）, seeds（乱数を変えた出力。G-3）, flicker（隣り合うスライス。G-4）。seeds・flicker の乱数は kind: v のときだけ意味がある"),
    "check.pseudo_eid_set":  Key(str,   "--pseudo_eid_set",  "pseudo の組に使う疑似 EID の「組」。空文字 = run が学習に使った組"),
    "check.sigma_s_list":    Key(ilist, "--sigma_s_list",    "試す σs（HU）の一覧（[50, 100, 200]。設計書 G-3 の 6。学習し直さない）。kind: u では使わない"),
    "check.steps":           Key(int,   "--steps",           "Euler の回数 K（25）"),
    "check.seed":            Key(int,   "--seed",            "乱数の種（seeds は seed, seed + 1, …）"),
    "check.n_seeds":         Key(int,   "--n_seeds",         "seeds の通り数（8。最初の 2 つで差の絵、全部で平均と標準偏差の絵）"),
    "check.precision":       Key(str,   "--precision",       "V の本体の精度 fp32 | bf16"),
    "check.old_output_dir":  Key(str,   "--old_output_dir",  "比べる相手（旧モデル）の出力フォルダ（<症例>/<スライス>.png、1024、16bit）。空文字なら無し。あるスライスだけ表の old の段に入る"),
    "check.out_root":        Key(str,   "--out_root",        "出力の根。空文字 = <run>/check/"),
    "mag.slices_per_case":   Key(int,   "--mag_slices_per_case", "虫眼鏡に使う枚数（症例ごと、等間隔。20）"),
    "mag.eid_cases":         Key(list,  "--mag_eid_cases",   "imagecas の組: eid_dir の下の症例名（分割の train 側は置けない）"),
    "mag.hospital_dirs":     Key(list,  "--mag_hospital_dirs", "hospital の組: 病院 EID512 の症例フォルダ（絶対パス）。[] で出さない"),
    "mag.tile":              Key(int,   "--mag_tile",        "帯域パワーのタイルの一辺（128）"),
    "seeds.slices":          Key(list,  "--seeds_slices",    "seeds に使うスライスと切り抜きの中心。'pseudo:<症例>/<スライス>.png:<y>:<x>'（<組>/k00 相対）か 'eid:<絶対パス>:<y>:<x>'（実 EID512。座標は補間後の 1024）"),
    "seeds.crop_size":       Key(int,   "--seeds_crop_size", "切り抜きの一辺（256）"),
    "seeds.crop_scale":      Key(int,   "--seeds_crop_scale", "拡大率（2）"),
    "flicker.series":        Key(list,  "--flicker_series",  "連続スライスの系列: 'pseudo:<症例>'（<組>/k00。正解 PCD1024 の帯も並べる）か 'eid:<症例フォルダの絶対パス>'"),
    "flicker.n_slices":      Key(int,   "--flicker_n_slices", "連続何枚か（32。症例の中央付近）"),
    "flicker.stretch":       Key(int,   "--flicker_stretch", "帯を厚み方向に何倍に伸ばして保存するか（8）"),
}

PRECISIONS = ("fp32", "bf16")
COMPILE_MODES = ("max-autotune", "reduce-overhead", "default")
GPU_GENS = (0, 30, 40, 50)
ARCHS = ("unet_ilumenate",)
INIT_TYPES = ("xavier_uniform", "torch")
FINAL_ACTS = ("linear", "tanh")
SOLVERS = ("euler",)
T_SAMPLINGS = ("uniform", "stratified_uniform")
KINDS = ("u", "v")
INFER_MODES = ("u", "uv")
CHECKS = ("magnifier", "seeds", "flicker")
INTERPS = ("nearest", "bilinear", "bicubic", "area", "lanczos")
# 再開で変えられないキー（重みの意味が変わる）。net.* は全部変えられない
FROZEN_DATA = ("data.hu_offset", "data.hu_min", "data.hu_max", "data.upsample_scale", "data.upsample_interp")
FROZEN_U = ()
FROZEN_V = ("flow.unit_hu", "flow.solver", "u.run", "u.tag")
# V を学習するとき、U の実効設定と一致していなければならない data のキー（U と V で絵の物差し・補間・分割が違うと μ の意味が変わる）
MUST_MATCH_U = FROZEN_DATA + ("data.train_cases", "data.val_cases", "data.test_cases")


# ---------------------------------------------------------------------------
# 検証・変換（stage1/configs/schema.py と同じ）
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


def sigma_px_1024(data, shared):
    """明るさを平均する広さ（出力の画素 = 1024 での σ）。configs/shared.yaml の値（512 基準）× 補間の倍率（RECONCILE #1: 16 × 2 = 32）。"""
    return float(shared["brightness.sigma_px_512"]) * data["data.upsample_scale"]


def lock_radius(sigma_px):
    """Gaussian の核の半径（3σ で打ち切り。models/lock.py と同じ式）。"""
    import math
    return int(math.ceil(3.0 * sigma_px))


def parse_seed_slice(text):
    """'pseudo:<症例>/<スライス>.png:<y>:<x>' | 'eid:<絶対パス>:<y>:<x>' → (kind, path, y, x)。"""
    parts = text.split(":")
    try:
        kind, path, y, x = parts[0], ":".join(parts[1:-2]), int(parts[-2]), int(parts[-1])
    except (ValueError, IndexError):
        raise ConfigError(f"seeds.slices は 'pseudo:<症例>/<スライス>.png:<y>:<x>' か 'eid:<絶対パス>:<y>:<x>' の形: {text!r}")
    if kind not in ("pseudo", "eid") or not path or (kind == "eid" and not path.startswith("/")):
        raise ConfigError(f"seeds.slices の種類は pseudo（<組>/k00 相対）か eid（絶対パス）: {text!r}")
    return kind, path, y, x


def parse_series(text):
    """'pseudo:<症例>' | 'eid:<症例フォルダの絶対パス>' → (kind, path)。"""
    kind, _, path = text.partition(":")
    if kind not in ("pseudo", "eid") or not path or (kind == "eid" and not path.startswith("/")):
        raise ConfigError(f"flicker.series は 'pseudo:<症例>' か 'eid:<症例フォルダの絶対パス>' の形: {text!r}")
    return kind, path


def _check_machine(machine, P):
    if machine["gpu_gen"] not in GPU_GENS: P.append(f"gpu_gen は {GPU_GENS}")
    if machine["num_threads"] < 0: P.append("num_threads ≥ 0")
    if not (1 <= machine["stage2_tb_port"] <= 65535): P.append("stage2_tb_port は 1..65535")


def _check_data(d, shared, P):
    if not d["data.pseudo_eid_set"]: P.append("data.pseudo_eid_set（疑似 EID の組のフォルダ名）を --pseudo_eid_set で指定")
    if "/" in d["data.pseudo_eid_set"]: P.append("data.pseudo_eid_set はフォルダ名だけ（pseudo_eid_dir の直下）")
    if d["data.hu_min"] >= d["data.hu_max"]: P.append("data.hu_min < data.hu_max")
    if d["data.upsample_scale"] < 1: P.append("data.upsample_scale ≥ 1")
    if d["data.upsample_interp"] not in INTERPS: P.append(f"data.upsample_interp は {INTERPS}")
    if d["data.patches_per_slice"] < 1: P.append("data.patches_per_slice ≥ 1")
    tr, va, te = d["data.train_cases"], d["data.val_cases"], d["data.test_cases"]
    if not tr or not va: P.append("data.train_cases / data.val_cases は 1 症例以上")
    dup = (set(tr) & set(va)) | (set(tr) & set(te)) | (set(va) & set(te))
    if dup or len(set(tr)) != len(tr) or len(set(va)) != len(va) or len(set(te)) != len(te): P.append(f"症例が複数のリストに入っている・重複している: {sorted(dup)}")
    if not d["data.eid_split_file"]: P.append("data.eid_split_file を指定")
    if d["display.hu_min"] >= d["display.hu_max"]: P.append("display.hu_min < display.hu_max")
    if d["display.diff_range_hu"] <= 0 or d["display.diff_blur_range_hu"] <= 0: P.append("display.diff_range_hu / diff_blur_range_hu は正")
    if d["display.tb_scale"] < 1: P.append("display.tb_scale ≥ 1")
    if shared["brightness.sigma_px_512"] <= 0: P.append("brightness.sigma_px_512 は正")


def _check_common(c, P):
    if c["optim.lr"] <= 0: P.append("optim.lr は正")
    if not (0.0 <= c["optim.beta1"] < 1.0 and 0.0 <= c["optim.beta2"] < 1.0): P.append("optim.beta1 / beta2 は [0, 1)")
    if c["optim.batch_size"] < 1: P.append("optim.batch_size ≥ 1")
    if c["optim.max_steps"] < 1 or c["optim.steps_per_epoch"] < 1: P.append("optim.max_steps / steps_per_epoch ≥ 1")
    if c["optim.max_hours"] <= 0: P.append("optim.max_hours は正")
    if not (0.0 <= c["optim.ema_decay"] < 1.0): P.append("optim.ema_decay は [0, 1)")
    if c["optim.precision"] not in PRECISIONS: P.append(f"optim.precision は {PRECISIONS}")
    if c["optim.compile_mode"] not in COMPILE_MODES: P.append(f"optim.compile_mode は {COMPILE_MODES}")
    if c["optim.compile_after_steps"] < 0: P.append("optim.compile_after_steps ≥ 0")
    if c["optim.seed"] < 0: P.append("optim.seed ≥ 0")
    for k in ("log.print_steps", "log.image_steps", "log.n_images", "log.save_latest_steps", "log.save_weights_epochs", "log.first_check_slices",
              "log.val_slices_per_case", "log.crop_size", "log.crop_scale", "mag.slices_per_case", "mag.tile"):
        if c[k] < 1: P.append(f"{k} ≥ 1")
    if c["log.first_check_step"] < 0 or c["log.n_random"] < 0: P.append("log.first_check_step / log.n_random ≥ 0")
    if not c["log.pcd_slice"] or c["log.pcd_slice"].startswith("/"): P.append("log.pcd_slice は <症例>/<スライス>.png（相対パス）")
    if not c["log.eid_slices"] or any(not s.startswith("/") for s in c["log.eid_slices"]): P.append("log.eid_slices は 1 つ以上、コンテナ内の絶対パスで書く")
    if any(not s.startswith("/") for s in c["mag.hospital_dirs"]): P.append("mag.hospital_dirs はコンテナ内の絶対パス")
    for k in ("log.pcd_crop_center", "log.eid_crop_center"):
        if len(c[k]) != 2 or min(c[k]) < 0: P.append(f"{k} は [y, x]")


def check_u_values(data, u, shared, machine):
    P = []
    _check_data(data, shared, P)
    _check_common(u, P)
    if u["net.arch"] not in ARCHS: P.append(f"net.arch は {ARCHS}")
    if u["net.base_ch"] < 1 or u["net.n_pool"] < 1: P.append("net.base_ch / net.n_pool ≥ 1")
    if u["net.init_type"] not in INIT_TYPES: P.append(f"net.init_type は {INIT_TYPES}")
    if u["net.final_act"] not in FINAL_ACTS: P.append(f"net.final_act は {FINAL_ACTS}")
    m = 2 ** max(u["net.n_pool"], 0)
    if u["patch.size"] < m or u["patch.size"] % m: P.append(f"patch.size は {m} の倍数")
    r = lock_radius(sigma_px_1024(data, shared))
    if u["patch.size"] <= 2 * r:
        P.append(f"patch.size（{u['patch.size']}）が明るさの核（半径 {r} = 幅 {2 * r + 1} 画素）より小さい（RECONCILE #11。パッチの中で広い平均を取れない）")
    _check_machine(machine, P)
    _raise("u values", P)


def v_size_multiple(v):
    """V の入力の一辺が満たす倍数（Haar 1 段の 2 × stride 2 が（段数 − 1）回。[1, 2, 4, 4] なら 16）。"""
    return 2 * 2 ** (len(v["net.ch_mult"]) - 1)


def check_v_values(data, v, shared, machine):
    P = []
    _check_data(data, shared, P)
    _check_common(v, P)
    if not v["u.run"]: P.append("u.run（固定して使う U の run 名）を --u_run で指定")
    if not (v["u.tag"] in ("latest", "best") or v["u.tag"].isdigit()): P.append("u.tag は best | latest | step 番号")
    if v["net.base_ch"] < 1: P.append("net.base_ch ≥ 1")
    if len(v["net.ch_mult"]) < 2 or any(m < 1 for m in v["net.ch_mult"]): P.append("net.ch_mult は 2 段以上・各 1 以上")
    if v["net.blocks_per_level"] < 1 or v["net.mid_blocks"] < 1: P.append("net.blocks_per_level / mid_blocks ≥ 1")
    if v["net.emb_dim"] < 1 or v["net.emb_freq_dim"] < 2 or v["net.emb_freq_dim"] % 2: P.append("net.emb_dim ≥ 1、net.emb_freq_dim は 2 以上の偶数")
    if v["flow.unit_hu"] <= 0: P.append("flow.unit_hu は正")
    if not (0 < v["flow.sigma_s_min_hu"] <= v["flow.sigma_s_max_hu"]): P.append("0 < flow.sigma_s_min_hu ≤ flow.sigma_s_max_hu")
    if not (v["flow.sigma_s_min_hu"] <= v["flow.sigma_s_hu"] <= v["flow.sigma_s_max_hu"]): P.append("flow.sigma_s_hu は学習する幅 [sigma_s_min_hu, sigma_s_max_hu] の中")
    if v["flow.steps"] < 1: P.append("flow.steps ≥ 1")
    if v["flow.solver"] not in SOLVERS: P.append(f"flow.solver は {SOLVERS}")
    if v["flow.t_sampling"] not in T_SAMPLINGS: P.append(f"flow.t_sampling は {T_SAMPLINGS}")
    if v["optim.weight_decay"] < 0 or v["optim.warmup_steps"] < 0: P.append("optim.weight_decay / warmup_steps ≥ 0")
    if v["optim.grad_clip"] <= 0: P.append("optim.grad_clip は正")
    if v["optim.u_precision"] not in PRECISIONS: P.append(f"optim.u_precision は {PRECISIONS}")
    m = v_size_multiple(v)
    if v["patch.size"] < m or v["patch.size"] % m: P.append(f"patch.size は {m} の倍数")
    r = lock_radius(sigma_px_1024(data, shared))
    if v["patch.margin"] < r:
        P.append(f"patch.margin（{v['patch.margin']}）は明るさの広さ（1024 で σ {sigma_px_1024(data, shared):g}）の 3 倍 = {r} 以上（RECONCILE #11。V に渡す中央で、広い平均がパッチの端の反射に触れないように）")
    if v["log.val_patches"] < 1 or v["log.val_seed"] < 0: P.append("log.val_patches ≥ 1、log.val_seed ≥ 0")
    _check_machine(machine, P)
    _raise("v values", P)


def _check_run_ref(c, sec, P):
    if c[f"{sec}.kind"] not in KINDS: P.append(f"{sec}.kind は {KINDS}")
    if not c[f"{sec}.run"]: P.append(f"{sec}.run（run 名）を --run で指定")
    if not (c[f"{sec}.tag"] in ("latest", "best") or c[f"{sec}.tag"].isdigit()): P.append(f"{sec}.tag は best | latest | step 番号")
    if c[f"{sec}.precision"] not in PRECISIONS: P.append(f"{sec}.precision は {PRECISIONS}")
    if c[f"{sec}.steps"] < 1 or c[f"{sec}.seed"] < 0 or c[f"{sec}.n_seeds"] < 1: P.append(f"{sec}.steps ≥ 1、{sec}.seed ≥ 0、{sec}.n_seeds ≥ 1")


def check_infer_values(infer, machine):
    P = []
    _check_run_ref(infer, "infer", P)
    if infer["infer.mode"] not in INFER_MODES: P.append(f"infer.mode は {INFER_MODES}")
    if infer["infer.mode"] == "uv" and infer["infer.kind"] != "v": P.append("infer.mode: uv には V の run が必要（--kind v）")
    if not infer["infer.input_dir"].startswith("/"): P.append("infer.input_dir をコンテナ内の絶対パスで指定（--input_dir）")
    if infer["infer.slice_step"] < 1 or infer["infer.max_slices"] < 0: P.append("infer.slice_step ≥ 1、infer.max_slices ≥ 0（0 で全部）")
    if infer["infer.sigma_s_hu"] <= 0: P.append("infer.sigma_s_hu は正")
    if not infer["infer.out_root"].startswith("/"): P.append("infer.out_root は絶対パス")
    _check_machine(machine, P)
    _raise("infer values", P)


def check_check_values(check, machine):
    P = []
    _check_run_ref(check, "check", P)
    bad = [w for w in check["check.what"] if w not in CHECKS]
    if bad or not check["check.what"]: P.append(f"check.what は {CHECKS} から 1 つ以上: {bad}")
    if check["check.kind"] == "v" and (not check["check.sigma_s_list"] or min(check["check.sigma_s_list"]) <= 0): P.append("check.sigma_s_list は正の HU を 1 つ以上")
    if check["check.old_output_dir"] and not check["check.old_output_dir"].startswith("/"): P.append("check.old_output_dir は絶対パスか空文字")
    if check["check.out_root"] and not check["check.out_root"].startswith("/"): P.append("check.out_root は絶対パスか空文字")
    for k in ("mag.slices_per_case", "mag.tile", "seeds.crop_size", "seeds.crop_scale", "flicker.n_slices", "flicker.stretch"):
        if check[k] < 1: P.append(f"{k} ≥ 1")
    if any(not s.startswith("/") for s in check["mag.hospital_dirs"]): P.append("mag.hospital_dirs はコンテナ内の絶対パス")
    _check_machine(machine, P)
    _raise("check values", P)
    for s in check["seeds.slices"]:
        parse_seed_slice(s)
    for s in check["flicker.series"]:
        parse_series(s)
