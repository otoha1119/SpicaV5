"""ablation/SRCycleGAN/configs/schema.py — この ablation の設定ファイル（train.yaml / infer.yaml）と、読む machines.yaml のキーの唯一の正。

原則は Stage 1 / Stage 2 と同じ: 既定値（フォールバック）は無い。ここに宣言したキーは**必ず**設定ファイルに存在しなければならず、未知のキー・型違いはエラー。
sh からの上書き（--flag value）は schema にあるフラグだけ受け付け、実効値を launch.yaml に残す。
検証・上書きの関数（validate / validate_shared / apply_overrides / flatten）は stage2/configs/schema.py から 2026-10-27 に複製した（ablation は Stage 1 / 2 から import しない）。

machines.yaml は Stage 共通ファイルなので、ここは「自分が使うキー」（MACHINE）の存在と型だけ検査し、他のキーは無視する（validate_shared）。
Stage 1 / Stage 2 の schema は未知キーを拒むので、この ablation 専用のキー（ablation_checkpoints_dir / ablation_tb_port）は
stage1/configs/schema.py と stage2/configs/schema.py の MACHINE にも flag None で宣言してある。キーを足すときは 3 箇所に書く。

各キーの宣言: Key(type, flag, doc)
  type : int | float | str | bool | list（str のリスト）。float は int の値も受け付ける。bool は YAML の true / false のみ
  flag : 上書き用のフラグ名（--flag value。bool は --flag true|false、list は --flag a,b,c、空リストは --flag ""）。None は上書き不可
  doc  : 意味と出典。論文 = Zheng et al., "SR-CycleGAN: super-resolution of clinical CT to micro-CT level with multi-modality super-resolution loss,"
         J. Med. Imaging 9(2) 024003 (2022)（JMI）/ SPIE Medical Imaging 2020, arXiv:1912.12838（SPIE）。コード = 著者公開コード（SpicaV2 commit 741924c）
"""

from collections import namedtuple

Key = namedtuple("Key", "type flag doc")


# ---------------------------------------------------------------------------
# configs/train.yaml — 学習パラメータ（論文 Sec. 3.3 / Table 1 とコード 741924c の値を全部ここに）
# ---------------------------------------------------------------------------
TRAIN = {
    # optim
    "optim.lr":              Key(float, "--lr",              'JMI Sec. 3.3 "learning rate 1e-5 ... epochs 1–100, then linear decay to 0 over epochs 100–200"。コード train_options lr 0.00001'),
    "optim.beta1":           Key(float, "--beta1",           "Adam β1。論文未記載 → コード beta1 0.5（junyanz 既定）"),
    "optim.beta2":           Key(float, "--beta2",           "Adam β2。論文未記載 → コード 0.999"),
    "optim.batch_size":      Key(int,   "--batch_size",      'JMI Sec. 3.3 "batch size 4"'),
    "optim.n_epochs":        Key(int,   "--n_epochs",        "lr 一定の epoch 数。論文 100（niter）"),
    "optim.n_epochs_decay":  Key(int,   "--n_epochs_decay",  "lr を 0 へ線形減衰させる epoch 数。論文 100（niter_decay）。合計 200 epoch"),
    "optim.seed":            Key(int,   "--seed",            "python / numpy / torch の乱数 seed（再現条件）。cudnn.benchmark は有効なので完全決定ではない"),
    # loss（論文 Eq. 8 の総和: λ5 L_GAN(G1) + λ6 L_GAN(G2) + λ7 L_cyc + λ1 L_S(G1) + λ2 L_S(G2) + λ3 L_D + λ4 L_U）
    "loss.gan_mode":         Key(str,   "--gan_mode",        "vanilla（論文 Eq. 2 の log 形 = BCEWithLogits。ユーザー決定 2026-10-27）| lsgan（コード 741924c の既定。ablation 用）"),
    "loss.lambda_gan_A":     Key(float, "--lambda_gan_A",    "λ5: GAN loss of G1 and D1（JMI Table 1 = 1.0）"),
    "loss.lambda_gan_B":     Key(float, "--lambda_gan_B",    "λ6: GAN loss of G2 and D2（JMI Table 1 = 1.0）"),
    "loss.lambda_cycle":     Key(float, "--lambda_cycle",    "λ7: cycle-consistency loss（JMI Table 1 = 1.0。両方向に同じ重み）"),
    "loss.cycle_norm":       Key(str,   "--cycle_norm",      "cycle loss のノルム: l2（論文 Eq. 1 squared l2-norm。ユーザー決定 2026-10-27）| l1（コード 741924c の L1Loss）"),
    "loss.lambda_ssim_A":    Key(float, "--lambda_ssim_A",   "λ1: SSIM loss of G1 = E[1 − SSIM(x, f↓(x_SR))]（JMI Table 1 = 1.0。SPIE 版は 2.0）"),
    "loss.lambda_ssim_B":    Key(float, "--lambda_ssim_B",   "λ2: SSIM loss of G2 = E[1 − SSIM(y, f↑(y_LR))]（JMI Table 1 = 1.0）"),
    "loss.lambda_down":      Key(float, "--lambda_down",     "λ3: downsample loss E‖x − f↓(x_SR)‖²（JMI Table 1 = 0.7。コードは 1.0）"),
    "loss.lambda_up":        Key(float, "--lambda_up",       "λ4: upsample loss E‖y − f↑(y_LR)‖²（JMI Table 1 = 0.3。コードは 1.0）"),
    "loss.lambda_identity":  Key(float, "--lambda_identity", "identity loss の重み。論文の総損失 Eq. 8 には無い → 0（ユーザー決定 2026-10-27）。コード 741924c は 0.2（idt_A = f↓(G1(y)) vs y、idt_B = f↑(G2(x)) vs x、重みは λ7 × この値）"),
    "loss.identity_b_scale": Key(float, "--identity_b_scale", "idt_B にさらに掛ける係数（コード 741924c の `* 0.1`）。lambda_identity = 0 なら使われない"),
    "loss.ssim_window":      Key(int,   "--ssim_window",     "SSIM の Gaussian window の一辺（コード pytorch_ssim = 11、σ 1.5）"),
    "loss.ssim_c1":          Key(float, "--ssim_c1",         "SSIM の C1（SPIE Sec. 3.2 \"C1 = 0.02\"。JMI は値未記載。コードは 0.01² = 0.0001）"),
    "loss.ssim_c2":          Key(float, "--ssim_c2",         "SSIM の C2（SPIE Sec. 3.2 \"C2 = 0.06\"。コードは 0.03² = 0.0009）"),
    "loss.pool_size":        Key(int,   "--pool_size",       "D 用の ImagePool（生成画像の履歴）の大きさ。論文未記載 → コード 50（CycleGAN 既定）。0 で無し"),
    # network
    "network.scale":         Key(int,   "--scale",           "SR 倍率（2 のべき）。論文は ×8（clinical 32 → µCT 256）、本研究は ×2（EID512 → PCD1024）。G1 の PixelShuffle 段数と G2 の downblock 段数 = log2(scale)、f↓ = AvgPool2d(scale)、f↑ = nearest ×scale"),
    "network.n_blocks":      Key(int,   "--n_blocks",        "G1 / G2 の ResBlock 数（JMI Sec. 4.2 \"9 ResBlocks\"、コード resnet_9blocks）"),
    "network.ngf":           Key(int,   "--ngf",             "G の基本チャネル数（コード 64）"),
    "network.ndf":           Key(int,   "--ndf",             "D 1 層目のチャネル数（コード 64）"),
    "network.norm":          Key(str,   "--norm",            "正規化層: batch（論文 Sec. 2.3.1 \"convolution layer ... batch normalization and activation\"。ユーザー決定 2026-10-27）| instance（コード 741924c の既定）"),
    "network.n_layers_D":    Key(int,   "--n_layers_D",      "D（PatchGAN）の層数。3 = 70×70（コード netD basic）"),
    "network.init_type":     Key(str,   "--init_type",       "重み初期化 normal | xavier | kaiming | orthogonal（コード normal）"),
    "network.init_gain":     Key(float, "--init_gain",       "初期化の標準偏差 / gain（コード 0.02）"),
    # data
    "data.split_file":       Key(str,   "--split_file",      "A 側（EID）の症例分割 YAML（train / val / test の症例名リスト。Stage 1 の bash start.sh split が作る configs/splits/eid_v5.yaml を流用）。リポジトリ直下からの相対パスか絶対パス。train = 学習、val = epoch 末のランダムプレビュー、test = 読まない（log.full_slice の 1 枚は例外）"),
    "data.b_exclude_cases":  Key(list,  "--b_exclude_cases", "B 側（PCD1024）から外す症例名のリスト。[] で全症例（Stage 1 task: sr と同じ決定）。ディスクに無い名前はエラー"),
    "data.patch_size":       Key(int,   "--patch_size",      "A（EID512）の patch 一辺。B（PCD1024）の patch は常に patch_size × scale（別々に書けると解剖スケールがずれるので書けない）。論文は 32 / 256（×8）、本研究 128 / 256（ユーザー決定 2026-10-27）"),
    "data.samples_per_epoch": Key(int,  "--samples_per_epoch", "1 epoch のサンプル数。論文は 2000 patch × 5 症例 = 10,000/ドメイン"),
    "data.patches_per_slice": Key(int,  "--patches_per_slice", "1 回読んだスライス（A / B それぞれ）から連続して切る patch 数（Stage 1 と同じ仕組み。位置は patch ごとに独立）。PCD1024 46 GB + EID512 49 GB は RAM に収まらないので 8"),
    "data.patch_seed":       Key(int,   "--patch_seed",      "fixed_batch / full_slices のランダムと serial_batches のときの決定的サンプリング用 seed"),
    "data.serial_batches":   Key(bool,  "--serial_batches",  "true で sampler を index 決定的に（再現用）"),
    "data.hu_offset":        Key(int,   "--hu_offset",       "stored = HU + hu_offset（1400）。Stage 1 と同じ"),
    "data.hu_min":           Key(int,   "--hu_min",          "正規化窓の下限 HU（−1400）→ −1。論文はモダリティごとの min-max だが本研究は HU 窓（HU が読める。Tanh の値域と一致）"),
    "data.hu_max":           Key(int,   "--hu_max",          "正規化窓の上限 HU（4096）→ +1"),
    # log
    "log.print_freq":        Key(int,   "--print_freq",      "バー末尾の損失・TensorBoard scalar・loss_log.txt の更新間隔（step = batch_size 枚）"),
    "log.image_freq":        Key(int,   "--image_freq",      "TensorBoard に学習中の patch グリッド（images/current, fixed, cycle_A, cycle_B）を出す間隔（step）"),
    "log.n_images":          Key(int,   "--n_images",        "patch グリッドの行数"),
    "log.full_slice":        Key(str,   "--full_slice",      "epoch 末にフル 512 → 1024 で書き出す固定スライス（eid_dir からの相対パス。test の症例 = 目視の holdout。EID-049/EID-049-079.png）"),
    "log.pcd_slice":         Key(str,   "--pcd_slice",       "パネル左端に並べる B 側（PCD1024）の代表スライス（pcd1024_dir からの相対パス。固定。PCD-002/PCD-002-236.png）"),
    "log.n_full_random":     Key(int,   "--n_full_random",   "epoch 末に書き出すランダムスライスの枚数（val の症例だけから。epoch ごとに別。0 で無し）"),
    "log.display_hu_min":    Key(int,   "--display_hu_min",  "表示の線形範囲の下限 HU（Stage 1 と同じ −1400）"),
    "log.display_hu_max":    Key(int,   "--display_hu_max",  "表示の線形範囲の上限 HU（Stage 1 と同じ 2100）"),
    "log.preview_bits":      Key(int,   "--preview_bits",    "output_images/preview_*/ のビット深度 16 | 8"),
    "log.diff_range_hu":     Key(int,   "--diff_range_hu",   "R = G1(z) − f↑(z) のカラー表示の ±範囲 HU（白 = 0、純青 = −、純赤 = +。util/residual_color.py）"),
    "log.tb_full_size":      Key(int,   "--tb_full_size",    "TensorBoard の images/full/* に出すパネルの列の一辺（1024 は大きいので 512 に縮小。ディスクの preview は縮小しない）"),
    "log.save_epoch_freq":   Key(int,   "--save_epoch_freq", "checkpoint（weights/epoch_NNN/）とフル画像を保存する epoch 間隔"),
    "log.save_latest_freq":  Key(int,   "--save_latest_freq", "latest/ を保存する間隔（画像枚数。batch_size の倍数）"),
}

# ---------------------------------------------------------------------------
# configs/machines.yaml（リポジトリ直上、Stage 共通）のうち、この ablation が使うキー。選択したマシンのエントリに対して検証（他のキーは無視）
# ---------------------------------------------------------------------------
MACHINE = {
    "gpu_gen":                 Key(int, None,                        "GPU 世代 30 | 40 | 50 | 0(CPU)。compose の選択（start_ablation.sh）と学習デバイスに使う"),
    "host_data_root":          Key(str, None,                        "Docker 起動時にマウントするホスト側のデータルート（start_ablation.sh が使う）"),
    "container_data_root":     Key(str, None,                        "マウント先（コンテナ内）"),
    "num_threads":             Key(int, "--num_threads",             "DataLoader の worker 数"),
    "eid_dir":                 Key(str, "--eid_dir",                 "A = 実 EID512（ImageCAS、EID_v5。Stage 1 の x 側と同じディレクトリ）。<case>/<slice>.png、1ch uint16"),
    "pcd1024_dir":             Key(str, "--pcd1024_dir",             "B = PCD1024（PCD512_v2 と同一スキャンの 1024 再構成。Stage 2 の教師と同じディレクトリ）"),
    "ablation_checkpoints_dir": Key(str, "--ablation_checkpoints_dir", "アブレーションの run の保存先ルート。この下に <ablation 名>/<run>/（util/run_paths.py）。2026-10-27"),
    "ablation_tb_port":        Key(int, None,                        "アブレーション用の TensorBoard のポート（Stage 1 の tb_port、Stage 2 の stage2_tb_port とは別。compose が 3 本とも公開する）。2026-10-27"),
}

# ---------------------------------------------------------------------------
# configs/infer.yaml — 推論の方式（bash start_ablation.sh <名前> infer → run_infer.py → inference_dir.py。フラグは inference_dir.py のもの）
# 重みディレクトリ・入力フォルダは infer.sh の変数、デバイスは machines.yaml の gpu_gen（0 → cpu）。G1 の構成と HU 正規化は run の launch.yaml、入力の一辺は run の dataset_info.yaml
# ---------------------------------------------------------------------------
INFER = {
    "max_slices":       Key(int,  "--max_slices",    "推論するスライス数の上限。0 で全部"),
    "batch_slices":     Key(int,  "--batch_slices",  "同時に G1 に通すスライス数（512 → 1024 のフル。G1 は 512² を 64 ch × 9 ResBlock で通すので 1 枚で数 GB。3070 8GB は 1〜2）"),
    "save_residual":    Key(bool, "--save_residual", "true で R = G1(z) − f↑(z)（0 HU = 32768、1024）と表示用カラーも full_R/ full_R_color/ に保存"),
    "save_panel":       Key(bool, "--save_panel",    "true で表示用パネル [PCD-like1024 | EID↑2 | R + ゲージ]（学習の preview と同じ正規化表示）も full_panel/ に保存"),
    "io.read_workers":  Key(int,  "--read_workers",  "PNG の読み込み・デコードを先読みするスレッド数（0 = 直列）"),
    "io.write_workers": Key(int,  "--write_workers", "PNG の書き込みを非同期にするスレッド数（0 = 直列）"),
}

GAN_MODES = ("vanilla", "lsgan")
CYCLE_NORMS = ("l1", "l2")
NORMS = ("batch", "instance")
INIT_TYPES = ("normal", "xavier", "kaiming", "orthogonal")
GPU_GENS = (0, 30, 40, 50)
SPLIT_KEYS = ("train", "val", "test")


# ---------------------------------------------------------------------------
# 検証・変換（stage2/configs/schema.py から複製）
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
        return isinstance(value, list) and all(isinstance(x, str) and x for x in value)
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


def to_argv(flat, schema):
    """検証済みの平坦 dict を inference_dir.py の引数列にする。bool は true のときだけフラグを付ける。list はカンマ区切り 1 語（空は空文字）。"""
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
        raise ConfigError(f"{flag} の値を {typ.__name__} に変換できません: {val!r}" + ("（list はカンマ区切り。空要素は不可、空リストは \"\"）" if typ is list else ""))
    raise TypeError(typ)


def apply_overrides(tokens, sections):
    """sh からの上書き列を平坦 dict に反映する（後勝ち）。
    sections: {"train": (flat_dict, TRAIN), "machine": (flat_dict, MACHINE), ...}。flat_dict はその場で書き換わる。
    受ける形: --flag value / --flag=value。bool は --flag（= true）または --flag true|false。list は --flag a,b,c（空は --flag ""）。
    schema に無いフラグ・値の無いフラグ・型違いは ConfigError。戻り値は適用した {flag: 値}（記録用）。"""
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
    if not (1 <= machine["ablation_tb_port"] <= 65535): P.append("ablation_tb_port は 1..65535")


def check_train_values(train, machine):
    P = []
    bs = train["optim.batch_size"]
    if train["optim.lr"] <= 0: P.append("optim.lr は正")
    if not (0.0 <= train["optim.beta1"] < 1.0 and 0.0 <= train["optim.beta2"] < 1.0): P.append("optim.beta1 / beta2 は [0, 1)")
    if bs < 1: P.append("optim.batch_size ≥ 1")
    if train["optim.n_epochs"] < 1: P.append("optim.n_epochs ≥ 1")
    if train["optim.n_epochs_decay"] < 0: P.append("optim.n_epochs_decay ≥ 0")
    if train["optim.seed"] < 0: P.append("optim.seed ≥ 0")
    if train["loss.gan_mode"] not in GAN_MODES: P.append(f"loss.gan_mode は {GAN_MODES}")
    if train["loss.cycle_norm"] not in CYCLE_NORMS: P.append(f"loss.cycle_norm は {CYCLE_NORMS}")
    for k in ("lambda_gan_A", "lambda_gan_B", "lambda_cycle", "lambda_ssim_A", "lambda_ssim_B", "lambda_down", "lambda_up", "lambda_identity", "identity_b_scale"):
        if train[f"loss.{k}"] < 0: P.append(f"loss.{k} ≥ 0")
    if train["loss.ssim_window"] < 3 or train["loss.ssim_window"] % 2 == 0: P.append("loss.ssim_window は 3 以上の奇数")
    if train["loss.ssim_c1"] <= 0 or train["loss.ssim_c2"] <= 0: P.append("loss.ssim_c1 / c2 は正")
    if train["loss.pool_size"] < 0: P.append("loss.pool_size ≥ 0")
    sc = train["network.scale"]
    if sc < 2 or (sc & (sc - 1)): P.append("network.scale は 2 以上の 2 のべき（PixelShuffle(2) の段数 = log2(scale)）")
    if train["network.n_blocks"] < 0: P.append("network.n_blocks ≥ 0")
    if train["network.ngf"] < 1 or train["network.ndf"] < 1: P.append("network.ngf / ndf ≥ 1")
    if train["network.norm"] not in NORMS: P.append(f"network.norm は {NORMS}")
    if train["network.n_layers_D"] < 1: P.append("network.n_layers_D ≥ 1")
    if train["network.init_type"] not in INIT_TYPES: P.append(f"network.init_type は {INIT_TYPES}")
    if train["network.init_gain"] <= 0: P.append("network.init_gain は正")
    if not train["data.split_file"]: P.append("data.split_file を指定（Stage 1 の bash start.sh split で作る）")
    if len(set(train["data.b_exclude_cases"])) != len(train["data.b_exclude_cases"]): P.append("data.b_exclude_cases に重複があります")
    ps = train["data.patch_size"]
    if ps < 4 or ps % 4: P.append("data.patch_size は 4 の倍数（G2 の 2 段 downsampling + downblock）")
    spe = train["data.samples_per_epoch"]
    if spe < bs or spe % bs: P.append(f"data.samples_per_epoch ({spe}) は optim.batch_size ({bs}) の倍数で batch_size 以上")
    if train["data.patches_per_slice"] < 1: P.append("data.patches_per_slice ≥ 1")
    if train["data.hu_min"] >= train["data.hu_max"]: P.append("data.hu_min < data.hu_max")
    if train["log.print_freq"] < 1 or train["log.image_freq"] < 1: P.append("log.print_freq / image_freq ≥ 1")
    if train["log.n_images"] < 1: P.append("log.n_images ≥ 1")
    if not train["log.full_slice"]: P.append("log.full_slice を指定")
    if not train["log.pcd_slice"]: P.append("log.pcd_slice を指定")
    if train["log.n_full_random"] < 0: P.append("log.n_full_random ≥ 0")
    if train["log.display_hu_min"] >= train["log.display_hu_max"]: P.append("log.display_hu_min < display_hu_max")
    if train["log.preview_bits"] not in (8, 16): P.append("log.preview_bits は 8 | 16")
    if train["log.diff_range_hu"] <= 0: P.append("log.diff_range_hu は正")
    if train["log.tb_full_size"] < 64: P.append("log.tb_full_size ≥ 64")
    if train["log.save_epoch_freq"] < 1: P.append("log.save_epoch_freq ≥ 1")
    slf = train["log.save_latest_freq"]
    if slf < bs or slf % bs: P.append(f"log.save_latest_freq ({slf}) は optim.batch_size ({bs}) の倍数（画像枚数単位）")
    _check_machine(machine, P)
    _problems_to_error("train values", P)


def check_infer_values(infer, machine):
    P = []
    if infer["max_slices"] < 0: P.append("max_slices ≥ 0（0 で全部）")
    if infer["batch_slices"] < 1: P.append("batch_slices ≥ 1")
    if infer["io.read_workers"] < 0 or infer["io.write_workers"] < 0: P.append("io.read_workers / write_workers ≥ 0（0 = 直列）")
    _check_machine(machine, P)
    _problems_to_error("infer values", P)


def check_split(split, where):
    """split_file の中身: train / val / test の文字列リスト（空リスト可）。重複・他のキー・型違いはエラー。戻り値 {kind: [case, ...]}（stage1/configs/schema.py と同じ規則）。"""
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
        if any(len(set(out[k])) != len(out[k]) for k in SPLIT_KEYS):
            P.append("同じリストの中に重複があります")
        if not out["train"]:
            P.append("train が空です")
    _problems_to_error(where, P)
    return out
