"""stage1/configs/schema.py — 設定ファイル 4 つ（train.yaml / mode.yaml / infer.yaml / machines.yaml）の唯一の正。

設計: docs/plans/20260905_config-design-plan.md
原則: 既定値（フォールバック）は無い。ここに宣言したキーは**必ず**設定ファイルに存在しなければならず、
      未知のキー・型違いはエラー。run_train.py はここを見て train.py の引数列を組み立てる。

各キーの宣言: Key(type, flag, doc)
  type : int | float | str | bool。float は int の値も受け付ける（10 → 10.0）。bool は YAML の true / false のみ
  flag : train.py（junyanz 本家の argparse）へ渡すフラグ名。None は train.py に渡さない（Docker 起動などで使う）
         bool のキーは store_true フラグに対応し、true のときだけフラグを付ける
  doc  : 意味と論文（FE-GAN = Park et al. 2019）の出典
"""

from collections import namedtuple

Key = namedtuple("Key", "type flag doc")


# ---------------------------------------------------------------------------
# stage1/configs/train.yaml — 学習パラメータ（FE-GAN が使うものを全部）
# ---------------------------------------------------------------------------
TRAIN = {
    # optim
    "optim.lr":             Key(float, "--lr",             '論文 "learning rate of 0.0002"'),
    "optim.beta1":          Key(float, "--beta1",          "Adam β1。論文未記載 (Q-T1)。junyanz 慣行 0.5"),
    "optim.beta2":          Key(float, "--beta2",          "Adam β2。論文未記載。0.999"),
    "optim.batch_size":     Key(int,   "--batch_size",     '論文 "mini-batch size of 40"'),
    "optim.n_epochs":       Key(int,   "--n_epochs",       '論文 "300 epochs"'),
    "optim.n_epochs_decay": Key(int,   "--n_epochs_decay", "lr 線形減衰の epoch 数。論文未記載 (Q-T2) → 0"),
    "optim.lr_policy":      Key(str,   "--lr_policy",      "junyanz の scheduler 名。再開が対応するのは linear のみ（F-12）。decay 0 なら定常"),
    "optim.seed":           Key(int,   "--seed",           "python / numpy / torch の乱数 seed（F-15）。再現条件として launch.yaml に残る。cudnn.benchmark は本家のままなので完全決定ではない"),
    # loss
    "loss.lambda_fid":      Key(float, "--lambda_fid",     '論文 "We empirically choose λ = 10"。fidelity ‖G(z)−z‖² の重み'),
    # network
    "network.ngf":          Key(int,   "--ngf",            "G 最初の段のチャネル数。Fig. 3 = 32"),
    "network.ndf":          Key(int,   "--ndf",            "D 1 層目のチャネル数。Fig. 3 = 32"),
    "network.norm":         Key(str,   "--norm",           "正規化層 (batch | instance | none)。論文 = batch。wavelet / paper は BN 固定なので ablation 用 G/D にのみ効く"),
    "network.init_type":    Key(str,   "--init_type",      '重み初期化。論文 "Gaussian distribution with a mean of 0" → normal'),
    "network.init_gain":    Key(float, "--init_gain",      '初期化の標準偏差。論文 "standard deviation of 0.01"'),
    "network.input_nc":     Key(int,   "--input_nc",       "入力チャネル数。grayscale = 1"),
    "network.output_nc":    Key(int,   "--output_nc",      "出力チャネル数。grayscale = 1"),
    # data
    "data.patch_size":      Key(int,   "--patch_size",     '論文 "patches of size 128 × 128"。0 でフル画像（推論用）'),
    "data.patch_stride":    Key(int,   "--patch_stride",   "patch 左上座標の刻み。random は 1（任意位置）。sampling=paper のときは論文の 8 を明示する（暗黙置換はしない。F-19）"),
    "data.patches_per_image": Key(int, "--patches_per_image", 'sampling=paper の 1 画像あたり patch 数。論文 "40 patches"'),
    "data.patch_seed":      Key(int,   "--patch_seed",     "sampling=paper の固定集合、--serial_batches 時の再現用 seed"),
    "data.samples_per_epoch": Key(int, "--samples_per_epoch", "sampling=random の 1 epoch のサンプル数。論文の patch 集合 24,000 に合わせる"),
    "data.hu_offset":       Key(int,   "--hu_offset",      "stored = HU + hu_offset（1400）"),
    "data.hu_min":          Key(int,   "--hu_min",         "正規化窓の下限 HU（−1400）"),
    "data.hu_max":          Key(int,   "--hu_max",         "正規化窓の上限 HU（4096）"),
    # log（util/monitor.py）。単位に注意: *_freq のうち print_freq / image_freq は **step（iteration）**、save_latest_freq は junyanz 由来で **画像枚数**
    "log.print_freq":       Key(int,   "--print_freq",     "バー末尾の損失・TensorBoard scalar・loss_log.txt を更新する間隔（step）"),
    "log.image_freq":       Key(int,   "--image_freq",     "TensorBoard に途中画像を出す間隔（step）"),
    "log.n_images":         Key(int,   "--n_images",       "途中画像の枚数（current / fixed それぞれ、128 patch）"),
    "log.n_full_images":    Key(int,   "--n_full_images",  "checkpoint 保存時にフル 512 で書き出すスライス数（PCD / EID-like / R を output_images/epoch_NNN/ に uint16 PNG）"),
    "log.display_hu_min":   Key(int,   "--display_hu_min", "表示用の線形範囲の下限 HU（この値以下を黒）"),
    "log.display_hu_max":   Key(int,   "--display_hu_max", "表示用の線形範囲の上限 HU（この値以上を白）"),
    "log.diff_range_hu":    Key(int,   "--diff_range_hu",  "差分パネル (G(z) − z) の ±範囲 HU（0 HU を中間グレー）"),
    "log.save_epoch_freq":  Key(int,   "--save_epoch_freq", "checkpoint を保存する epoch 間隔"),
    "log.save_latest_freq": Key(int,   "--save_latest_freq", "latest を保存する間隔（画像枚数。junyanz の total_iters 単位）"),
    "log.continue_train":   Key(bool,  "--continue_train", "true で latest から再開（通常は run_train.py --resume が付ける）"),
    "log.epoch_count":      Key(int,   "--epoch_count",    "開始 epoch 番号（再開時に使う。通常 1）"),
}

# ---------------------------------------------------------------------------
# stage1/configs/mode.yaml — アルゴリズムのモード切替
# ---------------------------------------------------------------------------
MODE = {
    "sampling":       Key(str,  "--sampling",       "random（ケース 1 完全ランダム）| paper（論文の固定 40 patch/画像）| aligned（ケース 2、未実装）"),
    "gan_mode":       Key(str,  "--gan_mode",       "fgan_kl（論文 Eq.6）| lsgan | vanilla（ablation、論文の式ではない）"),
    "netG":           Key(str,  "--netG",           "wavelet（論文 Fig.3）| unet_128 | unet_256 | resnet_6blocks | resnet_9blocks（ablation）"),
    "netD":           Key(str,  "--netD",           "paper（論文 Fig.3、受容野 22）| basic（70×70 PatchGAN）| n_layers | pixel"),
    "final_norm_act": Key(bool, "--final_norm_act", "G 最終 conv に bnorm+LReLU を付ける（確定: false。ablation 用）"),
    "serial_batches": Key(bool, "--serial_batches", "true で sampler を index 決定的に（再現用）"),
}

# ---------------------------------------------------------------------------
# configs/machines.yaml（リポジトリ直上、Stage 共通）— 選択したマシンのエントリに対して検証
# ---------------------------------------------------------------------------
MACHINE = {
    "gpu_gen":             Key(int, None,                "GPU 世代 30 | 40 | 50 | 0(CPU)。Docker（compose）の選択に使う"),
    "gpu_name":            Key(str, None,                "記録用"),
    "host_data_root":      Key(str, None,                "Docker 起動時にマウントするホスト側のデータルート"),
    "container_data_root": Key(str, "--dataroot",        "マウント先（コンテナ内）。junyanz の必須引数 --dataroot にも渡す（本ローダは dir_A/dir_B を使うので参照用）"),
    "pcd_dir":             Key(str, "--dir_A",           "PCD（論文の z 側 = A）。プロセスから見えるパス。直下に症例フォルダ"),
    "eid_dir":             Key(str, "--dir_B",           "EID（論文の x 側 = B）。同上"),
    "align_meta":          Key(str, "--align_meta",      "ケース 2 用の症例メタ YAML/JSON。sampling=aligned のときだけ存在を検査。空文字可"),
    "checkpoints_dir":     Key(str, "--checkpoints_dir", "重み・ログの保存先ルート。run 名 yyyy_mmdd_HHMM がこの下に付く（レイアウトは util/run_paths.py）"),
    "num_threads":         Key(int, "--num_threads",     "DataLoader の worker 数"),
    "tb_port":             Key(int, None,                "TensorBoard のポート（ホスト側・コンテナ側とも同じ番号で公開）。start.sh が使う"),
}

# ---------------------------------------------------------------------------
# stage1/configs/infer.yaml — 推論の方式（run_infer.py → inference_dir.py。フラグは inference_dir.py のもの）
# 重みディレクトリ・入力フォルダ・出力形式（png | dicom | both）・元 DICOM ルートは infer_stage1.sh の変数、デバイスは machines.yaml の gpu_gen（0 → cpu）
# ---------------------------------------------------------------------------
INFER = {
    "mode":             Key(str,  "--mode",             "full（512 を一発）| patch（patch.* で切って重なり平均）| both（両方保存 + |full − patch| を diff_stats.txt に）"),
    "patch.size":       Key(int,  "--patch_size",       "推論 patch の一辺。論文の学習 patch と同じ 128。8 の倍数"),
    "patch.stride":     Key(int,  "--patch_stride",     "patch 位置の刻み。論文の学習 grid と同じ 8（512 で 2,401 patch/枚）。端まで均等配置で被覆（nulmil image_cut 方式）"),
    "patch.blend":      Key(str,  "--patch_blend",      "重なりの合成: uniform（単純平均）| hann（2D Hann 重み）"),
    "patch.batch_size": Key(int,  "--patch_batch_size", "patch を G に通す枚数（VRAM に合わせる）"),
    "max_slices":       Key(int,  "--max_slices",       "推論するスライス数の上限。0 で全部（動作確認用に絞る）"),
    "save_residual":    Key(bool, "--save_residual",    "true で残差 R（0 HU = 32768）も <mode>_R/ に保存"),
    # DICOM 出力（OUTPUT_FORMAT = dicom | both のとき）。stored = (HU − intercept) / slope。参照 DICOM のタグと一致しなければエラー
    "dicom.rescale_slope":     Key(float, "--rescale_slope",     "元 DICOM の RescaleSlope。PCD（Siemens NAEOTOM Alpha）= 1"),
    "dicom.rescale_intercept": Key(float, "--rescale_intercept", "元 DICOM の RescaleIntercept。PCD（Siemens NAEOTOM Alpha）= −8192（格納値は符号なし 16bit）"),
}

FILES = {"train": TRAIN, "mode": MODE, "machine": MACHINE, "infer": INFER}


# ---------------------------------------------------------------------------
# 検証・変換
# ---------------------------------------------------------------------------
class ConfigError(Exception):
    pass


def flatten(d, prefix=""):
    """入れ子 dict を 'a.b.c' キーの平坦 dict にする。"""
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
    raise TypeError(typ)


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
    return {k: (float(flat[k]) if schema[k].type is float else flat[k]) for k in schema}


def to_argv(flat, schema):
    """検証済みの平坦 dict を train.py の引数列にする。bool は true のときだけフラグを付ける。"""
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


def opt_attr_names(schemas=(TRAIN, MODE, MACHINE)):
    """train.py の opt に必ず入っていなければならない属性名（bool 以外）。番兵 None の検査に使う。"""
    names = []
    for schema in schemas:
        for key in schema.values():
            if key.flag is not None and key.type is not bool:
                names.append(key.flag.lstrip("-"))
    return names


# ---------------------------------------------------------------------------
# 上書き（sh の --flag）を平坦 dict に反映する（F-02）。argv に生で足さず、実効値を dict に持ってから argv を作る
# ---------------------------------------------------------------------------
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
    except ValueError:
        raise ConfigError(f"{flag} の値を {typ.__name__} に変換できません: {val!r}")
    raise TypeError(typ)


def apply_overrides(tokens, sections):
    """sh からの上書き列を平坦 dict に反映する（後勝ち）。
    sections: {"train": (flat_dict, TRAIN), "mode": (flat_dict, MODE), ...}。flat_dict はその場で書き換わる。
    受ける形: --flag value / --flag=value。bool は --flag（= true）または --flag true|false / --flag=false。
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
# 値域・相互条件の検証（F-07）。型検査（validate）の後、実効値（上書き反映後）に対して行う
# ---------------------------------------------------------------------------
LR_POLICIES = ("linear",)  # 再開（scheduler の作り直し）が linear 前提。他方式は scheduler 状態の保存・復元を実装するまで受け付けない（F-12）
SAMPLINGS = ("random", "paper", "aligned")
GAN_MODES = ("fgan_kl", "lsgan", "vanilla")
NETGS = ("wavelet", "unet_128", "unet_256", "resnet_6blocks", "resnet_9blocks")
NETDS = ("paper", "basic", "n_layers", "pixel")
NORMS = ("batch", "instance", "none")
INIT_TYPES = ("normal", "xavier", "kaiming", "orthogonal")
GPU_GENS = (0, 30, 40, 50)
INFER_MODES = ("full", "patch", "both")
BLENDS = ("uniform", "hann")
HAAR_LEVELS = 3  # wavelet G は 2**3 の倍数の入力が必要


def _problems_to_error(where, problems):
    if problems:
        raise ConfigError(f"[{where}] " + " / ".join(problems))


def check_values(train, mode, machine):
    """train / mode / machine の実効値の値域と相互条件。"""
    P = []
    bs = train["optim.batch_size"]
    if train["optim.lr"] <= 0: P.append("optim.lr は正")
    if not (0.0 <= train["optim.beta1"] < 1.0 and 0.0 <= train["optim.beta2"] < 1.0): P.append("optim.beta1 / beta2 は [0, 1)")
    if bs < 1: P.append("optim.batch_size ≥ 1")
    if train["optim.n_epochs"] < 1: P.append("optim.n_epochs ≥ 1")
    if train["optim.n_epochs_decay"] < 0: P.append("optim.n_epochs_decay ≥ 0")
    if train["optim.lr_policy"] not in LR_POLICIES: P.append(f"optim.lr_policy は {LR_POLICIES} のみ（他方式は再開時の scheduler 復元が未実装）")
    if train["optim.seed"] < 0: P.append("optim.seed ≥ 0")
    if train["loss.lambda_fid"] < 0: P.append("loss.lambda_fid ≥ 0")
    if train["network.ngf"] < 1 or train["network.ndf"] < 1: P.append("network.ngf / ndf ≥ 1")
    if train["network.norm"] not in NORMS: P.append(f"network.norm は {NORMS}")
    if train["network.init_type"] not in INIT_TYPES: P.append(f"network.init_type は {INIT_TYPES}")
    if train["network.init_gain"] <= 0: P.append("network.init_gain は正")
    if train["network.input_nc"] < 1 or train["network.output_nc"] < 1: P.append("network.input_nc / output_nc ≥ 1")
    ps = train["data.patch_size"]
    m = 2**HAAR_LEVELS
    if ps < m or ps % m: P.append(f"data.patch_size は {m} の倍数（学習は patch 必須）")
    if train["data.patch_stride"] < 1: P.append("data.patch_stride ≥ 1")
    if train["data.patches_per_image"] < 1: P.append("data.patches_per_image ≥ 1")
    spe = train["data.samples_per_epoch"]
    if spe < bs or spe % bs: P.append(f"data.samples_per_epoch ({spe}) は optim.batch_size ({bs}) の倍数で batch_size 以上")
    if train["data.hu_min"] >= train["data.hu_max"]: P.append("data.hu_min < data.hu_max")
    if train["log.print_freq"] < 1 or train["log.image_freq"] < 1: P.append("log.print_freq / image_freq ≥ 1（step）")
    if train["log.n_images"] < 1: P.append("log.n_images ≥ 1")
    if train["log.n_full_images"] < 1: P.append("log.n_full_images ≥ 1")
    if train["log.display_hu_min"] >= train["log.display_hu_max"]: P.append("log.display_hu_min < display_hu_max")
    if train["log.diff_range_hu"] <= 0: P.append("log.diff_range_hu は正")
    if train["log.save_epoch_freq"] < 1: P.append("log.save_epoch_freq ≥ 1")
    slf = train["log.save_latest_freq"]
    if slf < bs or slf % bs: P.append(f"log.save_latest_freq ({slf}) は optim.batch_size ({bs}) の倍数（画像枚数単位。倍数でないと latest が保存されない）")
    if train["log.epoch_count"] < 1: P.append("log.epoch_count ≥ 1")
    if mode["sampling"] not in SAMPLINGS: P.append(f"sampling は {SAMPLINGS}")
    if mode["sampling"] == "paper" and train["data.patch_stride"] != 8: P.append("sampling=paper のときは data.patch_stride を論文の 8 にする（暗黙の置換はしない。F-19）")
    if mode["gan_mode"] not in GAN_MODES: P.append(f"gan_mode は {GAN_MODES}")
    if mode["netG"] not in NETGS: P.append(f"netG は {NETGS}")
    if mode["netD"] not in NETDS: P.append(f"netD は {NETDS}")
    if machine["gpu_gen"] not in GPU_GENS: P.append(f"gpu_gen は {GPU_GENS}")
    if machine["num_threads"] < 0: P.append("num_threads ≥ 0")
    if not (1 <= machine["tb_port"] <= 65535): P.append("tb_port は 1..65535")
    _problems_to_error("values", P)


def check_infer_values(infer):
    P = []
    if infer["mode"] not in INFER_MODES: P.append(f"mode は {INFER_MODES}")
    ps, st = infer["patch.size"], infer["patch.stride"]
    m = 2**HAAR_LEVELS
    if ps < m or ps % m: P.append(f"patch.size は {m} の倍数")
    if not (1 <= st <= ps): P.append(f"patch.stride は 1..patch.size ({ps})。大きいと被覆されない画素ができる")
    if infer["patch.blend"] not in BLENDS: P.append(f"patch.blend は {BLENDS}")
    if infer["patch.batch_size"] < 1: P.append("patch.batch_size ≥ 1")
    if infer["max_slices"] < 0: P.append("max_slices ≥ 0（0 で全部）")
    if infer["dicom.rescale_slope"] == 0: P.append("dicom.rescale_slope ≠ 0")
    _problems_to_error("infer values", P)


def check_opt(opt, where):
    """junyanz の opt に None（番兵）が残っていればエラー。run_train.py を経由せず train.py を直接叩いた事故を止める。"""
    missing = [n for n in opt_attr_names() if getattr(opt, n, None) is None]
    if missing:
        raise ConfigError(f"[{where}] 次の引数が指定されていません（既定値は無い。run_train.py 経由で起動すること）: " + ", ".join("--" + m for m in missing))
