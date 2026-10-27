"""ablation/CycleGAN/configs/schema.py — この ablation の設定ファイル（train.yaml / infer.yaml）と、読む machines.yaml のキーの唯一の正。

原則は Stage 1 / Stage 2 と同じ: 既定値（フォールバック）は無い。ここに宣言したキーは**必ず**設定ファイルに存在しなければならず、未知のキー・型違いはエラー。
sh からの上書き（--flag value）は schema にあるフラグだけ受け付け、実効値を launch.yaml に残す。
検証・上書きの関数（validate / validate_shared / apply_overrides / flatten）は stage2/configs/schema.py から 2026-10-27 に複製した（ablation は Stage 1 / 2 から import しない）。

machines.yaml は Stage 共通ファイルなので、ここは「自分が使うキー」（MACHINE）の存在と型だけ検査し、他のキーは無視する（validate_shared）。
Stage 1 / Stage 2 の schema は未知キーを拒むので、この ablation 専用のキー（ablation_checkpoints_dir / ablation_tb_port）は
stage1/configs/schema.py と stage2/configs/schema.py の MACHINE にも flag None で宣言してある。キーを足すときは 3 箇所に書く。

各キーの宣言: Key(type, flag, doc)
  type : int | float | str | bool | list（str のリスト）。float は int の値も受け付ける。bool は YAML の true / false のみ
  flag : 上書き用のフラグ名（--flag value。bool は --flag true|false、list は --flag a,b,c、空リストは --flag ""）。None は上書き不可
  doc  : 意味と出典。論文 = Zhu, Park, Isola, Efros, "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks," ICCV 2017（CycleGAN）。
         コード = junyanz/pytorch-CycleGAN-and-pix2pix 本家（stage1/ に vendoring 済みの commit 2a7afba）の cycle_gan_model.py の既定値
"""

from collections import namedtuple

Key = namedtuple("Key", "type flag doc")


# ---------------------------------------------------------------------------
# configs/train.yaml — 学習パラメータ（論文 Sec. 3.3 / Table 1 とコード 741924c の値を全部ここに）
# ---------------------------------------------------------------------------
TRAIN = {
    # optim
    "optim.lr":              Key(float, "--lr",              '本家 0.0002 = 論文 Sec. 4 "learning rate of 0.0002 ... the same learning rate for the first 100 epochs and linearly decay the rate to zero over the next 100 epochs"'),
    "optim.beta1":           Key(float, "--beta1",           "Adam β1。本家 0.5（論文 Sec. 4）"),
    "optim.beta2":           Key(float, "--beta2",           "Adam β2。本家 0.999"),
    "optim.batch_size":      Key(int,   "--batch_size",      "本家既定 1（論文も batch 1）。InstanceNorm なので batch に依らない。4 で速度を取る（ユーザー判断）"),
    "optim.n_epochs":        Key(int,   "--n_epochs",        "lr 一定の epoch 数。本家 100"),
    "optim.n_epochs_decay":  Key(int,   "--n_epochs_decay",  "lr を 0 へ線形減衰させる epoch 数。本家 100。合計 200 epoch"),
    "optim.seed":            Key(int,   "--seed",            "python / numpy / torch の乱数 seed（再現条件）。cudnn.benchmark は有効なので完全決定ではない"),
    # loss（CycleGAN 論文 Eq. 3: L_GAN(G, D_Y) + L_GAN(F, D_X) + λ L_cyc(G, F)。+ 本家の identity loss）
    "loss.gan_mode":         Key(str,   "--gan_mode",        "lsgan（本家既定 = 論文 Sec. 4 \"we replace the negative log likelihood objective by a least-squares loss\"）| vanilla（log 形）"),
    "loss.lambda_A":         Key(float, "--lambda_A",        "forward cycle ‖F(G(x)) − x‖₁ の重み（本家 10.0 = 論文 λ = 10）"),
    "loss.lambda_B":         Key(float, "--lambda_B",        "backward cycle ‖G(F(y)) − y‖₁ の重み（本家 10.0）"),
    "loss.lambda_identity":  Key(float, "--lambda_identity", "identity loss の重み（‖G(y) − y‖₁ × λ_B × これ + ‖F(x) − x‖₁ × λ_A × これ）。本家既定 0.5（論文 Sec. 5.2 の painting→photo で 0.5λ。一般目的関数 Eq. 3 には無い）。0 で無し"),
    "loss.pool_size":        Key(int,   "--pool_size",       "D 用の ImagePool（生成画像の履歴）の大きさ。本家 50（論文 Sec. 4 \"keep an image buffer that stores the 50 previously created images\"）。0 で無し"),
    # network
    "network.netG":          Key(str,   "--netG",            "G_A / G_B の構造: resnet_9blocks（本家既定、論文 256×256 用）| resnet_6blocks"),
    "network.ngf":           Key(int,   "--ngf",             "G の基本チャネル数（本家 64）"),
    "network.ndf":           Key(int,   "--ndf",             "D 1 層目のチャネル数（本家 64）"),
    "network.norm":          Key(str,   "--norm",            "正規化層: instance（本家既定 = 論文）| batch"),
    "network.n_layers_D":    Key(int,   "--n_layers_D",      "D（PatchGAN）の層数。3 = 70×70（本家 basic）"),
    "network.init_type":     Key(str,   "--init_type",       "重み初期化 normal | xavier | kaiming | orthogonal（本家 normal）"),
    "network.init_gain":     Key(float, "--init_gain",       "初期化の標準偏差 / gain（本家 0.02）"),
    # data
    "data.split_file":       Key(str,   "--split_file",      "A 側（EID1024）の症例分割 YAML（train / val / test。Stage 1 の bash start.sh split が作る configs/splits/eid_v5.yaml を流用）。train = 学習、val = epoch 末のランダムプレビュー、test = 読まない（log.full_slice の 1 枚は例外）"),
    "data.b_exclude_cases":  Key(list,  "--b_exclude_cases", "B 側（PCD1024）から外す症例名のリスト。[] で全症例。ディスクに無い名前はエラー"),
    "data.patch_size":       Key(int,   "--patch_size",      "A / B 共通の patch 一辺（同解像度）。1024 で 256（Stage 1 task: sr と同じ解剖スケール。本家の crop_size 256 とも一致）。4 の倍数"),
    "data.samples_per_epoch": Key(int,  "--samples_per_epoch", "1 epoch のサンプル数（本家は小さい方のデータセットの枚数。ここでは SRCycleGAN と揃えて 10,000）"),
    "data.patches_per_slice": Key(int,  "--patches_per_slice", "1 回読んだスライス（A / B それぞれ）から連続して切る patch 数（Stage 1 と同じ仕組み。位置は patch ごとに独立）。EID1024 130 GB + PCD1024 46 GB は RAM に収まらないので 8"),
    "data.patch_seed":       Key(int,   "--patch_seed",      "fixed_batch / full_slices のランダムと serial_batches のときの決定的サンプリング用 seed"),
    "data.serial_batches":   Key(bool,  "--serial_batches",  "true で sampler を index 決定的に（再現用）"),
    "data.hu_offset":        Key(int,   "--hu_offset",       "stored = HU + hu_offset（1400）。Stage 1 と同じ"),
    "data.hu_min":           Key(int,   "--hu_min",          "正規化窓の下限 HU（−1400）→ −1（G の Tanh と一致）"),
    "data.hu_max":           Key(int,   "--hu_max",          "正規化窓の上限 HU（4096）→ +1"),
    # log
    "log.print_freq":        Key(int,   "--print_freq",      "バー末尾の損失・TensorBoard scalar・loss_log.txt の更新間隔（step = batch_size 枚）"),
    "log.image_freq":        Key(int,   "--image_freq",      "TensorBoard に学習中の patch グリッド（images/current, fixed, cycle_A, cycle_B）を出す間隔（step）"),
    "log.n_images":          Key(int,   "--n_images",        "patch グリッドの行数"),
    "log.full_slice":        Key(str,   "--full_slice",      "epoch 末にフル 1024 で書き出す固定スライス（eid1024_dir からの相対パス。test の症例 = 目視の holdout。EID-049/EID-049-079.png）"),
    "log.pcd_slice":         Key(str,   "--pcd_slice",       "パネル左端に並べる B 側（PCD1024）の代表スライス（pcd1024_dir からの相対パス。固定。PCD-002/PCD-002-236.png）"),
    "log.n_full_random":     Key(int,   "--n_full_random",   "epoch 末に書き出すランダムスライスの枚数（val の症例だけから。epoch ごとに別。0 で無し）"),
    "log.display_hu_min":    Key(int,   "--display_hu_min",  "表示の線形範囲の下限 HU（Stage 1 と同じ −1400）"),
    "log.display_hu_max":    Key(int,   "--display_hu_max",  "表示の線形範囲の上限 HU（Stage 1 と同じ 2100）"),
    "log.preview_bits":      Key(int,   "--preview_bits",    "output_images/preview_*/ のビット深度 16 | 8"),
    "log.diff_range_hu":     Key(int,   "--diff_range_hu",   "R = G_A(z) − z のカラー表示の ±範囲 HU（白 = 0、純青 = −、純赤 = +。util/residual_color.py）"),
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
    "eid1024_dir":             Key(str, "--eid1024_dir",             "A = 実 EID を ×2 補間した EID1024（EID_v5 を bash start2.sh dataset --input_dir <eid_dir> --eidlike1024_dir <ここ> で作ったもの。manifest.yaml 付き。Stage 1 task: sr の A と同じ）。<case>/<slice>.png、1ch uint16"),
    "pcd1024_dir":             Key(str, "--pcd1024_dir",             "B = PCD1024（PCD512_v2 と同一スキャンの 1024 再構成。Stage 2 の教師と同じディレクトリ）。A と同じ大きさ"),
    "ablation_checkpoints_dir": Key(str, "--ablation_checkpoints_dir", "アブレーションの run の保存先ルート。この下に <ablation 名>/<run>/（util/run_paths.py）。2026-10-27"),
    "ablation_tb_port":        Key(int, None,                        "アブレーション用の TensorBoard のポート（Stage 1 の tb_port、Stage 2 の stage2_tb_port とは別。compose が 3 本とも公開する）。2026-10-27"),
}

# ---------------------------------------------------------------------------
# configs/infer.yaml — 推論の方式（bash start_ablation.sh <名前> infer → run_infer.py → inference_dir.py。フラグは inference_dir.py のもの）
# 重みディレクトリ・入力フォルダは infer.sh の変数、デバイスは machines.yaml の gpu_gen（0 → cpu）。G_A の構成と HU 正規化は run の launch.yaml、入力の一辺と 512 入力の補間方式（eid1024_dir/manifest.yaml）は launch の dataset / upsample
# ---------------------------------------------------------------------------
INFER = {
    "max_slices":       Key(int,  "--max_slices",    "推論するスライス数の上限。0 で全部"),
    "batch_slices":     Key(int,  "--batch_slices",  "同時に G_A に通すスライス数（1024 のフル。resnet_9blocks は 2 段 downsampling 後の 256² × 256 ch なので 1 枚で数 GB。3070 8GB は 1〜2）"),
    "save_residual":    Key(bool, "--save_residual", "true で R = G_A(z) − z（0 HU = 32768、1024）と表示用カラーも full_R/ full_R_color/ に保存"),
    "save_panel":       Key(bool, "--save_panel",    "true で表示用パネル [PCD-like1024 | EID1024 | R + ゲージ]（学習の preview と同じ正規化表示）も full_panel/ に保存"),
    "io.read_workers":  Key(int,  "--read_workers",  "PNG の読み込み・デコードを先読みするスレッド数（0 = 直列）"),
    "io.write_workers": Key(int,  "--write_workers", "PNG の書き込みを非同期にするスレッド数（0 = 直列）"),
}

GAN_MODES = ("lsgan", "vanilla")
NETGS = ("resnet_9blocks", "resnet_6blocks")
NORMS = ("instance", "batch")
INTERPS = ("nearest", "bilinear", "bicubic", "area", "lanczos")
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
    for k in ("lambda_A", "lambda_B", "lambda_identity"):
        if train[f"loss.{k}"] < 0: P.append(f"loss.{k} ≥ 0")
    if train["loss.pool_size"] < 0: P.append("loss.pool_size ≥ 0")
    if train["network.netG"] not in NETGS: P.append(f"network.netG は {NETGS}")
    if train["network.ngf"] < 1 or train["network.ndf"] < 1: P.append("network.ngf / ndf ≥ 1")
    if train["network.norm"] not in NORMS: P.append(f"network.norm は {NORMS}")
    if train["network.n_layers_D"] < 1: P.append("network.n_layers_D ≥ 1")
    if train["network.init_type"] not in INIT_TYPES: P.append(f"network.init_type は {INIT_TYPES}")
    if train["network.init_gain"] <= 0: P.append("network.init_gain は正")
    if not train["data.split_file"]: P.append("data.split_file を指定（Stage 1 の bash start.sh split で作る）")
    if len(set(train["data.b_exclude_cases"])) != len(train["data.b_exclude_cases"]): P.append("data.b_exclude_cases に重複があります")
    ps = train["data.patch_size"]
    if ps < 4 or ps % 4: P.append("data.patch_size は 4 の倍数（resnet G の 2 段 downsampling）")
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
