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
"""

from collections import namedtuple

Key = namedtuple("Key", "type flag doc")


# ---------------------------------------------------------------------------
# configs/train.yaml
# ---------------------------------------------------------------------------
TRAIN = {
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
    # degrade = 物理コア P（学習なし。R-S23: σ・鮮鋭度はスライスごとに範囲から抽選。範囲は R-S22 の実測（bash start1_v3.sh <machine> stats）を含むまで広げる）
    "degrade.enabled":       Key(bool,  "--degrade",         "false で P を掛けない（u = x。従来の fidelity GAN + 恒等項の対照）"),
    "degrade.mode":          Key(str,   "--degrade_mode",    "gaussian（一様ぼかし + ノイズ。1 本目）| bilateral_blur（エッジ保存の平滑化 → 弱いぼかし → ノイズ。中程度のエッジを実 EID 並みに残す。2026-10-30）"),
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
    "io.read_workers":  Key(int,  "--read_workers",  "PNG の読み込みを先読みするスレッド数（0 = 直列）"),
    "io.write_workers": Key(int,  "--write_workers", "PNG の書き込みを非同期にするスレッド数（0 = 直列）"),
}

GAN_MODES = ("fgan_kl", "lsgan", "vanilla")
DEGRADE_MODES = ("gaussian", "bilateral_blur")
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
                         "degrade.bilateral_sigma_space_px_min": 0.0, "degrade.bilateral_sigma_space_px_max": 0.0}

# 重みの形と P の有無を決めるキー。再開で変えられない
SHAPE_KEYS = ("network.netG", "network.netD", "network.ngf", "network.ndf", "network.norm", "degrade.enabled", "degrade.mode")


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
    if d["enabled"] and d["mode"] == "bilateral_blur" and (d["bilateral_sigma_color_hu_max"] <= 0 or d["bilateral_sigma_space_px_max"] <= 0):
        P.append("degrade.mode bilateral_blur では bilateral_sigma_color_hu_max / sigma_space_px_max を正に")


def check_train_values(train, machine):
    P = []
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
    for k in ("lambda_fid", "lambda_id", "lambda_id_final"):
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
