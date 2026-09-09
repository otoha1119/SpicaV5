"""stage2/configs/schema.py — Stage 2 の設定ファイル（dataset.yaml / train.yaml / mode.yaml）と、Stage 2 が読む machines.yaml のキーの唯一の正。

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
    "log.full_slice":              Key(str, "--full_slice",              "epoch 末にフル 1024 で書き出す固定スライス（eidlike1024_dir / pcd1024_dir からの相対パス。症例は train / val / test のどれかに入っていること。test の PCD-002/PCD-002-236.png、ユーザー決定 2026-09-09。best は val の指標で選ぶ）→ output_images/epoch_NNN/ と preview_fixed_<slice>/"),
    "log.eid_slice":               Key(str, "--eid_slice",               "実 EID のテストスライス（eid_dir = EID_v5 からの相対パス。EID-049/EID-049-079.png）。毎 epoch 512 → ×scale 補間（eidlike1024_dir の manifest.yaml の方式）→ U-Net に通し、固定パネルの 4 列目に出す"),
    "log.preview_bits":            Key(int, "--preview_bits",            "output_images/preview_*/（表示用）のビット深度。16 = 表示範囲を 0..65535 に伸ばす（Stage 1 と同じ）| 8"),
    "log.n_full_random":           Key(int, "--n_full_random",           "epoch 末に書き出すランダムスライスの枚数（epoch ごとに別。val の症例だけから。0 で無し）"),
    "log.display_hu_min":          Key(int, "--display_hu_min",          "TensorBoard 画像の線形表示範囲の下限 HU（この値以下を黒）。Stage 1 と同じ −1400"),
    "log.display_hu_max":          Key(int, "--display_hu_max",          "同上の上限 HU（この値以上を白）。Stage 1 と同じ 2100（stored 3500）"),
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
}

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
    "stage2_checkpoints_dir": Key(str, "--stage2_checkpoints_dir", "Stage 2 の run（yyyy_mmdd_HHMM）の保存先ルート。レイアウトは stage2/util/run_paths.py"),
}

INTERPS = ("nearest", "bilinear", "bicubic", "area", "lanczos")
GPU_GENS = (0, 30, 40, 50)
ARCHS = ("unet_ilumenate",)
INIT_TYPES = ("xavier_uniform", "torch")
LOSSES = ("mse", "l1")
FINAL_ACTS = ("linear", "tanh")


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
    if not train["log.full_slice"]: P.append("log.full_slice（固定スライスの相対パス）を指定")
    if not train["log.eid_slice"]: P.append("log.eid_slice（実 EID テストスライスの相対パス）を指定")
    if train["log.preview_bits"] not in (8, 16): P.append("log.preview_bits は 8 | 16")
    if train["log.n_full_random"] < 0: P.append("log.n_full_random ≥ 0")
    if train["log.display_hu_min"] >= train["log.display_hu_max"]: P.append("log.display_hu_min < display_hu_max")
    if mode["arch"] not in ARCHS: P.append(f"arch は {ARCHS}")
    if mode["base_ch"] < 1: P.append("base_ch ≥ 1")
    if mode["n_pool"] < 1: P.append("n_pool ≥ 1")
    if mode["init_type"] not in INIT_TYPES: P.append(f"init_type は {INIT_TYPES}")
    if mode["loss"] not in LOSSES: P.append(f"loss は {LOSSES}")
    if mode["final_act"] not in FINAL_ACTS: P.append(f"final_act は {FINAL_ACTS}")
    _check_machine(machine, P)
    _problems_to_error("train values", P)
