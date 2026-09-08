"""stage2/configs/schema.py — Stage 2 の設定ファイル（dataset.yaml。今後 train.yaml / infer.yaml を足す）と、Stage 2 が読む machines.yaml のキーの唯一の正。

原則は Stage 1（stage1/configs/schema.py）と同じ: 既定値（フォールバック）は無い。ここに宣言したキーは**必ず**設定ファイルに存在しなければならず、
未知のキー・型違いはエラー。sh からの上書き（--flag value）は schema にあるフラグだけ受け付け、実効値を manifest / launch に残す。

検証・上書きの関数（validate / apply_overrides / flatten）は stage1/configs/schema.py と同じものを 2026-09-08 に複製した
（Stage 間で import し合わない。両方の `configs` パッケージが名前衝突するため）。

machines.yaml は Stage 共通ファイルなので、Stage 2 は「自分が使うキー」（MACHINE）の存在と型だけ検査し、Stage 1 だけのキー
（pcd_dir / eid_dir / align_meta / checkpoints_dir ...）は無視する（validate_shared）。Stage 1 側は未知キーを拒むので、Stage 2 専用のキー
（pcd1024_dir / eidlike1024_dir）は stage1/configs/schema.py の MACHINE にも flag None で宣言してある。キーを足すときは両方に書く。

各キーの宣言: Key(type, flag, doc)
  type : int | float | str | bool。float は int の値も受け付ける。bool は YAML の true / false のみ
  flag : 上書き用のフラグ名（--flag value）。None は上書き不可
  doc  : 意味と出典
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
# configs/machines.yaml（リポジトリ直上、Stage 共通）のうち Stage 2 が使うキー。選択したマシンのエントリに対して検証（他のキーは無視）
# ---------------------------------------------------------------------------
MACHINE = {
    "gpu_gen":             Key(int, None, "GPU 世代 30 | 40 | 50 | 0(CPU)。compose の選択（start2.sh）と学習デバイスに使う"),
    "host_data_root":      Key(str, None, "Docker 起動時にマウントするホスト側のデータルート（start2.sh が使う）"),
    "container_data_root": Key(str, None, "マウント先（コンテナ内）。dataset の出力先はこの配下に限る"),
    "num_threads":         Key(int, None, "make_dataset.py の worker 数（0 で逐次）。学習では DataLoader の worker 数"),
    "tb_port":             Key(int, None, "TensorBoard のポート（start2.sh が compose に渡す。Stage 1 と共用）"),
    "pcd1024_dir":         Key(str, "--pcd1024_dir",     "Stage 2 の教師 PCD1024（<case>/<slice>.png、1ch uint16、512 と同じ命名）。学習で使う"),
    "eidlike1024_dir":     Key(str, "--eidlike1024_dir", "Stage 2 の学習入力。bash start2.sh dataset の出力先（EID-like512 を ×2 補間）で、学習はここを読む。container_data_root 配下"),
}

INTERPS = ("nearest", "bilinear", "bicubic", "area", "lanczos")
GPU_GENS = (0, 30, 40, 50)


# ---------------------------------------------------------------------------
# 検証・変換（stage1/configs/schema.py と同じ）
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
    return {k: (float(flat[k]) if schema[k].type is float else flat[k]) for k in schema}


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
    sections: {"dataset": (flat_dict, DATASET), ...}。flat_dict はその場で書き換わる。
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
# 値域の検証。型検査（validate）の後、実効値（上書き反映後）に対して行う
# ---------------------------------------------------------------------------
def _problems_to_error(where, problems):
    if problems:
        raise ConfigError(f"[{where}] " + " / ".join(problems))


def check_dataset_values(dataset, machine):
    P = []
    if dataset["scale"] < 2: P.append("scale ≥ 2（1 はコピーになるので作らない）")
    if dataset["interp"] not in INTERPS: P.append(f"interp は {INTERPS}")
    if dataset["input_size"] < 1: P.append("input_size ≥ 1")
    if machine["gpu_gen"] not in GPU_GENS: P.append(f"gpu_gen は {GPU_GENS}")
    if machine["num_threads"] < 0: P.append("num_threads ≥ 0")
    if not (1 <= machine["tb_port"] <= 65535): P.append("tb_port は 1..65535")
    _problems_to_error("dataset values", P)
