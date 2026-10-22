"""stage3/configs/schema.py — Stage 3（条件付き DDIM: EID-like1024 → PCD-like1024）の設定ファイルと、Stage 3 が読む machines.yaml のキーの唯一の正。

原則は Stage 1 / Stage 2 と同じ: 既定値（フォールバック）は無い。ここに宣言したキーは**必ず**設定ファイルに存在しなければならず、
未知のキー・型違いはエラー。sh からの上書き（--flag value）は schema にあるフラグだけ受け付け、実効値を launch.yaml / infer.yaml に残す。
検証・上書きの関数（flatten / validate / validate_shared / apply_overrides / to_argv）は stage2/configs/schema.py から 2026-10-13 に複製した
（Stage 間で import しない）。Stage 3 で足した型: ints（int のリスト。本家 ch_mult / attn_resolutions 用。上書きは --flag 1,1,2）と rois（nps.yaml の ROI 表。上書き不可）。

設定ファイル（決定: docs/decisions/20261013_pivot-stage2-diffusion.md 決定 7）:
  ddim.yml   本家 ermongroup/ddim の configs/church.yml を写したもの（data / model / diffusion / training / optim。キー名も本家のまま）。runners/diffusion.py が
             本家と同じ argparse.Namespace（dict2namespace）にして models/diffusion.Model・functions.get_optimizer に渡す。本家からの変更は ddim.yml の冒頭と UPSTREAM.md
  train.yaml SpicaV5 側（seed、症例分割・HU 正規化・patch の切り方、監視の頻度と表示、学習中のサンプリング設定）
  infer.yaml 推論（タイル合成の stride、DDIM の step 数 S / eta / skip_type、seed、入力の種類・教師・パネル）
  nps.yaml   NPS 評価（ROI の表と大きさ、トレンド除去、ビン数）
machines.yaml は Stage 共通ファイルなので、Stage 3 は自分が使うキー（MACHINE）の存在と型だけ検査する。Stage 1 は未知キーを拒むので、
Stage 3 専用のキー（stage3_checkpoints_dir / stage3_tb_port）は stage1/configs/schema.py の MACHINE にも flag None で宣言してある。キーを足すときは両方に書く。

各キーの宣言: Key(type, flag, doc)
  type : int | float | str | bool | list（str のリスト）| ints（int のリスト）| rois。float は int の値も受け付ける。bool は YAML の true / false のみ
  flag : 上書き用のフラグ名（--flag value）。None は上書き不可
  doc  : 意味と出典（本家 = ermongroup/ddim configs/church.yml、論文 = Song et al., DDIM, ICLR 2021）
"""

from collections import namedtuple

Key = namedtuple("Key", "type flag doc")


class ints:  # noqa: N801  型の印（int のリスト）
    pass


class rois:  # noqa: N801  型の印（nps.yaml の ROI 表: [{slice: str, y: int, x: int}, ...]）
    pass


# ---------------------------------------------------------------------------
# stage3/configs/ddim.yml — 本家 configs/church.yml（キー名は本家のまま。本家にあって使わないキーは置かない）
# ---------------------------------------------------------------------------
DDIM = {
    # data
    "data.image_size":                Key(int,   "--image_size",       "本家 256。学習 patch の一辺 = models/diffusion.Model の resolution（forward の assert はこの大きさしか通さない。1024 はタイル推論）"),
    "data.channels":                  Key(int,   None,                 "本家 3（RGB）→ 1（CT）。サンプリングの x_T の channel 数"),
    "data.random_flip":               Key(bool,  "--random_flip",      "本家 true（左右反転）→ false（Stage 2 と同じく augmentation なし。true なら入力と教師を同じく反転）"),
    # model（models/diffusion.Model がそのまま読む）
    "model.type":                     Key(str,   None,                 "本家 simple（bayesian は logvar を学習する別型。使わない）"),
    "model.in_channels":              Key(int,   None,                 "本家 3 → 2（条件 EID-like 1ch + x_t 1ch の channel 連結。Fast-DDPM / WeatherDiffusion と同じ形）"),
    "model.out_ch":                   Key(int,   None,                 "本家 3 → 1（予測する ε）"),
    "model.ch":                       Key(int,   "--ch",               "本家 128。最初の段のチャネル数（GroupNorm 32 なので 32 の倍数）"),
    "model.ch_mult":                  Key(ints,  "--ch_mult",          "本家 [1, 1, 2, 2, 4, 4]（256 → 8 の 6 段）"),
    "model.num_res_blocks":           Key(int,   "--num_res_blocks",   "本家 2"),
    "model.attn_resolutions":         Key(ints,  "--attn_resolutions", "本家 [16]（16×16 の段に self-attention）"),
    "model.dropout":                  Key(float, "--dropout",          "本家 0.0"),
    "model.var_type":                 Key(str,   None,                 "本家 fixedsmall（DDPM サンプラの分散。DDIM η=0 では使わない。記録のため本家どおり持つ）"),
    "model.ema_rate":                 Key(float, "--ema_rate",         "本家 0.999"),
    "model.ema":                      Key(bool,  None,                 "本家 true。サンプリングは EMA の重みで行う（本家 sample() と同じ）"),
    "model.resamp_with_conv":         Key(bool,  None,                 "本家 true"),
    # diffusion
    "diffusion.beta_schedule":        Key(str,   None,                 "本家 linear"),
    "diffusion.beta_start":           Key(float, None,                 "本家 0.0001"),
    "diffusion.beta_end":             Key(float, None,                 "本家 0.02"),
    "diffusion.num_diffusion_timesteps": Key(int, None,                "本家 1000（T）"),
    # training
    "training.batch_size":            Key(int,   "--batch_size",       "本家 64 → 8（ユーザー決定 2026-10-13）"),
    "training.n_iters":               Key(int,   "--n_iters",          "学習を止める step 数（本家 church.yml は 5000000 だが本家 train() は n_epochs で止まり n_iters を見ない）。ここでは上限として使い、snapshot を見て止める"),
    "training.snapshot_freq":         Key(int,   "--snapshot_freq",    "本家 5000。latest/ を保存する間隔（step）。本家は step 1 でも保存する（同じにした）"),
    "training.validation_freq":       Key(int,   "--validation_freq",  "本家 2000（church.yml にあるが本家 train() は使わない）。ここでは固定の val patch で ε の MSE を測る間隔（step）"),
    # optim（functions.get_optimizer がそのまま読む）
    "optim.weight_decay":             Key(float, None,                 "本家 0.0"),
    "optim.optimizer":                Key(str,   None,                 "本家 Adam（β2 は本家コードで 0.999 固定）"),
    "optim.lr":                       Key(float, "--lr",               "本家 0.00002（一定。scheduler なし）"),
    "optim.beta1":                    Key(float, None,                 "本家 0.9"),
    "optim.amsgrad":                  Key(bool,  None,                 "本家 false"),
    "optim.eps":                      Key(float, None,                 "本家 0.00000001"),
    "optim.grad_clip":                Key(float, "--grad_clip",        "church.yml に無い（本家 train() は config.optim.grad_clip の AttributeError を try/except で握りつぶすので clip しない）→ 0.0 = clip しない。正なら clip_grad_norm_（本家 cifar10.yml は 1.0）"),
}

# ---------------------------------------------------------------------------
# stage3/configs/train.yaml — SpicaV5 側の学習設定
# ---------------------------------------------------------------------------
TRAIN = {
    "optim.seed":                 Key(int,   "--seed",              "python / numpy / torch の乱数 seed（本家 main.py --seed の既定 1234）。cudnn.benchmark は本家どおり有効なので完全決定ではない"),
    # data
    "data.patch_seed":            Key(int,   "--patch_seed",        "固定 patch（監視の val patch・val の ε 損失の t と雑音）の位置を決める seed"),
    "data.patches_per_slice":     Key(int,   "--patches_per_slice", "1 回読んだペア（入力 1024 + 教師 1024）から連続して切る patch 数（Stage 2 と同じ仕組み。位置は patch ごとに独立）"),
    "data.hu_offset":             Key(int,   "--hu_offset",         "stored = HU + hu_offset（1400）。Stage 1 / 2 と同じ"),
    "data.hu_min":                Key(int,   "--hu_min",            "正規化窓の下限 HU（−1400）→ −1。Stage 2 と同じ（本家の rescaled: true = [−1, 1] と同じ値域）"),
    "data.hu_max":                Key(int,   "--hu_max",            "正規化窓の上限 HU（4096）→ +1"),
    "data.train_cases":           Key(list,  "--train_cases",       "学習に使う症例（Stage 2 と同じ PCD-003〜016）。eidlike1024_dir と pcd1024_dir の両方に存在すること"),
    "data.val_cases":             Key(list,  "--val_cases",         "検証に使う症例（PCD-017, 018）。学習には使わない"),
    "data.test_cases":            Key(list,  "--test_cases",        "最終評価用に避けておく症例（PCD-001, 002, 019, 020）。学習・検証で読まない（例外は log.full_slice の 1 枚）"),
    # log
    "log.steps_per_epoch":        Key(int,   "--steps_per_epoch",   "1 epoch の step 数（ユーザー決定 2026-10-13: 10000）。データはランダムに patch を切るので「1 周」の意味は無く、バー 1 本と epoch 末のログの区切りだけ。学習の長さは ddim.yml training.n_iters（step）"),
    "log.print_freq":             Key(int,   "--print_freq",        "バー末尾の損失・TB scalar・loss_log.txt の更新間隔（step）"),
    "log.image_freq":             Key(int,   "--image_freq",        "固定 val patch を EMA の重みでサンプリングして TB images/val_patch に出す間隔（step）"),
    "log.n_images":               Key(int,   "--n_images",          "その grid の行数（各行 [EID-like | PCD-like (sample) | PCD1024]）"),
    "log.n_val_patches":          Key(int,   "--n_val_patches",     "val の ε 損失（validation_freq ごと）に使う固定 val patch の数。t は [0, T) に等間隔、雑音は patch_seed で固定"),
    "log.full_freq":              Key(int,   "--full_freq",         "固定スライス（full_slice）と実 EID（eid_slice）を 1024 のままタイル合成でサンプリングして保存する間隔（step）。重いので所要時間を見て調整"),
    "log.save_weights_freq":      Key(int,   "--save_weights_freq", "weights/iter_NNNNNNN/（model + ema + state の 1 組）を残す間隔（step）。1 組 ≈ 1.8 GB（本家 128ch 256 で約 1.1 億 param）"),
    "log.crop_freq":              Key(int,   "--crop_freq",         "固定領域（crop_pcd / crop_eid）を EMA の重みでサンプリングして TB images/crop/* に出す間隔（step）。2026-10-13 ユーザー指示（Stage 1 の crop.yaml の領域を学習中に見る）"),
    "log.crop_size":              Key(int,   "--crop_size",         "元の切り抜きの一辺（1024 の画素。Stage 1 crop.yaml の 72（512）= 144）。y / x と合わせて**中心**を決めるのにだけ使い、表示は中心を保った ddim.yml data.image_size（256）の窓全体。image_size 以下"),
    "log.crop_scale":             Key(int,   "--crop_scale",        "TB に出すとき窓を最近傍で何倍にするか（1 = 256 のまま）"),
    "log.crop_pcd":               Key(rois,  None,                  "PCD 症例の固定領域 [{slice: <case>/<slice>.png, y: 左上の行, x: 左上の列}, ...]（1024 の画素。eidlike1024_dir / pcd1024_dir 相対）。[EID-like | PCD-like | PCD1024] の 3 列で出す。[] で無し。学習・検証に使わない test の症例のスライスでも読む（目視用、full_slice と同じ扱い）"),
    "log.crop_eid":               Key(rois,  None,                  "実 EID の固定領域（slice は eid_dir = EID_v5 相対の 512。y / x は ×scale 補間後の 1024 の画素）。[EID1024 | PCD-like] の 2 列で出す。[] で無し"),
    "log.full_slice":             Key(str,   "--full_slice",        "固定スライス（eidlike1024_dir / pcd1024_dir からの相対パス。Stage 2 と同じ PCD-002/PCD-002-236.png = test の症例を目視用に）"),
    "log.eid_slice":              Key(str,   "--eid_slice",         "実 EID のテストスライス（eid_dir = EID_v5 からの相対パス。EID-049/EID-049-079.png）。512 を eidlike1024_dir/manifest.yaml の方式で補間してから通す"),
    "log.sample_steps":           Key(int,   "--sample_steps",      "学習中のサンプリングの DDIM step 数 S（本家 main.py --timesteps。決定 7: 50）"),
    "log.sample_eta":             Key(float, "--sample_eta",        "同 η（本家 --eta。0 = 決定的 DDIM）"),
    "log.skip_type":              Key(str,   "--skip_type",         "同 step 列（本家 --skip_type: uniform = 論文の linear | quad）"),
    "log.sample_seed":            Key(int,   "--sample_seed",       "同 x_T の seed（スライス名と組み合わせて決まる。どの checkpoint でも同じ x_T なので推移を比べられる）"),
    "log.tile_stride":            Key(int,   "--tile_stride",       "1024 のタイル合成の stride r（決定 7: 128 = 49 タイル）。(1024 − image_size) の約数"),
    "log.tile_batch":             Key(int,   "--tile_batch",        "タイルを何枚ずつ ε-net に通すか（VRAM と相談。学習と同じ GPU なので小さめ）"),
    "log.preview_bits":           Key(int,   "--preview_bits",      "output_images/preview_*/ のビット深度。16 | 8（Stage 2 と同じ）"),
    "log.tb_full_size":           Key(int,   "--tb_full_size",      "TB の images/full/* の一辺（1024 を縮小。ディスクは縮小しない）"),
    "log.display_hu_min":         Key(int,   "--display_hu_min",    "表示の線形範囲の下限 HU（Stage 2 と同じ −1400）"),
    "log.display_hu_max":         Key(int,   "--display_hu_max",    "表示の線形範囲の上限 HU（Stage 2 と同じ 2100）"),
}

# ---------------------------------------------------------------------------
# configs/machines.yaml（Stage 共通）のうち Stage 3 が使うキー
# ---------------------------------------------------------------------------
MACHINE = {
    "gpu_gen":                Key(int, None,                      "GPU 世代 30 | 40 | 50 | 0(CPU)。compose の選択（start3.sh）とデバイス"),
    "host_data_root":         Key(str, None,                      "Docker 起動時にマウントするホスト側のデータルート（start3.sh が使う）"),
    "container_data_root":    Key(str, None,                      "マウント先（コンテナ内）"),
    "num_threads":            Key(int, "--num_threads",           "DataLoader の worker 数（0 で逐次）"),
    "stage3_tb_port":         Key(int, None,                      "Stage 3 の TensorBoard のポート（Stage 1 の tb_port、Stage 2 の stage2_tb_port とは別）"),
    "pcd1024_dir":            Key(str, "--pcd1024_dir",           "教師 PCD1024（Stage 2 と同じ）"),
    "eid_dir":                Key(str, "--eid_dir",               "実 EID512（EID_v5）。学習では log.eid_slice の 1 枚だけ読む"),
    "eidlike1024_dir":        Key(str, "--eidlike1024_dir",       "学習入力 EID-like1024（bash start2.sh dataset の出力。Stage 2 と同じデータを使う）"),
    "stage3_checkpoints_dir": Key(str, "--stage3_checkpoints_dir", "Stage 3 の run（yyyy_mmdd_HHMM）の保存先ルート。レイアウトは stage3/util/run_paths.py"),
}

# ---------------------------------------------------------------------------
# stage3/configs/infer.yaml — 推論（bash start3.sh infer → run_infer.py → inference_dir.py。フラグは inference_dir.py のもの）
# ---------------------------------------------------------------------------
INFER = {
    "input":           Key(str,   "--input",           "eidlike（PCD 症例の EID-like1024 / 512。教師あり可）| eid（実 EID512 = EID_v5。教師なし）"),
    "cases":           Key(list,  "--cases",           "推論する症例フォルダ名（[] で INPUT_DIR の全症例）。無い症例を書くとエラー"),
    "max_slices":      Key(int,   "--max_slices",      "推論するスライス数の上限（0 で全部）。cases・slices_per_case で絞った後に効く"),
    "slices_per_case": Key(int,   "--slices_per_case", "症例ごとに等間隔に間引く枚数（0 で全部）。決定 7 の 1 本目の評価は val 2 症例 × 10 枚"),
    "teacher":         Key(bool,  "--teacher",         "true で pcd1024_dir の同名スライスを教師に rmse / ssim / psnr（+ 入力そのままの参照値）を metrics.txt に。実 EID は false"),
    "save_input1024":  Key(bool,  "--save_input1024",  "512 入力を補間したとき、その 1024 入力も full_input1024/ に残すか（NPS 評価で入力側にも使う）"),
    "save_panel":      Key(bool,  "--save_panel",      "表示用パネル（Stage 2 と同じ 5 列）を full_panel/ に"),
    "eid_slice":       Key(str,   "--eid_slice",       "input=eidlike のパネル 4・5 列目の実 EID（eid_dir 相対）。空文字で無し。save_panel のときだけ"),
    "pcd_slice":       Key(str,   "--pcd_slice",       "input=eid のパネル 1〜3 列目の参照 PCD 症例（eidlike1024_dir / pcd1024_dir 相対）。空文字で無し。save_panel のときだけ"),
    "use_ema":         Key(bool,  "--use_ema",         "true で EMA の重み（本家 sample() と同じ）。false で学習中の生の重み"),
    "sample_steps":    Key(int,   "--sample_steps",    "DDIM の step 数 S（本家 --timesteps。決定 7: 既定 50、評価で 10 / 20 / 50 / 100 を振る）"),
    "eta":             Key(float, "--eta",             "DDIM の η（本家 --eta。0 = 決定的）"),
    "skip_type":       Key(str,   "--skip_type",       "step 列（本家 --skip_type: uniform | quad）"),
    "seed":            Key(int,   "--seed",            "x_T の seed（スライスの相対パスと組み合わせてスライスごとに別の x_T。同じ seed なら同じ結果）"),
    "tile_stride":     Key(int,   "--tile_stride",     "タイル合成の stride r（決定 7: 128）"),
    "tile_batch":      Key(int,   "--tile_batch",      "タイルを何枚ずつ ε-net に通すか"),
}

# ---------------------------------------------------------------------------
# stage3/configs/nps.yaml — NPS 評価（bash start3.sh nps → eval_nps.py）
# ---------------------------------------------------------------------------
NPS = {
    "roi_size": Key(int,  "--roi_size", "ROI の一辺（1024 の画素）"),
    "detrend":  Key(str,  "--detrend",  "ROI ごとのトレンド除去: mean（平均を引く）| poly2（2 次多項式面を引く）"),
    "n_bins":   Key(int,  "--n_bins",   "radial 平均のビン数（0 〜 0.5 cycles/pixel）"),
    "rois":     Key(rois, None,         "ROI の表 [{slice: <case>/<slice>.png, y: 左上の行, x: 左上の列}, ...]（1024 の画素）。推論の出力に含まれるスライスであること"),
}

# ---------------------------------------------------------------------------
# stage3/configs/crop.yaml — 学習中の images/crop と同じ固定領域を checkpoint 指定で単独実行
# 領域・crop_size は現在の train.yaml log.crop_* を読むので、ここにはサンプリング方式だけを書く。
# ---------------------------------------------------------------------------
CROP = {
    "panel_scale": Key(int,   "--panel_scale", "保存する144pxパネルの最近傍拡大倍率"),
    "device":      Key(str,   "--device",      "cpu | cuda"),
    "use_ema":     Key(bool,  "--use_ema",     "EMA の重みを使う（本家 sample と同じ）"),
    "sample_steps": Key(int,  "--sample_steps", "DDIM の step 数 S"),
    "eta":          Key(float,"--eta",          "DDIM eta"),
    "skip_type":    Key(str,  "--skip_type",    "uniform | quad"),
    "seed":         Key(int,  "--seed",         "固定領域ごとの x_T seed の基準"),
}

GPU_GENS = (0, 30, 40, 50)
INFER_INPUTS = ("eidlike", "eid")
SKIP_TYPES = ("uniform", "quad")
BETA_SCHEDULES = ("quad", "linear", "const", "jsd", "sigmoid")   # runners/diffusion.get_beta_schedule（本家）が持つもの
VAR_TYPES = ("fixedsmall", "fixedlarge")
OPTIMIZERS = ("Adam", "RMSProp", "SGD")                          # functions.get_optimizer（本家）が持つもの
DETRENDS = ("mean", "poly2")


# ---------------------------------------------------------------------------
# 検証・変換（stage2/configs/schema.py と同じ + ints / rois）
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


def unflatten(flat):
    """'a.b.c' キーの平坦 dict を入れ子に戻す（ddim.yml を本家の Namespace にするため）。"""
    out = {}
    for k, v in flat.items():
        d = out
        parts = k.split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = v
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
    if typ is ints:
        return isinstance(value, list) and all(_is_int(x) for x in value)
    if typ is rois:
        return isinstance(value, list) and all(isinstance(r, dict) and set(r) == {"slice", "y", "x"} and isinstance(r["slice"], str) and _is_int(r["y"]) and _is_int(r["x"]) for r in value)
    raise TypeError(typ)


def _coerce(value, typ):
    if typ is float:
        return float(value)
    if typ in (list, ints):
        return list(value)
    if typ is rois:
        return [dict(r) for r in value]
    return value


def _tname(typ):
    return {ints: "list[int]", rois: "list[{slice, y, x}]"}.get(typ, getattr(typ, "__name__", str(typ)))


def validate(name, data, schema):
    """平坦化した設定を schema と突き合わせる。欠落・未知・型違いをまとめて ConfigError にする。戻り値は平坦 dict。"""
    if not isinstance(data, dict):
        raise ConfigError(f"[{name}] トップレベルが dict ではありません: {type(data).__name__}")
    flat = flatten(data)
    missing = [k for k in schema if k not in flat]
    unknown = [k for k in flat if k not in schema]
    bad = [f"{k}: 期待 {_tname(schema[k].type)}, 実際 {type(flat[k]).__name__} ({flat[k]!r})" for k in flat if k in schema and not _type_ok(flat[k], schema[k].type)]
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
    """検証済みの平坦 dict を inference_dir.py の引数列にする（bool は true のときだけフラグを付ける）。"""
    argv = []
    for k, key in schema.items():
        if key.flag is None:
            continue
        v = flat[k]
        if key.type is bool:
            if v:
                argv.append(key.flag)
        elif key.type in (list, ints):
            argv += [key.flag, ",".join(str(x) for x in v)]
        else:
            argv += [key.flag, str(v)]
    return argv


def validate_shared(name, data, schema):
    """Stage 共通ファイル（machines.yaml）用: schema のキーの存在と型だけ検査し、他のキーは無視する。戻り値は schema のキーだけの平坦 dict。"""
    if not isinstance(data, dict):
        raise ConfigError(f"[{name}] トップレベルが dict ではありません: {type(data).__name__}")
    flat = flatten(data)
    missing = [k for k in schema if k not in flat]
    bad = [f"{k}: 期待 {_tname(schema[k].type)}, 実際 {type(flat[k]).__name__} ({flat[k]!r})" for k in schema if k in flat and not _type_ok(flat[k], schema[k].type)]
    problems = []
    if missing:
        problems.append("必須キーがありません: " + ", ".join(missing) + "（configs/machines.yaml に Stage 3 のキーを足すこと）")
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
        if typ in (list, ints):
            items = [x.strip() for x in str(val).split(",")]
            if any(not x for x in items):
                raise ValueError
            return [int(x) for x in items] if typ is ints else items
    except ValueError:
        raise ConfigError(f"{flag} の値を {_tname(typ)} に変換できません: {val!r}" + ("（リストはカンマ区切り。空要素は不可）" if typ in (list, ints) else ""))
    raise TypeError(typ)


def apply_overrides(tokens, sections):
    """sh からの上書き列を平坦 dict に反映する（後勝ち）。stage2 と同じ。
    sections: {"train": (flat_dict, TRAIN), "ddim": (flat_dict, DDIM), ...}。flat_dict はその場で書き換わる。
    受ける形: --flag value / --flag=value。bool は --flag（= true）または --flag true|false。list / ints は --flag a,b,c。
    schema に無いフラグ・値の無いフラグ・余った語・型違いは ConfigError。戻り値は適用した {flag: 値}（記録用）。"""
    index = {}
    for sec, (flat, schema) in sections.items():
        for key_name, key in schema.items():
            if key.flag:
                if key.flag in index:
                    raise ConfigError(f"schema のフラグが重複しています: {key.flag}（{index[key.flag][0]} と {sec}）")
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
# 値域・相互条件の検証（型検査の後、上書き反映後の実効値に対して）
# ---------------------------------------------------------------------------
def _problems_to_error(where, problems):
    if problems:
        raise ConfigError(f"[{where}] " + " / ".join(problems))


def _check_machine(machine, P):
    if machine["gpu_gen"] not in GPU_GENS: P.append(f"gpu_gen は {GPU_GENS}")
    if machine["num_threads"] < 0: P.append("num_threads ≥ 0")
    if not (1 <= machine["stage3_tb_port"] <= 65535): P.append("stage3_tb_port は 1..65535")


def _check_sampling(steps, eta, skip_type, stride, tile_batch, T, P, where, patch=None):
    if not (1 <= steps <= T): P.append(f"{where} の step 数 S は 1..{T}")
    if skip_type not in SKIP_TYPES: P.append(f"{where} の skip_type は {SKIP_TYPES}")
    if skip_type == "uniform" and 1 <= steps <= T and T % steps: P.append(f"{where}: skip_type uniform は T={T} を S={steps} で割り切れないと本家の range(0, T, T // S) が S+1 step 以上になる。T の約数（10 / 20 / 50 / 100 ...）にする")
    if eta < 0: P.append(f"{where} の eta ≥ 0")
    if stride < 1 or (patch is not None and stride > patch): P.append(f"{where} の tile_stride は 1..patch（{patch}）。patch より大きいとタイルの間に隙間ができる")
    if tile_batch < 1: P.append(f"{where} の tile_batch ≥ 1")


def check_ddim_values(ddim):
    """ddim.yml の値域と相互条件（models/diffusion.Model が組めること）。"""
    P = []
    size, ch, mult = ddim["data.image_size"], ddim["model.ch"], ddim["model.ch_mult"]
    if ddim["data.channels"] != 1: P.append("data.channels は 1（CT 1ch）")
    if ddim["model.in_channels"] != 2 * ddim["data.channels"]: P.append("model.in_channels = 2 × data.channels（条件と x_t の連結）")
    if ddim["model.out_ch"] != ddim["data.channels"]: P.append("model.out_ch = data.channels（ε）")
    if ddim["model.type"] != "simple": P.append("model.type は simple（bayesian は未対応）")
    if not mult or any(m < 1 for m in mult): P.append("model.ch_mult は正の int の非空リスト")
    elif size < 2 ** (len(mult) - 1) or size % (2 ** (len(mult) - 1)): P.append(f"data.image_size ({size}) は 2^(len(ch_mult) − 1) = {2 ** (len(mult) - 1)} の倍数")
    if ch < 32 or any((ch * m) % 32 for m in (mult or [1])): P.append("model.ch × ch_mult の各値は 32 の倍数（GroupNorm 32 群）")
    res = [size // (2 ** i) for i in range(len(mult))] if mult else []
    bad_attn = [r for r in ddim["model.attn_resolutions"] if r not in res]
    if bad_attn: P.append(f"model.attn_resolutions {bad_attn} がどの段の解像度 {res} にもありません")
    if ddim["model.num_res_blocks"] < 1: P.append("model.num_res_blocks ≥ 1")
    if not (0.0 <= ddim["model.dropout"] < 1.0): P.append("model.dropout は [0, 1)")
    if ddim["model.var_type"] not in VAR_TYPES: P.append(f"model.var_type は {VAR_TYPES}")
    if not (0.0 < ddim["model.ema_rate"] < 1.0): P.append("model.ema_rate は (0, 1)")
    if ddim["diffusion.beta_schedule"] not in BETA_SCHEDULES: P.append(f"diffusion.beta_schedule は {BETA_SCHEDULES}")
    if not (0.0 < ddim["diffusion.beta_start"] < ddim["diffusion.beta_end"] < 1.0): P.append("0 < diffusion.beta_start < beta_end < 1")
    if ddim["diffusion.num_diffusion_timesteps"] < 2: P.append("diffusion.num_diffusion_timesteps ≥ 2")
    if ddim["training.batch_size"] < 1: P.append("training.batch_size ≥ 1")
    for k in ("training.n_iters", "training.snapshot_freq", "training.validation_freq"):
        if ddim[k] < 1: P.append(f"{k} ≥ 1")
    if ddim["optim.optimizer"] not in OPTIMIZERS: P.append(f"optim.optimizer は {OPTIMIZERS}")
    if ddim["optim.lr"] <= 0: P.append("optim.lr は正")
    if not (0.0 <= ddim["optim.beta1"] < 1.0): P.append("optim.beta1 は [0, 1)")
    if ddim["optim.eps"] <= 0: P.append("optim.eps は正")
    if ddim["optim.weight_decay"] < 0: P.append("optim.weight_decay ≥ 0")
    if ddim["optim.grad_clip"] < 0: P.append("optim.grad_clip ≥ 0（0 = clip しない）")
    _problems_to_error("ddim values", P)


def check_train_values(train, ddim, machine):
    """train.yaml の値域と相互条件（学習）。症例リストとディスクの照合・画像サイズとタイルの整合は data/pair_dataset.py が行う。"""
    check_ddim_values(ddim)
    P = []
    if train["optim.seed"] < 0: P.append("optim.seed ≥ 0")
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
    if train["log.crop_freq"] < 1: P.append("log.crop_freq ≥ 1（step）")
    if not (8 <= train["log.crop_size"] <= ddim["data.image_size"]): P.append(f"log.crop_size は 8..ddim.yml data.image_size（{ddim['data.image_size']}）。領域を中央に置く 1 枚のタイルから切り出すため")
    if train["log.crop_scale"] < 1: P.append("log.crop_scale ≥ 1")
    for k in ("log.crop_pcd", "log.crop_eid"):
        for r in train[k]:
            if r["y"] < 0 or r["x"] < 0: P.append(f"{k} の y / x ≥ 0: {r}")
    for k in ("log.steps_per_epoch", "log.print_freq", "log.image_freq", "log.n_images", "log.n_val_patches", "log.full_freq", "log.save_weights_freq"):
        if train[k] < 1: P.append(f"{k} ≥ 1")
    if not train["log.full_slice"]: P.append("log.full_slice（固定スライスの相対パス）を指定")
    if not train["log.eid_slice"]: P.append("log.eid_slice（実 EID テストスライスの相対パス）を指定")
    _check_sampling(train["log.sample_steps"], train["log.sample_eta"], train["log.skip_type"], train["log.tile_stride"], train["log.tile_batch"],
                    ddim["diffusion.num_diffusion_timesteps"], P, "log", patch=ddim["data.image_size"])
    if train["log.sample_seed"] < 0: P.append("log.sample_seed ≥ 0")
    if train["log.preview_bits"] not in (8, 16): P.append("log.preview_bits は 8 | 16")
    if train["log.tb_full_size"] < 64: P.append("log.tb_full_size ≥ 64")
    if train["log.display_hu_min"] >= train["log.display_hu_max"]: P.append("log.display_hu_min < display_hu_max")
    _check_machine(machine, P)
    _problems_to_error("train values", P)


def check_infer_values(infer, machine, T):
    P = []
    if infer["input"] not in INFER_INPUTS: P.append(f"input は {INFER_INPUTS}")
    if infer["input"] == "eid" and infer["teacher"]: P.append("input=eid（実 EID）に教師は無いので teacher は false にする（--teacher false）")
    if infer["max_slices"] < 0: P.append("max_slices ≥ 0（0 で全部）")
    if infer["slices_per_case"] < 0: P.append("slices_per_case ≥ 0（0 で全部）")
    if infer["seed"] < 0: P.append("seed ≥ 0")
    _check_sampling(infer["sample_steps"], infer["eta"], infer["skip_type"], infer["tile_stride"], infer["tile_batch"], T, P, "infer")
    _check_machine(machine, P)
    _problems_to_error("infer values", P)


def check_crop_values(crop, train, machine, ddim):
    P = []
    p, n = ddim["data.image_size"], train["log.crop_size"]
    if not (1 <= n <= p): P.append(f"train.log.crop_size は 1..data.image_size ({p})")
    if (p - n) % 2: P.append("data.image_size - crop_size は偶数（中心を一致させるため）")
    if crop["panel_scale"] < 1: P.append("panel_scale >= 1")
    if crop["device"] not in ("cpu", "cuda"): P.append("device は cpu | cuda")
    if crop["device"] == "cuda" and machine["gpu_gen"] == 0: P.append("device=cuda だが machine は gpu_gen 0")
    if crop["use_ema"] and not ddim["model.ema"]: P.append("run は EMA なしなので use_ema false")
    if crop["seed"] < 0: P.append("seed >= 0")
    _check_sampling(crop["sample_steps"], crop["eta"], crop["skip_type"], 1, 1,
                    ddim["diffusion.num_diffusion_timesteps"], P, "crop", patch=p)
    _check_machine(machine, P)
    _problems_to_error("crop values", P)


def check_nps_values(nps):
    P = []
    if nps["roi_size"] < 8: P.append("roi_size ≥ 8")
    if nps["detrend"] not in DETRENDS: P.append(f"detrend は {DETRENDS}")
    if nps["n_bins"] < 4: P.append("n_bins ≥ 4")
    if not nps["rois"]: P.append("rois が空です。stage3/configs/nps.yaml に均一な ROI（スライス・左上の y / x）を書いてください（場所は推測で決めない）")
    for r in nps["rois"]:
        if r["y"] < 0 or r["x"] < 0: P.append(f"ROI の y / x ≥ 0: {r}")
    _problems_to_error("nps values", P)
