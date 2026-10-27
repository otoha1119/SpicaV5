"""stage1/configs/schema.py — 設定ファイル（train.yaml / mode.yaml / sr.yaml / infer.yaml / crop.yaml / machines.yaml）の唯一の正。

設計: docs/plans/20260905_config-design-plan.md
原則: 既定値（フォールバック）は無い。ここに宣言したキーは**必ず**設定ファイルに存在しなければならず、
      未知のキー・型違いはエラー。run_train.py はここを見て train.py の引数列を組み立てる。

各キーの宣言: Key(type, flag, doc)
  type : int | float | str | bool | list。float は int の値も受け付ける（10 → 10.0）。bool は YAML の true / false のみ。list は文字列のリスト（上書きはカンマ区切り --flag a,b）
  flag : train.py（junyanz 本家の argparse）へ渡すフラグ名。None は train.py に渡さない（Docker 起動などで使う）
         bool のキーは store_true フラグに対応し、true のときだけフラグを付ける
  doc  : 意味と論文（FE-GAN = Park et al. 2019）の出典
"""

from collections import namedtuple
from pathlib import Path

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
    "data.patches_per_slice": Key(int, "--patches_per_slice", "sampling=random で 1 回読んだスライス（A / B それぞれ）から連続して切る patch 数。1 = patch ごとに別スライス（2026-10-12 以前）。大きいほど読み出し・デコードが 1/k（既定 8。PC1 の sr で大幅高速化、degrade も同じ。2026-10-12）。位置は patch ごとに独立"),
    "data.hu_offset":       Key(int,   "--hu_offset",      "stored = HU + hu_offset（1400）"),
    "data.hu_min":          Key(int,   "--hu_min",         "正規化窓の下限 HU（−1400）"),
    "data.hu_max":          Key(int,   "--hu_max",         "正規化窓の上限 HU（4096）"),
    # log（util/monitor.py）。単位に注意: *_freq のうち print_freq / image_freq は **step（iteration）**、save_latest_freq は junyanz 由来で **画像枚数**
    "log.print_freq":       Key(int,   "--print_freq",     "バー末尾の損失・TensorBoard scalar・loss_log.txt を更新する間隔（step）"),
    "log.image_freq":       Key(int,   "--image_freq",     "TensorBoard に途中画像を出す間隔（step）"),
    "log.n_images":         Key(int,   "--n_images",       "途中画像の枚数（current / fixed それぞれ、128 patch）"),
    "log.full_slice":       Key(str,   "--full_slice",     "checkpoint 保存時にフル 512 で書き出す固定スライス（pcd_dir からの相対パス。例 PCD-002/PCD-002-215.png）。毎 epoch 同じ 1 枚"),
    "log.eid_slice":        Key(str,   "--eid_slice",      "パネルの右端に並べる EID の代表スライス（eid_dir からの相対パス。例 EID-001/EID-001-089.png）。固定。preview_fixed_<slice>/01_eid_<slice>.png にも 1 枚保存"),
    "log.n_full_random":    Key(int,   "--n_full_random",  "checkpoint 保存時にフル 512 で書き出すランダムスライスの枚数（epoch ごとに別のスライス。0 で無し）"),
    "log.display_hu_min":   Key(int,   "--display_hu_min", "表示用の線形範囲の下限 HU（この値以下を黒）"),
    "log.display_hu_max":   Key(int,   "--display_hu_max", "表示用の線形範囲の上限 HU（この値以上を白）。stored 4400 = HU 3000"),
    "log.preview_bits":     Key(int,   "--preview_bits",   "output_images/preview_*/（[EID | EID-like | PCD | R + ゲージ] のパネルと個別画像、表示用）のビット深度。16 = 表示範囲を 0..65535 に伸ばす（プレビューで正しく見え階調も細かい）| 8"),
    "log.diff_range_hu":    Key(int,   "--diff_range_hu",  "差分パネル (G(z) − z) のカラー表示の ±範囲 HU（白 = 0、純青 = −範囲、純赤 = +範囲。util/residual_color.py。推論の *_R_color/ も同じ値）"),
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
    "residual":       Key(bool, "--residual",       "true で G を residual 化: G(z) = z + F(z)（F = 生の generator、その出力が R。networks.ResidualGenerator）。損失は不変。false = 論文どおり G(z) を直接出力（段階 5-1、2026-09-11）"),
    "serial_batches": Key(bool, "--serial_batches", "true で sampler を index 決定的に（再現用）"),
    "task":           Key(str,  "--task",           "degrade（論文どおり A = PCD512 → B = EID512 の劣化生成。pcd_dir / eid_dir、分割なし）| sr（一発超解像: A = EID1024（eid1024_dir、EID_v5 を ×2 補間）→ B = PCD1024（pcd1024_dir）。configs/sr.yaml を読む: A の症例分割、B の除外、train.yaml の差し替え。2026-10-12）"),
}
# 導入前の run の launch.yaml / state.pth に無い mode のキーと、そのとき仮定する値。無い = その機構なしで学習された重みなので、推論・crop・resume では「導入前の意味」で扱う
# （「run 作成後に追加されたキーは今の yaml の値で補う」規則の例外。今の yaml の値で補うと旧 run の重みの意味が変わる。2026-09-11 residual、2026-10-12 task）
LEGACY_MODE_DEFAULTS = {"residual": False, "task": "degrade"}

# ---------------------------------------------------------------------------
# stage1/configs/sr.yaml — task: sr（一発超解像）のときだけ読む。degrade では一切読まない（既存の挙動は不変）
#   split_file / b_exclude_cases はここで宣言（train.py へのフラグ）。`train:` セクションは train.yaml の差し替え表で、書けるのは TRAIN のキーだけ（validate_sr）。
#   優先順位: train.yaml → sr.yaml の train → sh の --flag（後勝ち）。実効値と由来は launch.yaml に残る（run_train.py）。
#   補間の方式（scale / interp / input_size）はここに書かない: A = eid1024_dir の manifest.yaml（bash start2.sh dataset が書く）が正で、run_train.py が読んで launch に記録する
# ---------------------------------------------------------------------------
SR = {
    "split_file":      Key(str,  "--split_file",      "A 側（EID）の症例分割 YAML（train / val / test の症例名リスト。stage1/make_split.py が書く）。リポジトリ直下からの相対パスか絶対パス。train = 学習、val = epoch 末のランダムプレビュー、test = 読まない（log.full_slice の 1 枚は例外）"),
    "b_exclude_cases": Key(list, "--b_exclude_cases", "B 側（PCD1024）から外す症例名のリスト（Stage 2 の test と揃えて holdout にする）。空リスト [] で全症例。ディスクに無い名前はエラー。上書きは --b_exclude_cases PCD-001,PCD-002"),
}
SR_TRAIN_SECTION = "train"  # sr.yaml の中で train.yaml を差し替えるセクション名
SR_UPSAMPLE_KEYS = ("scale", "interp", "input_size")  # eid1024_dir/manifest.yaml の dataset.* から launch の sr.upsample に写すキー（推論が 512 入力を同じ方式で補間するため）

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
    # Stage 2 用（machines.yaml は Stage 共通ファイル。Stage 1 は使わないが、未知キーで落とさないためにここで宣言する。2026-09-08）
    "pcd1024_dir":         Key(str, "--pcd1024_dir",    "PCD1024（PCD512_v2 と同一スキャンの 1024 再構成）。Stage 2 の教師。Stage 1 は task: sr の B 側（x）に使う（degrade では使わない）"),
    "eid1024_dir":         Key(str, "--eid1024_dir",    "実 EID の 1024（EID_v5 を bash start2.sh dataset --input_dir <eid_dir> --eidlike1024_dir <ここ> で ×2 補間したもの。manifest.yaml 付き）。Stage 1 は task: sr の A 側（z）に使う（degrade では使わない）。2026-10-12"),
    "eidlike1024_dir":     Key(str, None,                "Stage 2 の学習入力（bash start2.sh dataset の出力先）。Stage 1 は使わない"),
    "stage2_checkpoints_dir": Key(str, None,           "Stage 2 の run の保存先ルート。Stage 1 は使わない"),
    "stage2_tb_port":      Key(int, None,                "Stage 2 の TensorBoard のポート（compose が両方公開するので start.sh も読む）。Stage 1 の TensorBoard は tb_port"),
    # Stage 3 用（2026-10-13。Stage 1 は使わないが、未知キーで落とさないためにここで宣言する）
    "stage3_checkpoints_dir": Key(str, None,           "Stage 3（条件付き DDIM）の run の保存先ルート。Stage 1 は使わない"),
    "stage3_tb_port":      Key(int, None,                "Stage 3 の TensorBoard のポート（compose が全 Stage のポートを公開するので start.sh も読む）"),
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
    "save_residual":    Key(bool, "--save_residual",    "true で残差 R（0 HU = 32768）と表示用カラーも <mode>_R/ <mode>_R_color/ に保存（Stage 2 用のデータ生成では不要なので既定 false。--save_residual true で一時的に有効）"),
    "save_panel":       Key(bool, "--save_panel",       "true で表示用パネル [EID-like | PCD | R + ゲージ]（学習の output_images/preview_* と同じ正規化表示、util/panel.py）も <mode>_panel/ に保存。学習中の目視確認用。既定 false"),
    # 速度（2026-09-09）: GPU は数 ms で終わるので、読み・計算・書きを同時に動かす
    "batch_slices":     Key(int,  "--batch_slices",     "同時に G に通すスライス数（full はそのまま 1 回の forward、patch は全スライスの patch を束ねて patch.batch_size ずつ。BN は eval なので結果は 1 枚ずつと同じ。3070 8GB で 8。CPU では 1 が最速）"),
    "io.read_workers":  Key(int,  "--read_workers",     "PNG の読み込み・デコードを先読みするスレッド数（0 = 直列）"),
    "io.write_workers": Key(int,  "--write_workers",    "PNG の書き込みを非同期にするスレッド数（0 = 直列）"),
    # DICOM 出力（OUTPUT_FORMAT = dicom | both のとき）。stored = (HU − intercept) / slope。参照 DICOM のタグと一致しなければエラー
    "dicom.rescale_slope":     Key(float, "--rescale_slope",     "元 DICOM の RescaleSlope。PCD（Siemens NAEOTOM Alpha）= 1"),
    "dicom.rescale_intercept": Key(float, "--rescale_intercept", "元 DICOM の RescaleIntercept。PCD（Siemens NAEOTOM Alpha）= −8192（格納値は符号なし 16bit）"),
}

# ---------------------------------------------------------------------------
# stage1/configs/crop.yaml — パッチ切り出し（bash start.sh crop）。cases（PCD / EID のペアと左上座標のリスト）は check_crop_cases で別に検証する
# ---------------------------------------------------------------------------
CROP = {
    "patch":       Key(int, "--patch",       "切り出す正方形の一辺（画素）。座標は左上 (x = 列, y = 行)"),
    "panel_scale": Key(int, "--panel_scale", "並べたパネルだけ最近傍で拡大する倍率（ラベルとゲージが読める大きさに）。個別画像は等倍"),
    "device":      Key(str, "--device",      "cpu | cuda。学習中に走らせるときは cpu（VRAM を取り合わない）"),
}
CROP_CASE_KEYS = {"pcd": str, "pcd_x": int, "pcd_y": int, "eid": str, "eid_x": int, "eid_y": int}

FILES = {"train": TRAIN, "mode": MODE, "machine": MACHINE, "sr": SR, "infer": INFER, "crop": CROP}


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
    if typ is list:
        return isinstance(value, list) and all(isinstance(x, str) and x for x in value)
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


def validate_sr(data):
    """sr.yaml: SR のキー + `train:`（train.yaml の差し替え表）。戻り値 (sr_flat, overlay_flat)。
    overlay は TRAIN のキー（入れ子は train.yaml と同じ書き方）だけを受け付け、型も TRAIN と同じ検査。無い・空でもよい（差し替え無し）。"""
    if not isinstance(data, dict):
        raise ConfigError(f"[sr] トップレベルが dict ではありません: {type(data).__name__}")
    data = dict(data)
    overlay = data.pop(SR_TRAIN_SECTION, None)
    sr = validate("sr", data, SR)
    if overlay is None:
        return sr, {}
    if not isinstance(overlay, dict):
        raise ConfigError(f"[sr] {SR_TRAIN_SECTION} は train.yaml と同じ入れ子の dict で書いてください")
    flat = flatten(overlay)
    unknown = [k for k in flat if k not in TRAIN]
    bad = [f"{k}: 期待 {TRAIN[k].type.__name__}, 実際 {type(flat[k]).__name__} ({flat[k]!r})" for k in flat if k in TRAIN and not _type_ok(flat[k], TRAIN[k].type)]
    problems = []
    if unknown:
        problems.append("train.yaml に無いキーです（差し替えできるのは TRAIN のキーだけ）: " + ", ".join(unknown))
    if bad:
        problems.append("型が違います: " + "; ".join(bad))
    if problems:
        raise ConfigError(f"[sr.{SR_TRAIN_SECTION}] " + " / ".join(problems))
    return sr, {k: (float(v) if TRAIN[k].type is float else v) for k, v in flat.items()}


def to_argv(flat, schema):
    """検証済みの平坦 dict を train.py の引数列にする。bool は true のときだけフラグを付ける。list はカンマ区切り 1 語（空リストは空文字）。"""
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
    sections: {"train": (flat_dict, TRAIN), "mode": (flat_dict, MODE), ...}。flat_dict はその場で書き換わる。
    受ける形: --flag value / --flag=value。bool は --flag（= true）または --flag true|false / --flag=false。list は --flag a,b,c（空リストは --flag ""）。
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
TASKS = ("degrade", "sr")
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
    if train["data.patches_per_slice"] < 1: P.append("data.patches_per_slice ≥ 1")
    if mode["sampling"] == "paper" and train["data.patches_per_slice"] != 1: P.append("sampling=paper のときは data.patches_per_slice を 1 にする（paper は固定集合を index で走査するので適用しない）")
    if train["data.hu_min"] >= train["data.hu_max"]: P.append("data.hu_min < data.hu_max")
    if train["log.print_freq"] < 1 or train["log.image_freq"] < 1: P.append("log.print_freq / image_freq ≥ 1（step）")
    if train["log.n_images"] < 1: P.append("log.n_images ≥ 1")
    if train["log.n_full_random"] < 0: P.append("log.n_full_random ≥ 0")
    if not train["log.full_slice"]: P.append("log.full_slice（固定スライスの相対パス）を指定")
    if not train["log.eid_slice"]: P.append("log.eid_slice（EID の代表スライスの相対パス）を指定")
    if train["log.display_hu_min"] >= train["log.display_hu_max"]: P.append("log.display_hu_min < display_hu_max")
    if train["log.preview_bits"] not in (8, 16): P.append("log.preview_bits は 8 | 16")
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
    if mode["task"] not in TASKS: P.append(f"task は {TASKS}")
    if machine["gpu_gen"] not in GPU_GENS: P.append(f"gpu_gen は {GPU_GENS}")
    if machine["num_threads"] < 0: P.append("num_threads ≥ 0")
    if not (1 <= machine["tb_port"] <= 65535): P.append("tb_port は 1..65535")
    _problems_to_error("values", P)


def check_sr_values(sr, root):
    """sr.yaml の実効値（task: sr のときだけ）。split_file は root（リポジトリ直下）基準で解決して存在を検査し、絶対パスに置き換える。"""
    P = []
    sf = Path(sr["split_file"])
    if not sr["split_file"]:
        P.append("split_file（症例分割 YAML）を指定。stage1/make_split.py で作る")
    else:
        sf = sf if sf.is_absolute() else Path(root) / sf
        if not sf.is_file():
            P.append(f"split_file がありません: {sf}（bash start.sh split で作る）")
        sr["split_file"] = str(sf)
    if len(set(sr["b_exclude_cases"])) != len(sr["b_exclude_cases"]): P.append("b_exclude_cases に重複があります")
    _problems_to_error("sr values", P)


SPLIT_KEYS = ("train", "val", "test")


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
    if infer["batch_slices"] < 1: P.append("batch_slices ≥ 1")
    if infer["io.read_workers"] < 0 or infer["io.write_workers"] < 0: P.append("io.read_workers / write_workers ≥ 0（0 = 直列）")
    if infer["dicom.rescale_slope"] == 0: P.append("dicom.rescale_slope ≠ 0")
    _problems_to_error("infer values", P)


def check_crop_values(crop):
    P = []
    if crop["patch"] < 1: P.append("patch ≥ 1")
    if crop["panel_scale"] < 1: P.append("panel_scale ≥ 1")
    if crop["device"] not in ("cpu", "cuda"): P.append("device は cpu | cuda")
    _problems_to_error("crop values", P)


def check_crop_cases(cases):
    """crop.yaml の cases: 1 件以上のリスト。各件は CROP_CASE_KEYS を全部持ち、余計なキー・型違い・負の座標はエラー。"""
    if not isinstance(cases, list) or not cases:
        raise ConfigError("[crop] cases は 1 件以上のリストで指定してください")
    P = []
    for i, c in enumerate(cases, 1):
        if not isinstance(c, dict):
            P.append(f"case{i}: dict ではありません"); continue
        missing = [k for k in CROP_CASE_KEYS if k not in c]
        unknown = [k for k in c if k not in CROP_CASE_KEYS]
        bad = [k for k, t in CROP_CASE_KEYS.items() if k in c and not _type_ok(c[k], t)]
        neg = [k for k in ("pcd_x", "pcd_y", "eid_x", "eid_y") if k in c and isinstance(c[k], int) and c[k] < 0]
        if missing: P.append(f"case{i}: キーがありません {missing}")
        if unknown: P.append(f"case{i}: 未知のキー {unknown}")
        if bad: P.append(f"case{i}: 型が違います {bad}")
        if neg: P.append(f"case{i}: 座標は 0 以上 {neg}")
        for k in ("pcd", "eid"):
            if k in c and isinstance(c[k], str) and c[k].count("-") != 2:
                P.append(f"case{i}: {k} はスライス名 PCD-nnn-sss / EID-nnn-sss の形: {c[k]!r}")
    _problems_to_error("crop cases", P)
    return cases


def check_opt(opt, where):
    """junyanz の opt に None（番兵）が残っていればエラー。run_train.py を経由せず train.py を直接叩いた事故を止める。"""
    missing = [n for n in opt_attr_names() if getattr(opt, n, None) is None]
    if missing:
        raise ConfigError(f"[{where}] 次の引数が指定されていません（既定値は無い。run_train.py 経由で起動すること）: " + ", ".join("--" + m for m in missing))
