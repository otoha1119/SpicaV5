"""ct_dataset.py — FE-GAN (Park et al. 2019) 用の CT データローダ。`--dataset_mode ct` で選択する。

計画: docs/plans/20260905_phase5-ct-dataset-plan.md
照合: docs/reference/park2019_implementation_checklist.md §5

【入力データ】 --dir_A = 論文の z 側 (A)、--dir_B = 論文の x 側 (B)。各直下に <症例>/<slice>.png
  （junyanz の <dataroot>/trainA|trainB 方式は使わない。パスは configs/machines.yaml → run_train.py が渡す）
  --task degrade: A = PCD512 (pcd_dir)、B = EID512 (eid_dir)。全症例を使う（従来どおり）
  --task sr     : A = EID1024 (eid1024_dir)、B = PCD1024 (pcd1024_dir)。[SpicaV5 2026-10-12、一発超解像の ablation]
                  A は --split_file (train / val / test の症例名) の train だけで学習し、val は epoch 末のランダムプレビュー、test は読まない
                  （例外は --full_slice の 1 枚 = 目視の holdout）。ディスク上の A の症例は全部どれかのリストに入っていること（未割当はエラー）。
                  B は --b_exclude_cases（カンマ区切り。空で無し）の症例を外す（無い名前はエラー）。A と B の画像サイズは同じであること。
                  --sr_scale / --sr_interp / --sr_input_size は A の manifest.yaml の方式（run_train.py が読む）。A の大きさ = input_size × scale を検査する
  - 1ch・uint16 の 16bit PNG。stored = HU + 1400。それ以外 (3ch / 8bit) は明示的にエラーにする
  - 症例フォルダの下にスライス PNG (サブフォルダがあれば再帰的に拾う)

【正規化】 (確定 2026-09-05、SpicaV3 load_ct_as_tensor と演算順序まで同一)
  hu = stored − 1400 → clip(−1400, 4096) → [0,1] → [−1,1]   (float32、(1,H,W))
  論文は正規化を記載していない。FE-GAN の G は末尾 Tanh が無く BN が入力のアフィン変換を吸収するため
  範囲に必然性は無い。fidelity λ‖G(z)−z‖² の実効値は範囲幅の 2 乗に比例するので λ=10 は再調整対象。

【サンプリング】 `--sampling` で切替。dataset 本体は sampler の返り値 (A のパスと位置、B のパスと位置) しか見ない
  random  : ケース 1 (ユーザー設計)。1 サンプルごとに A/B 独立に 症例一様 → スライス一様 → 位置一様。
            epoch 長は --samples_per_epoch (既定 24,000 = 論文の patch 集合と同じ計算量)
            --patches_per_slice k > 1 で、引いたスライス（A / B それぞれ）を k サンプル連続して使い回す（位置は毎回独立）。読み出し・デコードが 1/k。
            DataLoader の worker は 1 batch を丸ごと組むので、同じ batch に同じスライス由来の patch が k 個並ぶ。
            通常は worker ごとのカウンタ（k 回ごとに引き直す）、--serial_batches のときは index // k をグループにして決定的に引く [2026-10-12]
  paper   : 論文方式。"the patches of size 128 × 128 are extracted with strides of 8 in each direction ...
            Among them, 40 patches are randomly selected, and they are used as training image patches"
            → 学習前に 1 画像 40 位置 (stride 8 グリッド) を seed 付きで選び固定。A を走査、B はランダム
  aligned : ケース 2 (簡易位置合わせ)。未実装。症例 JSON (z_top / z_bottom / spine_x / spine_y) を使う予定

【augmentation】 なし (論文に記載なし)。反転・回転・強度変換は行わない。

【テスト・推論】 patch_size == 0 のとき A のスライスを順にフル画像で返す (B は A と同じ画像をダミーで返す)。
"""

import hashlib
import json
import os
import random
import time
import warnings
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from data.base_dataset import BaseDataset
from configs.schema import check_opt

# --- HU 定数 (option で上書き可。既定は SpicaV3 と同一) ---
HU_OFFSET = 1400
HU_MIN = -1400
HU_MAX = 4096


# ---------------------------------------------------------------------------
# 画像 I/O と正規化
# ---------------------------------------------------------------------------
def read_stored(path):
    """16bit 1ch PNG を uint16 (H, W) で読む。1ch・uint16 以外は受け付けない。"""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"画像を読み込めません: {path}")
    if img.ndim != 2:
        raise ValueError(f"1ch PNG のみ対応です (ndim={img.ndim}, shape={img.shape}): {path}")
    if img.dtype != np.uint16:
        raise ValueError(f"uint16 PNG のみ対応です (dtype={img.dtype}): {path}")
    return img


def normalize(stored, offset=HU_OFFSET, hu_min=HU_MIN, hu_max=HU_MAX):
    """stored (uint16) → [−1,1] float32。SpicaV3 load_ct_as_tensor と同じ演算順序。"""
    hu = stored.astype(np.float32) - offset
    hu = np.clip(hu, hu_min, hu_max)
    norm = (hu - hu_min) / (hu_max - hu_min)  # [0, 1]
    return norm * 2.0 - 1.0  # [−1, 1]


def denormalize(norm, offset=HU_OFFSET, hu_min=HU_MIN, hu_max=HU_MAX):
    """[−1,1] float → stored uint16。normalize の逆 (SpicaV3 tensor_to_ct_png と同じ round / clip)。推論 (フェーズ 7) が使う。"""
    n01 = np.clip((np.asarray(norm, dtype=np.float32) + 1.0) / 2.0, 0.0, 1.0)
    hu = n01 * (hu_max - hu_min) + hu_min
    stored = hu + offset
    return np.clip(np.round(stored), 0, 65535).astype(np.uint16)


def write_png(path, arr):
    """cv2.imwrite の戻り値を検査して書く（F-21）。失敗を黙って完了扱いにしない。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        ok = cv2.imwrite(str(path), arr)
    except cv2.error as e:  # 拡張子不明など（OpenCV 5 は False ではなく例外）
        raise IOError(f"PNG を書き込めません: {path}: {e}") from e
    if not ok:
        raise IOError(f"PNG を書き込めません: {path}")


R_ZERO = 32768  # 残差 PNG の 0 HU（uint16 で符号付きを表す）

# 512 → 1024 の補間（task: sr の推論で 512 の実 EID を受けたとき。Stage 2 の data/ct_io.upsample と同じ関数を複製。Stage 間で import しない）
INTERP_FLAGS = {"nearest": cv2.INTER_NEAREST, "bilinear": cv2.INTER_LINEAR, "bicubic": cv2.INTER_CUBIC, "area": cv2.INTER_AREA, "lanczos": cv2.INTER_LANCZOS4}


def upsample(stored, scale, interp):
    """uint16 (H, W) → uint16 (H·scale, W·scale)。float32 で cv2.resize（half-pixel 規約）→ 四捨五入 → [0, 65535] に clip。
    bash start2.sh dataset（EID1024_v1 の作成）と同じ演算なので、推論で 512 を受けて補間しても学習入力と同じ画像になる。"""
    if interp not in INTERP_FLAGS:
        raise ValueError(f"interp は {tuple(INTERP_FLAGS)} のどれか: {interp!r}")
    h, w = stored.shape
    out = cv2.resize(stored.astype(np.float32), (w * scale, h * scale), interpolation=INTERP_FLAGS[interp])
    return np.clip(np.rint(out), 0, 65535).astype(np.uint16)


def residual_stored(a_stored, g_stored):
    """残差 R = G(z) − z を uint16 で保存する規約: R_stored = (g_stored − a_stored) + 32768。
    stored 同士の差なので hu_offset は打ち消し、eidlike = pcd + (R − 32768) が厳密に成り立つ。"""
    r = g_stored.astype(np.int32) - a_stored.astype(np.int32) + R_ZERO
    return np.clip(r, 0, 65535).astype(np.uint16)


@lru_cache(maxsize=64)
def _load_normalized(path, offset, hu_min, hu_max):
    """読込 + 正規化をプロセス内で小さくキャッシュする (同じスライスから複数 patch を切るときの再デコード回避)。"""
    return normalize(read_stored(path), offset, hu_min, hu_max)


# ---------------------------------------------------------------------------
# 症例 → スライス一覧
# ---------------------------------------------------------------------------
def load_eval_only():
    """configs/eval_only.yaml の評価専用症例名（set）。学習系の列挙（drop_eval_only=True）だけが使う。ファイルが無い・形が違えばエラー（フォールバックしない）。[評価専用症例 2026-10-27]"""
    import yaml
    for d in Path(__file__).resolve().parents:
        p = d / "configs" / "eval_only.yaml"
        if p.is_file():
            with open(p, encoding="utf-8") as f:
                y = yaml.safe_load(f)
            cases = y.get("cases") if isinstance(y, dict) else None
            if not isinstance(cases, list) or not all(isinstance(c, str) for c in cases):
                raise RuntimeError(f"{p}: cases は症例名（文字列）のリストであること")
            return set(cases)
    raise RuntimeError("configs/eval_only.yaml が見つかりません（評価専用症例の定義。リポジトリ直下の configs/ に置く）")

class CaseIndex:
    """<root>/<症例>/<slice>.png を {症例: [パス, ...]} に整理する。症例フォルダ直下にサブフォルダがあれば再帰的に拾う。
    <root> 直下に PNG が直接ある場合は症例名 "_root" として 1 症例扱いにする。

    列挙は os.walk（scandir）+ 拡張子判定で、ファイルごとの stat はしない（Docker のバインドマウント越しでは stat が 1 件数 ms かかるため。旧 rglob + is_file は 2 倍遅かった）。
    '.' 始まりのファイル・フォルダは無視する（macOS の AppleDouble "._xxx.png" や .DS_Store が exFAT 経由のコピーで混ざる）。

    cache_dir を渡すと列挙結果を JSON に保存し、次回は「走査した全ディレクトリの mtime が一致」すれば再走査しない
    （ディレクトリの mtime は直下のファイルの追加・削除で変わる。数十万ファイルの列挙が数十回の stat になる）。ズレていれば再走査して上書き。"""

    EXT = ".png"

    def __init__(self, root, cache_dir=None, cases=None, exclude=(), drop_eval_only=False):
        """cases: この症例名だけに絞る（None で全部。ディスクに無い名前はエラー）。exclude: この症例名を外す（無い名前はエラー）。drop_eval_only: configs/eval_only.yaml の症例を黙って外す（学習系だけ True）。
        列挙とキャッシュはディレクトリ全体に対して行い、絞り込みはその後（キャッシュは分割に依らず共通）。[task: sr 用 2026-10-12]"""
        root = Path(root)
        if not root.is_dir():
            raise RuntimeError(f"データディレクトリが存在しません: {root}")
        self.root = root
        cache = self._cache_path(root, cache_dir)
        data = self._load_cache(cache, root) if cache else None
        if data is None:
            t0 = time.time()
            data = self._scan(root)
            n = sum(len(v) for v in data["slices"].values())
            print(f"[CaseIndex] {root}: {len(data['slices'])} cases / {n} files を列挙 ({time.time() - t0:.1f}s)" + (f" → cache {cache}" if cache else ""))
            if cache:
                cache.parent.mkdir(parents=True, exist_ok=True)
                with open(cache, "w", encoding="utf-8") as f:
                    json.dump(data, f)
        else:
            print(f"[CaseIndex] {root}: cache hit ({cache})")
        self.slices = data["slices"]
        if not self.slices:
            raise RuntimeError(f"PNG が見つかりません: {root}")
        self.eval_only_dropped = []  # 評価専用として外した症例（学習系だけ。推論は drop_eval_only=False で全症例を見る）
        if drop_eval_only:
            ev = load_eval_only()
            self.eval_only_dropped = sorted(c for c in self.slices if c in ev)
            self.slices = {c: v for c, v in self.slices.items() if c not in ev}
            if self.eval_only_dropped:
                print(f"[CaseIndex] {root}: 評価専用 {len(self.eval_only_dropped)} 症例を学習から除外 {self.eval_only_dropped}")
            if not self.slices:
                raise RuntimeError(f"評価専用を除くと症例が 0 になりました: {root}")
        self.disk_cases = sorted(self.slices.keys())  # 絞り込み前（ディスク上の全症例）
        if cases is not None:
            missing = [c for c in cases if c not in self.slices]
            if missing:
                raise RuntimeError(f"{root} に無い症例が指定されています: {missing}（ディスク上: {len(self.disk_cases)} 症例）")
            self.slices = {c: self.slices[c] for c in cases}
        if exclude:
            missing = [c for c in exclude if c not in self.slices]
            if missing:
                raise RuntimeError(f"{root} に無い症例を除外しようとしています: {missing}（黙って無視しない。sr.yaml の b_exclude_cases を確認）")
            self.slices = {c: v for c, v in self.slices.items() if c not in set(exclude)}
        if not self.slices:
            raise RuntimeError(f"絞り込みの結果、症例が 0 になりました: {root}")
        self.cases = sorted(self.slices.keys())
        self.all_paths = [p for c in self.cases for p in self.slices[c]]

    # --- 列挙 ---
    @classmethod
    def _scan(cls, root):
        slices, dir_mtimes = {}, {str(root): os.stat(root).st_mtime_ns}
        loose = []
        with os.scandir(root) as it:
            entries = sorted(it, key=lambda e: e.name)
        for e in entries:
            if e.name.startswith("."):
                continue
            if e.is_dir():
                files = []
                for d, dirs, names in os.walk(e.path):
                    dirs[:] = sorted(n for n in dirs if not n.startswith("."))
                    dir_mtimes[d] = os.stat(d).st_mtime_ns
                    files += [os.path.join(d, f) for f in names if f.lower().endswith(cls.EXT) and not f.startswith(".")]
                files.sort()
                if files:
                    slices[e.name] = files
            elif e.is_file() and e.name.lower().endswith(cls.EXT):
                loose.append(e.path)
        if loose:
            slices["_root"] = sorted(loose)
        return {"root": str(root), "dir_mtimes": dir_mtimes, "slices": slices}

    # --- キャッシュ ---
    @staticmethod
    def _cache_path(root, cache_dir):
        if cache_dir is None:
            return None
        key = hashlib.sha1(str(root.resolve()).encode("utf-8")).hexdigest()[:16]
        return Path(cache_dir) / f"{root.name}_{key}.json"

    @staticmethod
    def _load_cache(cache, root):
        if not cache.is_file():
            return None
        try:
            with open(cache, encoding="utf-8") as f:
                data = json.load(f)
            if data.get("root") != str(root):
                return None
            for d, m in data["dir_mtimes"].items():  # 走査した全ディレクトリの mtime が一致すれば有効
                if os.stat(d).st_mtime_ns != m:
                    return None
            return data
        except (OSError, KeyError, ValueError, TypeError):
            return None

    @property
    def n_cases(self):
        return len(self.cases)

    @property
    def n_slices(self):
        return len(self.all_paths)

    def random_slice(self, rng):
        """症例を一様に選び、その症例のスライスを一様に選ぶ (Q-D1: 症例一様)。"""
        case = rng.choice(self.cases)
        return rng.choice(self.slices[case])


# ---------------------------------------------------------------------------
# Sampler: sample(rng, index) -> (A_path, (yA, xA), B_path, (yB, xB))
# ---------------------------------------------------------------------------
def _grid_positions(img_hw, patch, stride):
    """patch の左上座標の候補 (y, x) を stride 刻みで列挙する。"""
    h, w = img_hw
    ys = list(range(0, h - patch + 1, stride))
    xs = list(range(0, w - patch + 1, stride))
    return ys, xs


class RandomSampler:
    """ケース 1: 完全ランダム。A/B とも 症例一様 → スライス一様 → 位置一様 (stride 刻み)。A と B は独立。
    k = patches_per_slice > 1 のとき、引いたスライス（A / B の組）を k サンプル使い回す（位置は毎回独立）。k = 1 は従来と同じ乱数列。"""

    def __init__(self, idx_a, idx_b, img_hw, patch, stride, k=1, seed=0):
        self.idx_a, self.idx_b = idx_a, idx_b
        self.ys, self.xs = _grid_positions(img_hw, patch, stride)
        self.k, self.seed = int(k), int(seed)
        self._left, self._cur, self._cur_group = 0, None, None  # worker ごとの状態（DataLoader の worker は fork されるので各 worker で独立に進む）

    def _pos(self, rng):
        return rng.choice(self.ys), rng.choice(self.xs)

    fixed_len = None  # 固定集合を持たない。epoch 長は dataset 側の samples_per_epoch

    def sample(self, rng, index, group=None):
        """group: None = worker のカウンタで k 回ごとに引き直す / int = そのグループ番号で決定的にスライスを引く（serial_batches、fixed_batch）。"""
        if self.k == 1:
            return self.idx_a.random_slice(rng), self._pos(rng), self.idx_b.random_slice(rng), self._pos(rng)
        if group is None:
            if self._left <= 0:
                self._cur, self._left = (self.idx_a.random_slice(rng), self.idx_b.random_slice(rng)), self.k
            self._left -= 1
        elif group != self._cur_group:
            g = random.Random(self.seed * 1_000_003 + 977 + int(group))
            self._cur_group, self._cur = group, (self.idx_a.random_slice(g), self.idx_b.random_slice(g))
        a_path, b_path = self._cur
        return a_path, self._pos(rng), b_path, self._pos(rng)


class PaperSampler:
    """論文方式: 学習前に 1 画像 patches_per_image 個の位置を stride グリッドから選んで固定集合にする。
    A は固定集合を index で走査、B は固定集合からランダム (非ペア)。"""

    def __init__(self, idx_a, idx_b, img_hw, patch, stride, per_image, seed):
        ys, xs = _grid_positions(img_hw, patch, stride)
        grid = [(y, x) for y in ys for x in xs]
        rng = random.Random(seed)
        self.items_a = [(p, pos) for p in idx_a.all_paths for pos in rng.sample(grid, min(per_image, len(grid)))]
        self.items_b = [(p, pos) for p in idx_b.all_paths for pos in rng.sample(grid, min(per_image, len(grid)))]
        self.fixed_len = len(self.items_a)  # 固定集合の大きさ = A 画像数 × per_image

    def sample(self, rng, index):
        a_path, a_pos = self.items_a[index % len(self.items_a)]
        b_path, b_pos = rng.choice(self.items_b)
        return a_path, a_pos, b_path, b_pos

    def export(self):
        return {"A": [(p, list(pos)) for p, pos in self.items_a], "B": [(p, list(pos)) for p, pos in self.items_b]}


class AlignedSampler:
    """ケース 2: 簡易位置合わせ (未実装。後日ユーザーと相談して設計する)。
    想定する症例メタ JSON:
      { "<case_id>": { "z_top": int, "z_bottom": int, "spine_x": int, "spine_y": int }, ... }
    想定する手順: PCD 症例ランダム → 有効範囲 [z_top, z_bottom] 内でスライスランダム →
                 EID 症例ランダム → z を (z − z_top)/(z_bottom − z_top) で対応付けたスライス →
                 patch 位置は脊椎頂点 (spine_x, spine_y) を基準にしたオフセットで A/B をおおよそ一致させる。"""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("--sampling aligned (ケース 2) は未実装です。docs/plans/20260905_phase5-ct-dataset-plan.md §1 を参照")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class CTDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """既定値は無い（docs/plans/20260905_config-design-plan.md）。すべて run_train.py が設定ファイルから明示して渡す。
        __init__ の check_opt で None が残っていればエラー。"""
        parser.add_argument("--dir_A", type=str, default=None, help="PCD（論文の z 側 = A）のディレクトリ。直下に症例フォルダ")
        parser.add_argument("--dir_B", type=str, default=None, help="EID（論文の x 側 = B）のディレクトリ。直下に症例フォルダ")
        parser.add_argument("--sampling", type=str, default=None, choices=["random", "paper", "aligned"], help="random=ケース1 完全ランダム | paper=論文の固定 40 patch/画像 | aligned=ケース2 (未実装)")
        parser.add_argument("--samples_per_epoch", type=int, default=None, help="sampling=random のときの 1 epoch のサンプル数")
        parser.add_argument("--patches_per_slice", type=int, default=None, help="sampling=random で 1 回読んだスライスから連続して切る patch 数（1 = 従来）")
        parser.add_argument("--use_eval_only_cases", action="store_true", help="configs/eval_only.yaml の評価専用症例も学習に使う（通常は除外。DataSet_new を学習に使う実験用）")
        parser.add_argument("--patch_size", type=int, default=None, help="patch の一辺。0 でフル画像 (テスト・推論用)")
        parser.add_argument("--patch_stride", type=int, default=None, help="patch 左上座標の刻み。1=任意位置、8=論文のグリッド")
        parser.add_argument("--patches_per_image", type=int, default=None, help="sampling=paper のときの 1 画像あたり patch 数 (論文 40)")
        parser.add_argument("--patch_seed", type=int, default=None, help="sampling=paper の固定集合、および --serial_batches 時の再現用 seed")
        parser.add_argument("--hu_offset", type=int, default=None, help="stored = HU + hu_offset")
        parser.add_argument("--hu_min", type=int, default=None, help="正規化窓の下限 (HU)")
        parser.add_argument("--hu_max", type=int, default=None, help="正規化窓の上限 (HU)")
        parser.add_argument("--align_meta", type=str, default=None, help="sampling=aligned 用の症例メタ (未実装)。空文字可")
        # task（mode.yaml）と sr 用（configs/sr.yaml + eid1024_dir/manifest.yaml。run_train.py が渡す。degrade では渡されない = None のまま）[2026-10-12]
        parser.add_argument("--task", type=str, default=None, choices=["degrade", "sr"], help="degrade（A = PCD512 → B = EID512）| sr（A = EID1024 → B = PCD1024、分割あり）")
        parser.add_argument("--eid1024_dir", type=str, default=None, help="machines.yaml eid1024_dir（記録用。sr では run_train.py が --dir_A にこれを入れる）")
        parser.add_argument("--pcd1024_dir", type=str, default=None, help="machines.yaml pcd1024_dir（記録用。sr では run_train.py が --dir_B にこれを入れる）")
        parser.add_argument("--split_file", type=str, default=None, help="sr: A 側の症例分割 YAML（train / val / test。絶対パス。run_train.py が解決）")
        parser.add_argument("--b_exclude_cases", type=str, default=None, help="sr: B 側から外す症例名（カンマ区切り。空文字で無し）")
        parser.add_argument("--sr_scale", type=int, default=None, help="sr: A の manifest.yaml の dataset.scale（512 → 1024 なら 2）")
        parser.add_argument("--sr_interp", type=str, default=None, help="sr: A の manifest.yaml の dataset.interp（bicubic 等）")
        parser.add_argument("--sr_input_size", type=int, default=None, help="sr: A の manifest.yaml の dataset.input_size（512）")
        # 切り出し・augmentation はこのローダの責務なので本家の前処理は無効化する（構造上の固定。チューニング対象ではない）
        parser.set_defaults(preprocess="none", no_flip=True, load_size=512, crop_size=128)
        return parser

    def __init__(self, opt):
        check_opt(opt, "CTDataset")  # dataset は model より先に作られるので、ここが番兵 None の最初の検査点
        BaseDataset.__init__(self, opt)
        self.hu = (opt.hu_offset, opt.hu_min, opt.hu_max)
        self.patch = opt.patch_size
        self.full_image = self.patch == 0

        dir_a = opt.dir_A
        dir_b = opt.dir_B
        cache_dir = Path(opt.checkpoints_dir) / ".case_index"  # 列挙結果のキャッシュ（run 共通。mtime が変われば自動で再走査）
        self.task = opt.task
        self.split = None       # sr: {"train": [...], "val": [...], "test": [...]}
        self.idx_a_val = None   # sr: val 症例の A（epoch 末のランダムプレビュー専用。学習には使わない）
        if self.task == "sr":
            self.split, b_exclude = self._load_sr_split(opt)
            self.idx_a = CaseIndex(dir_a, cache_dir, cases=self.split["train"], drop_eval_only=not opt.use_eval_only_cases)
            self.idx_a_val = CaseIndex(dir_a, cache_dir, cases=self.split["val"], drop_eval_only=not opt.use_eval_only_cases) if self.split["val"] else None
            unassigned = [c for c in self.idx_a.disk_cases if c not in set(self.split["train"]) | set(self.split["val"]) | set(self.split["test"])]
            if unassigned:
                raise RuntimeError(f"{dir_a} の症例が split_file のどのリストにも入っていません（分割を作り直す: bash start.sh split）: {unassigned[:10]}{' ...' if len(unassigned) > 10 else ''}（{len(unassigned)} 症例）")
            if not self.full_image:
                if not os.path.isdir(dir_b):
                    raise RuntimeError(f"学習には {dir_b} が必要です")
                self.idx_b = CaseIndex(dir_b, cache_dir, exclude=b_exclude, drop_eval_only=not opt.use_eval_only_cases)
            else:
                self.idx_b = self.idx_a
        else:
            self.idx_a = CaseIndex(dir_a, cache_dir, drop_eval_only=not opt.use_eval_only_cases)
            if os.path.isdir(dir_b):
                self.idx_b = CaseIndex(dir_b, cache_dir, drop_eval_only=not opt.use_eval_only_cases)
            else:
                if not self.full_image:
                    raise RuntimeError(f"学習には {dir_b} が必要です")
                warnings.warn(f"[CTDataset] {dir_b} が無いため B は A と同じ画像をダミーで返します")
                self.idx_b = self.idx_a

        # 固定スライス（毎 epoch のフル画像）は起動時に存在を検査する（無ければ学習前に止める）
        if not self.full_image and not (Path(dir_a) / opt.full_slice).is_file():
            raise RuntimeError(f"train.yaml log.full_slice が dir_A の下にありません: {Path(dir_a) / opt.full_slice}")
        if not self.full_image and not (Path(dir_b) / opt.eid_slice).is_file():
            raise RuntimeError(f"train.yaml log.eid_slice が dir_B の下にありません: {Path(dir_b) / opt.eid_slice}")
        # 先頭 1 枚で形式 (uint16 / 1ch) と画像サイズを確定する
        first = read_stored(self.idx_a.all_paths[0])
        self.img_hw = first.shape
        if not self.full_image and (self.patch > min(self.img_hw)):
            raise ValueError(f"patch_size {self.patch} が画像サイズ {self.img_hw} を超えています")
        if self.task == "sr" and not self.full_image:
            # A（EID1024）は manifest の方式で作った大きさ、B（PCD1024）は A と同じ大きさであること（B は _patch で遅延検査されるが、起動時に止める）
            expect = (opt.sr_input_size * opt.sr_scale,) * 2
            if tuple(self.img_hw) != expect:
                raise ValueError(f"A の画像サイズ {self.img_hw} が manifest の input_size × scale = {expect} と合いません: {self.idx_a.all_paths[0]}")
            b_hw = read_stored(self.idx_b.all_paths[0]).shape
            if tuple(b_hw) != tuple(self.img_hw):
                raise ValueError(f"A と B の画像サイズが違います（同解像度の G なので揃える）: A {self.img_hw} vs B {b_hw}（{self.idx_b.all_paths[0]}）")

        if self.full_image:
            self.sampler = None
        elif opt.sampling == "random":
            self.sampler = RandomSampler(self.idx_a, self.idx_b, self.img_hw, self.patch, opt.patch_stride, k=opt.patches_per_slice, seed=opt.patch_seed)
        elif opt.sampling == "paper":
            # 暗黙の置換はしない（F-19）。論文の stride 8 は yaml に明示する（schema.check_values が sampling=paper → patch_stride 8 を要求）
            self.sampler = PaperSampler(self.idx_a, self.idx_b, self.img_hw, self.patch, opt.patch_stride, opt.patches_per_image, opt.patch_seed)
            # 固定集合を保存 (再現条件)
            out = Path(opt.checkpoints_dir) / opt.name
            out.mkdir(parents=True, exist_ok=True)
            with open(out / "patch_index.json", "w") as f:
                json.dump(self.sampler.export(), f)
        elif opt.sampling == "aligned":
            if not opt.align_meta or not os.path.isfile(opt.align_meta):
                raise RuntimeError(f"--sampling aligned には症例メタファイルが必要です (--align_meta): {opt.align_meta!r}")
            self.sampler = AlignedSampler(self.idx_a, self.idx_b, self.img_hw, self.patch, opt.align_meta)
        else:
            raise ValueError(opt.sampling)

        print(f"[CTDataset] task {self.task}: A: {self.idx_a.n_cases} cases / {self.idx_a.n_slices} slices, B: {self.idx_b.n_cases} cases / {self.idx_b.n_slices} slices, image {self.img_hw}, patch {self.patch}, sampling {'full' if self.full_image else opt.sampling}, len {len(self)}"
              + (f", patches_per_slice {opt.patches_per_slice}" if not self.full_image and opt.sampling == "random" else ""))
        if self.task == "sr":
            n_val = self.idx_a_val.n_slices if self.idx_a_val is not None else 0
            print(f"[CTDataset] sr split ({opt.split_file}): A train {len(self.split['train'])} / val {len(self.split['val'])} ({n_val} slices, プレビュー専用) / test {len(self.split['test'])} (読まない) cases, "
                  f"B exclude {sorted(b_exclude) if b_exclude else '[]'} → {self.idx_b.n_cases} cases, upsample {opt.sr_input_size} × {opt.sr_scale} ({opt.sr_interp})")
            fixed_case = Path(opt.full_slice).parts[0] if Path(opt.full_slice).parts else ""
            if fixed_case in self.split["train"]:
                print(f"[CTDataset] 注意: log.full_slice の症例 {fixed_case} は train に入っている（目視の holdout にするなら test の症例を指定）")

    @staticmethod
    def _load_sr_split(opt):
        """sr 用: --split_file（train / val / test の症例名リスト）を読んで検査し、--b_exclude_cases（カンマ区切り）をリストにして返す。
        値の形の検査は configs/schema.check_split（run_train.py と同じ規則）。"""
        from configs.schema import ConfigError, check_split

        for k in ("split_file", "b_exclude_cases", "sr_scale", "sr_interp", "sr_input_size"):
            if getattr(opt, k, None) is None:
                raise RuntimeError(f"--task sr には --{k} が必要です（run_train.py 経由で起動すること）")
        path = Path(opt.split_file)
        if not path.is_file():
            raise RuntimeError(f"split_file がありません: {path}")
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        try:
            split = check_split(data, f"split_file {path}")
        except ConfigError as e:
            raise RuntimeError(str(e))
        b_exclude = [x.strip() for x in opt.b_exclude_cases.split(",") if x.strip()] if opt.b_exclude_cases.strip() else []
        if opt.sr_scale < 1 or opt.sr_input_size < 1:
            raise RuntimeError(f"--sr_scale / --sr_input_size は正: {opt.sr_scale} / {opt.sr_input_size}")
        if opt.sr_interp not in INTERP_FLAGS:
            raise RuntimeError(f"--sr_interp は {tuple(INTERP_FLAGS)} のどれか: {opt.sr_interp!r}")
        return split, b_exclude

    # --- 乱数 ---
    def _rng(self, index):
        # --serial_batches: index ごとに決定的 (再現用)。それ以外: プロセス内の random (DataLoader が worker ごとに seed する)
        if self.opt.serial_batches:
            return random.Random(self.opt.patch_seed * 1_000_003 + index)
        return random

    # --- 切り出し ---
    def _patch(self, path, pos):
        img = _load_normalized(path, *self.hu)
        if img.shape != self.img_hw:
            raise ValueError(f"画像サイズが揃っていません: {img.shape} != {self.img_hw}: {path}")
        y, x = pos
        return torch.from_numpy(np.ascontiguousarray(img[y : y + self.patch, x : x + self.patch])).unsqueeze(0)

    def __getitem__(self, index):
        if self.full_image:
            path = self.idx_a.all_paths[index % self.idx_a.n_slices]
            img = torch.from_numpy(_load_normalized(path, *self.hu)).unsqueeze(0)
            return {"A": img, "B": img, "A_paths": path, "B_paths": path}
        group = (index // self.opt.patches_per_slice) if (self.opt.serial_batches and self.opt.patches_per_slice > 1) else None  # 決定的モードだけ index でグループ化
        a_path, a_pos, b_path, b_pos = self.sampler.sample(self._rng(index), index, group)
        return {"A": self._patch(a_path, a_pos), "B": self._patch(b_path, b_pos), "A_paths": a_path, "B_paths": b_path}

    def fixed_batch(self, n):
        """監視用の固定サンプル。patch_seed から決まる決定的な乱数で index 0..n-1 を引く（学習の乱数列は消費しない、resume 後も同じ）。
        戻り値: {"A": (n,1,p,p), "B": (n,1,p,p)}。n == 0 なら None（F-17。schema は n ≥ 1 を要求するので防御）"""
        if n <= 0:
            return None
        items = [self.sampler.sample(random.Random(self.opt.patch_seed * 1_000_003 + i), i, group=-1 - i) for i in range(n)]  # 固定サンプルは worker のカウンタを使わず i ごとに決定的（負のグループ番号で学習側と衝突させない）
        return {
            "A": torch.stack([self._patch(a_path, a_pos) for a_path, a_pos, _, _ in items]),
            "B": torch.stack([self._patch(b_path, b_pos) for _, _, b_path, b_pos in items]),
        }

    def full_slices(self, epoch):
        """checkpoint 保存時の書き出し用: 固定スライス（--full_slice、dir_A からの相対パス）1 枚 + epoch ごとに別のランダムスライス n_full_random 枚。
        ランダムは patch_seed と epoch から決定的（resume しても同じ epoch は同じスライス）。
        ランダムの母集団: degrade = A の全スライス、sr = val の症例だけ（学習に使ったスライスは記憶で良く見えるので汎化の目視にならない。Stage 2 と同じ規則）。
        戻り値: {"paths": [str], "kinds": ["fixed", "random", ...], "A": (n,1,H,W)}"""
        fixed = Path(self.opt.dir_A) / self.opt.full_slice
        if not fixed.is_file():
            raise RuntimeError(f"固定スライスがありません（train.yaml log.full_slice は dir_A からの相対パス）: {fixed}")
        rng = random.Random(self.opt.patch_seed * 1_000_003 + int(epoch))
        if self.task == "sr":
            if self.idx_a_val is None and self.opt.n_full_random > 0:
                raise RuntimeError("sr: log.n_full_random > 0 なのに split_file の val が空です（val を入れるか n_full_random 0）")
            pool = [p for p in (self.idx_a_val.all_paths if self.idx_a_val is not None else []) if Path(p) != fixed]
        else:
            pool = [p for p in self.idx_a.all_paths if Path(p) != fixed]
        randoms = rng.sample(pool, min(self.opt.n_full_random, len(pool)))
        paths = [str(fixed)] + randoms
        kinds = ["fixed"] + ["random"] * len(randoms)
        return {"paths": paths, "kinds": kinds, "A": torch.stack([torch.from_numpy(_load_normalized(p, *self.hu)).unsqueeze(0) for p in paths])}

    def eid_reference(self):
        """パネルの右端に並べる EID の代表スライス（--eid_slice、dir_B からの相対パス。固定）。戻り値: {"path": str, "B": (1,1,H,W)}"""
        p = Path(self.opt.dir_B) / self.opt.eid_slice
        if not p.is_file():
            raise RuntimeError(f"EID の代表スライスがありません（train.yaml log.eid_slice は dir_B からの相対パス）: {p}")
        return {"path": str(p), "B": torch.from_numpy(_load_normalized(str(p), *self.hu)).unsqueeze(0).unsqueeze(0)}

    def __len__(self):
        if self.full_image:
            return self.idx_a.n_slices
        return self.opt.samples_per_epoch if self.sampler.fixed_len is None else self.sampler.fixed_len
