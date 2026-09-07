"""ct_dataset.py — FE-GAN (Park et al. 2019) 用の CT データローダ。`--dataset_mode ct` で選択する。

計画: docs/plans/20260905_phase5-ct-dataset-plan.md
照合: docs/reference/park2019_implementation_checklist.md §5

【入力データ】 --dir_A = PCD ディレクトリ (論文の z 側)、--dir_B = EID ディレクトリ (論文の x 側)。各直下に <症例>/<slice>.png
  （junyanz の <dataroot>/trainA|trainB 方式は使わない。パスは configs/machines.yaml → run_train.py が渡す）
  - 1ch・uint16 の 16bit PNG。stored = HU + 1400。それ以外 (3ch / 8bit) は明示的にエラーにする
  - 症例フォルダの下にスライス PNG (サブフォルダがあれば再帰的に拾う)

【正規化】 (確定 2026-09-05、SpicaV3 load_ct_as_tensor と演算順序まで同一)
  hu = stored − 1400 → clip(−1400, 4096) → [0,1] → [−1,1]   (float32、(1,H,W))
  論文は正規化を記載していない。FE-GAN の G は末尾 Tanh が無く BN が入力のアフィン変換を吸収するため
  範囲に必然性は無い。fidelity λ‖G(z)−z‖² の実効値は範囲幅の 2 乗に比例するので λ=10 は再調整対象。

【サンプリング】 `--sampling` で切替。dataset 本体は sampler の返り値 (A のパスと位置、B のパスと位置) しか見ない
  random  : ケース 1 (ユーザー設計)。1 サンプルごとに A/B 独立に 症例一様 → スライス一様 → 位置一様。
            epoch 長は --samples_per_epoch (既定 24,000 = 論文の patch 集合と同じ計算量)
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
class CaseIndex:
    """<root>/<症例>/<slice>.png を {症例: [パス, ...]} に整理する。症例フォルダ直下にサブフォルダがあれば再帰的に拾う。
    <root> 直下に PNG が直接ある場合は症例名 "_root" として 1 症例扱いにする。

    列挙は os.walk（scandir）+ 拡張子判定で、ファイルごとの stat はしない（Docker のバインドマウント越しでは stat が 1 件数 ms かかるため。旧 rglob + is_file は 2 倍遅かった）。
    '.' 始まりのファイル・フォルダは無視する（macOS の AppleDouble "._xxx.png" や .DS_Store が exFAT 経由のコピーで混ざる）。

    cache_dir を渡すと列挙結果を JSON に保存し、次回は「走査した全ディレクトリの mtime が一致」すれば再走査しない
    （ディレクトリの mtime は直下のファイルの追加・削除で変わる。数十万ファイルの列挙が数十回の stat になる）。ズレていれば再走査して上書き。"""

    EXT = ".png"

    def __init__(self, root, cache_dir=None):
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
    """ケース 1: 完全ランダム。A/B とも 症例一様 → スライス一様 → 位置一様 (stride 刻み)。A と B は独立。"""

    def __init__(self, idx_a, idx_b, img_hw, patch, stride):
        self.idx_a, self.idx_b = idx_a, idx_b
        self.ys, self.xs = _grid_positions(img_hw, patch, stride)

    def _pos(self, rng):
        return rng.choice(self.ys), rng.choice(self.xs)

    fixed_len = None  # 固定集合を持たない。epoch 長は dataset 側の samples_per_epoch

    def sample(self, rng, index):
        return self.idx_a.random_slice(rng), self._pos(rng), self.idx_b.random_slice(rng), self._pos(rng)


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
        parser.add_argument("--patch_size", type=int, default=None, help="patch の一辺。0 でフル画像 (テスト・推論用)")
        parser.add_argument("--patch_stride", type=int, default=None, help="patch 左上座標の刻み。1=任意位置、8=論文のグリッド")
        parser.add_argument("--patches_per_image", type=int, default=None, help="sampling=paper のときの 1 画像あたり patch 数 (論文 40)")
        parser.add_argument("--patch_seed", type=int, default=None, help="sampling=paper の固定集合、および --serial_batches 時の再現用 seed")
        parser.add_argument("--hu_offset", type=int, default=None, help="stored = HU + hu_offset")
        parser.add_argument("--hu_min", type=int, default=None, help="正規化窓の下限 (HU)")
        parser.add_argument("--hu_max", type=int, default=None, help="正規化窓の上限 (HU)")
        parser.add_argument("--align_meta", type=str, default=None, help="sampling=aligned 用の症例メタ (未実装)。空文字可")
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
        self.idx_a = CaseIndex(dir_a, cache_dir)
        if os.path.isdir(dir_b):
            self.idx_b = CaseIndex(dir_b, cache_dir)
        else:
            if not self.full_image:
                raise RuntimeError(f"学習には {dir_b} が必要です")
            warnings.warn(f"[CTDataset] {dir_b} が無いため B は A と同じ画像をダミーで返します")
            self.idx_b = self.idx_a

        # 固定スライス（毎 epoch のフル画像）は起動時に存在を検査する（無ければ学習前に止める）
        if not self.full_image and not (Path(dir_a) / opt.full_slice).is_file():
            raise RuntimeError(f"train.yaml log.full_slice が dir_A の下にありません: {Path(dir_a) / opt.full_slice}")
        # 先頭 1 枚で形式 (uint16 / 1ch) と画像サイズを確定する
        first = read_stored(self.idx_a.all_paths[0])
        self.img_hw = first.shape
        if not self.full_image and (self.patch > min(self.img_hw)):
            raise ValueError(f"patch_size {self.patch} が画像サイズ {self.img_hw} を超えています")

        if self.full_image:
            self.sampler = None
        elif opt.sampling == "random":
            self.sampler = RandomSampler(self.idx_a, self.idx_b, self.img_hw, self.patch, opt.patch_stride)
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

        print(f"[CTDataset] A: {self.idx_a.n_cases} cases / {self.idx_a.n_slices} slices, B: {self.idx_b.n_cases} cases / {self.idx_b.n_slices} slices, image {self.img_hw}, patch {self.patch}, sampling {'full' if self.full_image else opt.sampling}, len {len(self)}")

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
        a_path, a_pos, b_path, b_pos = self.sampler.sample(self._rng(index), index)
        return {"A": self._patch(a_path, a_pos), "B": self._patch(b_path, b_pos), "A_paths": a_path, "B_paths": b_path}

    def fixed_batch(self, n):
        """監視用の固定サンプル。patch_seed から決まる決定的な乱数で index 0..n-1 を引く（学習の乱数列は消費しない、resume 後も同じ）。
        戻り値: {"A": (n,1,p,p), "B": (n,1,p,p)}。n == 0 なら None（F-17。schema は n ≥ 1 を要求するので防御）"""
        if n <= 0:
            return None
        items = [self.sampler.sample(random.Random(self.opt.patch_seed * 1_000_003 + i), i) for i in range(n)]
        return {
            "A": torch.stack([self._patch(a_path, a_pos) for a_path, a_pos, _, _ in items]),
            "B": torch.stack([self._patch(b_path, b_pos) for _, _, b_path, b_pos in items]),
        }

    def full_slices(self, epoch):
        """checkpoint 保存時の書き出し用: 固定スライス（--full_slice、dir_A からの相対パス）1 枚 + epoch ごとに別のランダムスライス n_full_random 枚。
        ランダムは patch_seed と epoch から決定的（resume しても同じ epoch は同じスライス）。
        戻り値: {"paths": [str], "kinds": ["fixed", "random", ...], "A": (n,1,H,W)}"""
        fixed = Path(self.opt.dir_A) / self.opt.full_slice
        if not fixed.is_file():
            raise RuntimeError(f"固定スライスがありません（train.yaml log.full_slice は dir_A からの相対パス）: {fixed}")
        rng = random.Random(self.opt.patch_seed * 1_000_003 + int(epoch))
        pool = [p for p in self.idx_a.all_paths if Path(p) != fixed]
        randoms = rng.sample(pool, min(self.opt.n_full_random, len(pool)))
        paths = [str(fixed)] + randoms
        kinds = ["fixed"] + ["random"] * len(randoms)
        return {"paths": paths, "kinds": kinds, "A": torch.stack([torch.from_numpy(_load_normalized(p, *self.hu)).unsqueeze(0) for p in paths])}

    def __len__(self):
        if self.full_image:
            return self.idx_a.n_slices
        return self.opt.samples_per_epoch if self.sampler.fixed_len is None else self.sampler.fixed_len
