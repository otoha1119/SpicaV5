"""ablation/SRCycleGAN/data/dataset.py — SR-CycleGAN 用の非ペア CT データローダ（A = EID512 patch p、B = PCD1024 patch p×scale）。
stage1/data/ct_dataset.py（read_stored / normalize / denormalize / write_png / residual_stored / _load_normalized / CaseIndex / RandomSampler）を 2026-10-27 に複製し、
A と B の寸法が違う（×scale）ことに合わせて改変した。Stage 1 / 2 からは import しない。

【入力データ】 --eid_dir（A、512、<case>/<slice>.png、1ch uint16、stored = HU + 1400）、--pcd1024_dir（B、1024、同じ規約）
  A は split_file の train 症例だけで学習し、val は epoch 末のランダムプレビュー、test は読まない（例外は log.full_slice の 1 枚 = 目視の holdout）。
  ディスク上の A の症例が全部どれかのリストに入っていること（未割当はエラー）。B は b_exclude_cases を外す（無い名前はエラー）。
  起動時に A の先頭 1 枚と B の先頭 1 枚を読み、**B の一辺 = A の一辺 × scale** を検査する（違えばエラー）。

【patch】 A の patch = data.patch_size（128）、B の patch = patch_size × scale（256）。同じ解剖スケール（論文の 32 / 256 と同じ決め方）。
  位置の grid は A / B 別（stride 1）。非ペアなので A と B は症例・スライス・位置とも独立に引く（症例一様 → スライス一様 → 位置一様）。
  patches_per_slice k > 1 で、引いたスライス（A / B それぞれ）を k patch 使い回す（位置は毎回独立。Stage 1 と同じ）。

【正規化】 hu = stored − 1400 → clip(−1400, 4096) → [−1, 1]（Stage 1 と同じ。G の Tanh と値域が合う）。論文はモダリティごとの min-max。
【augmentation】 なし。背景マスクなし（論文の実装は肺マスク内に限定していたが、本研究は心臓 FOV）。
"""

import hashlib
import json
import os
import random
import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

from configs.schema import ConfigError, check_split

R_ZERO = 32768  # 残差 PNG の 0 HU


# ---------------------------------------------------------------------------
# 画像 I/O と正規化（stage1/data/ct_dataset.py と同じ演算）
# ---------------------------------------------------------------------------
def read_stored(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"画像を読み込めません: {path}")
    if img.ndim != 2:
        raise ValueError(f"1ch PNG のみ対応です (ndim={img.ndim}, shape={img.shape}): {path}")
    if img.dtype != np.uint16:
        raise ValueError(f"uint16 PNG のみ対応です (dtype={img.dtype}): {path}")
    return img


def normalize(stored, offset, hu_min, hu_max):
    hu = stored.astype(np.float32) - offset
    hu = np.clip(hu, hu_min, hu_max)
    norm = (hu - hu_min) / (hu_max - hu_min)
    return norm * 2.0 - 1.0


def denormalize(norm, offset, hu_min, hu_max):
    n01 = np.clip((np.asarray(norm, dtype=np.float32) + 1.0) / 2.0, 0.0, 1.0)
    hu = n01 * (hu_max - hu_min) + hu_min
    return np.clip(np.round(hu + offset), 0, 65535).astype(np.uint16)


def write_png(path, arr):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        ok = cv2.imwrite(str(path), arr)
    except cv2.error as e:
        raise IOError(f"PNG を書き込めません: {path}: {e}") from e
    if not ok:
        raise IOError(f"PNG を書き込めません: {path}")


def residual_stored(a_stored, g_stored):
    """R = G1(z) − f↑(z) を uint16 で保存する規約: R_stored = (g − a_up) + 32768。pcdlike = up(eid) + (R − 32768) が厳密に成り立つ。"""
    r = g_stored.astype(np.int32) - a_stored.astype(np.int32) + R_ZERO
    return np.clip(r, 0, 65535).astype(np.uint16)


def upsample_nearest(stored, scale):
    """uint16 (H, W) → (H·scale, W·scale) の最近傍（f↑ と同じ演算。表示と R の基準に使う）。"""
    return np.repeat(np.repeat(stored, scale, axis=0), scale, axis=1)


@lru_cache(maxsize=64)
def _load_normalized(path, offset, hu_min, hu_max):
    return normalize(read_stored(path), offset, hu_min, hu_max)


# ---------------------------------------------------------------------------
# 症例 → スライス一覧（stage1 CaseIndex の複製）
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
    EXT = ".png"

    def __init__(self, root, cache_dir=None, cases=None, exclude=(), drop_eval_only=False):
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
        self.disk_cases = sorted(self.slices.keys())
        if cases is not None:
            missing = [c for c in cases if c not in self.slices]
            if missing:
                raise RuntimeError(f"{root} に無い症例が指定されています: {missing}（ディスク上: {len(self.disk_cases)} 症例）")
            self.slices = {c: self.slices[c] for c in cases}
        if exclude:
            missing = [c for c in exclude if c not in self.slices]
            if missing:
                raise RuntimeError(f"{root} に無い症例を除外しようとしています: {missing}（黙って無視しない。train.yaml data.b_exclude_cases を確認）")
            self.slices = {c: v for c, v in self.slices.items() if c not in set(exclude)}
        if not self.slices:
            raise RuntimeError(f"絞り込みの結果、症例が 0 になりました: {root}")
        self.cases = sorted(self.slices.keys())
        self.all_paths = [p for c in self.cases for p in self.slices[c]]

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
            for d, m in data["dir_mtimes"].items():
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
        case = rng.choice(self.cases)
        return rng.choice(self.slices[case])


# ---------------------------------------------------------------------------
# Sampler: A / B で grid が別（寸法が違う）
# ---------------------------------------------------------------------------
def _grid_positions(img_hw, patch):
    """patch の左上座標の候補（stride 1）。"""
    h, w = img_hw
    return list(range(0, h - patch + 1)), list(range(0, w - patch + 1))


class RandomSampler:
    """完全ランダム。A / B とも 症例一様 → スライス一様 → 位置一様（それぞれの grid）。A と B は独立。
    k = patches_per_slice > 1 のとき、引いたスライス（A / B の組）を k サンプル使い回す（位置は毎回独立）。"""

    def __init__(self, idx_a, idx_b, hw_a, patch_a, hw_b, patch_b, k=1, seed=0):
        self.idx_a, self.idx_b = idx_a, idx_b
        self.grid_a = _grid_positions(hw_a, patch_a)
        self.grid_b = _grid_positions(hw_b, patch_b)
        self.k, self.seed = int(k), int(seed)
        self._left, self._cur, self._cur_group = 0, None, None  # worker ごとの状態

    @staticmethod
    def _pos(rng, grid):
        ys, xs = grid
        return rng.choice(ys), rng.choice(xs)

    def sample(self, rng, group=None):
        """group: None = worker のカウンタで k 回ごとに引き直す / int = そのグループ番号で決定的にスライスを引く（serial_batches、fixed_batch）。"""
        if self.k == 1:
            return self.idx_a.random_slice(rng), self._pos(rng, self.grid_a), self.idx_b.random_slice(rng), self._pos(rng, self.grid_b)
        if group is None:
            if self._left <= 0:
                self._cur, self._left = (self.idx_a.random_slice(rng), self.idx_b.random_slice(rng)), self.k
            self._left -= 1
        elif group != self._cur_group:
            g = random.Random(self.seed * 1_000_003 + 977 + int(group))
            self._cur_group, self._cur = group, (self.idx_a.random_slice(g), self.idx_b.random_slice(g))
        a_path, b_path = self._cur
        return a_path, self._pos(rng, self.grid_a), b_path, self._pos(rng, self.grid_b)


# ---------------------------------------------------------------------------
# split
# ---------------------------------------------------------------------------
def load_split(split_file):
    """split_file（train / val / test の症例名リスト）を読んで検査する。"""
    path = Path(split_file)
    if not path.is_file():
        raise RuntimeError(f"split_file がありません: {path}（Stage 1 の bash start.sh split で作る）")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    try:
        return check_split(data, f"split_file {path}")
    except ConfigError as e:
        raise RuntimeError(str(e))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class SRCGDataset(Dataset):
    def __init__(self, train, machine, cache_dir=None):
        """train / machine: 実効値（平坦 dict）。"""
        self.cfg = train
        self.hu = (train["data.hu_offset"], train["data.hu_min"], train["data.hu_max"])
        self.scale = train["network.scale"]
        self.patch_a = train["data.patch_size"]
        self.patch_b = self.patch_a * self.scale  # B の patch は常に A × scale
        dir_a, dir_b = Path(machine["eid_dir"]), Path(machine["pcd1024_dir"])
        self.dir_a, self.dir_b = dir_a, dir_b
        self.split = load_split(train["data.split_file"])
        self.idx_a = CaseIndex(dir_a, cache_dir, cases=self.split["train"], drop_eval_only=True)
        self.idx_a_val = CaseIndex(dir_a, cache_dir, cases=self.split["val"], drop_eval_only=True) if self.split["val"] else None
        assigned = set(self.split["train"]) | set(self.split["val"]) | set(self.split["test"])
        unassigned = [c for c in self.idx_a.disk_cases if c not in assigned]
        if unassigned:
            raise RuntimeError(f"{dir_a} の症例が split_file のどのリストにも入っていません（分割を作り直す: bash start.sh split）: {unassigned[:10]}{' ...' if len(unassigned) > 10 else ''}（{len(unassigned)} 症例）")
        self.b_exclude = list(train["data.b_exclude_cases"])
        self.idx_b = CaseIndex(dir_b, cache_dir, exclude=self.b_exclude, drop_eval_only=True)
        # 固定スライス・代表スライスの存在
        self.fixed_path = dir_a / train["log.full_slice"]
        if not self.fixed_path.is_file():
            raise RuntimeError(f"train.yaml log.full_slice が eid_dir の下にありません: {self.fixed_path}")
        self.pcd_ref_path = dir_b / train["log.pcd_slice"]
        if not self.pcd_ref_path.is_file():
            raise RuntimeError(f"train.yaml log.pcd_slice が pcd1024_dir の下にありません: {self.pcd_ref_path}")
        # 画像サイズ: A の一辺 × scale = B の一辺
        self.hw_a = tuple(read_stored(self.idx_a.all_paths[0]).shape)
        self.hw_b = tuple(read_stored(self.idx_b.all_paths[0]).shape)
        if tuple(x * self.scale for x in self.hw_a) != self.hw_b:
            raise ValueError(f"B の画像サイズ {self.hw_b} が A {self.hw_a} × scale {self.scale} と合いません（A {self.idx_a.all_paths[0]} / B {self.idx_b.all_paths[0]}）")
        if self.patch_a > min(self.hw_a):
            raise ValueError(f"patch_size {self.patch_a} が A の画像サイズ {self.hw_a} を超えています")
        self.sampler = RandomSampler(self.idx_a, self.idx_b, self.hw_a, self.patch_a, self.hw_b, self.patch_b, k=train["data.patches_per_slice"], seed=train["data.patch_seed"])
        self.n_val = self.idx_a_val.n_slices if self.idx_a_val is not None else 0
        print(f"[SRCGDataset] A (EID, {self.hw_a}, patch {self.patch_a}): train {self.idx_a.n_cases} cases / {self.idx_a.n_slices} slices, val {len(self.split['val'])} cases / {self.n_val} slices (preview only), test {len(self.split['test'])} cases (not read) | "
              f"B (PCD1024, {self.hw_b}, patch {self.patch_b}): {self.idx_b.n_cases} cases / {self.idx_b.n_slices} slices (exclude {self.b_exclude}) | len {len(self)}, patches_per_slice {train['data.patches_per_slice']}")
        fixed_case = Path(train["log.full_slice"]).parts[0]
        if fixed_case in self.split["train"]:
            print(f"[SRCGDataset] 注意: log.full_slice の症例 {fixed_case} は train に入っている（目視の holdout にするなら test の症例を指定）")

    def info(self):
        """dataset_info.yaml に書く内容。"""
        return {"image_hw_a": list(self.hw_a), "image_hw_b": list(self.hw_b), "scale": self.scale, "patch_a": self.patch_a, "patch_b": self.patch_b,
                "a_train_cases": self.idx_a.n_cases, "a_train_slices": self.idx_a.n_slices, "a_val_slices": self.n_val,
                "b_cases": self.idx_b.n_cases, "b_slices": self.idx_b.n_slices, "b_exclude": self.b_exclude,
                "split": {k: list(v) for k, v in self.split.items()}, "fixed_slice": str(self.fixed_path), "pcd_slice": str(self.pcd_ref_path)}

    # --- 乱数 ---
    def _rng(self, index):
        if self.cfg["data.serial_batches"]:
            return random.Random(self.cfg["data.patch_seed"] * 1_000_003 + index)
        return random

    # --- 切り出し ---
    def _patch(self, path, pos, size, hw):
        img = _load_normalized(path, *self.hu)
        if img.shape != hw:
            raise ValueError(f"画像サイズが揃っていません: {img.shape} != {hw}: {path}")
        y, x = pos
        return torch.from_numpy(np.ascontiguousarray(img[y : y + size, x : x + size])).unsqueeze(0)

    def _pair(self, a_path, a_pos, b_path, b_pos):
        return {"A": self._patch(a_path, a_pos, self.patch_a, self.hw_a), "B": self._patch(b_path, b_pos, self.patch_b, self.hw_b), "A_paths": a_path, "B_paths": b_path}

    def __getitem__(self, index):
        k = self.cfg["data.patches_per_slice"]
        group = (index // k) if (self.cfg["data.serial_batches"] and k > 1) else None
        return self._pair(*self.sampler.sample(self._rng(index), group))

    def __len__(self):
        return self.cfg["data.samples_per_epoch"]

    def fixed_batch(self, n):
        """監視用の固定サンプル（patch_seed から決定的。学習の乱数列は消費しない）。{"A": (n,1,p,p), "B": (n,1,p·s,p·s)}"""
        if n <= 0:
            return None
        items = [self.sampler.sample(random.Random(self.cfg["data.patch_seed"] * 1_000_003 + i), group=-1 - i) for i in range(n)]
        pairs = [self._pair(*it) for it in items]
        return {"A": torch.stack([p["A"] for p in pairs]), "B": torch.stack([p["B"] for p in pairs])}

    def full_slices(self, epoch):
        """epoch 末の書き出し用: 固定スライス（log.full_slice）1 枚 + val の症例から epoch ごとに別のランダム n_full_random 枚（A = EID512 フル）。
        戻り値 {"paths": [str], "kinds": ["fixed", "random", ...], "A": (n,1,H,W)}"""
        rng = random.Random(self.cfg["data.patch_seed"] * 1_000_003 + int(epoch))
        n_rand = self.cfg["log.n_full_random"]
        if self.idx_a_val is None and n_rand > 0:
            raise RuntimeError("log.n_full_random > 0 なのに split_file の val が空です（val を入れるか n_full_random 0）")
        pool = [p for p in (self.idx_a_val.all_paths if self.idx_a_val is not None else []) if Path(p) != self.fixed_path]
        randoms = rng.sample(pool, min(n_rand, len(pool)))
        paths = [str(self.fixed_path)] + randoms
        return {"paths": paths, "kinds": ["fixed"] + ["random"] * len(randoms),
                "A": torch.stack([torch.from_numpy(_load_normalized(p, *self.hu)).unsqueeze(0) for p in paths])}

    def pcd_reference(self):
        """パネル左端の PCD1024 代表（log.pcd_slice）。{"path": str, "B": (1,1,H,W)}"""
        return {"path": str(self.pcd_ref_path), "B": torch.from_numpy(_load_normalized(str(self.pcd_ref_path), *self.hu)).unsqueeze(0).unsqueeze(0)}
