"""stage1_v2/data/dataset.py — CycleGAN 用の非ペア CT データローダ。

Stage 1 ``task: degrade`` と同じ A=PCD512、B=EID512、症例列挙、patch sampling、HU 正規化を使う。

【入力データ】 pcd_dir（A）と eid_dir（B）。<case>/<slice>.png、1ch uint16、stored = HU + 1400。degrade と同じく split は使わない。
【patch】 A / B 共通の data.patch_size（128）。random / paper sampling と patches_per_slice は Stage 1 と同じ。
【正規化】 hu = stored − 1400 → clip(−1400, 4096) → [−1, 1]。augmentation なし、マスクなし。
【推論】 PCD512 を G_A に通して EID-like512 を生成する。
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
from torch.utils.data import Dataset

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
    """R = G_A(z) − z を uint16 で保存する規約: R_stored = (g − a) + 32768。pcdlike = eid + (R − 32768) が厳密に成り立つ。"""
    r = g_stored.astype(np.int32) - a_stored.astype(np.int32) + R_ZERO
    return np.clip(r, 0, 65535).astype(np.uint16)


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
                raise RuntimeError(f"{root} に無い症例を除外しようとしています: {missing}")
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
# Sampler（A / B 同じ大きさ・同じ patch）
# ---------------------------------------------------------------------------
def _grid_positions(img_hw, patch, stride):
    h, w = img_hw
    return list(range(0, h - patch + 1, stride)), list(range(0, w - patch + 1, stride))


class RandomSampler:
    """完全ランダム。A / B とも 症例一様 → スライス一様 → 位置一様。A と B は独立。k = patches_per_slice > 1 のとき、引いたスライス（A / B の組）を k サンプル使い回す（位置は毎回独立）。"""

    fixed_len = None

    def __init__(self, idx_a, idx_b, img_hw, patch, stride, k=1, seed=0):
        self.idx_a, self.idx_b = idx_a, idx_b
        self.ys, self.xs = _grid_positions(img_hw, patch, stride)
        self.k, self.seed = int(k), int(seed)
        self._left, self._cur, self._cur_group = 0, None, None

    def _pos(self, rng):
        return rng.choice(self.ys), rng.choice(self.xs)

    def sample(self, rng, index, group=None):
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
    """Stage 1 の paper sampling: A の各画像から固定位置を走査し、B は固定集合からランダムに選ぶ。"""

    def __init__(self, idx_a, idx_b, img_hw, patch, stride, per_image, seed):
        ys, xs = _grid_positions(img_hw, patch, stride)
        grid = [(y, x) for y in ys for x in xs]
        rng = random.Random(seed)
        self.items_a = [(p, pos) for p in idx_a.all_paths for pos in rng.sample(grid, min(per_image, len(grid)))]
        self.items_b = [(p, pos) for p in idx_b.all_paths for pos in rng.sample(grid, min(per_image, len(grid)))]
        self.fixed_len = len(self.items_a)

    def sample(self, rng, index, group=None):
        del group
        a_path, a_pos = self.items_a[index % len(self.items_a)]
        b_path, b_pos = rng.choice(self.items_b)
        return a_path, a_pos, b_path, b_pos

    def export(self):
        return {"A": [(p, list(pos)) for p, pos in self.items_a], "B": [(p, list(pos)) for p, pos in self.items_b]}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class CycleGANDataset(Dataset):
    def __init__(self, train, machine, cache_dir=None, run_dir=None):
        """train / machine: 実効値（平坦 dict）。"""
        self.cfg = train
        self.hu = (train["data.hu_offset"], train["data.hu_min"], train["data.hu_max"])
        self.patch = train["data.patch_size"]
        dir_a, dir_b = Path(machine["pcd_dir"]), Path(machine["eid_dir"])
        self.dir_a, self.dir_b = dir_a, dir_b
        drop_eval_only = not train["data.use_eval_only_cases"]
        self.idx_a = CaseIndex(dir_a, cache_dir, drop_eval_only=drop_eval_only)
        self.idx_b = CaseIndex(dir_b, cache_dir, drop_eval_only=drop_eval_only)
        self.fixed_path = dir_a / train["log.full_slice"]
        if not self.fixed_path.is_file():
            raise RuntimeError(f"train.yaml log.full_slice が pcd_dir の下にありません: {self.fixed_path}")
        self.eid_ref_path = dir_b / train["log.eid_slice"]
        if not self.eid_ref_path.is_file():
            raise RuntimeError(f"train.yaml log.eid_slice が eid_dir の下にありません: {self.eid_ref_path}")
        self.hw = tuple(read_stored(self.idx_a.all_paths[0]).shape)
        hw_b = tuple(read_stored(self.idx_b.all_paths[0]).shape)
        if hw_b != self.hw:
            raise ValueError(f"PCD と EID の画像サイズが違います: PCD {self.hw} vs EID {hw_b}")
        if self.patch > min(self.hw):
            raise ValueError(f"patch_size {self.patch} が画像サイズ {self.hw} を超えています")
        sampling = train["data.sampling"]
        if sampling == "random":
            self.sampler = RandomSampler(self.idx_a, self.idx_b, self.hw, self.patch, train["data.patch_stride"], k=train["data.patches_per_slice"], seed=train["data.patch_seed"])
        elif sampling == "paper":
            self.sampler = PaperSampler(self.idx_a, self.idx_b, self.hw, self.patch, train["data.patch_stride"], train["data.patches_per_image"], train["data.patch_seed"])
            if run_dir is not None:
                with open(Path(run_dir) / "patch_index.json", "w", encoding="utf-8") as f:
                    json.dump(self.sampler.export(), f)
        elif sampling == "aligned":
            raise NotImplementedError("data.sampling=aligned は Stage 1 と同じく未実装です")
        else:
            raise ValueError(sampling)
        print(f"[CycleGANDataset] A=PCD512: {self.idx_a.n_cases} cases / {self.idx_a.n_slices} slices | "
              f"B=EID512: {self.idx_b.n_cases} cases / {self.idx_b.n_slices} slices | image {self.hw}, patch {self.patch}, sampling {sampling}, len {len(self)}"
              + (f", patches_per_slice {train['data.patches_per_slice']}" if sampling == "random" else ""))

    def info(self):
        return {"image_hw": list(self.hw), "patch": self.patch, "sampling": self.cfg["data.sampling"],
                "a_domain": "PCD512", "a_cases": self.idx_a.n_cases, "a_slices": self.idx_a.n_slices,
                "b_domain": "EID512", "b_cases": self.idx_b.n_cases, "b_slices": self.idx_b.n_slices,
                "use_eval_only_cases": self.cfg["data.use_eval_only_cases"],
                "fixed_slice": str(self.fixed_path), "eid_slice": str(self.eid_ref_path)}

    def _rng(self, index):
        if self.cfg["data.serial_batches"]:
            return random.Random(self.cfg["data.patch_seed"] * 1_000_003 + index)
        return random

    def _patch(self, path, pos):
        img = _load_normalized(path, *self.hu)
        if img.shape != self.hw:
            raise ValueError(f"画像サイズが揃っていません: {img.shape} != {self.hw}: {path}")
        y, x = pos
        return torch.from_numpy(np.ascontiguousarray(img[y : y + self.patch, x : x + self.patch])).unsqueeze(0)

    def _pair(self, a_path, a_pos, b_path, b_pos):
        return {"A": self._patch(a_path, a_pos), "B": self._patch(b_path, b_pos), "A_paths": a_path, "B_paths": b_path}

    def __getitem__(self, index):
        k = self.cfg["data.patches_per_slice"]
        group = (index // k) if (self.cfg["data.serial_batches"] and k > 1) else None
        return self._pair(*self.sampler.sample(self._rng(index), index, group))

    def __len__(self):
        return self.cfg["data.samples_per_epoch"] if self.sampler.fixed_len is None else self.sampler.fixed_len

    def fixed_batch(self, n):
        if n <= 0:
            return None
        items = [self.sampler.sample(random.Random(self.cfg["data.patch_seed"] * 1_000_003 + i), i, group=-1 - i) for i in range(n)]
        pairs = [self._pair(*it) for it in items]
        return {"A": torch.stack([p["A"] for p in pairs]), "B": torch.stack([p["B"] for p in pairs])}

    def full_slices(self, epoch):
        """固定 PCD 1 枚 + PCD 全症例から epoch ごとに別のランダム画像。"""
        rng = random.Random(self.cfg["data.patch_seed"] * 1_000_003 + int(epoch))
        n_rand = self.cfg["log.n_full_random"]
        pool = [p for p in self.idx_a.all_paths if Path(p) != self.fixed_path]
        randoms = rng.sample(pool, min(n_rand, len(pool)))
        paths = [str(self.fixed_path)] + randoms
        return {"paths": paths, "kinds": ["fixed"] + ["random"] * len(randoms),
                "A": torch.stack([torch.from_numpy(_load_normalized(p, *self.hu)).unsqueeze(0) for p in paths])}

    def eid_reference(self):
        return {"path": str(self.eid_ref_path), "B": torch.from_numpy(_load_normalized(str(self.eid_ref_path), *self.hu)).unsqueeze(0).unsqueeze(0)}
