"""stage1_v3/data/dataset.py — 射影型 Stage 1 の非ペア CT データローダ（stage1_v2/data/dataset.py を土台に P を足したもの）。

Stage 1 ``task: degrade`` と同じ A = PCD512、B = EID512、症例列挙、patch sampling、HU 正規化。

【入力データ】 pcd_dir（A）と eid_dir（B）。<case>/<slice>.png、1ch uint16、stored = HU + 1400。split は使わない。
【P】 A のスライスを読むたびに degrade.sample_params でスライス 1 枚分の (σ_b, σ_n, c) を引き、フル 512 の HU 画像に apply_degrade を掛けてから patch を切る
      （patches_per_slice 枚の patch は同じ 1 回の P を共有 = スライス単位の抽選、R-S23）。B（実 EID）には P を掛けない。
【出力】 {"A": x patch, "U": P(x) patch, "B": y patch, "P": (3,) の params, "A_paths", "B_paths"}。すべて正規化 [−1, 1]。
【正規化】 hu = stored − 1400 → clip(−1400, 4096) → [−1, 1]。P は clip の前（HU）で掛ける。
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

from degrade import apply_degrade, degrade_config, params_to_array, sample_params

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


def stored_to_hu(stored, offset):
    return stored.astype(np.float32) - np.float32(offset)


def hu_to_norm(hu, hu_min, hu_max):
    hu = np.clip(hu, hu_min, hu_max)
    norm = (hu - hu_min) / (hu_max - hu_min)
    return (norm * 2.0 - 1.0).astype(np.float32)


def normalize(stored, offset, hu_min, hu_max):
    return hu_to_norm(stored_to_hu(stored, offset), hu_min, hu_max)


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
    """R = 出力 − 入力 を uint16 で保存する規約: R_stored = (g − a) + 32768。"""
    r = g_stored.astype(np.int32) - a_stored.astype(np.int32) + R_ZERO
    return np.clip(r, 0, 65535).astype(np.uint16)


@lru_cache(maxsize=64)
def _load_hu(path, offset):
    return stored_to_hu(read_stored(path), offset)


# ---------------------------------------------------------------------------
# 症例 → スライス一覧（stage1 CaseIndex の複製）
# ---------------------------------------------------------------------------
def load_eval_only():
    """configs/eval_only.yaml の評価専用症例名（set）。ファイルが無い・形が違えばエラー。"""
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
    raise RuntimeError("configs/eval_only.yaml が見つかりません（リポジトリ直下の configs/ に置く）")


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
        self.eval_only_dropped = []
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
                raise RuntimeError(f"{root} に無い症例が指定されています: {missing}")
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
# Sampler（stage1_v2 と同じ）
# ---------------------------------------------------------------------------
def _grid_positions(img_hw, patch, stride):
    h, w = img_hw
    return list(range(0, h - patch + 1, stride)), list(range(0, w - patch + 1, stride))


class RandomSampler:
    """完全ランダム。A / B とも 症例一様 → スライス一様 → 位置一様。k = patches_per_slice > 1 のとき、引いたスライスの組を k サンプル使い回す。"""

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
class ProjectionDataset(Dataset):
    def __init__(self, train, machine, cache_dir=None, run_dir=None):
        self.cfg = train
        self.offset, self.hu_min, self.hu_max = train["data.hu_offset"], train["data.hu_min"], train["data.hu_max"]
        self.patch = train["data.patch_size"]
        self.pcfg = degrade_config(train)
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
        else:
            raise ValueError(sampling)
        self._u_cache = None  # (a_path, u_hu, params): 同じスライスの連続 patch は同じ 1 回の P を共有
        self._np_rng = None
        print(f"[ProjectionDataset] A=PCD512: {self.idx_a.n_cases} cases / {self.idx_a.n_slices} slices | "
              f"B=EID512: {self.idx_b.n_cases} cases / {self.idx_b.n_slices} slices | image {self.hw}, patch {self.patch}, sampling {sampling}, len {len(self)}"
              + (f", patches_per_slice {train['data.patches_per_slice']}" if sampling == "random" else "")
              + f" | P {'on' if self.pcfg['enabled'] else 'off'} mode {self.pcfg['mode']}"
              + (" (P_θ はモデル側、models/degrade_net.py)" if self.pcfg['mode'] == 'learned' else
                 (f" bilateral color {self.pcfg['bil_color']} HU space {self.pcfg['bil_space']} px" if self.pcfg['mode'] == 'bilateral_blur' else "")
                 + f" blur {self.pcfg['blur']} px, noise {self.pcfg['noise']} HU, corr {self.pcfg['corr']} px"))

    def info(self):
        return {"image_hw": list(self.hw), "patch": self.patch, "sampling": self.cfg["data.sampling"],
                "a_domain": "PCD512", "a_cases": self.idx_a.n_cases, "a_slices": self.idx_a.n_slices,
                "b_domain": "EID512", "b_cases": self.idx_b.n_cases, "b_slices": self.idx_b.n_slices,
                "use_eval_only_cases": self.cfg["data.use_eval_only_cases"],
                "degrade": {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.pcfg.items()},
                "fixed_slice": str(self.fixed_path), "eid_slice": str(self.eid_ref_path)}

    # ------------------------------------------------------------------ 乱数
    def _rng(self, index):
        if self.cfg["data.serial_batches"]:
            return random.Random(self.cfg["data.patch_seed"] * 1_000_003 + index)
        return random

    def _nprng(self, index):
        """P 用の numpy Generator。serial_batches なら index 決定的、そうでなければ worker ごとの 1 本（torch の worker seed から派生）。"""
        if self.cfg["data.serial_batches"]:
            return np.random.default_rng(self.cfg["data.patch_seed"] * 1_000_003 + 7 + index)
        if self._np_rng is None:
            self._np_rng = np.random.default_rng(torch.initial_seed() % (2**32))
        return self._np_rng

    # ------------------------------------------------------------------ P
    def degrade_full(self, hu, rng, params=None):
        """フル画像の HU に P を 1 回掛ける。params を省略すればここで引く。戻り値 (u_hu, params)。"""
        if params is None:
            params = sample_params(self.pcfg, rng)
        return apply_degrade(hu, params, self.pcfg, rng), params

    def _u_full(self, a_path, index):
        """serial_batches では P の抽選と cache をグループ（index // patches_per_slice）単位にする
        （batch_size が patches_per_slice の倍数でないとグループが worker をまたぎ、先頭 index 依存の seed では同じスライスの patch が別の P になる）。"""
        if self.cfg["data.serial_batches"]:
            k = self.cfg["data.patches_per_slice"]
            key, seed_index = (a_path, index // k), (index // k) * k
        else:
            key, seed_index = a_path, index
        if self._u_cache is not None and self._u_cache[0] == key:
            return self._u_cache[1], self._u_cache[2]
        u, params = self.degrade_full(_load_hu(a_path, self.offset), self._nprng(seed_index))
        self._u_cache = (key, u, params)
        return u, params

    # ------------------------------------------------------------------ patch
    def _crop(self, hu, pos):
        if hu.shape != self.hw:
            raise ValueError(f"画像サイズが揃っていません: {hu.shape} != {self.hw}")
        y, x = pos
        return torch.from_numpy(hu_to_norm(hu[y : y + self.patch, x : x + self.patch], self.hu_min, self.hu_max)).unsqueeze(0)

    def _item(self, a_path, a_pos, b_path, b_pos, index):
        x_hu = _load_hu(a_path, self.offset)
        u_hu, params = self._u_full(a_path, index)
        y_hu = _load_hu(b_path, self.offset)
        return {"A": self._crop(x_hu, a_pos), "U": self._crop(u_hu, a_pos), "B": self._crop(y_hu, b_pos),
                "P": torch.from_numpy(params_to_array(params)), "A_paths": a_path, "B_paths": b_path}

    def __getitem__(self, index):
        k = self.cfg["data.patches_per_slice"]
        group = (index // k) if (self.cfg["data.serial_batches"] and k > 1) else None
        return self._item(*self.sampler.sample(self._rng(index), index, group), index)

    def __len__(self):
        return self.cfg["data.samples_per_epoch"] if self.sampler.fixed_len is None else self.sampler.fixed_len

    # ------------------------------------------------------------------ 表示用の固定集合
    def fixed_batch(self, n):
        if n <= 0:
            return None
        items = [self.sampler.sample(random.Random(self.cfg["data.patch_seed"] * 1_000_003 + i), i, group=-1 - i) for i in range(n)]
        out = []
        for i, it in enumerate(items):
            self._u_cache = None  # 固定集合は毎回同じ乱数で P を引く
            rng = np.random.default_rng(self.cfg["data.patch_seed"] * 1_000_003 + 7 + i)
            a_path, a_pos, b_path, b_pos = it
            u_hu, params = self.degrade_full(_load_hu(a_path, self.offset), rng)
            out.append({"A": self._crop(_load_hu(a_path, self.offset), a_pos), "U": self._crop(u_hu, a_pos), "B": self._crop(_load_hu(b_path, self.offset), b_pos),
                        "P": torch.from_numpy(params_to_array(params))})
        self._u_cache = None
        return {k: torch.stack([o[k] for o in out]) for k in ("A", "U", "B", "P")}

    def full_slices(self, epoch):
        """固定 PCD 1 枚 + ランダム PCD。各スライスに P を 1 回（epoch で変わる seed）。固定スライスは n_draws 回引き直した U_draws も返す（標準偏差画像用）。"""
        rng_py = random.Random(self.cfg["data.patch_seed"] * 1_000_003 + int(epoch))
        n_rand = self.cfg["log.n_full_random"]
        pool = [p for p in self.idx_a.all_paths if Path(p) != self.fixed_path]
        randoms = rng_py.sample(pool, min(n_rand, len(pool)))
        paths = [str(self.fixed_path)] + randoms
        rng = np.random.default_rng(self.cfg["data.patch_seed"] * 1_000_003 + 101 + int(epoch))
        A, U, P = [], [], []
        for p in paths:
            hu = _load_hu(p, self.offset)
            u, params = self.degrade_full(hu, rng)
            A.append(torch.from_numpy(hu_to_norm(hu, self.hu_min, self.hu_max)).unsqueeze(0))
            U.append(torch.from_numpy(hu_to_norm(u, self.hu_min, self.hu_max)).unsqueeze(0))
            P.append(torch.from_numpy(params_to_array(params)))
        draws = []
        n_draws = self.cfg["log.n_draws"]
        if n_draws > 1:
            hu = _load_hu(paths[0], self.offset)
            for _ in range(n_draws):
                u, _ = self.degrade_full(hu, rng)
                draws.append(torch.from_numpy(hu_to_norm(u, self.hu_min, self.hu_max)).unsqueeze(0))
        return {"paths": paths, "kinds": ["fixed"] + ["random"] * len(randoms),
                "A": torch.stack(A), "U": torch.stack(U), "P": torch.stack(P),
                "U_draws": torch.stack(draws) if draws else None}

    def eid_reference(self):
        hu = _load_hu(str(self.eid_ref_path), self.offset)
        return {"path": str(self.eid_ref_path), "B": torch.from_numpy(hu_to_norm(hu, self.hu_min, self.hu_max)).unsqueeze(0).unsqueeze(0)}
