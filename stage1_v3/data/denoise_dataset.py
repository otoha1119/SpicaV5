"""stage1_v3/data/denoise_dataset.py — v4 の Dn 学習用データ（PCD512 だけ。正解なし）。

data/dataset.py の CaseIndex（症例列挙、configs/eval_only.yaml による評価専用症例の除外）・読み込み・HU 正規化をそのまま使う。
  __getitem__ : 症例一様 → スライス一様 → 位置一様で patch_size の patch を切る。patches_per_slice 枚は同じスライスから連続して切る（RandomSampler と同じ）。
                戻り値 {"Y": (1,P,P) 正規化 [−1,1], "path"}
  fixed_slices(n): 症例列挙から等間隔に n 枚（preview と check/* 用。学習中ずっと同じ）。{"paths", "hu": [HU の ndarray], "Y": (n,1,H,W)}
"""

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from data.dataset import CaseIndex, _load_hu, hu_to_norm, read_stored


def evenly_spaced(paths, n):
    """paths から等間隔に n 本（n ≥ len なら全部）。"""
    if n >= len(paths):
        return list(paths)
    pos = np.linspace(0, len(paths) - 1, n).round().astype(int)
    return [paths[i] for i in sorted(set(pos.tolist()))]


class DenoiseDataset(Dataset):
    def __init__(self, dn, cache_dir=None):
        self.cfg = dn
        self.offset, self.hu_min, self.hu_max = dn["data.hu_offset"], dn["data.hu_min"], dn["data.hu_max"]
        self.patch = dn["data.patch_size"]
        self.k = dn["data.patches_per_slice"]
        self.root = Path(dn["data.input_dir"])
        self.idx = CaseIndex(self.root, cache_dir, drop_eval_only=not dn["data.use_eval_only_cases"])
        self.hw = tuple(read_stored(self.idx.all_paths[0]).shape)
        if self.patch > min(self.hw):
            raise ValueError(f"patch_size {self.patch} が画像サイズ {self.hw} を超えています")
        self._cur, self._left = None, 0
        print(f"[DenoiseDataset] PCD512: {self.idx.n_cases} cases / {self.idx.n_slices} slices | image {self.hw}, patch {self.patch}, patches_per_slice {self.k}, len {len(self)}")

    def info(self):
        return {"image_hw": list(self.hw), "patch": self.patch, "input_dir": str(self.root), "cases": self.idx.n_cases, "slices": self.idx.n_slices,
                "use_eval_only_cases": self.cfg["data.use_eval_only_cases"], "eval_only_dropped": list(self.idx.eval_only_dropped)}

    def __len__(self):
        return self.cfg["data.samples_per_epoch"]

    def __getitem__(self, index):
        if self._left <= 0:  # worker ごとの乱数は DataLoader が seed する（ProjectionDataset の非 serial と同じ）
            self._cur, self._left = self.idx.random_slice(random), self.k
        self._left -= 1
        hu = _load_hu(self._cur, self.offset)
        y = random.randrange(0, self.hw[0] - self.patch + 1)
        x = random.randrange(0, self.hw[1] - self.patch + 1)
        patch = hu_to_norm(hu[y : y + self.patch, x : x + self.patch], self.hu_min, self.hu_max)
        return {"Y": torch.from_numpy(patch).unsqueeze(0), "path": self._cur}

    def fixed_slices(self, n):
        paths = evenly_spaced(self.idx.all_paths, n)
        hus = [_load_hu(p, self.offset) for p in paths]
        return {"paths": paths, "hu": hus, "Y": torch.stack([torch.from_numpy(hu_to_norm(h, self.hu_min, self.hu_max)).unsqueeze(0) for h in hus])}
