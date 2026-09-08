"""stage2/data/ct_io.py — 16bit 1ch PNG の読み書き・HU 正規化・症例列挙。
stage1/data/ct_dataset.py（read_stored / normalize / denormalize / write_png / CaseIndex）から 2026-09-08 に複製（Stage 間で import しない）。
値の規約と演算順序は Stage 1 と同一（stored = HU + 1400 → clip(−1400, 4096) → [0,1] → [−1,1]）。
"""

import hashlib
import json
import os
import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


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


def normalize(stored, offset, hu_min, hu_max):
    """stored (uint16) → [−1,1] float32。Stage 1 と同じ演算順序。"""
    hu = stored.astype(np.float32) - offset
    hu = np.clip(hu, hu_min, hu_max)
    norm = (hu - hu_min) / (hu_max - hu_min)  # [0, 1]
    return norm * 2.0 - 1.0  # [−1, 1]


def denormalize(norm, offset, hu_min, hu_max):
    """[−1,1] float → stored uint16。normalize の逆（最後に 1 回だけ丸め・クリップ）。"""
    n01 = np.clip((np.asarray(norm, dtype=np.float32) + 1.0) / 2.0, 0.0, 1.0)
    hu = n01 * (hu_max - hu_min) + hu_min
    stored = hu + offset
    return np.clip(np.round(stored), 0, 65535).astype(np.uint16)


def hu_per_unit(hu_min, hu_max):
    """正規化空間の 1.0 が何 HU か（[−1,1] の幅 2 = hu_max − hu_min）。損失を HU 換算するときに使う。"""
    return (hu_max - hu_min) / 2.0


def write_png(path, arr):
    """cv2.imwrite の戻り値を検査して書く。失敗を黙って完了扱いにしない。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        ok = cv2.imwrite(str(path), arr)
    except cv2.error as e:
        raise IOError(f"PNG を書き込めません: {path}: {e}") from e
    if not ok:
        raise IOError(f"PNG を書き込めません: {path}")


@lru_cache(maxsize=32)
def load_normalized(path, offset, hu_min, hu_max):
    """読込 + 正規化をプロセス内で小さくキャッシュする（同じスライスから複数 patch を切るときの再デコード回避）。1024² float32 = 4 MB/枚。"""
    return normalize(read_stored(path), offset, hu_min, hu_max)


def list_case_names(root):
    """<root> 直下の症例フォルダ名だけを返す（ファイルは走査しない。起動器の事前検査用）。'.' 始まりは無視。"""
    root = Path(root)
    if not root.is_dir():
        raise RuntimeError(f"データディレクトリが存在しません: {root}")
    with os.scandir(root) as it:
        return sorted(e.name for e in it if e.is_dir() and not e.name.startswith("."))


class CaseIndex:
    """<root>/<症例>/<slice>.png を {症例: [パス, ...]} に整理する。症例フォルダ直下にサブフォルダがあれば再帰的に拾う。
    '.' 始まりのファイル・フォルダは無視する（macOS の AppleDouble "._xxx.png" など）。
    cache_dir を渡すと列挙結果を JSON に保存し、次回は「走査した全ディレクトリの mtime が一致」すれば再走査しない。"""

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
        return sum(len(v) for v in self.slices.values())
