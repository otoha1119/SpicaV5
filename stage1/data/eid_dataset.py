"""stage1/data/eid_dataset.py — 実 EID の学習パッチ（設計書 B・I-3）と、val の固定パッチ。

  分割   : data.split_file（configs/splits/eid_v5.yaml）の train だけで学習、val は損失の監視、test は読まない。
           評価専用（configs/eval_only.yaml）は CaseIndex(drop_eval_only=True) で除外。
           分割ファイルに無い症例は data.unlisted_cases（train = 学習に入れる | error）。分割ファイルにあってディスクに無い train / val の症例はエラー。
  引き方 : 症例を一様 → スライスを一様 → 位置を一様。1 回読んだスライスから patches_per_slice 個を連続して切る（位置は毎回独立）。
  切り方 : スライスを margin 画素 reflect で広げ、(patch_size + 2·margin) 四方を切る。GPU 側でぼかしてから中央 patch_size を使う。
  水増し : なし。
"""

import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import IterableDataset, get_worker_info

from data.ct_io import CaseIndex, read_stored, stored_to_x

REPO = Path(__file__).resolve().parents[2]  # リポジトリ直下（このファイルは <repo>/stage1/data/）


def split_path(train):
    p = Path(train["data.split_file"])
    return p if p.is_absolute() else REPO / p


def resolve_split(train, machine, cache_dir):
    """分割ファイルとディスクを突き合わせる。戻り値 dict: index（train の CaseIndex）, train / val / test / unlisted（症例名）, val_index, eval_only_dropped。
    食い違いは RuntimeError（起動器が run フォルダを作る前に呼ぶ）。"""
    sp = split_path(train)
    if not sp.is_file():
        raise RuntimeError(f"症例分割ファイルがありません: {sp}")
    with open(sp, encoding="utf-8") as f:
        split = yaml.safe_load(f)
    if not isinstance(split, dict) or any(not isinstance(split.get(k), list) or not all(isinstance(c, str) for c in split[k]) for k in ("train", "val", "test")):
        raise RuntimeError(f"{sp}: train / val / test の症例名のリストが必要です")
    tr, va, te = split["train"], split["val"], split["test"]
    dup = (set(tr) & set(va)) | (set(tr) & set(te)) | (set(va) & set(te))
    if dup:
        raise RuntimeError(f"{sp}: 複数の分割に入っている症例があります: {sorted(dup)[:5]}")
    full = CaseIndex(machine["eid_dir"], cache_dir=cache_dir, drop_eval_only=True)
    disk = set(full.disk_cases)
    missing = [c for c in tr + va if c not in disk]
    if missing:
        raise RuntimeError(f"{sp} の train / val にあって {machine['eid_dir']} に無い症例があります: {missing[:10]}（{len(missing)} 症例）")
    unlisted = sorted(disk - set(tr) - set(va) - set(te))
    if unlisted and train["data.unlisted_cases"] == "error":
        raise RuntimeError(f"分割ファイルに無い症例があります（data.unlisted_cases: error）: {unlisted[:10]}（{len(unlisted)} 症例）")
    if unlisted:
        print(f"[data] 分割ファイルに無い {len(unlisted)} 症例を学習に入れる（data.unlisted_cases: train）: {unlisted[:10]}{' ...' if len(unlisted) > 10 else ''}")
    train_cases = sorted(set(tr) | set(unlisted))
    return {
        "split_file": str(sp), "train": train_cases, "val": sorted(va), "test": sorted(te), "unlisted": unlisted,
        "eval_only_dropped": full.eval_only_dropped,
        "train_slices": {c: full.slices[c] for c in train_cases}, "val_slices": {c: full.slices[c] for c in sorted(va)},
    }


def padded_x(path, hu, margin):
    """スライスを読み、x にして margin 画素 reflect で広げる。(H + 2m, W + 2m) float32。"""
    x = stored_to_x(read_stored(path), hu)
    return np.pad(x, margin, mode="reflect") if margin > 0 else x


def cut(xp, y, x0, size):
    return torch.from_numpy(np.ascontiguousarray(xp[y:y + size, x0:x0 + size]))[None]


class EidPatchStream(IterableDataset):
    """学習パッチを無限に返す。各 worker が別の乱数列を持ち、1 スライスから patches_per_slice 個を連続して返す
    （DataLoader の worker は 1 batch を丸ごと組むので、batch 32 = 8 スライス × 4 個）。返す形 (1, S, S)、S = patch_size + 2·margin。"""

    def __init__(self, slices, train, hu, start_step):
        self.cases = sorted(slices)
        self.slices = slices
        self.patch, self.margin, self.k = train["data.patch_size"], train["data.margin"], train["data.patches_per_slice"]
        self.hu = hu
        self.seed, self.start_step = train["optim.seed"], int(start_step)

    def __iter__(self):
        info = get_worker_info()
        wid = info.id if info is not None else 0
        rng = random.Random(f"{self.seed}/{wid}/{self.start_step}")  # 再開のたびに別の列（同じパッチの繰り返しを避ける）
        size = self.patch + 2 * self.margin
        while True:
            path = rng.choice(self.slices[rng.choice(self.cases)])
            xp = padded_x(path, self.hu, self.margin)
            h, w = xp.shape
            if h < size or w < size:
                raise RuntimeError(f"スライスが patch_size {self.patch} より小さい: {path} {(h - 2 * self.margin, w - 2 * self.margin)}")
            for _ in range(self.k):
                yield cut(xp, rng.randint(0, h - size), rng.randint(0, w - size), size)


def worker_init(_worker_id):
    """worker は SIGTERM を無視する（監視ツールは train.py の全プロセスに SIGTERM を送る。worker が先に死ぬと main が latest を保存する前に DataLoader のエラーで落ちる。
    worker は main が終わるときに DataLoader が片付ける）。OpenCV のスレッドは 1 に（worker の数だけ並列なので）。"""
    import signal
    import cv2
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    cv2.setNumThreads(1)


def fixed_val_patches(val_slices, train, hu):
    """val の症例の固定パッチ (n, 1, S, S)。症例を順に回し、スライスと位置は log.val_seed で決める。"""
    n, seed = train["log.val_patches"], train["log.val_seed"]
    patch, margin = train["data.patch_size"], train["data.margin"]
    size = patch + 2 * margin
    rng = random.Random(f"val/{seed}")
    cases = sorted(val_slices)
    rng.shuffle(cases)
    out, names = [], []
    for i in range(n):
        path = rng.choice(val_slices[cases[i % len(cases)]])
        xp = padded_x(path, hu, margin)
        h, w = xp.shape
        if h < size or w < size:
            raise RuntimeError(f"スライスが patch_size {patch} より小さい: {path}")
        out.append(cut(xp, rng.randint(0, h - size), rng.randint(0, w - size), size))
        names.append(Path(path).name)
    return torch.stack(out), names


def load_full_x(path, hu):
    """1 枚を x の (1, 1, H, W) tensor で読む（途中経過・描画用）。"""
    return torch.from_numpy(stored_to_x(read_stored(path), hu))[None, None]
