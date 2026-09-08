"""stage2/data/pair_dataset.py — Stage 2 のペアローダ（EID-like1024 入力 ↔ PCD1024 教師）。

設計: docs/plans/20260908_stage2-unet-implementation-spec.md §4

【入力データ】 machines.yaml の eidlike1024_dir（入力。bash start2.sh dataset の出力）と pcd1024_dir（教師）。各直下に <症例>/<slice>.png（1ch uint16、一辺 1024）
【対応付け】 症例名（フォルダ名）とスライス名（ファイル名）が同じものをペアにする。症例ごとに両方にある名前の共通部分だけ使う
  （PCD1024_v1 は PCD-006 / 011 / 015 で末尾の枚数が PCD512 と違う → 先頭から min まで。切り捨てた枚数は dataset_info に残す）
【症例分割】 train.yaml の data.train_cases / val_cases / test_cases（ユーザー決定 2026-09-08）。
  ・train / val の症例は両方のディレクトリに存在すること（無ければエラー）
  ・test の症例は**一切読まない**（ディスクに無くてもよい）
  ・ディスク上（入力側・教師側）の症例は必ずどれかのリストに入っていること（未割当があればエラー = 意図せず学習に混ざるのを防ぐ）
【サンプリング】 Stage 1 のケース 1 と同じ: 症例一様 → スライス一様 → 位置一様（1 サンプルごと）。入力と教師は**同じスライス・同じ位置**の patch。
  augmentation なし（ユーザー決定: データが論文より多い）。epoch 長は samples_per_epoch
【正規化】 Stage 1 と同じ [−1,1]（data/ct_io.py）
【検証】 val_slices(): val 症例のフル 1024 ペアを、症例ごとに val_max_slices_per_case 枚まで等間隔に間引いて返す（毎 epoch 同じ集合）
"""

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from data.ct_io import CaseIndex, list_case_names, load_normalized, read_stored


def check_split_on_disk(train, machine):
    """train.yaml の症例リストとディスク（eidlike1024_dir / pcd1024_dir 直下の症例フォルダ）を照合する。run_train.py が run ディレクトリを作る前に呼ぶ。
    ・ディスク上の症例は train / val / test のどれかに入っていること（未割当 = 意図せず学習に混ざるのを防ぐ）
    ・train / val の症例は両方のディレクトリに存在すること。test はディスクに無くてもよい
    戻り値は {"input": [症例...], "pcd": [症例...]}（記録用）。問題があれば RuntimeError。"""
    train_cases, val_cases, test_cases = train["data.train_cases"], train["data.val_cases"], train["data.test_cases"]
    on_in, on_pcd = list_case_names(machine["eidlike1024_dir"]), list_case_names(machine["pcd1024_dir"])
    assigned = set(train_cases) | set(val_cases) | set(test_cases)
    unassigned = sorted((set(on_in) | set(on_pcd)) - assigned)
    if unassigned:
        raise RuntimeError(f"train.yaml のどのリストにも無い症例がディスクにあります（train_cases / val_cases / test_cases のどれかに入れること）: {unassigned}")
    for kind, cases in (("train_cases", train_cases), ("val_cases", val_cases)):
        missing = [c for c in cases if c not in on_in or c not in on_pcd]
        if missing:
            raise RuntimeError(f"{kind} の症例が入力（{machine['eidlike1024_dir']}）か教師（{machine['pcd1024_dir']}）にありません: {missing}")
    return {"input": on_in, "pcd": on_pcd}


def _by_name(paths):
    """[path, ...] → {ファイル名: path}。同じ名前が 2 回出たらエラー（サブフォルダで重複した場合）。"""
    d = {}
    for p in paths:
        name = Path(p).name
        if name in d:
            raise RuntimeError(f"同じスライス名が 2 つあります: {d[name]} と {p}")
        d[name] = p
    return d


def build_pairs(idx_in, idx_pcd, cases):
    """症例ごとに (入力 path, 教師 path) のリストを名前で対応付ける。戻り値 ({case: [(in, pcd), ...]}, {case: {"input": n, "pcd": n, "pair": n}})。"""
    pairs, info = {}, {}
    for c in cases:
        a, b = _by_name(idx_in.slices[c]), _by_name(idx_pcd.slices[c])
        common = sorted(set(a) & set(b))
        if not common:
            raise RuntimeError(f"症例 {c}: 入力と教師で同じ名前のスライスがありません（入力 {len(a)} 枚 / 教師 {len(b)} 枚）")
        pairs[c] = [(a[n], b[n]) for n in common]
        info[c] = {"input": len(a), "pcd": len(b), "pair": len(common), "dropped": len(a) + len(b) - 2 * len(common)}
    return pairs, info


class PairDataset(Dataset):
    def __init__(self, train, mode, machine, cache_dir):
        """train / mode / machine は run_train.py が検証した実効値（平坦 dict）。cache_dir は CaseIndex のキャッシュ置き場。"""
        self.hu = (train["data.hu_offset"], train["data.hu_min"], train["data.hu_max"])
        self.patch = train["data.patch_size"]
        self.samples_per_epoch = train["data.samples_per_epoch"]
        self.patch_seed = train["data.patch_seed"]
        self.serial = mode["serial_batches"]
        self.val_max = train["log.val_max_slices_per_case"]
        train_cases, val_cases, test_cases = train["data.train_cases"], train["data.val_cases"], train["data.test_cases"]

        check_split_on_disk(train, machine)  # 起動器でも検査済みだが、train.py 単独でも同じ規則で止まるように
        idx_in = CaseIndex(machine["eidlike1024_dir"], cache_dir)
        idx_pcd = CaseIndex(machine["pcd1024_dir"], cache_dir)
        for kind, cases in (("train_cases", train_cases), ("val_cases", val_cases)):
            empty = [c for c in cases if c not in idx_in.slices or c not in idx_pcd.slices]
            if empty:
                raise RuntimeError(f"{kind} の症例フォルダに PNG がありません: {empty}")

        self.train_cases, self.val_cases, self.test_cases = list(train_cases), list(val_cases), list(test_cases)
        self.pairs_train, info_train = build_pairs(idx_in, idx_pcd, self.train_cases)
        self.pairs_val, info_val = build_pairs(idx_in, idx_pcd, self.val_cases)
        self.info = {"train": info_train, "val": info_val}

        first_in, first_pcd = self.pairs_train[self.train_cases[0]][0]
        a, b = read_stored(first_in), read_stored(first_pcd)
        if a.shape != b.shape:
            raise ValueError(f"入力と教師の画像サイズが違います: {a.shape} != {b.shape}: {first_in} / {first_pcd}")
        self.img_hw = a.shape
        if self.patch > min(self.img_hw):
            raise ValueError(f"patch_size {self.patch} が画像サイズ {self.img_hw} を超えています")
        n_tr = sum(len(v) for v in self.pairs_train.values())
        n_va = sum(len(v) for v in self.pairs_val.values())
        dropped = sum(i["dropped"] for i in list(info_train.values()) + list(info_val.values()))
        print(f"[PairDataset] train {len(self.train_cases)} cases / {n_tr} pairs, val {len(self.val_cases)} cases / {n_va} pairs, "
              f"test {len(self.test_cases)} cases（読まない）, image {self.img_hw}, patch {self.patch}, len {len(self)}"
              + (f", 名前が片側にしか無いスライス {dropped} 枚は不使用" if dropped else ""))

    # --- 乱数 ---
    def _rng(self, index):
        if self.serial:
            return random.Random(self.patch_seed * 1_000_003 + index)
        return random  # DataLoader が worker ごとに seed する

    # --- 切り出し ---
    def _patch(self, path, pos):
        img = load_normalized(path, *self.hu)
        if img.shape != self.img_hw:
            raise ValueError(f"画像サイズが揃っていません: {img.shape} != {self.img_hw}: {path}")
        y, x = pos
        return torch.from_numpy(np.ascontiguousarray(img[y : y + self.patch, x : x + self.patch])).unsqueeze(0)

    def __getitem__(self, index):
        rng = self._rng(index)
        case = rng.choice(self.train_cases)
        in_path, pcd_path = rng.choice(self.pairs_train[case])
        h, w = self.img_hw
        pos = (rng.randrange(h - self.patch + 1), rng.randrange(w - self.patch + 1))
        return {"input": self._patch(in_path, pos), "target": self._patch(pcd_path, pos), "path": pcd_path, "y": pos[0], "x": pos[1]}

    def __len__(self):
        return self.samples_per_epoch

    # --- 検証 ---
    def val_slices(self):
        """val 症例のフル 1024 ペア [(case, in_path, pcd_path), ...]。症例ごとに val_max_slices_per_case 枚まで等間隔に間引く（0 で全部）。毎 epoch 同じ集合。"""
        out = []
        for c in self.val_cases:
            pairs = self.pairs_val[c]
            if self.val_max and len(pairs) > self.val_max:
                idx = np.linspace(0, len(pairs) - 1, self.val_max).round().astype(int)
                pairs = [pairs[i] for i in idx]
            out += [(c, a, b) for a, b in pairs]
        return out

    def load_full(self, path):
        """フル画像を (1,1,H,W) の正規化 tensor で返す（検証用）。"""
        return torch.from_numpy(load_normalized(path, *self.hu)).unsqueeze(0).unsqueeze(0)
