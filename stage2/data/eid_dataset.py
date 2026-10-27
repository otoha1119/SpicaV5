"""stage2/data/eid_dataset.py — 実 EID 枝（mode.yaml real_eid_branch: true）の実 EID512 小片ローダ。2026-10-13。

設計: docs/plans/20260908_stage2-unet-implementation-spec.md §9

【入力データ】 machines.yaml の eid_dir（EID_v5。<症例>/<slice>.png、1ch uint16、一辺 512）。教師は無い（D の本物側はペア枝の教師 PCD1024 の patch。ここでは読まない）
【症例】 train.yaml adv.eid_split_file（train / val / test の症例名リスト。Stage 1 の bash start.sh split が書く configs/splits/eid_v5.yaml）の **train だけ**。
  val / test は読まない（例外は監視用 log.eid_slice の 1 枚。その症例が train に入っていればエラー）。eid_dir 直下の症例は全部どれかのリストに入っていること（未割当があればエラー。Stage 1 の sr と同じ規則）。
  検査は check_eid_split_on_disk（run_train.py が run ディレクトリを作る前に呼ぶ。train.py 単独でも同じ規則で止まる）
【サンプリング】 PairDataset と同じ: 症例一様 → スライス一様 → 位置一様（512 グリッド上で adv.crop_eid 四方）。
  1 回読んだスライスから adv.patches_per_slice_eid 小片を連続して切る（data.patches_per_slice と同じ仕組み。位置は小片ごとに独立）。serial_batches なら index 決定的
【補間】 小片だけを run と同じ方式（eidlike1024_dir/manifest.yaml の scale / interp、data/ct_io.upsample）で ×scale する。512 全体は補間しない（§9.2「実 EID は 1024 のファイルを作らない」）。
  ただし補間は近傍画素を使う（bicubic 4 tap、lanczos 8 tap）ので、小片の縁だけ補間すると「512 全体を補間してから切る」（推論・学習中の実 EID テストと同じ経路）と縁の数画素が変わり、
  D がその縁を手掛かりにしかねない。そこで MARGIN 画素の余白を付けて切り、補間後に余白分（MARGIN × scale）を落とす → 内側は 512 全体を補間した結果と同一（画像の端で余白が取れない所は
  cv2.resize の端の扱いも同じなので一致する）。この等価性は smoke で確認した（2026-10-13）
【正規化】 PairDataset と同じ HU 窓（data/ct_io.normalize）
【長さ】 1 epoch のペア枝の step 数 × adv.batch_eid（ペアの loader と 1 対 1 で並走する。train.py は足りなければ cycle する）
"""

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from configs.schema import ConfigError, load_eid_split
from data.ct_io import CaseIndex, INTERP_FLAGS, list_case_names, normalize, read_stored, upsample

MARGIN = 4  # 補間の余白（512 上の画素）。lanczos（8 tap）でも足りる


def check_eid_crop(train, mode, upsample_cfg):
    """adv.crop_eid × scale（manifest の方式）が 2^n_pool の倍数か（U-Net に入る大きさか）。run_train.py が run ディレクトリを作る前にも呼ぶ。"""
    scale = upsample_cfg[0]
    m = 2 ** mode["n_pool"]
    out = train["adv.crop_eid"] * scale
    if out % m:
        raise RuntimeError(f"adv.crop_eid {train['adv.crop_eid']} × scale {scale} = {out} が 2^n_pool = {m} の倍数ではありません（U-Net に入らない）")
    return out


def check_eid_split_on_disk(train, machine):
    """adv.eid_split_file（絶対パス。schema.resolve_eid_split_file 済み）を読み、eid_dir の症例と照合する。戻り値 {train, val, test}（検査済み）。問題があれば RuntimeError。
    ・eid_dir 直下の症例は train / val / test のどれかに入っていること（未割当 = 意図せず学習に混ざるのを防ぐ）
    ・train の症例は eid_dir に存在すること（val / test は無くてもよい = 読まない）
    ・log.eid_slice（監視用）の症例は train に入っていないこと"""
    try:
        split = load_eid_split(train["adv.eid_split_file"])
    except ConfigError as e:
        raise RuntimeError(str(e))
    on_disk = list_case_names(machine["eid_dir"], drop_eval_only=True)  # 評価専用（configs/eval_only.yaml）は照合の対象外
    assigned = set(split["train"]) | set(split["val"]) | set(split["test"])
    unassigned = sorted(set(on_disk) - assigned)
    if unassigned:
        raise RuntimeError(f"{machine['eid_dir']} の症例が adv.eid_split_file のどのリストにも入っていません（分割を作り直す: bash start.sh split）: {unassigned[:10]}{' ...' if len(unassigned) > 10 else ''}（{len(unassigned)} 症例）")
    missing = [c for c in split["train"] if c not in on_disk]
    if missing:
        raise RuntimeError(f"adv.eid_split_file の train の症例が {machine['eid_dir']} にありません: {missing[:10]}{' ...' if len(missing) > 10 else ''}")
    eid_case = Path(train["log.eid_slice"]).parts[0]
    if eid_case in split["train"]:
        raise RuntimeError(f"監視用 log.eid_slice の症例 {eid_case} が adv.eid_split_file の train に入っています（学習に入れない。val か test にすること）: {train['adv.eid_split_file']}")
    return split


def crop_upsampled(stored, y, x, crop, scale, interp, margin=MARGIN):
    """stored (H,W) uint16 の (y, x) から crop 四方を、余白付きで補間してから余白を落として返す（(crop·scale, crop·scale) uint16）。
    結果は「stored 全体を upsample してから [y·scale : (y+crop)·scale, …] を切る」と同一。"""
    h, w = stored.shape
    y0, x0 = max(y - margin, 0), max(x - margin, 0)
    y1, x1 = min(y + crop + margin, h), min(x + crop + margin, w)
    up = upsample(np.ascontiguousarray(stored[y0:y1, x0:x1]), scale, interp)
    oy, ox = (y - y0) * scale, (x - x0) * scale
    return up[oy : oy + crop * scale, ox : ox + crop * scale]


class RealEIDDataset(Dataset):
    def __init__(self, train, mode, machine, upsample_cfg, n_steps, cache_dir):
        """train / mode / machine は実効値（平坦 dict）。upsample_cfg = (scale, interp)（PairDataset.upsample_cfg = manifest の方式）。n_steps = 1 epoch のペア枝の step 数。"""
        self.hu = (train["data.hu_offset"], train["data.hu_min"], train["data.hu_max"])
        self.crop = train["adv.crop_eid"]
        self.k = train["adv.patches_per_slice_eid"]
        self.batch = train["adv.batch_eid"]
        self.n_steps = n_steps
        self.scale, self.interp = upsample_cfg
        if self.interp not in INTERP_FLAGS:
            raise ValueError(self.interp)
        self.serial = mode["serial_batches"]
        self.patch_seed = train["data.patch_seed"]
        self._left, self._cur, self._cur_group = 0, None, None  # worker ごとの状態（PairDataset と同じ）
        out = check_eid_crop(train, mode, upsample_cfg)

        self.split = check_eid_split_on_disk(train, machine)  # 起動器でも検査済みだが、train.py 単独でも同じ規則で止まるように
        self.split_file = train["adv.eid_split_file"]
        idx = CaseIndex(machine["eid_dir"], cache_dir, drop_eval_only=True)
        self.cases = [c for c in self.split["train"] if c in idx.slices]  # train だけ。val / test のスライスは列挙結果を捨てる（読まない）
        empty = [c for c in self.split["train"] if c not in idx.slices]
        if empty:
            raise RuntimeError(f"adv.eid_split_file の train の症例フォルダに PNG がありません: {empty[:10]}")
        self.slices = {c: list(idx.slices[c]) for c in self.cases}
        first = read_stored(self.slices[self.cases[0]][0])
        self.img_hw = first.shape
        if self.crop > min(self.img_hw):
            raise ValueError(f"adv.crop_eid {self.crop} が実 EID の画像サイズ {self.img_hw} を超えています")
        self.n_slices = sum(len(v) for v in self.slices.values())
        print(f"[RealEIDDataset] {machine['eid_dir']}: split {self.split_file} → train {len(self.cases)} cases / {self.n_slices} slices（val {len(self.split['val'])} / test {len(self.split['test'])} cases は読まない）, "
              f"image {self.img_hw}, crop {self.crop} → x{self.scale} {self.interp} → {out}, "
              f"patches_per_slice_eid {self.k}, batch_eid {self.batch}, len {len(self)}")

    # --- 乱数（PairDataset と同じ。seed の系列は 2 つ目の定数を変えてペア枝と重ならないように） ---
    def _rng(self, index):
        if self.serial:
            return random.Random(self.patch_seed * 1_000_003 + 500_009 + index)
        return random

    def _draw_slice(self, rng):
        case = rng.choice(self.cases)
        return rng.choice(self.slices[case])

    def __getitem__(self, index):
        rng = self._rng(index)
        if self.k == 1:
            path = self._draw_slice(rng)
            stored = read_stored(path)
        elif self.serial:
            group = index // self.k
            if group != self._cur_group:
                p = self._draw_slice(random.Random(self.patch_seed * 1_000_003 + 500_009 + 977 + group))
                self._cur_group, self._cur = group, (p, read_stored(p))
            path, stored = self._cur
        else:
            if self._left <= 0:
                p = self._draw_slice(rng)
                self._cur, self._left = (p, read_stored(p)), self.k
            self._left -= 1
            path, stored = self._cur
        if stored.shape != self.img_hw:
            raise ValueError(f"実 EID の画像サイズが揃っていません: {stored.shape} != {self.img_hw}: {path}")
        h, w = self.img_hw
        y, x = rng.randrange(h - self.crop + 1), rng.randrange(w - self.crop + 1)
        up = crop_upsampled(stored, y, x, self.crop, self.scale, self.interp)
        return {"input": torch.from_numpy(normalize(up, *self.hu)).unsqueeze(0), "path": path, "y": y, "x": x}

    def __len__(self):
        return self.n_steps * self.batch

    def info(self):
        """dataset_info.yaml 用。"""
        return {"eid_split_file": self.split_file, "eid_train_cases": list(self.cases), "eid_train_slices": self.n_slices, "eid_val_cases": len(self.split["val"]), "eid_test_cases": len(self.split["test"]),
                "eid_image_hw": list(self.img_hw), "crop_eid": self.crop,
                "crop_out": self.crop * self.scale, "patches_per_slice_eid": self.k, "batch_eid": self.batch, "margin": MARGIN}
