"""stage3/data/pair_dataset.py — Stage 3 のペアローダ（条件 = EID-like1024 の patch ↔ 目標 x0 = PCD1024 の同位置 patch）。
stage2/data/pair_dataset.py から 2026-10-13 に複製し、拡散モデル用に変えた:
  ・patch の一辺 = ddim.yml data.image_size（本家 256。ε-net の assert がこの大きさしか通さない）
  ・data.random_flip（本家のキー。true なら入力と教師を同じく左右反転。Stage 3 の既定は false）
  ・監視用: 固定の val patch（val_patches。val の ε 損失と TB の val patch サンプル）、固定スライスと実 EID（1024 のままタイル合成）
  ・stage2 の「epoch 末の val フル画像の指標」「ランダムスライス」「serial_batches」は持たない（指標は推論 + metrics.txt で取る。決定 7）
症例分割・ディスクとの照合・名前による対応付け・patches_per_slice・manifest.yaml からの補間方式は stage2 と同じ規則。
"""

import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

from data.ct_io import INTERP_FLAGS, CaseIndex, list_case_names, load_normalized, normalize, read_stored, upsample

MANIFEST = "manifest.yaml"  # bash start2.sh dataset が eidlike1024_dir に書く（補間の方式）


def check_split_on_disk(train, machine):
    """train.yaml の症例リストとディスク（eidlike1024_dir / pcd1024_dir 直下の症例フォルダ）を照合する（stage2 と同じ規則）。run_train.py が run ディレクトリを作る前に呼ぶ。"""
    train_cases, val_cases, test_cases = train["data.train_cases"], train["data.val_cases"], train["data.test_cases"]
    on_in, on_pcd = list_case_names(machine["eidlike1024_dir"], drop_eval_only=True), list_case_names(machine["pcd1024_dir"], drop_eval_only=True)  # 評価専用（configs/eval_only.yaml）は照合の対象外
    assigned = set(train_cases) | set(val_cases) | set(test_cases)
    unassigned = sorted((set(on_in) | set(on_pcd)) - assigned)
    if unassigned:
        raise RuntimeError(f"train.yaml のどのリストにも無い症例がディスクにあります（train_cases / val_cases / test_cases のどれかに入れること）: {unassigned}")
    for kind, cases in (("train_cases", train_cases), ("val_cases", val_cases)):
        missing = [c for c in cases if c not in on_in or c not in on_pcd]
        if missing:
            raise RuntimeError(f"{kind} の症例が入力（{machine['eidlike1024_dir']}）か教師（{machine['pcd1024_dir']}）にありません: {missing}")
    fixed_case = Path(train["log.full_slice"]).parts[0]
    if fixed_case not in assigned:
        raise RuntimeError(f"train.yaml log.full_slice の症例 {fixed_case} が train_cases / val_cases / test_cases のどれにも含まれていません: {train['log.full_slice']}")
    if fixed_case in test_cases:
        print(f"[PairDataset] 注意: 固定スライス {train['log.full_slice']} は test の症例（学習・検証には使わず、この 1 枚だけ目視用に読む）")
    for root in (machine["eidlike1024_dir"], machine["pcd1024_dir"]):
        if not (Path(root) / train["log.full_slice"]).is_file():
            raise RuntimeError(f"train.yaml log.full_slice がありません: {Path(root) / train['log.full_slice']}（eidlike1024_dir / pcd1024_dir からの相対パス）")
    eid = Path(machine["eid_dir"]) / train["log.eid_slice"]
    if not eid.is_file():
        raise RuntimeError(f"train.yaml log.eid_slice がありません: {eid}（machines.yaml の eid_dir からの相対パス）")
    for r in train["log.crop_pcd"]:  # 固定領域（目視用。症例はどれかのリストに入っていること。test の症例でも読む）
        if Path(r["slice"]).parts[0] not in assigned:
            raise RuntimeError(f"train.yaml log.crop_pcd の症例 {Path(r['slice']).parts[0]} が train_cases / val_cases / test_cases のどれにも含まれていません: {r['slice']}")
        for root in (machine["eidlike1024_dir"], machine["pcd1024_dir"]):
            if not (Path(root) / r["slice"]).is_file():
                raise RuntimeError(f"train.yaml log.crop_pcd がありません: {Path(root) / r['slice']}（eidlike1024_dir / pcd1024_dir からの相対パス）")
    for r in train["log.crop_eid"]:
        if not (Path(machine["eid_dir"]) / r["slice"]).is_file():
            raise RuntimeError(f"train.yaml log.crop_eid がありません: {Path(machine['eid_dir']) / r['slice']}（machines.yaml の eid_dir からの相対パス）")
    scale = read_upsample_cfg(machine["eidlike1024_dir"])[0]
    size = train["log.crop_size"]
    for kind, root, mult in (("crop_pcd", machine["eidlike1024_dir"], 1), ("crop_eid", machine["eid_dir"], scale)):  # 固定領域（一辺 crop_size）は画像の中に収まること。run を作る前に止める
        for r in train[f"log.{kind}"]:
            h, w = (n * mult for n in read_stored(Path(root) / r["slice"]).shape)
            if r["y"] + size > h or r["x"] + size > w:
                raise RuntimeError(f"train.yaml log.{kind} {r}（一辺 {size}）が画像 {h}x{w} をはみ出します")
    return {"input": on_in, "pcd": on_pcd}


def read_upsample_cfg(eidlike1024_dir):
    """eidlike1024_dir/manifest.yaml（bash start2.sh dataset が書く）から補間の方式 (scale, interp) を読む。無い・不正ならエラー（推測で補間しない）。"""
    p = Path(eidlike1024_dir) / MANIFEST
    if not p.is_file():
        raise RuntimeError(f"{p} がありません。学習入力は bash start2.sh dataset で作ったもの（manifest.yaml 付き）を使ってください（実 EID テストスライスを同じ方式で補間するため）")
    with open(p, encoding="utf-8") as f:
        m = yaml.safe_load(f)
    try:
        scale, interp = int(m["dataset"]["scale"]), str(m["dataset"]["interp"])
    except (KeyError, TypeError, ValueError) as e:
        raise RuntimeError(f"{p} の dataset.scale / dataset.interp を読めません: {e}")
    if scale < 2 or interp not in INTERP_FLAGS:
        raise RuntimeError(f"{p} の補間の方式が不正です: scale {scale}, interp {interp!r}")
    return scale, interp


def _by_name(paths):
    d = {}
    for p in paths:
        name = Path(p).name
        if name in d:
            raise RuntimeError(f"同じスライス名が 2 つあります: {d[name]} と {p}")
        d[name] = p
    return d


def build_pairs(idx_in, idx_pcd, cases):
    """症例ごとに (入力 path, 教師 path) を名前で対応付ける（stage2 と同じ）。"""
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
    def __init__(self, train, ddim, machine, cache_dir):
        """train / ddim / machine は run_train.py が検証した実効値（平坦 dict）。"""
        self.hu = (train["data.hu_offset"], train["data.hu_min"], train["data.hu_max"])
        self.patch = ddim["data.image_size"]
        self.flip = ddim["data.random_flip"]
        self.length = train["log.steps_per_epoch"] * ddim["training.batch_size"]  # DataLoader 1 周 = 1 epoch = steps_per_epoch step 分の patch
        self.patch_seed = train["data.patch_seed"]
        self.k = train["data.patches_per_slice"]
        self._left, self._cur = 0, None  # worker ごとの状態（DataLoader の worker は fork されるので各 worker で独立に進む）
        self.full_slice = train["log.full_slice"]
        self.eid_path = str(Path(machine["eid_dir"]) / train["log.eid_slice"])
        self.dirs = {"eidlike": Path(machine["eidlike1024_dir"]), "pcd": Path(machine["pcd1024_dir"]), "eid": Path(machine["eid_dir"])}
        self.upsample_cfg = read_upsample_cfg(machine["eidlike1024_dir"])
        train_cases, val_cases, test_cases = train["data.train_cases"], train["data.val_cases"], train["data.test_cases"]

        check_split_on_disk(train, machine)
        idx_in = CaseIndex(machine["eidlike1024_dir"], cache_dir, drop_eval_only=True)
        idx_pcd = CaseIndex(machine["pcd1024_dir"], cache_dir, drop_eval_only=True)
        for kind, cases in (("train_cases", train_cases), ("val_cases", val_cases)):
            empty = [c for c in cases if c not in idx_in.slices or c not in idx_pcd.slices]
            if empty:
                raise RuntimeError(f"{kind} の症例フォルダに PNG がありません: {empty}")
        self.train_cases, self.val_cases, self.test_cases = list(train_cases), list(val_cases), list(test_cases)
        self.fixed_pair = (str(Path(machine["eidlike1024_dir"]) / self.full_slice), str(Path(machine["pcd1024_dir"]) / self.full_slice))
        self.pairs_train, info_train = build_pairs(idx_in, idx_pcd, self.train_cases)
        self.pairs_val, info_val = build_pairs(idx_in, idx_pcd, self.val_cases)
        self.info = {"train": info_train, "val": info_val}

        first_in, first_pcd = self.pairs_train[self.train_cases[0]][0]
        a, b = read_stored(first_in), read_stored(first_pcd)
        if a.shape != b.shape:
            raise ValueError(f"入力と教師の画像サイズが違います: {a.shape} != {b.shape}: {first_in} / {first_pcd}")
        self.img_hw = a.shape
        f = read_stored(self.fixed_pair[0])
        if f.shape != self.img_hw:
            raise ValueError(f"固定スライスの大きさ {f.shape} が学習画像 {self.img_hw} と違います: {self.fixed_pair[0]}")
        e = read_stored(self.eid_path)
        if tuple(x * self.upsample_cfg[0] for x in e.shape) != tuple(self.img_hw):
            raise ValueError(f"実 EID テストスライス {self.eid_path} の大きさ {e.shape} × scale {self.upsample_cfg[0]} が学習画像 {self.img_hw} と合いません")
        if self.patch > min(self.img_hw):
            raise ValueError(f"patch（ddim.yml data.image_size）{self.patch} が画像サイズ {self.img_hw} を超えています")
        n_tr = sum(len(v) for v in self.pairs_train.values())
        n_va = sum(len(v) for v in self.pairs_val.values())
        dropped = sum(i["dropped"] for i in list(info_train.values()) + list(info_val.values()))
        print(f"[PairDataset] train {len(self.train_cases)} cases / {n_tr} pairs, val {len(self.val_cases)} cases / {n_va} pairs, "
              f"test {len(self.test_cases)} cases（読まない）, image {self.img_hw}, patch {self.patch}, patches_per_slice {self.k}, random_flip {self.flip}, len {len(self)}"
              + (f", 名前が片側にしか無いスライス {dropped} 枚は不使用" if dropped else ""))

    # --- 切り出し ---
    def _patch(self, path, pos, flip=False):
        img = load_normalized(path, *self.hu)
        if img.shape != self.img_hw:
            raise ValueError(f"画像サイズが揃っていません: {img.shape} != {self.img_hw}: {path}")
        y, x = pos
        p = img[y : y + self.patch, x : x + self.patch]
        if flip:
            p = p[:, ::-1]
        return torch.from_numpy(np.ascontiguousarray(p)).unsqueeze(0)

    def _draw_pair(self):
        case = random.choice(self.train_cases)
        return random.choice(self.pairs_train[case])

    def __getitem__(self, index):
        """症例一様 → スライス一様 → 位置一様（stage2 と同じ）。patches_per_slice 回ごとにペアを引き直す。戻り値 input = 条件（EID-like）、target = x0（PCD）。"""
        if self.k == 1:
            in_path, pcd_path = self._draw_pair()
        else:
            if self._left <= 0:
                self._cur, self._left = self._draw_pair(), self.k
            self._left -= 1
            in_path, pcd_path = self._cur
        h, w = self.img_hw
        pos = (random.randrange(h - self.patch + 1), random.randrange(w - self.patch + 1))
        flip = self.flip and random.random() < 0.5
        return {"input": self._patch(in_path, pos, flip), "target": self._patch(pcd_path, pos, flip), "path": pcd_path, "y": pos[0], "x": pos[1]}

    def __len__(self):
        return self.length

    # --- 監視用（学習の乱数列を消費しない。resume 後も同じ） ---
    def val_patches(self, n, salt):
        """val 症例から決定的に選んだ n 組の patch（症例は順に巡回、スライス・位置は patch_seed と salt から）。
        戻り値 {"input": (n,1,p,p), "target": (n,1,p,p), "names": ["PCD-017-123 (y,x)", ...]}"""
        rng = random.Random(self.patch_seed * 1_000_003 + salt)
        h, w = self.img_hw
        xs, ts, names = [], [], []
        for i in range(n):
            case = self.val_cases[i % len(self.val_cases)]
            in_path, pcd_path = rng.choice(self.pairs_val[case])
            pos = (rng.randrange(h - self.patch + 1), rng.randrange(w - self.patch + 1))
            xs.append(self._patch(in_path, pos))
            ts.append(self._patch(pcd_path, pos))
            names.append(f"{Path(pcd_path).stem} ({pos[0]},{pos[1]})")
        return {"input": torch.stack(xs), "target": torch.stack(ts), "names": names}

    def load_full(self, path):
        """フル画像を (1,1,H,W) の正規化 tensor で返す。"""
        return torch.from_numpy(load_normalized(path, *self.hu)).unsqueeze(0).unsqueeze(0)

    def build_crops(self, train):
        """固定領域（train.yaml log.crop_pcd / crop_eid）を学習前に読む。元の切り出し（一辺 crop_size）の**中心を保ったまま**、周りを含めた patch の大きさ（256）の窓を切る。
        表示もこの窓全体（ユーザー指示 2026-10-13: 切り抜きの中心はそのままで周りの領域を入れた今のパッチサイズに）。窓は画像の端ではずれる（端に寄せる）。
        戻り値 [{"kind": "pcd" | "eid", "tag": スライス名, "cond": (1,1,p,p) 窓（EID-like / 実 EID 補間後）, "tgt": (1,1,p,p) 窓の教師 | None, "key": seed 用の名前}, ...]"""
        size, p = train["log.crop_size"], self.patch
        H, W = self.img_hw
        out = []
        for kind, items in (("pcd", train["log.crop_pcd"]), ("eid", train["log.crop_eid"])):
            for r in items:
                y, x = r["y"], r["x"]
                if y + size > H or x + size > W:
                    raise ValueError(f"固定領域 {r}（一辺 {size}）が画像 {H}x{W} をはみ出します")
                y0 = min(max(y + size // 2 - p // 2, 0), H - p)
                x0 = min(max(x + size // 2 - p // 2, 0), W - p)
                if kind == "pcd":
                    full = self.load_full(str(self.dirs["eidlike"] / r["slice"]))
                    tgt = self.load_full(str(self.dirs["pcd"] / r["slice"]))
                    if full.shape[-2:] != (H, W) or tgt.shape[-2:] != (H, W):
                        raise ValueError(f"固定領域のスライスの大きさが学習画像 {H}x{W} と違います: {r['slice']}")
                else:
                    scale, interp = self.upsample_cfg
                    full = torch.from_numpy(normalize(upsample(read_stored(str(self.dirs["eid"] / r["slice"])), scale, interp), *self.hu)).unsqueeze(0).unsqueeze(0)
                    tgt = None
                    if full.shape[-2:] != (H, W):
                        raise ValueError(f"実 EID の固定領域 {r['slice']} を ×{scale} 補間した大きさ {tuple(full.shape[-2:])} が学習画像 {H}x{W} と違います")
                out.append({"kind": kind, "tag": Path(r["slice"]).stem, "key": f"{r['slice']}@{y},{x}",
                            "cond": full[:, :, y0:y0 + p, x0:x0 + p].contiguous(),
                            "tgt": None if tgt is None else tgt[:, :, y0:y0 + p, x0:x0 + p].contiguous()})
        return out

    def load_eid_full(self):
        """実 EID テストスライス: eid_dir の 512 を manifest の方式で補間 → (1,1,H,W) の正規化 tensor。"""
        scale, interp = self.upsample_cfg
        up = upsample(read_stored(self.eid_path), scale, interp)
        return torch.from_numpy(normalize(up, *self.hu)).unsqueeze(0).unsqueeze(0)
