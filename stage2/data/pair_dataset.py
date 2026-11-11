"""stage2/data/pair_dataset.py — 疑似 EID512（K 通り）↔ PCD1024 の組の読み込み（設計書 B-5・E-2・F-1、RECONCILE #3・#9・#10・#12・#13）。

【入力】 <pseudo_eid_dir>/<組>/k00, k01, …/<症例>/<スライス>.png（512、Stage 1 の bash start.sh generate の出力）と <組>/manifest.yaml。
        「組」は data.yaml の data.pseudo_eid_set（フォルダ名）。K は manifest の k から読む（設定に書かない）。
【正解】 machines.yaml の pcd1024_dir/<症例>/<スライス>.png。症例名・ファイル名で対応させ、両方にある名前だけ使う（末尾の枚数差は共通部分だけ）。
【補間】 疑似 EID は読むたびにスライス全体を bicubic ×2（data/ct_io.upsample。float のまま）してからパッチを切る。1024 のファイルは作らない。
【分割】 data.train_cases / val_cases / test_cases。train・val の症例は全部の k と正解にあること。ディスク上の症例（評価専用を除く）は必ずどれかのリストに入っていること。
        test は読まない（例外は表示用の log.pcd_slice の 1 枚）。評価専用（configs/eval_only.yaml）は CaseIndex(drop_eval_only=True) で除外。
【引き方】 症例一様 → スライス一様 → 通り（k）一様 → 位置一様。1 回読んだスライスから patches_per_slice 個を連続して切る（同じ 1 通り、位置は毎回独立）。
          val・表示は k00 固定。水増しなし。
【起動前の検査】 check_before_launch(): 組と manifest があること、manifest の明るさの広さが configs/shared.yaml と同じこと（違えばエラー。data.allow_brightness_mismatch で明示したときだけ通す）、
          manifest の画像の大きさ × 倍率 = 正解の大きさ、表示用のスライスがあること、表示・確認用の実 EID が eid_split_file の train 側でないこと。
"""

import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import IterableDataset, get_worker_info

from data.ct_io import CaseIndex, hu_tuple, load_n, read_stored

REPO = Path(__file__).resolve().parents[2]  # リポジトリ直下（このファイルは <repo>/stage2/data/）
MANIFEST = "manifest.yaml"


def kdir(k):
    return f"k{int(k):02d}"


def set_dir(data, machine):
    return Path(machine["pseudo_eid_dir"]) / data["data.pseudo_eid_set"]


def read_manifest(sdir):
    """<組>/manifest.yaml を読む。無い・形が違えば RuntimeError（推測で補わない）。"""
    sdir = Path(sdir)
    if not sdir.is_dir():
        raise RuntimeError(f"疑似 EID の組のフォルダがありません: {sdir}（bash start.sh generate で作る。名前は --pseudo_eid_set）")
    p = sdir / MANIFEST
    if not p.is_file():
        raise RuntimeError(f"{p} がありません（作りかけの組か、Stage 1 の generate 以外で作ったフォルダ）")
    with open(p, encoding="utf-8") as f:
        m = yaml.safe_load(f)
    need = ("k", "brightness_sigma_px_512", "image_hw", "hu_offset")
    if not isinstance(m, dict) or any(k not in m for k in need):
        raise RuntimeError(f"{p} に必要なキーがありません: {[k for k in need if not isinstance(m, dict) or k not in m]}")
    if not isinstance(m["k"], int) or m["k"] < 1:
        raise RuntimeError(f"{p}: k は 1 以上の整数: {m['k']!r}")
    return m


def check_manifest(m, data, shared, where):
    """manifest と設定の突き合わせ（RECONCILE #9）。"""
    sm, ss = float(m["brightness_sigma_px_512"]), float(shared["brightness.sigma_px_512"])
    if abs(sm - ss) > 1e-9:
        msg = f"{where}: 疑似 EID を作ったときの明るさの広さ（brightness_sigma_px_512 = {sm:g}）が configs/shared.yaml（{ss:g}）と違います"
        if not data["data.allow_brightness_mismatch"]:
            raise RuntimeError(msg + "。そろえるか、分かったうえで通すなら --allow_brightness_mismatch true")
        print(f"[data] 注意: {msg}（data.allow_brightness_mismatch: true なので通す。Stage 2 は shared.yaml の {ss:g} を使う）", flush=True)
    if int(m["hu_offset"]) != data["data.hu_offset"]:
        raise RuntimeError(f"{where}: hu_offset {m['hu_offset']} が data.hu_offset {data['data.hu_offset']} と違います")
    cue = m.get("cue")
    if isinstance(cue, dict) and cue.get("use") is False:
        print(f"[data] 注意: {where} は手がかり無し（cue.use: false）で作った組です（診断用。PCD と対応しません）", flush=True)


def eid_train_cases(data):
    """実 EID の分割ファイルの train 側の症例名（Stage 1 の P が学んだ症例）。"""
    p = Path(data["data.eid_split_file"])
    p = p if p.is_absolute() else REPO / p
    if not p.is_file():
        raise RuntimeError(f"実 EID の症例分割ファイルがありません: {p}")
    with open(p, encoding="utf-8") as f:
        split = yaml.safe_load(f)
    if not isinstance(split, dict) or not isinstance(split.get("train"), list):
        raise RuntimeError(f"{p}: train の症例名のリストが必要です")
    return set(split["train"])


def check_eid_not_train(data, items, where):
    """items: [(表示名, 症例名), ...]。train 側が混ざっていたらエラー（RECONCILE #12: 確認に使う実 EID を P に見せていないこと）。"""
    tr = eid_train_cases(data)
    bad = [name for name, case in items if case in tr]
    if bad:
        raise RuntimeError(f"{where} に、Stage 1 の P が学習に使った症例（{data['data.eid_split_file']} の train 側）があります: {bad}（val / test 側か評価専用の症例にする）")


def _by_name(paths):
    d = {}
    for p in paths:
        name = Path(p).name
        if name in d:
            raise RuntimeError(f"同じスライス名が 2 つあります: {d[name]} と {p}")
        d[name] = p
    return d


def even_pick(items, n):
    """等間隔に n 個（n ≥ len なら全部）。"""
    if n <= 0 or len(items) <= n:
        return list(items)
    return [items[i] for i in np.linspace(0, len(items) - 1, n).round().astype(int)]


class PairData:
    """組と正解を突き合わせた結果。pairs[症例] = [(スライス名, 正解のパス), ...]（名前順）。疑似 EID のパスは pseudo_path(k, 症例, 名前)。"""

    def __init__(self, data, shared, machine, cache_dir):
        self.data, self.hu = data, hu_tuple(data)
        self.scale, self.interp = data["data.upsample_scale"], data["data.upsample_interp"]
        self.set_dir = set_dir(data, machine)
        self.manifest = read_manifest(self.set_dir)
        check_manifest(self.manifest, data, shared, str(self.set_dir / MANIFEST))
        self.k = int(self.manifest["k"])
        self.pcd_dir = Path(machine["pcd1024_dir"])
        tr, va, te = data["data.train_cases"], data["data.val_cases"], data["data.test_cases"]
        missing_k = [kdir(k) for k in range(self.k) if not (self.set_dir / kdir(k)).is_dir()]
        if missing_k:
            raise RuntimeError(f"{self.set_dir}: manifest は k = {self.k} だが {missing_k} がありません")
        idx_k = [CaseIndex(self.set_dir / kdir(k), cache_dir, drop_eval_only=True) for k in range(self.k)]
        idx_pcd = CaseIndex(self.pcd_dir, cache_dir, drop_eval_only=True)
        assigned = set(tr) | set(va) | set(te)
        unassigned = sorted((set(idx_k[0].disk_cases) | set(idx_pcd.disk_cases)) - assigned)
        if unassigned:
            raise RuntimeError(f"data.yaml のどのリストにも無い症例がディスクにあります（train_cases / val_cases / test_cases のどれかに入れること）: {unassigned}")
        self.pairs, self.info = {}, {}
        for c in list(tr) + list(va):
            lack = [kdir(k) for k in range(self.k) if c not in idx_k[k].slices] + ([] if c in idx_pcd.slices else [str(self.pcd_dir)])
            if lack:
                raise RuntimeError(f"症例 {c} が {lack} にありません（train / val の症例は全部の通りと正解に要る。組を作り直すか、分割から外す）")
            names = [_by_name(i.slices[c]) for i in idx_k]
            if any(set(n) != set(names[0]) for n in names[1:]):
                raise RuntimeError(f"症例 {c}: 通り（k）ごとにスライス名が違います（{[len(n) for n in names]} 枚）。組が作りかけの可能性")
            pcd = _by_name(idx_pcd.slices[c])
            common = sorted(set(names[0]) & set(pcd))
            if not common:
                raise RuntimeError(f"症例 {c}: 疑似 EID と正解で同じ名前のスライスがありません（{len(names[0])} 枚 / {len(pcd)} 枚）")
            self.pairs[c] = [(n, pcd[n]) for n in common]
            self.info[c] = {"pseudo": len(names[0]), "pcd": len(pcd), "pair": len(common)}
        self.train_cases, self.val_cases = list(tr), list(va)
        first = self.pairs[self.train_cases[0]][0]
        a, b = read_stored(self.pseudo_path(0, self.train_cases[0], first[0])), read_stored(first[1])
        if tuple(x * self.scale for x in a.shape) != tuple(b.shape):
            raise RuntimeError(f"疑似 EID の大きさ {a.shape} × {self.scale} が正解 {b.shape} と合いません: {first[1]}")
        self.hw = tuple(b.shape)
        n_tr, n_va = (sum(len(self.pairs[c]) for c in cs) for cs in (self.train_cases, self.val_cases))
        print(f"[data] 組 {self.set_dir.name}: K {self.k}、train {len(tr)} 症例 / {n_tr} 組、val {len(va)} 症例 / {n_va} 組、test {len(te)} 症例（読まない）、"
              f"正解 {self.hw}、補間 {self.interp} ×{self.scale}", flush=True)

    def pseudo_path(self, k, case, name):
        return self.set_dir / kdir(k) / case / name

    def load_x(self, path):
        """512 の入力（疑似 EID・実 EID）→ 補間した n の (1, 1, H, W)。"""
        return torch.from_numpy(load_n(path, self.hu, self.scale, self.interp))[None, None]

    def load_t(self, path):
        """1024 の画像（正解など）→ n の (1, 1, H, W)。"""
        return torch.from_numpy(load_n(path, self.hu))[None, None]

    def val_slices(self, n_per_case):
        """val のスライス [(症例, 名前, 正解のパス), ...]。症例ごとに等間隔で n_per_case 枚。毎回同じ集合。"""
        return [(c, n, p) for c in self.val_cases for n, p in even_pick(self.pairs[c], n_per_case)]

    def summary(self):
        return {"pseudo_eid_set": self.set_dir.name, "set_dir": str(self.set_dir), "k": self.k, "image_hw": list(self.hw),
                "n_train_pairs": sum(len(self.pairs[c]) for c in self.train_cases), "n_val_pairs": sum(len(self.pairs[c]) for c in self.val_cases),
                "cases": self.info, "manifest": self.manifest}


def check_before_launch(data, cfg, shared, machine, cache_dir):
    """起動器が run フォルダを作る前に呼ぶ（空の run を残さない。何時間も進んでから落ちない）。問題は RuntimeError。戻り値は PairData。"""
    pd = PairData(data, shared, machine, cache_dir)
    fixed = cfg["log.pcd_slice"]
    case = Path(fixed).parts[0]
    if case not in set(data["data.train_cases"]) | set(data["data.val_cases"]) | set(data["data.test_cases"]):
        raise RuntimeError(f"log.pcd_slice の症例 {case} が train_cases / val_cases / test_cases のどれにもありません: {fixed}")
    for p in (pd.set_dir / kdir(0) / fixed, pd.pcd_dir / fixed):
        if not p.is_file():
            raise RuntimeError(f"log.pcd_slice がありません: {p}（<組>/k00 と pcd1024_dir からの相対パス。疑似 EID は表示用の症例も作っておくこと = RECONCILE #13）")
    missing = [p for p in cfg["log.eid_slices"] if not Path(p).is_file()]
    if missing:
        raise RuntimeError(f"log.eid_slices のスライスがありません: {missing}（そのマシンに無いものは --eid_slices で外す）")
    check_eid_not_train(data, [(p, Path(p).parent.name) for p in cfg["log.eid_slices"]], "log.eid_slices")
    for c in cfg["mag.eid_cases"]:
        if not (Path(machine["eid_dir"]) / c).is_dir():
            raise RuntimeError(f"mag.eid_cases の症例がありません: {Path(machine['eid_dir']) / c}")
    check_eid_not_train(data, [(c, c) for c in cfg["mag.eid_cases"]], "mag.eid_cases")
    for d in cfg["mag.hospital_dirs"]:
        if not Path(d).is_dir():
            raise RuntimeError(f"mag.hospital_dirs のフォルダがありません: {d}（そのマシンに無いときは --mag_hospital_dirs \"\"）")
    check_eid_not_train(data, [(d, Path(d).name) for d in cfg["mag.hospital_dirs"]], "mag.hospital_dirs")
    return pd


def list_pngs(d):
    """フォルダ直下の PNG（名前順。'.' 始まりは無視）。"""
    return sorted(p for p in Path(d).iterdir() if p.suffix.lower() == ".png" and not p.name.startswith("."))


class PatchStream(IterableDataset):
    """学習パッチを無限に返す。各 worker が別の乱数列を持つ（再開のたびに別の列）。返すもの: {"x": 入力 (1, S, S), "t": 正解 (1, S, S)}（n の値、同じ場所）。
    S = patch + 2·margin。位置は S 四方が画像に収まる範囲から一様（設計書 C-4 の 3: 画像の外周 margin 画素は中央に来ない）。"""

    def __init__(self, pd, patch, margin, seed, start_step):
        self.pd, self.size, self.seed, self.start_step = pd, patch + 2 * margin, seed, int(start_step)
        self.k = pd.data["data.patches_per_slice"]
        if self.size > min(pd.hw):
            raise RuntimeError(f"パッチ（余白込み {self.size}）が画像 {pd.hw} より大きい")

    def __iter__(self):
        info = get_worker_info()
        wid = info.id if info is not None else 0
        rng = random.Random(f"{self.seed}/{wid}/{self.start_step}")
        pd, s = self.pd, self.size
        h, w = pd.hw
        while True:
            case = rng.choice(pd.train_cases)
            name, pcd_path = rng.choice(pd.pairs[case])
            k = rng.randrange(pd.k)
            x = load_n(pd.pseudo_path(k, case, name), pd.hu, pd.scale, pd.interp)
            t = load_n(pcd_path, pd.hu)
            if x.shape != (h, w) or t.shape != (h, w):
                raise RuntimeError(f"画像の大きさが揃っていません: {case}/{name} 入力 {x.shape} 正解 {t.shape}（期待 {(h, w)}）")
            for _ in range(self.k):
                y0, x0 = rng.randint(0, h - s), rng.randint(0, w - s)
                yield {"x": torch.from_numpy(np.ascontiguousarray(x[y0:y0 + s, x0:x0 + s]))[None],
                       "t": torch.from_numpy(np.ascontiguousarray(t[y0:y0 + s, x0:x0 + s]))[None]}


def worker_init(_worker_id):
    """worker は SIGTERM を無視する（監視ツールは train*.py の全プロセスに SIGTERM を送る。worker が先に死ぬと main が latest を保存する前に DataLoader のエラーで落ちる）。
    OpenCV の内部スレッドは使わない（worker の数だけ並列なので）。"""
    import signal
    import cv2
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    cv2.setNumThreads(0)  # スレッドプールを作らない（ct_io.py の冒頭と同じ理由。fork 後の停止を避ける）


def fixed_val_patches(pd, patch, margin, n, seed):
    """val の症例の固定パッチ（k00）。戻り値 {"x": (n, 1, S, S), "t": (n, 1, S, S)}。症例を順に回し、スライスと位置は seed で決める。"""
    s = patch + 2 * margin
    rng = random.Random(f"val/{seed}")
    h, w = pd.hw
    xs, ts = [], []
    for i in range(n):
        case = pd.val_cases[i % len(pd.val_cases)]
        name, pcd_path = rng.choice(pd.pairs[case])
        x, t = load_n(pd.pseudo_path(0, case, name), pd.hu, pd.scale, pd.interp), load_n(pcd_path, pd.hu)
        y0, x0 = rng.randint(0, h - s), rng.randint(0, w - s)
        xs.append(torch.from_numpy(np.ascontiguousarray(x[y0:y0 + s, x0:x0 + s]))[None])
        ts.append(torch.from_numpy(np.ascontiguousarray(t[y0:y0 + s, x0:x0 + s]))[None])
    return {"x": torch.stack(xs), "t": torch.stack(ts)}
