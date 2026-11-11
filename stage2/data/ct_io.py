"""stage2/data/ct_io.py — 16bit PNG の読み書き、HU ↔ 正規化の変換、補間（512 → 1024）、症例の列挙（評価専用の除外）。
read_stored / write_png / load_eval_only / CaseIndex は stage1/data/ct_io.py から複製した（Stage 間で import しない）。

データの決まり: 1ch・uint16 の PNG、stored = HU + hu_offset（1400）。
tensor の値（設計書 E-1）: 旧 Stage 2 と同じ線形の n = (HU − hu_min) ÷ (hu_max − hu_min) × 2 − 1（hu_min −1400 / hu_max 4096。n の 1.0 = 2,748 HU）。
  **範囲外も切らない**（設計書 A-3・E-1。旧 normalize は [hu_min, hu_max] に切っていた）。保存のときだけ四捨五入して uint16 に収める。
補間（設計書 E-2）: OpenCV の bicubic（half-pixel 規約）。旧 upsample と同じ cv2.resize だが、**float のまま返す**（四捨五入・0 での切り落としをしない。ファイル化しないので丸める理由が無い）。
  学習・val・推論・確認のすべてがこの 1 つの関数を、スライス全体に 1 回掛けてからパッチを切る。
"""

import hashlib
import json
import os
import time
from pathlib import Path

import cv2

# OpenCV の内部スレッドを使わない。main プロセスで cv2.resize（固定の val パッチやパネルの bicubic 補間）を呼んでスレッドプールが出来たあとに
# DataLoader の worker を fork すると、worker 側の最初の cv2.resize が futex で止まる（PC1 で 2026-11-11 に発生。U の学習が step 1 に進まず、親と全 worker が待機）。
# import の時点で 0 にしておけば、どのプロセスでもスレッドプールを作らない（並列は worker の数で取る）。
cv2.setNumThreads(0)
import numpy as np

INTERP_FLAGS = {"nearest": cv2.INTER_NEAREST, "bilinear": cv2.INTER_LINEAR, "bicubic": cv2.INTER_CUBIC, "area": cv2.INTER_AREA, "lanczos": cv2.INTER_LANCZOS4}


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


def hu_tuple(data):
    return (data["data.hu_offset"], data["data.hu_min"], data["data.hu_max"])


def hu_per_unit(hu):
    """n の 1.0 が何 HU か（[−1, 1] の幅 2 = hu_max − hu_min）。"""
    return (hu[2] - hu[1]) / 2.0


def stored_to_n(stored, hu):
    """stored（uint16 か float）→ n (float32)。切らない。"""
    offset, hu_min, hu_max = hu
    return ((np.asarray(stored, dtype=np.float32) - offset - hu_min) / float(hu_max - hu_min) * 2.0 - 1.0).astype(np.float32)


def n_to_stored(n, hu):
    """n (float) → stored uint16。HU に戻して四捨五入し、uint16 の範囲に収める（保存のときだけ）。"""
    offset, hu_min, hu_max = hu
    h = (np.asarray(n, dtype=np.float64) + 1.0) / 2.0 * (hu_max - hu_min) + hu_min
    return np.clip(np.rint(h + offset), 0, 65535).astype(np.uint16)


def n_to_hu(n, hu):
    """n（tensor か numpy）→ HU。"""
    return (n + 1.0) * hu_per_unit(hu) + hu[1]


def upsample(stored, scale, interp):
    """(H, W)（uint16 か float）→ float32 (H·scale, W·scale)。cv2.resize（half-pixel 規約）。丸めない・切らない。"""
    h, w = stored.shape
    return cv2.resize(np.asarray(stored, dtype=np.float32), (w * scale, h * scale), interpolation=INTERP_FLAGS[interp])


def load_n(path, hu, scale=1, interp="bicubic"):
    """1 枚を n の float32 (H, W) で読む。scale > 1 なら stored の上で補間してから n にする（線形なので順序に依らない）。"""
    s = read_stored(path)
    return stored_to_n(upsample(s, scale, interp) if scale > 1 else s, hu)


# ---------------------------------------------------------------------------
# 症例 → スライス一覧
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
    """<root>/<症例>/<slice>.png を {症例: [パス, ...]} に整理する。症例フォルダ直下にサブフォルダがあれば再帰的に拾う。
    <root> 直下に PNG が直接ある場合は症例名 "_root" として 1 症例扱いにする。

    列挙は os.walk（scandir）+ 拡張子判定で、ファイルごとの stat はしない（Docker のバインドマウント越しでは stat が 1 件数 ms かかるため。旧 rglob + is_file は 2 倍遅かった）。
    '.' 始まりのファイル・フォルダは無視する（macOS の AppleDouble "._xxx.png" や .DS_Store が exFAT 経由のコピーで混ざる）。

    cache_dir を渡すと列挙結果を JSON に保存し、次回は「走査した全ディレクトリの mtime が一致」すれば再走査しない
    （ディレクトリの mtime は直下のファイルの追加・削除で変わる。数十万ファイルの列挙が数十回の stat になる）。ズレていれば再走査して上書き。"""

    EXT = ".png"

    def __init__(self, root, cache_dir=None, cases=None, exclude=(), drop_eval_only=False):
        """cases: この症例名だけに絞る（None で全部。ディスクに無い名前はエラー）。exclude: この症例名を外す（無い名前はエラー）。drop_eval_only: configs/eval_only.yaml の症例を黙って外す（学習系だけ True）。
        列挙とキャッシュはディレクトリ全体に対して行い、絞り込みはその後（キャッシュは分割に依らず共通）。"""
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
        self.disk_cases = sorted(self.slices.keys())  # 絞り込み前（ディスク上の全症例）
        if cases is not None:
            missing = [c for c in cases if c not in self.slices]
            if missing:
                raise RuntimeError(f"{root} に無い症例が指定されています: {missing}（ディスク上: {len(self.disk_cases)} 症例）")
            self.slices = {c: self.slices[c] for c in cases}
        if exclude:
            missing = [c for c in exclude if c not in self.slices]
            if missing:
                raise RuntimeError(f"{root} に無い症例を除外しようとしています: {missing}（黙って無視しない）")
            self.slices = {c: v for c, v in self.slices.items() if c not in set(exclude)}
        if not self.slices:
            raise RuntimeError(f"絞り込みの結果、症例が 0 になりました: {root}")
        self.cases = sorted(self.slices.keys())
        self.all_paths = [p for c in self.cases for p in self.slices[c]]

    # --- 列挙 ---
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

    # --- キャッシュ ---
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
            for d, m in data["dir_mtimes"].items():  # 走査した全ディレクトリの mtime が一致すれば有効
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
        """症例を一様に選び、その症例のスライスを一様に選ぶ (Q-D1: 症例一様)。"""
        case = rng.choice(self.cases)
        return rng.choice(self.slices[case])
