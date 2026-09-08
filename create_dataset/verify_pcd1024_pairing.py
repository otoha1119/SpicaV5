"""
PCD1024_v1 と PCD512_v2 のスライス対応を検証する。

Stage 2 は教師あり学習なのでペアがずれると致命的。Z 間隔が 0.1mm で隣接
スライスがほぼ同一のため「同一インデックスで NCC が高い」だけでは足りない
（±5 枚ずれていても 0.99 は出る）。そこで周辺 ±W スライスを総当たりし、
**同一インデックスが最大になる**ことを確認する。

各症例で 12 箇所（先頭側 2・中央 8・末尾側 2）を検証点にし、それぞれ
1024[i] に対して 512[i-W..i+W] の NCC を全部計算して argmax を見る。
NCC の鋭さ（±1, +5, +10, +20 での低下）も出力し、指標がスライス単位の
分解能を持つことを同時に示す。

2026-09-08 の実行結果: 20 症例 × 12 点 × 41 候補 = 9,840 通りすべてで
argmax = 0。NCC は @0 = 0.999、@±1 = 0.991、@+5 = 0.947、@+20 = 0.865 で
1 枚のずれを識別できている。よって全症例が先頭から 1 対 1 対応であり、
枚数が違う 3 症例（PCD-006 / 011 / 015）は末尾が余っているだけなので
min(n1024, n512) までを使えばよい。
"""
import os, glob, cv2, numpy as np
from concurrent.futures import ProcessPoolExecutor

PCD1024 = "/Volumes/OTO-SSD/DataSet/PCD1024_v1"
PCD512  = "/Volumes/OTO-SSD/DataSet/PCD512_v2"
W       = 20   # 探索窓 ±20 スライス (= ±2.0mm)
N_MID   = 8    # 中央の検証点数
WORKERS = 8


def slice_paths(root, case):
    """症例内の PNG を連番順に返す（macOS の AppleDouble '._*' は除く）"""
    fs = [p for p in glob.glob(os.path.join(root, case, "*.png"))
          if not os.path.basename(p).startswith("._")]
    return sorted(fs, key=lambda p: int(os.path.basename(p).rsplit('-', 1)[1].split('.')[0]))


def ncc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def probe(f1, f5, i):
    """1024[i] に対し 512[i-W..i+W] の NCC を総当たりし (i, argmax, NCC群) を返す"""
    a = cv2.imread(f1[i], cv2.IMREAD_UNCHANGED).astype(np.float64)
    a = cv2.resize(a, (512, 512), interpolation=cv2.INTER_AREA)
    vals = {off: ncc(a, cv2.imread(f5[i + off], cv2.IMREAD_UNCHANGED).astype(np.float64))
            for off in range(-W, W + 1)}
    return i, max(vals, key=vals.get), vals


def check_case(case):
    f1, f5 = slice_paths(PCD1024, case), slice_paths(PCD512, case)
    n = min(len(f1), len(f5))
    idx = [W, W + 5]                                                   # 先頭側
    idx += [int(n * (k + 1) / (N_MID + 1)) for k in range(N_MID)]      # 中央
    idx += [n - 1 - W - 5, n - 1 - W]                                  # 末尾側
    idx = sorted({i for i in idx if i - W >= 0 and i + W < len(f5) and i < len(f1)})
    return case, len(f1), len(f5), n, [probe(f1, f5, i) for i in idx]


def main():
    print(f"探索窓 ±{W} スライス、症例あたり最大 {N_MID + 4} 点を総当たり\n")
    print(f"{'case':<9}{'n1024':>6}{'n512':>6}{'共通':>6}{'点数':>5}{'argmax≠0':>9}"
          f"{'NCC@0':>8}{'@±1':>8}{'@+5':>8}{'@+10':>8}{'@+20':>8}")
    allgood, total = True, 0
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for case, n1, n5, n, res in ex.map(check_case, [f"PCD-{i:03d}" for i in range(1, 21)]):
            bad = [(i, off) for i, off, _ in res if off != 0]
            m = lambda f: np.mean([f(v) for _, _, v in res])
            allgood &= not bad
            total += len(res) * (2 * W + 1)
            print(f"{case:<9}{n1:>6}{n5:>6}{n:>6}{len(res):>5}{len(bad):>9}"
                  f"{m(lambda v: v[0]):>8.4f}{m(lambda v: (v[1]+v[-1])/2):>8.4f}"
                  f"{m(lambda v: v[5]):>8.4f}{m(lambda v: v[10]):>8.4f}{m(lambda v: v[20]):>8.4f}"
                  + ("" if not bad else f"   <-- ずれあり {bad}"))
    print(f"\n比較総数 {total} 通り")
    print("全症例・全検証点で最良オフセット = 0:", "OK" if allgood else "NG")
    print("→ OK なら学習では min(n1024, n512) までを同一インデックスでペアにしてよい")


if __name__ == "__main__":
    main()
