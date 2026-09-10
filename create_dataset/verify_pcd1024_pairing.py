"""
PCD1024_v1 と PCD512_v2 のスライス対応を検証する。

Stage 2 は教師あり学習なのでペアがずれると致命的。Z 間隔が 0.1mm で隣接
スライスがほぼ同一のため「同一インデックスで NCC が高い」だけでは足りない
（±5 枚ずれていても 0.99 は出る）。そこで 1024[i] を 512 の周辺スライスと
比べ、**同一インデックスが最大になる**ことを確認する。

  --full  全 22,904 ペアを検査（1024[i] vs 512[i-1], 512[i], 512[i+1]）
  既定    症例あたり 12 点を ±20 スライスの窓で総当たり（速い抜き取り版）

2026-09-08 の実行結果:
  --full  22,904 ペアすべてで offset 0 が最大。NCC@0 最小 0.9921 / 平均 0.999、
          隣接に対する最小マージン +0.0024（全スライスで正）
  既定    20 症例 × 12 点 × 41 候補 = 9,840 通りすべてで argmax = 0。
          NCC は @0 0.999 / @±1 0.991 / @+5 0.947 / @+20 0.865 で 1 枚のずれを識別できる

結論: 全症例が先頭から 1 対 1 対応。枚数が違う 3 症例（PCD-006 1251/1290、
PCD-011 1058/1075、PCD-015 1290/1264）は末尾が余っているだけなので、
学習では min(n1024, n512) までを同一インデックスでペアにすればよい。
"""
import os, sys, glob, cv2, numpy as np
from concurrent.futures import ProcessPoolExecutor

PCD1024 = "/Volumes/OTO-SSD/DataSet/PCD1024_v1"
PCD512  = "/Volumes/OTO-SSD/DataSet/PCD512_v2"
W       = 20   # 抜き取り版の探索窓 ±20 スライス (= ±2.0mm)
N_MID   = 8    # 抜き取り版の中央の検証点数
WORKERS = 6


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


def down(path):
    """1024 を 512 に縮小して読む"""
    a = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float64)
    return cv2.resize(a, (512, 512), interpolation=cv2.INTER_AREA)


# ---------------- 全数版 ----------------

def check_full(case):
    f1, f5 = slice_paths(PCD1024, case), slice_paths(PCD512, case)
    n = min(len(f1), len(f5))
    cache = {}

    def g512(j):
        if j not in cache:
            cache[j] = cv2.imread(f5[j], cv2.IMREAD_UNCHANGED).astype(np.float64)
        return cache[j]

    bad, v0_min, v0_sum, margin_min = [], 1.0, 0.0, 1.0
    for i in range(n):
        a = down(f1[i])
        cands = {0: ncc(a, g512(i))}
        if i - 1 >= 0:      cands[-1] = ncc(a, g512(i - 1))
        if i + 1 < len(f5): cands[+1] = ncc(a, g512(i + 1))
        if max(cands, key=cands.get) != 0:
            bad.append((i, cands))
        v0_min = min(v0_min, cands[0]); v0_sum += cands[0]
        if len(cands) > 1:
            margin_min = min(margin_min, cands[0] - max(v for k, v in cands.items() if k != 0))
        for j in [j for j in cache if j < i - 1]:
            del cache[j]
    return case, len(f1), len(f5), n, bad, v0_min, v0_sum / n, margin_min


def run_full():
    print("全数検査: 各 i で 1024[i] を 512[i-1] / 512[i] / 512[i+1] と比較\n")
    print(f"{'case':<9}{'n1024':>6}{'n512':>6}{'ペア':>6}{'不一致':>7}"
          f"{'NCC@0 min':>11}{'平均':>9}{'最小マージン':>13}")
    total, allgood = 0, True
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for case, n1, n5, n, bad, lo, avg, mg in ex.map(check_full, [f"PCD-{i:03d}" for i in range(1, 21)]):
            total += n; allgood &= not bad
            print(f"{case:<9}{n1:>6}{n5:>6}{n:>6}{len(bad):>7}{lo:>11.4f}{avg:>9.4f}{mg:>13.4f}"
                  + ("" if not bad else f"   <-- ずれあり 先頭5件 {bad[:5]}"), flush=True)
    print(f"\n検査ペア総数 {total}")
    print("全ペアで offset 0 が最大:", "OK" if allgood else "NG")
    print("→ OK なら学習では min(n1024, n512) までを同一インデックスでペアにしてよい")


# ---------------- 抜き取り版 ----------------

def probe(f1, f5, i):
    a = down(f1[i])
    vals = {off: ncc(a, cv2.imread(f5[i + off], cv2.IMREAD_UNCHANGED).astype(np.float64))
            for off in range(-W, W + 1)}
    return i, max(vals, key=vals.get), vals


def check_sample(case):
    f1, f5 = slice_paths(PCD1024, case), slice_paths(PCD512, case)
    n = min(len(f1), len(f5))
    idx = [W, W + 5]                                                   # 先頭側
    idx += [int(n * (k + 1) / (N_MID + 1)) for k in range(N_MID)]      # 中央
    idx += [n - 1 - W - 5, n - 1 - W]                                  # 末尾側
    idx = sorted({i for i in idx if i - W >= 0 and i + W < len(f5) and i < len(f1)})
    return case, len(f1), len(f5), n, [probe(f1, f5, i) for i in idx]


def run_sample():
    print(f"抜き取り検査: 探索窓 ±{W} スライス、症例あたり最大 {N_MID + 4} 点を総当たり\n")
    print(f"{'case':<9}{'n1024':>6}{'n512':>6}{'ペア':>6}{'点数':>5}{'argmax≠0':>9}"
          f"{'NCC@0':>8}{'@±1':>8}{'@+5':>8}{'@+10':>8}{'@+20':>8}")
    allgood, total = True, 0
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for case, n1, n5, n, res in ex.map(check_sample, [f"PCD-{i:03d}" for i in range(1, 21)]):
            bad = [(i, off) for i, off, _ in res if off != 0]
            m = lambda f: np.mean([f(v) for _, _, v in res])
            allgood &= not bad
            total += len(res) * (2 * W + 1)
            print(f"{case:<9}{n1:>6}{n5:>6}{n:>6}{len(res):>5}{len(bad):>9}"
                  f"{m(lambda v: v[0]):>8.4f}{m(lambda v: (v[1]+v[-1])/2):>8.4f}"
                  f"{m(lambda v: v[5]):>8.4f}{m(lambda v: v[10]):>8.4f}{m(lambda v: v[20]):>8.4f}"
                  + ("" if not bad else f"   <-- ずれあり {bad}"), flush=True)
    print(f"\n比較総数 {total} 通り")
    print("全検証点で最良オフセット = 0:", "OK" if allgood else "NG")


if __name__ == "__main__":
    run_full() if "--full" in sys.argv else run_sample()
