"""
PCD1024_v1 と PCD512_v2 のペア対応を検証する。

  1. 全症例で 1024 を 1/2 に縮小し、同一インデックスのスライスと NCC を取る
  2. 枚数が一致しない症例は、スライス平均 HU プロファイルの相互相関で
     Z 方向の最適シフトを探し、実画像 NCC で裏を取る

2026-09-08 の実行結果: 全20症例 NCC 0.9973〜0.9996、枚数不一致の
PCD-006 / PCD-011 / PCD-015 も最適シフト 0（先頭から 1 対 1 対応）。
末尾の枚数だけが違うので min(n1024, n512) までを使えばよい。
"""
import os, glob, cv2, numpy as np

PCD1024 = "/Volumes/OTO-SSD/DataSet/PCD1024_v1"
PCD512  = "/Volumes/OTO-SSD/DataSet/PCD512_v2"
HU_OFFSET = 1400


def slice_paths(root, case):
    """症例内の PNG を連番順に返す（macOS の AppleDouble '._*' は除く）"""
    fs = [p for p in glob.glob(os.path.join(root, case, "*.png"))
          if not os.path.basename(p).startswith("._")]
    return sorted(fs, key=lambda p: int(os.path.basename(p).rsplit('-', 1)[1].split('.')[0]))


def ncc(a, b):
    a = a.astype(np.float64) - a.mean()
    b = b.astype(np.float64) - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else 0.0


def image_ncc(p1024, p512):
    """1024 を 1/2 に縮小して 512 と NCC を取る"""
    a = cv2.imread(p1024, cv2.IMREAD_UNCHANGED).astype(np.float64)
    b = cv2.imread(p512, cv2.IMREAD_UNCHANGED).astype(np.float64)
    return ncc(cv2.resize(a, (512, 512), interpolation=cv2.INTER_AREA), b)


def mean_profile(paths):
    return np.array([cv2.imread(p, cv2.IMREAD_UNCHANGED).mean() - HU_OFFSET for p in paths])


def best_shift(a, b, min_overlap=200):
    """a[i] <-> b[i+s] が最も合う s を返す"""
    az = (a - a.mean()) / (a.std() + 1e-12)
    bz = (b - b.mean()) / (b.std() + 1e-12)
    best = (None, -2.0)
    for s in range(-len(a) + min_overlap, len(b) - min_overlap):
        i0, i1 = max(0, -s), min(len(a), len(b) - s)
        if i1 - i0 < min_overlap:
            continue
        v = ncc(az[i0:i1], bz[i0 + s:i1 + s])
        if v > best[1]:
            best = (s, v)
    return best


def main():
    print(f"{'case':<9}{'n1024':>6}{'n512':>6}   NCC (1/4, 1/2, 3/4)")
    mismatched, ng = [], []
    for i in range(1, 21):
        case = f"PCD-{i:03d}"
        f1, f5 = slice_paths(PCD1024, case), slice_paths(PCD512, case)
        n = min(len(f1), len(f5))
        scores = [image_ncc(f1[int(n * f)], f5[int(n * f)]) for f in (0.25, 0.5, 0.75)]
        mark = "" if min(scores) > 0.99 else "   <-- 要確認"
        if min(scores) <= 0.99:
            ng.append(case)
        if len(f1) != len(f5):
            mismatched.append(case)
        print(f"{case:<9}{len(f1):>6}{len(f5):>6}   " + "  ".join(f"{s:.4f}" for s in scores) + mark)

    print("\n全症例 NCC > 0.99:", "OK" if not ng else f"NG {ng}")

    if mismatched:
        print(f"\n--- 枚数不一致 {len(mismatched)} 症例の Z シフト照合 ---")
        for case in mismatched:
            f1, f5 = slice_paths(PCD1024, case), slice_paths(PCD512, case)
            s, v = best_shift(mean_profile(f1), mean_profile(f5))
            print(f"{case}: n1024={len(f1)} n512={len(f5)}  best_shift={s}  profile_NCC={v:.5f}"
                  + ("  → 先頭から 1 対 1 対応" if s == 0 else "  → ずれあり、要調整"))


if __name__ == "__main__":
    main()
