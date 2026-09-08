"""
PCD-CT 1024 DICOM → 1ch 16bit PNG 変換（PCD512_v2 と同一仕様）

  HU = pixel_array * RescaleSlope + RescaleIntercept
  HU + 1400 (HU_OFFSET) → clip(0, 65535) → uint16 → 1ch 16bit PNG

入力: /Volumes/OTO-SSD/PCD1024_v5/PCD-XXX/PCD-XXX-YYY.dcm
出力: /Volumes/OTO-SSD/DataSet/PCD1024_v1/PCD-XXX/PCD-XXX-YYY.png
"""
import os, sys, cv2, numpy as np, pydicom
from concurrent.futures import ProcessPoolExecutor

INPUT_DIR  = "/Volumes/OTO-SSD/PCD1024_v5"
OUTPUT_DIR = "/Volumes/OTO-SSD/DataSet/PCD1024_v1"
HU_OFFSET  = 1400
WORKERS    = 8

def convert_case(case):
    src = os.path.join(INPUT_DIR, case)
    dst = os.path.join(OUTPUT_DIR, case)
    os.makedirs(dst, exist_ok=True)
    files = sorted((f for f in os.listdir(src) if f.lower().endswith('.dcm')),
                   key=lambda f: int(f.rsplit('-', 1)[1].split('.')[0]))
    n_done = n_over = 0
    hu_min, hu_max = np.inf, -np.inf
    for f in files:
        out = os.path.join(dst, f[:-4] + ".png")
        if os.path.exists(out):          # 再開時はスキップ
            n_done += 1
            continue
        ds = pydicom.dcmread(os.path.join(src, f))
        hu = ds.pixel_array.astype(np.float64) * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
        hu_min = min(hu_min, hu.min()); hu_max = max(hu_max, hu.max())
        if hu.min() < -HU_OFFSET or hu.max() > (65535 - HU_OFFSET):
            n_over += 1
        img = np.clip(hu + HU_OFFSET, 0, 65535).astype(np.uint16)
        if not cv2.imwrite(out, img):
            raise RuntimeError(f"書き込み失敗: {out}")
        n_done += 1
    return case, len(files), n_done, n_over, hu_min, hu_max

def main():
    only = sys.argv[1:] or None
    cases = sorted(n for n in os.listdir(INPUT_DIR)
                   if n.startswith("PCD-") and os.path.isdir(os.path.join(INPUT_DIR, n)))
    if only:
        cases = [c for c in cases if c in only]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"入力: {INPUT_DIR}\n出力: {OUTPUT_DIR}\n症例数: {len(cases)}\n", flush=True)
    total = over_total = 0
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for case, n, done, over, lo, hi in ex.map(convert_case, cases):
            total += done; over_total += over
            flag = f"  [WARN] uint16 範囲外 {over} 枚" if over else ""
            print(f"  {case}: {done}/{n} 枚  HU {lo:.0f}..{hi:.0f}{flag}", flush=True)
    print(f"\n{'='*50}\n変換完了  症例 {len(cases)} / 総スライス {total}")
    if over_total:
        print(f"  [WARN] uint16 範囲外（クリップ済み）: {over_total} 枚")
    print("="*50)

if __name__ == "__main__":
    main()
