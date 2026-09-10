"""
PCD1024_v5 の DICOMSAVE-<日時>-000 フォルダを PatientID ベースの PCD-XXX に
in-place リネームし、中の DICOM も PCD-XXX-YYY.dcm に揃える。

- 症例番号は DICOM の PatientID タグ（001〜020）
- スライス連番はファイル名ソート順（= InstanceNumber 順であることを検証済み）
- ロールバック用に rename_map.json を書き出す
"""
import os, sys, json, pydicom

ROOT = "/Volumes/OTO-SSD/PCD1024_v5"
MAP_PATH = os.path.join(ROOT, "rename_map.json")
DRY = "--apply" not in sys.argv

def main():
    src_dirs = sorted(n for n in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, n)))
    plan, seen = [], {}
    for name in src_dirs:
        d = os.path.join(ROOT, name)
        files = sorted(f for f in os.listdir(d) if f.lower().endswith('.dcm'))
        if not files:
            print(f"[ERROR] {name}: DICOM なし"); sys.exit(1)
        if name.startswith("PCD-"):
            print(f"[SKIP] {name}: 既にリネーム済み"); continue
        ds = pydicom.dcmread(os.path.join(d, files[0]), stop_before_pixels=True)
        pid = str(ds.PatientID).strip()
        case = f"PCD-{int(pid):03d}"
        if case in seen:
            print(f"[ERROR] PatientID 重複: {case} <- {name} / {seen[case]}"); sys.exit(1)
        seen[case] = name
        # InstanceNumber の連番性を検証
        inst = [int(pydicom.dcmread(os.path.join(d, f), stop_before_pixels=True,
                specific_tags=['InstanceNumber']).InstanceNumber) for f in (files[0], files[-1])]
        if inst != [1, len(files)]:
            print(f"[WARN] {name}: InstanceNumber が 1..{len(files)} でない (先頭={inst[0]}, 末尾={inst[1]})")
        # 新ファイル名の衝突チェック（元名は UID なので通常衝突しない）
        newnames = [f"{case}-{i:03d}.dcm" for i in range(1, len(files)+1)]
        if set(newnames) & set(files):
            print(f"[ERROR] {name}: 新旧ファイル名が衝突"); sys.exit(1)
        plan.append((name, case, files, newnames))

    print(f"{'旧フォルダ名':<32} -> {'新':<9} {'枚数':>6}  例")
    for name, case, files, newnames in plan:
        print(f"{name:<32} -> {case:<9} {len(files):>6}  {files[0][-20:]} -> {newnames[0]}")
    print(f"\n合計 {len(plan)} 症例 / {sum(len(p[2]) for p in plan)} ファイル")

    if DRY:
        print("\n[DRY RUN] 実行するには --apply を付けてください")
        return

    mapping = {}
    for name, case, files, newnames in plan:
        d = os.path.join(ROOT, name)
        for old, new in zip(files, newnames):
            os.rename(os.path.join(d, old), os.path.join(d, new))
        os.rename(d, os.path.join(ROOT, case))
        mapping[case] = {"src_dir": name, "files": dict(zip(newnames, files))}
        print(f"  {name} -> {case} ({len(files)} files)", flush=True)
    with open(MAP_PATH, "w") as f:
        json.dump(mapping, f, indent=1)
    print(f"\nロールバック用マッピング: {MAP_PATH}")

main()
