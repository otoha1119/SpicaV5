# create_dataset

CT データセット（DICOM → 学習用 PNG）の作成スクリプト。実行はホストの Python から（Docker 不要）。

## PCD-CT 1024（2026-09-08 作成）

| スクリプト | 役割 |
|---|---|
| `rename_pcd1024.py` | `/Volumes/OTO-SSD/PCD1024_v5` の `DICOMSAVE-<日時>-000` を `PCD-XXX/PCD-XXX-YYY.dcm` に in-place リネーム。`--apply` なしは dry run |
| `convert_pcd1024.py` | 上記 DICOM を 1ch 16bit PNG に変換して `/Volumes/OTO-SSD/DataSet/PCD1024_v1` へ出力。既存 PNG はスキップするので中断からの再開が効く |
| `verify_pcd1024_pairing.py` | 生成した 1024 と既存 `PCD512_v2` のペア対応を NCC で検証 |

### 症例番号は PatientID タグで決める

元フォルダ名の日時順とは一致しない（例: `DICOMSAVE-20240527074328-000` が PID=006）。
卒論時の `SeniorThesis/DataSet/photonCT/PhotonCT1024v3` は日時順で振られており、512 と番号がズレているので使わないこと。

リネームのロールバック用マッピングは `/Volumes/OTO-SSD/PCD1024_v5/rename_map.json`。

### PNG 変換仕様（`PCD512_v2` / `EID_v5` と共通）

```
HU  = pixel_array * RescaleSlope + RescaleIntercept
PNG = clip(HU + 1400, 0, 65535) を uint16 の 1ch 16bit PNG
```

### PCD512_v2 とのペア

全20症例が先頭スライスから 1 対 1 で対応する（検証済み、NCC 0.9973〜0.9996）。
ただし末尾の枚数が 3 症例で異なるため `min(n1024, n512)` までを使う:
`PCD-006` 1251/1290、`PCD-011` 1058/1075、`PCD-015` 1290/1264。
