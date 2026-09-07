"""util/dicom_io.py — 推論結果を元の DICOM に戻して書き出す（output_format = dicom | both）。[SpicaV5 新規]

前処理（SpicaV3 create_dataset/convert_pcd.py）の逆をやる:
  変換: DICOM → pixel × RescaleSlope + RescaleIntercept = HU → HU + 1400 → uint16 PNG、症例フォルダ名の数値部分 n → "PCD-nnn"、
        症例フォルダ内の *.dcm を再帰的に集めて**ファイル名順にソート**し、1 始まりの連番 → "PCD-nnn-sss.png"（512 のまま。切出し・リサンプルは無い）
  逆:   出力 PNG の名前 PCD-nnn-sss → 元 DICOM ルートの「数値部分が n の症例フォルダ」の sss 番目（同じソート順）の .dcm を参照として読み、
        EID-like の HU を infer.yaml の dicom.rescale_slope / rescale_intercept（参照のタグと一致することを検査）と参照の
        BitsStored / PixelRepresentation で格納値に戻し、ヘッダをそのまま継承して保存
        実データ（Siemens NAEOTOM Alpha）: slope 1, intercept −8192, 符号なし 16bit（2026-09-07 に DICOMSAVE-20240527074809-000 で確認）
        （SOPInstanceUID / SeriesInstanceUID は新規発行、SeriesDescription に由来を書く）。SpicaV2 inference_single.py の save_like_reference と同じ考え方

前提: 元 DICOM ルートは変換時と同じ構成（症例フォルダの数値部分と、フォルダ内 .dcm の名前順が変わっていない）であること。
      確認できない構成（名前が PCD-nnn-sss でない、スライス番号が範囲外、症例フォルダが無い）は**エラーで止める**（推測で埋めない）。
      さらに書く直前に「参照 DICOM から再現した stored 値 == 入力 PNG」を全画素で照合する（check_pixels。順序ズレの検出）。
      対応範囲は単一 Series・単一フレーム CT（check_series_unique）。出力は ImageType DERIVED\\SECONDARY、SeriesNumber = 元 + 1000。
"""

import os
import re
from pathlib import Path

import numpy as np
import pydicom
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

PNG_NAME_RE = re.compile(r"^(?P<prefix>[A-Za-z]+)-(?P<case>\d+)-(?P<slice>\d+)$")  # PCD-012-034


def parse_png_stem(stem):
    """'PCD-012-034' → (12, 34)。合わなければ ValueError。"""
    m = PNG_NAME_RE.match(stem)
    if not m:
        raise ValueError(f"PNG 名が <PREFIX>-<症例番号>-<スライス番号> の形ではないので元 DICOM を特定できません: {stem}")
    return int(m.group("case")), int(m.group("slice"))


def _numeric_key(name):
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else None


class DicomIndex:
    """元 DICOM ルート → {症例番号: [dcm パス（名前順）]}。convert_pcd.py と同じ規則。"""

    def __init__(self, root):
        root = Path(root)
        if not root.is_dir():
            raise RuntimeError(f"DICOM ルートが存在しません: {root}")
        self.root = root
        self.cases = {}
        for entry in sorted(root.iterdir()):
            if not entry.is_dir():
                continue
            n = _numeric_key(entry.name)
            if n is None:
                continue
            files = []
            for d, _, names in os.walk(entry):
                files += [os.path.join(d, f) for f in names if f.lower().endswith(".dcm")]
            files.sort()
            if files:
                if n in self.cases:
                    raise RuntimeError(f"症例番号 {n} のフォルダが複数あります（{self.cases[n][0]} と {files[0]}）。convert_pcd.py と同じく数値部分で識別するので一意である必要がある")
                self.cases[n] = files
        if not self.cases:
            raise RuntimeError(f"DICOM ルートに .dcm を含む症例フォルダがありません: {root}")

    def reference_for(self, stem):
        """出力 PNG の stem → 参照 DICOM のパス。"""
        case, idx = parse_png_stem(stem)
        if case not in self.cases:
            raise KeyError(f"症例番号 {case}（{stem}）に対応する DICOM フォルダがありません: {self.root}")
        files = self.cases[case]
        if not (1 <= idx <= len(files)):
            raise IndexError(f"{stem}: スライス番号 {idx} は症例 {case} の DICOM 数 {len(files)} の範囲外（連番は 1 始まり）")
        return files[idx - 1]


def expected_stored(ds, hu_offset):
    """参照 DICOM から、convert_pcd.py と同じ演算で PNG の stored 値を再現する（HU = pixel × slope + intercept → HU + offset → clip → uint16 に切り捨て）。"""
    hu = ds.pixel_array.astype(np.float64) * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
    return np.clip(hu + hu_offset, 0, 65535).astype(np.uint16)


def check_pixels(ref_path, stored_png, hu_offset, where=""):
    """入力 PNG が本当にこの参照 DICOM から作られたものか、画素で照合する（F-18a）。名前順の対応だけでは順序ズレを検出できないため。"""
    ds = pydicom.dcmread(ref_path)
    exp = expected_stored(ds, hu_offset)
    if exp.shape != stored_png.shape:
        raise ValueError(f"参照 DICOM {exp.shape} と入力 PNG {stored_png.shape} の形状が違います: {where or ref_path}")
    if not np.array_equal(exp, stored_png):
        n = int((exp != stored_png).sum())
        raise ValueError(f"入力 PNG と参照 DICOM の画素が一致しません（{n} 画素）。症例フォルダの数値・.dcm の名前順が変換時と違う可能性: {where or ref_path}")
    return ds


def check_series_unique(index, rels_by_case):
    """症例ごとに参照 DICOM の SeriesInstanceUID が 1 種であることを確認する（F-18c。単一 Series・単一フレーム CT に限定）。"""
    for case, stems in rels_by_case.items():
        uids = set()
        for stem in stems:
            ref = index.reference_for(stem)
            ds = pydicom.dcmread(ref, stop_before_pixels=True, specific_tags=["SeriesInstanceUID", "NumberOfFrames"])
            if int(getattr(ds, "NumberOfFrames", 1) or 1) != 1:
                raise ValueError(f"多フレーム DICOM は未対応: {ref}")
            uids.add(str(getattr(ds, "SeriesInstanceUID", "")))
        if len(uids) != 1:
            raise ValueError(f"症例 {case} の参照 DICOM に複数の Series が混ざっています（{len(uids)} 種）。単一 Series の症例フォルダにしてください")


def check_rescale(ds, slope, intercept, where=""):
    """参照 DICOM の RescaleSlope / Intercept が infer.yaml の宣言値と一致するか。無い・違う → エラー（黙って別の値で書かない）。"""
    if "RescaleSlope" not in ds or "RescaleIntercept" not in ds:
        raise ValueError(f"参照 DICOM に RescaleSlope / RescaleIntercept がありません: {where}")
    rs, ri = float(ds.RescaleSlope), float(ds.RescaleIntercept)
    if abs(rs - slope) > 1e-9 or abs(ri - intercept) > 1e-9:
        raise ValueError(f"参照 DICOM の Rescale (slope {rs}, intercept {ri}) が infer.yaml の dicom.rescale_* (slope {slope}, intercept {intercept}) と違います: {where}")


def hu_to_stored_like_ref(hu, ds, slope, intercept):
    """HU → 格納値 (HU − intercept) / slope。BitsStored / PixelRepresentation（参照 DICOM）の範囲でクリップ。"""
    if slope == 0:
        raise ValueError("rescale_slope が 0")
    stored = (np.asarray(hu, dtype=np.float64) - intercept) / slope
    bits = int(getattr(ds, "BitsStored", 16))
    signed = int(getattr(ds, "PixelRepresentation", 1)) == 1
    if signed:
        lo, hi, dtype = -(1 << (bits - 1)), (1 << (bits - 1)) - 1, np.int16
    else:
        lo, hi, dtype = 0, (1 << bits) - 1, np.uint16
    return np.clip(np.rint(stored), lo, hi).astype(dtype)


def write_like_reference(ref_path, hu, out_path, series_uid, description, slope, intercept, ds=None):
    """参照 DICOM のヘッダを継承し、画素だけ EID-like（HU）に置き換えて out_path に保存する。
    hu の形状は参照と同じでなければならない（512 のまま処理しているので通常同じ。違えばエラー）。
    slope / intercept は infer.yaml の宣言値。参照のタグと一致しなければエラー。ds を渡せば再読込しない（check_pixels の戻り値）。"""
    if ds is None:
        ds = pydicom.dcmread(ref_path)
    rows, cols = int(ds.Rows), int(ds.Columns)
    if tuple(hu.shape) != (rows, cols):
        raise ValueError(f"出力 {hu.shape} と参照 DICOM {(rows, cols)} の形状が違います: {ref_path}")
    check_rescale(ds, slope, intercept, ref_path)
    stored = hu_to_stored_like_ref(hu, ds, slope, intercept)
    ds.RescaleSlope = slope
    ds.RescaleIntercept = intercept
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = int(getattr(ds, "BitsStored", 16))
    ds.HighBit = ds.BitsStored - 1
    ds.PixelRepresentation = 1 if stored.dtype == np.int16 else 0
    ds.PixelData = stored.tobytes()
    for tag in ("SmallestImagePixelValue", "LargestImagePixelValue"):  # 参照の値は無効になるので消す
        if tag in ds:
            del ds[tag]
    ds.SOPInstanceUID = generate_uid()
    ds.SeriesInstanceUID = series_uid
    ds.SeriesNumber = int(getattr(ds, "SeriesNumber", 0) or 0) + 1000  # 元シリーズと区別（F-18b）
    orig_type = [str(v) for v in (getattr(ds, "ImageType", None) or [])]
    ds.ImageType = ["DERIVED", "SECONDARY"] + orig_type[2:]  # 画素から生成した画像は DERIVED（DICOM PS3.3 C.7.6.1.1.2）。3 値目以降は継承
    ds.SeriesDescription = description[:64]  # LO は 64 文字まで
    ds.DerivationDescription = description[:1024]
    ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian  # 参照が圧縮でも生の画素で書く
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(out_path), enforce_file_format=True)
    return ds
