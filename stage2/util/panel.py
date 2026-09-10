"""stage2/util/panel.py — 表示用パネル [EID-like1024 | PCD-like1024 | PCD1024 | 実 EID → PCD-like1024 | 実 EID1024] の組み立て。
stage1/util/panel.py（2026-09-08 ユーザー確定の見た目: 白い余白、画像の周りに 1px の薄いグレー枠、各列の下に白地・黒文字のラベル帯、cv2 Hershey フォント）から
同日に複製し、Stage 2 用に R 列とゲージを外した。並び順は 2026-09-10 ユーザー確定: 入力 → 出力 → 教師 → 実 EID の出力 → 実 EID の入力（元の EID を右端に）。
学習（util/monitor.py save_full_images: 固定 5 列、ランダム 3 列）と推論（inference_dir.py --save_panel: 教師・実 EID の有無で 2〜5 列）が同じ full_labels を使う。

入出力はすべて RGB float32 [0,1]（呼び出し側が 8bit / 16bit に量子化する）。
"""

import cv2
import numpy as np

PAD = 8            # 余白（白）
BORDER = 170 / 255.0  # 画像の 1px 枠
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _text_img(width, height, text, scale, color=(20, 20, 20)):
    """白地に黒文字の帯を 8bit で描いて float01 RGB に。"""
    img = np.full((height, width, 3), 255, np.uint8)
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, 1)
    cv2.putText(img, text, ((width - tw) // 2, (height + th) // 2), FONT, scale, color, 1, cv2.LINE_AA)
    return img.astype(np.float32) / 255.0


def label_strip(width, text, height=30, scale=0.6):
    """画像の下に付けるラベル帯（画像には重ねない）。"""
    return _text_img(width, height, text, scale)


def bordered(img01):
    """画像の周りに 1px の薄いグレー枠。"""
    return cv2.copyMakeBorder(img01, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(BORDER, BORDER, BORDER))


def build_panel(cols, labels):
    """cols: (H,W,3) float01 RGB の列。labels: 列ごとの文字列。戻り値 (Hp,Wp,3) float01 RGB。[列 + ラベル帯] を白の余白で横に並べる。"""
    if len(cols) != len(labels):
        raise ValueError(f"cols と labels の数が違います: {len(cols)} vs {len(labels)}")
    blocks = []
    for c, t in zip(cols, labels):
        b = bordered(np.ascontiguousarray(c, dtype=np.float32))
        blocks.append(np.concatenate([b, label_strip(b.shape[1], t)], axis=0))
    sep = np.ones((blocks[0].shape[0], PAD, 3), np.float32)
    row = [sep]
    for b in blocks:
        row += [b, sep]
    panel = np.concatenate(row, axis=1)
    top = np.ones((PAD, panel.shape[1], 3), np.float32)
    return np.concatenate([top, panel, top], axis=0)


def full_labels(model_tag, name=None, teacher=True, eid=None, input_name="EID-like1024", upsample=None):
    """フル画像パネルのラベル。並び順（2026-09-10 ユーザー確定）:
      [入力 | PCD-like1024 (output) | PCD1024 (teacher) | <eid> -> PCD-like1024 | <eid> (real EID input)]
    model_tag : "epoch 12"（学習）| 重みディレクトリ名 "best" / "epoch_030"（推論）
    name      : 入力のスライス名（ランダムのとき、1 列目に添える）
    teacher   : False で 3 列目（教師）を外す（推論で教師の無い入力）
    eid       : (実 EID のスライス名, scale, interp) で 4・5 列目を付ける。None なら 3 列まで
    input_name / upsample : 1 列目の表記（推論で実 EID512 を入力にしたとき "EID_v5 (input, x2 bicubic)" のように）"""
    up = f", x{upsample[0]} {upsample[1]}" if upsample else ""
    labels = [f"{input_name}  (input{up})" + (f"   {name}" if name else ""), f"PCD-like1024  (output)   {model_tag}"]
    if teacher:
        labels.append("PCD1024  (teacher)")
    if eid:
        eid_name, scale, interp = eid
        labels += [f"{eid_name} -> PCD-like1024   (real EID)", f"{eid_name}  (real EID input, x{scale} {interp})"]
    return labels


def eid_labels(model_tag, eid_name, scale, interp):
    """TB images/full/EID（実 EID テストだけの 2 列 [EID1024 | PCD-like1024]。ユーザー指示 2026-09-09）のラベル。"""
    return [f"{eid_name}  (real EID, x{scale} {interp})", f"PCD-like1024  (output)   {model_tag}"]
