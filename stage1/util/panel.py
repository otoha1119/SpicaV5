"""stage1/util/panel.py — 途中経過のパネルの組み立て。見た目は旧 old/stage2/util/panel.py（白い余白、1px の薄いグレー枠、列の下にラベル帯）から複製。
入出力は RGB float32 [0, 1]。行（度合いごと）を縦に積む grid を足した。
"""

import cv2
import numpy as np

PAD = 8
BORDER = 170 / 255.0
FONT = cv2.FONT_HERSHEY_SIMPLEX


def label_strip(width, text, height=30, scale=0.6):
    img = np.full((height, width, 3), 255, np.uint8)
    while scale > 0.3 and cv2.getTextSize(text, FONT, scale, 1)[0][0] > width - 4:
        scale -= 0.05
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, 1)
    cv2.putText(img, text, (max(2, (width - tw) // 2), (height + th) // 2), FONT, scale, (20, 20, 20), 1, cv2.LINE_AA)
    return img.astype(np.float32) / 255.0


def bordered(img01):
    return cv2.copyMakeBorder(img01, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(BORDER, BORDER, BORDER))


def build_row(cols, labels):
    """cols: (H, W, 3) float01 RGB の列（高さは同じ）。labels: 列ごとの文字列。[列 + ラベル帯] を白の余白で横に並べる。"""
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
    return np.concatenate(row, axis=1)


def build_grid(rows):
    """rows: [(cols, labels), ...]（どの行も同じ列数・同じ大きさ）。行を縦に積む。"""
    imgs = [build_row(c, l) for c, l in rows]
    sep = np.ones((PAD, imgs[0].shape[1], 3), np.float32)
    out = [sep]
    for r in imgs:
        out += [r, sep]
    return np.concatenate(out, axis=0)


def gray(img01):
    return np.repeat(np.asarray(img01, np.float32)[..., None], 3, axis=2)


def zoom(img, x, y, size, scale):
    """左上 (x, y) から size 四方を切り、最近傍で scale 倍。img は (H, W) か (H, W, 3)。"""
    h, w = img.shape[:2]
    if not (0 <= x and 0 <= y and x + size <= w and y + size <= h):
        raise ValueError(f"拡大の範囲 (x {x}, y {y}, size {size}) が画像 {(h, w)} の外")
    c = img[y:y + size, x:x + size]
    return np.repeat(np.repeat(c, scale, axis=0), scale, axis=1)


def to_u8(panel01):
    return (np.clip(panel01, 0.0, 1.0) * 255.0).round().astype(np.uint8)
