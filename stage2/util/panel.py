"""stage2/util/panel.py — 表示用パネル [EID-like1024 | PCD1024 | PCD-like1024] の組み立て。
stage1/util/panel.py（2026-09-08 ユーザー確定の見た目: 白い余白、画像の周りに 1px の薄いグレー枠、各列の下に白地・黒文字のラベル帯、cv2 Hershey フォント）から
同日に複製し、Stage 2 用に R 列とゲージを外した（並べるのは 3 列だけ。ユーザー指示 2026-09-08）。

入出力はすべて RGB float32 [0,1]（呼び出し側が 8bit に量子化する）。使う所: util/monitor.py log_full_images（TB images/full/*）
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


def full_labels(epoch, name=None, eid_name=None):
    """フル画像パネルのラベル（並び順 EID-like1024 → PCD1024 → PCD-like1024 [→ 実 EID テスト]。ユーザー指示 2026-09-08/09）。
    name はスライス名（ランダムのとき）、eid_name は実 EID テストスライス名（固定パネルの 4 列目。None なら 3 列）。"""
    tail = f"   {name}" if name else ""
    labels = [f"EID-like1024  (input){tail}", "PCD1024  (teacher)", f"PCD-like1024  (output)   epoch {int(epoch)}"]
    if eid_name:
        labels.append(f"{eid_name} -> PCD-like1024   (real EID test)")
    return labels
