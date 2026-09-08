"""util/panel.py — 表示用パネル [EID | EID-like | PCD | R + ゲージ] の組み立て。[SpicaV5]

2026-09-08 ユーザー確定: 並び順は EID（代表）→ EID-like G(z) → PCD z → R。各列の下に白地・黒文字のラベル帯、R の右に小さな縦ゲージ
（上 = +range_hu 赤、中央 = 0 白、下 = −range_hu 青、目盛り付き。補足扱いなので小さめ）。余白は白、画像の周りに 1px の薄いグレー枠
（R の白 = 0 が余白に溶けないように）。文字は cv2 の Hershey フォント（英数のみ。TrueType は商用フォントをリポジトリに入れられないため見送り）。

入出力はすべて RGB float32 [0,1]（呼び出し側が 8bit / 16bit に量子化する。画像列の 16bit 階調を落とさないため、文字帯だけ 8bit で描いて変換する）。
使う所: util/monitor.py save_full_images（TB images/full/* と output_images/preview_*/ のパネル）
"""

import cv2
import numpy as np

from util.residual_color import residual_rgb01

PAD = 8            # 余白（白）
BORDER = 170 / 255.0  # 画像の 1px 枠
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _text_img(width, height, text, scale, color=(20, 20, 20), center=True, x=0, baseline=None):
    """白地に黒文字の帯を 8bit で描いて float01 RGB に。"""
    img = np.full((height, width, 3), 255, np.uint8)
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, 1)
    if center:
        x = (width - tw) // 2
    y = (height + th) // 2 if baseline is None else baseline
    cv2.putText(img, text, (int(x), int(y)), FONT, scale, color, 1, cv2.LINE_AA)
    return img.astype(np.float32) / 255.0


def label_strip(width, text, height=30, scale=0.6):
    """画像の下に付けるラベル帯（画像には重ねない）。"""
    return _text_img(width, height, text, scale)


def gauge(height, range_hu, bar_w=12, width=72, bar_h_ratio=0.55, ticks=(1.0, 0.5, 0.0, -0.5, -1.0), scale=0.36):
    """縦ゲージ（補足扱いで小さめ）。上 = +range_hu（赤）、中央 = 0（白）、下 = −range_hu（青）。右に目盛りとラベル、上に HU。"""
    img = np.full((height, width, 3), 255, np.uint8)
    bar_h = int(height * bar_h_ratio)
    y0 = (height - bar_h) // 2
    y1 = y0 + bar_h
    x0 = 10
    t = np.linspace(range_hu, -range_hu, y1 - y0, dtype=np.float32)[:, None].repeat(bar_w, 1)
    img[y0:y1, x0 : x0 + bar_w] = (residual_rgb01(t, range_hu) * 255.0).round().astype(np.uint8)
    cv2.rectangle(img, (x0 - 1, y0 - 1), (x0 + bar_w, y1), (90, 90, 90), 1)  # 枠（白の中央が背景に溶けないように）
    for frac in ticks:
        v = int(round(frac * range_hu))
        y = int(round(y0 + (1.0 - frac) / 2.0 * (y1 - y0 - 1)))
        cv2.line(img, (x0 + bar_w + 1, y), (x0 + bar_w + 5, y), (40, 40, 40), 1)
        cv2.putText(img, f"{v:+d}" if v else "0", (x0 + bar_w + 8, y + 4), FONT, scale, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.putText(img, "HU", (x0, y0 - 8), FONT, scale, (30, 30, 30), 1, cv2.LINE_AA)
    return img.astype(np.float32) / 255.0


def bordered(img01):
    """画像の周りに 1px の薄いグレー枠。"""
    return cv2.copyMakeBorder(img01, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(BORDER, BORDER, BORDER))


def build_panel(cols, labels, range_hu):
    """cols: (H,W,3) float01 RGB の列（最後が R）。labels: 列ごとの文字列。戻り値 (Hp,Wp,3) float01 RGB。
    [列 + ラベル帯] を白の余白で横に並べ、最後に R のゲージ。"""
    if len(cols) != len(labels):
        raise ValueError(f"cols と labels の数が違います: {len(cols)} vs {len(labels)}")
    H = cols[0].shape[0]
    blocks = []
    for c, t in zip(cols, labels):
        b = bordered(np.ascontiguousarray(c, dtype=np.float32))
        blocks.append(np.concatenate([b, label_strip(b.shape[1], t)], axis=0))
    g = gauge(H + 2, range_hu)
    g = np.concatenate([g, np.ones((blocks[0].shape[0] - g.shape[0], g.shape[1], 3), np.float32)], axis=0)
    sep = np.ones((blocks[0].shape[0], PAD, 3), np.float32)
    row = [sep]
    for b in blocks:
        row += [b, sep]
    row += [g, sep]
    panel = np.concatenate(row, axis=1)
    top = np.ones((PAD, panel.shape[1], 3), np.float32)
    return np.concatenate([top, panel, top], axis=0)


def full_labels(epoch):
    """フル画像パネルのラベル（並び順 EID → EID-like → PCD → R）。"""
    return ["EID", f"EID-like  G(z)   epoch {int(epoch)}", "PCD  z", "R = G(z) - z   [HU]"]
