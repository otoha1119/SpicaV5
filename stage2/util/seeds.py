"""stage2/util/seeds.py — 画像ごとの乱数の種（設計書 E-3。stage1/util/seeds.py から複製）。

種 = sha256("<seed>/<症例>/<スライス名>") の先頭 63 bit（設計書は sha1 と書いているが、Stage 1 と同じ sha256 にした）。乱数は **CPU・FP32 で引いてから** GPU へ送る（機械が変わっても同じ乱数）。
同じ機械・同じ重み・同じ設定で同じ絵になり、batch の組み方に依らない。隣のスライスと乱数は共有しない。
"""

import hashlib

import torch


def seed_of(*parts):
    text = "/".join(str(p) for p in parts)
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big") >> 1


def generator_of(device, *parts):
    g = torch.Generator(device=device)
    g.manual_seed(seed_of(*parts))
    return g
