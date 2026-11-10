"""stage1/util/seeds.py — 画像ごとの乱数の種（設計書 F-3）。

種 = sha256("<base_seed>/<症例>/<スライス名>/<k>") の先頭 63 bit。画像ごとに乱数源を 1 つ持ち、そこから手がかりのノイズ → 描き始め → 各段の乱数を順に引く。
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
