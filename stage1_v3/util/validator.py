"""stage1_v3/util/validator.py — epoch ごとの自動採点と best/ の自動更新（2026-10-30、1 本目の run で HU 水準が epoch ごとに揺れたため。GAN は最終 epoch が最良とは限らない）。

  固定集合（run の最初に 1 回だけ決める。seed = data.patch_seed）:
    PCD: 学習症例から val.n_pcd_slices 枚（評価専用症例は含まない）。各 epoch で P を 1 回（固定 seed）掛けて G に通し、EID-like のフル 512 を作る
    EID: val.n_eid_slices 枚。参照統計（中央値と四分位）は最初に 1 回計算。各 epoch で G に通し（P なし）、恒等残差 RMS(R(y)) を eval mode で測る
  指標（util/metrics.analyze_slice、R-S22 の 3 軸）: local_sd（ノイズ）, nps_f_av（NPS の形）, edge_grad_p99（鮮鋭度）, hu_p50, hu_p90（HU 水準）
  score = Σ_k |median_gen,k − median_eid,k| / IQR_eid,k  +  RMS(R(y))_eval / val.target_ry_rms_hu   （小さいほど良い。0 なら全指標が EID の中央値に一致し恒等残差 0）
  TensorBoard: val/<k>_gen, val/<k>_eid（参照、定数）, val/<k>_dist（正規化距離）, val/ry_rms_HU, val/score, val/best_score
  ディスク: <run>/val_log.txt（1 epoch 1 行）。auto_best なら score 最良で model.save(run_dir, "best") + best.txt
"""

import random
import time
from pathlib import Path

import numpy as np
import torch

from data.dataset import _load_hu, hu_to_norm
from util.metrics import analyze_slice, summarize
from util.run_paths import BEST_NOTE

KEYS = ("local_sd", "nps_f_av", "edge_grad_p99", "hu_p50", "hu_p90")
VAL_LOG_FILE = "val_log.txt"


class Validator:
    def __init__(self, dataset, train, run_dir, tb):
        self.c = train
        self.run_dir = Path(run_dir)
        self.tb = tb
        self.enabled = train["val.n_pcd_slices"] > 0
        if not self.enabled:
            return
        self.ds = dataset
        self.thr = float(train["degrade.body_threshold_hu"])
        self.offset = train["data.hu_offset"]
        self.target_ry = train["val.target_ry_rms_hu"]
        seed = train["data.patch_seed"] * 1_000_003 + 4242
        rng = random.Random(seed)
        self.pcd_paths = rng.sample(dataset.idx_a.all_paths, min(train["val.n_pcd_slices"], dataset.idx_a.n_slices))
        self.eid_paths = rng.sample(dataset.idx_b.all_paths, min(train["val.n_eid_slices"], dataset.idx_b.n_slices))
        self.np_rng_seed = seed
        self.best_score = float("inf")
        self.best_epoch = None
        t0 = time.time()
        rows = [analyze_slice(_load_hu(p, self.offset), self.thr, np.random.default_rng(seed)) for p in self.eid_paths]
        self.ref = {k: summarize(rows, k) for k in KEYS}
        for k in KEYS:
            if self.ref[k]["n"] < 8 or not np.isfinite(self.ref[k]["median"]):
                raise RuntimeError(f"参照 EID の指標 {k} が計算できません（有効 {self.ref[k]['n']} 枚）。val.n_eid_slices を増やすか、体内判定 degrade.body_threshold_hu を確認")
            self.ref[k]["iqr"] = max(self.ref[k]["q75"] - self.ref[k]["q25"], 1e-6)
        with open(self.run_dir / VAL_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"================ Validation ({time.strftime('%Y-%m-%d %H:%M:%S')}) PCD {len(self.pcd_paths)} / EID {len(self.eid_paths)} ================\n")
            f.write("ref_eid " + " ".join(f"{k}={self.ref[k]['median']:.3f}[{self.ref[k]['q25']:.3f},{self.ref[k]['q75']:.3f}]" for k in KEYS) + "\n")
        print(f"[val] 参照 EID {len(self.eid_paths)} 枚の統計 ({time.time() - t0:.1f}s): " + " ".join(f"{k} {self.ref[k]['median']:.2f}" for k in KEYS))

    @torch.no_grad()
    def run(self, model, epoch, total_iters):
        """epoch 末に呼ぶ。戻り値 (score, is_best)。enabled でなければ (None, False)。"""
        if not self.enabled:
            return None, False
        t0 = time.time()
        rng = np.random.default_rng(self.np_rng_seed + int(epoch))
        hu_min, hu_max = self.c["data.hu_min"], self.c["data.hu_max"]
        to_hu = lambda t: ((t.float().cpu().numpy()[0, 0] + 1.0) / 2.0 * (hu_max - hu_min) + hu_min)  # noqa: E731
        rows = []
        for p in self.pcd_paths:
            hu = _load_hu(p, self.offset)
            u, _ = self.ds.degrade_full(hu, rng)
            y = model.predict(torch.from_numpy(hu_to_norm(u, hu_min, hu_max))[None, None])
            rows.append(analyze_slice(to_hu(y), self.thr, rng))
        gen = {k: summarize(rows, k) for k in KEYS}
        dist = {k: abs(gen[k]["median"] - self.ref[k]["median"]) / self.ref[k]["iqr"] if np.isfinite(gen[k]["median"]) else float("nan") for k in KEYS}
        sq, n = 0.0, 0
        for p in self.eid_paths:
            hu = _load_hu(p, self.offset)
            yn = torch.from_numpy(hu_to_norm(hu, hu_min, hu_max))[None, None]
            r = to_hu(model.predict(yn)) - np.clip(hu, hu_min, hu_max)
            sq += float((r ** 2).mean()); n += 1
        ry = float(np.sqrt(sq / max(n, 1)))
        score = sum(dist.values()) + ry / self.target_ry
        is_best = bool(np.isfinite(score) and score < self.best_score)
        if is_best:
            self.best_score, self.best_epoch = score, int(epoch)
        for k in KEYS:
            self.tb.add_scalar(f"val/{k}_gen", gen[k]["median"], total_iters)
            self.tb.add_scalar(f"val/{k}_eid", self.ref[k]["median"], total_iters)
            self.tb.add_scalar(f"val/{k}_dist", dist[k], total_iters)
        self.tb.add_scalar("val/ry_rms_HU", ry, total_iters)
        self.tb.add_scalar("val/score", score, total_iters)
        self.tb.add_scalar("val/best_score", self.best_score, total_iters)
        line = (f"val epoch {int(epoch)} | score {score:.3f}{' *best*' if is_best else ''} | ry_rms {ry:.1f} HU | "
                + " ".join(f"{k} {gen[k]['median']:.2f} (eid {self.ref[k]['median']:.2f}, d {dist[k]:.2f})" for k in KEYS) + f" | {time.time() - t0:.1f}s")
        with open(self.run_dir / VAL_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        print(line, flush=True)
        if is_best and self.c["val.auto_best"]:
            model.save(self.run_dir, "best")
            with open(self.run_dir / BEST_NOTE, "w", encoding="utf-8") as f:
                f.write(f"epoch: {int(epoch)}\nscore: {score:.4f}\nmarked_at: {time.strftime('%Y-%m-%dT%H:%M')}\nsource: auto (util/validator.py, R-S22 3 軸 + RMS(R(y)))\n"
                        + "".join(f"{k}: gen {gen[k]['median']:.3f} eid {self.ref[k]['median']:.3f} dist {dist[k]:.3f}\n" for k in KEYS) + f"ry_rms_HU: {ry:.2f}\n")
        return score, is_best
