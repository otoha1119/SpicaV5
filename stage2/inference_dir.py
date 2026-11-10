"""stage2/inference_dir.py — フォルダ丸ごとの推論の本体（設計書 E-3・H-2）。run_infer.py が設定を検証してから呼ぶ。

  入力  <input_dir>/<症例>/<スライス>.png か <input_dir>/<スライス>.png（1ch 16bit。infer.upsample: true なら 512 → 学習と同じ補間で 1024 に。false なら既に 1024）
  出力  <out_root>/<run>_<重み>_<入力名>_<時刻>/
          full/<症例>/<スライス>.png            最終出力（mode uv = U + V の 1 つ目の種。mode u = μ）。1024、1ch 16bit、stored = HU + 1400
          full_seed01/…, full_seed02/…          infer.n_seeds ≥ 2 のときの 2 つ目以降の種
          full_mu/…                             U の出力 μ（infer.save_mu。mode uv のとき）
          full_input1024/…                      補間した入力（infer.save_input1024。四捨五入して保存）
          full_panel/<症例>/<スライス>.png      [入力↑ | μ | U+V] の表示用パネル（infer.save_panel）
          infer.yaml                            実効設定、重み、枚数、1 枚あたりの秒数
  1024 一枚をそのまま通す（タイルなし → つなぎ目なし）。U・明るさの固定・V の入口と出口は FP32、V の本体だけ infer.precision。
  乱数はスライスごと（sha256('<seed>/<症例>/<スライス名>')、CPU で引く）。同じ seed なら同じ出力。
"""

import datetime
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from check.seed_diff import sample_seeds
from data.ct_io import CaseIndex, hu_tuple, load_n, n_to_stored, write_png
from models.pipeline import load_pipeline
from util.panel import build_grid, to_u8
from util.run_paths import JST
from util.views import View


def list_tasks(input_dir, infer):
    idx = CaseIndex(input_dir, cases=infer["infer.cases"] or None)  # 評価専用も除外しない（推論は全症例を見る）
    tasks = []
    for case in idx.cases:
        paths = idx.slices[case][::infer["infer.slice_step"]]
        if infer["infer.max_slices"] > 0:
            paths = paths[:infer["infer.max_slices"]]
        tasks += [(case, Path(p)) for p in paths]
    return idx, tasks


def run(infer, weight_dir, run_name, device):
    kind, mode = infer["infer.kind"], infer["infer.mode"]
    pipe, cfg, step = load_pipeline(weight_dir, kind, device, v_bf16=(infer["infer.precision"] == "bf16"))
    data = cfg["data"]
    hu = hu_tuple(data)
    scale = data["data.upsample_scale"] if infer["infer.upsample"] else 1
    sigma_s, steps = infer["infer.sigma_s_hu"], infer["infer.steps"]
    if mode == "uv":
        lo, hi = cfg["v"]["flow.sigma_s_min_hu"], cfg["v"]["flow.sigma_s_max_hu"]
        if not (lo <= sigma_s <= hi):
            raise RuntimeError(f"infer.sigma_s_hu {sigma_s} が、V が学習した幅 [{lo}, {hi}] HU の外です")
    input_dir = Path(infer["infer.input_dir"])
    idx, tasks = list_tasks(input_dir, infer)
    if not tasks:
        raise RuntimeError(f"推論する対象がありません: {input_dir}")
    now = datetime.datetime.now(JST)
    out = Path(infer["infer.out_root"]) / f"{run_name}_{Path(weight_dir).name}_{input_dir.name}_{now.strftime('%Y_%m%d_%H%M%S')}"
    if out.exists():
        raise RuntimeError(f"出力フォルダが既にあります: {out}")
    seeds = [infer["infer.seed"] + i for i in range(infer["infer.n_seeds"])] if mode == "uv" else []
    u_cfg = cfg["u"] if kind == "u" else cfg["u_config"]["u"]
    mult = max(2 ** u_cfg["net.n_pool"], pipe.v_net.multiple if mode == "uv" else 1)  # どちらも 2 のべき
    view = View(data)
    print(f"[infer] {out.name}: 重み {weight_dir}（step {step}、EMA）、mode {mode}、入力 {input_dir}（{idx.n_cases} 症例 / {len(tasks)} 枚、補間 ×{scale}）"
          + (f"、σs {sigma_s:g} HU、Euler {steps} 回、種 {seeds}、V の本体 {'bf16' if pipe.v_bf16 else 'fp32'}" if mode == "uv" else "") + f"、明るさの広さ σ {pipe.sigma_px:g} 画素", flush=True)
    out.mkdir(parents=True)
    t0, size = time.time(), None
    for i, (case, p) in enumerate(tasks):
        x = torch.from_numpy(load_n(p, hu, scale, data["data.upsample_interp"]))[None, None].to(device)
        size = size or tuple(x.shape[-2:])
        if tuple(x.shape[-2:]) != size:
            raise RuntimeError(f"画像の大きさが混ざっています: {p} {tuple(x.shape[-2:])} ≠ {size}")
        if size[0] % mult or size[1] % mult:
            raise RuntimeError(f"入力（補間後 {size}）の一辺は {mult} の倍数であること: {p}")
        mu, _ = pipe.mu(x)
        ys = sample_seeds(pipe, x, mu, case, p.name, seeds, sigma_s, steps) if mode == "uv" else []
        final = ys[0] if ys else mu
        if not bool(torch.isfinite(final).all()):
            raise RuntimeError(f"出力に NaN / Inf があります: {p}")
        rel = Path(case) / p.name if case != "_root" else Path(p.name)
        save = lambda sub, n: write_png(out / sub / rel, n_to_stored(n[0, 0].float().cpu().numpy(), hu))
        save("full", final)
        for j, y in enumerate(ys[1:], start=1):
            save(f"full_seed{j:02d}", y)
        if infer["infer.save_mu"] and mode == "uv":
            save("full_mu", mu)
        if infer["infer.save_input1024"]:
            save("full_input1024", x)
        if infer["infer.save_panel"]:
            cols = [(x, f"input up ({p.stem})"), (mu, "mu = U")] + ([(ys[0], f"U+V (sigma_s {sigma_s:g}HU, K {steps})")] if ys else [])
            panel = build_grid([([view.img(pipe.units.hu(n)) for n, _ in cols], [l for _, l in cols])])
            write_png(out / "full_panel" / rel, np.ascontiguousarray(to_u8(panel)[..., ::-1]))
        if i == 0 or (i + 1) % 20 == 0 or i + 1 == len(tasks):
            el = time.time() - t0
            print(f"[infer] {i + 1}/{len(tasks)} 枚、{el / (i + 1):.2f} s/枚、残り 約 {(len(tasks) - i - 1) * el / (i + 1) / 60:.0f} 分", flush=True)
    with open(out / "infer.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"created": now.isoformat(timespec="seconds"), "run": run_name, "weight_dir": str(weight_dir), "weight_step": step, "weights": "ema",
                        "infer": infer, "n_cases": idx.n_cases, "n_images": len(tasks), "image_hw": list(size), "brightness_sigma_px_1024": pipe.sigma_px,
                        "seeds": seeds, "seed_rule": "sha256('<seed>/<case>/<slice file name>') >> 1 (first 8 bytes), drawn on CPU",
                        "sec_per_image": (time.time() - t0) / len(tasks), "run_config": {k: v for k, v in cfg.items() if k != "machine"}}, f, allow_unicode=True, sort_keys=False)
    print(f"[infer] 完了: {out}（{len(tasks)} 枚）", flush=True)
    return out
