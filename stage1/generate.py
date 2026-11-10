"""stage1/generate.py — 疑似 EID の作成の本体（設計書 F）。run_generate.py が設定を検証してから呼ぶ。

  入力  <input_dir>/<症例>/<スライス>.png（512、1ch 16bit。PCD512 でも実 EID でも同じ手順）
  出力  <pseudo_eid_dir>/<組>/k00, k01, …/<症例>/<スライス>.png（512、1ch 16bit、stored = HU + 1400。入力と同じファイル名）+ <組>/manifest.yaml
        <組>.tmp に書く → 枚数を検算 → <組> に改名。既にある組には書かない。

  手順は diffusion/sampler.py（手がかり → 広い平均 → ただのノイズ + 広い平均から Euler ancestral、毎段に錨）。重みは EMA。
  U-Net 本体だけ bf16（sample.precision）、ほかは FP32。sample.compile のときは batch を固定し、端数は最後の 1 枚を詰めて捨てる。
  乱数は画像ごと（util/seeds.py: sha256('<seed>/<症例>/<スライス名>/<k>')）。batch の組み方に依らず同じ絵になる。
"""

import datetime
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from data.ct_io import CaseIndex, hu_tuple, read_stored, stored_to_x, write_png, x_to_stored
from diffusion.sampler import redraw
from models.painter import denoise, load_ema, want_bf16
from util.run_paths import JST
from util.seeds import generator_of

REPO = Path(__file__).resolve().parents[1]


def git_commit():
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        try:  # コンテナに git が無いとき: .git を直接読む
            head = (REPO / ".git" / "HEAD").read_text(encoding="utf-8").strip()
            return (REPO / ".git" / head[5:]).read_text(encoding="utf-8").strip() if head.startswith("ref: ") else head
        except OSError:
            return "unknown"


def group_name(run, step, gen, sigma_a):
    cue = f"b{gen['cue.blur_px']:.1f}_c{int(round(gen['cue.noise_hu'])):03d}" if gen["cue.use"] else "nocue"
    name = f"{run}_step{step:07d}_{cue}_a{sigma_a:g}"
    return name + (f"_{gen['generate.name_suffix']}" if gen["generate.name_suffix"] else "")


def list_tasks(input_dir, gen):
    idx = CaseIndex(input_dir, cases=gen["generate.cases"] or None)  # 評価専用も除外しない（推論側は全症例を見る）
    tasks = []
    for case in idx.cases:
        paths = idx.slices[case][::gen["generate.slice_step"]]
        if gen["generate.max_slices"] > 0:
            paths = paths[:gen["generate.max_slices"]]
        tasks += [(case, Path(p)) for p in paths]
    return idx, tasks


def run(gen, shared, machine, weight_dir, run_name, device):
    net, cfg, step = load_ema(weight_dir, device)
    train = cfg["train"]
    hu = hu_tuple(train)
    scale = train["data.hu_scale"]
    sigma_a = shared["brightness.sigma_px_512"]
    input_dir = Path(gen["generate.input_dir"] or machine["pcd_dir"])
    name = group_name(run_name, step, gen, sigma_a)
    root = Path(machine["pseudo_eid_dir"])
    final, tmp = root / name, root / (name + ".tmp")
    if final.exists():
        raise RuntimeError(f"組が既にあります（上書きしない）: {final}（別の組にするなら --name_suffix）")
    if tmp.exists():
        raise RuntimeError(f"前回の途中の出力が残っています: {tmp}（消してから実行）")
    idx, tasks = list_tasks(input_dir, gen)
    if not tasks:
        raise RuntimeError(f"作る対象がありません: {input_dir}")
    k_total, bs = gen["generate.k"], gen["sample.batch"]
    jobs = [(k, case, path) for k in range(k_total) for case, path in tasks]
    bf16 = want_bf16(gen["sample.precision"], device)
    channels_last = bool(train["optim.channels_last"])
    compiled = bool(gen["sample.compile"]) and device.type == "cuda"
    body = torch.compile(net.body, mode=gen["sample.compile_mode"]) if compiled else None
    sigma_data = train["diffusion.sigma_data"]

    def fn(x, sigma, cue, blur_px, noise_x, has):
        return denoise(net, x, sigma, cue, blur_px, noise_x, has, sigma_data, body=body, bf16=bf16, channels_last=channels_last, clone=compiled)[0]

    print(f"[generate] {name}: 重み {weight_dir}（step {step}、EMA）、入力 {input_dir}（{idx.n_cases} 症例 / {len(tasks)} 枚）× K {k_total} = {len(jobs)} 枚、"
          f"σ_b {gen['cue.blur_px']} 画素 / σ_c {gen['cue.noise_hu']} HU / σ_a {sigma_a} 画素、{gen['sample.steps']} 段 η {gen['sample.eta']}、bf16 {bf16}、compile {compiled}", flush=True)
    tmp.mkdir(parents=True)
    t0, done, size = time.time(), 0, None
    for i in range(0, len(jobs), bs):
        chunk = jobs[i:i + bs]
        imgs = [stored_to_x(read_stored(p), hu) for _, _, p in chunk]
        for (_, _, p), im in zip(chunk, imgs):
            size = size or im.shape
            if im.shape != size:
                raise RuntimeError(f"画像の大きさが混ざっています: {p} {im.shape} ≠ {size}")
        n_real = len(chunk)
        pad = bs - n_real if compiled else 0  # compile のときは形を固定（最後の 1 枚で詰めて、結果は捨てる）
        src = torch.from_numpy(np.stack(imgs + [imgs[-1]] * pad))[:, None].to(device)
        gens = [generator_of(device, gen["generate.seed"], case, p.name, k) for k, case, p in chunk + [chunk[-1]] * pad]
        b = src.shape[0]
        blur = torch.full((b,), gen["cue.blur_px"], device=device)
        noise = torch.full((b,), gen["cue.noise_hu"] / scale, device=device)
        out, _ = redraw(fn, src, blur, noise, gen["cue.use"], gens, gen["sample.steps"], gen["sample.eta"], gen["sample.rho"],
                        train["diffusion.sigma_min"], train["diffusion.sigma_max"], sigma_a)
        if not bool(torch.isfinite(out).all()):
            raise RuntimeError(f"出力に NaN / Inf があります（{chunk[0][2]} を含む batch）。--precision fp32 で試す")
        out = out[:n_real, 0].float().cpu().numpy()
        for (k, case, p), o in zip(chunk, out):
            write_png(tmp / f"k{k:02d}" / case / p.name, x_to_stored(o, hu))
        done += n_real
        if i == 0 or (i // bs) % 20 == 0 or done == len(jobs):
            el = time.time() - t0
            print(f"[generate] {done}/{len(jobs)} 枚、{el / done:.2f} s/枚、残り 約 {(len(jobs) - done) * el / done / 60:.0f} 分", flush=True)
    n_png = sum(1 for _ in tmp.rglob("*.png"))
    if n_png != len(jobs):
        raise RuntimeError(f"枚数が合いません: 書いた {n_png} ≠ 予定 {len(jobs)}（{tmp} を確認）")
    manifest = {
        "name": name, "created": datetime.datetime.now(JST).isoformat(timespec="seconds"), "git_commit": git_commit(),
        "run": run_name, "weight_dir": str(weight_dir), "weight_step": step, "weights": "ema",
        "input_dir": str(input_dir), "n_cases": idx.n_cases, "cases": idx.cases, "n_slices": len(tasks), "k": k_total, "n_images": len(jobs),
        "image_hw": list(size), "format": "1ch uint16 PNG, stored = HU + hu_offset", "hu_offset": hu[0], "hu_clip": [hu[1], hu[2]],
        "cue": {"use": gen["cue.use"], "blur_px": gen["cue.blur_px"], "noise_hu": gen["cue.noise_hu"]},
        "brightness_sigma_px_512": sigma_a,
        "sample": {"sampler": "euler_ancestral", "steps": gen["sample.steps"], "eta": gen["sample.eta"], "rho": gen["sample.rho"],
                   "sigma_min": train["diffusion.sigma_min"], "sigma_max": train["diffusion.sigma_max"], "precision": "bf16" if bf16 else "fp32", "compile": compiled, "batch": bs},
        "seed": gen["generate.seed"], "seed_rule": "sha256('<seed>/<case>/<slice file name>/<k>') >> 1 (first 8 bytes)",
        "slice_step": gen["generate.slice_step"], "max_slices": gen["generate.max_slices"],
        "sec_per_image": (time.time() - t0) / len(jobs), "generate": gen,
    }
    with open(tmp / "manifest.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, allow_unicode=True, sort_keys=False)
    for _ in range(10):  # Windows のバインドマウントで改名が拒否されることがある
        try:
            os.replace(tmp, final)
            break
        except PermissionError as e:
            err = e
            time.sleep(3.0)
    else:
        shutil.copytree(tmp, final)
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"[generate] 注意: 改名が拒否されたのでコピーで置いた（{err}）", flush=True)
    print(f"[generate] 完了: {final}（{len(jobs)} 枚、{manifest['sec_per_image']:.2f} s/枚）。manifest.yaml を確認", flush=True)
    return final
