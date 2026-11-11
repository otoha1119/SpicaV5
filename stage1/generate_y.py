"""stage1/generate_y.py — 型 Y の疑似 EID の作成（ホストからは bash start.sh PC1 ygen --run <型 Y の run> [--flag value]）。

  python stage1/generate_y.py --machine PC1 --config stage1/configs/y_generate.yaml --shared configs/shared.yaml --machines configs/machines.yaml -- --run <run> --s_hu 100

手順: PCD → 消し手 A → m。redraw.s_hu > 0 なら m に強さ s のノイズを足し、実 EID の描き手 E（run の init.e_run）で s から 0 まで描き直す
（Euler ancestral η = 1、毎段に新しい乱数。手がかり = Gaussian(m, cue_blur_px)、redraw.anchor なら毎段に明るさの錨）。s = 0 は A の出力をそのまま書く。
出力: <pseudo_eid_dir>/Y_<run>-<step>_s<s>_b<cue>[_anc][_<suffix>]/k00…/<症例>/<スライス>.png と manifest.yaml（既にある組には書かない）。
乱数は画像ごと（util/seeds.py、'<seed>/<症例>/<スライス名>/<k>'）。
"""

import argparse
import datetime
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from configs.schema import MACHINE, PRECISIONS, SHARED, Y_GENERATE, Y_TRAIN, ConfigError, _raise, apply_overrides, validate, validate_shared  # noqa: E402
from data.ct_io import CaseIndex, hu_tuple, read_stored, stored_to_x, write_png, x_to_stored  # noqa: E402
from diffusion.anchor import anchor, wide_mean  # noqa: E402
from diffusion.cue import gaussian_blur  # noqa: E402
from diffusion.edm import ancestral_step, karras_sigmas  # noqa: E402
from generate import git_commit  # noqa: E402
from models.eraser import build_eraser, fill_legacy  # noqa: E402
from models.painter import denoise, load_ema, want_bf16  # noqa: E402
from util.run_paths import JST, ckpt_dir  # noqa: E402
from util.seeds import generator_of  # noqa: E402


def load_yaml(path):
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"設定ファイルが見つかりません: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def redraw(E, m, src, s, steps, cue_blur, anchor_px, use_anchor, sigma_min, sd, bf16, cl, gens):
    b, d = m.shape[0], m.device
    rn = lambda: torch.stack([torch.randn(m.shape[1:], generator=g, device=d) for g in gens])
    blur, noise0, has = torch.full((b,), cue_blur, device=d), torch.zeros(b, device=d), torch.ones(b, device=d, dtype=torch.bool)
    cue = gaussian_blur(m, cue_blur)
    wm = wide_mean(src, anchor_px)
    sig = karras_sigmas(steps, sigma_min, s, 7.0, d)
    x = m + s * rn()
    for i in range(steps):
        a, a2 = float(sig[i]), float(sig[i + 1])
        x0 = denoise(E, x, torch.full((b,), a, device=d), cue, blur, noise0, has, sd, bf16=bf16, channels_last=cl)[0]
        if use_anchor:
            x0 = anchor(x0, wm, anchor_px)
        if a2 == 0:
            return x0
        down, up = ancestral_step(a, a2, 1.0)
        x = x0 + (down / a) * (x - x0) + up * rn()
    return x


def main():
    p = argparse.ArgumentParser(description="型 Y の疑似 EID の作成")
    p.add_argument("--machine", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--shared", required=True)
    p.add_argument("--machines", required=True)
    p.add_argument("overrides", nargs=argparse.REMAINDER)
    a = p.parse_args()
    overrides = a.overrides[1:] if a.overrides and a.overrides[0] == "--" else a.overrides
    try:
        g = validate("y_generate", load_yaml(a.config), Y_GENERATE)
        shared = validate_shared("shared", load_yaml(a.shared), SHARED)
        machines = load_yaml(a.machines)
        if not isinstance(machines, dict) or a.machine not in machines:
            raise ConfigError(f"machines.yaml にエントリ '{a.machine}' がありません")
        machine = validate_shared(f"machines.{a.machine}", machines[a.machine], MACHINE)
        apply_overrides(overrides, {"y_generate": (g, Y_GENERATE), "shared": (shared, SHARED), "machine": (machine, MACHINE)})
        P = []
        if not g["generate.run"]: P.append("generate.run を --run で指定")
        if g["generate.k"] < 1 or g["generate.slice_step"] < 1 or g["redraw.steps"] < 1 or g["redraw.batch"] < 1: P.append("k / slice_step / steps / batch ≥ 1")
        if g["generate.max_slices"] < 0 or g["redraw.s_hu"] < 0 or g["redraw.cue_blur_px"] <= 0: P.append("max_slices ≥ 0、s_hu ≥ 0、cue_blur_px は正")
        if g["redraw.precision"] not in PRECISIONS: P.append(f"redraw.precision は {PRECISIONS}")
        _raise("y_generate values", P)
        run_dir = Path(machine["painter_checkpoints_dir"]) / g["generate.run"]
        wdir = ckpt_dir(run_dir, g["generate.tag"])
        if not (wdir / "A.pth").is_file():
            raise ConfigError(f"A の重みがありません: {wdir}")
        state = torch.load(wdir / "state.pth", map_location="cpu", weights_only=False)
        yc = validate("y(state)", fill_legacy(dict(state["config"])), Y_TRAIN)
        e_dir = ckpt_dir(Path(machine["painter_checkpoints_dir"]) / yc["init.e_run"], yc["init.e_tag"])
    except ConfigError as e:
        print(f"[generate_y] 設定エラー: {e}", file=sys.stderr)
        sys.exit(2)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    A = build_eraser(yc).to(device)
    A.load_state_dict(torch.load(wdir / "A.pth", map_location=device, weights_only=True))
    A.eval()
    E, e_cfg, e_step = load_ema(e_dir, device)
    et = e_cfg["train"]
    hu, scale = hu_tuple(et), et["data.hu_scale"]
    bf16, cl = want_bf16(g["redraw.precision"], device), bool(et["optim.channels_last"])
    s = g["redraw.s_hu"] / scale
    name = f"Y_{g['generate.run']}-{int(state['step']):07d}_s{g['redraw.s_hu']:g}_b{g['redraw.cue_blur_px']:g}{'_anc' if g['redraw.anchor'] else ''}"
    name += f"_{g['generate.name_suffix']}" if g["generate.name_suffix"] else ""
    root = Path(machine["pseudo_eid_dir"])
    final, tmp = root / name, root / (name + ".tmp")
    if final.exists() or tmp.exists():
        print(f"[generate_y] エラー: 組（か途中の出力）が既にあります: {final}", file=sys.stderr)
        sys.exit(1)
    input_dir = Path(g["generate.input_dir"] or machine["pcd_dir"])
    idx = CaseIndex(input_dir, cases=g["generate.cases"] or None)
    tasks = []
    for case in idx.cases:
        paths = idx.slices[case][::g["generate.slice_step"]]
        if g["generate.slices"]:
            paths = [q for q in idx.slices[case] if Path(q).name in set(g["generate.slices"])]
        if g["generate.max_slices"] > 0:
            paths = paths[:g["generate.max_slices"]]
        tasks += [(case, Path(q)) for q in paths]
    jobs = [(k, case, q) for k in range(g["generate.k"]) for case, q in tasks]
    if not jobs:
        print(f"[generate_y] エラー: 作る対象がありません: {input_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"[generate_y] {name}: A {wdir}（step {state['step']}）、E {e_dir}（step {e_step}）、{len(tasks)} 枚 × K {g['generate.k']}、s {g['redraw.s_hu']:g} HU、{g['redraw.steps']} 段", flush=True)
    tmp.mkdir(parents=True)
    t0, bs = time.time(), g["redraw.batch"]
    for i in range(0, len(jobs), bs):
        chunk = jobs[i:i + bs]
        src = torch.from_numpy(np.stack([stored_to_x(read_stored(q), hu) for _, _, q in chunk]))[:, None].to(device)
        with torch.no_grad():
            m = A(src)
            if s > 0:
                gens = [generator_of(device, g["generate.seed"], case, q.name, k) for k, case, q in chunk]
                out = redraw(E, m, src, s, g["redraw.steps"], g["redraw.cue_blur_px"], shared["brightness.sigma_px_512"], g["redraw.anchor"],
                             et["diffusion.sigma_min"], et["diffusion.sigma_data"], bf16, cl, gens)
            else:
                out = m
        if not bool(torch.isfinite(out).all()):
            raise RuntimeError("出力に NaN / Inf があります")
        for (k, case, q), o in zip(chunk, out[:, 0].float().cpu().numpy()):
            write_png(tmp / f"k{k:02d}" / case / q.name, x_to_stored(o, hu))
        done = i + len(chunk)
        print(f"[generate_y] {done}/{len(jobs)} 枚、{(time.time() - t0) / done:.2f} s/枚", flush=True)
    manifest = {"name": name, "created": datetime.datetime.now(JST).isoformat(timespec="seconds"), "git_commit": git_commit(), "method": "type Y (eraser A + redraw by E)",
                "a_run": g["generate.run"], "a_weight_dir": str(wdir), "a_step": int(state["step"]), "e_weight_dir": str(e_dir), "e_step": e_step,
                "input_dir": str(input_dir), "cases": idx.cases, "n_slices": len(tasks), "k": g["generate.k"], "n_images": len(jobs),
                "format": "1ch uint16 PNG, stored = HU + hu_offset", "brightness_sigma_px_512": shared["brightness.sigma_px_512"],
                "generate": g, "sec_per_image": (time.time() - t0) / len(jobs)}
    with open(tmp / "manifest.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, allow_unicode=True, sort_keys=False)
    os.replace(tmp, final)
    print(f"[generate_y] 完了: {final}（{len(jobs)} 枚）", flush=True)


if __name__ == "__main__":
    main()
