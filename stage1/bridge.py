"""stage1/bridge.py — 型 X の疑似 EID の作成の本体。run_bridge.py が設定を検証してから呼ぶ。

  入力  <input_dir>/<症例>/<スライス>.png（512、1ch 16bit）
  出力  <pseudo_eid_dir>/<組>/k00/<症例>/<スライス>.png（512、1ch 16bit、stored = HU + 1400）+ <組>/manifest.yaml
        乱数を使わないので 1 枚につき 1 通り（k00 だけ。Stage 2 の読み方と同じ置き方）。<組>.tmp に書く → 枚数を検算 → <組> に改名。既にある組には書かない。

  手順（diffusion/ode.py）:
    1. 条件の絵 c = ぼかし(入力, σ_b)（+ σ_c のノイズ）。cue.use: false なら条件なし
    2. 行き: enc の描き手の ODE で、入力 → ノイズ z
    3. 帰り: dec の描き手の ODE で、z → 出力（条件は同じ c。anchor.decode なら各段で広い平均を入力に合わせ直す）
  enc = PCD の描き手、dec = 実 EID の描き手で PCD → 疑似 EID。enc = dec にすると往復の確認（元の絵に戻るはず）。
  確認用の数字（manifest と表示）: z を σ_max で割ったものの標準偏差（1 に近いはず）と、隣の画素との相関（0 に近いはず）、出力 − 入力の RMS（HU）。
"""

import datetime
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from data.ct_io import CaseIndex, hu_tuple, read_stored, stored_to_x, write_png, x_to_stored
from diffusion.anchor import wide_mean
from diffusion.cue import make_cue
from diffusion.ode import decode, encode
from generate import git_commit
from models.painter import denoise, load_ema, want_bf16
from util.run_paths import JST
from util.seeds import generator_of

SAME = ("data.hu_offset", "data.hu_min", "data.hu_max", "data.hu_scale", "cue.blur_max_px", "diffusion.sigma_min", "diffusion.sigma_max", "diffusion.sigma_data")


def group_name(br, enc_step, dec_step):
    cue = f"b{br['cue.blur_px']:.1f}_c{int(round(br['cue.noise_hu'])):03d}" if br["cue.use"] else "nocue"
    name = (f"bridge_{br['bridge.enc_run']}-{enc_step:07d}_to_{br['bridge.dec_run']}-{dec_step:07d}_{cue}"
            f"_n{br['ode.steps']}{'h' if br['ode.heun'] else 'e'}{'_anc' if br['anchor.decode'] else ''}{'_eta%g' % br['ode.decode_eta'] if br['ode.decode_eta'] > 0 else ''}")
    return name + (f"_{br['bridge.name_suffix']}" if br["bridge.name_suffix"] else "")


def list_tasks(input_dir, br):
    idx = CaseIndex(input_dir, cases=br["bridge.cases"] or None)
    tasks = []
    for case in idx.cases:
        paths = idx.slices[case][::br["bridge.slice_step"]]
        if br["bridge.slices"]:
            paths = [p for p in idx.slices[case] if Path(p).name in set(br["bridge.slices"])]
        if br["bridge.max_slices"] > 0:
            paths = paths[:br["bridge.max_slices"]]
        tasks += [(case, Path(p)) for p in paths]
    return idx, tasks


def run(br, shared, machine, enc_dir, dec_dir, device):
    enc_net, enc_cfg, enc_step = load_ema(enc_dir, device)
    same_weights = Path(enc_dir).resolve() == Path(dec_dir).resolve()
    dec_net, dec_cfg, dec_step = (enc_net, enc_cfg, enc_step) if same_weights else load_ema(dec_dir, device)
    train = dec_cfg["train"]
    diff = [k for k in SAME if enc_cfg["train"][k] != train[k]]
    if diff:
        raise RuntimeError(f"行きと帰りの描き手で、重みの意味を決める設定が違います: {diff}")
    hu, scale = hu_tuple(train), train["data.hu_scale"]
    sigma_a = shared["brightness.sigma_px_512"]
    s_min, s_max, sigma_data = train["diffusion.sigma_min"], train["diffusion.sigma_max"], train["diffusion.sigma_data"]
    input_dir = Path(br["bridge.input_dir"] or machine["pcd_dir"])
    name = group_name(br, enc_step, dec_step)
    root = Path(machine["pseudo_eid_dir"])
    final, tmp = root / name, root / (name + ".tmp")
    if final.exists():
        raise RuntimeError(f"組が既にあります（上書きしない）: {final}（別の組にするなら --name_suffix）")
    if tmp.exists():
        raise RuntimeError(f"前回の途中の出力が残っています: {tmp}（消してから実行）")
    idx, tasks = list_tasks(input_dir, br)
    if not tasks:
        raise RuntimeError(f"作る対象がありません: {input_dir}")
    bf16 = want_bf16(br["ode.precision"], device)

    def fn_of(net, cfg):
        cl = bool(cfg["train"]["optim.channels_last"])
        return lambda x, sigma, cue, blur_px, noise_x, has: denoise(net, x, sigma, cue, blur_px, noise_x, has, sigma_data, bf16=bf16, channels_last=cl)[0]

    enc_fn, dec_fn = fn_of(enc_net, enc_cfg), fn_of(dec_net, dec_cfg)
    steps, rho, heun, bs = br["ode.steps"], br["ode.rho"], br["ode.heun"], br["ode.batch"]
    print(f"[bridge] {name}: 行き {enc_dir}（step {enc_step}）→ 帰り {dec_dir}（step {dec_step}）、入力 {input_dir}（{len(tasks)} 枚）、"
          f"条件 {'σ_b %g 画素 / σ_c %g HU' % (br['cue.blur_px'], br['cue.noise_hu']) if br['cue.use'] else 'なし'}、錨 {br['anchor.decode']}、"
          f"{steps} 段 {'Heun' if heun else 'Euler'}、bf16 {bf16}", flush=True)
    tmp.mkdir(parents=True)
    t0, done, size, rows = time.time(), 0, None, []
    for i in range(0, len(tasks), bs):
        chunk = tasks[i:i + bs]
        imgs = [stored_to_x(read_stored(p), hu) for _, p in chunk]
        for (_, p), im in zip(chunk, imgs):
            size = size or im.shape
            if im.shape != size:
                raise RuntimeError(f"画像の大きさが混ざっています: {p} {im.shape} ≠ {size}")
        src = torch.from_numpy(np.stack(imgs))[:, None].to(device)
        b = src.shape[0]
        use = bool(br["cue.use"])
        has = torch.full((b,), use, device=device, dtype=torch.bool)
        blur = torch.full((b,), br["cue.blur_px"] if use else 0.0, device=device)
        noise = torch.full((b,), br["cue.noise_hu"] / scale if use else 0.0, device=device)
        if use:
            n_c = torch.stack([torch.randn(src.shape[1:], generator=generator_of(device, br["bridge.seed"], case, p.name, 0), device=device) for case, p in chunk])
            cue = make_cue(src, blur, noise, n_c)
        else:
            cue = torch.zeros_like(src)
        cond = (cue, blur, noise, has)
        z = encode(enc_fn, src, cond, steps, rho, s_min, s_max, heun)
        m = wide_mean(src, sigma_a) if br["anchor.decode"] else None
        eta = br["ode.decode_eta"]
        gens = [generator_of(device, br["bridge.seed"], case, p.name, 1) for case, p in chunk] if eta > 0 else None
        out = decode(dec_fn, z, cond, steps, rho, s_min, s_max, heun, anchor_m=m, anchor_px=sigma_a, eta=eta, gens=gens)
        if not bool(torch.isfinite(out).all()):
            raise RuntimeError(f"出力に NaN / Inf があります（{chunk[0][1]} を含む batch）。--precision fp32 で試す")
        zn = z / s_max
        zc = zn - zn.mean(dim=(1, 2, 3), keepdim=True)
        z_std = zn.std(dim=(1, 2, 3))
        z_corr = (zc[..., :, 1:] * zc[..., :, :-1]).mean(dim=(1, 2, 3)) / zc.pow(2).mean(dim=(1, 2, 3))
        rms = (out - src).pow(2).mean(dim=(1, 2, 3)).sqrt() * scale
        out_np = out[:, 0].float().cpu().numpy()
        for j, ((case, p), o) in enumerate(zip(chunk, out_np)):
            write_png(tmp / "k00" / case / p.name, x_to_stored(o, hu))
            rows.append({"case": case, "slice": p.name, "z_std": float(z_std[j]), "z_neighbor_corr": float(z_corr[j]), "out_minus_in_rms_hu": float(rms[j])})
        done += b
        el = time.time() - t0
        r = rows[-1]
        print(f"[bridge] {done}/{len(tasks)} 枚、{el / done:.1f} s/枚、z の標準偏差 {r['z_std']:.4f}・隣との相関 {r['z_neighbor_corr']:+.4f}、出力 − 入力 RMS {r['out_minus_in_rms_hu']:.1f} HU", flush=True)
    n_png = sum(1 for _ in tmp.rglob("*.png"))
    if n_png != len(tasks):
        raise RuntimeError(f"枚数が合いません: 書いた {n_png} ≠ 予定 {len(tasks)}（{tmp} を確認）")
    mean = lambda k: float(np.mean([r[k] for r in rows]))
    manifest = {
        "name": name, "created": datetime.datetime.now(JST).isoformat(timespec="seconds"), "git_commit": git_commit(), "method": "bridge (type X: encode with enc, decode with dec, probability-flow ODE)",
        "enc": {"run": br["bridge.enc_run"], "weight_dir": str(enc_dir), "step": enc_step}, "dec": {"run": br["bridge.dec_run"], "weight_dir": str(dec_dir), "step": dec_step}, "weights": "ema",
        "input_dir": str(input_dir), "n_cases": idx.n_cases, "cases": idx.cases, "n_slices": len(tasks), "k": 1, "n_images": len(tasks),
        "image_hw": list(size), "format": "1ch uint16 PNG, stored = HU + hu_offset", "hu_offset": hu[0], "hu_clip": [hu[1], hu[2]],
        "cue": {"use": br["cue.use"], "blur_px": br["cue.blur_px"], "noise_hu": br["cue.noise_hu"]}, "anchor_decode": br["anchor.decode"], "brightness_sigma_px_512": sigma_a,
        "ode": {"steps": steps, "solver": "heun" if heun else "euler", "rho": rho, "decode_eta": br["ode.decode_eta"], "sigma_min": s_min, "sigma_max": s_max, "precision": "bf16" if bf16 else "fp32", "batch": bs},
        "check": {"z_std_mean": mean("z_std"), "z_neighbor_corr_mean": mean("z_neighbor_corr"), "out_minus_in_rms_hu_mean": mean("out_minus_in_rms_hu"), "per_slice": rows},
        "sec_per_image": (time.time() - t0) / len(tasks), "bridge": br,
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
        print(f"[bridge] 注意: 改名が拒否されたのでコピーで置いた（{err}）", flush=True)
    print(f"[bridge] 完了: {final}（{len(tasks)} 枚、{manifest['sec_per_image']:.1f} s/枚）。z の標準偏差 平均 {manifest['check']['z_std_mean']:.4f}、"
          f"隣との相関 平均 {manifest['check']['z_neighbor_corr_mean']:+.4f}、出力 − 入力 RMS 平均 {manifest['check']['out_minus_in_rms_hu_mean']:.1f} HU", flush=True)
    return final
