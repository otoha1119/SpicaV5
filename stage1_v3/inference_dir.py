"""stage1_v3/inference_dir.py — 学習済み G で 512 スライスを書き出す（run_infer.py から全引数明示で exec される）。

  入力: <input_dir>/<case>/<slice>.png（1ch uint16、512）
  --apply_degrade : 入力 x に P（run の launch.yaml と同じ範囲、スライスごとに抽選）を掛けてから G に通す（PCD → EID-like。Stage 2 の学習データ）
                    無しなら G だけ（実 EID → 恒等の確認）
  --n_draws N     : P を N 回引き直す。full/ は draw 0、full_std/ に N 回の出力の標準偏差 [HU]（uint16。R-S21 の判定）
  出力: <out_dir>/full/<case>/<slice>.png（出力）、full_u/（P(x)、--save_residual かつ apply_degrade）、full_R/ と full_R_color/（R = 出力 − G の入力、--save_residual）、
        full_std/（n_draws > 1）、full_panel/（[出力 | P(x) | 入力 | R + ゲージ]、--save_panel）、R_colorbar_pm<range>HU.png、draws_summary.yaml（n_draws > 1 と恒等 RMS の要約）
  G は 512 フル画像を一発で通す。
  v4（run の launch が pipeline.mode v4）: --apply_degrade なら全経路 x → Dn → G → ⊛ k(z) → + ノイズ（models/projection_model.generate_v4）。full/ = u（draw 0）、full_std/、
        --eps_zero で full_eps0/（ε = 0 の u = 指紋。z は draw 0 と同じ = generator の状態を保存して引き直すので full − full_eps0 がノイズの実現値そのもの）、
        --save_residual で full_xc/（Dn(x)）と full_R/（R(x_c) = G(x_c) − x_c）。パネルは [u | Dn(x) | x | R(x_c)]。
        --apply_degrade 無しなら v3 と同じく G だけ（恒等の確認）。manifest.yaml に G / P / Dn の重みのパスと sha256 を残す
"""

import argparse
import collections
import hashlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from data.dataset import CaseIndex, denormalize, hu_to_norm, read_stored, residual_stored, stored_to_hu, write_png  # noqa: E402
from degrade import apply_degrade, sample_params  # noqa: E402
from models import networks  # noqa: E402
from models.degrade_net import LearnedDegrade  # noqa: E402
from models.denoiser import load_denoiser  # noqa: E402
from models.projection_model import generate_v4  # noqa: E402
from util.panel import build_panel  # noqa: E402
from util.residual_color import colorbar_filename, colorbar_rgb01, residual_rgb01, residual_rgb8, rgb_to_bgr  # noqa: E402
from util.run_paths import DN_WEIGHT, INFER_WEIGHT, find_run_dir, latest_launch  # noqa: E402

DEVICES = ("cuda", "cpu")


def parse_bool(s):
    return str(s).lower() in ("true", "1", "yes")


def parse_args():
    p = argparse.ArgumentParser(description="射影型 Stage 1 推論")
    req = lambda *a, **k: p.add_argument(*a, required=True, **k)  # noqa: E731
    req("--weight_dir")
    req("--input_dir")
    req("--out_dir")
    req("--device", choices=DEVICES)
    req("--index_cache_dir")
    # G
    req("--netG"); req("--ngf", type=int); req("--norm")
    req("--hu_offset", type=int); req("--hu_min", type=int); req("--hu_max", type=int)
    req("--input_size", type=int)
    # P（run の launch.yaml から）
    req("--p_enabled", type=parse_bool); req("--p_mode"); req("--p_blur_min", type=float); req("--p_blur_max", type=float)
    req("--p_learned_z_dim", type=int); req("--p_learned_kernel_size", type=int); req("--p_learned_noise_kernel_size", type=int); req("--p_learned_amp_ch", type=int); req("--p_learned_sigma_max_hu", type=float)
    req("--p_learned_sigma_min_hu", type=float)
    # v4（run の launch から）
    req("--pipeline_mode", choices=("v3", "v4")); req("--dn_weights")  # v3 では ""
    req("--p_bil_color_min", type=float); req("--p_bil_color_max", type=float); req("--p_bil_space_min", type=float); req("--p_bil_space_max", type=float)
    req("--p_noise_min", type=float); req("--p_noise_max", type=float); req("--p_corr_min", type=float); req("--p_corr_max", type=float)
    req("--p_body_only", type=parse_bool); req("--p_body_threshold_hu", type=float)
    # 表示
    req("--diff_range_hu", type=int); req("--display_hu_min", type=int); req("--display_hu_max", type=int); req("--preview_bits", type=int, choices=(8, 16))
    # 方式（configs/infer.yaml）
    p.add_argument("--apply_degrade", action="store_true")
    req("--n_draws", type=int)
    req("--max_slices", type=int)
    req("--batch_slices", type=int)
    p.add_argument("--save_residual", action="store_true")
    p.add_argument("--save_panel", action="store_true")
    p.add_argument("--eps_zero", action="store_true")
    req("--read_workers", type=int)
    req("--write_workers", type=int)
    p.add_argument("--seed", type=int, default=0, help="P の乱数 seed（run_infer は渡さない = 0）")
    return p.parse_args()


def build_generator(a, device):
    cfg = {"network.norm": a.norm, "network.ngf": a.ngf, "network.netG": a.netG, "network.init_type": "normal", "network.init_gain": 0.02}
    net = networks.build_G(cfg, device)
    path = Path(a.weight_dir) / INFER_WEIGHT
    if not path.is_file():
        raise FileNotFoundError(f"G の重みがありません: {path}")
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    net.to(device).eval()
    return net, path


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    a = parse_args()
    device = torch.device(a.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda ですが CUDA が使えません")
        torch.backends.cudnn.benchmark = True
    input_dir, out_dir = Path(a.input_dir), Path(a.out_dir)
    hu = (a.hu_offset, a.hu_min, a.hu_max)
    size = a.input_size
    pcfg = {"enabled": bool(a.p_enabled and a.apply_degrade), "mode": a.p_mode, "bil_color": (a.p_bil_color_min, a.p_bil_color_max), "bil_space": (a.p_bil_space_min, a.p_bil_space_max),
            "blur": (a.p_blur_min, a.p_blur_max), "noise": (a.p_noise_min, a.p_noise_max),
            "corr": (a.p_corr_min, a.p_corr_max), "body_only": a.p_body_only, "body_threshold_hu": a.p_body_threshold_hu}
    n_draws = a.n_draws if pcfg["enabled"] else 1
    rng = np.random.default_rng(a.seed)
    net, weight_path = build_generator(a, device)
    netP, netDn = None, None
    v4 = a.pipeline_mode == "v4" and pcfg["enabled"]
    learned = pcfg["enabled"] and a.p_mode == "learned"
    eps_zero = bool(a.eps_zero and v4)
    weights = {"G": {"path": str(weight_path), "sha256": sha256(weight_path)}}
    if learned:
        hu_per_unit = (a.hu_max - a.hu_min) / 2.0
        netP = LearnedDegrade(a.p_learned_z_dim, a.p_learned_kernel_size, a.p_learned_noise_kernel_size, a.p_learned_amp_ch, a.p_learned_sigma_max_hu / hu_per_unit,
                              sigma_min_norm=(a.p_learned_sigma_min_hu / hu_per_unit if v4 else 0.0))
        p_path = Path(a.weight_dir) / "net_P.pth"
        netP.load_state_dict(torch.load(p_path, map_location="cpu", weights_only=True))
        netP.to(device).eval()
        gen = torch.Generator(device=device).manual_seed(a.seed)
        weights["P"] = {"path": str(p_path), "sha256": sha256(p_path)}
    if v4:
        if not a.dn_weights or not Path(a.dn_weights).is_file():
            raise FileNotFoundError(f"v4 の Dn の重みがありません: {a.dn_weights!r}")
        netDn = load_denoiser(a.dn_weights, device)
        weights["Dn"] = {"path": str(a.dn_weights), "sha256": sha256(a.dn_weights)}
        run_dir = find_run_dir(a.weight_dir)
        dn_at_train = None
        if run_dir is not None:
            with open(latest_launch(run_dir), encoding="utf-8") as f:
                dn_at_train = yaml.safe_load(f).get("dn")  # run_train が書いた {path, sha256}
        weights["Dn"]["at_train"] = dn_at_train
        weights["Dn"]["matches_train"] = bool(isinstance(dn_at_train, dict) and dn_at_train.get("sha256") == weights["Dn"]["sha256"])
        if not weights["Dn"]["matches_train"]:
            print(f"[infer] 注意: 使う Dn の sha256 が学習時（launch.yaml の dn:）と一致しません: {dn_at_train}")
    idx = CaseIndex(input_dir, a.index_cache_dir)
    paths = idx.all_paths if a.max_slices == 0 else idx.all_paths[: a.max_slices]
    print(f"[infer] device={device} pipeline={a.pipeline_mode} G={weight_path} P={'on' if pcfg['enabled'] else 'off'} n_draws={n_draws}" + (f" Dn={a.dn_weights} eps_zero={eps_zero}" if v4 else ""))
    print(f"[infer] input={input_dir} ({idx.n_cases} cases, {idx.n_slices} slices, 推論 {len(paths)} 枚) {size} -> {size}: {out_dir}")
    if a.save_panel and a.display_hu_min >= a.display_hu_max:
        raise ValueError("--display_hu_min < --display_hu_max にしてください")
    in_label = ("Dn(x)  x_c" if v4 else "P(x)  u") if pcfg["enabled"] else "input  (no P)"
    panel_labels = [f"output  {Path(a.weight_dir).name}", in_label, "input  x", "R(x_c) = G(x_c) - x_c   [HU]" if v4 else "R = output - G input   [HU]"]
    panel_gray01 = lambda stored: np.repeat(np.clip((stored.astype(np.float32) - a.hu_offset - a.display_hu_min) / float(a.display_hu_max - a.display_hu_min), 0.0, 1.0)[..., None], 3, axis=2)  # noqa: E731
    quant_panel = (lambda x01: (np.clip(x01, 0, 1) * 65535.0).round().astype(np.uint16)) if a.preview_bits == 16 else (lambda x01: (np.clip(x01, 0, 1) * 255.0).round().astype(np.uint8))  # noqa: E731
    out_dir.mkdir(parents=True, exist_ok=True)
    if a.save_residual or a.save_panel:
        write_png(out_dir / colorbar_filename(a.diff_range_hu), rgb_to_bgr((colorbar_rgb01(a.diff_range_hu, 512) * 255.0).round().astype(np.uint8)))

    reader = ThreadPoolExecutor(max_workers=a.read_workers) if a.read_workers > 0 else None
    writer = ThreadPoolExecutor(max_workers=a.write_workers) if a.write_workers > 0 else None
    pending = []
    tm = {"read": 0.0, "g": 0.0, "write": 0.0}
    summary = {"n_slices": 0, "R_rms_HU_sum": 0.0, "std_out_body_HU_sum": 0.0, "std_u_body_HU_sum": 0.0, "params": []}

    def load(p):
        stored = read_stored(p)
        if stored.shape != (size, size):
            raise ValueError(f"入力は {size}×{size} のみです: {stored.shape} {p}")
        return stored, stored_to_hu(stored, a.hu_offset)

    def drain(limit):
        t = time.time()
        while len(pending) > limit:
            pending.pop(0).result()
        tm["write"] += time.time() - t

    def put(path, arr):
        if writer is None:
            write_png(path, arr)
        else:
            pending.append(writer.submit(write_png, path, arr))
            drain(4 * a.write_workers + 16)

    queue = collections.deque()
    it = iter(paths)
    prefetch = 2 * a.batch_slices + (a.read_workers if reader is not None else 0)

    def take(n):
        items = []
        for _ in range(n):
            if reader is None:
                p = next(it, None)
                if p is None:
                    break
                t = time.time()
                stored, x_hu = load(p)
                tm["read"] += time.time() - t
            else:
                while len(queue) < prefetch:
                    p = next(it, None)
                    if p is None:
                        break
                    queue.append((p, reader.submit(load, p)))
                if not queue:
                    break
                p, fut = queue.popleft()
                t = time.time()
                stored, x_hu = fut.result()
                tm["read"] += time.time() - t
            items.append((p, stored, x_hu))
        return items

    t0 = time.time()
    bar = tqdm(total=len(paths), unit="slice", dynamic_ncols=True)
    try:
        while True:
            items = take(a.batch_slices)
            if not items:
                break
            # P: スライスごと・draw ごとに抽選。(n_items, n_draws) の u
            us, params = [], []
            t = time.time()
            if v4:  # 全経路。outs = u（draw 0 が full/）、gin = x_c（R の基準）、rsrc = G(x_c)、u0 = ε = 0 の u（z は draw 0 と同じ）
                xn = torch.from_numpy(np.stack([hu_to_norm(x_hu, a.hu_min, a.hu_max) for _, _, x_hu in items])[:, None]).to(device)
                with torch.inference_mode():
                    state0 = gen.get_state()  # draw 0 の直前の状態（z が先に引かれる）
                    draws = [generate_v4(netDn, net, netP, xn, generator=gen) for _ in range(n_draws)]
                    u0 = None
                    if eps_zero:
                        state_end = gen.get_state()
                        gen.set_state(state0)  # 同じ z で ε = 0 → full − full_eps0 = ノイズの実現値
                        u0 = generate_v4(netDn, net, netP, xn, generator=gen, eps_zero=True)["u"].cpu().numpy()[:, 0]
                        gen.set_state(state_end)
                ys = np.stack([d["u"].cpu().numpy()[:, 0] for d in draws], axis=1).reshape(len(items) * n_draws, *xn.shape[-2:])  # item-major
                gin = draws[0]["x_c"].cpu().numpy()[:, 0]
                rsrc = draws[0]["g"].cpu().numpy()[:, 0]
                params = [[] for _ in items]  # v4 に draw ごとのパラメータは無い（z は乱数）
                tm["g"] += time.time() - t
            elif learned:  # P_θ を GPU で（draw ごとに別の z, ε）
                xn = torch.from_numpy(np.stack([hu_to_norm(x_hu, a.hu_min, a.hu_max) for _, _, x_hu in items])[:, None]).to(device)
                with torch.inference_mode():
                    un = torch.cat([netP(xn, generator=gen)[0].float() for _ in range(n_draws)], dim=0)  # (n_draws*n_items) draw-major
                un = un.view(n_draws, len(items), 1, *un.shape[-2:]).transpose(0, 1).reshape(-1, 1, *un.shape[-2:])  # item-major
                u_hu = (un.cpu().numpy()[:, 0] + 1.0) / 2.0 * (a.hu_max - a.hu_min) + a.hu_min
                for k in range(len(items)):
                    us.append([u_hu[k * n_draws + d] for d in range(n_draws)])
                    params.append([{"learned": True} for _ in range(n_draws)])
                xs = un
            else:
                for _, _, x_hu in items:
                    ud, pd = [], []
                    for _ in range(n_draws):
                        prm = sample_params(pcfg, rng)
                        ud.append(apply_degrade(x_hu, prm, pcfg, rng))
                        pd.append(prm)
                    us.append(ud)
                    params.append(pd)
                xs = torch.from_numpy(np.stack([hu_to_norm(u, a.hu_min, a.hu_max) for ud in us for u in ud])[:, None]).to(device)
            if not v4:
                t = time.time()
                with torch.inference_mode():
                    ys = net(xs).cpu().numpy()[:, 0]
                tm["g"] += time.time() - t
            for k, (p, stored, x_hu) in enumerate(items):
                rel = Path(p).relative_to(input_dir)
                outs = ys[k * n_draws : (k + 1) * n_draws]
                out16 = denormalize(outs[0], *hu)
                u16 = denormalize(gin[k] if v4 else hu_to_norm(us[k][0], a.hu_min, a.hu_max), *hu)  # G の入力（v3: P(x)、v4: Dn(x)）
                r16 = denormalize(rsrc[k], *hu) if v4 else out16  # R の出力側（v3: 出力、v4: G(x_c)）
                in16 = denormalize(hu_to_norm(x_hu, a.hu_min, a.hu_max), *hu)
                put(out_dir / "full" / rel, out16)
                if v4 and eps_zero:
                    put(out_dir / "full_eps0" / rel, denormalize(u0[k], *hu))
                delta_hu = r16.astype(np.int32) - u16.astype(np.int32)
                summary["n_slices"] += 1
                summary["R_rms_HU_sum"] += float(np.sqrt(np.mean(delta_hu.astype(np.float64) ** 2)))
                summary["params"].append({"slice": str(rel), "draws": params[k]})
                if a.save_residual:
                    if pcfg["enabled"]:
                        put(out_dir / ("full_xc" if v4 else "full_u") / rel, u16)
                    put(out_dir / "full_R" / rel, residual_stored(u16, r16))
                    put(out_dir / "full_R_color" / rel, rgb_to_bgr(residual_rgb8(delta_hu, a.diff_range_hu)))
                if n_draws > 1:
                    out_hu = (outs + 1.0) / 2.0 * (a.hu_max - a.hu_min) + a.hu_min
                    std_out = out_hu.std(axis=0)
                    std_u = np.zeros_like(std_out) if v4 else np.stack(us[k]).std(axis=0)  # v4: 決定的な x_c の引き直しは無い（std_u = 0）
                    body = x_hu > a.p_body_threshold_hu
                    summary["std_out_body_HU_sum"] += float(std_out[body].mean()) if body.any() else float(std_out.mean())
                    summary["std_u_body_HU_sum"] += float(std_u[body].mean()) if body.any() else float(std_u.mean())
                    put(out_dir / "full_std" / rel, np.clip(np.round(std_out), 0, 65535).astype(np.uint16))
                if a.save_panel:
                    panel01 = build_panel([panel_gray01(out16), panel_gray01(u16), panel_gray01(in16), residual_rgb01(delta_hu, a.diff_range_hu)], panel_labels, a.diff_range_hu)
                    put(out_dir / "full_panel" / rel, rgb_to_bgr(quant_panel(panel01)))
                bar.update(1)
        drain(0)
    finally:
        bar.close()
        if reader is not None:
            reader.shutdown(wait=False, cancel_futures=True)
        if writer is not None:
            writer.shutdown(wait=True)
    n = max(summary["n_slices"], 1)
    res = {"n_slices": summary["n_slices"], "apply_degrade": pcfg["enabled"], "pipeline_mode": a.pipeline_mode, "n_draws": n_draws, "R_rms_HU_mean": summary["R_rms_HU_sum"] / n}
    if n_draws > 1:
        res["std_out_body_HU_mean"] = summary["std_out_body_HU_sum"] / n
        if not v4:
            res["std_u_body_HU_mean"] = summary["std_u_body_HU_sum"] / n
            res["std_ratio"] = res["std_out_body_HU_mean"] / max(res["std_u_body_HU_mean"], 1e-6)
    with open(out_dir / "draws_summary.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"summary": res, "params": summary["params"]}, f, allow_unicode=True, sort_keys=False)
    with open(out_dir / "manifest.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"pipeline_mode": a.pipeline_mode, "apply_degrade": pcfg["enabled"], "eps_zero": eps_zero, "eps0_shares_z_with_draw0": eps_zero, "n_draws": n_draws, "weights": weights,
                        "outputs": ["full"] + (["full_eps0"] if eps_zero else []) + (["full_std"] if n_draws > 1 else [])
                        + ((["full_xc" if v4 else "full_u"] if pcfg["enabled"] else []) + ["full_R", "full_R_color"] if a.save_residual else []) + (["full_panel"] if a.save_panel else [])},
                       f, allow_unicode=True, sort_keys=False)
    elapsed = time.time() - t0
    print(f"[infer] R rms {res['R_rms_HU_mean']:.1f} HU" + (f" | draws std_out {res['std_out_body_HU_mean']:.1f} HU" + (f" / std_u {res['std_u_body_HU_mean']:.1f} HU (ratio {res['std_ratio']:.2f})" if not v4 else "") if n_draws > 1 else ""))
    print(f"[infer] time: 読み待ち {tm['read']:.1f}s | G {tm['g']:.1f}s | 書き待ち {tm['write']:.1f}s")
    print(f"[infer] done: {len(paths)} slices, {elapsed:.1f}s ({elapsed / max(len(paths), 1):.2f} s/slice) -> {out_dir}")


if __name__ == "__main__":
    main()
