"""stage1_v3/train_denoiser.py — v4 の Dn（PCD512 の自己教師あり denoiser）の学習。bash start1_v3.sh <machine> dn → denoise.sh から起動。
  dn.method n2n（Neighbor2Neighbor）| blindspot（構造化 blind-spot、本命）。bm3d は学習なしなので dn では使えない（bash start1_v3.sh <machine> dnfixed）。

  1. configs/denoise.yaml と ../configs/machines.yaml（使うキーだけ）を configs/schema.py で検証（既定値なし）→ sh の上書き → 値域検査
  2. run = checkpoints/denoiser/<yyyy_mmdd_HHMM>/（util/run_paths.py）。launch.yaml に実効値を保存
  3. DenoiseDataset（PCD512 の patch、評価専用症例は除外）→ DenoiserUNet → 損失（models/denoiser.py）
     n2n: Neighbor2Neighbor（決定 §6.2）。γ は 0 → gamma_max を gamma_ramp_epochs で線形に上げる。compile 時は勾配側だけ compile した module を使い、
          勾配なしのフル解像度 forward は素の net で行う（n2n_loss）
     blindspot: フル解像度の patch に円盤盲点（radius / stride）を置き、環からの乱択で埋めた ỹ を通して中心画素だけの損失（blindspot_loss）。γ・lr の半減以外のスケジュールは無い。
          最初の step で置換の検査（置換元が隠した画素に無い）を 1 回行う。起動行に U-Net の受容野（receptive_field）を出す（2r+1 より十分大きいこと）
     lr は epoch ≥ lr_half_epoch（含む）で 1/2。Adam（optim.beta1 / beta2）。check/* は FP32（util/dn_check.py）
  4. epoch 末: latest/net_Dn.pth（毎 epoch）、weights/epoch_NNN/net_Dn.pth（save_epoch_freq）、固定スライス n_preview 枚の preview（[x | Dn(x) | x − Dn(x)]）と
     check/*（util/dn_check.evaluate_denoiser、§6.4 の指標）を TensorBoard に出す。最後に checkpoints/denoiser/latest/ へ net_Dn.pth と launch.yaml を複製
     （train.yaml の pipeline.dn_weights の既定の参照先。symlink ではなく複製 = Windows バインドマウント対策）
  再開（resume）は無い。dn は毎回新規に始める（約 50 epoch で足りる想定。途中の重みは weights/epoch_NNN/ に残る）。
  TensorBoard は bash start1_v3.sh <machine> tb（checkpoints 全体が logdir なので denoiser/<run>/tb も見える）。
"""

import argparse
import datetime
import os
import random
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from configs.schema import DENOISE, MACHINE, ConfigError, apply_overrides, check_denoise_values, validate, validate_shared  # noqa: E402
from data.dataset import write_png  # noqa: E402
from data.denoise_dataset import DenoiseDataset  # noqa: E402
from models.denoiser import DenoiserUNet, blindspot_loss, n2n_loss, receptive_field, save_denoiser  # noqa: E402
from util.dn_check import LINE_KEYS, SUMMARY_KEYS, evaluate_denoiser, make_denoise_fn  # noqa: E402
from util.panel import build_panel  # noqa: E402
from util.residual_color import residual_rgb01, rgb_to_bgr  # noqa: E402
from util.run_paths import DATASET_INFO_FILE, DN_WEIGHT, LAUNCH_FILE, LOSS_LOG_FILE, STAGE, TB_DIR, WEIGHTS_DIR, checkpoints_root, denoiser_latest_dir, denoiser_run_dir, epoch_dirname, epoch_images_dir, run_name  # noqa: E402


def gamma_at(dn, epoch):
    """epoch（1 始まり）の γ: ramp 中は epoch 開始時の値で一定（0 → gamma_max を gamma_ramp_epochs で線形）。"""
    n = dn["n2n.gamma_ramp_epochs"]
    if n <= 0:
        return dn["n2n.gamma_max"]
    return dn["n2n.gamma_max"] * min(max((epoch - 1) / float(n), 0.0), 1.0)


def lr_at(dn, epoch):
    """epoch（1 始まり）≥ lr_half_epoch（含む）で lr を 1/2。0 なら一定。"""
    h = dn["optim.lr_half_epoch"]
    return dn["optim.lr"] * (0.5 if (h > 0 and epoch >= h) else 1.0)


def _copy_tree_files(src_dir, dst_dir, names):
    tmp = dst_dir.with_name(dst_dir.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    for n in names:
        shutil.copy2(src_dir / n, tmp / n)
    old = dst_dir.with_name(dst_dir.name + ".old")
    if old.exists():
        shutil.rmtree(old)
    if dst_dir.exists():
        os.replace(dst_dir, old)
    os.replace(tmp, dst_dir)
    if old.exists():
        shutil.rmtree(old, ignore_errors=True)


def main():
    p = argparse.ArgumentParser(description=f"{STAGE} Dn（denoiser）学習")
    p.add_argument("--machine", required=True)
    p.add_argument("--denoise", required=True)
    p.add_argument("--machines", required=True)
    p.add_argument("--write_run_dir", default=None, help="run ディレクトリのパスをこのファイルに書く（pipeline_v4.sh が次の段に渡すため）")
    p.add_argument("overrides", nargs=argparse.REMAINDER)
    a = p.parse_args()
    overrides = a.overrides[1:] if a.overrides and a.overrides[0] == "--" else a.overrides
    jst = datetime.timezone(datetime.timedelta(hours=9), name="JST")
    now = datetime.datetime.now(jst)
    try:
        with open(a.denoise, encoding="utf-8") as f:
            dn = validate("denoise", yaml.safe_load(f), DENOISE)
        with open(a.machines, encoding="utf-8") as f:
            machines = yaml.safe_load(f)
        if not isinstance(machines, dict) or a.machine not in machines:
            raise ConfigError(f"machines.yaml にエントリ '{a.machine}' がありません")
        machine = validate_shared(f"machines.{a.machine}", machines[a.machine], MACHINE)
        applied = apply_overrides(overrides, {"denoise": (dn, DENOISE)})
        check_denoise_values(dn, machine)
        if dn["dn.method"] == "bm3d":
            raise ConfigError("dn.method bm3d は学習しない（固定 denoiser）。bash start1_v3.sh <machine> dnfixed でデータを denoise し、train は --dn_source precomputed --dn_dir <出力> で読む")
        if not Path(dn["data.input_dir"]).is_dir():
            raise ConfigError(f"data.input_dir がありません: {dn['data.input_dir']}")
        name = run_name(now)
        run_dir = denoiser_run_dir(machine, name)
        if run_dir.exists():
            raise ConfigError(f"run ディレクトリが既に存在します: {run_dir}（同一分の再起動。1 分待つか、不要なら消してから起動）")
    except ConfigError as e:
        print(f"[dn] 設定エラー: {e}", file=sys.stderr)
        sys.exit(2)

    run_dir.mkdir(parents=True)
    with open(run_dir / LAUNCH_FILE, "w", encoding="utf-8") as f:
        yaml.safe_dump({"stage": STAGE, "kind": "denoiser", "timestamp": now.isoformat(timespec="seconds"), "machine_name": a.machine,
                        "denoise": dn, "machine": machine, "overrides": list(overrides), "applied": applied}, f, allow_unicode=True, sort_keys=False)
    if a.write_run_dir:
        Path(a.write_run_dir).write_text(str(run_dir) + "\n", encoding="utf-8")
    print(f"[dn] {STAGE} denoiser run {name} → {run_dir}", flush=True)

    seed = dn["optim.seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if machine["gpu_gen"] != 0 and device.type != "cuda":
        raise RuntimeError("machines.yaml の gpu_gen ≠ 0 なのに CUDA が使えません（gpu_gen: 0 にすれば cpu で動く）")
    torch.backends.cudnn.benchmark = True
    use_bf16 = dn["optim.precision"] == "bf16" and device.type == "cuda"
    compile_enabled = dn["optim.compile"] and device.type == "cuda"

    dataset = DenoiseDataset(dn, cache_dir=checkpoints_root(machine) / ".case_index")
    with open(run_dir / DATASET_INFO_FILE, "w", encoding="utf-8") as f:
        yaml.safe_dump(dataset.info(), f, allow_unicode=True, sort_keys=False)
    bs = dn["optim.batch_size"]
    loader = DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=machine["num_threads"], pin_memory=(device.type == "cuda"), drop_last=True)
    net = DenoiserUNet(dn["model.ch"], dn["model.n_levels"]).to(device)
    run_net = net
    if compile_enabled:
        run_net = torch.compile(net, mode="max-autotune")
    n_params = sum(p.numel() for p in net.parameters())
    opt = torch.optim.Adam(net.parameters(), lr=dn["optim.lr"], betas=(dn["optim.beta1"], dn["optim.beta2"]))
    method = dn["dn.method"]
    s, d0 = dn["n2n.cell_size"], dn["n2n.min_dist"]
    r, stride, bs_loss = dn["blindspot.radius"], dn["blindspot.stride"], dn["blindspot.loss"]
    rf = receptive_field(net, device)
    if method == "n2n":
        desc = f"N2N cell {s} min_dist {d0} gamma→{dn['n2n.gamma_max']} over {dn['n2n.gamma_ramp_epochs']} ep (sub-sampled {dn['data.patch_size'] // s}²)"
    else:
        desc = f"blindspot radius {r} (disk {2 * r + 1}²) stride {stride} fill {dn['blindspot.fill']} loss {bs_loss} (full-res {dn['data.patch_size']}², centres {100.0 / stride ** 2:.2f} %)"
        if rf < 3 * (2 * r + 1):
            raise RuntimeError(f"U-Net の受容野 {rf} px が盲点の 3 倍 {3 * (2 * r + 1)} px に足りません（model.n_levels / ch を増やすか radius を下げる）")
    print(f"[dn] method {method} | U-Net ch {dn['model.ch']} levels {dn['model.n_levels']} ({n_params / 1e6:.2f}M, receptive field {rf} px) | {desc} "
          f"| lr {dn['optim.lr']:.1e} (1/2 from epoch {dn['optim.lr_half_epoch']}) betas ({dn['optim.beta1']}, {dn['optim.beta2']}) | batch {bs} | precision {dn['optim.precision']} | compile {compile_enabled} | device {device}", flush=True)

    fixed = dataset.fixed_slices(dn["log.n_preview"])
    tb = SummaryWriter(log_dir=str(run_dir / TB_DIR))
    if os.environ.get("SPICA_TB_PORT"):  # bash start1_v3.sh <machine> dn / v4 から起動したとき: この run の tb を同じポートで配信（v4 通しでは後段の run_train が同じポートで自分の run に切り替える。全 run は bash start1_v3.sh <machine> tb）
        from run_train import start_tensorboard
        try:
            start_tensorboard(run_dir, int(os.environ["SPICA_TB_PORT"]))
        except ConfigError as e:
            print(f"[dn] 設定エラー: {e}", file=sys.stderr)
            sys.exit(2)
    log_path = run_dir / LOSS_LOG_FILE
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"================ Denoiser Training ({time.strftime('%Y-%m-%d %H:%M:%S')}) ================\n")
    hu_norm = (dn["data.hu_offset"], dn["data.hu_min"], dn["data.hu_max"])
    thr = float(dn["check.body_threshold_hu"])
    lo, hi, rng_hu = dn["log.display_hu_min"], dn["log.display_hu_max"], dn["log.diff_range_hu"]
    gray = lambda hu: np.repeat(np.clip((hu - lo) / float(hi - lo), 0.0, 1.0)[..., None], 3, axis=2).astype(np.float32)  # noqa: E731
    autocast = lambda: torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16)  # noqa: E731
    sub_gen = torch.Generator(device=device).manual_seed(dn["data.patch_seed"])

    total_steps = 0
    n_epochs, print_freq, save_epoch = dn["optim.n_epochs"], dn["log.print_freq"], dn["log.save_epoch_freq"]
    last_latest = None
    for epoch in range(1, n_epochs + 1):
        t_epoch = time.time()
        gamma, lr = (gamma_at(dn, epoch) if method == "n2n" else 0.0), lr_at(dn, epoch)
        for g in opt.param_groups:
            g["lr"] = lr
        net.train()
        sums, n_sum = {"rec": 0.0, "reg": 0.0, "total": 0.0}, 0
        bar = tqdm(total=len(dataset) // bs * bs, desc=f"Dn {epoch}/{n_epochs}", unit="img", leave=False, dynamic_ncols=True, mininterval=0.5)
        try:
            for batch in loader:
                y = batch["Y"].to(device, non_blocking=True)
                with autocast():
                    if method == "n2n":
                        loss, l_rec, l_reg = n2n_loss(run_net, y, s, d0, gamma, sub_gen, net_full=net)
                    else:
                        loss, _ = blindspot_loss(run_net, y, r, stride, bs_loss, sub_gen, verify=(total_steps == 0))
                        l_rec, l_reg = loss, torch.zeros((), device=device)
                opt.zero_grad(set_to_none=True)
                loss.float().backward()
                opt.step()
                total_steps += 1
                nb = int(y.shape[0])
                vals = {"rec": float(l_rec.detach()), "reg": float(l_reg.detach()), "total": float(loss.detach())}
                for k, v in vals.items():
                    sums[k] += v * nb
                n_sum += nb
                bar.update(nb)
                if total_steps % print_freq == 0:
                    bar.set_postfix({k: f"{v:.5f}" for k, v in vals.items()}, refresh=False)
                    for k, v in vals.items():
                        tb.add_scalar(f"loss/{k}", v, total_steps)
                    if method == "n2n":
                        tb.add_scalar("n2n/gamma", gamma, total_steps)
                    tb.add_scalar("lr", lr, total_steps)
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(f"(epoch: {epoch}, step: {total_steps}) " + " ".join(f"{k}: {v:.6f}" for k, v in vals.items()) + f" gamma: {gamma:.3f} lr: {lr:.2e}\n")
        finally:
            bar.close()
        # ---- epoch 末: 保存
        net.eval()
        latest = run_dir / "latest"
        tmp = latest.with_name("latest.tmp")
        tmp.mkdir(parents=True, exist_ok=True)
        save_denoiser(net, tmp / DN_WEIGHT)
        if latest.exists():
            shutil.rmtree(latest)
        os.replace(tmp, latest)
        last_latest = latest / DN_WEIGHT
        if epoch % save_epoch == 0 or epoch == n_epochs:
            ep_dir = run_dir / WEIGHTS_DIR / epoch_dirname(epoch)
            ep_dir.mkdir(parents=True, exist_ok=True)
            save_denoiser(net, ep_dir / DN_WEIGHT)
        # ---- preview と check/*
        rows, summary, images = evaluate_denoiser(make_denoise_fn(net, device, hu_norm), fixed["hu"], hu_norm, thr, seed=dn["data.patch_seed"], return_images=True)
        for k in SUMMARY_KEYS:
            tb.add_scalar(f"check/{k}", summary[k], total_steps)
        out_dir = epoch_images_dir(run_dir, epoch)
        for i, (pth, (x, xc, resid)) in enumerate(zip(fixed["paths"], images)):
            panel = build_panel([gray(x), gray(xc), residual_rgb01(resid, rng_hu)], ["PCD512  x", f"Dn(x)   epoch {epoch}", "x - Dn(x)   [HU]"], rng_hu)
            write_png(out_dir / f"{Path(pth).stem}_panel.png", rgb_to_bgr((np.clip(panel, 0, 1) * 65535.0).round().astype(np.uint16)))
            if i < 2:
                small = cv2.resize(panel, (panel.shape[1] * 512 // max(panel.shape[0], 1), 512), interpolation=cv2.INTER_AREA)
                tb.add_image(f"images/preview_{i}", (np.clip(small, 0, 1) * 255.0).round().astype(np.uint8), total_steps, dataformats="HWC")
        means = {k: v / max(n_sum, 1) for k, v in sums.items()}
        line = (f"dn {method} epoch {epoch}/{n_epochs} | " + " ".join(f"{k} {v:.6f}" for k, v in means.items()) + (f" | gamma {gamma:.2f}" if method == "n2n" else "") + f" lr {lr:.1e} | "
                + " ".join(f"{k} {summary[k]:.2f}" for k in LINE_KEYS) + f" ({summary['n_valid']}/{summary['n_slices']} slices) | {time.time() - t_epoch:.0f}s")
        tqdm.write(line)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        tb.add_scalar("train/epoch", epoch, total_steps)
    tb.flush(); tb.close()
    # ---- denoiser/latest/ へ複製（train.yaml pipeline.dn_weights の既定の参照先）
    latest_dir = denoiser_latest_dir(machine)
    _copy_tree_files(run_dir, latest_dir, [LAUNCH_FILE])
    shutil.copy2(last_latest, latest_dir / DN_WEIGHT)
    print(f"[dn] 完了: {run_dir}（latest: {last_latest}）→ 複製 {latest_dir / DN_WEIGHT}", flush=True)


if __name__ == "__main__":
    main()
