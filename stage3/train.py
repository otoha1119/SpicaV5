"""stage3/train.py — Stage 3（条件付き DDIM）の学習ループ。**run_train.py 経由で起動する**（直接叩くと launch.yaml が無くて止まる）。
本家 ermongroup/ddim runners/diffusion.Diffusion.train() のループを、SpicaV5 の流儀（launch.yaml・run レイアウト・TB・原子的保存・再開）で回す。1 step の中身は runners/diffusion.CondDiffusion.train_step。

  1. run_train.py が書いた launch.yaml（実効値）を読み、schema で再検証する
  2. seed → device（gpu_gen ≠ 0 なら CUDA 必須）→ PairDataset → DataLoader → CondDiffusion
  3. 再開（--resume_state）なら model / ema / optimizer / RNG / step を復元する
  4. step ループ（ddim.yml training.n_iters まで。1 epoch = train.yaml log.steps_per_epoch step（10,000）。バー 1 本と epoch 末のログの区切りだけ）:
       print_freq        TB loss/*・バー・loss_log.txt
       validation_freq   固定 val patch の ε の MSE（学習中の重み / EMA）
       image_freq        固定 val patch を EMA の重みで DDIM サンプリング → TB images/val_patch
       snapshot_freq     latest/（本家と同じ間隔。本家どおり step 1 でも保存する）
       save_weights_freq weights/iter_NNNNNNN/
       full_freq         固定スライス + 実 EID を 1024 のままタイル合成でサンプリング → output_images/ と TB
  5. n_iters に達したら latest/ と weights/ を保存して終わる
"""

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from configs.schema import DDIM, MACHINE, TRAIN, ConfigError, check_train_values, validate, validate_shared  # noqa: E402
from data.pair_dataset import PairDataset  # noqa: E402
from runners.diffusion import CondDiffusion, count_tiles, describe_seq, slice_seed  # noqa: E402
from util.monitor import TrainMonitor  # noqa: E402
from util.run_paths import DATASET_INFO_FILE, ckpt_dir  # noqa: E402

VAL_SALT, IMG_SALT = 101, 202  # 固定 val patch の選び方（ε 損失用と画像用で別の patch）


def main():
    p = argparse.ArgumentParser(description="Stage 3 学習ループ（run_train.py から exec される）")
    p.add_argument("--launch", required=True)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--resume_state", default=None, help="再開に使う重みディレクトリの state.pth（同じディレクトリの model.pth / ema.pth も読む）")
    p.add_argument("--require_cuda", action="store_true")
    a = p.parse_args()

    launch = Path(a.launch)
    if not launch.is_file():
        print(f"[train] 設定エラー: launch.yaml がありません（run_train.py 経由で起動すること）: {launch}", file=sys.stderr)
        sys.exit(2)
    with open(launch, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    try:
        ddim = validate("ddim(launch)", cfg["ddim"], DDIM)
        train = validate("train(launch)", cfg["train"], TRAIN)
        machine = validate_shared("machine(launch)", cfg["machine"], MACHINE)
        check_train_values(train, ddim, machine)
    except (ConfigError, KeyError, TypeError) as e:
        print(f"[train] 設定エラー: {launch}: {e}", file=sys.stderr)
        sys.exit(2)
    run_dir = Path(a.run_dir)

    seed = train["optim.seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.backends.cudnn.benchmark = True  # 本家 main.py と同じ
    if a.require_cuda and device.type != "cuda":
        raise RuntimeError("machines.yaml の gpu_gen ≠ 0 なのに CUDA が使えません（gpu_gen: 0 にすれば cpu で動く）")

    dataset = PairDataset(train, ddim, machine, cache_dir=Path(machine["stage3_checkpoints_dir"]) / ".case_index")
    h, w = dataset.img_hw
    patch = ddim["data.image_size"]
    try:
        n_tiles = count_tiles(h, w, patch, train["log.tile_stride"])
    except ValueError as e:
        print(f"[train] 設定エラー: log.tile_stride: {e}", file=sys.stderr)
        sys.exit(2)
    with open(run_dir / DATASET_INFO_FILE, "w", encoding="utf-8") as f:
        yaml.safe_dump({"train_cases": dataset.train_cases, "val_cases": dataset.val_cases, "test_cases": dataset.test_cases,
                        "image_hw": list(dataset.img_hw), "patch": patch, "counts": dataset.info,
                        "fixed_slice": list(dataset.fixed_pair), "eid_test_slice": dataset.eid_path,
                        "eid_upsample": {"scale": dataset.upsample_cfg[0], "interp": dataset.upsample_cfg[1], "from": f"{machine['eidlike1024_dir']}/manifest.yaml"}},
                       f, allow_unicode=True, sort_keys=False)
    bs = ddim["training.batch_size"]
    loader = DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=machine["num_threads"], pin_memory=(device.type == "cuda"), drop_last=True)

    diffusion = CondDiffusion(ddim, device, train=True)
    diffusion.run_config = {"ddim": dict(ddim), "train": dict(train), "machine": dict(machine)}
    if a.resume_state:
        diffusion.load_weights(Path(a.resume_state).parent, use_ema=False)
        diffusion.resume(a.resume_state)
    s = {"steps": train["log.sample_steps"], "eta": train["log.sample_eta"], "skip_type": train["log.skip_type"], "seed": train["log.sample_seed"],
         "stride": train["log.tile_stride"], "tile_batch": train["log.tile_batch"], "n_tiles": n_tiles}
    print(f"[train] ε-net（本家 models/diffusion.Model）: {diffusion.n_params / 1e6:.1f}M params, ch {ddim['model.ch']} ch_mult {ddim['model.ch_mult']} attn {ddim['model.attn_resolutions']}, "
          f"in {ddim['model.in_channels']} out {ddim['model.out_ch']}, patch {patch}, batch {bs}, lr {ddim['optim.lr']}, EMA {ddim['model.ema']} ({ddim['model.ema_rate']}), "
          f"T {ddim['diffusion.num_diffusion_timesteps']} {ddim['diffusion.beta_schedule']}, device {device}")
    print(f"[train] 監視のサンプリング: DDIM η {s['eta']}, {describe_seq(diffusion.num_timesteps, s['steps'], s['skip_type'])}, {h}x{w} は {patch} タイル × {n_tiles} 枚（r {s['stride']}）")

    val_fixed = dataset.val_patches(train["log.n_val_patches"], VAL_SALT)
    img_fixed = dataset.val_patches(train["log.n_images"], IMG_SALT)
    n_pixels = ddim["data.channels"] * patch * patch
    monitor = TrainMonitor(run_dir, train, bs, n_pixels)
    n_iters = ddim["training.n_iters"]
    snap, vfreq = ddim["training.snapshot_freq"], ddim["training.validation_freq"]
    print_freq, image_freq, full_freq, wfreq = train["log.print_freq"], train["log.image_freq"], train["log.full_freq"], train["log.save_weights_freq"]
    seed_of = lambda name: slice_seed(s["seed"], name)  # noqa: E731
    last_saved = None
    spe = train["log.steps_per_epoch"]
    n_epochs = -(-n_iters // spe)
    print(f"[train] 1 epoch = {spe} step（= {spe * bs} patch）、n_iters {n_iters} step = {n_epochs} epoch。"
          f"監視: val ε {vfreq} / val patch 画像 {image_freq} / latest {snap} / weights {wfreq} / 1024 合成 {full_freq} step ごと")

    while diffusion.step < n_iters:
        diffusion.epoch = diffusion.step // spe + 1  # epoch は step から決まる（再開で epoch の途中から始めても数え方が変わらない）
        offset = diffusion.step % spe
        todo = min(spe - offset, n_iters - diffusion.step)
        t_epoch = time.time()
        monitor.epoch_bar(diffusion.epoch, n_epochs, spe, offset, diffusion.step, n_iters)
        iter_data_time = time.time()
        for k, batch in enumerate(loader):  # DataLoader 1 周 = spe step 分。再開時は残りの todo step だけ回す
            t0 = time.time()
            t_data = t0 - iter_data_time
            losses = diffusion.train_step(batch)
            step = diffusion.step
            monitor.accumulate(losses, time.time() - t0, t_data)
            monitor.tick()
            if step % print_freq == 0:
                monitor.step(diffusion.epoch, step)
            need_ema = step % vfreq == 0 or step % image_freq == 0 or step % full_freq == 0
            ema = diffusion.ema_model() if need_ema else None
            if step % vfreq == 0:
                val = {"model": diffusion.eps_loss(val_fixed["input"], val_fixed["target"])}
                if diffusion.ema_helper is not None:
                    val["ema"] = diffusion.eps_loss(val_fixed["input"], val_fixed["target"], model=ema)
                monitor.log_val(val, step)
            if step % image_freq == 0:
                monitor.log_val_patches(diffusion, img_fixed, s, step, ema)
            if step % snap == 0 or step == 1:  # 本家: if step % snapshot_freq == 0 or step == 1
                diffusion.save(run_dir, "latest")
                last_saved = step
            if step % wfreq == 0:
                diffusion.save(run_dir, step)
            if step % full_freq == 0:
                monitor.save_full_images(diffusion, dataset, s, step, ema, seed_of)
            del ema
            iter_data_time = time.time()
            if k + 1 >= todo:
                break
        monitor.end_epoch(diffusion.epoch, diffusion.step, diffusion.optimizer.param_groups[0]["lr"], time.time() - t_epoch, done=diffusion.step % spe == 0)

    step = diffusion.step
    if last_saved != step:
        diffusion.save(run_dir, "latest")
    if step % wfreq != 0:
        diffusion.save(run_dir, step)
    monitor.write(f"[train] 完了: step {step} / n_iters {n_iters} → {ckpt_dir(run_dir, 'latest')}, {ckpt_dir(run_dir, step)}")
    monitor.close()


if __name__ == "__main__":
    main()
