"""stage2/train.py — Stage 2（U-Net + MSE、ILUMENATE 準拠）の学習ループ。junyanz は使わない。**run_train.py 経由で起動する**（直接叩くと launch.yaml が無くて止まる）。

設計: docs/plans/20260908_stage2-unet-implementation-spec.md §4
  1. run_train.py が書いた launch.yaml（実効値）を読み、schema で再検証する（既定値なし）
  2. seed → device（machines.yaml の gpu_gen ≠ 0 なら CUDA 必須。黙って CPU に落ちない）→ PairDataset → DataLoader → RegressionModel
  3. 再開（--resume_state）なら重みと optimizer / RNG / 進捗を復元する
  4. epoch ループ: print_freq step ごとに scalar、image_freq step ごとに 128 patch グリッド（TB images/current, images/fixed）、save_latest_freq 枚ごとに latest/、epoch 末に val 指標 + latest/ + weights/epoch_NNN/ + フル 1024 画像
     （固定 = train.yaml log.full_slice、ランダム = n_full_random 枚、実 EID テスト = log.eid_slice。output_images/ の生 16bit と preview、TB のパネル。util/monitor.py）
  5. 最終 epoch は save_epoch_freq の倍数でなくても保存する
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
from configs.schema import MACHINE, MODE, TRAIN, ConfigError, validate, validate_shared  # noqa: E402
from data.pair_dataset import PairDataset  # noqa: E402
from models.regression_model import RegressionModel  # noqa: E402
from util.monitor import TrainMonitor  # noqa: E402
from util.run_paths import DATASET_INFO_FILE, ckpt_dir  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="Stage 2 学習ループ（run_train.py から exec される）")
    p.add_argument("--launch", required=True, help="run_train.py が書いた解決済み設定（launch.yaml / launch_resume_*.yaml）")
    p.add_argument("--run_dir", required=True, help="run ディレクトリ")
    p.add_argument("--epoch_count", type=int, required=True, help="開始 epoch（新規 1、再開は run_train.py が state から決める）")
    p.add_argument("--resume_state", default=None, help="再開に使う重みディレクトリの state.pth（同じディレクトリの net_G.pth も読む）")
    p.add_argument("--require_cuda", action="store_true", help="CUDA が無ければ止める（gpu_gen ≠ 0 のとき run_train.py が付ける）")
    a = p.parse_args()

    launch = Path(a.launch)
    if not launch.is_file():
        print(f"[train] 設定エラー: launch.yaml がありません（run_train.py 経由で起動すること）: {launch}", file=sys.stderr)
        sys.exit(2)
    with open(launch, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    try:
        train = validate("train(launch)", cfg["train"], TRAIN)
        mode = validate("mode(launch)", cfg["mode"], MODE)
        machine = validate_shared("machine(launch)", cfg["machine"], MACHINE)
    except (ConfigError, KeyError, TypeError) as e:
        print(f"[train] 設定エラー: {launch}: {e}", file=sys.stderr)
        sys.exit(2)
    run_dir = Path(a.run_dir)

    random.seed(train["optim.seed"]); np.random.seed(train["optim.seed"]); torch.manual_seed(train["optim.seed"]); torch.cuda.manual_seed_all(train["optim.seed"])
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.backends.cudnn.benchmark = True  # Stage 1（junyanz base_model）と同じ。patch は固定サイズなので速くなる（完全な決定性は無い。seed の doc 参照）
    if a.require_cuda and device.type != "cuda":
        raise RuntimeError("machines.yaml の gpu_gen ≠ 0 なのに CUDA が使えません（gpu_gen: 0 にすれば cpu で動く）")

    dataset = PairDataset(train, mode, machine, cache_dir=Path(machine["stage2_checkpoints_dir"]) / ".case_index")
    with open(run_dir / DATASET_INFO_FILE, "w", encoding="utf-8") as f:
        yaml.safe_dump({"train_cases": dataset.train_cases, "val_cases": dataset.val_cases, "test_cases": dataset.test_cases,
                        "image_hw": list(dataset.img_hw), "counts": dataset.info,
                        "fixed_slice": list(dataset.fixed_pair), "eid_test_slice": dataset.eid_path,
                        "eid_upsample": {"scale": dataset.upsample_cfg[0], "interp": dataset.upsample_cfg[1], "from": f"{machine['eidlike1024_dir']}/manifest.yaml"},
                        "val_slices_per_epoch": len(dataset.val_slices())}, f, allow_unicode=True, sort_keys=False)
    bs = train["optim.batch_size"]
    loader = DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=machine["num_threads"], pin_memory=(device.type == "cuda"), drop_last=False)

    model = RegressionModel(train, mode, machine, device)
    print(f"[train] {mode['arch']} base_ch {mode['base_ch']} n_pool {mode['n_pool']} final_act {mode['final_act']} residual {mode['residual']}: {model.n_params / 1e6:.2f}M params, loss {mode['loss']}, device {device}")
    if a.resume_state:
        model.load_weights(Path(a.resume_state).parent)
        model.resume(a.resume_state)
    total_iters = model.total_iters
    val_slices = dataset.val_slices()
    fixed = dataset.fixed_patches(train["log.n_images"])  # TB images/fixed 用（log.full_slice から。位置は patch_seed で固定）
    monitor = TrainMonitor(run_dir, train, mode["loss"], bs, len(dataset))
    n_epochs, print_freq, save_latest, save_epoch = train["optim.n_epochs"], train["log.print_freq"], train["log.save_latest_freq"], train["log.save_epoch_freq"]
    image_freq = train["log.image_freq"]
    last_epoch, last_saved_epoch = None, None

    for epoch in range(a.epoch_count, n_epochs + 1):
        t_epoch = time.time()
        iter_data_time = time.time()
        model.cur_epoch, model.epoch_done = epoch, False
        for batch in monitor.epoch_bar(loader, epoch, n_epochs):
            t0 = time.time()
            t_data = t0 - iter_data_time
            n_batch = int(batch["input"].shape[0])
            total_iters += n_batch
            model.total_iters = total_iters
            losses = model.optimize(batch)
            monitor.accumulate(losses, time.time() - t0, t_data)
            step = total_iters // bs
            if step % print_freq == 0:
                monitor.step(epoch, total_iters)
            if step % image_freq == 0:
                monitor.log_images(model, fixed, total_iters)  # 128 patch グリッド（current / fixed）を TB へ
            if total_iters % save_latest == 0:
                model.save(run_dir, "latest")
            iter_data_time = time.time()
        model.epoch_done = True
        val = model.evaluate(dataset, val_slices)
        monitor.log_val(val, total_iters)
        if epoch % save_epoch == 0:
            model.save(run_dir, "latest")
            model.save(run_dir, epoch)
            monitor.save_full_images(model, dataset, dataset.full_slices(epoch), epoch, total_iters)  # 生 16bit / preview / TB（固定 + ランダム + 実 EID テスト）
            last_saved_epoch = epoch
        last_epoch = epoch
        monitor.end_epoch(epoch, n_epochs, total_iters, model.optimizer.param_groups[0]["lr"], time.time() - t_epoch, val)

    if last_epoch is not None and last_saved_epoch != last_epoch:
        monitor.write(f"saving the final model (epoch {last_epoch}, iters {total_iters})")
        model.save(run_dir, "latest")
        model.save(run_dir, last_epoch)
        monitor.save_full_images(model, dataset, dataset.full_slices(last_epoch), last_epoch, total_iters)
    monitor.write(f"[train] 完了: {run_dir}（latest: {ckpt_dir(run_dir, 'latest')}）")
    monitor.close()


if __name__ == "__main__":
    main()
