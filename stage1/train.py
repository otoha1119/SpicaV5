"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""

import os
import random
import time

import numpy as np
import torch
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.monitor import TrainMonitor  # [SpicaV5] 進捗バー + TensorBoard + loss_log.txt（本家 util/visualizer.py は使わない）
from util.util import init_ddp, cleanup_ddp

# [SpicaV5] 本家 train.py からの変更点（docs/plans/20260907_training-monitor-plan.md, stage1/UPSTREAM.md）
#   - 表示: Visualizer → TrainMonitor（tqdm バー / TensorBoard / loss_log.txt / uint16 サンプル画像）。display_freq / update_html_freq は使わない
#   - 再開: total_iters を model.resume_total_iters から復元、model.cur_epoch / epoch_done / total_iters を state 保存用に渡す
#   - print はバーを崩さないよう monitor.write（tqdm.write）にする

if __name__ == "__main__":
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:  # [SpicaV5] F-25: DDP 経路は未対応（ログ・画像・集計が単一プロセス前提）
        raise NotImplementedError("複数 GPU（DDP）は未対応です。単一プロセスで起動してください")
    opt = TrainOptions().parse()  # get training options
    # [SpicaV5] F-15: 乱数 seed（DataLoader worker の seed も torch の RNG から派生する）。resume 時はこの後の setup で保存済み RNG に上書きされる
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    opt.device = init_ddp()
    if opt.require_cuda and opt.device.type != "cuda":  # [SpicaV5] F-16: gpu_gen ≠ 0 のマシンで CUDA が無ければ止める（推論側と同じ挙動）
        raise RuntimeError("machines.yaml の gpu_gen ≠ 0 なのに CUDA が使えません（gpu_gen: 0 にすれば cpu で動く）")
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)  # get the number of images in the dataset.
    print(f"The number of training images = {dataset_size}")

    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers（[SpicaV5] --resume_state があれば optimizer / RNG も復元）
    fixed = dataset.dataset.fixed_batch(opt.n_images) if hasattr(dataset.dataset, "fixed_batch") else None  # [SpicaV5] 監視用の固定サンプル（128 patch）
    full = dataset.dataset.fixed_full(opt.n_full_images) if hasattr(dataset.dataset, "fixed_full") else None  # [SpicaV5] checkpoint 時に書き出すフル 512 スライス
    monitor = TrainMonitor(opt, dataset_size, fixed)  # [SpicaV5]
    total_iters = getattr(model, "resume_total_iters", 0)  # [SpicaV5] 再開時は保存された画像枚数から続ける
    total_epochs = opt.n_epochs + opt.n_epochs_decay
    last_epoch, last_saved_epoch = None, None  # [SpicaV5] F-06: ループ終了時に最終 epoch が未保存なら保存する
    for epoch in range(opt.epoch_count, total_epochs + 1):
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
        model.cur_epoch, model.epoch_done = epoch, False  # [SpicaV5] 再開用: いま何 epoch 目か / その epoch を終えたか
        # Set epoch for DistributedSampler
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)

        for i, data in enumerate(monitor.epoch_bar(dataset, epoch, total_epochs)):  # [SpicaV5] tqdm バーで包む
            iter_start_time = time.time()  # timer for computation per iteration
            t_data = iter_start_time - iter_data_time

            n_batch = int(data["A"].shape[0])  # [SpicaV5] F-20: 端数バッチは実枚数で数える
            total_iters += n_batch
            epoch_iter += n_batch
            model.total_iters = total_iters  # [SpicaV5] 再開用: state 保存時に使う
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()  # calculate loss functions, get gradients, update network weights
            monitor.accumulate(model, n_batch)  # [SpicaV5] F-22: epoch 要約は全 step の平均

            step = total_iters // opt.batch_size  # [SpicaV5] print_freq / image_freq は step 単位
            if step % opt.print_freq == 0:
                monitor.step(epoch, total_iters, model, time.time() - iter_start_time, t_data)
            if step % opt.image_freq == 0:
                monitor.log_images(model, total_iters)

            if total_iters % opt.save_latest_freq == 0:  # cache our latest model every <save_latest_freq> iterations（画像枚数）
                monitor.write(f"saving the latest model (epoch {epoch}, total_iters {total_iters})")
                save_suffix = f"iter_{total_iters}" if opt.save_by_iter else "latest"
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        model.epoch_done = True  # [SpicaV5] この epoch は完了（epoch 末の保存はここより後）
        model.update_learning_rate()  # update learning rates at the end of every epoch

        if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
            monitor.write(f"saving the model at the end of epoch {epoch}, iters {total_iters}")
            model.save_networks("latest")
            model.save_networks(epoch)
            monitor.save_full_images(model, full, epoch, total_iters)  # [SpicaV5] PCD / EID-like / R をフル 512 で保存
            last_saved_epoch = epoch
        last_epoch = epoch

        monitor.end_epoch(epoch, total_epochs, total_iters, model.optimizers[0].param_groups[0]["lr"], time.time() - epoch_start_time)

    if last_epoch is not None and last_saved_epoch != last_epoch:  # [SpicaV5] F-06: 最終 epoch が save_epoch_freq の倍数でなくても最終状態を残す
        monitor.write(f"saving the final model (epoch {last_epoch}, iters {total_iters})")
        model.save_networks("latest")
        model.save_networks(last_epoch)
        monitor.save_full_images(model, full, last_epoch, total_iters)

    monitor.close()
    cleanup_ddp()
