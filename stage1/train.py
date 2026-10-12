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

import math
import os
import random
import time

import numpy as np
import torch
import yaml
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.monitor import TrainMonitor  # [SpicaV5] 進捗バー + TensorBoard + loss_log.txt（本家 util/visualizer.py は使わない）
from util.util import init_ddp, cleanup_ddp

# [SpicaV5] 本家 train.py からの変更点（docs/plans/20260907_training-monitor-plan.md, stage1/UPSTREAM.md）
#   - 表示: Visualizer → TrainMonitor（tqdm バー / TensorBoard / loss_log.txt / uint16 サンプル画像）。display_freq / update_html_freq は使わない
#   - 再開: total_iters を model.resume_total_iters から復元、model.cur_epoch / epoch_done / total_iters を state 保存用に渡す
#   - print はバーを崩さないよう monitor.write（tqdm.write）にする
#   - 2 モデル同時学習（bash start.sh multi、2026-10-13、docs/plans/20261012_multi-model-training-plan.md）: opt / model / monitor をリストで持ち、
#     DataLoader は 1 つ。1 batch ごとにモデル 1 → モデル 2 の順に set_input + optimize_parameters。単独実行は長さ 1 の同じループ（分岐は増やさない）。
#     2 つ目以降の opt は --co_launch（run_train.py が付ける）の launch.yaml の argv から組む。片方の損失が非有限（NaN / inf）か例外なら全部止める


def parse_opts():
    """先頭モデルの opt は sys.argv（従来どおり）。--co_launch にある launch.yaml（run_train.py が書いた実効設定。argv を持つ）から 2 つ目以降の opt を組む。"""
    opt = TrainOptions().parse()
    opts = [opt]
    for path in opt.co_launch:
        with open(path, encoding="utf-8") as f:
            launch = yaml.safe_load(f)
        if not isinstance(launch, dict) or not isinstance(launch.get("argv"), list):
            raise RuntimeError(f"--co_launch {path} に argv がありません（run_train.py が書く launch.yaml を指定する）")
        co = TrainOptions().parse([str(t) for t in launch["argv"]])
        if co.co_launch:
            raise RuntimeError(f"--co_launch {path} の argv にさらに --co_launch があります（入れ子は不可）")
        opts.append(co)
    return opts


def check_finite(model, label):
    """[SpicaV5] 損失が非有限なら止める（組の片方が NaN になったら両方止める。各 run の latest/ は直前の保存のまま）。"""
    bad = {k: v for k, v in model.get_current_losses().items() if not math.isfinite(v)}
    if bad:
        raise RuntimeError(f"損失が非有限です（{label}）: {bad} → 組の全モデルを止める（各 run の latest/ は直前の保存のまま。続きは bash start.sh resume）")


if __name__ == "__main__":
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:  # [SpicaV5] F-25: DDP 経路は未対応（ログ・画像・集計が単一プロセス前提）
        raise NotImplementedError("複数 GPU（DDP）は未対応です。単一プロセスで起動してください")
    opts = parse_opts()  # get training options（[SpicaV5] 組なら複数）
    opt = opts[0]  # [SpicaV5] データ・epoch・ログ頻度など組で共有するキーは先頭の opt を使う（run_train.py が組で同一を保証）
    # [SpicaV5] F-15: 乱数 seed（DataLoader worker の seed も torch の RNG から派生する）。resume 時はこの後の setup で保存済み RNG に上書きされる。
    #   グローバル RNG はデータ側（patch 抽出）用。モデル側（重みの初期化、乱数画像 n）は各モデルの torch.Generator（FidelityGANModel.rng。2026-10-13）
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    device = init_ddp()
    for o in opts:
        o.device = device
    if opt.require_cuda and device.type != "cuda":  # [SpicaV5] F-16: gpu_gen ≠ 0 のマシンで CUDA が無ければ止める（推論側と同じ挙動）
        raise RuntimeError("machines.yaml の gpu_gen ≠ 0 なのに CUDA が使えません（gpu_gen: 0 にすれば cpu で動く）")
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options（[SpicaV5] 組で 1 つ）
    dataset_size = len(dataset)  # get the number of images in the dataset.
    print(f"The number of training images = {dataset_size}")

    labels = [o.name.split("-", 1)[1] if "-" in o.name else o.name for o in opts] if len(opts) > 1 else [None]  # [SpicaV5] 組のときだけ run 名の接尾辞（mode ファイル名）で区別
    models = []
    for o in opts:
        model = create_model(o)  # create a model given opt.model and other options
        model.setup(o)  # regular setup: load and print networks; create schedulers（[SpicaV5] --resume_state があれば optimizer / RNG も復元）
        models.append(model)
    fixed = dataset.dataset.fixed_batch(opt.n_images) if hasattr(dataset.dataset, "fixed_batch") else None  # [SpicaV5] 監視用の固定サンプル（128 patch）
    full_slices = dataset.dataset.full_slices if hasattr(dataset.dataset, "full_slices") else (lambda epoch: None)  # [SpicaV5] checkpoint 時に書き出すフル 512（固定 1 + epoch ごとのランダム）
    eid_ref = dataset.dataset.eid_reference() if hasattr(dataset.dataset, "eid_reference") else None  # [SpicaV5] パネルの右端に並べる EID の代表（固定）
    monitors = [TrainMonitor(o, dataset_size, fixed, eid_ref, label=lab) for o, lab in zip(opts, labels)]  # [SpicaV5] run ごとに 1 つ（TB / loss_log.txt）。バーは先頭だけ
    resume_iters = [getattr(model, "resume_total_iters", 0) for model in models]  # [SpicaV5] 再開時は保存された画像枚数から続ける
    if len(set(resume_iters)) != 1:
        raise RuntimeError(f"組の各モデルの進捗（total_iters）が一致しません: {dict(zip([o.name for o in opts], resume_iters))}（同じ時点の checkpoint を組にしてください）")
    total_iters = resume_iters[0]
    total_epochs = opt.n_epochs + opt.n_epochs_decay
    last_epoch, last_saved_epoch = None, None  # [SpicaV5] F-06: ループ終了時に最終 epoch が未保存なら保存する
    for epoch in range(opt.epoch_count, total_epochs + 1):
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
        for model in models:
            model.cur_epoch, model.epoch_done = epoch, False  # [SpicaV5] 再開用: いま何 epoch 目か / その epoch を終えたか
        # Set epoch for DistributedSampler
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)
        for monitor in monitors[1:]:
            monitor.begin_epoch()  # [SpicaV5] バーを持たない monitor の epoch 要約をリセット（先頭は epoch_bar がやる）

        for i, data in enumerate(monitors[0].epoch_bar(dataset, epoch, total_epochs)):  # [SpicaV5] tqdm バーで包む（組でも 1 本）
            iter_start_time = time.time()  # timer for computation per iteration
            t_data = iter_start_time - iter_data_time

            n_batch = int(data["A"].shape[0])  # [SpicaV5] F-20: 端数バッチは実枚数で数える
            total_iters += n_batch
            epoch_iter += n_batch
            for model, monitor, label in zip(models, monitors, labels):  # [SpicaV5] 同じ batch をモデル 1 → 2 の順に
                model.total_iters = total_iters  # [SpicaV5] 再開用: state 保存時に使う
                model.set_input(data)  # unpack data from dataset and apply preprocessing
                model.optimize_parameters()  # calculate loss functions, get gradients, update network weights
                check_finite(model, label or model.opt.name)
                monitor.accumulate(model, n_batch)  # [SpicaV5] F-22: epoch 要約は全 step の平均

            step = total_iters // opt.batch_size  # [SpicaV5] print_freq / image_freq は step 単位
            if step % opt.print_freq == 0:
                t_comp = time.time() - iter_start_time
                postfix = {}
                for model, monitor in zip(models, monitors):
                    postfix.update(monitor.step(epoch, total_iters, model, t_comp, t_data))
                monitors[0].set_postfix(postfix)  # [SpicaV5] バー末尾に全モデルの主要損失を名前付きで並べる
            if step % opt.image_freq == 0:
                for model, monitor in zip(models, monitors):
                    monitor.log_images(model, total_iters)

            if total_iters % opt.save_latest_freq == 0:  # cache our latest model every <save_latest_freq> iterations（画像枚数）。表示はしない（2026-09-08 ユーザー指示。保存はする）
                save_suffix = f"iter_{total_iters}" if opt.save_by_iter else "latest"
                for model in models:
                    model.save_networks(save_suffix)

            iter_data_time = time.time()

        for model in models:
            model.epoch_done = True  # [SpicaV5] この epoch は完了（epoch 末の保存はここより後）
            model.update_learning_rate()  # update learning rates at the end of every epoch

        if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
            monitors[0].write(f"saving the model at the end of epoch {epoch}, iters {total_iters}")
            full = full_slices(epoch)  # [SpicaV5] 組で同じスライス（patch_seed と epoch から決定的）
            for model, monitor in zip(models, monitors):
                model.save_networks("latest")
                model.save_networks(epoch)
                monitor.save_full_images(model, full, epoch, total_iters)  # [SpicaV5] PCD / EID-like / R をフル 512 で保存
            last_saved_epoch = epoch
        last_epoch = epoch

        for model, monitor in zip(models, monitors):
            monitor.end_epoch(epoch, total_epochs, total_iters, model.optimizers[0].param_groups[0]["lr"], time.time() - epoch_start_time)

    if last_epoch is not None and last_saved_epoch != last_epoch:  # [SpicaV5] F-06: 最終 epoch が save_epoch_freq の倍数でなくても最終状態を残す
        monitors[0].write(f"saving the final model (epoch {last_epoch}, iters {total_iters})")
        full = full_slices(last_epoch)
        for model, monitor in zip(models, monitors):
            model.save_networks("latest")
            model.save_networks(last_epoch)
            monitor.save_full_images(model, full, last_epoch, total_iters)

    for monitor in monitors:
        monitor.close()
    cleanup_ddp()
