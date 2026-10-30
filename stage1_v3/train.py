"""stage1_v3/train.py — 射影型 Stage 1 の学習ループ（run_train.py から exec される）。

  1. launch.yaml（実効値）を読み schema で再検証（既定値なし）→ seed → device（gpu_gen ≠ 0 なら CUDA 必須）
  2. ProjectionDataset（A = PCD512 → P → U、B = EID512、patch 128、非ペア）→ ProjectionModel
  3. 再開（--resume_state）なら重み 2 つと optimizer / RNG / 進捗を復元
  4. epoch ループ: print_freq step ごとに scalar、image_freq step ごとに patch グリッド、save_latest_freq 枚ごとに latest/、
     epoch 末に lr 更新 → latest/ + weights/epoch_NNN/ + フル 512 画像（固定 PCD + ランダム PCD + 恒等経路 + n_draws 標準偏差）
     → 自動採点（util/validator.py、R-S22 の 3 軸 + 恒等残差）。score 最良なら best/ を更新
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
from configs.schema import MACHINE, TRAIN, ConfigError, validate, validate_shared  # noqa: E402
from data.dataset import ProjectionDataset  # noqa: E402
from models.projection_model import ProjectionModel  # noqa: E402
from util.monitor import TrainMonitor  # noqa: E402
from util.validator import Validator  # noqa: E402
from util.run_paths import DATASET_INFO_FILE, ckpt_dir, checkpoints_root  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="射影型 Stage 1 学習ループ（run_train.py から exec される）")
    p.add_argument("--launch", required=True)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--epoch_count", type=int, required=True)
    p.add_argument("--resume_state", default=None)
    p.add_argument("--require_cuda", action="store_true")
    a = p.parse_args()

    launch = Path(a.launch)
    if not launch.is_file():
        print(f"[train] 設定エラー: launch.yaml がありません（run_train.py 経由で起動すること）: {launch}", file=sys.stderr)
        sys.exit(2)
    with open(launch, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    try:
        train = validate("train(launch)", cfg["train"], TRAIN)
        machine = validate_shared("machine(launch)", cfg["machine"], MACHINE)
    except (ConfigError, KeyError, TypeError) as e:
        print(f"[train] 設定エラー: {launch}: {e}", file=sys.stderr)
        sys.exit(2)
    run_dir = Path(a.run_dir)

    seed = train["optim.seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.backends.cudnn.benchmark = True
    if a.require_cuda and device.type != "cuda":
        raise RuntimeError("machines.yaml の gpu_gen ≠ 0 なのに CUDA が使えません（gpu_gen: 0 にすれば cpu で動く）")

    dataset = ProjectionDataset(train, machine, cache_dir=checkpoints_root(machine) / ".case_index", run_dir=run_dir)
    with open(run_dir / DATASET_INFO_FILE, "w", encoding="utf-8") as f:
        yaml.safe_dump(dataset.info(), f, allow_unicode=True, sort_keys=False)
    bs = train["optim.batch_size"]
    loader = DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=machine["num_threads"], pin_memory=(device.type == "cuda"), drop_last=False)

    model = ProjectionModel(train, device, launch_config={"train": dict(train), "machine": dict(machine)})
    npar = model.n_params()
    print(f"[train] projection G {train['network.netG']} D {train['network.netD']} gan {train['loss.gan_mode']} "
          f"lambda_fid {train['loss.lambda_fid']} lambda_id {train['loss.lambda_id']}→{train['loss.lambda_id_final']} over {train['loss.lambda_id_decay_epochs']} ep | "
          f"P {'on' if train['degrade.enabled'] else 'off'} | " + " ".join(f"{k} {v / 1e6:.2f}M" for k, v in npar.items())
          + f" | lr_G {model.lr_G:.2e} lr_D {model.lr_D:.2e} | precision {model.precision} | compile {model.compile_enabled}"
          + (" (requested, CUDA only)" if model.compile_requested and not model.compile_enabled else "") + f" | device {device}")
    if a.resume_state:
        model.load_weights(Path(a.resume_state).parent)
        model.resume(a.resume_state, a.epoch_count)
    else:
        model.make_schedulers(a.epoch_count)
    total_iters = model.total_iters
    fixed = dataset.fixed_batch(train["log.n_images"])
    monitor = TrainMonitor(run_dir, train, len(dataset), fixed, dataset.eid_reference(), train["degrade.body_threshold_hu"])
    validator = Validator(dataset, train, run_dir, monitor.tb)
    total_epochs = train["optim.n_epochs"] + train["optim.n_epochs_decay"]
    print_freq, image_freq, save_latest, save_epoch = train["log.print_freq"], train["log.image_freq"], train["log.save_latest_freq"], train["log.save_epoch_freq"]
    last_epoch, last_saved_epoch = None, None

    for epoch in range(a.epoch_count, total_epochs + 1):
        t_epoch = time.time()
        iter_data_time = time.time()
        model.cur_epoch, model.epoch_done = epoch, False
        for batch in monitor.epoch_bar(loader, epoch, total_epochs):
            t0 = time.time()
            t_data = t0 - iter_data_time
            n_batch = int(batch["A"].shape[0])
            total_iters += n_batch
            model.total_iters = total_iters
            model.set_input(batch)
            losses = model.optimize_parameters()
            monitor.accumulate(losses, n_batch, time.time() - t0, t_data)
            step = total_iters // bs
            if step % print_freq == 0:
                monitor.step(epoch, total_iters, losses)
            if step % image_freq == 0:
                monitor.log_images(model, model.views(fixed) if fixed is not None else None, total_iters, model.views(batch))
            if total_iters % save_latest == 0:
                model.save(run_dir, "latest")
            iter_data_time = time.time()
        model.epoch_done = True
        model.update_learning_rate()
        if epoch % save_epoch == 0:
            monitor.write(f"saving the model at the end of epoch {epoch}, iters {total_iters}")
            model.save(run_dir, "latest")
            model.save(run_dir, epoch)
            monitor.save_full_images(model, dataset.full_slices(epoch), epoch, total_iters)
            validator.run(model, epoch, total_iters)
            last_saved_epoch = epoch
        last_epoch = epoch
        monitor.end_epoch(epoch, total_epochs, total_iters, model.lr_G, model.lr_D, time.time() - t_epoch)

    if last_epoch is not None and last_saved_epoch != last_epoch:
        monitor.write(f"saving the final model (epoch {last_epoch}, iters {total_iters})")
        model.save(run_dir, "latest")
        model.save(run_dir, last_epoch)
        monitor.save_full_images(model, dataset.full_slices(last_epoch), last_epoch, total_iters)
        validator.run(model, last_epoch, total_iters)
    if validator.enabled and validator.best_epoch is not None:
        monitor.write(f"[val] best: epoch {validator.best_epoch} (score {validator.best_score:.3f}){' → best/' if train['val.auto_best'] else ''}")
    monitor.write(f"[train] 完了: {run_dir}（latest: {ckpt_dir(run_dir, 'latest')}）")
    monitor.close()


if __name__ == "__main__":
    main()
