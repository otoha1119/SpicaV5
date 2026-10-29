"""stage2/train.py — Stage 2（U-Net + MSE、ILUMENATE 準拠）の学習ループ。junyanz は使わない。**run_train.py 経由で起動する**（直接叩くと launch.yaml が無くて止まる）。

設計: docs/plans/20260908_stage2-unet-implementation-spec.md §4
  1. run_train.py が書いた launch.yaml（実効値）を読み、schema で再検証する（既定値なし）
  2. seed → device（machines.yaml の gpu_gen ≠ 0 なら CUDA 必須。黙って CPU に落ちない）→ PairDataset → DataLoader → RegressionModel
  3. 再開（--resume_state）なら重みと optimizer / RNG / 進捗を復元する。新規 run の --init_G_from なら G だけを読み、D / optimizer / 進捗は新規にする
  4. epoch ループ: print_freq step ごとに scalar、image_freq step ごとに 128 patch グリッド（TB images/current, images/fixed）、save_latest_freq 枚ごとに latest/、epoch 末に val 指標 →（log.best_metric が最良なら best/ + best.txt）→ latest/ + weights/epoch_NNN/ + フル 1024 画像
     （固定 = train.yaml log.full_slice、ランダム = n_full_random 枚、実 EID テスト = log.eid_slice。output_images/ の生 16bit と preview、TB のパネル。util/monitor.py）
  5. 最終 epoch は save_epoch_freq の倍数でなくても保存する
  6. 実 EID 枝（mode.real_eid_branch、2026-10-13）: RealEIDDataset（data/eid_dataset.py。adv.eid_split_file の train の症例から実 EID512 の小片 → ×scale）を別の DataLoader で
     ペアの loader と並走させ（長さはペア枝の step 数 × adv.batch_eid。足りなければ cycle）、毎 step 両方のバッチを model.optimize に渡す。D の本物側はペア枝の教師（models/regression_model.py）
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
from data.eid_dataset import RealEIDDataset  # noqa: E402
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
    p.add_argument("--init_G_from", default=None, help="新規 run で読む G-only の net_G.pth（D / optimizer / 進捗は新規）")
    p.add_argument("--require_cuda", action="store_true", help="CUDA が無ければ止める（gpu_gen ≠ 0 のとき run_train.py が付ける）")
    a = p.parse_args()
    if a.resume_state and a.init_G_from:
        p.error("--resume_state と --init_G_from は同時に使えません")
    if a.init_G_from and a.epoch_count != 1:
        p.error("--init_G_from は新規 run 専用なので --epoch_count 1 が必要です")

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
    loader_eid, eid_iter = None, None
    if mode["real_eid_branch"]:  # 実 EID 枝（2026-10-13）: ペアの loader と同じ step 数になる長さで別 loader を作り並走させる
        dataset_eid = RealEIDDataset(train, mode, machine, dataset.upsample_cfg, n_steps=len(loader), cache_dir=Path(machine["stage2_checkpoints_dir"]) / ".case_index")
        with open(run_dir / DATASET_INFO_FILE, "a", encoding="utf-8") as f:
            yaml.safe_dump({"real_eid_branch": dataset_eid.info()}, f, allow_unicode=True, sort_keys=False)
        loader_eid = DataLoader(dataset_eid, batch_size=train["adv.batch_eid"], shuffle=False, num_workers=machine["num_threads"], pin_memory=(device.type == "cuda"), drop_last=False)

    def next_eid():
        """実 EID のバッチを 1 つ取る（loader_eid が尽きたら頭から = cycle）。"""
        nonlocal eid_iter
        if eid_iter is None:
            eid_iter = iter(loader_eid)
        try:
            return next(eid_iter)
        except StopIteration:
            eid_iter = iter(loader_eid)
            return next(eid_iter)

    model = RegressionModel(train, mode, machine, device)
    print(f"[train] {mode['arch']} base_ch {mode['base_ch']} n_pool {mode['n_pool']} final_act {mode['final_act']} residual {mode['residual']}: {model.n_params / 1e6:.2f}M params, loss {mode['loss']}, device {device}"
          + (f" | real_eid_branch: D {model.n_params_D / 1e6:.2f}M params, {train['adv.gan_mode']}, lambda_gan {train['adv.lambda_gan']}, lr_D {train['adv.lr_D']}, batch_eid {train['adv.batch_eid']}" if mode["real_eid_branch"] else ""))
    if a.resume_state:
        model.load_weights(Path(a.resume_state).parent)
        model.resume(a.resume_state)
    elif a.init_G_from:
        model.load_G_only(a.init_G_from)
        print(f"[train] pretrained G を読み込み: {a.init_G_from}（D / G optimizer / D optimizer は新規初期化）")
    total_iters = model.total_iters
    val_slices = dataset.val_slices()
    fixed = dataset.fixed_patches(train["log.n_images"])  # TB images/fixed 用（log.full_slice から。位置は patch_seed で固定）
    monitor = TrainMonitor(run_dir, train, mode["loss"], bs, len(dataset), real_eid_branch=mode["real_eid_branch"])
    n_epochs, print_freq, save_latest, save_epoch = train["optim.n_epochs"], train["log.print_freq"], train["log.save_latest_freq"], train["log.save_epoch_freq"]
    best_metric = train["log.best_metric"]
    image_freq = train["log.image_freq"]
    last_epoch, last_saved_epoch = None, None

    if a.init_G_from:
        # 学習前の事前学習 G を必ず残し、その val を best の初期基準にする。
        # pretrained checkpoint に含まれる D は新規初期化、optimizer も新規。epoch 1 から通常どおり学習する。
        model.cur_epoch, model.epoch_done, model.total_iters = 0, True, 0
        val0 = model.evaluate(dataset, val_slices)
        monitor.log_val(val0, 0)
        improved = model.update_best(run_dir, best_metric, val0, 0, 0)
        monitor.log_best(model.best, improved, 0, 0)
        model.save(run_dir, "pretrained")
        monitor.write(f"[init_G] epoch 0 baseline: {best_metric} {val0[best_metric]:.6f} on {val0['n']} slices | source {a.init_G_from} | saved {ckpt_dir(run_dir, 'pretrained')}")

    for epoch in range(a.epoch_count, n_epochs + 1):
        t_epoch = time.time()
        iter_data_time = time.time()
        model.cur_epoch, model.epoch_done = epoch, False
        if mode["real_eid_branch"]:  # 崩壊対策（2026-10-13）: ウォームアップ終了後は lr を落とす（通算 epoch で判定。resume 後も同じ）
            lr_now = model.apply_epoch_lr()
            monitor.write(f"[epoch {epoch}] real_eid_branch: GAN {'有効' if model.gan_active() else f'無効（warmup_epochs {model.warmup_epochs}）'}, lr {lr_now:.6f}")
        for batch in monitor.epoch_bar(loader, epoch, n_epochs):
            t0 = time.time()
            t_data = t0 - iter_data_time
            n_batch = int(batch["input"].shape[0])
            total_iters += n_batch
            model.total_iters = total_iters
            step = total_iters // bs
            losses = model.optimize(batch, next_eid() if loader_eid is not None else None, diag=(step % print_freq == 0))  # diag: 勾配ノルムの内訳（print_freq ごと）
            monitor.accumulate(losses, time.time() - t0, t_data)
            monitor.check_divergence(losses, epoch, total_iters)
            if step % print_freq == 0:
                monitor.step(epoch, total_iters)
            if step % image_freq == 0:
                monitor.log_images(model, fixed, total_iters)  # 128 patch グリッド（current / fixed）を TB へ
            if total_iters % save_latest == 0:
                model.save(run_dir, "latest")
            iter_data_time = time.time()
        eid_iter = None  # 実 EID の loader も epoch ごとに頭から（worker の乱数は DataLoader が epoch ごとに seed し直す）
        model.epoch_done = True
        val = model.evaluate(dataset, val_slices)
        monitor.log_val(val, total_iters)
        monitor.check_collapse(val, epoch)
        improved = model.update_best(run_dir, best_metric, val, epoch, total_iters)  # val が最良なら best/ + best.txt（save_epoch_freq とは独立）
        monitor.log_best(model.best, improved, epoch, total_iters)
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
    best_msg = f", best: {ckpt_dir(run_dir, 'best')} = epoch {model.best['epoch']} ({model.best['metric']} {model.best['value']:.6f})" if model.best else ""
    monitor.write(f"[train] 完了: {run_dir}（latest: {ckpt_dir(run_dir, 'latest')}{best_msg}）")
    monitor.close()


if __name__ == "__main__":
    main()
