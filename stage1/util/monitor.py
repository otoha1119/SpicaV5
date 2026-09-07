"""util/monitor.py — 学習表示: ターミナル進捗バー（tqdm）+ TensorBoard + loss_log.txt

設計: docs/plans/20260907_training-monitor-plan.md
junyanz の util/visualizer.py（wandb / HTML）は使わない。train.py はこのクラスだけを呼ぶ。

【出すもの】
  ターミナル : 1 epoch = 1 本の tqdm バー（単位 = 画像枚数。1 step で batch_size 枚進む）。末尾に損失と d_in（HU）。
               epoch 末に 1 行の要約（tqdm.write。バーを崩さない）
  TensorBoard: <run>/tb/  scalar は print_freq step ごと、画像は image_freq step ごと。global_step = total_iters（画像枚数。resume で復元されるので曲線がつながる）
     loss/D, loss/G_GAN, loss/G_fid, loss/G_total（= G_GAN + λ·G_fid）
     diag/D_real, diag/D_fake（E[D(x)], E[D(G(z))]。最適で 1 − p_G/p_x。D_fake の急落 = KL 発散の兆候）
     diag/d_in_HU（mean|G(z) − z| を HU 換算。恒等退行 ≈ 0 / 暴走の監視）
     time/sec_per_step, time/data_sec_per_step（データ律速の検出）
     train/lr（epoch ごと）
     images/current（いま学習中のバッチ先頭 n_images 枚）, images/fixed（学習開始時に固定した同じ patch）
        各行 = [ PCD z | EID-like G(z) | 差分 G(z)−z | 実 EID x ]。128 patch のまま（元サイズに戻さない）
  ディスク   : <run>/output_images/samples/<total_iters>_{current,fixed}.png — TB と同じ 8bit グリッド（表示用。16bit ではない）
               <run>/output_images/epoch_NNN/<slice>_{pcd,eidlike,R}.png — checkpoint 保存時にフル 512 のスライス n_full_images 枚を
               PCD / EID-like / 残差 R = G(z) − z で保存（uint16。pcd/eidlike は HU + hu_offset、R は 32768 + ΔHU）。TB にも images/full/<slice>
               （画像パネルは stored = HU + hu_offset、差分パネルは 32768 + ΔHU。TB は 8bit しか描けないため厳密値はこちら）
  loss_log.txt: print_freq ごとの損失（テキスト。grep 用の保険）

【表示の線形範囲】 TB の 8bit 表示は HU を [display_hu_min, display_hu_max] で線形に 0..255 へ（窓は掛けない。既定案 −1400..1600 HU = stored 0..3000）。
   差分パネルは ±diff_range_hu を 0..255 に（0 HU が中間グレー）。
"""

import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

from data.ct_dataset import denormalize, residual_stored, write_png
from util.run_paths import epoch_images_dir, samples_dir

LOSS_KEYS_BAR = ("D", "G_GAN", "G_fid")  # バー末尾に出す損失
DIAG_KEYS = ("D_real", "D_fake")           # TB で diag/ に分類する損失名


class TrainMonitor:
    def __init__(self, opt, dataset_size, fixed_batch):
        """
        opt          : junyanz の opt（print_freq / image_freq / n_images / display_hu_* / diff_range_hu / hu_* / lambda_fid / batch_size を使う）
        dataset_size : 1 epoch の画像枚数（バーの total）
        fixed_batch  : {"A": (n,1,p,p), "B": (n,1,p,p)} 監視用の固定サンプル（CTDataset.fixed_batch）。None なら fixed は出さない
        """
        self.opt = opt
        self.bs = opt.batch_size
        self.dataset_size = dataset_size
        self.run_dir = Path(opt.checkpoints_dir) / opt.name
        self.tb = SummaryWriter(log_dir=str(self.run_dir / "tb"))
        self.samples_dir = samples_dir(self.run_dir)  # <run>/output_images/samples/
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.run_dir / "loss_log.txt"
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"================ Training Loss ({time.strftime('%Y-%m-%d %H:%M:%S')}) ================\n")
        self.hu_per_unit = (opt.hu_max - opt.hu_min) / 2.0  # 正規化 [-1,1] の 1.0 が何 HU か
        self.fixed = fixed_batch
        self.bar = None
        self._sum = {}
        self._n = 0

    # ------------------------------------------------------------------ ターミナル
    def epoch_bar(self, dataset, epoch, total_epochs):
        """dataset を包んで yield する。バーは画像枚数で進む（1 step = batch_size 枚）。"""
        # mininterval=0.1 / miniters=1: 1 step ごとに描画する（0.5 だと GPU では表示が数 step 飛ぶ）。10 Hz なら端末 I/O は問題にならない
        self.bar = tqdm(total=self.dataset_size, desc=f"Epoch {epoch}/{total_epochs}", unit="img", leave=False, dynamic_ncols=True, mininterval=0.1, miniters=1)
        self._sum, self._n = {}, 0
        try:
            for data in dataset:
                yield data
                self.bar.update(int(data["A"].shape[0]))  # F-20: 実枚数
        finally:
            self.bar.close()
            self.bar = None

    def write(self, msg):
        """バーを崩さずに 1 行出す（print の代わり）。"""
        tqdm.write(msg)

    # ------------------------------------------------------------------ 損失
    def accumulate(self, model, n_batch):
        """毎 step 呼ぶ（F-22）: epoch 要約用に損失をサンプル数で重み付けして積む。"""
        for k, v in model.get_current_losses().items():
            self._sum[k] = self._sum.get(k, 0.0) + v * n_batch
        self._n += n_batch

    def step(self, epoch, total_iters, model, t_comp, t_data):
        """print_freq step ごとに呼ぶ: バー末尾・TB scalar・loss_log.txt を更新。"""
        losses = model.get_current_losses()
        d_in_hu = float((model.fake_B.detach() - model.real_A).abs().mean()) * self.hu_per_unit
        if self.bar is not None:
            postfix = {k: f"{losses[k]:.3f}" for k in LOSS_KEYS_BAR if k in losses}
            postfix["d_in"] = f"{d_in_hu:.0f}HU"
            self.bar.set_postfix(postfix, refresh=False)
        for k, v in losses.items():
            self.tb.add_scalar(("diag/" if k in DIAG_KEYS else "loss/") + k, v, total_iters)
        if "G_GAN" in losses and "G_fid" in losses:
            self.tb.add_scalar("loss/G_total", losses["G_GAN"] + self.opt.lambda_fid * losses["G_fid"], total_iters)
        self.tb.add_scalar("diag/d_in_HU", d_in_hu, total_iters)
        self.tb.add_scalar("time/sec_per_step", t_comp, total_iters)
        self.tb.add_scalar("time/data_sec_per_step", t_data, total_iters)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"(epoch: {epoch}, samples: {total_iters}, step: {total_iters // self.bs}, sec/step: {t_comp:.3f}, data: {t_data:.3f}) "
                    + " ".join(f"{k}: {v:.4f}" for k, v in losses.items()) + f" d_in_HU: {d_in_hu:.1f}\n")

    def end_epoch(self, epoch, total_epochs, total_iters, lr, elapsed):
        means = {k: v / max(1, self._n) for k, v in self._sum.items()}  # 全 step のサンプル重み付き平均（F-22）
        line = f"epoch {epoch}/{total_epochs} | " + " ".join(f"{k} {v:.4f}" for k, v in means.items()) + f" | lr {lr:.2e} | {elapsed:.0f}s"
        self.write(line)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        self.tb.add_scalar("train/lr", lr, total_iters)
        self.tb.add_scalar("train/epoch", epoch, total_iters)

    # ------------------------------------------------------------------ 画像
    def log_images(self, model, total_iters):
        """image_freq step ごとに呼ぶ: images/current と images/fixed を TB へ、同じ 8bit グリッドを output_images/samples/ へ。"""
        n = self.opt.n_images
        sets = [("current", model.real_A[:n], model.fake_B[:n].detach(), model.real_B[:n])]
        if self.fixed is not None:
            net = model.netG
            was_training = net.training
            net.eval()  # 監視用の順伝播は BN の running 統計で（学習の統計を汚さない）
            with torch.no_grad():
                a = self.fixed["A"].to(model.device)
                b = self.fixed["B"].to(model.device)
                g = net(a)
            net.train(was_training)
            sets.append(("fixed", a, g, b))
        for tag, a, g, b in sets:
            grid8 = self._grid(a, g, b)
            self.tb.add_image(f"images/{tag}", grid8, total_iters, dataformats="HW")
            write_png(self.samples_dir / f"{total_iters:09d}_{tag}.png", grid8)

    def _grid(self, a, g, b):
        """各行 [z | G(z) | G(z)−z | x] のグリッドを 8bit（表示用。HU を display_hu_min..max で線形に 0..255、差分は ±diff_range_hu）で返す。"""
        o = self.opt
        to_hu = lambda t: (t.detach().float().cpu() + 1.0) / 2.0 * (o.hu_max - o.hu_min) + o.hu_min
        A, G, B = to_hu(a), to_hu(g), to_hu(b)
        D = G - A
        lin = lambda hu: ((hu - o.display_hu_min) / float(o.display_hu_max - o.display_hu_min)).clamp(0.0, 1.0)
        dif = lambda d: ((d + o.diff_range_hu) / float(2 * o.diff_range_hu)).clamp(0.0, 1.0)
        n = A.shape[0]
        panels8 = torch.stack([p for i in range(n) for p in (lin(A[i]), lin(G[i]), dif(D[i]), lin(B[i]))])  # (4n,1,p,p)
        return (make_grid(panels8, nrow=4, padding=2, pad_value=1.0)[0] * 255.0).round().clamp(0, 255).to(torch.uint8).numpy()

    # ------------------------------------------------------------------ フル画像（checkpoint と一緒に）
    def save_full_images(self, model, full, epoch, total_iters):
        """checkpoint 保存時に、固定スライス（CTDataset.fixed_full）をフル 512 で G に通し、
        <run>/output_images/epoch_NNN/<slice>_{pcd,eidlike,R}.png を uint16 で保存する（TB にも images/full/<slice> を 8bit で）。
        推論は full-image モード（G は全畳み込み + 3 段 Haar なので H, W が 8 の倍数なら一発で通る）。
        uint16 化は inference_dir.py と同じ関数（denormalize / residual_stored）で、eidlike = pcd + (R − 32768) が厳密に成り立つ。"""
        if full is None or full["A"].shape[0] == 0:
            return
        o = self.opt
        out_dir = epoch_images_dir(self.run_dir, epoch)
        out_dir.mkdir(parents=True, exist_ok=True)
        net = model.netG
        was_training = net.training
        net.eval()
        with torch.no_grad():
            a = full["A"].to(model.device)
            g = torch.cat([net(a[i : i + 1]) for i in range(a.shape[0])])  # 1 枚ずつ（512 なのでメモリを抑える）
        net.train(was_training)
        to_hu = lambda t: (t.detach().float().cpu() + 1.0) / 2.0 * (o.hu_max - o.hu_min) + o.hu_min
        A, G = to_hu(a), to_hu(g)
        R = G - A
        hu = (o.hu_offset, o.hu_min, o.hu_max)
        for i, path in enumerate(full["paths"]):
            stem = Path(path).stem
            pcd16 = denormalize(a[i, 0].detach().cpu().numpy(), *hu)
            eid16 = denormalize(g[i, 0].detach().cpu().numpy(), *hu)
            write_png(out_dir / f"{stem}_pcd.png", pcd16)
            write_png(out_dir / f"{stem}_eidlike.png", eid16)
            write_png(out_dir / f"{stem}_R.png", residual_stored(pcd16, eid16))  # 0 HU = 32768
            lin = lambda hu: ((hu - o.display_hu_min) / float(o.display_hu_max - o.display_hu_min)).clamp(0.0, 1.0)
            dif = lambda d: ((d + o.diff_range_hu) / float(2 * o.diff_range_hu)).clamp(0.0, 1.0)
            grid = make_grid(torch.stack([lin(A[i]), lin(G[i]), dif(R[i])]), nrow=3, padding=4, pad_value=1.0)[0]
            self.tb.add_image(f"images/full/{stem}", (grid * 255.0).round().clamp(0, 255).to(torch.uint8).numpy(), total_iters, dataformats="HW")
        self.write(f"saved full-size images ({len(full['paths'])} slices) -> {out_dir}")

    def close(self):
        self.tb.flush()
        self.tb.close()
