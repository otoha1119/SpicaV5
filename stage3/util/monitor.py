"""stage3/util/monitor.py — Stage 3 学習の表示: ターミナル進捗バー（tqdm）+ TensorBoard + loss_log.txt。stage2/util/monitor.py から 2026-10-13 に複製して拡散モデル用に変えた。

global_step は本家と同じ step（optimizer の更新回数）。stage2 の total_iters（画像枚数）とは単位が違う。
  ターミナル : 1 epoch（data.samples_per_epoch 枚）= 1 本の tqdm バー。末尾に loss と ε の画素平均 MSE
  TensorBoard: <run>/tb/
     loss/eps               本家の損失そのもの（ε の二乗誤差の (C,H,W) 和の batch 平均。本家 tb_logger "loss" と同じ量）。print_freq の区間平均
     loss/eps_mse_per_pixel 同じ値を画素数で割ったもの（ε は標準正規なので 1 が「何も分からない」、0 が完全）
     val/eps_mse_per_pixel, val/eps_mse_per_pixel_ema   validation_freq ごと: 固定 val patch（n_val_patches 枚、t は [0, T) に等間隔、雑音固定）での同じ量。学習中の重み / EMA の重み
     time/sec_per_step, time/data_sec_per_step
     images/val_patch       image_freq ごと: 固定 val patch（n_images 枚）を EMA の重みで DDIM サンプリング（log.sample_steps）。各行 [EID-like (cond) | PCD-like (sample) | PCD1024 (teacher)]
     images/full/fixed      full_freq ごと: 固定スライス（log.full_slice）と実 EID（log.eid_slice）をタイル合成でサンプリングした 5 列パネル（stage2 と同じ並び・util/panel.py）
     images/full/EID        同: 実 EID だけの 2 列
     time/full_sample_sec   その 2 枚のサンプリングにかかった秒数（full_freq を決める目安。決定 7）
  ディスク（stage2 と同じ 2 系統。区切りは iter_NNNNNNN）: output_images/iter_NNNNNNN/ の生 16bit、preview_fixed_<slice>/ の表示用
"""

import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

from data.ct_io import denormalize, to_display, write_png
from util.panel import build_panel, eid_labels, full_labels, label_strip
from util.run_paths import LOSS_LOG_FILE, TB_DIR, iter_dirname, iter_images_dir, preview_dir

to_u8 = lambda x01: (np.clip(x01, 0.0, 1.0) * 255.0).round().astype(np.uint8)  # noqa: E731


class TrainMonitor:
    PATCH_LABELS = ("EID-like (cond)", "PCD-like (sample)", "PCD1024 (teacher)")

    def __init__(self, run_dir, train, batch_size, dataset_size, n_pixels):
        self.run_dir = Path(run_dir)
        self.hu = (train["data.hu_offset"], train["data.hu_min"], train["data.hu_max"])
        self.disp_min, self.disp_max = train["log.display_hu_min"], train["log.display_hu_max"]
        self.bits = train["log.preview_bits"]
        self.tb_size = train["log.tb_full_size"]
        self.batch_size = batch_size
        self.dataset_size = dataset_size
        self.n_pixels = n_pixels  # C·H·W（本家の損失を画素平均に直す）
        self.tb = SummaryWriter(str(self.run_dir / TB_DIR))
        self.log = open(self.run_dir / LOSS_LOG_FILE, "a", encoding="utf-8")
        self.log.write(f"================ Training ({time.strftime('%Y-%m-%d %H:%M:%S')}) ================\n")
        self.bar = None
        self._reset_window()

    def _reset_window(self):
        self.win = {"loss": 0.0, "t_comp": 0.0, "t_data": 0.0, "n": 0}

    def epoch_bar(self, loader, epoch, step, n_iters):
        self.bar = tqdm(loader, total=len(loader), unit="img", unit_scale=self.batch_size, dynamic_ncols=True, mininterval=0.5,
                        desc=f"epoch {epoch} (step {step}/{n_iters})", leave=True)
        return self.bar

    def write(self, msg):
        tqdm.write(msg)
        self.log.write(msg + "\n")
        self.log.flush()

    def accumulate(self, losses, t_comp, t_data):
        w = self.win
        w["loss"] += losses["loss"]; w["t_comp"] += t_comp; w["t_data"] += t_data; w["n"] += 1

    def step(self, epoch, step):
        w = self.win
        if w["n"] == 0:
            return
        n = w["n"]
        loss = w["loss"] / n
        if not np.isfinite(loss):
            self.write(f"[warn] step {step}: loss が有限でない（{loss}）。発散している可能性")
        self.tb.add_scalar("loss/eps", loss, step)
        self.tb.add_scalar("loss/eps_mse_per_pixel", loss / self.n_pixels, step)
        self.tb.add_scalar("time/sec_per_step", w["t_comp"] / n, step)
        self.tb.add_scalar("time/data_sec_per_step", w["t_data"] / n, step)
        if self.bar is not None:
            self.bar.set_postfix_str(f"step {step} | loss {loss:.2f} | eps mse/px {loss / self.n_pixels:.5f} | {w['t_comp'] / n:.2f} s/step | data {w['t_data'] / n * 1000:.0f} ms", refresh=False)
        self.log.write(f"(epoch: {epoch}, step: {step}, time: {w['t_comp'] / n:.3f}, data: {w['t_data'] / n:.3f}) loss: {loss:.4f} eps_mse_per_pixel: {loss / self.n_pixels:.6f}\n")
        self.log.flush()
        self._reset_window()

    def log_val(self, val, step):
        self.tb.add_scalar("val/eps_mse_per_pixel", val["model"] / self.n_pixels, step)
        if "ema" in val:
            self.tb.add_scalar("val/eps_mse_per_pixel_ema", val["ema"] / self.n_pixels, step)
        self.log.write(f"[val] step {step}: eps_mse_per_pixel {val['model'] / self.n_pixels:.6f}" + (f" (ema {val['ema'] / self.n_pixels:.6f})" if "ema" in val else "") + "\n")
        self.log.flush()

    # ------------------------------------------------------------------ 表示の部品（stage2 と同じ）
    def _stored(self, t):
        return denormalize(t[0, 0].detach().cpu().numpy(), *self.hu)

    def _lin01(self, stored):
        g = np.clip((stored.astype(np.float32) - self.hu[0] - self.disp_min) / float(self.disp_max - self.disp_min), 0.0, 1.0)
        return np.repeat(g[:, :, None], 3, axis=2)

    def _gray(self, stored):
        return to_display(stored, self.hu[0], self.disp_min, self.disp_max, self.bits)

    def _tb(self, col01):
        if col01.shape[0] == self.tb_size and col01.shape[1] == self.tb_size:
            return col01
        return cv2.resize(col01, (self.tb_size, self.tb_size), interpolation=cv2.INTER_AREA)

    def _rgb(self, panel01):
        arr = (np.clip(panel01, 0.0, 1.0) * (65535.0 if self.bits == 16 else 255.0)).round().astype(np.uint16 if self.bits == 16 else np.uint8)
        return np.ascontiguousarray(arr[:, :, ::-1])

    def _patch_grid(self, x, y, t):
        """(n,1,p,p) ×3 → 各行 [cond | sample | teacher] のグリッド (H,W,3) uint8。上に列ラベル（stage2 と同じ見た目）。"""
        rows = []
        for i in range(x.shape[0]):
            for tsr in (x, y, t):
                g = self._lin01(self._stored(tsr[i : i + 1]))
                rows.append(torch.from_numpy(g).permute(2, 0, 1))
        pad = 2
        grid = make_grid(torch.stack(rows), nrow=3, padding=pad, pad_value=0.5).permute(1, 2, 0).numpy()
        p = x.shape[-1]
        scale = 0.45
        while scale > 0.25 and max(cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] for s in self.PATCH_LABELS) > p - 4:
            scale -= 0.05
        blank = np.ones((22, pad, 3), np.float32)
        header = [blank]
        for s in self.PATCH_LABELS:
            header += [label_strip(p, s, height=22, scale=scale), blank]
        return to_u8(np.concatenate([np.concatenate(header, axis=1), grid], axis=0))

    # ------------------------------------------------------------------ サンプル
    def log_val_patches(self, diffusion, fixed, s, step, model):
        """image_freq ごと: 固定 val patch を EMA の重みでサンプリングして TB images/val_patch。"""
        t0 = time.time()
        seeds = [s["seed"] + i for i in range(fixed["input"].shape[0])]
        y = diffusion.sample_patches(fixed["input"], s["steps"], s["eta"], s["skip_type"], seeds, model).cpu()
        self.tb.add_image("images/val_patch", self._patch_grid(fixed["input"], y, fixed["target"]), step, dataformats="HWC")
        self.tb.add_scalar("time/val_patch_sample_sec", time.time() - t0, step)

    def save_full_images(self, diffusion, dataset, s, step, model, seed_of):
        """full_freq ごと: 固定スライス（入力 / 教師あり）と実 EID テストを 1024 のままタイル合成でサンプリングし、
          output_images/iter_NNNNNNN/ に生 uint16、preview_fixed_<slice>/ に表示用、TB に images/full/fixed（5 列）と images/full/EID（2 列）。"""
        t0 = time.time()
        out_dir = iter_images_dir(self.run_dir, step)
        out_dir.mkdir(parents=True, exist_ok=True)
        it = iter_dirname(step)
        model_tag = f"step {int(step)}"
        scale, interp = dataset.upsample_cfg
        in_path, pcd_path = dataset.fixed_pair
        stem, eid_stem = Path(pcd_path).stem, Path(dataset.eid_path).stem
        sample = lambda cond, name: diffusion.sample_tiled(cond, s["steps"], s["eta"], s["skip_type"], seed_of(name), s["stride"], s["tile_batch"], model).cpu()  # noqa: E731
        x_e = dataset.load_eid_full()
        e_in, e_out = self._stored(x_e), self._stored(sample(x_e, Path(dataset.eid_path).name))
        x, t = dataset.load_full(in_path), dataset.load_full(pcd_path)
        a, b, y = self._stored(x), self._stored(t), self._stored(sample(x, Path(pcd_path).name))
        elapsed = time.time() - t0
        write_png(out_dir / f"{eid_stem}_eid1024.png", e_in)
        write_png(out_dir / f"{eid_stem}_pcdlike.png", e_out)
        write_png(out_dir / f"{stem}_eidlike.png", a)
        write_png(out_dir / f"{stem}_pcd1024.png", b)
        write_png(out_dir / f"{stem}_pcdlike.png", y)
        cols = [self._lin01(a), self._lin01(y), self._lin01(b), self._lin01(e_out), self._lin01(e_in)]
        labels = full_labels(model_tag, None, True, (eid_stem, scale, interp))
        panel01 = build_panel(cols, labels)
        d = preview_dir(self.run_dir, f"fixed_{stem}")
        if not (d / f"00_eidlike1024_{stem}.png").is_file():
            write_png(d / f"00_eidlike1024_{stem}.png", self._gray(a))
            write_png(d / f"01_pcd1024_{stem}.png", self._gray(b))
            write_png(d / f"02_eid1024_{eid_stem}.png", self._gray(e_in))
        write_png(d / f"{it}_pcdlike.png", self._gray(y))
        write_png(d / f"{it}_eid_pcdlike.png", self._gray(e_out))
        write_png(d / f"{it}_panel.png", self._rgb(panel01))
        self.tb.add_image("images/full/fixed", to_u8(build_panel([self._tb(c) for c in cols], labels)), step, dataformats="HWC")
        eid_panel = build_panel([self._tb(self._lin01(e_in)), self._tb(self._lin01(e_out))], eid_labels(model_tag, eid_stem, scale, interp))
        self.tb.add_image("images/full/EID", to_u8(eid_panel), step, dataformats="HWC")
        self.tb.add_scalar("time/full_sample_sec", elapsed, step)
        self.write(f"[full] step {step}: 固定スライス + 実 EID を {s['steps']} step × タイル {s['n_tiles']} 枚（r {s['stride']}）でサンプリング {elapsed:.0f}s -> {out_dir}")

    def end_epoch(self, epoch, step, lr, elapsed):
        self.tb.add_scalar("train/lr", lr, step)
        self.tb.flush()
        self.write(f"[epoch {epoch}] {elapsed:.0f}s, step {step}, lr {lr:.2e}")

    def close(self):
        self.tb.close()
        self.log.close()
