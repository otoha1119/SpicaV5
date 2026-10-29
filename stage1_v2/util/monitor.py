"""stage1_v2/util/monitor.py — tqdm、TensorBoard、loss log、フル画像 preview。

  ターミナル : 1 epoch = 1 本の tqdm バー（単位 = 画像枚数）。末尾に D_A / G_A / cycle_A / idt_A と d_in（HU）。epoch 末に 1 行の要約
  TensorBoard: <run>/tb/  global_step = total_iters（画像枚数。resume でつながる）
     loss/*            D_A, G_A, cycle_A, idt_A, D_B, G_B, cycle_B, idt_B, G_total（重み込み。本家 loss_names）
     diag/d_in_HU      mean|G_A(x) − x| の HU（入力からどれだけ動いたか）
     time/sec_per_step, time/data_sec_per_step, train/lr, train/epoch
     images/current, images/fixed   各行 [PCD x | EID-like G_A(x) | R | EID y]（128 patch）
     images/cycle_A    現在のバッチ: [x | G_A(x) | G_B(G_A(x))]（forward cycle）、images/cycle_B: [y | G_B(y) | G_A(G_B(y))]（backward cycle）
     images/full/fixed, images/full/random   epoch 末のパネル [EID代表 | EID-like | PCD | R + ゲージ]
  ディスク   : <run>/output_images/epoch_NNN/<slice>_pcd.png, _eidlike.png, _R.png（512 生）
               <run>/output_images/preview_random/epoch_NNN_<slice>.png（パネルだけ）
  表示の線形範囲は HU を [display_hu_min, display_hu_max] で 0..白、R は util/residual_color（白 = 0、青 = −、赤 = +）。
"""

import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

from data.dataset import denormalize, residual_stored, write_png
from util.panel import build_panel, full_labels
from util.residual_color import colorbar_rgb01, residual_rgb01, rgb_to_bgr
from util.run_paths import LOSS_LOG_FILE, TB_DIR, epoch_images_dir, preview_dir

LOSS_KEYS_BAR = ("D_A", "G_A", "cycle_A", "idt_A")
GRID_PAD = 0.5
to_u8 = lambda x01: (np.clip(x01, 0.0, 1.0) * 255.0).round().astype(np.uint8)  # noqa: E731
to_u16 = lambda x01: (np.clip(x01, 0.0, 1.0) * 65535.0).round().astype(np.uint16)  # noqa: E731
NAMES = {"a": "pcd", "gen": "eidlike", "b": "eid"}


class TrainMonitor:
    def __init__(self, run_dir, train, dataset_size, fixed_batch, eid_ref):
        self.c = train
        self.bs = train["optim.batch_size"]
        self.dataset_size = dataset_size
        self.run_dir = Path(run_dir)
        self.tb = SummaryWriter(log_dir=str(self.run_dir / TB_DIR))
        self.log_path = self.run_dir / LOSS_LOG_FILE
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"================ Training Loss ({time.strftime('%Y-%m-%d %H:%M:%S')}) ================\n")
        self.fixed = fixed_batch
        self.eid_ref = eid_ref
        self.eid_stem = Path(eid_ref["path"]).stem
        self.hu = (train["data.hu_offset"], train["data.hu_min"], train["data.hu_max"])
        self.bar = None
        self._sum, self._n, self._t_comp, self._t_data = {}, 0, 0.0, 0.0

    # ------------------------------------------------------------------ ターミナル
    def epoch_bar(self, loader, epoch, total_epochs):
        self.bar = tqdm(total=self.dataset_size, desc=f"Epoch {epoch}/{total_epochs}", unit="img", leave=False, dynamic_ncols=True, mininterval=0.5, miniters=1)
        self._sum, self._n, self._t_comp, self._t_data = {}, 0, 0.0, 0.0
        try:
            for data in loader:
                yield data
                self.bar.update(int(data["A"].shape[0]))
        finally:
            self.bar.close()
            self.bar = None

    def write(self, msg):
        tqdm.write(msg)

    # ------------------------------------------------------------------ 損失
    def accumulate(self, losses, n_batch, t_comp, t_data):
        for k, v in losses.items():
            self._sum[k] = self._sum.get(k, 0.0) + v * n_batch
        self._n += n_batch
        self._t_comp, self._t_data = t_comp, t_data

    def step(self, epoch, total_iters, losses):
        if self.bar is not None:
            postfix = {k: f"{losses[k]:.3f}" for k in LOSS_KEYS_BAR if k in losses}
            postfix["d_in"] = f"{losses['d_in_HU']:.0f}HU"
            self.bar.set_postfix(postfix, refresh=False)
        for k, v in losses.items():
            self.tb.add_scalar(("diag/" if k == "d_in_HU" else "loss/") + k, v, total_iters)
        self.tb.add_scalar("time/sec_per_step", self._t_comp, total_iters)
        self.tb.add_scalar("time/data_sec_per_step", self._t_data, total_iters)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"(epoch: {epoch}, samples: {total_iters}, step: {total_iters // self.bs}, sec/step: {self._t_comp:.3f}, data: {self._t_data:.3f}) "
                    + " ".join(f"{k}: {v:.4f}" for k, v in losses.items()) + "\n")

    def end_epoch(self, epoch, total_epochs, total_iters, lr, elapsed):
        means = {k: v / max(1, self._n) for k, v in self._sum.items()}
        line = f"epoch {epoch}/{total_epochs} | " + " ".join(f"{k} {v:.4f}" for k, v in means.items()) + f" | lr {lr:.2e} | {elapsed:.0f}s"
        self.write(line)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        self.tb.add_scalar("train/lr", lr, total_iters)
        self.tb.add_scalar("train/epoch", epoch, total_iters)

    # ------------------------------------------------------------------ 表示用の変換
    def _to_hu(self, t):
        return (t.detach().float().cpu() + 1.0) / 2.0 * (self.hu[2] - self.hu[1]) + self.hu[1]

    def _lin(self, hu):
        """(1,H,W) HU → (3,H,W) [0,1] グレー。"""
        return ((hu - self.c["log.display_hu_min"]) / float(self.c["log.display_hu_max"] - self.c["log.display_hu_min"])).clamp(0.0, 1.0).expand(3, -1, -1)

    def _col(self, d):
        """(1,H,W) ΔHU → (3,H,W) [0,1] カラー。"""
        return torch.from_numpy(residual_rgb01(d[0].numpy(), self.c["log.diff_range_hu"])).permute(2, 0, 1)

    # ------------------------------------------------------------------ patch グリッド
    def log_images(self, model, fixed_views, total_iters, current):
        """image_freq step ごと。current / fixed_views: cycle_views の戻り値（A, fake_B, rec_A, B, fake_A, rec_B。eval で通したもの）。"""
        n = self.c["log.n_images"]
        for tag, v in (("current", current), ("fixed", fixed_views)):
            if v is None:
                continue
            self.tb.add_image(f"images/{tag}", self._grid(v["A"][:n], v["fake_B"][:n], v["B"][:n]), total_iters, dataformats="HWC")
        if current is not None:
            self.tb.add_image("images/cycle_A", self._cycle_grid([current["A"][:n], current["fake_B"][:n], current["rec_A"][:n]]), total_iters, dataformats="HWC")
            self.tb.add_image("images/cycle_B", self._cycle_grid([current["B"][:n], current["fake_A"][:n], current["rec_B"][:n]]), total_iters, dataformats="HWC")

    def _grid(self, a, g, b):
        """各行 [x | G_A(x) | G_A(x) − x（カラー）| y]。最下段に凡例。"""
        A, G, B = self._to_hu(a), self._to_hu(g), self._to_hu(b)
        D = G - A
        n = A.shape[0]
        panels = torch.stack([p for i in range(n) for p in (self._lin(A[i]), self._lin(G[i]), self._col(D[i]), self._lin(B[i]))])
        grid = make_grid(panels, nrow=4, padding=2, pad_value=GRID_PAD).permute(1, 2, 0).numpy()
        return to_u8(np.concatenate([grid, colorbar_rgb01(self.c["log.diff_range_hu"], grid.shape[1])], axis=0))

    def _cycle_grid(self, cols):
        """各行 [cols[0][i] | cols[1][i] | cols[2][i]]（グレー）。"""
        hus = [self._to_hu(c) for c in cols]
        n = hus[0].shape[0]
        panels = torch.stack([self._lin(h[i]) for i in range(n) for h in hus])
        return to_u8(make_grid(panels, nrow=len(cols), padding=2, pad_value=GRID_PAD).permute(1, 2, 0).numpy())

    # ------------------------------------------------------------------ フル画像（checkpoint と一緒に）
    def save_full_images(self, model, full, epoch, total_iters):
        """固定 + ランダムの PCD512 を G_A に通し、EID-like512 を保存する。"""
        if full is None or full["A"].shape[0] == 0:
            return
        out_dir = epoch_images_dir(self.run_dir, epoch)
        out_dir.mkdir(parents=True, exist_ok=True)
        a = full["A"]
        g = torch.cat([model.predict(a[i : i + 1]).cpu() for i in range(a.shape[0])])
        A, G = self._to_hu(a), self._to_hu(g)
        R = G - A
        E = self._lin(self._to_hu(self.eid_ref["B"])[0])
        bits = to_u16 if self.c["log.preview_bits"] == 16 else to_u8
        gray_png = lambda p3: bits(p3[0].numpy())  # noqa: E731
        rgb_png = lambda hwc: rgb_to_bgr(bits(hwc))  # noqa: E731
        hwc = lambda p3: p3.permute(1, 2, 0).numpy()  # noqa: E731
        ep = out_dir.name
        tbs = self.c["log.tb_full_size"]
        for i, (path, kind) in enumerate(zip(full["paths"], full["kinds"])):
            stem = Path(path).stem
            pcd16 = denormalize(a[i, 0].numpy(), *self.hu)
            eidlike16 = denormalize(g[i, 0].numpy(), *self.hu)
            write_png(out_dir / f"{stem}_{NAMES['a']}.png", pcd16)
            write_png(out_dir / f"{stem}_{NAMES['gen']}.png", eidlike16)
            write_png(out_dir / f"{stem}_R.png", residual_stored(pcd16, eidlike16))
            cols = [hwc(E), hwc(self._lin(G[i])), hwc(self._lin(A[i])), hwc(self._col(R[i]))]
            labels = full_labels(f"epoch {int(epoch)}")
            panel01 = build_panel(cols, labels, self.c["log.diff_range_hu"])
            tag = "fixed" if kind == "fixed" else ("random" if full["kinds"].count("random") == 1 else f"random{i}")
            small = cv2.resize(panel01, (int(round(panel01.shape[1] * tbs / cols[0].shape[0])), int(round(panel01.shape[0] * tbs / cols[0].shape[0]))), interpolation=cv2.INTER_AREA)
            self.tb.add_image(f"images/full/{tag}", to_u8(small), total_iters, dataformats="HWC")
            if kind == "fixed":
                d = preview_dir(self.run_dir, f"fixed_{stem}")
                if not (d / f"00_{NAMES['a']}_{stem}.png").is_file():
                    write_png(d / f"00_{NAMES['a']}_{stem}.png", gray_png(self._lin(A[i])))
                if not (d / f"01_{NAMES['b']}_{self.eid_stem}.png").is_file():
                    write_png(d / f"01_{NAMES['b']}_{self.eid_stem}.png", gray_png(E))
                write_png(d / f"{ep}_{NAMES['gen']}.png", gray_png(self._lin(G[i])))
                write_png(d / f"{ep}_R_color.png", rgb_png(hwc(self._col(R[i]))))
                write_png(d / f"{ep}_panel.png", rgb_png(panel01))
            else:
                write_png(preview_dir(self.run_dir, "random") / f"{ep}_{stem}.png", rgb_png(panel01))
        self.write(f"saved full-size images ({len(full['paths'])} slices, {g.shape[-1]}) -> {out_dir}")

    def close(self):
        self.tb.flush()
        self.tb.close()
