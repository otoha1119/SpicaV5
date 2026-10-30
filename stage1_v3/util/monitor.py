"""stage1_v3/util/monitor.py — tqdm、TensorBoard、loss log、フル画像 preview（stage1_v2 を土台に、P と恒等経路と標準偏差画像を足したもの）。

  ターミナル : 1 epoch = 1 本の tqdm バー。末尾に D / G_GAN / G_fid / G_id と R_u / R_y（HU）。epoch 末に 1 行の要約
  TensorBoard: <run>/tb/  global_step = total_iters（画像枚数。resume でつながる）
     loss/*          D, D_real, D_fake, G_GAN, G_fid, G_id, G_total
     diag/*          R_u_rms_HU, R_y_rms_HU, R_ratio, P_rms_HU, d_in_HU, lambda_id, P_k_sigma_px, P_noise_amp_HU（learned）, D_sep, D_acc（D の分離。0 / 0.5 = 見分けていない）
     draws/*         epoch 末、固定スライスを n_draws 回引き直した出力の統計（R-S21 とレビューの検出指標）:
                       std_out_body_HU   出力の引き直し間標準偏差の体内平均（0 に近い = 決定的）
                       std_ratio         std(出力) / std(P(x))（< 1 = R が P のノイズを消している = denoise。≈ 1 で透過）
                       corr_dR_du        corr(R(u_i) − R(u_j), u_i − u_j)（負が大きい = R が実現を打ち消す）
     time/*, train/lr_G, train/lr_D, train/epoch
     images/current, images/fixed   各行 [x | P(x) | ŷ = u + R(u) | R(u) | y | y + R(y) | R(y)]（128 patch）
     images/full/fixed, images/full/random   [EID代表 | EID-like | P(x) | PCD | R + ゲージ]。images/full/identity [y | G(y) | R(y)]。images/full/std 標準偏差画像
  ディスク   : <run>/output_images/epoch_NNN/ と preview_*/（util/run_paths.py）
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
from util.panel import build_panel
from util.residual_color import colorbar_rgb01, residual_rgb01, rgb_to_bgr
from util.run_paths import LOSS_LOG_FILE, TB_DIR, epoch_images_dir, preview_dir

LOSS_KEYS_BAR = ("D", "G_GAN", "G_fid", "G_id")
DIAG_KEYS = ("R_u_rms_HU", "R_y_rms_HU", "R_ratio", "P_rms_HU", "d_in_HU", "lambda_id", "P_k_sigma_px", "P_noise_amp_HU", "D_sep", "D_acc")
GRID_PAD = 0.5
STD_DISPLAY_MAX_HU = 100.0  # 標準偏差画像の表示レンジ（0 = 黒 … 100 HU = 白）
to_u8 = lambda x01: (np.clip(x01, 0.0, 1.0) * 255.0).round().astype(np.uint8)  # noqa: E731
to_u16 = lambda x01: (np.clip(x01, 0.0, 1.0) * 65535.0).round().astype(np.uint16)  # noqa: E731


def full_labels(tag):
    return ["EID512  y (ref)", f"EID-like512  u + R(u)   {tag}", "P(x)  u", "PCD512  x", "R(u) = out - u   [HU]"]


def identity_labels(tag):
    return ["EID512  y", f"y + R(y)   {tag}", "R(y)   [HU]"]


class TrainMonitor:
    def __init__(self, run_dir, train, dataset_size, fixed_batch, eid_ref, body_threshold_hu):
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
        self.body_thr = body_threshold_hu
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
            postfix["R_u"] = f"{losses['R_u_rms_HU']:.0f}HU"
            postfix["R_y"] = f"{losses['R_y_rms_HU']:.0f}HU"
            self.bar.set_postfix(postfix, refresh=False)
        for k, v in losses.items():
            self.tb.add_scalar(("diag/" if k in DIAG_KEYS else "loss/") + k, v, total_iters)
        self.tb.add_scalar("time/sec_per_step", self._t_comp, total_iters)
        self.tb.add_scalar("time/data_sec_per_step", self._t_data, total_iters)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"(epoch: {epoch}, samples: {total_iters}, step: {total_iters // self.bs}, sec/step: {self._t_comp:.3f}, data: {self._t_data:.3f}) "
                    + " ".join(f"{k}: {v:.4f}" for k, v in losses.items()) + "\n")

    def end_epoch(self, epoch, total_epochs, total_iters, lr_G, lr_D, elapsed):
        means = {k: v / max(1, self._n) for k, v in self._sum.items()}
        line = f"epoch {epoch}/{total_epochs} | " + " ".join(f"{k} {v:.4f}" for k, v in means.items()) + f" | lr_G {lr_G:.2e} lr_D {lr_D:.2e} | {elapsed:.0f}s"
        self.write(line)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        self.tb.add_scalar("train/lr_G", lr_G, total_iters)
        self.tb.add_scalar("train/lr_D", lr_D, total_iters)
        self.tb.add_scalar("train/epoch", epoch, total_iters)

    # ------------------------------------------------------------------ 表示用の変換
    def _to_hu(self, t):
        return (t.detach().float().cpu() + 1.0) / 2.0 * (self.hu[2] - self.hu[1]) + self.hu[1]

    def _lin(self, hu):
        return ((hu - self.c["log.display_hu_min"]) / float(self.c["log.display_hu_max"] - self.c["log.display_hu_min"])).clamp(0.0, 1.0).expand(3, -1, -1)

    def _col(self, d):
        return torch.from_numpy(residual_rgb01(d[0].numpy(), self.c["log.diff_range_hu"])).permute(2, 0, 1)

    @staticmethod
    def _std_gray(std_hu):
        """(1,H,W) 標準偏差 [HU] → (3,H,W) [0,1]。"""
        return (std_hu / STD_DISPLAY_MAX_HU).clamp(0.0, 1.0).expand(3, -1, -1)

    # ------------------------------------------------------------------ patch グリッド
    def log_images(self, model, fixed_views, total_iters, current):
        n = self.c["log.n_images"]
        for tag, v in (("current", current), ("fixed", fixed_views)):
            if v is None:
                continue
            self.tb.add_image(f"images/{tag}", self._grid(v, n), total_iters, dataformats="HWC")

    def _grid(self, v, n):
        """各行 [x | u | ŷ | R(u) | y | y+R(y) | R(y)]。最下段に凡例。"""
        A, U, G, B, I = (self._to_hu(v[k][:n]) for k in ("A", "U", "fake_B", "B", "idt_B"))
        Ru, Ry = G - U, I - B
        m = A.shape[0]
        panels = torch.stack([p for i in range(m) for p in (self._lin(A[i]), self._lin(U[i]), self._lin(G[i]), self._col(Ru[i]), self._lin(B[i]), self._lin(I[i]), self._col(Ry[i]))])
        grid = make_grid(panels, nrow=7, padding=2, pad_value=GRID_PAD).permute(1, 2, 0).numpy()
        return to_u8(np.concatenate([grid, colorbar_rgb01(self.c["log.diff_range_hu"], grid.shape[1])], axis=0))

    # ------------------------------------------------------------------ フル画像（checkpoint と一緒に）
    def save_full_images(self, model, full, epoch, total_iters):
        """固定 + ランダムの PCD512（x と P(x)）を G に通して保存。固定スライスでは n_draws の標準偏差画像、代表 EID では恒等経路も保存。"""
        if full is None or full["A"].shape[0] == 0:
            return
        out_dir = epoch_images_dir(self.run_dir, epoch)
        out_dir.mkdir(parents=True, exist_ok=True)
        a, u = full["A"], full["U"]
        g = torch.cat([model.predict(u[i : i + 1]).cpu() for i in range(u.shape[0])])
        A, U, G = self._to_hu(a), self._to_hu(u), self._to_hu(g)
        R = G - U
        y = self.eid_ref["B"]
        gy = model.predict(y).cpu()
        Y, GY = self._to_hu(y)[0], self._to_hu(gy)[0]
        RY = GY - Y
        bits = to_u16 if self.c["log.preview_bits"] == 16 else to_u8
        gray_png = lambda p3: bits(p3[0].numpy())  # noqa: E731
        rgb_png = lambda hwc: rgb_to_bgr(bits(hwc))  # noqa: E731
        hwc = lambda p3: p3.permute(1, 2, 0).numpy()  # noqa: E731
        ep = out_dir.name
        tbs = self.c["log.tb_full_size"]
        E = self._lin(Y)

        def tb_panel(tag, panel01):
            small = cv2.resize(panel01, (int(round(panel01.shape[1] * tbs / A.shape[-1])), int(round(panel01.shape[0] * tbs / A.shape[-1]))), interpolation=cv2.INTER_AREA)
            self.tb.add_image(f"images/full/{tag}", to_u8(small), total_iters, dataformats="HWC")

        for i, (path, kind) in enumerate(zip(full["paths"], full["kinds"])):
            stem = Path(path).stem
            pcd16, u16, eidlike16 = denormalize(a[i, 0].numpy(), *self.hu), denormalize(u[i, 0].numpy(), *self.hu), denormalize(g[i, 0].numpy(), *self.hu)
            write_png(out_dir / f"{stem}_pcd.png", pcd16)
            write_png(out_dir / f"{stem}_u.png", u16)
            write_png(out_dir / f"{stem}_eidlike.png", eidlike16)
            write_png(out_dir / f"{stem}_R.png", residual_stored(u16, eidlike16))
            cols = [hwc(E), hwc(self._lin(G[i])), hwc(self._lin(U[i])), hwc(self._lin(A[i])), hwc(self._col(R[i]))]
            panel01 = build_panel(cols, full_labels(f"epoch {int(epoch)}"), self.c["log.diff_range_hu"])
            tag = "fixed" if kind == "fixed" else ("random" if full["kinds"].count("random") == 1 else f"random{i}")
            tb_panel(tag, panel01)
            if kind == "fixed":
                d = preview_dir(self.run_dir, f"fixed_{stem}")
                if not (d / f"00_pcd_{stem}.png").is_file():
                    write_png(d / f"00_pcd_{stem}.png", gray_png(self._lin(A[i])))
                if not (d / f"01_eid_{self.eid_stem}.png").is_file():
                    write_png(d / f"01_eid_{self.eid_stem}.png", gray_png(E))
                write_png(d / f"{ep}_eidlike.png", gray_png(self._lin(G[i])))
                write_png(d / f"{ep}_u.png", gray_png(self._lin(U[i])))
                write_png(d / f"{ep}_R_color.png", rgb_png(hwc(self._col(R[i]))))
                write_png(d / f"{ep}_panel.png", rgb_png(panel01))
            else:
                write_png(preview_dir(self.run_dir, "random") / f"{ep}_{stem}.png", rgb_png(panel01))

        # 恒等経路（代表 EID）
        eid16, gy16 = denormalize(y[0, 0].numpy(), *self.hu), denormalize(gy[0, 0].numpy(), *self.hu)
        write_png(out_dir / f"{self.eid_stem}_idt.png", gy16)
        write_png(out_dir / f"{self.eid_stem}_Ry.png", residual_stored(eid16, gy16))
        panel_id = build_panel([hwc(E), hwc(self._lin(GY)), hwc(self._col(RY))], identity_labels(f"epoch {int(epoch)}"), self.c["log.diff_range_hu"])
        tb_panel("identity", panel_id)
        write_png(preview_dir(self.run_dir, f"identity_{self.eid_stem}") / f"{ep}_panel.png", rgb_png(panel_id))
        r_y_rms = float(RY.pow(2).mean().sqrt())
        self.tb.add_scalar("draws/R_y_full_rms_HU", r_y_rms, total_iters)

        # n_draws の標準偏差画像（固定スライス）
        msg = ""
        if full.get("U_draws") is not None:
            ud = full["U_draws"]
            gd = torch.cat([model.predict(ud[i : i + 1]).cpu() for i in range(ud.shape[0])])
            Ud, Gd = self._to_hu(ud), self._to_hu(gd)
            std_out, std_u = Gd.std(dim=0), Ud.std(dim=0)  # (1,H,W)
            body = (A[0] > self.body_thr)
            s_out = float(std_out[body].mean()) if body.any() else float(std_out.mean())
            s_u = float(std_u[body].mean()) if body.any() else float(std_u.mean())
            d_out, d_u = (Gd[1:] - Gd[:-1]), (Ud[1:] - Ud[:-1])  # 隣り合う draw の差
            dR = (d_out - d_u).flatten()
            du = d_u.flatten()
            corr = float(((dR - dR.mean()) * (du - du.mean())).mean() / (dR.std() * du.std() + 1e-6)) if du.std() > 0 else 0.0  # Pearson（平均を引く）
            self.tb.add_scalar("draws/std_out_body_HU", s_out, total_iters)
            self.tb.add_scalar("draws/std_u_body_HU", s_u, total_iters)
            self.tb.add_scalar("draws/std_ratio", s_out / max(s_u, 1e-6), total_iters)
            self.tb.add_scalar("draws/corr_dR_du", corr, total_iters)
            stem0 = Path(full["paths"][0]).stem
            write_png(out_dir / f"{stem0}_std.png", np.clip(np.round(std_out[0].numpy()), 0, 65535).astype(np.uint16))
            std_img = hwc(self._std_gray(std_out))
            self.tb.add_image("images/full/std", to_u8(cv2.resize(std_img, (tbs, tbs), interpolation=cv2.INTER_AREA)), total_iters, dataformats="HWC")
            write_png(preview_dir(self.run_dir, f"fixed_{stem0}") / f"{ep}_std.png", rgb_png(std_img))
            msg = f" | draws {ud.shape[0]}: std_out {s_out:.1f} HU, std_u {s_u:.1f} HU, ratio {s_out / max(s_u, 1e-6):.2f}, corr(dR,du) {corr:+.2f}"
        self.write(f"saved full-size images ({len(full['paths'])} slices, {g.shape[-1]}) -> {out_dir} | R(y) full RMS {r_y_rms:.1f} HU{msg}")

    def close(self):
        self.tb.flush()
        self.tb.close()
