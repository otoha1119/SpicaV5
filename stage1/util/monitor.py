"""stage1/util/monitor.py — 学習の途中経過: TensorBoard の scalar と画像、loss_log.txt、パネル、生の 16bit（設計書 G-3）。

scalar のタグ（監視ツールは名前に loss / val/ / diag/ / sigma を含むものを Mac 側の要約に出す。設計書の train/* と check/* は diag/ の下に置いた）:
  loss/train、loss_bin/0〜5（σ の帯。境 2 / 6 / 20 / 60 / 200 / 2,000 / 40,000 HU）、loss_cue/none・clean・blur_only・noise_only・both
  val/loss、val/loss_bin/0〜5、diag/bf16_val_ratio/bin0〜5（学習が bf16 のとき。同じ入力の bf16 ÷ FP32）
  diag/lr、diag/grad_norm、diag/sec_per_step、diag/imgs_per_sec、diag/data_time_frac、diag/hours
  diag/check_redraw_wide_mean_diff_hu、diag/check_redraw_fine_sd_ratio、diag/check_redraw_fine_corr_src        実 EID（log.eid_slices の 1 つ目）の描き直し
  diag/check_pseudo_wide_mean_diff_hu、diag/check_pseudo_fine_corr_pcd、diag/check_pseudo_fine_sd_ratio_vs_eid  PCD（log.pcd_slices の 1 つ目）からの疑似 EID
    （どれも度合い 3 つの平均。度合いごとは末尾に /b2_c0 など。確認用で、合格線は置かない）

画像（TensorBoard と <run>/preview/<名前>/step_NNNNNNN.png。生の 16bit は <run>/output_images/step_NNNNNNN/）:
  preview/train_patch                 学習中のパッチ [元 | 手がかり | ノイズ入り | x̂0]
  preview/nocue_256                   手がかり無しで描いた patch_size 四方
  preview/eid_redraw/<スライス>       行 = 度合い、列 = [実 EID | 手がかり | 描き直し | 描き直し − 実 EID]
  preview/pcd_pseudo/<スライス>       行 = 度合い、列 = [PCD | 手がかり | 疑似 EID | 実 EID（参照）| 疑似 EID − PCD]
  preview/zoom/…                      上の 2 つの拡大（log.zoom_size 四方を log.zoom_scale 倍、最近傍）。各一覧の 1 つ目だけ（2 つ目以降 = 病院 EID などは拡大の位置を決めていないので出さない）
乱数は毎回同じ種（step が違っても同じ描き始め）。表示は HU −1400〜3000、差の列は白 = 0・青 −・赤 +（±log.diff_range_hu）。
"""

import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from data.ct_io import write_png, x_to_stored
from data.eid_dataset import load_full_x
from diffusion.anchor import wide_mean
from diffusion.cue import gaussian_blur
from diffusion.sampler import redraw
from models.painter import NonFiniteError
from util.panel import build_grid, gray, to_u8, zoom
from util.residual_color import residual_rgb01
from util.run_paths import LOSS_LOG_FILE, TB_DIR, preview_path, step_images_dir
from util.seeds import generator_of


def cue_tag(b, c):
    return f"b{b:g}_c{c:g}"


def _corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).mean() / (a.std() * b.std()).clamp_min(1e-12))


class Monitor:
    def __init__(self, run_dir, train, shared, machine, hu, cues, device):
        self.run_dir, self.t, self.hu, self.device = Path(run_dir), train, hu, device
        self.scale = train["data.hu_scale"]
        self.anchor_px = shared["brightness.sigma_px_512"]
        self.cues = cues  # [(σ_b 画素, σ_c HU), ...]
        self.writer = SummaryWriter(str(self.run_dir / TB_DIR))
        self.log = open(self.run_dir / LOSS_LOG_FILE, "a", encoding="utf-8")
        self.t_window, self.t_data, self.n_window = time.time(), 0.0, 0
        # (種類, 番号, パス, x の (1, 1, H, W))。各種類の 1 つ目が基準（check の scalar・拡大・参照列）
        self.items = [(kind, i, Path(p), load_full_x(p, hu).to(device)) for kind, key in (("eid", "log.eid_slices"), ("pcd", "log.pcd_slices")) for i, p in enumerate(train[key])]

    # --- 文字と scalar ---
    def write(self, msg):
        print(msg, flush=True)
        self.log.write(msg + "\n")
        self.log.flush()

    def scalars(self, step, values):
        """有限の値だけ TensorBoard に書く（NaN / Inf の scalar は監視側が学習を止める合図になるので書かない）。書かなかったタグを返す。"""
        bad = [k for k, v in values.items() if not math.isfinite(v)]
        for k, v in values.items():
            if k not in bad:
                self.writer.add_scalar(k, v, step)
        if bad:
            self.write(f"[monitor] 警告: step {step} の scalar が有限でないので書かなかった: {bad}")
        return bad

    def tick(self, t_data):
        self.t_data += t_data
        self.n_window += 1

    def log_train(self, step, painter, hours):
        now = time.time()
        sec = (now - self.t_window) / max(self.n_window, 1)
        vals = painter.window_stats()
        vals.update({"diag/lr": painter.opt.param_groups[0]["lr"], "diag/sec_per_step": sec, "diag/imgs_per_sec": self.t["optim.batch_size"] / max(sec, 1e-9),
                     "diag/data_time_frac": self.t_data / max(now - self.t_window, 1e-9), "diag/hours": hours})
        self.scalars(step, vals)
        bins = " ".join(f"{vals[k]:.4f}" for k in sorted(vals) if k.startswith("loss_bin/"))
        self.write(f"[train] step {step} (epoch {step / self.t['log.steps_per_epoch']:.2f}) loss {vals['loss/train']:.5f} | bin {bins} | grad {vals['diag/grad_norm']:.3f} lr {vals['diag/lr']:.2e} | "
                   f"{sec:.3f} s/step {vals['diag/imgs_per_sec']:.1f} img/s data {vals['diag/data_time_frac'] * 100:.0f}% | {hours:.2f} h")
        painter.reset_window()
        self.t_window, self.t_data, self.n_window = time.time(), 0.0, 0
        return sec

    def log_val(self, step, vals):
        bad = [k for k in self.scalars(step, vals) if k.startswith("val/")]
        if bad:
            raise NonFiniteError(f"val の損失が有限でない: {bad}（step {step}）")
        bins = " ".join(f"{vals[k]:.4f}" for k in sorted(vals) if k.startswith("val/loss_bin/"))
        ratio = " ".join(f"{vals[k]:.2f}" for k in sorted(vals) if k.startswith("diag/bf16_val_ratio/"))
        self.write(f"[val] step {step} loss {vals['val/loss']:.5f} | bin {bins}" + (f" | bf16/fp32 {ratio}" if ratio else ""))

    # --- 表示の変換 ---
    def disp(self, x):
        """x（tensor (H, W) か numpy）→ 表示用 [0, 1]（HU display_hu_min〜display_hu_max を線形に）。"""
        a = x.detach().float().cpu().numpy() if torch.is_tensor(x) else np.asarray(x, np.float32)
        lo, hi = self.t["log.display_hu_min"], self.t["log.display_hu_max"]
        return np.clip((a * self.scale - lo) / (hi - lo), 0.0, 1.0)

    def diff(self, a, b):
        d = (a - b).detach().float().cpu().numpy() * self.scale
        return residual_rgb01(d, self.t["log.diff_range_hu"])

    def _emit(self, tag, name, step, panel01):
        u8 = to_u8(panel01)
        write_png(preview_path(self.run_dir, name, step), np.ascontiguousarray(u8[..., ::-1]))
        self.writer.add_image(tag, u8, step, dataformats="HWC")

    def _raw(self, step, name, x):
        write_png(step_images_dir(self.run_dir, step) / f"{name}.png", x_to_stored(x.detach().float().cpu().numpy(), self.hu))

    # --- 学習中のパッチ ---
    def log_train_patch(self, step, last):
        x0, cue, x, x0_hat, sigma, blur_px, noise_x, has = last
        rows = []
        for i in range(x0.shape[0]):
            lab = f"cue b{float(blur_px[i]):.2f} c{float(noise_x[i]) * self.scale:.0f}HU" if bool(has[i]) else "cue (hidden)"
            rows.append(([gray(self.disp(v[i, 0])) for v in (x0, cue, x, x0_hat)],
                         ["x0", lab, f"noisy sigma {float(sigma[i]) * self.scale:.0f}HU", "x0_hat"]))
        self._emit("preview/train_patch", "train_patch", step, build_grid(rows))

    # --- 途中経過の描画 ---
    def _draw(self, painter, key, src1=None, use_cue=True, n=None, size=None):
        t = self.t
        if use_cue:
            b = len(self.cues)
            src = src1.expand(b, -1, -1, -1).contiguous()
            blur = torch.tensor([c[0] for c in self.cues], device=self.device, dtype=torch.float32)
            noise = torch.tensor([c[1] / self.scale for c in self.cues], device=self.device, dtype=torch.float32)
        else:
            b = n
            src = torch.zeros(b, 1, size, size, device=self.device)
            blur = noise = torch.zeros(b, device=self.device)
        gens = [generator_of(self.device, "preview", t["log.preview_seed"], key, i) for i in range(b)]
        return redraw(painter.ema_denoiser(), src, blur, noise, use_cue, gens, t["sample.steps"], t["sample.eta"], t["sample.rho"],
                      t["diffusion.sigma_min"], t["diffusion.sigma_max"], self.anchor_px)

    def _fine(self, x):
        return x - gaussian_blur(x, self.t["log.check_highpass_px"])

    def preview(self, painter, step):
        """描き直し・疑似 EID・手がかり無しを描いて、パネル・生の 16bit・diag/check_* を出す。かかった秒を返す。"""
        t0 = time.time()
        t = self.t
        zs, zk = t["log.zoom_size"], t["log.zoom_scale"]
        eid = self.items[0][3]  # 基準の実 EID（eid_slices の 1 つ目）
        eid_fine_sd = float(self._fine(eid).std())
        checks = {}

        def add(name, i, v):
            checks.setdefault(name, []).append((cue_tag(*self.cues[i]), v))

        for key, idx, path, src in self.items:
            stem = path.stem
            out, cue = self._draw(painter, stem, src)
            rows, zrows = [], []
            zxy = {"eid": (t["log.eid_zoom_x"], t["log.eid_zoom_y"]), "pcd": (t["log.pcd_zoom_x"], t["log.pcd_zoom_y"])}[key] if idx == 0 else None
            for i, (b, c) in enumerate(self.cues):
                tag = cue_tag(b, c)
                o = out[i:i + 1]
                kind = "pseudo" if key == "pcd" else "redraw"
                self._raw(step, f"{stem}_{tag}_{kind}", o[0, 0])
                self._raw(step, f"{stem}_{tag}_cue", cue[i, 0])
                wide = float((wide_mean(o, self.anchor_px) - wide_mean(src, self.anchor_px)).abs().mean()) * self.scale
                fo, fs = self._fine(o), self._fine(src)
                if key == "eid" and idx == 0:
                    add("diag/check_redraw_wide_mean_diff_hu", i, wide)
                    add("diag/check_redraw_fine_sd_ratio", i, float(fo.std() / fs.std().clamp_min(1e-12)))
                    add("diag/check_redraw_fine_corr_src", i, _corr(fo, fs))
                elif key == "pcd" and idx == 0:
                    add("diag/check_pseudo_wide_mean_diff_hu", i, wide)
                    add("diag/check_pseudo_fine_corr_pcd", i, _corr(fo, fs))
                    add("diag/check_pseudo_fine_sd_ratio_vs_eid", i, float(fo.std()) / max(eid_fine_sd, 1e-12))
                if key == "pcd":
                    cols = [gray(self.disp(src[0, 0])), gray(self.disp(cue[i, 0])), gray(self.disp(o[0, 0])), gray(self.disp(eid[0, 0])), self.diff(o[0, 0], src[0, 0])]
                    labels = ["PCD", f"cue b{b:g}px c{c:g}HU", "pseudo EID", "real EID (ref)", f"pseudo - PCD (+-{t['log.diff_range_hu']:g}HU)"]
                else:
                    cols = [gray(self.disp(src[0, 0])), gray(self.disp(cue[i, 0])), gray(self.disp(o[0, 0])), self.diff(o[0, 0], src[0, 0])]
                    labels = ["real EID", f"cue b{b:g}px c{c:g}HU", "redraw", f"redraw - EID (+-{t['log.diff_range_hu']:g}HU)"]
                same = all(col.shape == cols[0].shape for col in cols)  # 参照の実 EID と PCD の大きさが違うときは参照列を外す
                if not same:
                    keep = [j for j, col in enumerate(cols) if col.shape == cols[0].shape]
                    cols, labels = [cols[j] for j in keep], [labels[j] for j in keep]
                rows.append((cols, labels))
                if zxy is not None:
                    if key == "pcd" and same:  # 参照の実 EID は実 EID 側の位置で切る
                        zc = [zoom(col, zxy[0], zxy[1], zs, zk) for col in cols[:3]] + [zoom(cols[3], t["log.eid_zoom_x"], t["log.eid_zoom_y"], zs, zk), zoom(cols[4], zxy[0], zxy[1], zs, zk)]
                    else:
                        zc = [zoom(col, zxy[0], zxy[1], zs, zk) for col in cols]
                    zrows.append((zc, labels))
            name = {"eid": "eid_redraw", "pcd": "pcd_pseudo"}[key]
            self._emit(f"preview/{name}/{stem}", f"{name}_{stem}", step, build_grid(rows))
            if zrows:
                self._emit(f"preview/zoom/{name}/{stem}", f"zoom_{name}_{stem}", step, build_grid(zrows))
        if t["log.nocue_n"] > 0:
            out, _ = self._draw(painter, "nocue", None, use_cue=False, n=t["log.nocue_n"], size=t["data.patch_size"])
            for i in range(out.shape[0]):
                self._raw(step, f"nocue_{i:02d}", out[i, 0])
            self._emit("preview/nocue_256", "nocue", step, build_grid([([gray(self.disp(out[i, 0])) for i in range(out.shape[0])], [f"no cue #{i}" for i in range(out.shape[0])])]))
        vals = {}
        for name, items in checks.items():
            vals[name] = sum(v for _, v in items) / len(items)
            for tag, v in items:
                vals[f"{name}/{tag}"] = v
        self.scalars(step, vals)
        sec = time.time() - t0
        main = " ".join(f"{k.replace('diag/check_', '')} {v:.3f}" for k, v in vals.items() if k.count("/") == 1)
        self.write(f"[preview] step {step} ({sec:.1f} s) {main}")
        self.writer.flush()
        return sec

    def close(self):
        self.writer.close()
        self.log.close()
