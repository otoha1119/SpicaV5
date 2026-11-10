"""stage2/util/monitor.py — Stage 2 学習の表示: ターミナル進捗バー（tqdm）+ TensorBoard（scalar のみ）+ loss_log.txt。

Stage 1 の util/monitor.py に相当する。画像は学習中の 128 patch グリッド（image_freq step ごと。2026-09-09 追加）と、epoch 末のフル 1024（固定 + ランダム + 実 EID テスト）。
保存先は Stage 1 と同じ 2 系統（util/run_paths.py）: output_images/epoch_NNN/ の生 16bit と preview_*/ の表示用。TensorBoard にも同じパネル（2026-09-09 ユーザー確定）。

  ターミナル : 1 epoch = 1 本の tqdm バー（単位 = 画像枚数）。末尾に loss、rmse（正規化空間、0 へ）、ssim（1 へ）。epoch 末に 1 行の要約（tqdm.write）
  TensorBoard: <run>/tb/  global_step = total_iters（画像枚数。resume で復元されるので曲線がつながる）
     loss/<mse|l1>          学習の損失（正規化空間 [−1,1]。print_freq step ごと、直近 print_freq step の平均）
     train/rmse, train/ssim  同じ区間の RMSE（正規化空間。0 へ。HU は × 2748）と SSIM（1 へ）。出力 vs 教師、128 patch。指標は util/metrics.py（2026-09-09: HU 表示と MAE を廃止）
     time/sec_per_step, time/data_sec_per_step
     train/lr（epoch ごと）
     val/rmse, val/ssim, val/psnr   epoch 末、val 症例のフル 1024（出力 vs 教師）。rmse は正規化空間（0 へ）、ssim は 1 へ、psnr は dB（大きいほど良い）
     val/best_<metric>, val/best_epoch   best/ の記録（train.yaml log.best_metric。更新した epoch は loss_log.txt に [best] 行）
     val/rmse_input, val/ssim_input, val/psnr_input   参照線: 入力そのまま vs 教師（何もしない場合）。出力がこれを超えなければ学習の意味が無い
     images/current                 image_freq step ごと: いま学習中のバッチ先頭 n_images 枚の 128 patch。各行 [EID-like1024 (input) | PCD-like1024 (output) | PCD1024 (teacher)]
     images/fixed                   同じ間隔: 固定スライス（log.full_slice）から patch_seed で決めた位置の n_images 枚（同じ patch で推移を追う。eval で通す）
     images/full/fixed              epoch 末（save_epoch_freq ごと）: 固定スライス（train.yaml log.full_slice、PCD-002-236）の 5 列パネル（並びは 2026-09-10 ユーザー確定）
                                    [EID-like1024 (input) | PCD-like1024 (output) | PCD1024 (teacher) | 実 EID（log.eid_slice）→ PCD-like1024 | 実 EID1024（元の EID）]（util/panel.py。8bit、HU を display_hu_min..max で線形）
     images/full/random（複数なら random0, random1, ...）  epoch ごとに別のランダムスライス（val だけ。ラベルに名前）の 3 列パネル [入力 | 出力 | 教師]
     images/full/EID                epoch 末: 実 EID テスト（log.eid_slice）だけの 2 列 [EID1024（512 を補間）| PCD-like1024（出力）]（ユーザー指示 2026-09-09）
     ※ images/full/* は log.tb_full_size（512）に縮小して出す（1024 は TB で拡大が効かない）。ディスクの preview_*/ と epoch_NNN/ は 1024 のまま
  ディスク（Stage 1 と同じ）: <run>/output_images/epoch_NNN/<slice>_{eidlike,pcd1024,pcdlike}.png, <eid_slice>_{eid1024,pcdlike}.png（生 uint16、stored = HU + 1400）
     <run>/output_images/preview_fixed_<slice>/ 00_eidlike1024_ / 01_pcd1024_ / 02_eid1024_（代表、初回だけ）, epoch_NNN_pcdlike.png, epoch_NNN_eid_pcdlike.png, epoch_NNN_panel.png（表示用、preview_bits）
     <run>/output_images/preview_random/epoch_NNN_<slice>.png（3 列パネルだけ）
  loss_log.txt: print_freq ごとの損失（テキスト。grep 用の保険）と epoch 要約
"""

import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.ct_io import denormalize, to_display, write_png
from torchvision.utils import make_grid

from util.panel import build_panel, eid_labels, full_labels, label_strip
from util.run_paths import LOSS_LOG_FILE, TB_DIR, epoch_images_dir, preview_dir

to_u8 = lambda x01: (np.clip(x01, 0.0, 1.0) * 255.0).round().astype(np.uint8)  # noqa: E731


class TrainMonitor:
    def __init__(self, run_dir, train, loss_name, batch_size, dataset_size):
        """train は launch の実効値（平坦 dict。hu_* と display_hu_* を使う）。"""
        self.run_dir = Path(run_dir)
        self.hu = (train["data.hu_offset"], train["data.hu_min"], train["data.hu_max"])
        self.hu_min, self.hu_max = train["data.hu_min"], train["data.hu_max"]
        self.disp_min, self.disp_max = train["log.display_hu_min"], train["log.display_hu_max"]
        self.bits = train["log.preview_bits"]
        self.tb_size = train["log.tb_full_size"]  # TB の images/full/* は 1024 → この一辺に縮小（ディスクは縮小しない）
        self.loss_name = loss_name
        self.batch_size = batch_size
        self.dataset_size = dataset_size
        self.tb = SummaryWriter(str(self.run_dir / TB_DIR))
        self.log = open(self.run_dir / LOSS_LOG_FILE, "a", encoding="utf-8")
        self.log.write(f"================ Training ({time.strftime('%Y-%m-%d %H:%M:%S')}) ================\n")
        self.bar = None
        self._reset_window()

    def _reset_window(self):
        self.win = {"loss": 0.0, "mse": 0.0, "ssim": 0.0, "t_comp": 0.0, "t_data": 0.0, "n": 0}

    def epoch_bar(self, loader, epoch, total_epochs):
        self.bar = tqdm(loader, total=len(loader), unit="img", unit_scale=self.batch_size, dynamic_ncols=True, mininterval=0.5,
                        desc=f"epoch {epoch}/{total_epochs}", leave=True)
        return self.bar

    def write(self, msg):
        tqdm.write(msg)
        self.log.write(msg + "\n")
        self.log.flush()

    def accumulate(self, losses, t_comp, t_data):
        w = self.win
        w["loss"] += losses["loss"]; w["mse"] += losses["mse"]; w["ssim"] += losses["ssim"]
        w["t_comp"] += t_comp; w["t_data"] += t_data; w["n"] += 1

    def step(self, epoch, total_iters):
        """print_freq step ごと: 直近区間の平均を TB / バー / loss_log に出す。"""
        w = self.win
        if w["n"] == 0:
            return
        n = w["n"]
        loss, rmse, sim = w["loss"] / n, (w["mse"] / n) ** 0.5, w["ssim"] / n
        self.tb.add_scalar(f"loss/{self.loss_name}", loss, total_iters)
        self.tb.add_scalar("train/rmse", rmse, total_iters)
        self.tb.add_scalar("train/ssim", sim, total_iters)
        self.tb.add_scalar("time/sec_per_step", w["t_comp"] / n, total_iters)
        self.tb.add_scalar("time/data_sec_per_step", w["t_data"] / n, total_iters)
        if self.bar is not None:
            self.bar.set_postfix_str(f"{self.loss_name} {loss:.5f} | rmse {rmse:.4f} | ssim {sim:.4f} | data {w['t_data'] / n * 1000:.0f} ms", refresh=False)
        self.log.write(f"(epoch: {epoch}, iters: {total_iters}, time: {w['t_comp'] / n:.3f}, data: {w['t_data'] / n:.3f}) {self.loss_name}: {loss:.6f} rmse: {rmse:.5f} ssim: {sim:.4f}\n")
        self.log.flush()
        self._reset_window()

    def log_val(self, val, total_iters):
        for k, v in val.items():
            if k != "n":
                self.tb.add_scalar(f"val/{k}", v, total_iters)

    def log_best(self, best, improved, epoch, total_iters):
        """epoch 末: best の記録を TB（val/best_<metric> と val/best_epoch）に出し、更新した epoch は 1 行書く。"""
        if best is None:
            return
        self.tb.add_scalar(f"val/best_{best['metric']}", best["value"], total_iters)
        self.tb.add_scalar("val/best_epoch", best["epoch"], total_iters)
        if improved:
            self.write(f"[best] epoch {epoch}: val/{best['metric']} {best['value']:.6f} が最良 → best/ を更新")

    def end_epoch(self, epoch, total_epochs, total_iters, lr, elapsed, val):
        self.tb.add_scalar("train/lr", lr, total_iters)
        self.tb.flush()
        msg = (f"[epoch {epoch}/{total_epochs}] {elapsed:.0f}s, iters {total_iters}, lr {lr:.6f} | "
               f"val rmse {val['rmse']:.5f} (input {val['rmse_input']:.5f}) ssim {val['ssim']:.4f} (input {val['ssim_input']:.4f}) psnr {val['psnr']:.2f} dB (input {val['psnr_input']:.2f}) on {val['n']} slices")
        self.write(msg)

    # ------------------------------------------------------------------ 128 patch グリッド（image_freq step ごと。TB だけ）
    PATCH_LABELS = ("EID-like (input)", "PCD-like (output)", "PCD1024 (teacher)")

    def _patch_grid(self, x, y, t):
        """(n,1,p,p) 正規化 tensor ×3 → 各行 [input | output | teacher] のグリッド (H,W,3) uint8。上に列ラベル。表示は HU を display_hu_min..max で線形。"""
        rows = []
        for i in range(x.shape[0]):
            for tsr in (x, y, t):
                g = self._lin01(self._stored(tsr[i : i + 1]))  # (p,p,3)
                rows.append(torch.from_numpy(g).permute(2, 0, 1))
        pad = 2
        grid = make_grid(torch.stack(rows), nrow=3, padding=pad, pad_value=0.5).permute(1, 2, 0).numpy()  # (H, 3p+4*pad, 3)
        p = x.shape[-1]
        scale = 0.45
        while scale > 0.25 and max(cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] for s in self.PATCH_LABELS) > p - 4:
            scale -= 0.05  # 列幅（patch_size）に収まるまで文字を小さく
        blank = np.ones((22, pad, 3), np.float32)
        header = [blank]
        for s in self.PATCH_LABELS:
            header += [label_strip(p, s, height=22, scale=scale), blank]
        return to_u8(np.concatenate([np.concatenate(header, axis=1), grid], axis=0))

    def log_images(self, model, fixed, total_iters):
        """image_freq step ごと: images/current（直近バッチ先頭 n_images 枚）と images/fixed（固定スライスの固定 patch を eval で通す）を TB へ。ディスクには書かない。"""
        n = fixed["input"].shape[0]
        if model.last is not None:
            x, y, t = (a[:n].cpu() for a in model.last)
            self.tb.add_image("images/current", self._patch_grid(x, y, t), total_iters, dataformats="HWC")
        y_f = model.predict(fixed["input"])  # eval で通す（BN は無いが学習の勾配を汚さない）
        model.net.train()
        self.tb.add_image("images/fixed", self._patch_grid(fixed["input"], y_f, fixed["target"]), total_iters, dataformats="HWC")

    # ------------------------------------------------------------------ フル画像（epoch 末。生 16bit / preview / TB）
    def _stored(self, t):
        """(1,1,H,W) 正規化 tensor → stored uint16 (H,W)（inference と同じ denormalize。丸め・クリップは最後に 1 回）。"""
        return denormalize(t[0, 0].numpy(), *self.hu)

    def _lin01(self, stored):
        """stored uint16 → (H,W,3) float01 グレー（HU を display_hu_min..max で線形に黒..白。窓は掛けない。パネル用）。"""
        g = np.clip((stored.astype(np.float32) - self.hu[0] - self.disp_min) / float(self.disp_max - self.disp_min), 0.0, 1.0)
        return np.repeat(g[:, :, None], 3, axis=2)

    def _gray(self, stored):
        """表示用のグレー 1ch（preview_bits の深度）。"""
        return to_display(stored, self.hu[0], self.disp_min, self.disp_max, self.bits)

    def _tb(self, col01):
        """TB 用に (H,W,3) float01 を一辺 tb_full_size に縮小（INTER_AREA）。すでにその大きさなら何もしない。"""
        if col01.shape[0] == self.tb_size and col01.shape[1] == self.tb_size:
            return col01
        return cv2.resize(col01, (self.tb_size, self.tb_size), interpolation=cv2.INTER_AREA)

    def _rgb(self, panel01):
        """表示用のパネル (H,W,3) float01 → cv2 用 BGR（preview_bits の深度）。"""
        arr = (np.clip(panel01, 0.0, 1.0) * (65535.0 if self.bits == 16 else 255.0)).round().astype(np.uint16 if self.bits == 16 else np.uint8)
        return np.ascontiguousarray(arr[:, :, ::-1])

    def save_full_images(self, model, dataset, full, epoch, total_iters):
        """epoch 末（save_epoch_freq ごと）: 固定 1 組 + ランダム n 組（PairDataset.full_slices）と実 EID テスト（log.eid_slice）をフル 1024 で通し、
          output_images/epoch_NNN/  生 uint16（<slice>_{eidlike,pcd1024,pcdlike}.png、<eid>_{eid1024,pcdlike}.png）
          preview_fixed_<slice>/    代表 00_/01_/02_（初回だけ）、epoch_NNN_pcdlike.png、epoch_NNN_eid_pcdlike.png、epoch_NNN_panel.png（5 列）
          preview_random/           epoch_NNN_<slice>.png（3 列パネル）
        を書き、TB にも images/full/fixed（5 列）と images/full/random（3 列）を 8bit で出す。タグは固定なので TB のスライダーで epoch を追える。"""
        out_dir = epoch_images_dir(self.run_dir, epoch)
        out_dir.mkdir(parents=True, exist_ok=True)
        ep = out_dir.name  # epoch_NNN
        model_tag = f"epoch {int(epoch)}"
        scale, interp = dataset.upsample_cfg
        # 実 EID テスト（固定パネルの 4・5 列目 = 出力と元の EID）
        x_e = dataset.load_eid_full()
        e_in, e_out = self._stored(x_e), self._stored(model.predict(x_e))
        eid_stem = Path(dataset.eid_path).stem
        write_png(out_dir / f"{eid_stem}_eid1024.png", e_in)
        write_png(out_dir / f"{eid_stem}_pcdlike.png", e_out)
        n_rand = full["kinds"].count("random")
        for i, ((in_path, pcd_path), kind, name) in enumerate(zip(full["pairs"], full["kinds"], full["names"])):
            stem = Path(pcd_path).stem
            x, t = dataset.load_full(in_path), dataset.load_full(pcd_path)
            a, b, y = self._stored(x), self._stored(t), self._stored(model.predict(x))
            write_png(out_dir / f"{stem}_eidlike.png", a)
            write_png(out_dir / f"{stem}_pcd1024.png", b)
            write_png(out_dir / f"{stem}_pcdlike.png", y)
            cols = [self._lin01(a), self._lin01(y), self._lin01(b)]  # 入力 | 出力 | 教師（並びは 2026-09-10 ユーザー確定）
            if kind == "fixed":
                cols += [self._lin01(e_out), self._lin01(e_in)]  # + 実 EID → PCD-like1024 | 実 EID1024（元）
                labels = full_labels(model_tag, None, True, (eid_stem, scale, interp))
                panel01 = build_panel(cols, labels)
                tag = "fixed"
                d = preview_dir(self.run_dir, f"fixed_{stem}")
                if not (d / f"00_eidlike1024_{stem}.png").is_file():  # 代表は初回の checkpoint で 1 回だけ
                    write_png(d / f"00_eidlike1024_{stem}.png", self._gray(a))
                    write_png(d / f"01_pcd1024_{stem}.png", self._gray(b))
                    write_png(d / f"02_eid1024_{eid_stem}.png", self._gray(e_in))
                write_png(d / f"{ep}_pcdlike.png", self._gray(y))
                write_png(d / f"{ep}_eid_pcdlike.png", self._gray(e_out))
                write_png(d / f"{ep}_panel.png", self._rgb(panel01))
            else:
                labels = full_labels(model_tag, name)
                panel01 = build_panel(cols, labels)
                tag = "random" if n_rand == 1 else f"random{i - 1}"
                write_png(preview_dir(self.run_dir, "random") / f"{ep}_{stem}.png", self._rgb(panel01))
            self.tb.add_image(f"images/full/{tag}", to_u8(build_panel([self._tb(c) for c in cols], labels)), total_iters, dataformats="HWC")  # TB は縮小版
        eid_panel = build_panel([self._tb(self._lin01(e_in)), self._tb(self._lin01(e_out))], eid_labels(model_tag, eid_stem, scale, interp))
        self.tb.add_image("images/full/EID", to_u8(eid_panel), total_iters, dataformats="HWC")  # 実 EID テストだけの 2 列（縮小版）
        model.net.train()
        self.write(f"saved full-size images ({len(full['pairs'])} slices + real EID test) -> {out_dir}")

    def close(self):
        self.tb.close()
        self.log.close()
