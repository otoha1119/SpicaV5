"""stage2/util/monitor.py — Stage 2 学習の表示: ターミナル進捗バー（tqdm）+ TensorBoard（scalar のみ）+ loss_log.txt。

Stage 1 の util/monitor.py に相当する。画像は epoch 末のフル 1024（固定 + ランダム + 実 EID テスト）だけで、128 patch のグリッドは出さない。
保存先は Stage 1 と同じ 2 系統（util/run_paths.py）: output_images/epoch_NNN/ の生 16bit と preview_*/ の表示用。TensorBoard にも同じパネル（2026-09-09 ユーザー確定）。

  ターミナル : 1 epoch = 1 本の tqdm バー（単位 = 画像枚数）。末尾に loss（正規化空間）と RMSE [HU]。epoch 末に 1 行の要約（tqdm.write）
  TensorBoard: <run>/tb/  global_step = total_iters（画像枚数。resume で復元されるので曲線がつながる）
     loss/<mse|l1>          学習の損失（正規化空間 [−1,1]。print_freq step ごと、直近 print_freq step の平均）
     train/rmse_HU, train/mae_HU   同じ区間の HU 換算
     time/sec_per_step, time/data_sec_per_step
     train/lr（epoch ごと）
     val/rmse_HU, val/mae_HU       epoch 末、val 症例のフル 1024（出力 − 教師）
     val/rmse_input_HU, val/mae_input_HU   参照線: 入力 − 教師（何もしない場合）。出力がこれを下回らなければ学習の意味が無い
     images/full/fixed              epoch 末（save_epoch_freq ごと）: 固定スライス（train.yaml log.full_slice、PCD-002-236）の 4 列パネル
                                    [EID-like1024 (input) | PCD1024 (teacher) | PCD-like1024 (output) | 実 EID（log.eid_slice）→ PCD-like1024]（util/panel.py。8bit、HU を display_hu_min..max で線形）
     images/full/random（複数なら random0, random1, ...）  epoch ごとに別のランダムスライス（train ∪ val。ラベルに名前と train/val）の 3 列パネル
  ディスク（Stage 1 と同じ）: <run>/output_images/epoch_NNN/<slice>_{eidlike,pcd1024,pcdlike}.png, <eid_slice>_{eid1024,pcdlike}.png（生 uint16、stored = HU + 1400）
     <run>/output_images/preview_fixed_<slice>/ 00_eidlike1024_ / 01_pcd1024_ / 02_eid1024_（代表、初回だけ）, epoch_NNN_pcdlike.png, epoch_NNN_eid_pcdlike.png, epoch_NNN_panel.png（表示用、preview_bits）
     <run>/output_images/preview_random/epoch_NNN_<slice>.png（3 列パネルだけ）
  loss_log.txt: print_freq ごとの損失（テキスト。grep 用の保険）と epoch 要約
"""

import time
from pathlib import Path

import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.ct_io import denormalize, to_display, write_png
from util.panel import build_panel, full_labels
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
        self.loss_name = loss_name
        self.batch_size = batch_size
        self.dataset_size = dataset_size
        self.tb = SummaryWriter(str(self.run_dir / TB_DIR))
        self.log = open(self.run_dir / LOSS_LOG_FILE, "a", encoding="utf-8")
        self.log.write(f"================ Training ({time.strftime('%Y-%m-%d %H:%M:%S')}) ================\n")
        self.bar = None
        self._reset_window()

    def _reset_window(self):
        self.win = {"loss": 0.0, "rmse_hu": 0.0, "mae_hu": 0.0, "t_comp": 0.0, "t_data": 0.0, "n": 0}

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
        w["loss"] += losses["loss"]; w["rmse_hu"] += losses["rmse_hu"]; w["mae_hu"] += losses["mae_hu"]
        w["t_comp"] += t_comp; w["t_data"] += t_data; w["n"] += 1

    def step(self, epoch, total_iters):
        """print_freq step ごと: 直近区間の平均を TB / バー / loss_log に出す。"""
        w = self.win
        if w["n"] == 0:
            return
        n = w["n"]
        loss, rmse, mae = w["loss"] / n, w["rmse_hu"] / n, w["mae_hu"] / n
        self.tb.add_scalar(f"loss/{self.loss_name}", loss, total_iters)
        self.tb.add_scalar("train/rmse_HU", rmse, total_iters)
        self.tb.add_scalar("train/mae_HU", mae, total_iters)
        self.tb.add_scalar("time/sec_per_step", w["t_comp"] / n, total_iters)
        self.tb.add_scalar("time/data_sec_per_step", w["t_data"] / n, total_iters)
        if self.bar is not None:
            self.bar.set_postfix_str(f"{self.loss_name} {loss:.5f} | rmse {rmse:.1f} HU | data {w['t_data'] / n * 1000:.0f} ms", refresh=False)
        self.log.write(f"(epoch: {epoch}, iters: {total_iters}, time: {w['t_comp'] / n:.3f}, data: {w['t_data'] / n:.3f}) {self.loss_name}: {loss:.6f} rmse_HU: {rmse:.2f} mae_HU: {mae:.2f}\n")
        self.log.flush()
        self._reset_window()

    def log_val(self, val, total_iters):
        for k, v in val.items():
            if k != "n":
                self.tb.add_scalar(f"val/{k.replace('_hu', '_HU')}", v, total_iters)

    def end_epoch(self, epoch, total_epochs, total_iters, lr, elapsed, val):
        self.tb.add_scalar("train/lr", lr, total_iters)
        self.tb.flush()
        msg = (f"[epoch {epoch}/{total_epochs}] {elapsed:.0f}s, iters {total_iters}, lr {lr:.6f} | "
               f"val rmse {val['rmse_hu']:.1f} HU (input {val['rmse_input_hu']:.1f}) mae {val['mae_hu']:.1f} HU (input {val['mae_input_hu']:.1f}) on {val['n']} slices")
        self.write(msg)

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

    def _rgb(self, panel01):
        """表示用のパネル (H,W,3) float01 → cv2 用 BGR（preview_bits の深度）。"""
        arr = (np.clip(panel01, 0.0, 1.0) * (65535.0 if self.bits == 16 else 255.0)).round().astype(np.uint16 if self.bits == 16 else np.uint8)
        return np.ascontiguousarray(arr[:, :, ::-1])

    def save_full_images(self, model, dataset, full, epoch, total_iters):
        """epoch 末（save_epoch_freq ごと）: 固定 1 組 + ランダム n 組（PairDataset.full_slices）と実 EID テスト（log.eid_slice）をフル 1024 で通し、
          output_images/epoch_NNN/  生 uint16（<slice>_{eidlike,pcd1024,pcdlike}.png、<eid>_{eid1024,pcdlike}.png）
          preview_fixed_<slice>/    代表 00_/01_/02_（初回だけ）、epoch_NNN_pcdlike.png、epoch_NNN_eid_pcdlike.png、epoch_NNN_panel.png（4 列）
          preview_random/           epoch_NNN_<slice>.png（3 列パネル）
        を書き、TB にも images/full/fixed（4 列）と images/full/random（3 列）を 8bit で出す。タグは固定なので TB のスライダーで epoch を追える。"""
        out_dir = epoch_images_dir(self.run_dir, epoch)
        out_dir.mkdir(parents=True, exist_ok=True)
        ep = out_dir.name  # epoch_NNN
        # 実 EID テスト（固定パネルの 4 列目）
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
            cols = [self._lin01(a), self._lin01(b), self._lin01(y)]
            if kind == "fixed":
                cols.append(self._lin01(e_out))
                panel01 = build_panel(cols, full_labels(epoch, None, eid_stem))
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
                panel01 = build_panel(cols, full_labels(epoch, name))
                tag = "random" if n_rand == 1 else f"random{i - 1}"
                write_png(preview_dir(self.run_dir, "random") / f"{ep}_{stem}.png", self._rgb(panel01))
            self.tb.add_image(f"images/full/{tag}", to_u8(panel01), total_iters, dataformats="HWC")
        model.net.train()
        self.write(f"saved full-size images ({len(full['pairs'])} slices + real EID test) -> {out_dir}")

    def close(self):
        self.tb.close()
        self.log.close()
