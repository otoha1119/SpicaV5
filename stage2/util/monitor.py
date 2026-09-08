"""stage2/util/monitor.py — Stage 2 学習の表示: ターミナル進捗バー（tqdm）+ TensorBoard（scalar のみ）+ loss_log.txt。

Stage 1 の util/monitor.py に相当する。画像は epoch 末のフル 1024 パネルだけ（128 patch のグリッドは出さない）。ディスクの preview は別フェーズ（ユーザー指示 2026-09-08）。

  ターミナル : 1 epoch = 1 本の tqdm バー（単位 = 画像枚数）。末尾に loss（正規化空間）と RMSE [HU]。epoch 末に 1 行の要約（tqdm.write）
  TensorBoard: <run>/tb/  global_step = total_iters（画像枚数。resume で復元されるので曲線がつながる）
     loss/<mse|l1>          学習の損失（正規化空間 [−1,1]。print_freq step ごと、直近 print_freq step の平均）
     train/rmse_HU, train/mae_HU   同じ区間の HU 換算
     time/sec_per_step, time/data_sec_per_step
     train/lr（epoch ごと）
     val/rmse_HU, val/mae_HU       epoch 末、val 症例のフル 1024（出力 − 教師）
     val/rmse_input_HU, val/mae_input_HU   参照線: 入力 − 教師（何もしない場合）。出力がこれを下回らなければ学習の意味が無い
     images/full/fixed              epoch 末（save_epoch_freq ごと）: 固定スライス（train.yaml log.full_slice = Stage 1 と同じ PCD-002-215）のパネル
                                    [EID-like1024 (input) | PCD1024 (teacher) | PCD-like1024 (output)]（util/panel.py。8bit、HU を display_hu_min..max で線形に黒..白）
     images/full/random（複数なら random0, random1, ...）  epoch ごとに別のランダムスライス（train ∪ val。ラベルに名前と train/val）
  loss_log.txt: print_freq ごとの損失（テキスト。grep 用の保険）と epoch 要約
"""

import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from util.panel import build_panel, full_labels
from util.run_paths import LOSS_LOG_FILE, TB_DIR

to_u8 = lambda x01: (np.clip(x01, 0.0, 1.0) * 255.0).round().astype(np.uint8)  # noqa: E731


class TrainMonitor:
    def __init__(self, run_dir, train, loss_name, batch_size, dataset_size):
        """train は launch の実効値（平坦 dict。hu_* と display_hu_* を使う）。"""
        self.run_dir = Path(run_dir)
        self.hu_min, self.hu_max = train["data.hu_min"], train["data.hu_max"]
        self.disp_min, self.disp_max = train["log.display_hu_min"], train["log.display_hu_max"]
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

    # ------------------------------------------------------------------ フル画像（epoch 末に TB へ）
    def _lin(self, t):
        """(1,H,W) 正規化 tensor → (H,W,3) float01 グレー（HU を display_hu_min..max で線形に黒..白。窓は掛けない）。"""
        hu = (t[0].float() + 1.0) / 2.0 * (self.hu_max - self.hu_min) + self.hu_min
        g = ((hu - self.disp_min) / float(self.disp_max - self.disp_min)).clamp(0.0, 1.0).numpy()
        return np.repeat(g[:, :, None], 3, axis=2)

    def log_full_images(self, model, dataset, full, epoch, total_iters):
        """固定 1 組 + ランダム n 組（PairDataset.full_slices(epoch)）をフル 1024 で通し、[EID-like1024 | PCD1024 | PCD-like1024] のパネルを TB へ（8bit）。
        タグは固定（fixed / random / randomN）なので TB のスライダーで epoch を追える。ディスクには書かない（preview は別フェーズ）。"""
        n_rand = full["kinds"].count("random")
        for i, ((in_path, pcd_path), kind, name) in enumerate(zip(full["pairs"], full["kinds"], full["names"])):
            x, t = dataset.load_full(in_path), dataset.load_full(pcd_path)
            y = model.predict(x)
            cols = [self._lin(x[0]), self._lin(t[0]), self._lin(y[0])]
            panel = build_panel(cols, full_labels(epoch, name if kind == "random" else None))
            tag = "fixed" if kind == "fixed" else ("random" if n_rand == 1 else f"random{i - 1}")
            self.tb.add_image(f"images/full/{tag}", to_u8(panel), total_iters, dataformats="HWC")
        model.net.train()

    def close(self):
        self.tb.close()
        self.log.close()
