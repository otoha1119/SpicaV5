"""stage2/util/monitor.py — Stage 2 学習の表示: ターミナル進捗バー（tqdm）+ TensorBoard（scalar のみ）+ loss_log.txt。

Stage 1 の util/monitor.py の scalar 部分に相当する。画像（TB のグリッド・フル画像・preview パネル）は別フェーズで詰める（ユーザー指示 2026-09-08）。

  ターミナル : 1 epoch = 1 本の tqdm バー（単位 = 画像枚数）。末尾に loss（正規化空間）と RMSE [HU]。epoch 末に 1 行の要約（tqdm.write）
  TensorBoard: <run>/tb/  global_step = total_iters（画像枚数。resume で復元されるので曲線がつながる）
     loss/<mse|l1>          学習の損失（正規化空間 [−1,1]。print_freq step ごと、直近 print_freq step の平均）
     train/rmse_HU, train/mae_HU   同じ区間の HU 換算
     time/sec_per_step, time/data_sec_per_step
     train/lr（epoch ごと）
     val/rmse_HU, val/mae_HU       epoch 末、val 症例のフル 1024（出力 − 教師）
     val/rmse_input_HU, val/mae_input_HU   参照線: 入力 − 教師（何もしない場合）。出力がこれを下回らなければ学習の意味が無い
  loss_log.txt: print_freq ごとの損失（テキスト。grep 用の保険）と epoch 要約
"""

import time
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from util.run_paths import LOSS_LOG_FILE, TB_DIR


class TrainMonitor:
    def __init__(self, run_dir, loss_name, batch_size, dataset_size):
        self.run_dir = Path(run_dir)
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

    def close(self):
        self.tb.close()
        self.log.close()
