"""stage2/util/monitor.py — 学習の途中経過: TensorBoard の scalar と画像、loss_log.txt、epoch 末の確認（数字・パネル・生の 16bit）。設計書 F-2・F-3・G。

scalar のタグ（監視ツールは名前に loss / val/ / diag/ / sigma を含むものを Mac 側の要約に出す。設計書の speed/* lock/* leak/* var/* mag/* fake/* は diag/ の下に置いた）:
  U: loss/mse、diag/train_rmse、diag/grad_norm、diag/lr、diag/sec_per_step、diag/imgs_per_sec、diag/data_time_frac、diag/hours
     val/rmse、val/rmse_input（フル 1024 の μ 対 X、x↑ 対 X。n の単位 = 旧 Stage 2 の val/rmse と同じ計算。**入力そのままの値に近いほど正直**。best の選択には使わない）
     diag/leak_noise_explained、diag/var_gap、diag/lock_target_shift_hu、diag/lock_f_shift_hu_<組>、diag/lock_out_shift_hu_<組>
  V: loss/flow、loss/flow_t0〜t3、val/flow（固定パッチ）、val/rmse_mu・val/rmse_out・val/rmse_input（見るだけ）
     diag/leak_corr_fine、diag/leak_noise_explained、diag/lock_corr_rms_hu_<組>、diag/lock_out_shift_hu_<組>、diag/lock_target_shift_hu
     diag/fake_seed_sd_hu、diag/fake_seed_low_ratio、diag/fake_ref_low_ratio（log.pcd_slice の 1 枚で、乱数の種を変えた 2 出力の差）
  共通: diag/mag/<数字>_<段>_<組>（虫眼鏡。数字 = fine_sd / edge / band1〜5 / shift_hu、段 = in / mu / out / ref、組 = pseudo / imagecas / hospital / pcd）、diag/check_sec
  どれも確認用で、合格線は無い（設計書 G）。

画像（TensorBoard の images/… と <run>/preview/<名前>/<印>.png。生の 16bit は <run>/output_images/<印>/。<印> = epoch_NNN、最初の確認は first_step_NNNNNNN）:
  images/train_patch          学習中のパッチ。U: [入力 | μ | 正解 X′]、V: [μ | 1 段での見積もり | 正解 X′]（TensorBoard だけ）
  images/fixed                固定パネル。U: [疑似 EID↑ | μ | PCD1024 | 実 EID → μ | 実 EID↑]、V: [疑似 EID↑ | μ | U+V | PCD1024 | 実 EID → μ | 実 EID → U+V | 実 EID↑]
  images/eid/<スライス>       log.eid_slices の 2 つ目以降（病院 EID など）: [EID↑ | → μ | （→ U+V）]
  images/random/<番号>        val から epoch ごとに別の 1 枚: [入力 | μ | （U+V）| 正解]
  images/crop_pcd、crop_eid   固定パネルの切り抜き（log.crop_size 四方を log.crop_scale 倍）
  images/seed_diff            V だけ: 切り抜きで [seed a | seed b | 差 | ぼかした差]（1 行目 = 疑似 EID、2 行目 = 実 EID）
乱数は毎回同じ種（log.preview_seed とスライス名から）。表示は HU −1400〜3000。
"""

import math
import random
import time
from pathlib import Path

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from check.magnifier import Stopped, build_slices, evaluate
from check.seed_diff import diff_row, sample_seeds, seed_stats
from data.ct_io import n_to_stored, write_png
from models.trainer import NonFiniteError
from util import metrics as M
from util.panel import build_grid, to_u8
from util.run_paths import LOSS_LOG_FILE, TB_DIR, check_dir, images_dir, preview_path
from util.views import View, crop_zoom, shrink, to_np

U_TAGS = {"rmse_mu": "val/rmse", "rmse_input": "val/rmse_input"}
V_TAGS = {"rmse_mu": "val/rmse_mu", "rmse_out": "val/rmse_out", "rmse_input": "val/rmse_input"}


class Monitor:
    def __init__(self, run_dir, kind, data, cfg, machine, pd, device, should_stop):
        self.run_dir, self.kind, self.data, self.c, self.machine, self.pd, self.device = Path(run_dir), kind, data, cfg, machine, pd, device
        self.should_stop = should_stop
        self.view = View(data)
        self.writer = SummaryWriter(str(self.run_dir / TB_DIR))
        self.log = open(self.run_dir / LOSS_LOG_FILE, "a", encoding="utf-8")
        self.t_window, self.t_data, self.n_window = time.time(), 0.0, 0
        fixed = cfg["log.pcd_slice"]
        self.fixed = {"case": Path(fixed).parts[0], "name": Path(fixed).name, "x": pd.load_x(pd.set_dir / "k00" / fixed),
                      "t": pd.load_t(pd.pcd_dir / fixed)}
        self.eids = [{"case": Path(p).parent.name, "name": Path(p).name, "x": pd.load_x(p)} for p in cfg["log.eid_slices"]]

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

    def log_train(self, step, model, hours):
        now = time.time()
        sec = (now - self.t_window) / max(self.n_window, 1)
        vals = model.window_stats()
        bs = self.c["optim.batch_size"]
        vals.update({"diag/lr": model.opt.param_groups[0]["lr"], "diag/sec_per_step": sec, "diag/imgs_per_sec": bs / max(sec, 1e-9),
                     "diag/data_time_frac": self.t_data / max(now - self.t_window, 1e-9), "diag/hours": hours})
        self.scalars(step, vals)
        main = " ".join(f"{k.split('/', 1)[1]} {v:.5f}" for k, v in vals.items() if k.startswith("loss"))
        self.write(f"[train] step {step} (epoch {step / self.c['optim.steps_per_epoch']:.2f}) {main} | grad {vals['diag/grad_norm']:.3f} lr {vals['diag/lr']:.2e} | "
                   f"{sec:.3f} s/step {vals['diag/imgs_per_sec']:.1f} img/s data {vals['diag/data_time_frac'] * 100:.0f}% | {hours:.2f} h")
        model.reset_window()
        self.t_window, self.t_data, self.n_window = time.time(), 0.0, 0
        return sec

    def log_val(self, step, vals):
        bad = [k for k in self.scalars(step, vals) if k.startswith("val/")]
        if bad:
            raise NonFiniteError(f"val の値が有限でない: {bad}（step {step}）")

    # --- 画像 ---
    def _hu(self, pipe, n):
        return pipe.units.hu(n)

    def _emit(self, tag, name, mark, step, panel01, tb_scale=1):
        write_png(preview_path(self.run_dir, name, mark), np.ascontiguousarray(to_u8(panel01)[..., ::-1]))
        self.writer.add_image(tag, to_u8(shrink(panel01, tb_scale)), step, dataformats="HWC")

    def _raw(self, mark, name, n):
        write_png(images_dir(self.run_dir, mark) / f"{name}.png", n_to_stored(to_np(n), self.pd.hu))

    def log_images(self, step, model, units):
        """学習中のパッチ（TensorBoard だけ）。"""
        last, v = model.last, self.view
        if last is None:
            return
        rows = []
        if self.kind == "u":
            for i in range(last["x"].shape[0]):
                rows.append(([v.img(units.hu(last[k][i])) for k in ("x", "mu", "target")], ["input (pseudo EID up)", "mu = U", "target X'"]))
        else:
            for i in range(last["mu"].shape[0]):
                rows.append(([v.img(units.hu(last[k][i])) for k in ("mu", "x1_hat", "target")],
                             ["mu = U", f"1-step est. t {float(last['t'][i]):.2f} sigma_s {float(last['sigma_s_hu'][i]):.0f}HU", "target X'"]))
        self.writer.add_image("images/train_patch", to_u8(build_grid(rows)), step, dataformats="HWC")

    def _one(self, pipe, item, mode, seeds):
        """1 枚を通す。戻り値 {"x", "mu", "ys": [種ごとの出力]}（n、device 上）。"""
        x = item["x"].to(self.device)
        mu, _ = pipe.mu(x)
        ys = sample_seeds(pipe, x, mu, item["case"], item["name"], seeds, self.c["flow.sigma_s_hu"], self.c["flow.steps"]) if mode == "uv" else []
        return {"x": x, "mu": mu, "ys": ys}

    def epoch_end(self, step, mark, pipe, reduced=False):
        """epoch 末の確認: 数字（val・虫眼鏡・見張り）→ magnifier.md → パネルと生の 16bit。SIGTERM を受けていたら途中で Stopped。戻り値は書いた scalar。"""
        t0 = time.time()
        c, v, pd = self.c, self.view, self.pd
        mode = "uv" if self.kind == "v" else "u"
        sig, steps = (c["flow.sigma_s_hu"], c["flow.steps"]) if mode == "uv" else (None, None)
        seed = c["log.preview_seed"]
        n_val = c["log.first_check_slices"] if reduced else c["log.val_slices_per_case"]
        n_mag = c["log.first_check_slices"] if reduced else c["mag.slices_per_case"]
        slices = build_slices(pd, self.machine["eid_dir"], n_val, n_mag, c["mag.eid_cases"], c["mag.hospital_dirs"], with_alt=(mode == "u"))
        sc, table = evaluate(pipe, pd, slices, mode, sig, steps, seed, c["mag.tile"], self.data["display.body_hu"], self.should_stop)
        names = U_TAGS if mode == "u" else V_TAGS
        vals = {names.get(k, f"diag/{k}"): val for k, val in sc.items()}
        vals.update(table.scalars())
        H = lambda n: self._hu(pipe, n)

        # 固定パネル（疑似 EID と 1 つ目の実 EID）
        seeds = [seed, seed + 1] if mode == "uv" else []
        fx, e0 = self.fixed, self.eids[0]
        a, b = self._one(pipe, fx, mode, seeds), self._one(pipe, e0, mode, seeds)
        if self.should_stop():
            raise Stopped()
        t = fx["t"].to(self.device)
        stem, estem = Path(fx["name"]).stem, Path(e0["name"]).stem
        for name, n in ((f"{stem}_input", a["x"]), (f"{stem}_mu", a["mu"]), (f"{stem}_pcd1024", t), (f"{estem}_input", b["x"]), (f"{estem}_mu", b["mu"])):
            self._raw(mark, name, n)
        left = [(a["x"], "pseudo EID up"), (a["mu"], "mu = U")] + ([(a["ys"][0], "U+V")] if a["ys"] else []) + [(t, "PCD1024 (target)")]
        right = [(b["mu"], "real EID -> mu")] + ([(b["ys"][0], "real EID -> U+V")] if b["ys"] else []) + [(b["x"], f"real EID up ({estem})")]
        if a["ys"]:
            self._raw(mark, f"{stem}_out", a["ys"][0]); self._raw(mark, f"{estem}_out", b["ys"][0])
        if b["x"].shape == a["x"].shape:
            self._emit("images/fixed", f"fixed_{stem}", mark, step, build_grid([([v.img(H(n)) for n, _ in left + right], [l for _, l in left + right])]), self.data["display.tb_scale"])
        else:  # 実 EID の大きさが違うときは別のパネル
            self._emit("images/fixed", f"fixed_{stem}", mark, step, build_grid([([v.img(H(n)) for n, _ in left], [l for _, l in left])]), self.data["display.tb_scale"])
            self._emit("images/fixed_eid", f"fixed_{estem}", mark, step, build_grid([([v.img(H(n)) for n, _ in right], [l for _, l in right])]), self.data["display.tb_scale"])
        cs, ck = c["log.crop_size"], c["log.crop_scale"]
        for tag, cols, center in (("crop_pcd", left, c["log.pcd_crop_center"]), ("crop_eid", right, c["log.eid_crop_center"])):
            self._emit(f"images/{tag}", tag, mark, step, build_grid([([crop_zoom(v.img(H(n)), center, cs, ck) for n, _ in cols], [l for _, l in cols])]))
        if mode == "uv":
            xh = H(a["x"])
            mask = M.body_mask(xh, pipe.sigma_px, self.data["display.body_hu"])
            vals.update({f"diag/{k}": val for k, val in seed_stats(H(a["ys"][0]), H(a["ys"][1]), mask, c["mag.tile"], H(t), H(a["mu"])).items()})
            rows = [diff_row(v, H(a["ys"][0]), H(a["ys"][1]), c["log.pcd_crop_center"], cs, ck, "pseudo")]
            if b["x"].shape == a["x"].shape:
                rows.append(diff_row(v, H(b["ys"][0]), H(b["ys"][1]), c["log.eid_crop_center"], cs, ck, "real EID"))
            self._emit("images/seed_diff", "seed_diff", mark, step, build_grid(rows))

        # 2 つ目以降の実 EID（病院 EID など）
        for e in self.eids[1:]:
            if self.should_stop():
                raise Stopped()
            r = self._one(pipe, e, mode, seeds[:1])
            es = Path(e["name"]).stem
            cols = [(r["x"], f"EID up ({es})"), (r["mu"], "-> mu")] + ([(r["ys"][0], "-> U+V")] if r["ys"] else [])
            self._raw(mark, f"{es}_input", r["x"]); self._raw(mark, f"{es}_mu", r["mu"])
            if r["ys"]:
                self._raw(mark, f"{es}_out", r["ys"][0])
            self._emit(f"images/eid/{es}", f"eid_{es}", mark, step, build_grid([([v.img(H(n)) for n, _ in cols], [l for _, l in cols])]), self.data["display.tb_scale"])

        # val から毎回別の 1 枚
        pool = [(cc, n, p) for cc in pd.val_cases for n, p in pd.pairs[cc]]
        for i, (cc, n, p) in enumerate(random.Random(f"random/{seed}/{step}").sample(pool, min(c["log.n_random"], len(pool)))):
            if self.should_stop():
                raise Stopped()
            r = self._one(pipe, {"case": cc, "name": n, "x": pd.load_x(pd.pseudo_path(0, cc, n))}, mode, seeds[:1])
            cols = [(r["x"], f"input ({Path(n).stem})"), (r["mu"], "mu = U")] + ([(r["ys"][0], "U+V")] if r["ys"] else []) + [(pd.load_t(p).to(self.device), "PCD1024 (target)")]
            self._emit(f"images/random/{i}", f"random_{i}", mark, step, build_grid([([v.img(H(x)) for x, _ in cols], [l for _, l in cols])]), self.data["display.tb_scale"])

        sec = time.time() - t0
        vals["diag/check_sec"] = sec
        d = check_dir(self.run_dir, mark)
        d.mkdir(parents=True, exist_ok=True)
        extra = {"step": step, "mode": mode, **({"sigma_s_hu": sig, "steps": steps} if mode == "uv" else {}), **{k: val for k, val in vals.items() if "/mag/" not in k}}
        (d / "magnifier.md").write_text(table.markdown(f"{self.run_dir.name} {mark}（{'縮小版' if reduced else 'epoch 末'}）", extra), encoding="utf-8")
        self.log_val(step, vals)
        main = " ".join(f"{k.split('/', 1)[1]} {val:.4g}" for k, val in vals.items() if "/mag/" not in k)
        self.write(f"[check] step {step} {mark} ({sec:.0f} s, {len(slices)} 枚) {main}")
        self.writer.flush()
        return vals

    def close(self):
        self.writer.close()
        self.log.close()
