"""util/validator.py — epoch ごとの自動採点と best_eNNN/ の自動更新（2026-11-02、stage1_v3/util/validator.py の v1 版。Stage 間で import しない）。

  固定集合（run の最初に 1 回だけ決める。seed = data.patch_seed）:
    PCD: 学習症例（dataset の idx_a）から val.n_pcd_slices 枚。各 epoch で G に通し（eval、no_grad、FP32）、EID-like のフル 512 を作る
    EID: 学習症例（idx_b）から val.n_eid_slices 枚。参照統計（中央値と四分位）は最初に 1 回計算
  指標（util/metrics.analyze_slice、R-S22 の 3 軸）: local_sd（ノイズ量）, nps_f_av（NPS の形）, edge_grad_p90 / p99（鮮鋭度）, hu_p50 / hu_p90（HU 水準）
  score = Σ_k |median_gen,k − median_eid,k| / IQR_eid,k（小さいほど良い。0 なら 6 指標が全部 EID の中央値に一致）
    v3 にあった恒等残差 RMS(R(y)) の項は v1 に恒等項が無いので入れない
  TensorBoard: val/<k>_gen, val/<k>_eid（参照、定数）, val/<k>_dist, val/score, val/best_score
  ディスク: <run>/val_log.txt（1 epoch 1 行）。val.auto_best なら score 最良で <run>/best_eNNN/ を保存（model.save_networks、原子的）し、
    他の best_e*（と旧 best/）を消して best.txt に記録。bash start.sh best <run> <epoch>（mark_best.py）で手動上書きできる
  task: sr では採点しない（A = EID1024、B = PCD1024 で指標の意味が変わる。val.n_pcd_slices を 0 にする）
"""

import random
import time
from pathlib import Path

import numpy as np
import torch

from data.ct_dataset import normalize, read_stored
from models.spectral_loss import parse_floats
from util.metrics import analyze_slice, body_mask, summarize
from util.run_paths import BEST_NOTE, best_dirname, retire_other_bests

KEYS = ("local_sd", "nps_f_av", "edge_grad_p90", "edge_grad_p99", "hu_p50", "hu_p90")
VAL_LOG_FILE = "val_log.txt"


class Validator:
    def __init__(self, idx_a, idx_b, opt, run_dir, tb):
        self.opt = opt
        self.run_dir = Path(run_dir)
        self.tb = tb
        self.enabled = opt.val_pcd_slices > 0 and opt.task == "degrade"
        if not self.enabled:
            return
        self.thr = float(opt.body_threshold_hu)
        seed = int(opt.patch_seed) * 1_000_003 + 4242
        rng = random.Random(seed)
        self.pcd_paths = rng.sample(list(idx_a.all_paths), min(opt.val_pcd_slices, len(idx_a.all_paths)))
        self.eid_paths = rng.sample(list(idx_b.all_paths), min(opt.val_eid_slices, len(idx_b.all_paths)))
        self.np_rng_seed = seed
        self.best_score, self.best_epoch = float("inf"), None
        note = self.run_dir / BEST_NOTE  # resume のとき: 自動 best の score を引き継ぐ（無いと再開直後の epoch が必ず best になる）。手動 best（score 無し）は次の採点で上書きされ得る
        if note.is_file():
            kv = dict(l.split(":", 1) for l in note.read_text(encoding="utf-8").splitlines() if ":" in l)
            if "score" in kv and "epoch" in kv:
                try:
                    self.best_score, self.best_epoch = float(kv["score"]), int(kv["epoch"])
                except ValueError:
                    pass
        t0 = time.time()
        rows = [analyze_slice(self._hu(p), self.thr, np.random.default_rng(seed)) for p in self.eid_paths]
        self.ref = {k: summarize(rows, k) for k in KEYS}
        for k in KEYS:
            if self.ref[k]["n"] < 8 or not np.isfinite(self.ref[k]["median"]):
                raise RuntimeError(f"参照 EID の指標 {k} が計算できません（有効 {self.ref[k]['n']} 枚）。val.n_eid_slices を増やすか val.body_threshold_hu を確認")
            self.ref[k]["iqr"] = max(self.ref[k]["q75"] - self.ref[k]["q25"], 1e-6)
        with open(self.run_dir / VAL_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"================ Validation ({time.strftime('%Y-%m-%d %H:%M:%S')}) PCD {len(self.pcd_paths)} / EID {len(self.eid_paths)} ================\n")
            f.write("ref_eid " + " ".join(f"{k}={self.ref[k]['median']:.3f}[{self.ref[k]['q25']:.3f},{self.ref[k]['q75']:.3f}]" for k in KEYS) + "\n")
        print(f"[val] 参照 EID {len(self.eid_paths)} 枚の統計 ({time.time() - t0:.1f}s): " + " ".join(f"{k} {self.ref[k]['median']:.2f}" for k in KEYS), flush=True)

    def _hu(self, path):
        return read_stored(path).astype(np.float32) - np.float32(self.opt.hu_offset)

    @torch.no_grad()
    def run(self, model, epoch, total_iters):
        """epoch 末（checkpoint 保存の後）に呼ぶ。戻り値 (score, is_best)。enabled でなければ (None, False)。"""
        if not self.enabled:
            return None, False
        t0 = time.time()
        o = self.opt
        rng = np.random.default_rng(self.np_rng_seed + int(epoch))
        net = model.netG
        was_training = net.training
        net.eval()  # 採点は BN の running 統計で（monitor と同じ）
        rows = []; tiles_x, tiles_y = [], []
        dn = Path(o.dn_dir) if getattr(o, "dn_dir", None) else None  # gen_mode: mean_noise は G の入力が BM3D(x)（2026-11-05）
        for p in self.pcd_paths:
            src = dn / Path(p).relative_to(o.dir_A) if dn is not None else p
            x = torch.from_numpy(normalize(read_stored(src), o.hu_offset, o.hu_min, o.hu_max))[None, None].to(model.device)
            if hasattr(net, "noise_like"):  # mean_noise（Codex レビュー 3 の 4）: 伝達・コヒーレンスは μ、見た目の統計（local SD・NPS・縁）はノイズ付きの出力。ノイズはスライスで決まる seed
                from models.eid_noise import slice_seed
                mu_t = net.net(x).clone().float()
                y = (mu_t + net.noise_like(mu_t, seeds=[slice_seed(Path(p).relative_to(o.dir_A))])).cpu().numpy()[0, 0]
                y_tr = mu_t.cpu().numpy()[0, 0]
            else:
                y = net(x).clone().float().cpu().numpy()[0, 0]  # clone: compile 時の CUDA Graph 出力の上書き対策
                y_tr = y
            xn = x.float().cpu().numpy()[0, 0]
            for ty in range(0, xn.shape[0] - 127, 128):  # 帯域の伝達・コヒーレンス用の 128 タイル（学習の patch と同じ大きさ）
                for tx in range(0, xn.shape[1] - 127, 128):
                    tiles_x.append(xn[ty:ty + 128, tx:tx + 128]); tiles_y.append(y_tr[ty:ty + 128, tx:tx + 128])
            hu = (y + 1.0) / 2.0 * (o.hu_max - o.hu_min) + o.hu_min
            rows.append(analyze_slice(hu, self.thr, rng))
        # EID 側: 実 EID を G に通したときの体内の差 RMS(G(y) − y)（恒等項の効き。identity のオン・オフどちらでも測る。score には入れない = オン・オフで比較できる）
        sq = []
        for p in ([] if hasattr(net, "noise_like") else self.eid_paths[:32]):  # mean_noise では実 EID を G に通す恒等の診断は意味を持たない（入力は BM3D(PCD) だけ）
            hu_y = self._hu(p)
            yn = torch.from_numpy(normalize(read_stored(p), o.hu_offset, o.hu_min, o.hu_max))[None, None].to(model.device)
            gy = net(yn).clone().float().cpu().numpy()[0, 0]
            gy_hu = (gy + 1.0) / 2.0 * (o.hu_max - o.hu_min) + o.hu_min
            m = body_mask(hu_y, self.thr)
            d = (gy_hu - np.clip(hu_y, o.hu_min, o.hu_max))[m]
            sq.append(float((d ** 2).mean()) if d.size else float("nan"))
        ry = float(np.sqrt(np.nanmean(sq))) if sq else float("nan")
        net.train(was_training)
        gen = {k: summarize(rows, k) for k in KEYS}
        spec_a, spec_g, spec_p = [], [], []
        if getattr(o, "spec_edges", None) and tiles_x:  # 入力 PCD と出力の帯域ごとの伝達 a・コヒーレンス γ²（漏れの監視。score には入れない。2026-11-04）
            from models.spectral_loss import band_transfer
            tx_ = torch.from_numpy(np.stack(tiles_x))[:, None]; ty_ = torch.from_numpy(np.stack(tiles_y))[:, None]
            av, gv, pv = band_transfer(tx_, ty_, parse_floats(o.spec_edges, "spec_edges"), window=getattr(o, "spec_window", "hann"))
            spec_a, spec_g, spec_p = [float(v) for v in av], [float(v) for v in gv], [float(v) for v in pv]
            for i, (va, vg, vp) in enumerate(zip(spec_a, spec_g, spec_p)):
                self.tb.add_scalar(f"val/spec_a{i + 1}", va, total_iters); self.tb.add_scalar(f"val/spec_g{i + 1}", vg, total_iters); self.tb.add_scalar(f"val/spec_p{i + 1}", vp, total_iters)
        dist = {k: abs(gen[k]["median"] - self.ref[k]["median"]) / self.ref[k]["iqr"] if np.isfinite(gen[k]["median"]) else float("nan") for k in KEYS}
        if getattr(o, "score_spec", False) and spec_a:  # 帯域の目標からの距離（2026-11-05。6 指標だけでは縁のぼけを見落とし best_e020 を選んだ）
            w_ = parse_floats(o.spec_weights, "spec_weights"); at_ = parse_floats(o.spec_a_target, "spec_a_target"); gt_ = parse_floats(o.spec_g2_target, "spec_g2_target")
            on = [i for i, wi in enumerate(w_) if wi > 0]
            dist["spec_a"] = float(np.mean([abs(spec_a[i] - at_[i]) for i in on])) / 0.05
            if not hasattr(net, "noise_like"):  # mean_noise の μ は s̃ の決定的な関数なので γ² は 1 に近く、目標との比較に意味がない（伝達 a だけで選ぶ）
                dist["spec_g"] = float(np.mean([abs(spec_g[i] - gt_[i]) for i in on])) / 0.10
            self.tb.add_scalar("val/spec_a_dist", dist["spec_a"], total_iters)
            if "spec_g" in dist:
                self.tb.add_scalar("val/spec_g_dist", dist["spec_g"], total_iters)
        score = float(sum(dist.values()))
        is_best = bool(np.isfinite(score) and score < self.best_score)
        if is_best:
            self.best_score, self.best_epoch = score, int(epoch)
        for k in KEYS:
            self.tb.add_scalar(f"val/{k}_gen", gen[k]["median"], total_iters)
            self.tb.add_scalar(f"val/{k}_eid", self.ref[k]["median"], total_iters)
            self.tb.add_scalar(f"val/{k}_dist", dist[k], total_iters)
        self.tb.add_scalar("val/ry_rms_HU", ry, total_iters)  # 実 EID の体内 RMS(G(y) − y)
        self.tb.add_scalar("val/score", score, total_iters)
        self.tb.add_scalar("val/best_score", self.best_score, total_iters)
        line = (f"val epoch {int(epoch)} | score {score:.3f}{' *best*' if is_best else ''} | ry_rms {ry:.1f} HU | "
                + (("spec a " + "/".join(f"{v:.3f}" for v in spec_a) + " γ² " + "/".join(f"{v:.2f}" for v in spec_g) + " P " + "/".join(f"{v:.3f}" for v in spec_p) + " | ") if spec_a else "")
                + " ".join(f"{k} {gen[k]['median']:.2f} (eid {self.ref[k]['median']:.2f}, d {dist[k]:.2f})" for k in KEYS)
                + (f" spec_a_d {dist['spec_a']:.2f}" if "spec_a" in dist else "") + (f" spec_g_d {dist['spec_g']:.2f}" if "spec_g" in dist else "") + f" | {time.time() - t0:.1f}s")
        with open(self.run_dir / VAL_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        print(line, flush=True)
        if is_best and o.auto_best:
            tag = best_dirname(epoch)
            model.save_networks(tag)  # <run>/best_eNNN/（原子的）
            retire_other_bests(self.run_dir, self.run_dir / tag)
            with open(self.run_dir / BEST_NOTE, "w", encoding="utf-8") as f:
                f.write(f"epoch: {int(epoch)}\ndir: {tag}\nscore: {score:.4f}\nmarked_at: {time.strftime('%Y-%m-%dT%H:%M')}\nsource: auto (util/validator.py, R-S22 の 6 指標)\n"
                        + "".join(f"{k}: gen {gen[k]['median']:.3f} eid {self.ref[k]['median']:.3f} dist {dist[k]:.3f}\n" for k in KEYS) + f"ry_rms_HU: {ry:.2f}\n")
        return score, is_best
