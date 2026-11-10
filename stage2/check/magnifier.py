"""stage2/check/magnifier.py — 「虫眼鏡の確認」の数字（設計書 G-1〜G-3）。学習中の epoch 末（util/monitor.py）と確認コマンド（run_check.py）の両方がこれを使う。

同じ重みに 3 つの組を通し、各組・各段で同じ数字を出して並べる:
  組: pseudo（val 症例の疑似 EID k00）/ imagecas（実 EID、ImageCAS）/ hospital（病院 EID。評価専用 = 見るだけ）/ pcd（参照。val の正解 PCD1024 そのもの。段は ref）
  段: in（補間した入力 x↑）/ mu（U の出力）/ out（U + V の最終出力。V があるとき）/ old（比べる相手の出力フォルダがあるとき）
  数字: fine_sd（ノイズ感）、edge（縁の鋭さ）、band1〜5（粒の細かさ。pcd の組との比）、shift_hu（広い平均が入力から動いた量 = G∗(絵 − x↑) の体内 RMS）
合格線は無い。見るのは「実 EID の出力が、疑似 EID の出力と入力のどちらに近いか」。

正解がある組（pseudo）では、あわせて次を出す:
  rmse_mu / rmse_input / rmse_out   フル 1024 で μ 対 X、x↑ 対 X、y 対 X（n の単位。旧 Stage 2 の val/rmse と同じ計算。入力そのままの値に近いほど「正直」。設計書 B-4）
  leak_noise_explained              U が正解のノイズをどれだけ当てているか（G-2 の 3）
  leak_corr_fine                    最終出力の細かい成分と正解の細かい成分の相関（G-3 の 3。0 に近い = V のノイズは正解のノイズと無関係）
  var_gap                           RMS(μ(k00) − μ(k01)) ÷ RMS(x↑(k00) − x↑(k01))（G-2 の 4。K ≥ 2 のとき）
  lock_target_shift_hu              G∗(x↑ − X) の体内 RMS（Stage 1 の明るさの錨のずれ）
  lock_f_shift_hu                   G∗(x↑ − f) の体内 RMS（U の素の出力の広い平均が流れていないか）
  lock_corr_rms_hu                  最後に差し替えた量 G∗(x↑ − Z1) の体内 RMS（V が広い平均を動かそうとしているか）
"""

import math

import torch

from models.lock import wide_mean
from models.pipeline import cpu_noise
from util import metrics as M

STAGES = ("in", "mu", "out", "old", "ref")
GROUPS = ("pseudo", "imagecas", "hospital", "pcd")


class Stopped(Exception):
    """確認の途中で止める合図（SIGTERM を受けた。学習ループが latest を保存して終わる）。"""


def _mean(vals):
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    return sum(vals) / len(vals) if vals else float("nan")


class MagTable:
    def __init__(self, tile, sigma_px):
        self.tile, self.sigma_px, self.acc = tile, sigma_px, {}

    def add(self, stage, group, img, mask, x=None):
        a = self.acc.setdefault((stage, group), {"fine_sd": [], "edge": [], "band": [], "shift_hu": []})
        a["fine_sd"].append(M.fine_sd(img, mask))
        a["edge"].append(M.edge_ratio(img, mask))
        b = M.band_power(img, mask, self.tile)
        if b is not None:
            a["band"].append(b.cpu())
        if x is not None:
            a["shift_hu"].append(M.rms(wide_mean(img - x, self.sigma_px), mask))

    def result(self):
        """{(段, 組): {"n", "fine_sd", "edge", "shift_hu", "band1"…"band5"（pcd との比。pcd の組が無ければ出さない）}}。"""
        ref = self.acc.get(("ref", "pcd"))
        ref_band = torch.stack(ref["band"]).mean(dim=0) if ref and ref["band"] else None
        out = {}
        for (stage, group), a in self.acc.items():
            r = {"n": len(a["fine_sd"]), "fine_sd": _mean(a["fine_sd"]), "edge": _mean(a["edge"])}
            if a["shift_hu"]:
                r["shift_hu"] = _mean(a["shift_hu"])
            if a["band"] and ref_band is not None:
                ratio = torch.stack(a["band"]).mean(dim=0) / ref_band.clamp_min(1e-12)
                for i in range(M.N_BANDS):
                    r[f"band{i + 1}"] = float(ratio[i])
            out[(stage, group)] = r
        return out

    def scalars(self, prefix="diag/mag/"):
        out = {}
        for (stage, group), r in self.result().items():
            for k, v in r.items():
                if k != "n" and math.isfinite(v):
                    out[f"{prefix}{k}_{stage}_{group}"] = v
        return out

    def markdown(self, title, extra=None):
        res = self.result()
        groups = [g for g in GROUPS if any(k[1] == g for k in res)]
        stages = [s for s in STAGES if any(k[0] == s for k in res)]
        lines = [f"# {title}", "", "数字に合格線は無い（どちらに近いかを見る材料）。段: in = 補間した入力、mu = U の出力、out = U + V、old = 比べる相手、ref = 正解 PCD1024。", ""]
        if extra:
            lines += [f"- {k}: {v:.5g}" if isinstance(v, float) else f"- {k}: {v}" for k, v in extra.items()] + [""]
        stats = ["fine_sd", "edge", "shift_hu"] + [f"band{i + 1}" for i in range(M.N_BANDS)]
        notes = {"fine_sd": "細かい成分の局所 SD の体内の中央値（HU）", "edge": "縁で σ 1 の勾配 ÷ σ 3 の勾配", "shift_hu": "G∗(絵 − 入力) の体内 RMS（HU）"}
        for st in stats:
            if not any(st in r for r in res.values()):
                continue
            band = st.startswith("band")
            lo, hi = (M.BAND_EDGES[int(st[4:]) - 1], M.BAND_EDGES[int(st[4:])]) if band else (None, None)
            lines += [f"## {st}（{notes.get(st) or f'{lo}〜{hi} 周期/画素のパワー ÷ pcd'}）", "", "| 段 | " + " | ".join(groups) + " |", "|---|" + "---|" * len(groups)]
            for s in stages:
                cells = [f"{res[(s, g)][st]:.4g}" if (s, g) in res and st in res[(s, g)] and math.isfinite(res[(s, g)][st]) else "—" for g in groups]
                if any(c != "—" for c in cells):
                    lines.append(f"| {s} | " + " | ".join(cells) + " |")
            lines.append("")
        lines.append("枚数: " + ", ".join(f"{s}/{g} {res[(s, g)]['n']}" for s in stages for g in groups if (s, g) in res))
        return "\n".join(lines) + "\n"


@torch.no_grad()
def evaluate(pipe, io, slices, mode, sigma_s_hu, steps, seed, tile, body_hu, should_stop=None, progress=None):
    """slices: [{"group", "case", "name", "x": 512 の入力のパス, "t": 正解のパス | None, "x_alt": k01 のパス | None, "mag": bool, "old": 旧出力のパス | None}, ...]
    io: load_x(path)（補間した n）・load_t(path)（1024 の n）を持つもの（data.pair_dataset.PairData か同じ形のもの）。
    mode: "u"（μ まで）| "uv"。戻り値 (scalars の dict, MagTable)。"""
    un, sig = pipe.units, pipe.sigma_px
    table = MagTable(tile, sig)
    acc = {}

    def put(k, v):
        if v is not None and math.isfinite(v):
            acc.setdefault(k, []).append(v)

    gap_num = gap_den = 0.0
    for i, s in enumerate(slices):
        if should_stop is not None and should_stop():
            raise Stopped()
        x = io.load_x(s["x"]).to(pipe.device)
        mu, f = pipe.mu(x)
        y = z1 = None
        if mode == "uv":
            eps = cpu_noise(x.shape[-2:], seed, s["case"], s["name"])[None, None]
            y, z1 = pipe.flow(mu, x, eps, sigma_s_hu, steps)
        xh, mh = un.hu(x), un.hu(mu)
        yh = un.hu(y) if y is not None else None
        mask = M.body_mask(xh, sig, body_hu)
        g = s["group"]
        if s["mag"]:
            table.add("in", g, xh, mask)
            table.add("mu", g, mh, mask, xh)
            if yh is not None:
                table.add("out", g, yh, mask, xh)
            if s.get("old"):
                table.add("old", g, un.hu(io.load_t(s["old"]).to(pipe.device)), mask, xh)
        put(f"lock_f_shift_hu_{g}", M.rms(wide_mean(un.hu(x) - un.hu(f), sig), mask))
        put(f"lock_out_shift_hu_{g}", M.rms(wide_mean((yh if yh is not None else mh) - xh, sig), mask))
        if z1 is not None:
            put(f"lock_corr_rms_hu_{g}", M.rms(wide_mean(xh - un.hu(z1), sig), mask))
        if s.get("t"):
            t = io.load_t(s["t"]).to(pipe.device)
            th = un.hu(t)
            if s["mag"]:
                table.add("ref", "pcd", th, mask)
            put("mse_mu", float((mu - t).square().mean()))
            put("mse_input", float((x - t).square().mean()))
            put("lock_target_shift_hu", M.rms(wide_mean(xh - th, sig), mask))
            put("leak_noise_explained", M.noise_explained(th, mh, mask))
            if yh is not None:
                put("mse_out", float((y - t).square().mean()))
                put("leak_corr_fine", M.corr(M.fine(yh), M.fine(th), mask))
        if s.get("x_alt"):
            x2 = io.load_x(s["x_alt"]).to(pipe.device)
            mu2, _ = pipe.mu(x2)
            m = mask.float()
            gap_num += float(((mh - un.hu(mu2)).square() * m).sum())
            gap_den += float(((xh - un.hu(x2)).square() * m).sum())
        if progress is not None:
            progress(i + 1, len(slices))
    out = {}
    for k, v in acc.items():
        out["r" + k if k.startswith("mse_") else k] = (_mean(v) ** 0.5) if k.startswith("mse_") else _mean(v)
    if gap_den > 0:
        out["var_gap"] = (gap_num / gap_den) ** 0.5
    return out, table


def build_slices(pd, eid_dir, n_val, n_mag, eid_cases, hospital_dirs, with_alt, old_dir=None):
    """確認に通すスライスの一覧を作る。pseudo = val 症例の k00（症例ごとに等間隔 n_val 枚。うち等間隔 n_mag 枚が虫眼鏡の対象）、
    imagecas = eid_dir/<症例> の等間隔 n_mag 枚、hospital = 症例フォルダの等間隔 n_mag 枚。with_alt で、虫眼鏡の対象のスライスに k01（var_gap 用。K ≥ 2 のとき）を付ける。"""
    from pathlib import Path

    from data.pair_dataset import even_pick, list_pngs

    def old(case, name):
        p = Path(old_dir) / case / name if old_dir else None
        return str(p) if p is not None and p.is_file() else None

    out = []
    for c in pd.val_cases:
        vs = even_pick(pd.pairs[c], n_val)
        mag = {n for n, _ in even_pick(vs, n_mag)}
        for name, pcd in vs:
            out.append({"group": "pseudo", "case": c, "name": name, "x": pd.pseudo_path(0, c, name), "t": pcd,
                        "x_alt": pd.pseudo_path(1, c, name) if with_alt and pd.k > 1 and name in mag else None, "mag": name in mag, "old": old(c, name)})
    for group, dirs in (("imagecas", [Path(eid_dir) / c for c in eid_cases]), ("hospital", [Path(d) for d in hospital_dirs])):
        for d in dirs:
            for p in even_pick(list_pngs(d), n_mag):
                out.append({"group": group, "case": d.name, "name": p.name, "x": p, "t": None, "x_alt": None, "mag": True, "old": old(d.name, p.name)})
    return out
