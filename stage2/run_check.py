"""stage2/run_check.py — 確認の道具（設計書 G。Docker 内で動く。ホストからは bash start2.sh check --kind v --run <run> [--flag value]）。

  python stage2/run_check.py --machine PC1 --check stage2/configs/check.yaml --machines configs/machines.yaml -- --kind v --run 2026_1113_1000 --what magnifier,seeds,flicker

  magnifier（G-1〜G-3「虫眼鏡の確認」）: 同じ重みに pseudo（val の疑似 EID k00）/ imagecas（実 EID）/ hospital（病院 EID）を通し、入力・μ・最終出力の
            fine_sd / edge / band1〜5 / shift_hu を並べた表（magnifier_s<σs>.md）と数字（numbers.yaml）。V は check.sigma_s_list の σs ごとに出す（学習し直さない）。
            見るのは「実 EID の出力が、疑似 EID の出力と入力のどちらに近いか」。合格線は無い。
  seeds    （G-3 の 4・5）: seeds.slices の各スライスで乱数の種を check.n_seeds 通り。最初の 2 つの差 [seed a | seed b | 差 | ぼかした差]（切り抜き）と、
            全部の画素ごとの平均・標準偏差の絵。fake_seed_sd_hu / fake_seed_low_ratio。V の重みのときだけ。
  flicker  （G-4）: flicker.series の連続 flicker.n_slices 枚を通し、中央の行・列で縦に切った帯（スライスごとに別の乱数 / 全スライス同じ乱数）と adj_corr。
  出力: <run>/check/<重み>_<時刻>/（check.out_root で変えられる）。ネットワーク・データの決まりは重みの state.pth の実効設定から。U・明るさ・V の入口と出口は FP32。
  表示・確認に使う実 EID が eid_split_file の train 側（Stage 1 の P が学んだ症例）なら起動前にエラー（RECONCILE #12）。
"""

import argparse
import datetime
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from check import flicker as FL  # noqa: E402
from check.magnifier import build_slices, evaluate  # noqa: E402
from check.seed_diff import diff_row, mean_std, sample_seeds, seed_stats  # noqa: E402
from configs.schema import CHECK, MACHINE, ConfigError, apply_overrides, check_check_values, parse_seed_slice, parse_series, validate  # noqa: E402
from data.ct_io import n_to_stored, write_png  # noqa: E402
from data.pair_dataset import PairData, check_eid_not_train, list_pngs  # noqa: E402
from models.pipeline import load_pipeline  # noqa: E402
from run_infer import find_weights  # noqa: E402
from util import metrics as M  # noqa: E402
from util.launcher import cache_dir, load_machine, load_yaml  # noqa: E402
from util.panel import build_grid, gray, to_u8  # noqa: E402
from util.run_paths import JST, check_dir  # noqa: E402
from util.views import View, crop_zoom, to_np  # noqa: E402


def save_panel(path, panel01):
    write_png(path, np.ascontiguousarray(to_u8(panel01)[..., ::-1]))


def do_magnifier(pipe, pd, c, machine, data, mode, sigmas, out, numbers):
    slices = build_slices(pd, machine["eid_dir"], c["mag.slices_per_case"], c["mag.slices_per_case"], c["mag.eid_cases"], c["mag.hospital_dirs"],
                          with_alt=True, old_dir=c["check.old_output_dir"] or None)
    for s in sigmas:
        t0 = time.time()
        sc, table = evaluate(pipe, pd, slices, mode, s, c["check.steps"], c["check.seed"], c["mag.tile"], data["display.body_hu"])
        name = "magnifier" + (f"_s{s:g}" if s is not None else "")
        extra = {"mode": mode, **({"sigma_s_hu": s, "steps": c["check.steps"]} if s is not None else {}), **sc}
        (out / f"{name}.md").write_text(table.markdown(f"{out.parent.parent.name} {out.name} {name}", extra), encoding="utf-8")
        numbers[name] = {**sc, **table.scalars(prefix="")}
        print(f"[check] {name}: {len(slices)} 枚、{time.time() - t0:.0f} s → {out / (name + '.md')}", flush=True)


def do_seeds(pipe, pd, c, data, sigmas, out, numbers):
    view, un = View(data), pipe.units
    seeds = [c["check.seed"] + i for i in range(c["check.n_seeds"])]
    cs, ck = c["seeds.crop_size"], c["seeds.crop_scale"]
    for text in c["seeds.slices"]:
        kind, path, cy, cx = parse_seed_slice(text)
        p = pd.set_dir / "k00" / path if kind == "pseudo" else Path(path)
        tp = pd.pcd_dir / path if kind == "pseudo" and (pd.pcd_dir / path).is_file() else None
        x = pd.load_x(p).to(pipe.device)
        mu, _ = pipe.mu(x)
        xh, mh = un.hu(x), un.hu(mu)
        th = un.hu(pd.load_t(tp).to(pipe.device)) if tp else None
        mask = M.body_mask(xh, pipe.sigma_px, data["display.body_hu"])
        for s in sigmas:
            ys = [un.hu(y) for y in sample_seeds(pipe, x, mu, p.parent.name, p.name, seeds, s, c["check.steps"])]
            name = f"{p.stem}_s{s:g}"
            rows = [diff_row(view, ys[0], ys[1 % len(ys)], [cy, cx], cs, ck, f"{p.stem} s{s:g}HU")]
            mean, std = mean_std(ys) if len(ys) > 1 else (ys[0], torch.zeros_like(ys[0]))
            cols = [view.img(xh), view.img(mh), view.img(mean), gray(np.clip(to_np(std) / (view.diff_range / 2.0), 0.0, 1.0))]
            labels = ["input up", "mu = U", f"mean of {len(ys)} seeds", f"std of {len(ys)} seeds (0-{view.diff_range / 2:g}HU)"]
            rows.append(([crop_zoom(col, [cy, cx], cs, ck) for col in cols], labels))
            save_panel(out / "seeds" / f"{name}_crop.png", build_grid(rows))
            save_panel(out / "seeds" / f"{name}_full.png", build_grid([(cols + ([view.img(th)] if th is not None else []), labels + (["PCD1024 (target)"] if th is not None else []))]))
            write_png(out / "seeds" / f"{name}_mean.png", n_to_stored(to_np(un.n_from_z(mean / un.unit)), pd.hu))
            if len(ys) > 1:
                numbers[f"seeds/{name}"] = seed_stats(ys[0], ys[1], mask, c["mag.tile"], th, mh)
            print(f"[check] seeds {name}: {numbers.get(f'seeds/{name}')}", flush=True)


def do_flicker(pipe, pd, c, data, mode, sigma, out, numbers):
    view = View(data)
    for text in c["flicker.series"]:
        kind, path = parse_series(text)
        if kind == "pseudo":
            if path not in pd.pairs:
                raise RuntimeError(f"flicker.series の症例 {path} は train / val の症例ではありません（正解と組になっている症例だけ）")
            run_ = FL.middle_run(pd.pairs[path], c["flicker.n_slices"])
            paths, targets, case = [pd.pseudo_path(0, path, n) for n, _ in run_], [t for _, t in run_], path
        else:
            paths, targets, case = FL.middle_run(list_pngs(path), c["flicker.n_slices"]), None, Path(path).name
        vols = FL.run_series(pipe, pd, paths, case, mode, sigma, c["check.steps"], c["check.seed"], False, data["display.body_hu"])
        items = [("input up", vols["in"]), ("mu = U", vols["mu"])]
        if mode == "uv":
            same = FL.run_series(pipe, pd, paths, case, mode, sigma, c["check.steps"], c["check.seed"], True, data["display.body_hu"])
            items += [("U+V (noise per slice)", vols["out"]), ("U+V (same noise)", same["out"])]
        if targets:
            items.append(("PCD1024 (target)", pipe.units.hu(torch.stack([pd.load_t(t)[0, 0] for t in targets]))))
        numbers[f"flicker/{case}"] = {f"adj_corr/{label}": FL.adj_corr(vol, vols["mask"]) for label, vol in items}
        for axis, idx in (("row", 0), ("col", 1)):
            cols = [view.img(FL.strips(vol, c["flicker.stretch"])[idx]) for _, vol in items]
            save_panel(out / "flicker" / f"{case}_{axis}.png", build_grid([(cols, [l for l, _ in items])]))
        print(f"[check] flicker {case}（{len(paths)} 枚）: {numbers[f'flicker/{case}']}", flush=True)


def main():
    p = argparse.ArgumentParser(description="新 Stage 2 の確認の道具")
    p.add_argument("--machine", required=True)
    p.add_argument("--check", required=True)
    p.add_argument("--machines", required=True)
    p.add_argument("overrides", nargs=argparse.REMAINDER)
    a = p.parse_args()
    overrides = a.overrides[1:] if a.overrides and a.overrides[0] == "--" else a.overrides
    try:
        c = validate("check", load_yaml(a.check), CHECK)
        machine = load_machine(a.machines, a.machine)
        apply_overrides(overrides, {"check": (c, CHECK), "machine": (machine, MACHINE)})
        check_check_values(c, machine)
        kind = c["check.kind"]
        run_dir, wdir = find_weights(machine, kind, c["check.run"], c["check.tag"])
    except ConfigError as e:
        print(f"[check] 設定エラー: {e}", file=sys.stderr)
        sys.exit(2)
    if machine["gpu_gen"] != 0 and not torch.cuda.is_available():
        raise RuntimeError("machines.yaml の gpu_gen ≠ 0 なのに CUDA が使えません")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    pipe, cfg, step = load_pipeline(wdir, kind, device, v_bf16=(c["check.precision"] == "bf16"))
    data, shared = dict(cfg["data"]), cfg["shared"]
    if c["check.pseudo_eid_set"]:
        data["data.pseudo_eid_set"] = c["check.pseudo_eid_set"]
    what, mode = c["check.what"], ("uv" if kind == "v" else "u")
    try:
        eids = [(d, Path(d).name) for d in c["mag.hospital_dirs"]] + [(e, e) for e in c["mag.eid_cases"]]
        eids += [(t, Path(parse_seed_slice(t)[1]).parent.name) for t in c["seeds.slices"] if parse_seed_slice(t)[0] == "eid"] if "seeds" in what else []
        eids += [(t, Path(parse_series(t)[1]).name) for t in c["flicker.series"] if parse_series(t)[0] == "eid"] if "flicker" in what else []
        check_eid_not_train(data, eids, "確認に使う実 EID")
        pd = PairData(data, shared, machine, cache_dir(machine))
        sigmas = [None]
        if mode == "uv":
            lo, hi = cfg["v"]["flow.sigma_s_min_hu"], cfg["v"]["flow.sigma_s_max_hu"]
            sigmas = [float(s) for s in c["check.sigma_s_list"]]
            if any(not (lo <= s <= hi) for s in sigmas):
                raise RuntimeError(f"check.sigma_s_list {sigmas} に、V が学習した幅 [{lo}, {hi}] HU の外の値があります")
        now = datetime.datetime.now(JST)
        mark = f"{wdir.name}_{now.strftime('%Y_%m%d_%H%M%S')}"
        out = Path(c["check.out_root"]) / f"{run_dir.name}_{mark}" if c["check.out_root"] else check_dir(run_dir, mark)
        out.mkdir(parents=True)
        print(f"[check] {kind} の run {run_dir.name} / {wdir.name}（step {step}）、組 {pd.set_dir.name}、やること {what}" + (f"、σs {sigmas} HU" if mode == "uv" else "") + f" → {out}", flush=True)
        numbers = {}
        if "magnifier" in what:
            do_magnifier(pipe, pd, c, machine, data, mode, sigmas, out, numbers)
        if "seeds" in what:
            if mode == "uv":
                do_seeds(pipe, pd, c, data, sigmas, out, numbers)
            else:
                print("[check] seeds は V の重みのときだけ（U は乱数を使わない）。とばす", flush=True)
        if "flicker" in what:
            do_flicker(pipe, pd, c, data, mode, sigmas[len(sigmas) // 2] if mode == "uv" else None, out, numbers)
        with open(out / "numbers.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump({"created": now.isoformat(timespec="seconds"), "kind": kind, "run": run_dir.name, "weight_dir": str(wdir), "weight_step": step,
                            "pseudo_eid_set": pd.set_dir.name, "check": c, "numbers": numbers}, f, allow_unicode=True, sort_keys=False)
        print(f"[check] 完了: {out}", flush=True)
    except RuntimeError as e:
        print(f"[check] エラー: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
