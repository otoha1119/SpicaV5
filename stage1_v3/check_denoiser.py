"""stage1_v3/check_denoiser.py — v4 の Dn 単体の合否（docs/decisions/20261031_stage1-v3-noise-path-analysis.md §6.4）。dncheck。

  bash start1_v3.sh <machine> dncheck [--weights PATH] [--method n2n|blindspot|bm3d] [--psd_path P] [--input_dir DIR] [--n_slices N] [--out_name NAME]
    --weights   : net_Dn.pth（学習型 n2n / blindspot）。省略時は configs/train.yaml の pipeline.dn_weights（v4 学習がそのまま使う重み）を見る（どちらを見たか表示する）
    --method    : denoise.yaml の dn.method の上書き。bm3d なら --weights は不要（指定するとエラー）で、bm3d.psd_path の PSD で固定 denoiser（BM3DDenoiser）を同じ合否表に掛ける
    --input_dir : PCD512 のフォルダ（= data.input_dir の上書き。省略時は configs/denoise.yaml の値）
    --n_slices  : 測るスライス数（= check.n_slices の上書き。症例列挙から等間隔）
  閾値は configs/denoise.yaml の check.*。--input_dir / --n_slices を含め schema（DENOISE）のフラグは summary.yaml の applied に残る。
  計測は FP32（bf16 の量子化段差 5〜11 HU は 6 HU の閾値に対して無視できない。util/dn_check.py）。
  出力: <repo>/output/stage1_v3/dncheck_<時刻>/summary.yaml（method、Dn の素性、スライスごとの値と中央値、合否）と <slice>_panel.png（[x | Dn(x) | x − Dn(x)]、最初の 4 枚）。
        表の下に「比較:」1 行（method と中央値を横並び。blindspot の重み vs bm3d を見比べる）
  終了コード: 全項目 PASS なら 0、それ以外 1（pipeline_v4.sh はここで止まる）。
"""

import argparse
import datetime
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from configs.schema import DENOISE, MACHINE, TRAIN, ConfigError, apply_overrides, check_denoise_values, validate, validate_shared  # noqa: E402
from data.dataset import CaseIndex, read_stored, stored_to_hu, write_png  # noqa: E402
from data.denoise_dataset import evenly_spaced  # noqa: E402
from models.denoiser import BM3DDenoiser, load_denoiser  # noqa: E402
from util.dn_check import compare_line, evaluate_denoiser, format_table, judge, make_denoise_fn  # noqa: E402
from util.panel import build_panel  # noqa: E402
from util.residual_color import residual_rgb01, rgb_to_bgr  # noqa: E402
from util.run_paths import STAGE, checkpoints_root, dncheck_output_dir  # noqa: E402

LOCAL_FLAGS = ("--weights", "--out_name")  # --input_dir / --n_slices は schema（DENOISE）の上書きとして applied に残る
N_PANELS = 4


def split_local(tokens):
    local, rest, i = {}, [], 0
    while i < len(tokens):
        t = tokens[i]
        if t in LOCAL_FLAGS:
            if i + 1 >= len(tokens) or tokens[i + 1].startswith("--"):
                raise ConfigError(f"{t} には値が必要です")
            local[t] = tokens[i + 1]
            i += 2
            continue
        rest.append(t)
        i += 1
    return local, rest


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def panel_image(x, xc, resid, dn, label):
    lo, hi, rng = dn["log.display_hu_min"], dn["log.display_hu_max"], dn["log.diff_range_hu"]
    gray = lambda hu: np.repeat(np.clip((hu - lo) / float(hi - lo), 0.0, 1.0)[..., None], 3, axis=2).astype(np.float32)  # noqa: E731
    panel = build_panel([gray(x), gray(xc), residual_rgb01(resid, rng)], ["PCD512  x", f"Dn(x)   {label}", "x - Dn(x)   [HU]"], rng)
    return rgb_to_bgr((np.clip(panel, 0, 1) * 65535.0).round().astype(np.uint16))


def main():
    p = argparse.ArgumentParser(description=f"{STAGE} Dn 単体の合否")
    p.add_argument("--machine", required=True)
    p.add_argument("--denoise", required=True)
    p.add_argument("--train", required=True)
    p.add_argument("--machines", required=True)
    p.add_argument("--out_root", default=None)
    p.add_argument("overrides", nargs=argparse.REMAINDER)
    a = p.parse_args()
    overrides = a.overrides[1:] if a.overrides and a.overrides[0] == "--" else a.overrides
    try:
        with open(a.denoise, encoding="utf-8") as f:
            dn = validate("denoise", yaml.safe_load(f), DENOISE)
        with open(a.machines, encoding="utf-8") as f:
            machines = yaml.safe_load(f)
        if not isinstance(machines, dict) or a.machine not in machines:
            raise ConfigError(f"machines.yaml にエントリ '{a.machine}' がありません")
        machine = validate_shared(f"machines.{a.machine}", machines[a.machine], MACHINE)
        local, rest = split_local(overrides)
        applied = apply_overrides(rest, {"denoise": (dn, DENOISE)})
        check_denoise_values(dn, machine)
        method = dn["dn.method"]
        weights, source = None, None
        if method == "bm3d":
            if "--weights" in local:
                raise ConfigError("--method bm3d に --weights は指定できません（固定 denoiser。PSD は bm3d.psd_path / --psd_path）")
            if not Path(dn["bm3d.psd_path"]).is_file():
                raise ConfigError(f"bm3d.psd_path がありません: {dn['bm3d.psd_path']}（bash start1_v3.sh <machine> stats --dirs <PCD512> の noise_psd_<名前>.npy）")
        elif "--weights" in local:
            weights, source = Path(local["--weights"]), "--weights"
        else:
            with open(a.train, encoding="utf-8") as f:
                train = validate("train", yaml.safe_load(f), TRAIN)
            if not train["pipeline.dn_weights"]:
                raise ConfigError("--weights が無く、configs/train.yaml の pipeline.dn_weights も空です")
            weights, source = Path(train["pipeline.dn_weights"]), "configs/train.yaml pipeline.dn_weights"
        if weights is not None and not weights.is_file():
            raise ConfigError(f"net_Dn.pth がありません（{source}）: {weights}")
        input_dir = Path(dn["data.input_dir"])
        if not input_dir.is_dir():
            raise ConfigError(f"入力フォルダがありません: {input_dir}")
    except (ConfigError, ValueError) as e:
        print(f"[dncheck] 設定エラー: {e}", file=sys.stderr)
        sys.exit(2)

    device = torch.device("cuda") if (machine["gpu_gen"] != 0 and torch.cuda.is_available()) else torch.device("cpu")
    if machine["gpu_gen"] != 0 and device.type != "cuda":
        raise RuntimeError("machines.yaml の gpu_gen ≠ 0 なのに CUDA が使えません")
    jst = datetime.timezone(datetime.timedelta(hours=9), name="JST")
    now = datetime.datetime.now(jst)
    out_dir = dncheck_output_dir(now, root=a.out_root)
    if "--out_name" in local:
        out_dir = out_dir.with_name(local["--out_name"])
    out_dir.mkdir(parents=True)
    offset = dn["data.hu_offset"]
    hu_norm = (offset, dn["data.hu_min"], dn["data.hu_max"])
    if method == "bm3d":
        fixed = BM3DDenoiser(dn["bm3d.psd_path"], dn["bm3d.profile"], dn["data.hu_min"], dn["data.hu_max"], dn["bm3d.sigma_scale"])
        denoise, label = fixed, f"bm3d {dn['bm3d.profile']} x{dn['bm3d.sigma_scale']:g}"
        dn_desc = {"method": "bm3d", **fixed.info()}
        dn_text = f"bm3d profile {dn['bm3d.profile']} sigma_scale {dn['bm3d.sigma_scale']:g} psd {fixed.psd_path}（白色換算 σ {fixed.sigma_white_hu:.1f} HU）"
    else:
        net = load_denoiser(weights, device)
        denoise, label = make_denoise_fn(net, device, hu_norm), weights.parent.name
        dn_desc = {"method": method, "weights": str(weights), "weights_sha256": sha256(weights), "weights_source": source}
        dn_text = f"weights {weights}（{source}）"
    idx = CaseIndex(input_dir, checkpoints_root(machine) / ".case_index")
    paths = evenly_spaced(idx.all_paths, dn["check.n_slices"])
    print(f"[dncheck] method {method} | {dn_text} | {input_dir}: {idx.n_cases} cases, {len(paths)}/{idx.n_slices} slices | device {device}")
    hus = [stored_to_hu(read_stored(p), offset) for p in paths]
    rows, summary, images = evaluate_denoiser(denoise, hus, hu_norm, float(dn["check.body_threshold_hu"]), seed=dn["data.patch_seed"], return_images=True)
    res = judge(summary, dn)
    table = format_table(res)
    for i, (pth, (x, xc, resid)) in enumerate(zip(paths, images)):
        if i >= N_PANELS:
            break
        write_png(out_dir / f"{Path(pth).stem}_panel.png", panel_image(x, xc, resid, dn, label))
    cmp = compare_line(method, label, summary)
    with open(out_dir / "summary.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"stage": STAGE, "kind": "dncheck", "timestamp": now.isoformat(timespec="seconds"), "method": method, "dn": dn_desc,
                        "weights": (str(weights) if weights else None), "weights_sha256": (sha256(weights) if weights else None), "weights_source": source,
                        "input_dir": str(input_dir), "n_slices": len(paths), "applied": applied, "thresholds": {k: dn[k] for k in dn if k.startswith("check.")},
                        "summary": summary, "judge": {k: {"value": v, "cond": c, "pass": ok} for k, (v, c, ok) in res.items()}, "all_pass": all(ok for _, _, ok in res.values()),
                        "compare": cmp,
                        "slices": [{"slice": str(Path(p).relative_to(input_dir)), **({k: float(v) for k, v in r.items()} if r else {"skipped": True})} for p, r in zip(paths, rows)]},
                       f, allow_unicode=True, sort_keys=False)
    print(table)
    print(f"[dncheck] {cmp}")
    print(f"[dncheck] -> {out_dir}")
    sys.exit(0 if all(ok for _, _, ok in res.values()) else 1)


if __name__ == "__main__":
    main()
