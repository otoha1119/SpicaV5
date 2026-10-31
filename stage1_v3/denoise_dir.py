"""stage1_v3/denoise_dir.py — 固定 Dn（学習なし）で PCD512 を先に denoise し、v4 の学習（pipeline.dn_source precomputed）と推論が読む形で書き出す。dnfixed。

  bash start1_v3.sh <machine> dnfixed [--input_dir DIR] [--max_slices N] [--workers K] [--out_name NAME] [--psd_path P 等 configs/schema.py DENOISE の上書き]
    方式は configs/denoise.yaml の dn.method（bm3d のみ。n2n / blindspot は学習型なので bash start1_v3.sh <machine> dn）。
    bm3d: models/denoiser.BM3DDenoiser（PyPI bm3d に bm3d.psd_path の実測 PSD を渡す。PSD は bash start1_v3.sh <machine> stats --dirs <入力> が書く noise_psd_<名前>.npy）
    --input_dir : 省略時は denoise.yaml の data.input_dir。--max_slices 0 = 全部。--workers = multiprocessing のプロセス数（0 = 直列。BM3D は CPU で数秒 / 512²）
  出力: <repo>/output/stage1_v3/dnfixed_<入力名>_<時刻>/full/<case>/<slice>.png（uint16、stored = Dn(x) + hu_offset、入力と同じ相対パス）と
        manifest.yaml（method、PSD のパスと sha256、profile、HU 窓、スライス一覧）。学習: --pipeline_mode v4 --dn_source precomputed --dn_dir <この出力> --dn_weights ""
        合否: bash start1_v3.sh <machine> dncheck --method bm3d（同じ PSD で固定 denoiser を掛け直して測る）
"""

import argparse
import datetime
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from configs.schema import DENOISE, MACHINE, ConfigError, apply_overrides, check_denoise_values, validate, validate_shared  # noqa: E402
from data.dataset import CaseIndex, read_stored, stored_to_hu, write_png  # noqa: E402
from models.denoiser import BM3DDenoiser  # noqa: E402
from util.run_paths import STAGE, checkpoints_root, dnfixed_output_dir  # noqa: E402

LOCAL_FLAGS = ("--input_dir", "--out_name", "--max_slices", "--workers")
_FIXED = None


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


def _init(psd_path, profile, hu_min, hu_max, sigma_scale):
    global _FIXED
    _FIXED = BM3DDenoiser(psd_path, profile, hu_min, hu_max, sigma_scale)


def _work(args):
    """1 スライス: 読む → Dn → 書く。戻り値 (rel, RMS(x − Dn(x)) [HU], 秒)。"""
    src, dst, rel, offset = args
    t = time.time()
    hu = stored_to_hu(read_stored(src), offset)
    out = _FIXED(hu)
    write_png(dst, np.clip(np.round(out + offset), 0, 65535).astype(np.uint16))
    return rel, float(np.sqrt(((np.clip(hu, _FIXED.hu_min, _FIXED.hu_max) - out) ** 2).mean())), time.time() - t


def main():
    p = argparse.ArgumentParser(description=f"{STAGE} 固定 Dn でデータを denoise")
    p.add_argument("--machine", required=True)
    p.add_argument("--denoise", required=True)
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
        if dn["dn.method"] != "bm3d":
            raise ConfigError(f"dnfixed は固定 denoiser（dn.method bm3d）専用です（今は {dn['dn.method']!r}。学習型は bash start1_v3.sh <machine> dn）。--method bm3d で上書きできる")
        if not Path(dn["bm3d.psd_path"]).is_file():
            raise ConfigError(f"bm3d.psd_path がありません: {dn['bm3d.psd_path']}（bash start1_v3.sh <machine> stats --dirs <入力> の noise_psd_<名前>.npy）")
        input_dir = Path(local.get("--input_dir", dn["data.input_dir"]))
        if not input_dir.is_dir():
            raise ConfigError(f"入力フォルダがありません: {input_dir}")
        max_slices, workers = int(local.get("--max_slices", 0)), int(local.get("--workers", machine["num_threads"]))
        if max_slices < 0 or workers < 0:
            raise ConfigError("--max_slices / --workers ≥ 0")
    except (ConfigError, ValueError) as e:
        print(f"[dnfixed] 設定エラー: {e}", file=sys.stderr)
        sys.exit(2)

    jst = datetime.timezone(datetime.timedelta(hours=9), name="JST")
    now = datetime.datetime.now(jst)
    out_dir = dnfixed_output_dir(input_dir, now, root=a.out_root)
    if "--out_name" in local:
        out_dir = out_dir.with_name(local["--out_name"])
    if out_dir.exists():
        print(f"[dnfixed] 設定エラー: 出力先が既に存在します: {out_dir}", file=sys.stderr)
        sys.exit(2)
    offset, hu_min, hu_max = dn["data.hu_offset"], dn["data.hu_min"], dn["data.hu_max"]
    fixed = BM3DDenoiser(dn["bm3d.psd_path"], dn["bm3d.profile"], hu_min, hu_max, dn["bm3d.sigma_scale"])  # 設定の検査（PSD の形）はここで 1 回
    idx = CaseIndex(input_dir, checkpoints_root(machine) / ".case_index")
    paths = idx.all_paths if max_slices == 0 else idx.all_paths[:max_slices]
    out_dir.mkdir(parents=True)
    print(f"[dnfixed] bm3d profile {dn['bm3d.profile']} sigma_scale {dn['bm3d.sigma_scale']} | PSD {fixed.psd_path} {fixed.psd.shape}（白色換算 σ {fixed.sigma_white_hu:.1f} HU, sha256 {fixed.psd_sha256[:12]}…）"
          f" | {input_dir} ({idx.n_cases} cases, {len(paths)}/{idx.n_slices} slices) → {out_dir} | workers {workers}", flush=True)
    jobs = [(pth, out_dir / "full" / Path(pth).relative_to(input_dir), str(Path(pth).relative_to(input_dir)), offset) for pth in paths]
    init_args = (dn["bm3d.psd_path"], dn["bm3d.profile"], hu_min, hu_max, dn["bm3d.sigma_scale"])
    results, t0 = [], time.time()
    bar = tqdm(total=len(jobs), unit="slice", dynamic_ncols=True)
    try:
        if workers == 0:
            _init(*init_args)
            for j in jobs:
                results.append(_work(j)); bar.update(1)
        else:
            with mp.get_context("spawn").Pool(workers, initializer=_init, initargs=init_args) as pool:
                for r in pool.imap(_work, jobs, chunksize=1):
                    results.append(r); bar.update(1)
    finally:
        bar.close()
    rms = [r[1] for r in results]
    manifest = {"stage": STAGE, "kind": "dnfixed", "timestamp": now.isoformat(timespec="seconds"), "input_dir": str(input_dir), "n_slices": len(results),
                "dn": fixed.info(), "denoise_config": {k: v for k, v in dn.items() if k.startswith(("dn.", "bm3d.", "data.hu_"))}, "applied": applied,
                "dn_resid_rms_HU_mean": (float(np.mean(rms)) if rms else float("nan")), "sec_per_slice": (float(np.mean([r[2] for r in results])) if results else float("nan")),
                "slices": [{"slice": r[0], "dn_resid_rms_HU": r[1]} for r in results]}
    with open(out_dir / "manifest.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, allow_unicode=True, sort_keys=False)
    el = time.time() - t0
    print(f"[dnfixed] done: {len(results)} slices, {el:.0f}s ({el / max(len(results), 1):.1f} s/slice wall) | RMS(x − Dn(x)) {manifest['dn_resid_rms_HU_mean']:.1f} HU -> {out_dir}")
    print(f"[dnfixed] 学習: bash start1_v3.sh <machine> --pipeline_mode v4 --dn_source precomputed --dn_dir {out_dir} --dn_weights \"\"")


if __name__ == "__main__":
    main()
