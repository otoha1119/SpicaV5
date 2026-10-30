"""stage1_v3/measure_dir.py — R-S22 の計測（学習なし）。複数フォルダ（実 EID / PCD512 / EID-like）の 512 スライスで次を並べる。

  (a) ノイズ: 体内の「一様 ROI」（gradient energy が最小の 32×32 窓、各スライス上位 n_roi 個）の局所 SD [HU] と、その ROI の 1D radial NPS（HU²·px²）の形（f_peak, f_av [cycles/px]）
  (b) 鮮鋭度の代理: 体内の強いエッジ（|∇| 上位 1 %）での勾配の大きさ [HU/px]（線計測 eval/line_metrics.py の edge 10–90 % 幅の代わり。参考値）
  (c) HU: 体内（HU > body_threshold）の p50 / p90 / p99、空気（体外）の平均

  bash start1_v3.sh <machine> stats --dirs A,B,C [--max_slices 200] [--seed 0]
  出力: <repo>/output/stage1_v3/stats_<時刻>/stats.yaml（フォルダごとの中央値と四分位）と summary.txt（横並びの表）
  degrade.* の範囲を決める手順: EID の局所 SD² − PCD の局所 SD² の平方根が noise_sigma_hu の中心、NPS の f_peak の比が corr、エッジ勾配の比が blur の目安。
"""

import argparse
import datetime
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from configs.schema import MACHINE, TRAIN, ConfigError, validate, validate_shared  # noqa: E402
from data.dataset import CaseIndex, read_stored, stored_to_hu  # noqa: E402
from util.metrics import ROI, N_ROI, analyze_slice, summarize  # noqa: E402
from util.run_paths import STAGE, checkpoints_root, stats_output_dir  # noqa: E402

def main():
    p = argparse.ArgumentParser(description=f"{STAGE} R-S22 計測")
    p.add_argument("--machine", required=True)
    p.add_argument("--train", required=True)
    p.add_argument("--machines", required=True)
    p.add_argument("--out_root", default=None)
    p.add_argument("overrides", nargs=argparse.REMAINDER)
    a = p.parse_args()
    overrides = a.overrides[1:] if a.overrides and a.overrides[0] == "--" else a.overrides
    opts = {"--dirs": None, "--max_slices": "200", "--seed": "0"}
    i = 0
    try:
        while i < len(overrides):
            t = overrides[i]
            if t not in opts or i + 1 >= len(overrides):
                raise ConfigError(f"stats の引数は --dirs A,B,C [--max_slices N] [--seed N] です: {t}")
            opts[t] = overrides[i + 1]
            i += 2
        if not opts["--dirs"]:
            raise ConfigError("--dirs を指定してください（カンマ区切り）")
        with open(a.train, encoding="utf-8") as f:
            train = validate("train", yaml.safe_load(f), TRAIN)
        with open(a.machines, encoding="utf-8") as f:
            machines = yaml.safe_load(f)
        if not isinstance(machines, dict) or a.machine not in machines:
            raise ConfigError(f"machines.yaml にエントリ '{a.machine}' がありません")
        machine = validate_shared(f"machines.{a.machine}", machines[a.machine], MACHINE)
        dirs = [Path(d) for d in opts["--dirs"].split(",") if d]
        for d in dirs:
            if not d.is_dir():
                raise ConfigError(f"フォルダがありません: {d}")
        max_slices, seed = int(opts["--max_slices"]), int(opts["--seed"])
    except (ConfigError, ValueError) as e:
        print(f"[stats] 設定エラー: {e}", file=sys.stderr)
        sys.exit(2)
    thr = float(train["degrade.body_threshold_hu"])
    offset = train["data.hu_offset"]
    jst = datetime.timezone(datetime.timedelta(hours=9), name="JST")
    now = datetime.datetime.now(jst)
    out_dir = stats_output_dir(now, root=a.out_root)
    out_dir.mkdir(parents=True)
    keys = ("local_sd", "nps_f_peak", "nps_f_av", "edge_grad_p99", "edge_grad_top1_mean", "hu_p50", "hu_p90", "hu_p99", "air_mean")
    result = {"stage": STAGE, "timestamp": now.isoformat(timespec="seconds"), "body_threshold_hu": thr, "roi": ROI, "n_roi_per_slice": N_ROI, "dirs": {}}
    rng = np.random.default_rng(seed)
    for d in dirs:
        idx = CaseIndex(d, checkpoints_root(machine) / ".case_index")
        paths = list(idx.all_paths)
        if max_slices and len(paths) > max_slices:
            paths = [paths[i] for i in sorted(rng.choice(len(paths), max_slices, replace=False))]
        rows = []
        for pth in paths:
            hu = stored_to_hu(read_stored(pth), offset)
            rows.append(analyze_slice(hu, thr, rng))
        valid = [r for r in rows if r is not None]
        nps_mean = np.mean([r["nps_radial"] for r in valid], axis=0).tolist() if valid else []
        result["dirs"][str(d)] = {"n_slices": len(paths), "n_valid": len(valid), **{k: summarize(valid, k) for k in keys}, "nps_radial_mean": nps_mean}
        print(f"[stats] {d}: {len(valid)}/{len(paths)} slices")
    with open(out_dir / "stats.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(result, f, allow_unicode=True, sort_keys=False)
    names = [Path(d).name for d in dirs]
    lines = ["metric".ljust(22) + "".join(n[:22].rjust(24) for n in names)]
    for k in keys:
        lines.append(k.ljust(22) + "".join(f"{result['dirs'][str(d)][k]['median']:12.2f} [{result['dirs'][str(d)][k]['q25']:.1f},{result['dirs'][str(d)][k]['q75']:.1f}]".rjust(24) for d in dirs))
    txt = "\n".join(lines)
    with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt)
    print(f"[stats] -> {out_dir}")


if __name__ == "__main__":
    main()
