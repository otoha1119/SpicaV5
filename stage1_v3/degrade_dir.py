"""stage1_v3/degrade_dir.py — P 単独（学習なし）で PCD512 → EID-like512 を生成する。推薦の最初の対照（レビュー I: P-only で R-S22 の校正と R-S27 の漏れ量を先に測る）。

  bash start1_v3.sh <machine> ponly [--input_dir DIR] [--out_name NAME] [--max_slices N] [--seed N] [--blur_min .. 等 degrade.* の上書き]
  P の範囲は configs/train.yaml の degrade.*（schema の --flag で上書き）。出力 <repo>/output/stage1_v3/ponly_<入力名>_<時刻>/full/<case>/<slice>.png と manifest.yaml
  （P の設定とスライスごとの σ_b, σ_n, c）。Stage 2 の dataset（bash start2.sh dataset）の入力にそのまま使える。
"""

import argparse
import datetime
import sys
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from configs.schema import MACHINE, TRAIN, ConfigError, apply_overrides, check_degrade_values, validate, validate_shared  # noqa: E402
from data.dataset import CaseIndex, read_stored, stored_to_hu, write_png  # noqa: E402
from degrade import apply_degrade, degrade_config, sample_params  # noqa: E402
from util.run_paths import STAGE, checkpoints_root, ponly_output_dir  # noqa: E402

LOCAL_FLAGS = ("--input_dir", "--out_name", "--max_slices", "--seed")


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


def main():
    p = argparse.ArgumentParser(description=f"{STAGE} P 単独生成")
    p.add_argument("--machine", required=True)
    p.add_argument("--train", required=True)
    p.add_argument("--machines", required=True)
    p.add_argument("--input_dir", required=True)
    p.add_argument("--out_root", default=None)
    p.add_argument("overrides", nargs=argparse.REMAINDER)
    a = p.parse_args()
    overrides = a.overrides[1:] if a.overrides and a.overrides[0] == "--" else a.overrides
    try:
        with open(a.train, encoding="utf-8") as f:
            train = validate("train", yaml.safe_load(f), TRAIN)
        with open(a.machines, encoding="utf-8") as f:
            machines = yaml.safe_load(f)
        if not isinstance(machines, dict) or a.machine not in machines:
            raise ConfigError(f"machines.yaml にエントリ '{a.machine}' がありません")
        machine = validate_shared(f"machines.{a.machine}", machines[a.machine], MACHINE)
        local, rest = split_local(overrides)
        applied = apply_overrides(rest, {"train": (train, TRAIN)})
        bad = [k for k in applied if not k.lstrip("-").startswith(("blur", "noise", "corr", "body", "degrade", "hu_"))]
        if bad:
            raise ConfigError(f"ponly で上書きできるのは degrade.* と data.hu_* だけです: {bad}")
        P = []
        check_degrade_values(train, P)
        if P:
            raise ConfigError(" / ".join(P))
        if not train["degrade.enabled"]:
            raise ConfigError("degrade.enabled が false（P が無い）。--degrade true にしてください")
        input_dir = Path(local.get("--input_dir", a.input_dir))
        if not input_dir.is_dir():
            raise ConfigError(f"入力フォルダがありません: {input_dir}")
        max_slices = int(local.get("--max_slices", 0))
        seed = int(local.get("--seed", 0))
    except (ConfigError, ValueError, KeyError, TypeError) as e:
        print(f"[ponly] 設定エラー: {e}", file=sys.stderr)
        sys.exit(2)

    jst = datetime.timezone(datetime.timedelta(hours=9), name="JST")
    now = datetime.datetime.now(jst)
    out_dir = ponly_output_dir(input_dir, now, root=a.out_root)
    if "--out_name" in local:
        out_dir = out_dir.with_name(local["--out_name"])
    if out_dir.exists():
        print(f"[ponly] 設定エラー: 出力先が既に存在します: {out_dir}", file=sys.stderr)
        sys.exit(2)
    out_dir.mkdir(parents=True)
    pcfg = degrade_config(train)
    offset = train["data.hu_offset"]
    idx = CaseIndex(input_dir, checkpoints_root(machine) / ".case_index")
    paths = idx.all_paths if max_slices == 0 else idx.all_paths[:max_slices]
    rng = np.random.default_rng(seed)
    print(f"[ponly] {input_dir} ({idx.n_cases} cases, {len(paths)} slices) → {out_dir} | blur {pcfg['blur']} px, noise {pcfg['noise']} HU, corr {pcfg['corr']} px, body_only {pcfg['body_only']}")
    manifest = {"stage": STAGE, "kind": "ponly", "timestamp": now.isoformat(timespec="seconds"), "input_dir": str(input_dir), "seed": seed,
                "degrade": {k: (list(v) if isinstance(v, tuple) else v) for k, v in pcfg.items()}, "applied": applied, "slices": []}
    for pth in tqdm(paths, unit="slice", dynamic_ncols=True):
        rel = Path(pth).relative_to(input_dir)
        hu = stored_to_hu(read_stored(pth), offset)
        prm = sample_params(pcfg, rng)
        u = apply_degrade(hu, prm, pcfg, rng)
        write_png(out_dir / "full" / rel, np.clip(np.round(u + offset), 0, 65535).astype(np.uint16))
        manifest["slices"].append({"slice": str(rel), **prm})
    with open(out_dir / "manifest.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, allow_unicode=True, sort_keys=False)
    print(f"[ponly] done: {len(paths)} slices -> {out_dir}")


if __name__ == "__main__":
    main()
