"""Stage 2 の固定プラーク領域だけを推論する起動器（bash start2.sh crop）。"""

import argparse
import datetime
import os
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from configs.schema import CROP, MACHINE, TRAIN, ConfigError, apply_overrides, check_crop_cases, check_crop_values, to_argv, validate, validate_shared  # noqa: E402
from run_infer import DISPLAY_KEYS, load_yaml, net_config, upsample_config  # noqa: E402
from util.run_paths import LAUNCH_FILE, crop_output_dir, find_run_dir, latest_launch  # noqa: E402


def split_overrides(tokens):
    path, rest, i = None, [], 0
    while i < len(tokens):
        if tokens[i] == "--weight_dir":
            if i + 1 >= len(tokens) or tokens[i + 1].startswith("--"):
                raise ConfigError("--weight_dir には値が必要です")
            path, i = tokens[i + 1], i + 2
        else:
            rest.append(tokens[i])
            i += 1
    return path, rest


def main():
    p = argparse.ArgumentParser(description="Stage 2 局所 crop 起動器")
    p.add_argument("--machine", required=True)
    p.add_argument("--crop", required=True)
    p.add_argument("--train", required=True)
    p.add_argument("--machines", required=True)
    p.add_argument("--weight_dir", required=True)
    p.add_argument("--out_root", default=None)
    p.add_argument("overrides", nargs=argparse.REMAINDER)
    a = p.parse_args()
    overrides = a.overrides[1:] if a.overrides and a.overrides[0] == "--" else a.overrides
    try:
        raw = load_yaml(a.crop)
        if not isinstance(raw, dict):
            raise ConfigError(f"[crop] トップレベルが dict ではありません: {a.crop}")
        pcd_cases = check_crop_cases("pcd_cases", raw.pop("pcd_cases", None))
        eid_cases = check_crop_cases("eid_cases", raw.pop("eid_cases", None))
        if not pcd_cases and not eid_cases:
            raise ConfigError("[crop] pcd_cases / eid_cases の合計を1件以上にしてください")
        crop = validate("crop", raw, CROP)
        machines = load_yaml(a.machines)
        if not isinstance(machines, dict) or a.machine not in machines:
            raise ConfigError(f"machines.yaml に '{a.machine}' がありません")
        machine = validate_shared(f"machines.{a.machine}", machines[a.machine], MACHINE)
        weight_over, rest = split_overrides(overrides)
        applied = apply_overrides(rest, {"crop": (crop, CROP)})
        weight_dir = Path(weight_over or a.weight_dir)
        if not (weight_dir / "net_G.pth").is_file():
            raise ConfigError(f"net_G.pth がありません: {weight_dir}")
        run_dir = find_run_dir(weight_dir)
        if run_dir is None:
            raise ConfigError(f"{LAUNCH_FILE} を持つ run が見つかりません: {weight_dir}")
        launch_path = latest_launch(run_dir)
        mode, hu = net_config(load_yaml(launch_path), launch_path)
        scale, interp, image_size = upsample_config(run_dir)
        check_crop_values(crop, machine, image_size, mode["n_pool"], len(pcd_cases), len(eid_cases))
        selected_pcd = [(crop["pcd_index"], pcd_cases[crop["pcd_index"] - 1])]
        selected_eid = [(len(pcd_cases) + crop["eid_index"], eid_cases[crop["eid_index"] - 1])]
        for k in ("eidlike1024_dir", "pcd1024_dir", "eid_dir"):
            if not Path(machine[k]).is_dir():
                raise ConfigError(f"machines.yaml の {k} がありません: {machine[k]}")
        cur_train = validate("train", load_yaml(a.train), TRAIN)
        display = {k: cur_train[k] for k in DISPLAY_KEYS}
    except ConfigError as e:
        print(f"[run_crop] 設定エラー: {e}", file=sys.stderr)
        sys.exit(2)

    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9), name="JST"))
    out_dir = crop_output_dir(run_dir, weight_dir, now, a.out_root)
    if out_dir.exists():
        print(f"[run_crop] 出力先が既にあります: {out_dir}", file=sys.stderr)
        sys.exit(2)
    out_dir.mkdir(parents=True)
    argv = ["--weight_dir", str(weight_dir), "--out_dir", str(out_dir),
            "--eidlike1024_dir", machine["eidlike1024_dir"], "--pcd1024_dir", machine["pcd1024_dir"], "--eid_dir", machine["eid_dir"],
            "--arch", mode["arch"], "--base_ch", str(mode["base_ch"]), "--n_pool", str(mode["n_pool"]), "--final_act", mode["final_act"],
            "--hu_offset", str(hu["data.hu_offset"]), "--hu_min", str(hu["data.hu_min"]), "--hu_max", str(hu["data.hu_max"]),
            "--scale", str(scale), "--interp", interp, "--image_size", str(image_size),
            "--display_hu_min", str(display["log.display_hu_min"]), "--display_hu_max", str(display["log.display_hu_max"]), "--preview_bits", str(display["log.preview_bits"])]
    if mode["residual"]:
        argv.append("--residual")
    argv += to_argv(crop, CROP)
    for tag, c in selected_pcd:
        argv += ["--pcd_case", f"{tag:02d}", c["slice"], str(c["x"]), str(c["y"])]
    for tag, c in selected_eid:
        argv += ["--eid_case", f"{tag:02d}", c["slice"], str(c["x"]), str(c["y"])]
    with open(out_dir / "crop.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"timestamp": now.isoformat(timespec="seconds"), "machine_name": a.machine,
                        "run_dir": str(run_dir), "launch": str(launch_path), "weight_dir": str(weight_dir),
                        "crop": crop, "pcd_cases": pcd_cases, "eid_cases": eid_cases,
                        "selected_pcd": selected_pcd, "selected_eid": selected_eid, "net": mode, "hu": hu,
                        "upsample": {"scale": scale, "interp": interp, "image_size": image_size}, "display": display,
                        "overrides": overrides, "applied": applied, "argv": argv}, f, allow_unicode=True, sort_keys=False)
    print(f"[run_crop] {weight_dir} -> {out_dir}（全引数は crop.yaml）", flush=True)
    os.chdir(HERE)
    os.execv(sys.executable, [sys.executable, str(HERE / "crop_patches.py")] + argv)


if __name__ == "__main__":
    main()
