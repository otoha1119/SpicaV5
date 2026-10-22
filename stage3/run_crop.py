"""Stage 3 の TensorBoard 固定プラーク領域を checkpoint 指定で単独実行する起動器。"""

import argparse
import datetime
import os
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from configs.schema import CROP, MACHINE, TRAIN, ConfigError, apply_overrides, check_crop_values, to_argv, validate, validate_shared  # noqa: E402
from run_infer import DISPLAY_KEYS, load_yaml, run_config, upsample_config  # noqa: E402
from util.run_paths import CKPT_EMA, CKPT_MODEL, LAUNCH_FILE, crop_output_dir, find_run_dir, latest_launch  # noqa: E402


def split_overrides(tokens):
    path, rest, i = None, [], 0
    while i < len(tokens):
        if tokens[i] == "--weight_dir":
            if i + 1 >= len(tokens) or tokens[i + 1].startswith("--"):
                raise ConfigError("--weight_dir には値が必要です")
            path, i = tokens[i + 1], i + 2
        else:
            rest.append(tokens[i]); i += 1
    return path, rest


def main():
    p = argparse.ArgumentParser(description="Stage 3 固定プラーク crop 起動器")
    p.add_argument("--machine", required=True); p.add_argument("--crop", required=True)
    p.add_argument("--train", required=True); p.add_argument("--machines", required=True)
    p.add_argument("--weight_dir", required=True); p.add_argument("--out_root", default=None)
    p.add_argument("overrides", nargs=argparse.REMAINDER)
    a = p.parse_args()
    overrides = a.overrides[1:] if a.overrides and a.overrides[0] == "--" else a.overrides
    try:
        crop = validate("crop", load_yaml(a.crop), CROP)
        train = validate("train", load_yaml(a.train), TRAIN)
        machines = load_yaml(a.machines)
        if not isinstance(machines, dict) or a.machine not in machines:
            raise ConfigError(f"machines.yaml に '{a.machine}' がありません")
        machine = validate_shared(f"machines.{a.machine}", machines[a.machine], MACHINE)
        weight_over, rest = split_overrides(overrides)
        applied = apply_overrides(rest, {"crop": (crop, CROP)})
        weight_dir = Path(weight_over or a.weight_dir)
        for name in (CKPT_MODEL, CKPT_EMA):
            if not (weight_dir / name).is_file(): raise ConfigError(f"{name} がありません: {weight_dir}")
        run_dir = find_run_dir(weight_dir)
        if run_dir is None: raise ConfigError(f"{LAUNCH_FILE} を持つ run が見つかりません: {weight_dir}")
        launch_path = latest_launch(run_dir)
        ddim, hu = run_config(load_yaml(launch_path), launch_path)
        scale, interp, image_size = upsample_config(run_dir)
        pcd_cases, eid_cases = train["log.crop_pcd"], train["log.crop_eid"]
        check_crop_values(crop, train, machine, ddim, len(pcd_cases), len(eid_cases))
        selected_pcd = [(crop["pcd_index"], pcd_cases[crop["pcd_index"] - 1])]
        selected_eid = [(len(pcd_cases) + crop["eid_index"], eid_cases[crop["eid_index"] - 1])]
        for k in ("eidlike1024_dir", "pcd1024_dir", "eid_dir"):
            if not Path(machine[k]).is_dir(): raise ConfigError(f"machines.yaml の {k} がありません: {machine[k]}")
        display = {k: train[k] for k in DISPLAY_KEYS}
    except ConfigError as e:
        print(f"[run_crop] 設定エラー: {e}", file=sys.stderr); sys.exit(2)

    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9), name="JST"))
    out_dir = crop_output_dir(run_dir, weight_dir, now, a.out_root)
    if out_dir.exists(): print(f"[run_crop] 出力先が既にあります: {out_dir}", file=sys.stderr); sys.exit(2)
    out_dir.mkdir(parents=True)
    argv = ["--weight_dir", str(weight_dir), "--launch", str(launch_path), "--out_dir", str(out_dir),
            "--eidlike1024_dir", machine["eidlike1024_dir"], "--pcd1024_dir", machine["pcd1024_dir"], "--eid_dir", machine["eid_dir"],
            "--hu_offset", str(hu["data.hu_offset"]), "--hu_min", str(hu["data.hu_min"]), "--hu_max", str(hu["data.hu_max"]),
            "--scale", str(scale), "--interp", interp, "--image_size", str(image_size),
            "--crop_size", str(train["log.crop_size"]),
            "--display_hu_min", str(display["log.display_hu_min"]), "--display_hu_max", str(display["log.display_hu_max"]), "--preview_bits", str(display["log.preview_bits"])]
    argv += to_argv(crop, CROP)
    for tag, c in selected_pcd: argv += ["--pcd_case", f"{tag:02d}", c["slice"], str(c["x"]), str(c["y"])]
    for tag, c in selected_eid: argv += ["--eid_case", f"{tag:02d}", c["slice"], str(c["x"]), str(c["y"])]
    with open(out_dir / "crop.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"timestamp": now.isoformat(timespec="seconds"), "machine_name": a.machine, "run_dir": str(run_dir), "launch": str(launch_path),
                        "weight_dir": str(weight_dir), "crop": crop, "crop_size": train["log.crop_size"], "pcd_cases": pcd_cases, "eid_cases": eid_cases,
                        "selected_pcd": selected_pcd, "selected_eid": selected_eid,
                        "ddim": ddim, "hu": hu, "upsample": {"scale": scale, "interp": interp, "image_size": image_size}, "display": display,
                        "overrides": overrides, "applied": applied, "argv": argv}, f, allow_unicode=True, sort_keys=False)
    print(f"[run_crop] {weight_dir} -> {out_dir}（全引数は crop.yaml）", flush=True)
    os.chdir(HERE); os.execv(sys.executable, [sys.executable, str(HERE / "crop_patches.py")] + argv)


if __name__ == "__main__": main()
