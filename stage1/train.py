"""stage1/train.py — 新 Stage 1（描き手 P: 実 EID だけで学ぶ拡散モデル）の学習ループ。**run_train.py 経由で起動する**（直接叩くと launch.yaml が無くて止まる）。

  python stage1/train.py --run_dir /workspace/runs/stage1/<run> --launch <run>/launch.yaml [--resume_state <重みディレクトリ>/state.pth] [--require_cuda]

  1. run_train.py が書いた launch.yaml（実効値）を読み、schema で再検証する（既定値なし）
  2. seed → device（gpu_gen ≠ 0 なら CUDA 必須）→ 症例の分割 → DataLoader（実 EID の train のパッチを無限に）→ Painter → 再開なら復元
  3. step のループ: 学習 1 step → scalar（log.print_steps ごと。最初は 1, 2, 4, …, 64 step 目にも）→ 学習パッチの絵 → val → 途中経過のパネル → 保存
  4. 止まり方:
       optim.max_steps / optim.max_hours（累計）に達した → latest を保存して終了コード 0
       SIGTERM / SIGINT                                  → いまの step を終えて latest を保存、終了コード 143（保存は原子的なので、途中で殺されても前の latest から再開できる）
       損失・勾配・val が NaN / Inf                      → 重みを更新・保存せずに終了コード 3（NaN の scalar は書かない）
  最初の 5 step と compile の初回は、かかった秒を 1 行ずつ出す（compile の間、何も出ない時間を作らない）。
"""

import argparse
import random
import signal
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from configs.schema import MACHINE, MODEL, SHARED, TRAIN, ConfigError, parse_cue, validate, validate_shared  # noqa: E402
from data.ct_io import hu_tuple  # noqa: E402
from data.eid_dataset import EidPatchStream, fixed_val_patches, resolve_split, worker_init  # noqa: E402
from diffusion.cue import draw_cue_params  # noqa: E402
from diffusion.edm import stratified_sigma  # noqa: E402
from models.painter import NonFiniteError, Painter  # noqa: E402
from util.monitor import Monitor  # noqa: E402
from util.run_paths import DATASET_INFO_FILE, ckpt_dir  # noqa: E402

EARLY_STEPS = (1, 2, 4, 8, 16, 32, 64)  # この起動の最初はここでも scalar を出す（最初の scalar を待たせない）


def build_val(split, train, hu):
    """val の固定の入力（CPU）: パッチ、σ（帯に均等）、ノイズ、度合い、手がかりのノイズ。log.val_seed で毎回同じ。"""
    big, names = fixed_val_patches(split["val_slices"], train, hu)
    g = torch.Generator().manual_seed(train["log.val_seed"])
    n, p = big.shape[0], train["data.patch_size"]
    sigma = stratified_sigma(n, train["diffusion.sigma_min"], train["diffusion.sigma_max"], g)
    blur_px, noise_x, has = draw_cue_params(n, train, torch.device("cpu"), g)
    return {"big": big, "sigma": sigma, "eps": torch.randn(n, 1, p, p, generator=g), "cue_noise": torch.randn(n, 1, p, p, generator=g),
            "blur_px": blur_px, "noise_x": noise_x, "has": has}, names


def stop_workers(it):
    """DataLoader の worker を即座に止める。worker は SIGTERM を無視するので（data/eid_dataset.py worker_init）、普通の片付けだと 1 つずつ時間切れを待って
    終了が数十秒遅れる（監視ツールは SIGTERM の 30 秒後に SIGKILL する）。保存が済んでから呼ぶ。"""
    for w in getattr(it, "_workers", []):
        try:
            w.kill()
        except Exception:
            pass


def main():
    p = argparse.ArgumentParser(description="新 Stage 1 学習ループ（run_train.py から exec される）")
    p.add_argument("--run_dir", required=True, help="run フォルダ（/workspace/runs/stage1/<run>）")
    p.add_argument("--launch", required=True, help="run_train.py が書いた実効設定（launch.yaml / launch_resume_*.yaml）")
    p.add_argument("--resume_state", default=None, help="再開に使う重みディレクトリの state.pth（同じ所の net.pth / ema.pth も読む）")
    p.add_argument("--require_cuda", action="store_true", help="CUDA が無ければ止める（gpu_gen ≠ 0 のとき run_train.py が付ける）")
    a = p.parse_args()

    launch = Path(a.launch)
    if not launch.is_file():
        print(f"[train] 設定エラー: launch.yaml がありません（run_train.py 経由で起動すること）: {launch}", file=sys.stderr)
        sys.exit(2)
    with open(launch, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    try:
        train = validate("train(launch)", cfg["train"], TRAIN)
        model = validate("model(launch)", cfg["model"], MODEL)
        shared = validate_shared("shared(launch)", cfg["shared"], SHARED)
        machine = validate_shared("machine(launch)", cfg["machine"], MACHINE)
        cues = [parse_cue(c) for c in train["log.preview_cues"]]
    except (ConfigError, KeyError, TypeError) as e:
        print(f"[train] 設定エラー: {launch}: {e}", file=sys.stderr)
        sys.exit(2)
    run_dir = Path(a.run_dir)
    hu = hu_tuple(train)

    seed = train["optim.seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.backends.cudnn.benchmark = True
    if device.type == "cuda":  # 専用メモリの上限。Windows（WSL）は超えると共有メモリにあふれて極端に遅くなるので、その前に OOM で止める
        torch.cuda.set_per_process_memory_fraction(train["optim.max_gpu_mem_fraction"], torch.cuda.current_device())
    if a.require_cuda and device.type != "cuda":
        raise RuntimeError("machines.yaml の gpu_gen ≠ 0 なのに CUDA が使えません（gpu_gen: 0 のマシン定義なら cpu で動く）")

    split = resolve_split(train, machine, cache_dir=Path(machine["painter_checkpoints_dir"]) / ".case_index")
    painter = Painter(train, model, shared, machine, device)
    if a.resume_state:
        painter.resume(a.resume_state)
    elif train["init.run"]:
        painter.init_from(ckpt_dir(Path(machine["painter_checkpoints_dir"]) / train["init.run"], train["init.tag"]))
    info = {"domain": train["data.domain"], "init_run": train["init.run"], "init_tag": train["init.tag"], "split_file": split["split_file"], "n_train_cases": len(split["train"]), "n_val_cases": len(split["val"]), "n_test_cases_unread": len(split["test"]),
            "n_train_slices": sum(len(v) for v in split["train_slices"].values()), "n_val_slices": sum(len(v) for v in split["val_slices"].values()),
            "unlisted_cases_added_to_train": split["unlisted"], "eval_only_dropped": split["eval_only_dropped"],
            "train_cases": split["train"], "val_cases": split["val"], "n_params": painter.n_params, "start_step": painter.step, "device": str(device)}

    def write_info():
        with open(run_dir / DATASET_INFO_FILE, "w", encoding="utf-8") as f:
            yaml.safe_dump(info, f, allow_unicode=True, sort_keys=False)

    val, val_names = build_val(split, train, hu)
    info["val_patches"] = len(val_names)
    write_info()
    monitor = Monitor(run_dir, train, shared, machine, hu, cues, device)
    monitor.write(f"[train] run {run_dir.name}: {painter.n_params / 1e6:.2f}M params, device {device}, precision {train['optim.precision']}（本体 bf16 = {painter.use_bf16}）, "
                  f"compile {painter.compile_enabled}（{train['optim.compile_mode']}、{train['optim.compile_after_steps']} step 後）, "
                  f"train {info['n_train_cases']} 症例 / {info['n_train_slices']} 枚, val {info['n_val_cases']} 症例, start step {painter.step}")

    bs = train["optim.batch_size"]
    dataset = EidPatchStream(split["train_slices"], train, hu, start_step=painter.step)
    loader = DataLoader(dataset, batch_size=bs, num_workers=machine["num_threads"], pin_memory=(device.type == "cuda"), drop_last=True, worker_init_fn=worker_init)

    term = {"sig": None}

    def on_term(signum, _frame):
        term["sig"] = signum
        print(f"[train] signal {signum} を受けた。いまの step を終えて latest を保存して止まる", flush=True)

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)

    t = train
    max_steps, max_hours = t["optim.max_steps"], t["optim.max_hours"]
    elapsed0, t_start = painter.elapsed, time.time()
    hours = lambda: (elapsed0 + time.time() - t_start) / 3600.0
    stats_left, s_n, s_sum, s_sq = t["data.stats_patches"], 0, 0.0, 0.0
    local, reason, first_compiled_step = 0, None, None
    it = iter(loader)
    while True:
        if painter.step >= max_steps:
            reason = f"optim.max_steps {max_steps} に達した"
            break
        if hours() >= max_hours:
            reason = f"optim.max_hours {max_hours} に達した"
            break
        if term["sig"] is not None:
            reason = f"signal {term['sig']}"
            break
        local += 1
        if painter.maybe_compile(local):
            first_compiled_step = local
            monitor.write(f"[train] step {painter.step + 1}: torch.compile（{t['optim.compile_mode']}）を始める。初回は数分かかる。この間 scalar は出ない")
        t0 = time.time()
        try:
            batch = next(it)
        except RuntimeError as e:  # worker が殺されたとき（SIGKILL など）
            if term["sig"] is not None:
                reason = f"signal {term['sig']}（DataLoader 停止: {e}）"
                break
            raise
        t1 = time.time()
        try:
            painter.train_step(batch)
        except NonFiniteError as e:
            monitor.write(f"[train] 異常終了: {e}。重みは更新・保存しない（最後の保存: {ckpt_dir(run_dir, 'latest')}）。lr を下げて再開するなら bash start.sh resume {run_dir.name} --lr 0.0001")
            monitor.close()
            stop_workers(it)
            sys.exit(3)
        step = painter.step
        painter.elapsed = elapsed0 + time.time() - t_start
        monitor.tick(t1 - t0)
        if local <= 5 or local == first_compiled_step or (first_compiled_step and local == first_compiled_step + 1):
            note = "（compile 込み）" if local == first_compiled_step else ""
            monitor.write(f"[train] step {step}: {time.time() - t0:.2f} s{note}（うちデータ待ち {t1 - t0:.2f} s）、起動から {time.time() - t_start:.0f} s")
        if stats_left > 0:
            x = batch.double()
            s_n += x.numel(); s_sum += float(x.sum()); s_sq += float((x * x).sum())
            stats_left -= batch.shape[0]
            if stats_left <= 0:
                mean = s_sum / s_n
                info["measured_x"] = {"patches": t["data.stats_patches"], "mean": mean, "std": (s_sq / s_n - mean * mean) ** 0.5, "note": "x = HU / hu_scale。確認用（設定には使わない。diffusion.sigma_data と比べる）"}
                write_info()
                monitor.write(f"[train] 学習データ {t['data.stats_patches']} パッチの x: 平均 {mean:.4f}、標準偏差 {info['measured_x']['std']:.4f}（sigma_data {t['diffusion.sigma_data']}）")
        if step % t["log.print_steps"] == 0 or local in EARLY_STEPS:
            sec = monitor.log_train(step, painter, hours())
            if step % t["log.print_steps"] == 0:
                info["measured_speed"] = {"step": step, "sec_per_step": sec, "imgs_per_sec": bs / max(sec, 1e-9), "compiled": painter.compiled, "bf16": painter.use_bf16}
        t_side = time.time()
        if step % t["log.train_patch_steps"] == 0:
            monitor.log_train_patch(step, painter.last)
        first = local == t["log.preview_first_step"]
        try:
            if step % t["log.val_steps"] == 0 or first:
                monitor.log_val(step, painter.evaluate(val))
        except NonFiniteError as e:
            monitor.write(f"[train] 異常終了: {e}。latest は上書きしない")
            monitor.close()
            stop_workers(it)
            sys.exit(3)
        if step % t["log.preview_steps"] == 0 or first:
            try:
                monitor.preview(painter, step)
            except Exception:  # 途中経過の絵が失敗しても学習は止めない（夜間に 1 枚の絵のために止まらないように。ログに残す）
                monitor.write(f"[preview] step {step} の途中経過に失敗（学習は続ける）:\n{traceback.format_exc()}")
        if step % t["log.save_weights_steps"] == 0:
            painter.save(run_dir, step)
            monitor.write(f"[save] step {step} → {ckpt_dir(run_dir, step)}")
        if step % t["log.save_latest_steps"] == 0 or first:
            painter.save(run_dir, "latest")
            write_info()
            monitor.write(f"[save] step {step} → latest")
        monitor.t_window += time.time() - t_side  # 絵・val・保存の時間は sec_per_step に入れない

    painter.elapsed = elapsed0 + time.time() - t_start
    saved = painter.save(run_dir, "latest") if local > 0 else True
    write_info()
    monitor.write(f"[train] 終了: {reason}。step {painter.step}、累計 {painter.elapsed / 3600:.2f} 時間。latest {'を保存した' if saved else 'の保存に失敗した'}: {ckpt_dir(run_dir, 'latest')}"
                  f"（続きは bash start.sh resume {run_dir.name}）")
    monitor.close()
    stop_workers(it)
    sys.exit(143 if term["sig"] is not None else 0)


if __name__ == "__main__":
    main()
