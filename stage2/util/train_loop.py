"""stage2/util/train_loop.py — U・V の学習ループの本体（train_u.py / train_v.py が呼ぶ）。作りは stage1/train.py と同じ。

  1. 起動器が書いた launch.yaml（実効値）を読み、schema で再検証する（既定値なし）
  2. seed → device（gpu_gen ≠ 0 なら CUDA 必須）→ 疑似 EID の組と正解の突き合わせ → 司令塔（UModel / VModel）→ 再開なら復元 → DataLoader（パッチを無限に）
  3. step のループ: 学習 1 step → scalar（log.print_steps ごと。最初は 1, 2, 4, …, 64 step 目にも）→ 学習パッチの絵 → 保存 → epoch 末の確認（val・虫眼鏡・パネル）
     epoch 末は「latest と weights の保存 → 確認」の順（確認の途中で止められても、その epoch の重みは残っている）
  4. 止まり方:
       optim.max_steps / optim.max_hours（累計）に達した → latest と best（rule: last = 最後の EMA。自動では選ばない。設計書 B-4・C-5）を保存して終了コード 0
       SIGTERM / SIGINT                                  → いまの step（確認の途中ならそのスライス）を終えて latest を保存、終了コード 143
       損失・勾配・val が NaN / Inf                      → 重みを更新・保存せずに終了コード 3（NaN の scalar は書かない）
  最初の 5 step と compile の初回は、かかった秒を 1 行ずつ出す。速度の実測（diag/sec_per_step）は dataset_info.yaml にも残す。
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

from check.magnifier import Stopped
from configs.schema import DATA, MACHINE, SHARED, U, V, ConfigError, validate, validate_shared
from data.pair_dataset import PatchStream, check_before_launch, fixed_val_patches, worker_init
from models.trainer import NonFiniteError
from util.monitor import Monitor
from util.run_paths import DATASET_INFO_FILE, ckpt_dir, write_best_note

EARLY_STEPS = (1, 2, 4, 8, 16, 32, 64)  # この起動の最初はここでも scalar を出す（最初の scalar を待たせない）
SCHEMAS = {"u": U, "v": V}


def finish(it, code):
    """DataLoader の worker を即座に止めて、このプロセスを終える（worker は SIGTERM を無視するので、普通の片付けだと 1 つずつ時間切れを待って終了が数十秒遅れる。
    監視ツールは SIGTERM の 30 秒後に SIGKILL する）。保存とログを閉じ終えてから呼ぶ。os._exit で抜けるのは、殺した worker を DataLoader の後始末が待ち直さないようにするため。"""
    import os
    for w in getattr(it, "_workers", []):
        try:
            w.kill()
        except Exception:
            pass
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(code)


def build_model(kind, a, cfg_all, data, cfg, shared, machine, device):
    if kind == "u":
        from models.u_model import UModel
        return UModel(data, cfg, shared, machine, device)
    from models.v_model import VModel
    if a.resume_state:  # 再開: V の重みディレクトリに同梱した U の写しと実効設定を使う
        d = Path(a.resume_state).parent
        st = torch.load(a.resume_state, map_location="cpu", weights_only=True)
        u_config, u_source = st["config"]["u_config"], st["config"]["u_source"]
        weights = torch.load(d / "net_U.pth", map_location=device, weights_only=True)
    else:
        d = Path(cfg_all["u_weight_dir"])
        st = torch.load(d / "state.pth", map_location="cpu", weights_only=True)
        u_config, u_source = {k: dict(st["config"][k]) for k in ("data", "u", "shared")}, cfg_all["u_source"]
        weights = torch.load(d / "net_U_ema.pth", map_location=device, weights_only=True)
    return VModel(data, cfg, shared, machine, u_config, weights, u_source, device)


def main(kind):
    p = argparse.ArgumentParser(description=f"新 Stage 2 の {kind.upper()} の学習ループ（run_train_{kind}.py から exec される）")
    p.add_argument("--run_dir", required=True, help=f"run フォルダ（/workspace/runs/stage2/{kind}/<run>）")
    p.add_argument("--launch", required=True, help="起動器が書いた実効設定（launch.yaml / launch_resume_*.yaml）")
    p.add_argument("--resume_state", default=None, help="再開に使う重みディレクトリの state.pth")
    p.add_argument("--require_cuda", action="store_true", help="CUDA が無ければ止める（gpu_gen ≠ 0 のとき起動器が付ける）")
    a = p.parse_args()
    tag = f"train_{kind}"

    launch = Path(a.launch)
    if not launch.is_file():
        print(f"[{tag}] 設定エラー: launch.yaml がありません（run_train_{kind}.py 経由で起動すること）: {launch}", file=sys.stderr)
        sys.exit(2)
    with open(launch, encoding="utf-8") as f:
        cfg_all = yaml.safe_load(f)
    try:
        data = validate("data(launch)", cfg_all["data"], DATA)
        cfg = validate(f"{kind}(launch)", cfg_all[kind], SCHEMAS[kind])
        shared = validate_shared("shared(launch)", cfg_all["shared"], SHARED)
        machine = validate_shared("machine(launch)", cfg_all["machine"], MACHINE)
    except (ConfigError, KeyError, TypeError) as e:
        print(f"[{tag}] 設定エラー: {launch}: {e}", file=sys.stderr)
        sys.exit(2)
    run_dir = Path(a.run_dir)

    seed = cfg["optim.seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.backends.cudnn.benchmark = True
    if a.require_cuda and device.type != "cuda":
        raise RuntimeError("machines.yaml の gpu_gen ≠ 0 なのに CUDA が使えません（gpu_gen: 0 のマシン定義なら cpu で動く）")

    pd = check_before_launch(data, cfg, shared, machine, Path(machine["flow_checkpoints_dir"]) / ".case_index")
    model = build_model(kind, a, cfg_all, data, cfg, shared, machine, device)
    if a.resume_state:
        model.resume(a.resume_state)
    info = {**pd.summary(), "kind": kind, "n_params": model.n_params, "start_step": model.step, "device": str(device),
            "brightness_sigma_px_1024": model.sigma_px}
    if kind == "v":
        info["u_source"] = model.config["u_source"]
        info["n_params_u"] = sum(q.numel() for q in model.u_net.parameters())

    def write_info():
        with open(run_dir / DATASET_INFO_FILE, "w", encoding="utf-8") as f:
            yaml.safe_dump(info, f, allow_unicode=True, sort_keys=False)

    write_info()
    term = {"sig": None}

    def on_term(signum, _frame):
        term["sig"] = signum
        print(f"[{tag}] signal {signum} を受けた。いまの step を終えて latest を保存して止まる", flush=True)

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)

    monitor = Monitor(run_dir, kind, data, cfg, machine, pd, device, should_stop=lambda: term["sig"] is not None)
    margin = cfg["patch.margin"] if kind == "v" else 0
    val = model.build_val(fixed_val_patches(pd, cfg["patch.size"], margin, cfg["log.val_patches"], cfg["log.val_seed"]), cfg["log.val_seed"]) if kind == "v" else None
    monitor.write(f"[{tag}] run {run_dir.name}: {model.n_params / 1e6:.2f}M params, device {device}, precision {cfg['optim.precision']}（bf16 = {model.use_bf16}）, "
                  f"compile {model.compile_enabled}（{cfg['optim.compile_mode']}、{cfg['optim.compile_after_steps']} step 後）, 明るさの広さ σ {model.sigma_px:g} 画素（1024）, "
                  f"パッチ {cfg['patch.size']}{f' + 余白 {margin}' if margin else ''} × batch {cfg['optim.batch_size']}, 組 {pd.set_dir.name}（K {pd.k}）, start step {model.step}"
                  + (f", U = {model.config['u_source']}" if kind == "v" else ""))

    bs, spe = cfg["optim.batch_size"], cfg["optim.steps_per_epoch"]
    loader = DataLoader(PatchStream(pd, cfg["patch.size"], margin, seed, model.step), batch_size=bs, num_workers=machine["num_threads"],
                        pin_memory=(device.type == "cuda"), drop_last=True, worker_init_fn=worker_init)
    max_steps, max_hours = cfg["optim.max_steps"], cfg["optim.max_hours"]
    elapsed0, t_start = model.elapsed, time.time()
    hours = lambda: (elapsed0 + time.time() - t_start) / 3600.0
    local, reason, limit, first_compiled = 0, None, False, None
    it = iter(loader)

    def fail(msg):
        monitor.write(f"[{tag}] 異常終了: {msg}。重みは更新・保存しない（最後の保存: {ckpt_dir(run_dir, 'latest')}）。lr を下げて再開するなら bash start2.sh resume {kind} {run_dir.name} --lr <値>")
        monitor.close()
        finish(it, 3)

    while True:
        if model.step >= max_steps:
            reason, limit = f"optim.max_steps {max_steps} に達した", True
            break
        if hours() >= max_hours:
            reason, limit = f"optim.max_hours {max_hours} に達した", True
            break
        if term["sig"] is not None:
            reason = f"signal {term['sig']}"
            break
        local += 1
        if model.maybe_compile(local):
            first_compiled = local
            monitor.write(f"[{tag}] step {model.step + 1}: torch.compile（{cfg['optim.compile_mode']}）を始める。初回は数分かかる。この間 scalar は出ない")
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
            model.train_step(batch)
        except NonFiniteError as e:
            fail(str(e))
        step = model.step
        model.elapsed = elapsed0 + time.time() - t_start
        monitor.tick(t1 - t0)
        if local <= 5 or local == first_compiled or (first_compiled and local == first_compiled + 1):
            note = "（compile 込み）" if local == first_compiled else ""
            monitor.write(f"[{tag}] step {step}: {time.time() - t0:.2f} s{note}（うちデータ待ち {t1 - t0:.2f} s）、起動から {time.time() - t_start:.0f} s")
        if step % cfg["log.print_steps"] == 0 or local in EARLY_STEPS:
            sec = monitor.log_train(step, model, hours())
            if step % cfg["log.print_steps"] == 0:
                info["measured_speed"] = {"step": step, "sec_per_step": sec, "imgs_per_sec": bs / max(sec, 1e-9), "compiled": model.compiled, "bf16": model.use_bf16}
        t_side = time.time()
        if step % cfg["log.image_steps"] == 0:
            try:
                monitor.log_images(step, model, model.pipeline().units)
            except Exception:
                monitor.write(f"[images] step {step} の学習パッチの絵に失敗（学習は続ける）:\n{traceback.format_exc()}")
        first = local == cfg["log.first_check_step"]
        epoch_end = step % spe == 0
        if step % cfg["log.save_latest_steps"] == 0 or first or epoch_end:
            model.save(run_dir, "latest")
            write_info()
            monitor.write(f"[save] step {step} → latest")
        if epoch_end and (step // spe) % cfg["log.save_weights_epochs"] == 0:
            model.save(run_dir, step)
            monitor.write(f"[save] step {step}（epoch {step // spe}）→ {ckpt_dir(run_dir, step)}")
        if epoch_end or first:
            mark = f"epoch_{step // spe:03d}" if epoch_end else f"first_step_{step:07d}"
            try:
                if kind == "v":
                    vf = model.evaluate(val)
                    monitor.log_val(step, vf)
                    monitor.write(f"[val] step {step} flow {vf['val/flow']:.5f}（固定パッチ {cfg['log.val_patches']} 個、EMA、FP32）")
                monitor.epoch_end(step, mark, model.pipeline(), reduced=not epoch_end)
            except Stopped:
                monitor.write(f"[check] step {step} の確認を途中で止めた（signal）")
            except NonFiniteError as e:
                fail(str(e))
            except Exception:  # 確認の絵・数字が失敗しても学習は止めない（夜間に 1 枚の絵のために止まらないように。ログに残す）
                monitor.write(f"[check] step {step} の確認に失敗（学習は続ける）:\n{traceback.format_exc()}")
            if device.type == "cuda":
                torch.cuda.empty_cache()
        monitor.t_window += time.time() - t_side  # 絵・確認・保存の時間は sec_per_step に入れない

    model.elapsed = elapsed0 + time.time() - t_start
    saved = model.save(run_dir, "latest") if local > 0 else True
    if limit and model.step > 0:
        if model.save(run_dir, "best"):
            write_best_note(run_dir, model.step, "last", "latest", f"{reason}。自動では選ばない（最後の EMA）。選び直すのは bash start2.sh best {kind} {run_dir.name} <step>")
            monitor.write(f"[save] step {model.step} → best（rule: last）")
    write_info()
    monitor.write(f"[{tag}] 終了: {reason}。step {model.step}、累計 {model.elapsed / 3600:.2f} 時間。latest {'を保存した' if saved else 'の保存に失敗した'}: {ckpt_dir(run_dir, 'latest')}"
                  + ("" if limit else f"（続きは bash start2.sh resume {kind} {run_dir.name}）"))
    monitor.close()
    finish(it, 143 if term["sig"] is not None else 0)
