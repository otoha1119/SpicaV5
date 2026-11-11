"""stage1/train_y.py — 型 Y の学習（起動器と学習ループを 1 本に。ホストからは bash start.sh PC1 y [--flag value] / bash start.sh PC1 y resume <run>）。

  python stage1/train_y.py --machine PC1 --config stage1/configs/y.yaml --shared configs/shared.yaml --machines configs/machines.yaml [--resume <run>] -- [--flag value]

部品（docs/plans/20261111_detail_design/stage1_redesign_ideas/persona_skeptic.md 案 1。式の出典は UPSTREAM.md の 7）
  A（消し手、学習）: PCD512 → m。近くの画素の重みつき平均しかできない（models/eraser.py）
  E（実 EID の描き手、固定）: init.e_run の EMA。実 EID だけで学んだもの
  F（疑似側の物差し、学習）: E の重みから始め、「m に強めのノイズをかぶせた絵」の拡散の損失で m の集まりを追う
1 step
  1. PCD のパッチ → m = A(x)。手がかり c = Gaussian(m, dmd.cue_blur_px)（明るさの配置をそろえて比べるため。勾配は流さない）
  2. σ ~ loguniform[sigma_lo, sigma_hi]、y = m + σε。e = E(y)、f = F(y)（勾配なし）
  3. A の更新: g = (f − e) / mean|m − e|（サンプルごと）、損失 = ½ Σ (m − sg(m − g))² / B（DMD。∂損失/∂m = g、つまり m を e − f の向きに動かす）
  4. F の更新（optim.f_steps 回）: m（勾配なし）に σ' のノイズを足し、EDM の損失（F の側で重み 1）
  PCD と画素を比べる損失・判別器は無い。E と F は勾配を止めた答えだけを使う。
精度: A・損失・ぼかし・EDM の前処理は FP32。E・F の U-Net 本体だけ bf16（optim.precision）。F の学習の本体だけ torch.compile（optim.compile）。
出力: <painter_checkpoints_dir>/<run>/（launch.yaml、loss_log.txt、tb/、latest/ と weights/step_N/ に A.pth・F.pth・state.pth、output_images/step_N/*.png（16bit）、preview/zoom/step_N.png）
止まり方: max_steps / max_hours → latest を保存して 0。SIGTERM / SIGINT → latest を保存して 143。損失・勾配が NaN → 保存せず 3。
"""

import argparse
import datetime
import math
import os
import random
import signal
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from configs.schema import COMPILE_MODES, MACHINE, PRECISIONS, SHARED, Y_TRAIN, ConfigError, _check_machine, _raise, apply_overrides, validate, validate_shared  # noqa: E402
from data.ct_io import hu_tuple, write_png, x_to_stored  # noqa: E402
from data.eid_dataset import EidPatchStream, load_full_x, resolve_split, worker_init  # noqa: E402
from diffusion.anchor import anchor, wide_mean  # noqa: E402
from diffusion.cue import center_crop, gaussian_blur  # noqa: E402
from diffusion.edm import ancestral_step, karras_sigmas, precond  # noqa: E402
from models.eraser import build_eraser, fill_legacy  # noqa: E402
from models.painter import _swap_dirs, denoise, load_ema, want_bf16  # noqa: E402
from models.unet import build_net  # noqa: E402
from util.run_paths import JST, LAUNCH_FILE, ckpt_dir, run_name, saved_tags  # noqa: E402
from run_train import other_trainings  # noqa: E402

KINDS = ("A.pth", "F.pth", "state.pth")


def load_yaml(path):
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"設定ファイルが見つかりません: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def e_dir_of(cfg, machine):
    d = ckpt_dir(Path(machine["painter_checkpoints_dir"]) / cfg["init.e_run"], cfg["init.e_tag"])
    missing = [k for k in ("net.pth", "ema.pth", "state.pth") if not (d / k).is_file()]
    if missing:
        raise ConfigError(f"init.e_run の重みが揃っていません: {d} に {missing} が無い")
    return d


def check_values(c, shared, machine, e_train):
    P = []
    for k in ("optim.lr_a", "optim.lr_f", "optim.grad_clip", "optim.max_hours", "dmd.sigma_lo_hu", "dmd.sigma_hi_hu", "dmd.cue_blur_px"):
        if c[k] <= 0: P.append(f"{k} は正")
    if not (0.0 <= c["optim.beta1"] < 1.0 and 0.0 <= c["optim.beta2"] < 1.0): P.append("optim.beta1 / beta2 は [0, 1)")
    for k in ("optim.batch_size", "optim.f_steps", "optim.max_steps", "data.patches_per_slice", "eraser.radius", "eraser.width", "eraser.depth",
              "log.print_steps", "log.preview_steps", "log.save_latest_steps", "log.save_weights_steps", "log.preview_steps_redraw"):
        if c[k] < 1: P.append(f"{k} ≥ 1")
    if c["eraser.kind"] not in ("free", "isotropic"): P.append("eraser.kind は free | isotropic")
    if c["optim.compile_after_steps"] < 0: P.append("optim.compile_after_steps ≥ 0")
    if c["optim.precision"] not in PRECISIONS: P.append(f"optim.precision は {PRECISIONS}")
    if c["optim.compile_mode"] not in COMPILE_MODES: P.append(f"optim.compile_mode は {COMPILE_MODES}")
    if c["dmd.sigma_lo_hu"] >= c["dmd.sigma_hi_hu"]: P.append("dmd.sigma_lo_hu < sigma_hi_hu")
    if c["data.patch_size"] % 16: P.append("data.patch_size は 16 の倍数")
    if c["data.margin"] < max(c["eraser.radius"], 3 * c["dmd.cue_blur_px"]): P.append("data.margin ≥ max(eraser.radius, 3 × dmd.cue_blur_px)")
    if c["dmd.cue_blur_px"] > e_train["cue.blur_max_px"]: P.append(f"dmd.cue_blur_px は E の学習の上端 {e_train['cue.blur_max_px']} 以下")
    if not c["data.pcd_val_cases"]: P.append("data.pcd_val_cases に 1 症例以上")
    if not c["log.pcd_slices"]: P.append("log.pcd_slices に 1 枚以上")
    for p in c["log.pcd_slices"]:
        if not p.startswith("/"): P.append(f"log.pcd_slices はコンテナ内の絶対パス: {p}")
        elif Path(p).parent.name not in c["data.pcd_val_cases"]: P.append(f"log.pcd_slices は val の症例から: {p}")
        elif not Path(p).is_file(): P.append(f"log.pcd_slices が無い: {p}")
    try:
        if not c["log.preview_s_hu"] or any(float(v) < 0 for v in c["log.preview_s_hu"]): P.append("log.preview_s_hu は 0 以上の数のリスト")
    except ValueError:
        P.append("log.preview_s_hu は数のリスト（文字列で書く）")
    _check_machine(machine, P)
    _raise("y values", P)


class Trainer:
    def __init__(self, c, machine, device, e_dir):
        self.c, self.device = c, device
        self.E, e_cfg, self.e_step = load_ema(e_dir, device)
        self.e_cfg = e_cfg
        et = e_cfg["train"]
        self.hu = hu_tuple(et)
        self.scale = et["data.hu_scale"]
        self.sd = et["diffusion.sigma_data"]
        self.cl = bool(et["optim.channels_last"])
        self.F = build_net(e_cfg["model"], et).to(device)
        self.F.load_state_dict(torch.load(Path(e_dir) / "ema.pth", map_location=device, weights_only=True))
        if self.cl:
            self.F = self.F.to(memory_format=torch.channels_last)
        self.F.train()
        self.A = build_eraser(c).to(device)
        self.bf16 = want_bf16(c["optim.precision"], device)
        self.opt_a = torch.optim.AdamW(self.A.parameters(), lr=c["optim.lr_a"], betas=(c["optim.beta1"], c["optim.beta2"]), weight_decay=0.0)
        self.opt_f = torch.optim.AdamW(self.F.parameters(), lr=c["optim.lr_f"], betas=(c["optim.beta1"], c["optim.beta2"]), weight_decay=0.0)
        self.compile_enabled = bool(c["optim.compile"]) and device.type == "cuda"
        self.f_body, self.compiled = self.F.body, False
        self.step, self.elapsed = 0, 0.0
        self.lo, self.hi = c["dmd.sigma_lo_hu"] / self.scale, c["dmd.sigma_hi_hu"] / self.scale
        self.reset_window()

    def reset_window(self):
        self.win = {k: 0.0 for k in ("n", "loss_a", "loss_f", "grad_a", "grad_f", "gap_hu", "change_hu", "center_w")}

    def maybe_compile(self, local):
        if self.compile_enabled and not self.compiled and local > self.c["optim.compile_after_steps"]:
            self.f_body = torch.compile(self.F.body, mode=self.c["optim.compile_mode"])
            self.compiled = True
            return True
        return False

    def _sigma(self, b):
        u = torch.rand(b, device=self.device)
        return torch.exp(math.log(self.lo) + u * (math.log(self.hi) - math.log(self.lo)))

    def _cond(self, b):
        d = self.device
        return torch.full((b,), self.c["dmd.cue_blur_px"], device=d), torch.zeros(b, device=d), torch.ones(b, device=d, dtype=torch.bool)

    def train_step(self, batch):
        c, d, M = self.c, self.device, self.c["data.margin"]
        big = batch.to(d, non_blocking=True).float()
        b = big.shape[0]
        m_big, wts = self.A(big, return_weights=True)
        m = center_crop(m_big, M)
        blur, noise0, has = self._cond(b)
        with torch.no_grad():
            cue = center_crop(gaussian_blur(m_big.detach(), c["dmd.cue_blur_px"]), M)
            md = m.detach()
            sigma = self._sigma(b)
            y = md + sigma.view(-1, 1, 1, 1) * torch.randn_like(md)
            e = denoise(self.E, y, sigma, cue, blur, noise0, has, self.sd, bf16=self.bf16, channels_last=self.cl)[0]
            f = denoise(self.F, y, sigma, cue, blur, noise0, has, self.sd, bf16=self.bf16, channels_last=self.cl)[0]
            g = (f - e) / (md - e).abs().mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
            target = md - g
        loss_a = 0.5 * ((m - target) ** 2).sum() / b
        if not torch.isfinite(loss_a):
            raise FloatingPointError(f"A の損失が有限でない（step {self.step + 1}）")
        self.opt_a.zero_grad(set_to_none=True)
        loss_a.backward()
        grad_a = float(torch.nn.utils.clip_grad_norm_(self.A.parameters(), c["optim.grad_clip"]))
        if not math.isfinite(grad_a):
            raise FloatingPointError(f"A の勾配が有限でない（step {self.step + 1}）")
        self.opt_a.step()
        loss_f_sum, grad_f = 0.0, 0.0
        for _ in range(c["optim.f_steps"]):
            s2 = self._sigma(b)
            x = md + s2.view(-1, 1, 1, 1) * torch.randn_like(md)
            c_skip, c_out, _ = precond(s2, self.sd)
            tgt = (md - c_skip * x) / c_out
            _, fo = denoise(self.F, x, s2, cue, blur, noise0, has, self.sd, body=self.f_body, bf16=self.bf16, channels_last=self.cl, clone=self.compiled)
            loss_f = ((fo - tgt) ** 2).mean()
            if not torch.isfinite(loss_f):
                raise FloatingPointError(f"F の損失が有限でない（step {self.step + 1}）")
            self.opt_f.zero_grad(set_to_none=True)
            loss_f.backward()
            grad_f = float(torch.nn.utils.clip_grad_norm_(self.F.parameters(), c["optim.grad_clip"]))
            if not math.isfinite(grad_f):
                raise FloatingPointError(f"F の勾配が有限でない（step {self.step + 1}）")
            self.opt_f.step()
            loss_f_sum += loss_f.item()
        self.step += 1
        with torch.no_grad():
            w = self.win
            w["n"] += 1
            w["loss_a"] += loss_a.item()
            w["loss_f"] += loss_f_sum / c["optim.f_steps"]
            w["grad_a"] += grad_a
            w["grad_f"] += grad_f
            w["gap_hu"] += float((f - e).abs().mean()) * self.scale
            w["change_hu"] += float(((md - center_crop(big, M)) ** 2).mean().sqrt()) * self.scale
            w["center_w"] += float(center_crop(wts, M)[:, wts.shape[1] // 2].mean())
            if hasattr(self.A, "sigma_px"):
                w["sigma_px"] = w.get("sigma_px", 0.0) + float(center_crop(self.A.sigma_px(big), M).mean())

    def window_stats(self):
        w, n = self.win, max(self.win["n"], 1)
        return {"loss/a_dmd": w["loss_a"] / n, "loss/f_denoise": w["loss_f"] / n, "diag/grad_norm_a": w["grad_a"] / n, "diag/grad_norm_f": w["grad_f"] / n,
                "diag/gap_f_minus_e_hu": w["gap_hu"] / n, "diag/a_change_rms_hu": w["change_hu"] / n, "diag/a_center_weight": w["center_w"] / n,
                **({"diag/a_sigma_px_mean": w["sigma_px"] / n} if "sigma_px" in w else {})}

    # --- 途中経過 ---
    @torch.no_grad()
    def redraw(self, m, src, s, steps, gen=None):
        """m（A の出力）に強さ s のノイズを足し、E で s から 0 まで描き直す（Euler ancestral η = 1、毎段に錨）。"""
        b = m.shape[0]
        blur, noise0, has = self._cond(b)
        cue = gaussian_blur(m, self.c["dmd.cue_blur_px"])
        wm = wide_mean(src, self.anchor_px)
        sig = karras_sigmas(steps, self.e_cfg["train"]["diffusion.sigma_min"], s, 7.0, m.device)
        rn = (lambda: torch.randn(m.shape, generator=gen, device=m.device)) if gen is not None else (lambda: torch.randn_like(m))
        x = m + s * rn()
        for i in range(steps):
            a, a2 = float(sig[i]), float(sig[i + 1])
            x0 = denoise(self.E, x, torch.full((b,), a, device=m.device), cue, blur, noise0, has, self.sd, bf16=self.bf16, channels_last=self.cl)[0]
            x0 = anchor(x0, wm, self.anchor_px)
            if a2 == 0:
                return x0
            down, up = ancestral_step(a, a2, 1.0)
            x = x0 + (down / a) * (x - x0) + up * rn()
        return x


def fine_stats(x, ref, scale):
    """確認用（設計には使わない）: 軟部の平らな所で、細かい成分（x − Gaussian(x, 2)）の SD（HU）と、ref の細かい成分との相関。"""
    sm = gaussian_blur(ref, 3.0)
    gy, gx = torch.gradient(sm[0, 0])
    g = torch.hypot(gx, gy)
    soft = (sm[0, 0] > -0.1) & (sm[0, 0] < 0.2)
    if int(soft.sum()) < 100:
        return float("nan"), float("nan")
    flat = soft & (g < torch.quantile(g[soft], 0.4))
    hx, hr = (x - gaussian_blur(x, 2.0))[0, 0][flat], (ref - gaussian_blur(ref, 2.0))[0, 0][flat]
    hx0, hr0 = hx - hx.mean(), hr - hr.mean()
    return float(hx.std()) * scale, float((hx0 * hr0).sum() / (hx0.norm() * hr0.norm()).clamp_min(1e-12))


def save_panel(path, imgs, labels, scale, zoom):
    """見やすい途中経過（8bit、表示は HU −1400〜3000 = stored 0〜4400）。上段 = 全体、下段 = 拡大（zoom = (x, y, size, 倍率)）。下に名前。"""
    from PIL import Image, ImageDraw
    to8 = lambda a: (np.clip((a[0, 0].float().cpu().numpy() * scale + 1400.0) / 4400.0, 0, 1) * 255).astype(np.uint8)
    zx, zy, zs, zk = zoom
    tops = [to8(a) for a in imgs]
    h, w = tops[0].shape
    zooms = [np.kron(t[zy:zy + zs, zx:zx + zs], np.ones((zk, zk), dtype=np.uint8)) for t in tops]
    pad, th, zh = 6, 22, zs * zk
    W = pad + len(imgs) * (max(w, zh) + pad)
    canvas = Image.new("L", (W, pad + h + pad + zh + th), 255)
    dr = ImageDraw.Draw(canvas)
    for i, (t, z, lab) in enumerate(zip(tops, zooms, labels)):
        x0 = pad + i * (max(w, zh) + pad)
        canvas.paste(Image.fromarray(t), (x0, pad))
        canvas.paste(Image.fromarray(z), (x0, pad + h + pad))
        dr.text((x0, pad + h + pad + zh + 4), lab, fill=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def save(tr, run_dir, tag, config):
    final = ckpt_dir(run_dir, tag)
    tmp, old = final.with_name(final.name + ".tmp"), final.with_name(final.name + ".old")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    torch.save(tr.A.state_dict(), tmp / "A.pth")
    torch.save(tr.F.state_dict(), tmp / "F.pth")
    torch.save({"step": tr.step, "elapsed_sec": tr.elapsed, "opt_a": tr.opt_a.state_dict(), "opt_f": tr.opt_f.state_dict(), "config": config,
                "e_step": tr.e_step, "torch_rng": torch.get_rng_state()}, tmp / "state.pth")
    return _swap_dirs(tmp, final, old)


def main():
    p = argparse.ArgumentParser(description="型 Y の学習")
    p.add_argument("--machine", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--shared", required=True)
    p.add_argument("--machines", required=True)
    p.add_argument("--resume", default=None, metavar="RUN")
    p.add_argument("--allow_concurrent", action="store_true", help="ほかの学習が同じコンテナで動いていても起動する（既定は止める）")
    p.add_argument("overrides", nargs=argparse.REMAINDER)
    a = p.parse_args()
    overrides = a.overrides[1:] if a.overrides and a.overrides[0] == "--" else a.overrides
    now = datetime.datetime.now(JST)
    others = other_trainings()
    if others and not a.allow_concurrent:
        print("[train_y] 設定エラー: このコンテナでほかの学習が動いています。止めてから起動するか --allow_concurrent: " + "; ".join(f"pid {pid}: {cmd}" for pid, cmd in others), file=sys.stderr)
        sys.exit(2)
    try:
        shared = validate_shared("shared", load_yaml(a.shared), SHARED)
        machines = load_yaml(a.machines)
        if not isinstance(machines, dict) or a.machine not in machines:
            raise ConfigError(f"machines.yaml にエントリ '{a.machine}' がありません")
        machine = validate_shared(f"machines.{a.machine}", machines[a.machine], MACHINE)
        if a.resume:
            run_dir = Path(machine["painter_checkpoints_dir"]) / a.resume
            sp = ckpt_dir(run_dir, "latest") / "state.pth"
            if not all((sp.parent / k).is_file() for k in KINDS):
                raise ConfigError(f"再開用の重みが揃っていません: {sp.parent}（保存済み: {saved_tags(run_dir)}）")
            state = torch.load(sp, map_location="cpu", weights_only=False)
            c = validate("y(state)", fill_legacy(dict(state["config"])), Y_TRAIN)
            frozen = {k: c[k] for k in c if k.startswith(("eraser.", "init."))}
            applied = apply_overrides(overrides, {"y": (c, Y_TRAIN), "shared": (shared, SHARED), "machine": (machine, MACHINE)})
            changed = [k for k in frozen if frozen[k] != c[k]]
            if changed:
                raise ConfigError(f"再開では変えられない設定です: {changed}")
            launch_name = f"launch_resume_{now.strftime('%Y%m%d_%H%M%S')}.yaml"
        else:
            c = validate("y", load_yaml(a.config), Y_TRAIN)
            applied = apply_overrides(overrides, {"y": (c, Y_TRAIN), "shared": (shared, SHARED), "machine": (machine, MACHINE)})
            run_dir = Path(machine["painter_checkpoints_dir"]) / run_name(now)  # 有無の検査は値の検査の後
            launch_name, state = LAUNCH_FILE, None
        e_dir = e_dir_of(c, machine)
        e_train = torch.load(e_dir / "state.pth", map_location="cpu", weights_only=True)["config"]["train"]
        check_values(c, shared, machine, e_train)
        if not a.resume and run_dir.exists():
            raise ConfigError(f"run フォルダが既にあります: {run_dir}（同じ分の再起動。1 分待つ）")
        split = resolve_split({"data.domain": "pcd", "data.pcd_val_cases": c["data.pcd_val_cases"]}, machine,
                              cache_dir=Path(machine["painter_checkpoints_dir"]) / ".case_index")
    except (ConfigError, RuntimeError) as e:
        print(f"[train_y] 設定エラー: {e}", file=sys.stderr)
        sys.exit(2)

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / launch_name, "w", encoding="utf-8") as f:
        yaml.safe_dump({"timestamp": now.isoformat(timespec="seconds"), "stage": "stage1_type_y", "machine_name": a.machine, "y": c, "shared": shared,
                        "machine": machine, "overrides": list(overrides), "applied": applied, "resume": a.resume, "e_dir": str(e_dir),
                        "train_cases": split["train"], "val_cases": split["val"]}, f, allow_unicode=True, sort_keys=False)
    seed = c["optim.seed"] + (state["step"] if state else 0)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if machine["gpu_gen"] != 0 and device.type != "cuda":
        raise RuntimeError("machines.yaml の gpu_gen ≠ 0 なのに CUDA が使えません")
    torch.backends.cudnn.benchmark = True
    tr = Trainer(c, machine, device, e_dir)
    tr.anchor_px = shared["brightness.sigma_px_512"]
    if state:
        d = sp.parent
        tr.A.load_state_dict(torch.load(d / "A.pth", map_location=device, weights_only=True))
        tr.F.load_state_dict(torch.load(d / "F.pth", map_location=device, weights_only=True))
        tr.opt_a.load_state_dict(state["opt_a"]); tr.opt_f.load_state_dict(state["opt_f"])
        for o, lr in ((tr.opt_a, c["optim.lr_a"]), (tr.opt_f, c["optim.lr_f"])):
            for g in o.param_groups:
                g["lr"] = lr
        tr.step, tr.elapsed = int(state["step"]), float(state["elapsed_sec"])
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(run_dir / "tb"))
    log = open(run_dir / "loss_log.txt", "a", encoding="utf-8")

    def say(msg):
        print(msg, flush=True)
        log.write(msg + "\n"); log.flush()

    n_a = sum(p.numel() for p in tr.A.parameters())
    say(f"[train_y] run {run_dir.name}: E {e_dir}（step {tr.e_step}）、A {n_a / 1e3:.1f}k params（近傍 {2 * c['eraser.radius'] + 1}²）、"
        f"F {sum(p.numel() for p in tr.F.parameters()) / 1e6:.1f}M、train {len(split['train'])} 症例、val {split['val']}、batch {c['optim.batch_size']}、"
        f"見比べ σ {c['dmd.sigma_lo_hu']:g}〜{c['dmd.sigma_hi_hu']:g} HU、bf16 {tr.bf16}、compile {tr.compile_enabled}、start step {tr.step}")
    ds_cfg = {"data.patch_size": c["data.patch_size"], "data.margin": c["data.margin"], "data.patches_per_slice": c["data.patches_per_slice"], "optim.seed": c["optim.seed"]}
    loader = torch.utils.data.DataLoader(EidPatchStream(split["train_slices"], ds_cfg, tr.hu, start_step=tr.step), batch_size=c["optim.batch_size"],
                                         num_workers=machine["num_threads"], pin_memory=device.type == "cuda", drop_last=True, worker_init_fn=worker_init)
    prev = [(Path(pth), load_full_x(pth, tr.hu).to(device)) for pth in c["log.pcd_slices"]]
    zoom = (266, 222, 72, 4)  # 途中経過の拡大（PCD-002-236 のステント。stage1/configs/train.yaml の pcd_zoom と同じ所）
    term = {"sig": None}

    def on_term(signum, _f):
        term["sig"] = signum
        print(f"[train_y] signal {signum}。いまの step を終えて latest を保存して止まる", flush=True)

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)
    el0, t0 = tr.elapsed, time.time()
    hours = lambda: (el0 + time.time() - t0) / 3600.0
    it, local, reason, t_win = iter(loader), 0, None, time.time()
    while True:
        if tr.step >= c["optim.max_steps"]:
            reason = "max_steps"; break
        if hours() >= c["optim.max_hours"]:
            reason = "max_hours"; break
        if term["sig"] is not None:
            reason = f"signal {term['sig']}"; break
        local += 1
        if tr.maybe_compile(local):
            say(f"[train_y] step {tr.step + 1}: F の本体を torch.compile（{c['optim.compile_mode']}）。初回は数分")
        try:
            batch = next(it)
        except RuntimeError:
            if term["sig"] is not None:
                reason = f"signal {term['sig']}"; break
            raise
        try:
            tr.train_step(batch)
        except FloatingPointError as e:
            say(f"[train_y] 異常終了: {e}。保存しない")
            sys.exit(3)
        tr.elapsed = el0 + time.time() - t0
        step = tr.step
        if step % c["log.print_steps"] == 0 or local in (1, 2, 5, 10):
            st = tr.window_stats()
            sec = (time.time() - t_win) / max(tr.win["n"], 1)
            st["diag/sec_per_step"], st["diag/hours"] = sec, hours()
            for k, v in st.items():
                writer.add_scalar(k, v, step)
            say(f"[train_y] step {step} " + " ".join(f"{k.split('/')[-1]} {v:.4g}" for k, v in st.items()))
            tr.reset_window(); t_win = time.time()
        if step % c["log.preview_steps"] == 0 or local == 10:
            ts = time.time()
            tr.A.eval()
            with torch.no_grad():
                out_dir = run_dir / "output_images" / f"step_{step:07d}"
                for pth, x in prev:
                    m = tr.A(x)
                    vals = {"val/a_change_rms_hu": float(((m - x) ** 2).mean().sqrt()) * tr.scale}
                    redraws = {}
                    sd, corr = fine_stats(m, x, tr.scale)
                    vals["val/m_flat_fine_sd_hu"], vals["val/m_fine_corr_pcd"] = sd, corr
                    write_png(out_dir / f"{pth.stem}__m.png", x_to_stored(m[0, 0].float().cpu().numpy(), tr.hu))
                    for s in c["log.preview_s_hu"]:
                        g = torch.Generator(device=device); g.manual_seed(0)
                        r = tr.redraw(m, x, float(s) / tr.scale, c["log.preview_steps_redraw"], gen=g)
                        redraws[s] = r
                        sd, corr = fine_stats(r, x, tr.scale)
                        vals[f"val/s{s}_flat_fine_sd_hu"], vals[f"val/s{s}_fine_corr_pcd"] = sd, corr
                        write_png(out_dir / f"{pth.stem}__s{s}.png", x_to_stored(r[0, 0].float().cpu().numpy(), tr.hu))
                    panel = [x, m]
                    for s in c["log.preview_s_hu"]:
                        panel.append(redraws[s])
                    save_panel(run_dir / "preview" / pth.stem / f"step_{step:07d}.png", panel, ["PCD", "A(PCD)"] + [f"s{s}" for s in c["log.preview_s_hu"]], tr.scale, zoom)
                    for k, v in vals.items():
                        if math.isfinite(v):
                            writer.add_scalar(k, v, step)
                    say(f"[preview] step {step} {pth.name} ({time.time() - ts:.1f} s) " + " ".join(f"{k.split('/')[-1]} {v:.4g}" for k, v in vals.items()))
            tr.A.train()
            t_win += time.time() - ts
        if step % c["log.save_weights_steps"] == 0:
            save(tr, run_dir, step, c)
        if step % c["log.save_latest_steps"] == 0:
            save(tr, run_dir, "latest", c)
            say(f"[save] step {step} → latest")
    tr.elapsed = el0 + time.time() - t0
    ok = save(tr, run_dir, "latest", c) if local > 0 else True
    say(f"[train_y] 終了: {reason}。step {tr.step}、累計 {tr.elapsed / 3600:.2f} 時間。latest {'保存' if ok else '保存失敗'}（続きは bash start.sh PC1 y resume {run_dir.name}）")
    writer.close(); log.close()
    for w in getattr(it, "_workers", []):
        try:
            w.kill()
        except Exception:
            pass
    sys.exit(143 if term["sig"] is not None else 0)


if __name__ == "__main__":
    main()
