"""stage2/models/regression_model.py — Stage 2 の教師あり回帰（U-Net + MSE / L1）の司令塔。junyanz の BaseModel は使わない。

  forward   : ns_forward（range / null-space 分離、2026-11-10 ブランチ 027）: x_in = L_in(x)（網に見せる入力。無効なら x）→ f = net(x_in)（mode.residual=true なら x_in + net(x_in)）
              → y = L_R(x) + (f − L_R(f))（無効なら f。x は生の入力）。L は train.ns.kernel で選ぶ: gaussian = g_σ ∗ x（ns.input_sigma / range_sigma [px]）、
              ideal = FFT の理想低域射影 P（ns.input_cutoff / range_cutoff [cycles/px @1024]、ns.taper）。gaussian で両 σ が 0 なら従来の net(x) / x + net(x) と同一。
              学習の損失・D の偽物・EMA・val・preview・推論（inference_dir.py が同じ ns_forward を import）はすべて y を使う
  loss      : mse | l1 を正規化空間 [−1,1] で計算。表示用に HU 換算（1.0 = hu_per_unit HU）の RMSE / MAE も返す
  optimizer : Adam(lr, (beta1, beta2))。lr は一定（論文に減衰の記載なし）→ scheduler は持たない
  checkpoint: 重みディレクトリ（util/run_paths.py）に net_G.pth と state.pth（optimizer / RNG / epoch / total_iters / **保存時の実効設定 config**）を**原子的に**保存
              （<dir>.tmp に書く → 既存 <dir> を .old に → .tmp を <dir> に → .old を消す。Stage 1 の save_networks と同じ手順）
  resume    : state.pth から optimizer と RNG（python / torch / numpy）を復元し、lr / betas は今の実効値を再適用（上書きが最優先）
"""

import copy
import math
import os
import random
import shutil
import time
from collections import namedtuple
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from util.metrics import mse, psnr, ssim
from models.unet import build_net
from util.run_paths import ckpt_dir, write_best_note

BETTER = {"rmse": lambda new, old: new < old, "ssim": lambda new, old: new > old, "psnr": lambda new, old: new > old}  # log.best_metric ごとの「更新」の向き


def _swap_dirs(tmp, final, old, retries=10, wait=3.0):
    """tmp を final に差し替える（Stage 1 と同じ。Windows バインドマウントで os.replace が PermissionError になる対策、2026-11-03）。
    再試行 → ファイル単位の置き換え → それでも駄目なら警告して学習を続ける（次回の保存で再試行）。"""
    err = None
    for _ in range(retries):
        try:
            if final.exists() and old.exists():
                shutil.rmtree(old)
            if final.exists():
                os.replace(final, old)
            os.replace(tmp, final)
            if old.exists():
                shutil.rmtree(old, ignore_errors=True)
            return True
        except PermissionError as e:
            err = e
            time.sleep(wait)
    try:
        final.mkdir(parents=True, exist_ok=True)
        for f in sorted(tmp.iterdir()):
            os.replace(f, final / f.name)
        shutil.rmtree(tmp, ignore_errors=True)
        if old.exists():
            shutil.rmtree(old, ignore_errors=True)
        print(f"[save] 注意: {final} のフォルダ改名が拒否されたのでファイル単位で置き換えた（{err}）", flush=True)
        return True
    except (PermissionError, FileNotFoundError) as e2:
        print(f"[save] 警告: checkpoint {final} を保存できなかった（{e2}）。学習は続け、次回の保存で再試行する", flush=True)
        return False


def _to_tuple(x):
    """random.setstate は入れ子の tuple を要求する。torch.load 後に list になっている場合に戻す。"""
    if isinstance(x, list):
        return tuple(_to_tuple(v) for v in x)
    return x


# --- range / null-space 分離（2026-11-10 ブランチ 027）。inference_dir.py も同じ関数を使う ---
# 低域化 L の種類（train.ns.kernel。schema.NS_KERNELS と同じ値。schema は torch を読まないので別に持つ）
#   gaussian: 分離 Gaussian（σ は px @1024）。核のスペクトルに零が無く可逆なので、網が逆畳み込みして入力の高域（鍵）を復元・増幅できた（EXP-20261110-01）
#   ideal   : FFT の理想低域射影 P（cutoff は cycles/px @1024、taper で遷移帯）。P² = P なので入力の高域は完全に消え、出力の低域 P y = P x が厳密に固定される（2026-11-10 第 2 版）
NS_KERNELS = ("gaussian", "ideal")
_GAUSS_1D = {}     # (σ, device, dtype) → (1,1,1,K) の正規化 1 次元 Gaussian。lowpass_gaussian が使い回す
_IDEAL_MASK = {}   # (2H, 2W, cutoff, taper, device) → (1,1,2H,W+1) float32 の放射状マスク M(f)。lowpass_ideal が使い回す


def gaussian_kernel1d(sigma, device, dtype):
    """半径 ceil(3σ) の正規化 1 次元 Gaussian を (1,1,1,K) で返す（(σ, device, dtype) ごとにキャッシュ。σ 2.3 → K = 15）。"""
    key = (float(sigma), str(device), dtype)
    k = _GAUSS_1D.get(key)
    if k is None:
        r = math.ceil(3 * sigma)
        ax = torch.arange(-r, r + 1, dtype=torch.float64)
        k1 = torch.exp(-ax ** 2 / (2.0 * sigma ** 2))
        k = (k1 / k1.sum()).to(device=device, dtype=dtype)[None, None, None, :]
        _GAUSS_1D[key] = k
    return k


def lowpass_gaussian(x, sigma):
    """g_σ ∗ x: 1ch (N,1,H,W) に分離 Gaussian（横 → 縦）を reflect padding で掛ける（線形。正規化空間のままでよい）。σ ≤ 0 なら x そのもの（コピーしない）。
    半径 ceil(3σ)。128 patch でも 1024 フルでも同じ核（reflect は半径 < 一辺 が条件。schema.check_train_values が patch_size と照合）。第 1 版（33e3673）の lowpass そのもの。"""
    if sigma <= 0.0:
        return x
    k = gaussian_kernel1d(sigma, x.device, x.dtype)
    r = k.shape[-1] // 2
    x = F.pad(x, (r, r, r, r), mode="reflect")
    return F.conv2d(F.conv2d(x, k), k.transpose(-1, -2))


def ideal_mask(hp, wp, cutoff, taper, device):
    """延長後 (hp, wp) グリッドの rfft2 スペクトル (hp, wp//2+1) に掛ける放射状マスク M(f)（(hp, wp, cutoff, taper, device) ごとにキャッシュ。常に float32）。
    f = √(fy² + fx²) [cycles/px]（fftfreq / rfftfreq = 1 標本あたりの周期数。延長しても画素間隔は変わらないので 128 patch でも 1024 フルでも同じ単位 = 1024 グリッドの cycles/px）。
    M = 1 (f ≤ c − t/2) | 0 (f ≥ c + t/2) | raised cosine ½(1 + cos(π (f − (c − t/2)) / t)) の間（c = cutoff、t = taper。t = 0 なら f ≤ c で 1、他は 0 の急峻な遮断）。"""
    key = (int(hp), int(wp), float(cutoff), float(taper), str(device))
    m = _IDEAL_MASK.get(key)
    if m is None:
        fy = torch.fft.fftfreq(hp, dtype=torch.float64)
        fx = torch.fft.rfftfreq(wp, dtype=torch.float64)
        f = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
        if taper > 0.0:
            lo, hi = cutoff - taper / 2.0, cutoff + taper / 2.0
            ramp = 0.5 * (1.0 + torch.cos(math.pi * (f - lo) / taper))
            m = torch.where(f <= lo, torch.ones_like(f), torch.where(f >= hi, torch.zeros_like(f), ramp))
        else:
            m = (f <= cutoff).to(torch.float64)
        m = m.to(device=device, dtype=torch.float32)[None, None]
        _IDEAL_MASK[key] = m
    return m


def lowpass_ideal(x, cutoff, taper):
    """P x: 1ch (N,1,H,W) の理想低域射影（cutoff / taper は cycles/px）。cutoff ≤ 0 なら x そのもの（コピーしない）。
    FFT は周期境界なので、patch の左右・上下端が隣り合って段差になり、その輪帯（ringing）が内側へ回り込む。これを避けるため x を半画素対称に 2 倍へ延長
    （横に [x, x↔]、縦に [·, ·↕] の (2H, 2W)。DCT-II と同じ周期化で、境界は連続・段差なし）してから rfft2 → M(f) を掛ける → irfft2 → 左上 H×W を切り出す。
    対称な信号に偶な核（M は放射状）を掛けても対称性は保たれるので、延長 → 切り出しを含めた P は直交射影: taper 0 なら P² = P で P(y) = P(x) が厳密に成り立つ
    （taper > 0 なら遷移帯 c ± t/2 の中だけ近似）。16 px の reflect padding + 切り出しは射影にならない（P² ≠ P、境界から鍵が漏れる）ので採らない。
    128 patch（学習）でも 1024 フル（推論）でも同じ関数・同じ単位。FFT は bf16 autocast の中でも float32 で計算する（入口で float に、出口で x の dtype に戻す）。"""
    if cutoff <= 0.0:
        return x
    h, w = x.shape[-2:]
    with torch.autocast(device_type=x.device.type, enabled=False):
        xe = x.float()
        xe = torch.cat([xe, xe.flip(-1)], dim=-1)
        xe = torch.cat([xe, xe.flip(-2)], dim=-2)   # (N,1,2H,2W) 半画素対称延長
        spec = torch.fft.rfft2(xe) * ideal_mask(2 * h, 2 * w, cutoff, taper, x.device)
        y = torch.fft.irfft2(spec, s=(2 * h, 2 * w))[..., :h, :w]
    return y.to(x.dtype)


def lowpass(x, kernel, sigma, cutoff, taper):
    """低域化 L の振り分け（train.ns.kernel）: gaussian → lowpass_gaussian(x, σ)、ideal → lowpass_ideal(x, cutoff, taper)。無効（σ / cutoff が 0）なら x そのもの。"""
    if kernel == "gaussian":
        return lowpass_gaussian(x, sigma)
    if kernel == "ideal":
        return lowpass_ideal(x, cutoff, taper)
    raise ValueError(f"ns.kernel は {NS_KERNELS}: {kernel!r}")


class NSConfig(namedtuple("NSConfig", "kernel input_sigma range_sigma input_cutoff range_cutoff taper")):
    """train.ns.* の実効値（RegressionModel と inference_dir.py が同じものを持つ）。kernel が gaussian なら σ [px]、ideal なら cutoff / taper [cycles/px @1024] を使い、
    使わない側は 0.0（schema.check_train_values が検査。launch.yaml を読む側はこれで単位を取り違えない）。"""
    __slots__ = ()
    KEYS = ("ns.kernel", "ns.input_sigma", "ns.range_sigma", "ns.input_cutoff", "ns.range_cutoff", "ns.taper")   # = schema.NS_KEYS（順序も同じ）

    @classmethod
    def from_flat(cls, train):
        return cls(str(train["ns.kernel"]), float(train["ns.input_sigma"]), float(train["ns.range_sigma"]),
                   float(train["ns.input_cutoff"]), float(train["ns.range_cutoff"]), float(train["ns.taper"]))

    def check(self):
        """値の整合（schema.check_train_values の ns の規則のうち patch_size に依らない部分。inference_dir.py の引数検査用）。問題があれば ValueError。"""
        P = []
        if self.kernel not in NS_KERNELS: P.append(f"kernel は {NS_KERNELS}: {self.kernel!r}")
        for name, v in (("input_sigma", self.input_sigma), ("range_sigma", self.range_sigma)):
            if v < 0.0: P.append(f"{name} ≥ 0.0: {v}")
            elif v > 0.0 and self.kernel != "gaussian": P.append(f"{name} {v} は kernel gaussian のときだけ（ideal では 0.0 にして *_cutoff を使う）")
        for name, v in (("input_cutoff", self.input_cutoff), ("range_cutoff", self.range_cutoff)):
            if not (0.0 <= v <= 0.5): P.append(f"{name} は 0.0（無し）〜 0.5 cycles/px: {v}")
            elif v > 0.0 and self.kernel != "ideal": P.append(f"{name} {v} は kernel ideal のときだけ（gaussian では 0.0 にして *_sigma を使う）")
            elif v > 0.0 and self.taper / 2.0 >= v: P.append(f"taper {self.taper} の半分が {name} {v} 以上（直流まで減衰する。taper < 2 × cutoff）")
        if self.taper < 0.0: P.append(f"taper ≥ 0.0: {self.taper}")
        elif self.taper > 0.0 and self.kernel != "ideal": P.append(f"taper {self.taper} は kernel ideal のときだけ（gaussian では 0.0）")
        if P:
            raise ValueError("ns: " + " / ".join(P))
        return self

    @property
    def range_on(self):
        """range 項（y = L_R(x) + f − L_R(f)）が有効か。無効なら y = f。"""
        return (self.range_sigma if self.kernel == "gaussian" else self.range_cutoff) > 0.0

    def input_lowpass(self, x):
        """x_in = L_in(x)（網に見せる入力）。"""
        return lowpass(x, self.kernel, self.input_sigma, self.input_cutoff, self.taper)

    def range_lowpass(self, x):
        """L_R(x)（range 項 = 出力の低域）。"""
        return lowpass(x, self.kernel, self.range_sigma, self.range_cutoff, self.taper)

    def describe(self):
        """起動ログ用。"""
        if self.kernel == "gaussian":
            return f"gaussian σ_in {self.input_sigma} σ_R {self.range_sigma} [px@1024]"
        return f"ideal c_in {self.input_cutoff} c_R {self.range_cutoff} taper {self.taper} [cycles/px@1024]"


def ns_combine(x, x_in, out, residual, ns):
    """網の生出力 out から最終出力 y を作る。f = x_in + out（mode.residual）| out、y = L_R(x) + (f − L_R(f))（range が有効）| f。
    range 項は**生の x**（input 側の低域化を掛ける前）。出力の低域 = 入力の低域で固定し、網の寄与は零空間（高域）だけになる（ideal なら厳密、gaussian は近似）。"""
    f = x_in + out if residual else out
    if ns.range_on:
        return ns.range_lowpass(x) + f - ns.range_lowpass(f)
    return f


def ns_forward(x, net, residual, ns):
    """x → y の全体（RegressionModel.forward と inference_dir.py が共用）。ns は NSConfig。gaussian で両 σ が 0 なら従来の net(x) / x + net(x) とビット一致。"""
    x_in = ns.input_lowpass(x)
    return ns_combine(x, x_in, net(x_in), residual, ns)


class RegressionModel:
    def __init__(self, train, mode, machine, device):
        self.device = device
        self.config = {"train": dict(train), "mode": dict(mode), "machine": dict(machine)}  # この重みに対応する実効設定（state.pth に入れる。再開の基準）
        self.residual = mode["residual"]
        # range / null-space 分離（2026-11-10 ブランチ 027。gaussian で両 σ 0.0 なら 024 と同一。resume で変更不可 = schema.RESUME_FIXED_TRAIN_KEYS）。NSConfig / lowpass / ns_combine / ns_forward（モジュール関数）
        self.ns = NSConfig.from_flat(train)
        self.net = build_net(mode).to(device)
        # 高速化（Stage 1 と同じ、2026-11-03）: 学習の forward だけ bf16 autocast + torch.compile。val・監視・推論は compile しない FP32 の self.net（同じ重み）。
        # compile は torch.compile で別の呼び出し口を作る（state_dict のキーは self.net のまま）。CUDA Graph の出力 buffer 再利用に備えて学習の出力は clone する
        self.use_bf16 = train["optim.precision"] == "bf16" and device.type == "cuda"
        self.compile_enabled = bool(train["optim.compile"]) and device.type == "cuda"
        if self.compile_enabled and not (shutil.which(os.environ.get("CC", "cc")) or shutil.which("gcc")):
            raise RuntimeError("torch.compile に必要な C コンパイラがありません。Docker image を再ビルドしてください（down → build）")
        self.net_train = torch.compile(self.net, mode="max-autotune") if self.compile_enabled else self.net
        if mode["loss"] == "mse":
            self.criterion = nn.MSELoss()
        elif mode["loss"] == "l1":
            self.criterion = nn.L1Loss()
        else:
            raise ValueError(mode["loss"])
        self.loss_name = mode["loss"]
        self.lr, self.betas = train["optim.lr"], (train["optim.beta1"], train["optim.beta2"])
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, betas=self.betas)
        # 出力側 D（2026-11-09 ブランチ 023。研究者の設計 2026-10-12「MSE + 実 EID 出力側 D」）。adv.lambda 0 なら無し（main と同じ）
        self.adv_lambda, self.adv_source = float(train["adv.lambda"]), train["adv.source"]
        self.net_D, self.optimizer_D, self.gan = None, None, None
        if self.adv_lambda > 0.0:
            from models.discriminator import GANLoss, PatchDiscriminator
            self.net_D = PatchDiscriminator(1, int(train["adv.ndf"])).to(device)
            self.gan = GANLoss(train["adv.gan_mode"])
            self.gan_thresh = 0.5 if train["adv.gan_mode"] == "lsgan" else 0.0   # D_acc の判定の境界（lsgan は 0.5、vanilla は logit 0）
            self.optimizer_D = torch.optim.Adam(self.net_D.parameters(), lr=float(train["adv.lr_d"]), betas=(0.5, 0.999))
        # G の EMA（2026-11-09 ブランチ 024）: decay > 0 なら学習中の self.net とは別に平均した重み self.net_ema を持ち、val / best / 推論用の net_G.pth はそれを使う
        self.ema_decay = float(train["ema.decay"])
        self.net_ema = None
        if self.ema_decay > 0.0:
            self.net_ema = copy.deepcopy(self.net).eval()
            for q in self.net_ema.parameters():
                q.requires_grad_(False)
        self.cur_epoch, self.epoch_done, self.total_iters = 0, False, 0
        self.best = None  # best/ の記録 {"metric", "value", "epoch", "total_iters"}。state.pth に入れて resume で引き継ぐ（無ければ None = 次の val が必ず best）
        self.last = None  # 直近の学習バッチ (input, pred, target)。optimize が更新
        self.n_params = sum(p.numel() for p in self.net.parameters())

    # --- 前向き ---
    def forward(self, x, net=None):
        net = self.net if net is None else net
        return ns_forward(x, net, self.residual, self.ns)

    @property
    def net_eval(self):
        """val / 監視 / 推論に使う G。EMA があれば EMA、無ければ学習中の重み。"""
        return self.net_ema if self.net_ema is not None else self.net

    @torch.no_grad()
    def _ema_update(self):
        d = self.ema_decay
        for q_ema, q in zip(self.net_ema.parameters(), self.net.parameters()):
            q_ema.mul_(d).add_(q.detach(), alpha=1.0 - d)
        for b_ema, b in zip(self.net_ema.buffers(), self.net.buffers()):
            b_ema.copy_(b)

    def optimize(self, batch):
        """1 step。戻り値は表示用の dict: loss（正規化空間の損失）、mse（出力 vs 教師。monitor が √ を取って train/rmse に。0 へ）、ssim（1 へ）。
        forward は bf16 autocast（optim.precision）+ compile（optim.compile）。出力は FP32 に戻してから損失を計算する。"""
        x = batch["input"].to(self.device, non_blocking=True)
        t = batch["target"].to(self.device, non_blocking=True)
        x_in = self.ns.input_lowpass(x)   # 網に見せる入力（ns の input 側が有効なら低域化。FP32、autocast の外）。range 項と損失は生の x / 教師 t
        self.net.train()
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_bf16):
            y = self.net_train(x_in)
        y = y.float()
        if self.compile_enabled:
            y = y.clone()
        pred = ns_combine(x, x_in, y, self.residual, self.ns)   # = ns_forward(x) と同じ値（autocast / compile の都合で分けて書く）
        loss_mse = self.criterion(pred, t)
        loss, extra, fake = loss_mse, {}, None
        if self.net_D is not None:
            # 偽物 = 実 EID patch の出力（adv.source eid）、ペアの出力（eidlike）、両方（both）。本物 = 教師 PCD1024 patch
            fakes = []
            if self.adv_source in ("eid", "both"):
                e = batch["eid"].to(self.device, non_blocking=True)
                e_in = self.ns.input_lowpass(e)
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_bf16):
                    ye = self.net_train(e_in)
                ye = ye.float()
                if self.compile_enabled:
                    ye = ye.clone()
                fakes.append(ns_combine(e, e_in, ye, self.residual, self.ns))
            if self.adv_source in ("eidlike", "both"):
                fakes.append(pred)
            fake = torch.cat(fakes, 0)
            self.net_D.train()
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_bf16):
                d_fake_g = self.net_D(fake)
            loss_adv = self.gan(d_fake_g.float(), True)
            loss = loss_mse + self.adv_lambda * loss_adv
            extra["adv_G"] = float(loss_adv.detach())
            extra["mse_only"] = float(loss_mse.detach())
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        if self.net_ema is not None:
            self._ema_update()
        if self.net_D is not None:
            self.optimizer_D.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_bf16):
                d_real = self.net_D(t)
                d_fake = self.net_D(fake.detach())
            d_real, d_fake = d_real.float(), d_fake.float()
            loss_D = 0.5 * (self.gan(d_real, True) + self.gan(d_fake, False))
            loss_D.backward()
            self.optimizer_D.step()
            with torch.no_grad():
                extra["D_loss"] = float(loss_D); extra["D_real"] = float(d_real.mean()); extra["D_fake"] = float(d_fake.mean())
                extra["D_acc"] = 0.5 * (float((d_real > self.gan_thresh).float().mean()) + float((d_fake <= self.gan_thresh).float().mean()))
        with torch.no_grad():
            self.last = (x.detach(), pred.detach(), t.detach())  # TB の images/current 用（util/monitor.py log_images）
            return {"loss": float(loss), "mse": mse(pred, t), "ssim": ssim(pred, t), **extra}

    @torch.no_grad()
    def predict(self, x):
        """監視・検証用の順伝播（eval、勾配なし）。x は (N,1,H,W) 正規化 tensor（CPU 可）。"""
        net = self.net_eval
        net.eval()
        return self.forward(x.to(self.device), net).cpu()

    @torch.no_grad()
    def evaluate(self, dataset, slices):
        """val のフル画像で rmse（正規化空間。0 へ）/ ssim（1 へ）/ psnr（大きいほど良い）をスライス平均で出す。入力そのまま（何もしない場合）の値も *_input として返す（参照線）。"""
        net = self.net_eval
        net.eval()
        acc = {"mse": 0.0, "ssim": 0.0, "psnr": 0.0, "mse_input": 0.0, "ssim_input": 0.0, "psnr_input": 0.0}
        n = 0
        for _case, in_path, pcd_path in slices:
            x = dataset.load_full(in_path).to(self.device)
            t = dataset.load_full(pcd_path).to(self.device)
            y = self.forward(x, net)
            for name, pred in (("", y), ("_input", x)):
                acc["mse" + name] += mse(pred, t); acc["ssim" + name] += ssim(pred, t); acc["psnr" + name] += psnr(pred, t)
            n += 1
        if n == 0:
            raise RuntimeError("検証スライスがありません")
        out = {k: v / n for k, v in acc.items()}
        out = {"rmse": out["mse"] ** 0.5, "ssim": out["ssim"], "psnr": out["psnr"], "rmse_input": out["mse_input"] ** 0.5, "ssim_input": out["ssim_input"], "psnr_input": out["psnr_input"], "n": n}
        return out

    def update_best(self, run_dir, metric, val, epoch, total_iters):
        """epoch 末の val で log.best_metric が最良を更新したら best/ に保存し best.txt を書く。戻り値: 更新したか。
        self.best を先に更新してから保存するので、best/state.pth の best には自分自身の記録が入る。"""
        value = float(val[metric])
        if self.best is not None and self.best.get("metric") == metric and not BETTER[metric](value, float(self.best["value"])):
            return False
        self.best = {"metric": metric, "value": value, "epoch": int(epoch), "total_iters": int(total_iters)}
        self.save(run_dir, "best")
        write_best_note(run_dir, epoch, "auto", metric=metric, value=value, total_iters=total_iters, source=f"val on {val['n']} slices")
        return True

    # --- checkpoint ---
    def _state(self):
        np_state = np.random.get_state()
        return {
            "epoch": int(self.cur_epoch), "epoch_done": bool(self.epoch_done), "total_iters": int(self.total_iters),
            "best": self.best,  # best/ の記録（自動更新の比較対象。resume で引き継ぐ）
            "optimizer": self.optimizer.state_dict(),
            "optimizer_D": self.optimizer_D.state_dict() if self.optimizer_D is not None else None,   # 出力側 D（無ければ None）
            "lr": float(self.optimizer.param_groups[0]["lr"]),
            "config": self.config,  # 保存時の実効設定（train / mode / machine、平坦キー）。resume はこれを基準にする（失敗した起動の launch を引き継がない。レビュー指摘 2026-09-09）
            "rng": {
                "python": random.getstate(),
                "torch": torch.get_rng_state(),
                "numpy": (np_state[0], torch.from_numpy(np_state[1].astype(np.int64)), int(np_state[2]), int(np_state[3]), float(np_state[4])),
            },
        }

    def save(self, run_dir, tag):
        """net_G.pth と state.pth を重みディレクトリに原子的に保存する。"""
        final = ckpt_dir(run_dir, tag)
        tmp, old = final.with_name(final.name + ".tmp"), final.with_name(final.name + ".old")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        torch.save(self.net_eval.state_dict(), tmp / "net_G.pth")   # 推論・評価用（EMA があれば EMA）
        if self.net_ema is not None:
            torch.save(self.net.state_dict(), tmp / "net_G_raw.pth")   # 学習中の重み（resume 用）
        if self.net_D is not None:
            torch.save(self.net_D.state_dict(), tmp / "net_D.pth")
        torch.save(self._state(), tmp / "state.pth")
        _swap_dirs(tmp, final, old)  # Windows バインドマウントの PermissionError 対策

    def load_weights(self, weight_dir):
        g = torch.load(Path(weight_dir) / "net_G.pth", map_location=self.device, weights_only=True)
        raw = Path(weight_dir) / "net_G_raw.pth"
        if self.net_ema is not None:
            self.net_ema.load_state_dict(g)
            if raw.is_file():
                self.net.load_state_dict(torch.load(raw, map_location=self.device, weights_only=True))
            else:
                self.net.load_state_dict(g)   # EMA なしの run から EMA 付きで再開: 学習中の重みも net_G.pth から
                print("[resume] 注意: net_G_raw.pth が無いので学習中の重みも net_G.pth から読む（EMA なしの run からの再開）")
        else:
            self.net.load_state_dict(torch.load(raw, map_location=self.device, weights_only=True) if raw.is_file() else g)   # EMA 付きの run から EMA なしで再開: 学習中の重みを引き継ぐ
        if self.net_D is not None:
            p = Path(weight_dir) / "net_D.pth"
            if p.is_file():
                self.net_D.load_state_dict(torch.load(p, map_location=self.device, weights_only=True))
            else:
                print(f"[resume] 注意: {p} が無いので D は初期化のまま（D なしの run から D 付きで再開）")

    def resume(self, state_path):
        """state.pth から optimizer / RNG / 進捗を復元する。state は CPU に読む（RNG 用 Tensor を CUDA に載せない）。lr / betas は今の実効値を再適用。"""
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        self.optimizer.load_state_dict(state["optimizer"])
        for g in self.optimizer.param_groups:
            g["lr"] = self.lr
            g["betas"] = self.betas
        if self.optimizer_D is not None and state.get("optimizer_D"):
            self.optimizer_D.load_state_dict(state["optimizer_D"])
        rng = state["rng"]
        random.setstate(_to_tuple(rng["python"]))
        torch.set_rng_state(rng["torch"].cpu())
        kind, keys, pos, has_gauss, cached = rng["numpy"]
        np.random.set_state((kind, keys.cpu().numpy().astype(np.uint32), pos, has_gauss, cached))
        self.total_iters = int(state["total_iters"])
        self.best = state.get("best")  # 古い state（best 実装前）には無い → None（次の val が best になる。best/ が既にあれば上書きされる）
        best_msg = f"best {self.best['metric']} {self.best['value']:.6f} @ epoch {self.best['epoch']}" if self.best else "best 記録なし"
        print(f"[resume] {state_path}: epoch {state['epoch']} (done={state['epoch_done']}), total_iters {state['total_iters']}, lr {self.optimizer.param_groups[0]['lr']:.7f}, {best_msg}")
        return state
