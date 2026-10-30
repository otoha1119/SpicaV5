"""stage1_v3/models/projection_model.py — 射影型 Stage 1: 物理コア P（dataset 側）+ 学習残差 R + 実 EID 恒等項。

【記号】（docs/surveys/20261030_stage1-projection-identity.html）
  real_A = x（PCD512 patch）   U = u = P(x)（dataset が掛けた劣化。確率的）   real_B = y（実 EID512 patch）
  G(in) = in + R(in)（networks.ResidualGenerator。R は決定的、乱数入力なし）
  fake_B = u + R(u)（EID-like。D に見せる）     idt_B = y + R(y)（恒等経路。D には見せない）

【損失】
  D : 論文 Eq.(12)（gan_mode fgan_kl: D := 1 − exp(v)、stage1/models/fidelity_gan_model.py と同じ）。real = y、fake = fake_B
  G : L_GAN(fake_B) + λ_fid · mean(R(u)²) + λ_id(epoch) · mean(R(y)²)
      fidelity は 1 draw の ‖R(u)‖²（レビュー B: 2 draw 内積 mean(R(u₁)⊙R(u₂)) は R が P のノイズを打ち消す denoise に盲目なので不採用。
      ‖R(u)‖² = ‖E R‖² + Var_u R で、Var_u R は「P の実現への反応」なので罰してよい。確率性は P が担保する（R-S21））
      λ_id は lambda_id → lambda_id_final へ lambda_id_decay_epochs で線形減衰（0 なら一定）
【BatchNorm】 PCD 経路（u）と EID 経路（y）は 1 つのバッチに連結して G に通す（別 forward だと BN のバッチ統計の違いだけで L_id ≈ 0 になり eval で再現しない。レビュー G1）。
【高速化（008 と同じ）】 optim.precision bf16 = CUDA autocast（G / D の forward だけ。損失は FP32 に戻して計算。fgan_kl の exp は FP32）。optim.compile = nn.Module.compile(mode="max-autotune")。
  compile 時は CUDA Graph が出力 buffer を再利用するので、保持する出力は _run_net で clone する（D は 1 iteration に 3 回呼ぶ）
【診断値（HU）】 R_u_rms / R_y_rms（残差の RMS）、R_ratio = R_u_rms / R_y_rms（≫ 1 かつ D 分離維持 = R がドメイン検出器化、レビュー D）、P_rms = RMS(u − x)、d_in = mean|fake_B − x|
"""

import os
import random
import shutil
import time

import numpy as np
import torch

from models import networks
from util.run_paths import CKPT_KINDS, NET_NAMES, ckpt_dir


def _to_tuple(x):
    return tuple(_to_tuple(v) for v in x) if isinstance(x, (list, tuple)) else x


def _quiet_inductor_autotune_success_logs():
    """探索は維持し、成功した max-autotune 候補の大量表示だけ止める（008 と同じ）。"""
    import torch._inductor.select_algorithm as select_algorithm

    select_algorithm.PRINT_AUTOTUNE = False
    import torch._inductor.config as inductor_config
    if hasattr(inductor_config, "autotune_num_choices_displayed"):
        inductor_config.autotune_num_choices_displayed = 0
    if hasattr(inductor_config, "max_autotune_report_choices_stats"):
        inductor_config.max_autotune_report_choices_stats = False


def _swap_dirs(tmp, final, old, retries=10, wait=3.0):
    """tmp を final に差し替える（stage1_v2 / ablation と同じ。Windows バインドマウントの PermissionError 対策）。"""
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


class ProjectionModel:
    LOSS_NAMES = ("D", "D_real", "D_fake", "G_GAN", "G_fid", "G_id")
    DIAG_NAMES = ("R_u_rms_HU", "R_y_rms_HU", "R_ratio", "P_rms_HU", "d_in_HU", "lambda_id")

    def __init__(self, train, device, launch_config=None, is_train=True):
        self.cfg = train
        self.device = device
        self.is_train = is_train
        self.launch_config = launch_config
        self.hu_per_unit = (train["data.hu_max"] - train["data.hu_min"]) / 2.0
        self.precision = train["optim.precision"]
        self.use_bf16 = self.precision == "bf16" and device.type == "cuda"
        self.compile_requested = train["optim.compile"]
        self.compile_enabled = self.compile_requested and device.type == "cuda"
        if self.compile_enabled and not (shutil.which(os.environ.get("CC", "cc")) or shutil.which("gcc")):
            raise RuntimeError("torch.compile に必要な C コンパイラがありません。Docker image を再ビルドしてください: bash start1_v3.sh <machine> down → bash start1_v3.sh <machine> build")
        self.netG = networks.build_G(train, device)
        self.model_names = list(NET_NAMES) if is_train else ["G"]
        if is_train:
            self.netD = networks.build_D(train, device)
            if self.compile_enabled:
                _quiet_inductor_autotune_success_logs()
                for net in self.nets().values():  # nn.Module.compile は identity と state_dict のキーを保つ
                    net.compile(mode="max-autotune")
            self.gan_mode = train["loss.gan_mode"]
            if self.gan_mode != "fgan_kl":
                self.criterionGAN = networks.GANLoss(self.gan_mode).to(device)
            self.lambda_fid = train["loss.lambda_fid"]
            betas = (train["optim.beta1"], train["optim.beta2"])
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=train["optim.lr_G"], betas=betas)
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=train["optim.lr_D"], betas=betas)
            self.optimizers = [self.optimizer_G, self.optimizer_D]
            self.schedulers = []
            self.cur_epoch, self.epoch_done, self.total_iters = 1, False, 0
        self.losses = {}

    # ------------------------------------------------------------------ 補助
    def nets(self):
        return {n: getattr(self, "net" + n) for n in self.model_names}

    def n_params(self):
        return {n: networks.count_params(net) for n, net in self.nets().items()}

    def make_schedulers(self, epoch_count):
        self.schedulers = [networks.linear_scheduler(o, epoch_count, self.cfg["optim.n_epochs"], self.cfg["optim.n_epochs_decay"]) for o in self.optimizers]

    def update_learning_rate(self):
        for s in self.schedulers:
            s.step()

    @property
    def lr_G(self):
        return float(self.optimizer_G.param_groups[0]["lr"])

    @property
    def lr_D(self):
        return float(self.optimizer_D.param_groups[0]["lr"])

    def lambda_id(self, epoch=None):
        """λ_id のスケジュール: epoch 1 で lambda_id、lambda_id_decay_epochs 経過で lambda_id_final、以後一定。0 なら一定。"""
        c = self.cfg
        lam0, lam1, n = c["loss.lambda_id"], c["loss.lambda_id_final"], c["loss.lambda_id_decay_epochs"]
        if n <= 0:
            return lam0
        e = (self.cur_epoch if epoch is None else epoch) - 1
        t = min(max(e / float(n), 0.0), 1.0)
        return lam0 + (lam1 - lam0) * t

    @staticmethod
    def set_requires_grad(nets, requires_grad=False):
        for net in nets:
            for p in net.parameters():
                p.requires_grad = requires_grad

    def train_mode(self, flag=True):
        for net in self.nets().values():
            net.train(flag)

    def _autocast(self):
        """BF16 は GradScaler 不要。重みと optimizer state は FP32 のまま。"""
        return torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_bf16)

    def _run_net(self, net, x):
        """compiled module の CUDA Graph 出力を次の呼び出しによる上書きから守る（出力を FP32 に戻す）。"""
        with self._autocast():
            y = net(x)
        y = y.float()
        return y.clone() if self.compile_enabled else y

    # ------------------------------------------------------------------ 論文 Eq.(6) の D の値域: D := 1 − exp(v)
    @staticmethod
    def _D_from_v(v):
        return 1.0 - torch.exp(v)

    # ------------------------------------------------------------------ forward / backward
    def set_input(self, batch):
        self.real_A = batch["A"].to(self.device, non_blocking=True)
        self.u = batch["U"].to(self.device, non_blocking=True)
        self.real_B = batch["B"].to(self.device, non_blocking=True)

    def forward(self):
        """PCD 経路と EID 経路を連結して 1 回で G に通す（BN 対策）。"""
        n = self.u.shape[0]
        inp = torch.cat([self.u, self.real_B], dim=0)
        out = self._run_net(self.netG, inp)
        self.fake_B, self.idt_B = out[:n], out[n:]
        self.R_u, self.R_y = self.fake_B - self.u, self.idt_B - self.real_B

    def backward_D(self):
        v_real = self._run_net(self.netD, self.real_B)
        v_fake = self._run_net(self.netD, self.fake_B.detach())
        if self.gan_mode == "fgan_kl":
            D_real = self._D_from_v(v_real)
            J_D = D_real.mean() + v_fake.mean()  # E[D(x)] + E[log(1 − D(G(z)))]、log(1 − D) = v
            self.loss_D = -J_D
            self.loss_D_real = D_real.mean().detach()
            self.loss_D_fake = self._D_from_v(v_fake.clamp(max=80.0)).mean().detach()
        else:
            self.loss_D_real = self.criterionGAN(v_real, True)
            self.loss_D_fake = self.criterionGAN(v_fake, False)
            self.loss_D = self.loss_D_real + self.loss_D_fake
        self.loss_D.backward()

    def backward_G(self):
        v_fake = self._run_net(self.netD, self.fake_B)
        if self.gan_mode == "fgan_kl":
            self.loss_G_GAN = v_fake.mean()  # E[log(1 − D(G(z)))] = E[v]
        else:
            self.loss_G_GAN = self.criterionGAN(v_fake, True)
        self.loss_G_fid = torch.mean(self.R_u ** 2)
        self.loss_G_id = torch.mean(self.R_y ** 2)
        lam_id = self.lambda_id()
        self.loss_G = self.loss_G_GAN + self.lambda_fid * self.loss_G_fid + lam_id * self.loss_G_id
        self.loss_G.backward()
        return lam_id

    def optimize_parameters(self):
        """1 iteration: forward → G 更新 → D 更新（stage1 と同じ順序）。"""
        self.forward()
        self.set_requires_grad([self.netD], False)
        self.optimizer_G.zero_grad(set_to_none=True)
        lam_id = self.backward_G()
        self.optimizer_G.step()
        self.set_requires_grad([self.netD], True)
        self.optimizer_D.zero_grad(set_to_none=True)
        self.backward_D()
        self.optimizer_D.step()
        with torch.no_grad():
            h = self.hu_per_unit
            r_u = float(self.R_u.detach().pow(2).mean().sqrt()) * h
            r_y = float(self.R_y.detach().pow(2).mean().sqrt()) * h
            diag = {"R_u_rms_HU": r_u, "R_y_rms_HU": r_y, "R_ratio": r_u / max(r_y, 1e-6),
                    "P_rms_HU": float((self.u - self.real_A).pow(2).mean().sqrt()) * h,
                    "d_in_HU": float((self.fake_B.detach() - self.real_A).abs().mean()) * h, "lambda_id": lam_id}
        self.losses = {k: float(getattr(self, "loss_" + k).detach()) for k in self.LOSS_NAMES}
        self.losses["G_total"] = float(self.loss_G.detach())
        self.losses.update(diag)
        return self.losses

    # ------------------------------------------------------------------ 推論（G だけ。P は呼び出し側）
    @torch.no_grad()
    def predict(self, x):
        """x: (N,1,H,W) 正規化済み → G(x) = x + R(x)。eval で通す（BN は running 統計）。"""
        was = self.netG.training
        self.netG.eval()
        y = self._run_net(self.netG, x.to(self.device))
        self.netG.train(was)
        return y

    @torch.no_grad()
    def views(self, batch):
        """表示用: バッチ先頭の A / U / fake_B / B / idt_B（eval で通す）。"""
        self.train_mode(False)
        a, u, b = batch["A"].to(self.device), batch["U"].to(self.device), batch["B"].to(self.device)
        out = self._run_net(self.netG, torch.cat([u, b], dim=0))
        self.train_mode(True)
        n = u.shape[0]
        return {"A": a, "U": u, "fake_B": out[:n], "B": b, "idt_B": out[n:]}

    # ------------------------------------------------------------------ 保存 / 再開
    def _state(self):
        np_state = np.random.get_state()
        return {
            "epoch": int(self.cur_epoch), "epoch_done": bool(self.epoch_done), "total_iters": int(self.total_iters),
            "optimizer_G": self.optimizer_G.state_dict(), "optimizer_D": self.optimizer_D.state_dict(),
            "lr_G": self.lr_G, "lr_D": self.lr_D, "config": self.launch_config,
            "rng": {"python": random.getstate(), "torch": torch.get_rng_state(),
                    "numpy": (np_state[0], torch.from_numpy(np_state[1].astype(np.int64)), int(np_state[2]), int(np_state[3]), float(np_state[4]))},
        }

    def save(self, run_dir, tag):
        final = ckpt_dir(run_dir, tag)
        tmp, old = final.with_name(final.name + ".tmp"), final.with_name(final.name + ".old")
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True, exist_ok=True)
        for name, net in self.nets().items():
            torch.save(net.state_dict(), tmp / f"net_{name}.pth")
        torch.save(self._state(), tmp / "state.pth")
        _swap_dirs(tmp, final, old)

    def load_weights(self, weight_dir):
        for name, net in self.nets().items():
            path = weight_dir / f"net_{name}.pth"
            if not path.is_file():
                raise FileNotFoundError(f"重みがありません: {path}（重みディレクトリの中身: {CKPT_KINDS}）")
            net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            net.to(self.device)

    def resume(self, state_path, epoch_count):
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        self.optimizer_G.load_state_dict(state["optimizer_G"])
        self.optimizer_D.load_state_dict(state["optimizer_D"])
        for optimizer, lr in ((self.optimizer_G, self.cfg["optim.lr_G"]), (self.optimizer_D, self.cfg["optim.lr_D"])):
            for g in optimizer.param_groups:
                g["lr"] = lr
                g["initial_lr"] = lr
                g["betas"] = (self.cfg["optim.beta1"], self.cfg["optim.beta2"])
        self.make_schedulers(epoch_count)
        rng = state["rng"]
        random.setstate(_to_tuple(rng["python"]))
        torch.set_rng_state(rng["torch"].cpu())
        kind, keys, pos, has_gauss, cached = rng["numpy"]
        np.random.set_state((kind, keys.cpu().numpy().astype(np.uint32), pos, has_gauss, cached))
        self.total_iters = int(state["total_iters"])
        print(f"[resume] {state_path}: epoch {state['epoch']} (done={state['epoch_done']}), total_iters {state['total_iters']}, lr_G {self.lr_G:.2e}, lr_D {self.lr_D:.2e}")
        return state
