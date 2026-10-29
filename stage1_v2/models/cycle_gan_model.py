"""stage1_v2/models/cycle_gan_model.py — 標準 CycleGAN（Zhu et al., ICCV 2017）。
junyanz の BaseModel には依存せず、メソッド構成（forward / backward_D_basic / backward_D_A / backward_D_B / backward_G / optimize_parameters）は本家と同じに保つ。
保存・再開・表示用の補助は SpicaV5 の run レイアウトに合わせている。

【記号の対応】（論文 Fig. 3 / 本家の変数名）
  real_A = z（PCD512 patch）   netG_A = G : PCD → EID    netD_A = D_EID : real_B vs fake_B
  real_B = x（EID512 patch）   netG_B = F : EID → PCD    netD_B = D_PCD : real_A vs fake_A
  fake_B = G(z),  rec_A = F(G(z)),  fake_A = F(x),  rec_B = G(F(x))。

【損失】 論文 Eq. 3 + 本家の identity（train.yaml loss.*）
  G: L_GAN(D_Y(fake_B)) + L_GAN(D_X(fake_A)) + λ_A ‖rec_A − real_A‖₁ + λ_B ‖rec_B − real_B‖₁ (+ λ_B λ_idt ‖G(real_B) − real_B‖₁ + λ_A λ_idt ‖F(real_A) − real_A‖₁)
  D: 0.5 (L_GAN(D(real), 1) + L_GAN(D(fake_pool), 0))（ImagePool の履歴から fake を引く）
  L_GAN は gan_mode: lsgan（本家既定 = 論文 Sec. 4）| vanilla
【診断値】 diag/d_in_HU = mean|G(z) − z| の HU 換算
"""

import itertools
import os
import random
import shutil
import time

import numpy as np
import torch
import torch.nn as nn

from models import networks
from models.image_pool import ImagePool
from util.run_paths import CKPT_KINDS, NET_NAMES, ckpt_dir


def _to_tuple(x):
    return tuple(_to_tuple(v) for v in x) if isinstance(x, (list, tuple)) else x


def _swap_dirs(tmp, final, old, retries=10, wait=3.0):
    """tmp を final に差し替える。既存の final は old に退避してから消す。改名が拒否されたら wait 秒待って再試行し、最後はファイル単位の置き換えに落とす。
    final が無く old だけあるのは「退避まで済んで差し替えで止まった」状態なので、old は直前の checkpoint として消さない。戻り値: 成功したか。"""
    err = None
    for _ in range(retries):
        try:
            if final.exists() and old.exists():  # 前回の中断の残骸
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
    try:  # フォールバック: ディレクトリは触らず、中のファイルを 1 つずつ置き換える（フォルダのハンドルが掴まれていても通る）
        final.mkdir(parents=True, exist_ok=True)
        for f in sorted(tmp.iterdir()):
            os.replace(f, final / f.name)
        shutil.rmtree(tmp, ignore_errors=True)
        if old.exists():
            shutil.rmtree(old, ignore_errors=True)
        print(f"[save] 注意: {final} のフォルダ改名が拒否されたのでファイル単位で置き換えた（{err}）", flush=True)
        return True
    except (PermissionError, FileNotFoundError) as e2:
        print(f"[save] 警告: checkpoint {final} を保存できなかった（{e2}）。学習は続け、次回の保存で再試行する。{tmp} に今回の重みが残っている", flush=True)
        return False


class CycleGANModel:
    LOSS_NAMES = ("D_A", "G_A", "cycle_A", "idt_A", "D_B", "G_B", "cycle_B", "idt_B")  # 本家 loss_names

    def __init__(self, train, machine, device, launch_config=None, is_train=True):
        self.cfg = train
        self.device = device
        self.is_train = is_train
        self.launch_config = launch_config
        self.hu_per_unit = (train["data.hu_max"] - train["data.hu_min"]) / 2.0
        # 本家 __init__: define networks (both Generators and discriminators)
        self.netG_A = networks.build_G(train, device)  # G: A → B
        self.netG_B = networks.build_G(train, device)  # F: B → A
        self.model_names = list(NET_NAMES) if is_train else ["G_A"]
        if is_train:
            self.netD_A = networks.build_D(train, device)  # D_Y
            self.netD_B = networks.build_D(train, device)  # D_X
            self.fake_A_pool = ImagePool(train["loss.pool_size"])
            self.fake_B_pool = ImagePool(train["loss.pool_size"])
            self.criterionGAN = networks.GANLoss(train["loss.gan_mode"]).to(device)
            self.criterionCycle = nn.L1Loss()
            self.criterionIdt = nn.L1Loss()
            lr, betas = train["optim.lr"], (train["optim.beta1"], train["optim.beta2"])
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=lr, betas=betas)
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=lr, betas=betas)
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
    def lr(self):
        return float(self.optimizers[0].param_groups[0]["lr"])

    @staticmethod
    def set_requires_grad(nets, requires_grad=False):
        for net in nets:
            for p in net.parameters():
                p.requires_grad = requires_grad

    def train_mode(self, flag=True):
        for net in self.nets().values():
            net.train(flag)

    # ------------------------------------------------------------------ 本家と同じメソッド構成
    def set_input(self, input):
        self.real_A = input["A"].to(self.device, non_blocking=True)
        self.real_B = input["B"].to(self.device, non_blocking=True)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator（本家。(real + fake) × 0.5）"""
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B（本家）"""
        c = self.cfg
        lambda_idt, lambda_A, lambda_B = c["loss.lambda_identity"], c["loss.lambda_A"], c["loss.lambda_B"]
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = torch.zeros((), device=self.device)
            self.loss_idt_B = torch.zeros((), device=self.device)
        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()

    def optimize_parameters(self):
        """1 iteration: forward → G 更新 → D 更新（本家と同じ順序）"""
        self.forward()
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.backward_D_A()
        self.backward_D_B()
        self.optimizer_D.step()
        with torch.no_grad():
            d_in_hu = float((self.fake_B.detach() - self.real_A).abs().mean()) * self.hu_per_unit
        self.losses = {k: float(getattr(self, "loss_" + k).detach()) for k in self.LOSS_NAMES}
        self.losses["G_total"] = float(self.loss_G.detach())
        self.losses["d_in_HU"] = d_in_hu
        return self.losses

    # ------------------------------------------------------------------ 推論（G_A のみ）
    @torch.no_grad()
    def predict(self, x):
        """x: (N,1,H,W) 正規化済み → G_A(x)（同じ大きさ）。eval で通す（InstanceNorm は train / eval で同じ）。"""
        was = self.netG_A.training
        self.netG_A.eval()
        y = self.netG_A(x.to(self.device))
        self.netG_A.train(was)
        return y

    @torch.no_grad()
    def cycle_views(self, batch):
        """表示用: バッチ先頭 n 枚の fake_B / rec_A / fake_A / rec_B（eval で通す）。"""
        self.train_mode(False)
        a, b = batch["A"].to(self.device), batch["B"].to(self.device)
        fake_B = self.netG_A(a)
        rec_A = self.netG_B(fake_B)
        fake_A = self.netG_B(b)
        rec_B = self.netG_A(fake_A)
        self.train_mode(True)
        return {"A": a, "fake_B": fake_B, "rec_A": rec_A, "B": b, "fake_A": fake_A, "rec_B": rec_B}

    # ------------------------------------------------------------------ 保存 / 再開
    def _state(self):
        np_state = np.random.get_state()
        return {
            "epoch": int(self.cur_epoch),
            "epoch_done": bool(self.epoch_done),
            "total_iters": int(self.total_iters),
            "optimizer_G": self.optimizer_G.state_dict(),
            "optimizer_D": self.optimizer_D.state_dict(),
            "lr": self.lr,
            "config": self.launch_config,
            "rng": {
                "python": random.getstate(),
                "torch": torch.get_rng_state(),
                "numpy": (np_state[0], torch.from_numpy(np_state[1].astype(np.int64)), int(np_state[2]), int(np_state[3]), float(np_state[4])),
            },
        }

    def save(self, run_dir, tag):
        """重み 4 つと state.pth を**同じ重みディレクトリに原子的に**保存する（<dir>.tmp に書く → 既存を .old に → rename → .old を消す）。
        Windows のバインドマウント（Docker Desktop）では、フォルダ内のファイルを別プロセス（Explorer / ウイルス対策 / 同期ソフト）が掴んでいるとフォルダの改名が
        PermissionError になる（PC3 で epoch 52 に発生、2026-10-28）。_swap_dirs が再試行し、それでも駄目ならファイル単位で置き換え、それも駄目なら**学習を止めずに警告**して次回の保存に委ねる。"""
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
        for optimizer in self.optimizers:
            for g in optimizer.param_groups:
                g["lr"] = self.cfg["optim.lr"]
                g["initial_lr"] = self.cfg["optim.lr"]
                g["betas"] = (self.cfg["optim.beta1"], self.cfg["optim.beta2"])
        self.make_schedulers(epoch_count)
        rng = state["rng"]
        random.setstate(_to_tuple(rng["python"]))
        torch.set_rng_state(rng["torch"].cpu())
        kind, keys, pos, has_gauss, cached = rng["numpy"]
        np.random.set_state((kind, keys.cpu().numpy().astype(np.uint32), pos, has_gauss, cached))
        self.total_iters = int(state["total_iters"])
        print(f"[resume] {state_path}: epoch {state['epoch']} (done={state['epoch_done']}), total_iters {state['total_iters']}, lr {self.lr:.2e}")
        return state
