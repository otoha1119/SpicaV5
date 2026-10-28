"""ablation/SRCycleGAN/models/sr_cyclegan_model.py — SR-CycleGAN（Zheng et al., JMI 2022）のモデル。
著者公開コード models/medical_cycle_gan_model.py（SpicaV2 commit 741924c）の移植。junyanz の BaseModel には依存せず、メソッド構成（forward / backward_D_basic /
backward_D_A / backward_D_B / backward_G / optimize_parameters）は 741924c と同じに保ち、行対応で照合できるようにしてある。

【記号の対応】（論文 Fig. 3 / Eq. 1〜8、コードの変数名）
  real_A = x   : LR（論文 clinical CT、本研究 EID512 patch 128）      netG_A = G1 : A → B（×scale、PixelShuffle）   netD_A = D1 : real_B vs fake_B
  real_B = y   : HR（論文 µCT、本研究 PCD1024 patch 256）             netG_B = G2 : B → A（1/scale、downblock）    netD_B = D2 : real_A vs fake_A
  fake_B = x_SR = G1(x),  rec_A = G2(G1(x)),  fake_A = y_LR = G2(y),  rec_B = G1(G2(y))
  f↓ = AvgPool2d(scale)（論文 "f↓ is average pooling"）, f↑ = UpsamplingNearest2d(scale)（"f↑ is nearest-neighbor upsampling"）

【損失】 論文 Eq. 8（重みは train.yaml loss.*。JMI Table 1: λ1 = λ2 = 1.0, λ3 = 0.7, λ4 = 0.3, λ5 = λ6 = λ7 = 1.0）
  G: λ5 L_GAN(D1(fake_B)) + λ6 L_GAN(D2(fake_A)) + λ7 (L_cyc(rec_A, real_A) + L_cyc(rec_B, real_B))
     + λ1 (1 − SSIM(f↓(fake_B), real_A)) + λ2 (1 − SSIM(f↑(fake_A), real_B)) + λ3 MSE(f↓(fake_B), real_A) + λ4 MSE(f↑(fake_A), real_B)
     (+ identity: λ7·λ_idt·‖f↓(G1(real_B)) − real_B‖₁ + λ7·λ_idt·identity_b_scale·‖f↑(G2(real_A)) − real_A‖₁。コード 741924c にだけある。論文 Eq. 8 に無いので既定 λ_idt = 0)
  D: 0.5 (L_GAN(D(real), 1) + L_GAN(D(fake_pool), 0))（CycleGAN 本家。ImagePool の履歴から fake を引く）
  L_GAN は gan_mode: vanilla（論文 Eq. 2 の log 形 = BCEWithLogits）| lsgan（コード）。L_cyc は cycle_norm: l2（論文 Eq. 1）| l1（コード）
  コードとの差（ユーザー決定 2026-10-27 で論文に合わせた所）: cycle L1 → L2、λ3 / λ4 1.0 → 0.7 / 0.3、identity 0.2 → 0、lsgan → vanilla、instance → batch、SSIM C1 / C2 → SPIE 版 0.02 / 0.06。
  741924c の idt_B は UpsamplingBilinear2d だったが、f↑ は論文どおり nearest に統一（identity は既定で無効）。

【診断値】 diag/d_in_HU = mean|f↓(G1(x)) − x| の HU 換算（downsample loss の HU 表示。入力の強度を保っているか）
【保存 / 再開】 Stage 1 fidelity_gan_model.py と同じ流儀（原子的保存、state.pth に optimizer ×2 / RNG / 進捗 / 実効設定）。util/run_paths.py のレイアウト。
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
from models.ssim import SSIM
from util.run_paths import CKPT_KINDS, NET_NAMES, ckpt_dir


def _to_tuple(x):
    """random.setstate は入れ子の tuple を要求する。torch.load 後に list になっている場合に戻す。"""
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


class SRCycleGANModel:
    LOSS_NAMES = ("D_A", "G_A", "cycle_A", "idt_A", "D_B", "G_B", "cycle_B", "idt_B", "ssim_A", "ssim_B", "down", "up")  # 741924c の loss_names + MMSR の内訳

    def __init__(self, train, machine, device, launch_config=None, is_train=True):
        """train: train.yaml の実効値（平坦 dict）。machine: machines.yaml の実効値（平坦 dict）。launch_config: state.pth に写す実効設定（{"train":..., "machine":...}）。"""
        self.cfg = train
        self.device = device
        self.is_train = is_train
        self.launch_config = launch_config
        self.scale = train["network.scale"]
        self.hu_per_unit = (train["data.hu_max"] - train["data.hu_min"]) / 2.0  # 正規化 [−1,1] の 1.0 が何 HU か
        # 741924c __init__: define networks (both Generators and discriminators)
        self.netG_A = networks.build_G1(train, device)  # G1: clinical → micro（A → B、×scale）
        self.netG_B = networks.build_G2(train, device)  # G2: micro → clinical（B → A、1/scale）
        self.model_names = list(NET_NAMES) if is_train else ["G_A"]
        self.f_down = nn.AvgPool2d(self.scale, stride=self.scale)      # 741924c: AvgPool2d(8, stride=8)
        self.f_up = nn.UpsamplingNearest2d(scale_factor=self.scale)   # 741924c: UpsamplingNearest2d(scale_factor=8)
        if is_train:
            self.netD_A = networks.build_D(train, device)  # D1: real_B vs fake_B（HR 側）
            self.netD_B = networks.build_D(train, device)  # D2: real_A vs fake_A（LR 側）
            self.fake_A_pool = ImagePool(train["loss.pool_size"])
            self.fake_B_pool = ImagePool(train["loss.pool_size"])
            self.criterionGAN = networks.GANLoss(train["loss.gan_mode"]).to(device)
            self.criterionCycle = nn.MSELoss() if train["loss.cycle_norm"] == "l2" else nn.L1Loss()  # 論文 Eq. 1 = squared l2 / コード = L1
            self.criterionIdt = nn.L1Loss()  # 741924c（identity はコードにだけある）
            self.criterionMSE = nn.MSELoss()  # 論文 Eq. 5 / 6 の MSE（741924c は criterionCycle = L1 を流用していたが、論文は MSE と明記）
            self.ssim = SSIM(train["loss.ssim_window"], train["loss.ssim_c1"], train["loss.ssim_c2"]).to(device)
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

    # ------------------------------------------------------------------ 741924c と同じメソッド構成
    def set_input(self, input):
        """dataloader の出力 {"A": (N,1,p,p), "B": (N,1,p·scale,p·scale)} を展開する（741924c の 'clinical' / 'micro'）。"""
        self.real_A = input["A"].to(self.device, non_blocking=True)
        self.real_B = input["B"].to(self.device, non_blocking=True)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator（741924c と同一。(real + fake) × 0.5）"""
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B（論文 Eq. 8。741924c backward_G の構成）"""
        c = self.cfg
        lambda_idt = c["loss.lambda_identity"]
        lambda_cyc = c["loss.lambda_cycle"]  # 741924c の lambda_A = lambda_B（= 論文 λ7）
        # Identity loss（741924c。論文 Eq. 8 に無いので既定 0）
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||f↓(G_A(B)) − B||（741924c: idt_A = AvgPool2d(8)(G_A(real_B))）
            self.idt_A = self.f_down(self.netG_A(self.real_B))
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_cyc * lambda_idt
            # G_B should be identity if real_A is fed: ||f↑(G_B(A)) − A||（741924c: idt_B = Upsampling(8)(G_B(real_A))、× 0.1）
            self.idt_B = self.f_up(self.netG_B(self.real_A))
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_cyc * lambda_idt * c["loss.identity_b_scale"]
        else:
            self.loss_idt_A = torch.zeros((), device=self.device)
            self.loss_idt_B = torch.zeros((), device=self.device)
        # GAN loss D_A(G_A(A))（λ5）
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True) * c["loss.lambda_gan_A"]
        # GAN loss D_B(G_B(B))（λ6）
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True) * c["loss.lambda_gan_B"]
        # Forward cycle loss || G_B(G_A(A)) - A||（λ7）
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_cyc
        # Backward cycle loss || G_A(G_B(B)) - B||（λ7）
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_cyc
        # MMSR loss（論文 Eq. 4〜7）: downsample fake micro to clinical scale / upsample fake clinical to micro scale
        self.downsample_fake_B = self.f_down(self.fake_B)  # f↓(x_SR)
        self.upsample_fake_A = self.f_up(self.fake_A)      # f↑(y_LR)
        self.loss_down = self.criterionMSE(self.downsample_fake_B, self.real_A) * c["loss.lambda_down"]            # λ3 L_D（Eq. 5）
        self.loss_up = self.criterionMSE(self.upsample_fake_A, self.real_B) * c["loss.lambda_up"]                  # λ4 L_U（Eq. 6）
        self.loss_ssim_A = (1.0 - self.ssim(self.downsample_fake_B, self.real_A)) * c["loss.lambda_ssim_A"]       # λ1 L_S(x, f↓(x_SR))（Eq. 4）
        self.loss_ssim_B = (1.0 - self.ssim(self.upsample_fake_A, self.real_B)) * c["loss.lambda_ssim_B"]         # λ2 L_S(y, f↑(y_LR))
        # combined loss and calculate gradients（741924c: loss_G = G_A + G_B + cycle_A + cycle_B + idt_A + idt_B + downsample + upsample + clinical_ssim + micro_ssim）
        self.loss_G = (self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
                       + self.loss_down + self.loss_up + self.loss_ssim_A + self.loss_ssim_B)
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration（741924c と同じ順序: G → D）"""
        self.forward()
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.backward_D_A()
        self.backward_D_B()
        self.optimizer_D.step()
        with torch.no_grad():
            d_in_hu = float((self.downsample_fake_B.detach() - self.real_A).abs().mean()) * self.hu_per_unit
        self.losses = {k: float(getattr(self, "loss_" + k).detach()) for k in self.LOSS_NAMES}
        self.losses["G_total"] = float(self.loss_G.detach())
        self.losses["d_in_HU"] = d_in_hu
        return self.losses

    # ------------------------------------------------------------------ 推論（G1 のみ）
    @torch.no_grad()
    def predict(self, x):
        """x: (N,1,H,W) 正規化済み → G1(x): (N,1,H·scale,W·scale)。BN は eval（running 統計）。"""
        was = self.netG_A.training
        self.netG_A.eval()
        y = self.netG_A(x.to(self.device))
        self.netG_A.train(was)
        return y

    @torch.no_grad()
    def cycle_views(self, batch):
        """表示用: 現在のバッチ先頭 n 枚の fake_B / rec_A / fake_A / rec_B（eval で通す）。"""
        self.train_mode(False)
        a, b = batch["A"].to(self.device), batch["B"].to(self.device)
        fake_B = self.netG_A(a)
        rec_A = self.netG_B(fake_B)
        fake_A = self.netG_B(b)
        rec_B = self.netG_A(fake_A)
        self.train_mode(True)
        return {"A": a, "fake_B": fake_B, "rec_A": rec_A, "B": b, "fake_A": fake_A, "rec_B": rec_B}

    # ------------------------------------------------------------------ 保存 / 再開（Stage 1 fidelity_gan_model.py と同じ流儀）
    def _state(self):
        np_state = np.random.get_state()
        return {
            "epoch": int(self.cur_epoch),
            "epoch_done": bool(self.epoch_done),
            "total_iters": int(self.total_iters),
            "optimizer_G": self.optimizer_G.state_dict(),
            "optimizer_D": self.optimizer_D.state_dict(),
            "lr": self.lr,
            "config": self.launch_config,  # 保存時の実効設定（train / machine）。再開はこれを基準にする
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
        """重みディレクトリの net_*.pth を strict に読む（学習時は 4 つ、推論時は G_A だけ）。"""
        for name, net in self.nets().items():
            path = weight_dir / f"net_{name}.pth"
            if not path.is_file():
                raise FileNotFoundError(f"重みがありません: {path}（重みディレクトリの中身: {CKPT_KINDS}）")
            state = torch.load(path, map_location="cpu", weights_only=True)
            net.load_state_dict(state)
            net.to(self.device)

    def resume(self, state_path, epoch_count):
        """state.pth から optimizer ×2 / RNG / 進捗を復元し、lr / betas を今の実効値に戻してから scheduler を作り直す。"""
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        self.optimizer_G.load_state_dict(state["optimizer_G"])
        self.optimizer_D.load_state_dict(state["optimizer_D"])
        for optimizer in self.optimizers:  # 再開時の --lr / --beta を最優先に（load_state_dict は保存時の値を戻す）
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
