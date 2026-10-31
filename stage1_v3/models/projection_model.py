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
  D も本物 y と偽物 ŷ を連結して通す（D 更新・G 更新とも。別 forward だと偽物バッチの HU の水準差が 1 層目の BN で消えて D に見えず、R の HU 水準が揺れる。2026-10-30 の 1 本目で判明）。
【高速化（008 と同じ）】 optim.precision bf16 = CUDA autocast（G / D の forward だけ。損失は FP32 に戻して計算。fgan_kl の exp は FP32）。optim.compile = nn.Module.compile(mode="max-autotune")。
  compile 時は CUDA Graph が出力 buffer を再利用するので、保持する出力は _run_net で clone する（D は 1 iteration に 3 回呼ぶ）
【learned P（degrade.mode learned、2026-10-30 の転換）】 u = P_θ(x, z, ε) を models/degrade_net.py が GPU で作る（dataset は u = x を渡す）。P_θ は G と同じ optimizer。
  損失に λ_k_fid · mean((x ⊛ k − x)²)（カーネル側だけの fidelity）と λ_com · COM² を足す。ノイズ側には fidelity を掛けない。net_P.pth を保存し推論でも使う
【診断値（HU）】 R_u_rms / R_y_rms（残差の RMS）、R_ratio = R_u_rms / R_y_rms（≫ 1 かつ D 分離維持 = R がドメイン検出器化、レビュー D）、P_rms = RMS(u − x)、d_in = mean|fake_B − x|
【v4（pipeline.mode v4、2026-10-31、docs/decisions/20261031_stage1-v3-noise-path-analysis.md §3）】 順序を入れ替え、PCD ノイズの除去を凍結 denoiser Dn に任せる:
  x → Dn（no_grad、eval、optimizer に入れない、compile しない）→ x_c → G: g = x_c + R(x_c) → v = g ⊛ k(z) → u = v + σ(x_c) ⊙ (h(z) ⊛ ε)、σ ∈ [σ_min, σ_max] → D vs y
  y → G → y + R(y)（Dn も k もノイズも通さない。v3 と同じ）
  損失: L_GAN(u) + λ_fid mean(R(x_c)²) + λ_id mean(R(y)²) + λ_k mean((x_c ⊛ k − x_c)²)（同じ z。x_c 基準）+ λ_com COM²。
  fidelity・恒等・カーネル項のどれも ε と σ に依存しない（u を経由する項は L_GAN だけ。C2・C3。この性質を崩す変更をしないこと）
  Dn の重みは pipeline.dn_weights から読み、最初の save で run 直下と各重みディレクトリに net_Dn.pth として複製する（run の自己完結。resume / infer はそちらを優先）
  generate(x, generator, eps_zero) が {x_c, g, v, u} を返す（表示・採点・推論。eps_zero=True = 指紋）。degrade(x) は generate()["u"]。predict(x) は G だけ（恒等の確認）
  追加の診断値: Dn_resid_rms_HU = 体内の RMS(x − x_c)。P_rms_HU は v4 では全経路の RMS(u − x)（v3 の RMS(P(x) − x) とは意味が違う）
【FP32 の約束（v4）】 凍結 Dn の forward と、ε = 0 の指紋（generate(eps_zero=True)）は常に FP32（autocast 無効）。正規化単位 1 = 2748 HU なので bf16 の量子化段差は
  軟部組織で 5〜11 HU になり、指紋 SD ≤ 6 HU の判定に対して無視できない。学習時の G / D / P_θ は従来どおり bf16 autocast
"""

import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from models import networks
from models.degrade_net import LearnedDegrade
from models.denoiser import load_denoiser, save_denoiser
from util.run_paths import CKPT_KINDS, DN_WEIGHT, NET_NAMES, ckpt_dir


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


@torch.no_grad()
def generate_v4(netDn, netG, netP, x, generator=None, eps_zero=False, autocast=None):
    """v4 の生成経路（eval、勾配なし）。x: (N,1,H,W) 正規化済み → {"x_c", "g", "v", "u", "sigma"}（すべて FP32）。
    z と ε は generator から引く。eps_zero=True なら ε = 0（z は通常どおり）= 指紋（決定的に残るノイズ）。学習（ProjectionModel.forward）と同じ順序・同じ部品。
    Dn は常に FP32。eps_zero=True（指紋の計測）は G / P も FP32（autocast を渡されても無効にする。bf16 の段差 5〜11 HU が 6 HU の閾値に効くため）。"""
    off = lambda: torch.autocast(device_type=x.device.type, enabled=False)  # noqa: E731
    ac = autocast if (autocast is not None and not eps_zero) else off
    with off():
        x_c = netDn(x.float()).float()
    with ac():
        g = netG(x_c).float()
        n = x.shape[0]
        z = netP.sample_z(n, x.device, generator)
        eps = torch.zeros_like(x) if eps_zero else torch.randn(x.shape, device=x.device, generator=generator)
        v, info = netP.blur(g, z)
        noise, sigma = netP.noise(x_c, z, eps, h=info["h"])
        u = v + noise
    return {"x_c": x_c, "g": g, "v": v.float(), "u": u.float(), "sigma": sigma.float()}


class ProjectionModel:
    LOSS_NAMES = ("D", "D_real", "D_fake", "G_GAN", "G_fid", "G_id", "P_kfid", "P_com")
    DIAG_NAMES = ("R_u_rms_HU", "R_y_rms_HU", "R_ratio", "P_rms_HU", "d_in_HU", "lambda_id", "P_k_sigma_px", "P_noise_amp_HU", "D_sep", "D_acc", "Dn_resid_rms_HU")

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
        self.v4 = train["pipeline.mode"] == "v4"
        self.learned_P = train["degrade.enabled"] and train["degrade.mode"] == "learned"
        if self.v4 and not self.learned_P:
            raise ValueError("pipeline.mode v4 は degrade.mode learned が必要")
        self.model_names = (["G", "P"] if self.learned_P else ["G"]) + (["D"] if is_train else [])
        if self.learned_P:
            sigma_min = train["degrade.learned_sigma_min_hu"] / self.hu_per_unit if self.v4 else 0.0  # v3 は従来どおり σ = σ_max·sigmoid
            self.netP = LearnedDegrade(train["degrade.learned_z_dim"], train["degrade.learned_kernel_size"], train["degrade.learned_noise_kernel_size"],
                                       train["degrade.learned_amp_ch"], train["degrade.learned_sigma_max_hu"] / self.hu_per_unit, sigma_min_norm=sigma_min).to(device)
            self.lambda_k_fid, self.lambda_k_com = train["loss.lambda_k_fid"], train["loss.lambda_k_com"]
        self.netDn, self.dn_source = None, None
        if self.v4:  # 凍結 denoiser。optimizer に入れず、compile もしない（小さく、出力を Python 側で使うので CUDA Graph の上書きを避ける。P_θ と同じ理由）
            dn_path = Path(train["pipeline.dn_weights"])
            if dn_path.is_file():
                self.netDn, self.dn_source = load_denoiser(dn_path, device), dn_path
            self._dn_copied_to_run = False
        if is_train:
            self.netD = networks.build_D(train, device)
            if self.compile_enabled:
                _quiet_inductor_autotune_success_logs()
                for name, net in self.nets().items():  # nn.Module.compile は identity と state_dict のキーを保つ
                    if name == "P":  # P_θ は小さく（MLP + サンプル別 grouped conv）、途中結果を Python 側で使うので compile しない（CUDA Graph の出力上書きで落ちる。PC1 2026-10-30）
                        continue
                    net.compile(mode="max-autotune")
            self.gan_mode = train["loss.gan_mode"]
            if self.gan_mode != "fgan_kl":
                self.criterionGAN = networks.GANLoss(self.gan_mode).to(device)
            self.lambda_fid = train["loss.lambda_fid"]
            betas = (train["optim.beta1"], train["optim.beta2"])
            g_params = list(self.netG.parameters()) + (list(self.netP.parameters()) if self.learned_P else [])
            self.optimizer_G = torch.optim.Adam(g_params, lr=train["optim.lr_G"], betas=betas)
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=train["optim.lr_D"], betas=betas)
            self.optimizers = [self.optimizer_G, self.optimizer_D]
            self.schedulers = []
            self.cur_epoch, self.epoch_done, self.total_iters = 1, False, 0
        self.losses = {}

    # ------------------------------------------------------------------ 補助
    def nets(self):
        return {n: getattr(self, "net" + n) for n in self.model_names}

    def n_params(self):
        out = {n: networks.count_params(net) for n, net in self.nets().items()}
        if self.netDn is not None:
            out["Dn(frozen)"] = networks.count_params(self.netDn)
        return out

    def _require_dn(self):
        if self.netDn is None:
            raise RuntimeError(f"v4 の Dn の重みが読めていません（pipeline.dn_weights = {self.cfg['pipeline.dn_weights']!r} が無く、run にも net_Dn.pth が無い）")
        return self.netDn

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
        self.u = batch["U"].to(self.device, non_blocking=True)  # learned なら u = x（P_θ は forward で掛ける）
        self.real_B = batch["B"].to(self.device, non_blocking=True)

    def degrade(self, x, generator=None):
        """learned のとき x → u = P_θ(x, z, ε)。eval で通す（表示・採点・推論用。勾配なし）。learned でなければ x をそのまま返す。v4 は全経路（Dn → G → k → ノイズ）の u。"""
        if self.v4:
            return self.generate(x, generator)["u"]
        if not self.learned_P:
            return x
        was = self.netP.training
        self.netP.eval()
        with torch.no_grad(), self._autocast():
            u, _ = self.netP(x.to(self.device), generator=generator)
        self.netP.train(was)
        return u.float()

    def forward(self):
        """PCD 経路と EID 経路を連結して 1 回で G に通す（BN 対策）。v4 は Dn → G → ⊛ k(z) → + ノイズ。"""
        if self.v4:
            return self._forward_v4()
        n = self.u.shape[0]
        if self.learned_P:
            with self._autocast():
                u, info = self.netP(self.real_A)
            self.u = u.float().clone()
            self.P_info = {k: (v.float().clone() if torch.is_tensor(v) else v) for k, v in info.items()}  # 後続の compiled graph に上書きされないよう clone
            self.kfid_ref = self.real_A
        inp = torch.cat([self.u, self.real_B], dim=0)
        out = self._run_net(self.netG, inp)
        self.fake_B, self.idt_B = out[:n], out[n:]
        self.R_u, self.R_y = self.fake_B - self.u, self.idt_B - self.real_B

    def _forward_v4(self):
        x, y = self.real_A, self.real_B
        n = x.shape[0]
        with torch.no_grad(), torch.autocast(device_type=self.device.type, enabled=False):  # Dn は FP32（上の FP32 の約束）
            x_c = self._require_dn()(x.float())
        self.x_c = x_c.float()
        out = self._run_net(self.netG, torch.cat([self.x_c, y], dim=0))
        self.g, self.idt_B = out[:n], out[n:]
        self.R_u, self.R_y = self.g - self.x_c, self.idt_B - y  # fidelity は R(x_c)。ε・σ を含まない
        with self._autocast():
            z = self.netP.sample_z(n, x.device)
            eps = torch.randn(x.shape, device=x.device)
            v, info = self.netP.blur(self.g, z)
            noise, sigma = self.netP.noise(self.x_c, z, eps, h=info["h"])
            u = v + noise
            kfid_blur, _ = self.netP.blur(self.x_c, z)  # カーネル項は x_c 基準・同じ z（1 回余分な conv）
        self.u = u.float()
        self.fake_B = self.u
        self.P_info = {"blurred": kfid_blur.float(), "sigma": sigma.float(), "com2": info["com2"].float(), "k_sigma_px": info["k_sigma_px"].float(), "z": z}
        self.kfid_ref = self.x_c

    def _D_real_fake(self, fake):
        """本物 y と偽物を 1 バッチに連結して D に通す（BN のドメイン分割対策。別 forward だと偽物バッチ全体の HU のずれが BN で消え、D が HU の水準を見られない。2026-10-30 run 2026_1030_1856 で HU 水準が epoch ごとに揺れた原因）。"""
        n = self.real_B.shape[0]
        v = self._run_net(self.netD, torch.cat([self.real_B, fake], dim=0))
        return v[:n], v[n:]

    def backward_D(self):
        v_real, v_fake = self._D_real_fake(self.fake_B.detach())
        with torch.no_grad():  # D の分離の診断: D_sep = mean(v_fake) − mean(v_real)（0 = 見分けていない、正 = D 優勢）、D_acc = patch 単位で v_fake > v_real の割合（0.5 = 当てずっぽう、1 = 完全に分離）
            self.diag_D_sep = float(v_fake.mean() - v_real.mean())
            self.diag_D_acc = float((v_fake.flatten() > v_real.flatten()).float().mean())
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
        _, v_fake = self._D_real_fake(self.fake_B)  # G の更新でも本物と連結（偽物だけだと BN が水準差を消し、G に HU の勾配が届かない）
        if self.gan_mode == "fgan_kl":
            self.loss_G_GAN = v_fake.mean()  # E[log(1 − D(G(z)))] = E[v]
        else:
            self.loss_G_GAN = self.criterionGAN(v_fake, True)
        self.loss_G_fid = torch.mean(self.R_u ** 2)
        self.loss_G_id = torch.mean(self.R_y ** 2)
        lam_id = self.lambda_id()
        self.loss_G = self.loss_G_GAN + self.lambda_fid * self.loss_G_fid + lam_id * self.loss_G_id
        if self.learned_P:  # カーネル側の fidelity と重心の正則化（ノイズ側には掛けない）
            self.loss_P_kfid = torch.mean((self.P_info["blurred"].float() - self.kfid_ref) ** 2)  # v3: x 基準、v4: x_c 基準
            self.loss_P_com = self.P_info["com2"].float().mean()
            self.loss_G = self.loss_G + self.lambda_k_fid * self.loss_P_kfid + self.lambda_k_com * self.loss_P_com
        else:
            self.loss_P_kfid = torch.zeros((), device=self.device)
            self.loss_P_com = torch.zeros((), device=self.device)
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
                    "d_in_HU": float((self.fake_B.detach() - self.real_A).abs().mean()) * h, "lambda_id": lam_id,
                    "D_sep": self.diag_D_sep, "D_acc": self.diag_D_acc}
            if self.learned_P:
                body = self.real_A > (-500.0 - self.cfg["data.hu_min"]) / h - 1.0
                sig = self.P_info["sigma"].detach().float()
                diag["P_k_sigma_px"] = float(self.P_info["k_sigma_px"].detach().float().mean())
                diag["P_noise_amp_HU"] = float(sig[body].mean()) * h if body.any() else float(sig.mean()) * h
                if self.v4:
                    dr = (self.real_A - self.x_c)
                    diag["Dn_resid_rms_HU"] = float(dr[body].pow(2).mean().sqrt()) * h if body.any() else float(dr.pow(2).mean().sqrt()) * h
        self.losses = {k: float(getattr(self, "loss_" + k).detach()) for k in self.LOSS_NAMES}
        self.losses["G_total"] = float(self.loss_G.detach())
        self.losses.update(diag)
        return self.losses

    @torch.no_grad()
    def generate(self, x, generator=None, eps_zero=False):
        """v4: x → {x_c, g, v, u, sigma}（eval、勾配なし。generate_v4）。eps_zero=True で ε = 0（指紋）。"""
        if not self.v4:
            raise RuntimeError("generate は pipeline.mode v4 のみ（v3 は degrade / predict）")
        was_g, was_p = self.netG.training, self.netP.training
        self.netG.eval(); self.netP.eval()
        out = generate_v4(self._require_dn(), self.netG, self.netP, x.to(self.device), generator=generator, eps_zero=eps_zero, autocast=self._autocast)
        self.netG.train(was_g); self.netP.train(was_p)
        if self.compile_enabled:
            out = {k: v.clone() for k, v in out.items()}
        return out

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
        if self.v4:  # [x | Dn(x) | u | R(x_c) | y | y+R(y) | R(y)]（U = x_c、fake_B = u、R = g − x_c）
            gen = self.generate(a, generator=torch.Generator(device=self.device).manual_seed(0))
            idt = self._run_net(self.netG, b)
            self.train_mode(True)
            return {"A": a, "U": gen["x_c"], "fake_B": gen["u"], "B": b, "idt_B": idt, "R": gen["g"] - gen["x_c"]}
        if self.learned_P:
            u = self.degrade(a, generator=torch.Generator(device=self.device).manual_seed(0))
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
        if self.v4:  # 凍結 Dn も各重みディレクトリに複製（best / infer / mark_best が重みディレクトリだけで完結する）。run 直下には最初の 1 回だけ
            save_denoiser(self._require_dn(), tmp / DN_WEIGHT)
            if not self._dn_copied_to_run and not (Path(run_dir) / DN_WEIGHT).is_file():
                save_denoiser(self.netDn, Path(run_dir) / DN_WEIGHT)
            self._dn_copied_to_run = True
        _swap_dirs(tmp, final, old)

    def load_weights(self, weight_dir):
        for name, net in self.nets().items():
            path = weight_dir / f"net_{name}.pth"
            if not path.is_file():
                raise FileNotFoundError(f"重みがありません: {path}（重みディレクトリの中身: {CKPT_KINDS}）")
            net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            net.to(self.device)
        if self.v4:  # Dn: 重みディレクトリ → run 直下 → pipeline.dn_weights の順（run の複製を優先）
            weight_dir = Path(weight_dir)
            cands = [weight_dir / DN_WEIGHT] + [d / DN_WEIGHT for d in (weight_dir.parent, weight_dir.parent.parent)]
            found = next((c for c in cands if c.is_file()), None)
            if found is not None:
                self.netDn, self.dn_source = load_denoiser(found, self.device), found
                self._dn_copied_to_run = True
            self._require_dn()
            print(f"[load] Dn: {self.dn_source}")

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
