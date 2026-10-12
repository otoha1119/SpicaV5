"""fidelity_gan_model.py — Park et al. 2019 の fidelity-embedded GAN を junyanz 学習基盤の上に実装する。

対象論文: H. S. Park, J. Baek, S. K. You, J. K. Choi, J. K. Seo,
          "Unpaired Image Denoising Using a Generative Adversarial Network in X-Ray CT,"
          IEEE Access, 2019. DOI 10.1109/ACCESS.2019.2934178 (arXiv:1903.06257v2)

【このファイルの由来】 models/cycle_gan_model.py をコピーし、論文 Fig. 3 に無い部位を削除したもの (フェーズ 1)。
  削除: netG_B, netD_B, cycle loss (lambda_A / lambda_B), identity loss (lambda_identity), ImagePool
  残存: netG 1 個 (A→B), netD 1 個 (B 側), optimizer / scheduler の配線

【論文の記号との対応】 (論文 Sec. II, Fig. 3)
  real_A  = z        : 変換元 (論文 LDCT。本研究では PCD)
  real_B  = x        : 分布の参照先 (論文 SDCT。本研究では EID)
  fake_B  = G(z)     : 生成画像
  netD    = D        : real_B (x) と fake_B (G(z)) を判別する PatchGAN

【損失 (フェーズ 4)】 論文 Eq.(6), (12) と Theorem II.1
  J(D,G) := E_{x~p_x}[D(x)] + E_{z~p_z}[log(1 − D(G(z)))] + λ E_{z~p_z}[‖G(z) − z‖²]      ... Eq.(6)
  G_* = argmin_G max_D J(D,G)                                                            ... Eq.(5)
  経験版 Eq.(12):
    G  = argmin_G (1/N) Σ_z [ log(1 − D(G(z))) + λ‖G(z) − z‖² ]
    D  = arg max_D (1/M) Σ_x D(x) + (1/N) Σ_z log(1 − D(G(z)))
       ※ 論文の Eq.(12) は D について "argmin" と書かれているが、Eq.(7) と Theorem II.1 の証明 (Eq.7–11) は max。
         証明に従い max を採用 (docs/reference/park2019_implementation_checklist.md Q-L1)
  最適 D は Eq.(9): D_opt(w) = 1 − p_G(w)/p_x(w) ∈ (−∞, 1]。log(1 − D) が定義されるには D < 1 が必要だが、
  論文は D の出力制約を書いていない。→ 採用 (Q-L2, ユーザー承認 2026-09-05):
    D の 1×1 conv の生出力を v とし  D := 1 − exp(v)   (常に D < 1)
    このとき  log(1 − D) = v,  E[D(x)] = E[1 − exp(v(x))]
    固定 w での最大化条件は  p_x(w)·(−exp(v)) + p_G(w) = 0  ⇔  exp(v) = p_G/p_x  ⇔  D = 1 − p_G/p_x  で Eq.(9) と一致する
  fidelity ‖G(z) − z‖² は画素平均で計算する (Q-L3)。λ = 10 (論文 "We empirically choose λ = 10")

【フェーズ進行】 (docs/plans/20260905_stage1-fidelity-gan-implementation-spec.md)
  1 骨格 → 2 netG='wavelet' (models/wavelet_generator.py) → 3 netD='paper' (models/paper_discriminator.py)
  → 4 損失 (このファイル) → 6 学習条件の既定値 (modify_commandline_options)
"""

import os
import random
import shutil

import numpy as np
import yaml
import torch
import torch.distributed as dist
from .base_model import BaseModel
from . import networks
from configs.schema import check_opt


def _to_tuple(x):
    """random.setstate は入れ子の tuple を要求する。torch.load 後に list になっている場合に戻す。"""
    return tuple(_to_tuple(v) for v in x) if isinstance(x, (list, tuple)) else x


class FidelityGANModel(BaseModel):
    """一方向 GAN (A→B) + fidelity。cycle_gan_model.CycleGANModel から片側だけを残し、損失を Park 2019 に置き換えたもの。"""

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """モデル固有 option と、既定値の廃止（番兵化）。

        原則（docs/plans/20260905_config-design-plan.md）: FE-GAN が使う引数に既定値は無い。
        すべて run_train.py が configs/train.yaml / mode.yaml / machines.yaml から明示して渡す。
        junyanz 本家の argparse 既定は set_defaults(None) で潰し、__init__ の check_opt で None が残っていればエラー。
        （checkpoints_dir は BaseOptions.parse が保存先として先に使うため本家既定のまま。起動器が必ず渡す）

        論文に記載のある値は configs/train.yaml のコメントに出典を書いてある。
        """
        parser.set_defaults(
            netG=None, netD=None, ngf=None, ndf=None, norm=None,
            input_nc=None, output_nc=None, init_type=None, init_gain=None,
            batch_size=None, num_threads=None,
            no_dropout=True,  # 論文に dropout の記載なし（構造上の固定。チューニング対象ではない）
        )
        parser.add_argument("--final_norm_act", action="store_true", help="G 最終 conv に bnorm+LReLU を付ける（確定: 付けない。ablation 用）")
        parser.add_argument("--residual", action="store_true", help="G を residual 化: fake_B = real_A + F(real_A)（F = 生の generator、その出力が R。networks.ResidualGenerator）。損失は不変（段階 5-1、2026-09-11）")
        if is_train:
            parser.set_defaults(
                lr=None, beta1=None, n_epochs=None, n_epochs_decay=None, lr_policy=None,
                gan_mode=None, print_freq=None, save_epoch_freq=None, save_latest_freq=None, epoch_count=None,
                pool_size=0,  # 論文に image buffer の記載なし（構造上の固定）
            )
            parser.add_argument("--beta2", type=float, default=None, help="Adam β2（論文未記載）")
            # 学習表示（util/monitor.py）。値は configs/train.yaml の log: から
            parser.add_argument("--image_freq", type=int, default=None, help="TensorBoard に途中画像を出す間隔（step）")
            parser.add_argument("--n_images", type=int, default=None, help="途中画像の枚数")
            parser.add_argument("--full_slice", type=str, default=None, help="checkpoint 保存時にフル 512 で書き出す固定スライス（dir_A からの相対パス）")
            parser.add_argument("--eid_slice", type=str, default=None, help="パネルの右端に並べる EID の代表スライス（dir_B からの相対パス。固定）")
            parser.add_argument("--n_full_random", type=int, default=None, help="checkpoint 保存時にフル 512 で書き出すランダムスライスの枚数（epoch ごとに別）")
            parser.add_argument("--display_hu_min", type=int, default=None, help="表示用の線形範囲の下限 HU")
            parser.add_argument("--display_hu_max", type=int, default=None, help="表示用の線形範囲の上限 HU")
            parser.add_argument("--diff_range_hu", type=int, default=None, help="差分パネル（R）のカラー表示の ±範囲 HU（白 = 0、純青 = −、純赤 = +）")
            parser.add_argument("--preview_bits", type=int, default=None, help="output_images/preview_<slice>/ のビット深度（8 | 16）")
            parser.add_argument("--lambda_fid", type=float, default=None, help="weight λ of the fidelity term ‖G(z) − z‖² (paper: λ = 10)")
            parser.add_argument("--seed", type=int, default=None, help="乱数 seed（F-15。train.py が起動直後に python / numpy / torch に適用）")
            parser.add_argument("--require_cuda", action="store_true", help="CUDA が使えなければ止める（F-16。run_train が machines.yaml の gpu_gen ≠ 0 のとき付ける）")
            # 再開用（run_train.py --resume が付ける。設定値ではなく起動器の内部フラグなので schema には無い）
            parser.add_argument("--resume_state", type=str, default=None, help="重みディレクトリの state.pth のパス。--continue_train と併用。optimizer / scheduler / RNG / iteration 数を復元する")
            parser.add_argument("--launch_path", type=str, default=None, help="この起動の実効設定（launch.yaml）。state.pth に train / mode / machine を写し、再開の基準にする（run_train.py が付ける）")
        return parser

    def __init__(self, opt):
        check_opt(opt, "FidelityGANModel")  # 番兵 None が残っていればここで止める
        BaseModel.__init__(self, opt)
        # 表示・保存する損失名。<BaseModel.get_current_losses> が loss_<name> を読む
        #   D      : D の最小化損失 (= −J_D)
        #   D_real : E[D(x)]   (診断値。最適で 1 − p_G/p_x)
        #   D_fake : E[D(G(z))] (診断値)
        #   G_GAN  : E[log(1 − D(G(z)))]
        #   G_fid  : E[‖G(z) − z‖²] (λ を掛ける前)
        self.loss_names = ["D", "D_real", "D_fake", "G_GAN", "G_fid"]
        # 表示・保存する画像名
        self.visual_names = ["real_A", "fake_B", "real_B"]
        # 保存・読込するネットワーク名。テスト時は G のみ
        self.model_names = ["G", "D"] if self.isTrain else ["G"]

        # G: A→B (論文 G: z → G(z))
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, final_norm_act=opt.final_norm_act, residual=opt.residual)

        if self.isTrain:
            # D: B 側の判別器 (論文 D: x vs G(z))
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain)
            self.gan_mode = opt.gan_mode
            if self.gan_mode != "fgan_kl":
                self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # ablation 用 (論文の式ではない)
            self.lambda_fid = opt.lambda_fid
            # optimizer。scheduler は <BaseModel.setup> が opt.lr_policy から自動生成
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers = [self.optimizer_G, self.optimizer_D]

    # ------------------------------------------------------------------
    # 論文 Eq.(6) の D の値域: D := 1 − exp(v)。v は netD の生出力
    # ------------------------------------------------------------------
    @staticmethod
    def _D_from_v(v):
        return 1.0 - torch.exp(v)

    @staticmethod
    def _log1mD_from_v(v):
        # log(1 − D) = log(exp(v)) = v
        return v

    def set_input(self, input):
        """dataloader の出力を展開する。--direction で A/B を入れ替えられる (本家と同じ)。"""
        AtoB = self.opt.direction == "AtoB"
        self.real_A = input["A" if AtoB else "B"].to(self.device)
        self.real_B = input["B" if AtoB else "A"].to(self.device)
        self.image_paths = input["A_paths" if AtoB else "B_paths"]

    def forward(self):
        """fake_B = G(real_A)。論文 Fig. 3 の generator G の順伝播。--residual のときは G = z + F(z)（networks.ResidualGenerator）なので fake_B − real_A が F の生出力 R。"""
        self.fake_B = self.netG(self.real_A)

    def backward_D(self):
        """D の損失と勾配。論文 Eq.(12) の D:  max_D  E_x[D(x)] + E_z[log(1 − D(G(z)))]
        → 最小化するのは  loss_D = −( E_x[D(x)] + E_z[log(1 − D(G(z)))] )。係数 0.5 は付けない (Eq.(12) に無い)。
        """
        v_real = self.netD(self.real_B)
        v_fake = self.netD(self.fake_B.detach())
        if self.gan_mode == "fgan_kl":
            D_real = self._D_from_v(v_real)              # D(x) = 1 − exp(v)
            log1mD_fake = self._log1mD_from_v(v_fake)    # log(1 − D(G(z))) = v
            J_D = D_real.mean() + log1mD_fake.mean()
            self.loss_D = -J_D
            self.loss_D_real = D_real.mean().detach()
            self.loss_D_fake = self._D_from_v(v_fake.clamp(max=80.0)).mean().detach()  # 診断値。exp の overflow（−inf 表示）を避ける（F-23）
        else:  # ablation: 本家 GANLoss (lsgan / vanilla)
            self.loss_D_real = self.criterionGAN(v_real, True)
            self.loss_D_fake = self.criterionGAN(v_fake, False)
            self.loss_D = self.loss_D_real + self.loss_D_fake
        self.loss_D.backward()

    def backward_G(self):
        """G の損失と勾配。論文 Eq.(12) の G:  min_G  E_z[log(1 − D(G(z)))] + λ E_z[‖G(z) − z‖²]"""
        v_fake = self.netD(self.fake_B)
        if self.gan_mode == "fgan_kl":
            self.loss_G_GAN = self._log1mD_from_v(v_fake).mean()   # E[log(1 − D(G(z)))] = E[v]
        else:  # ablation: 本家 GANLoss
            self.loss_G_GAN = self.criterionGAN(v_fake, True)
        self.loss_G_fid = torch.mean((self.fake_B - self.real_A) ** 2)  # ‖G(z) − z‖² の画素平均
        self.loss_G = self.loss_G_GAN + self.lambda_fid * self.loss_G_fid
        self.loss_G.backward()

    # ------------------------------------------------------------------
    # 学習の再開: 本家は重み (*_net_G/D.pth) しか保存しないので、optimizer / RNG / 進捗を同じ重みディレクトリの state.pth に足す
    # ------------------------------------------------------------------
    def save_networks(self, epoch):
        """重み（net_G / net_D）と state.pth を**同じ重みディレクトリに原子的に**保存する（F-14）。tag は 'latest' / 数値 / 'iter_N'。置き場所は util/run_paths.py の規則。
        手順: <dir>.tmp/ に 3 ファイルを書く → 既存 <dir> を <dir>.old に rename → .tmp を <dir> に rename → .old を消す。
        途中で止まっても <dir> に新旧の混在は起きない（.tmp / .old が残るだけ。resume は 3 ファイル揃った <dir> だけを見る）。本家の save_networks は使わない。"""
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        final = self.ckpt_path(epoch, "state.pth").parent
        tmp, old = final.with_name(final.name + ".tmp"), final.with_name(final.name + ".old")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        for name in self.model_names:
            net = getattr(self, "net" + name)
            net = net.module if hasattr(net, "module") else net  # DDP を外す
            net = net._orig_mod if hasattr(net, "_orig_mod") else net  # torch.compile を外す
            torch.save(net.state_dict(), tmp / f"net_{name}.pth")
        if self.isTrain:
            torch.save(self._resume_state(), tmp / "state.pth")
        if old.exists():
            shutil.rmtree(old)
        if final.exists():
            os.replace(final, old)
        os.replace(tmp, final)
        if old.exists():
            shutil.rmtree(old)

    def _launch_config(self):
        """--launch_path（run_train.py が書いた実効設定）の train / mode / machine を返す。無ければ None（手動起動・scratch）。"""
        path = getattr(self.opt, "launch_path", None)
        if not path:
            return None
        with open(path, encoding="utf-8") as f:
            launch = yaml.safe_load(f)
        return {k: dict(launch[k]) for k in ("train", "mode", "machine")}

    def _resume_state(self):
        """state.pth の中身（optimizer / RNG / 進捗）。"""
        np_state = np.random.get_state()  # ('MT19937', ndarray(624, uint32), pos, has_gauss, cached_gaussian)
        state = {
            "epoch": int(getattr(self, "cur_epoch", self.opt.epoch_count)),
            "epoch_done": bool(getattr(self, "epoch_done", False)),
            "total_iters": int(getattr(self, "total_iters", 0)),
            "optimizer_G": self.optimizer_G.state_dict(),
            "optimizer_D": self.optimizer_D.state_dict(),
            # scheduler の state は保存しない: junyanz の LambdaLR は --epoch_count を起点に 0 から数える設計なので、
            # 復元すると二重計上になる（lr が 0 や負になる）。再開時は optimizer state を読んだ後に scheduler を作り直す
            "lr": float(self.optimizers[0].param_groups[0]["lr"]),
            "config": self._launch_config(),  # 保存時の実効設定（train / mode / machine）。resume はこれを基準にする（失敗した起動の launch を引き継がない。2026-09-09）
            "rng": {
                "python": random.getstate(),
                "torch": torch.get_rng_state(),
                "numpy": (np_state[0], torch.from_numpy(np_state[1].astype(np.int64)), int(np_state[2]), int(np_state[3]), float(np_state[4])),
            },
        }
        return state

    def setup(self, opt):
        """本家の setup（重み読込・scheduler 生成）のあと、--resume_state があれば optimizer / RNG / 進捗を復元し、scheduler を作り直す。
        lr / betas は opt の値（yaml + sh の上書き）で上書きする（F-10）。"""
        super().setup(opt)
        if not (self.isTrain and opt.continue_train and opt.resume_state):
            return
        if opt.lr_policy != "linear":  # F-12: scheduler の作り直しは linear（epoch_count 起点）でしか成立しない。schema でも制限済み
            raise NotImplementedError(f"lr_policy={opt.lr_policy} の再開は未対応（scheduler 状態の保存・復元が無い）。linear を使ってください")
        # F-01: state は CPU に読む。map_location=device だと RNG 用の Tensor（numpy の keys）まで CUDA に載り、.numpy() で落ちる。
        #       optimizer の state は load_state_dict が param のデバイスへ自動でキャストする
        state = torch.load(opt.resume_state, map_location="cpu", weights_only=True)
        self.optimizer_G.load_state_dict(state["optimizer_G"])  # Adam のモーメント・step を復元
        self.optimizer_D.load_state_dict(state["optimizer_D"])
        # F-10(b): load_state_dict は保存時の lr / betas / initial_lr を param_groups に戻すので、opt（= yaml + sh の上書き）の値を再適用する。
        #          これで resume 時の --lr / --beta1 / --beta2 が最優先になる。同じ値なら何も変わらない
        for optimizer in self.optimizers:
            for g in optimizer.param_groups:
                g["lr"] = opt.lr
                g["initial_lr"] = opt.lr
                g["betas"] = (opt.beta1, opt.beta2)
        # scheduler は作り直す: LambdaLR が param_groups の initial_lr を base に、新しい --epoch_count を起点として lr を再設定する（linear のみ対応。F-12）
        self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        rng = state["rng"]
        random.setstate(_to_tuple(rng["python"]))
        torch.set_rng_state(rng["torch"].cpu())
        kind, keys, pos, has_gauss, cached = rng["numpy"]
        np.random.set_state((kind, keys.cpu().numpy().astype(np.uint32), pos, has_gauss, cached))
        self.resume_total_iters = state["total_iters"]
        print(f"[resume] {opt.resume_state}: epoch {state['epoch']} (done={state['epoch_done']}), total_iters {state['total_iters']}, lr {self.optimizers[0].param_groups[0]['lr']:.7f}")

    def optimize_parameters(self):
        """1 iteration: forward → G 更新 → D 更新 (本家 cycle_gan_model と同じ順序。論文は更新順を明記していない)。"""
        self.forward()
        # G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
