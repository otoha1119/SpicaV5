"""stage2/models/v_model.py — V（質感を作るフロー。設計書 C）の司令塔。式は PMRF（Ohayon et al., ICLR 2025。照合は stage2/UPSTREAM.md）。

  毎 step（設計書 C-4 の 3）:
    x↑・X の (size + 2·margin) 四方 → 固定した U で μ = lock(U(x↑), x↑)、正解 X′ = lock(X, x↑) を同じ広さで計算 → 中央 size を切る → V の中の値（HU ÷ unit_hu）へ
    σs ~ 対数一様 [sigma_s_min_hu, sigma_s_max_hu]、ε ~ N(0, I)、t ~ U[0, 1]（サンプルごと）
    Z0 = μ + σs·ε、Zt = (1 − t)·Z0 + t·X′、損失 = mean((V(Zt, t, σs) − (X′ − Z0))²)（重み付けなし）
  V に渡すもの: Zt・t・σs の 3 つだけ（models/flow_net.py の forward の引数）。x↑ は固定した U と明るさの固定（G）にだけ入る。
  精度: 固定した U の順伝播は optim.u_precision（bf16）+ optim.u_compile、勾配なし。V は本体だけ bf16 autocast + torch.compile、入口・出口・埋め込み・損失・明るさは FP32。
  U: 起動時に U の run の重み（net_U_ema.pth）を読んで固定する（eval・勾配なし・学習しない）。V の重みディレクトリに写し（net_U.pth）と U の実効設定を同梱する。
"""

import math

import torch

from models.flow_net import build_flow, flow_forward
from models.lock import center_crop, lock
from models.pipeline import Pipeline, Units
from models.trainer import Trainer, check_compiler, want_bf16
from models.unet import build_net
from data.ct_io import hu_tuple

N_TBINS = 4  # loss/flow_t0〜t3（t を 4 等分）


def draw_t(n, kind, device, generator=None):
    u = torch.rand(n, device=device, generator=generator)
    if kind == "uniform":
        return u
    if kind == "stratified_uniform":  # PMRF の公式実装の既定: batch を n 等分した区間から 1 つずつ（並びは混ぜる）
        return ((torch.arange(n, device=device) + u) / n)[torch.randperm(n, device=device, generator=generator)]
    raise ValueError(kind)


def draw_sigma_s(n, lo_hu, hi_hu, unit, device, generator=None):
    """σs（V の中の値の単位）。[lo, hi] HU を対数一様。lo = hi なら定数。"""
    u = torch.rand(n, device=device, generator=generator)
    return torch.exp(math.log(lo_hu) + u * (math.log(hi_hu) - math.log(lo_hu))) / unit


class VModel(Trainer):
    NET_NAME = "net_V"

    def __init__(self, data, v, shared, machine, u_config, u_weights, u_source, device):
        """u_config: U の実効設定 {"data", "u", "shared"}（U の state.pth から）。u_weights: U の EMA の state_dict。u_source: 記録用（run・tag・step）。"""
        self.data, self.sigma_px = data, float(shared["brightness.sigma_px_512"]) * data["data.upsample_scale"]
        self.units = Units(hu_tuple(data), v["flow.unit_hu"])
        self.margin = v["patch.margin"]
        self.u_residual = bool(u_config["u"]["net.residual"])
        self.u_net = build_net(u_config["u"]).to(device)
        self.u_net.load_state_dict(u_weights)
        self.u_net.eval().requires_grad_(False)
        self.u_bf16 = want_bf16(v["optim.u_precision"], device)
        self.u_compile_enabled = bool(v["optim.u_compile"]) and device.type == "cuda"
        if self.u_compile_enabled:
            check_compiler("--u_compile")
        self.u_fn = self.u_net
        self.net = build_flow(v).to(device)
        config = {"data": dict(data), "v": dict(v), "shared": dict(shared), "machine": dict(machine), "u_config": u_config, "u_source": u_source}
        self._setup(v, config, device, weight_decay=v["optim.weight_decay"], warmup_steps=v["optim.warmup_steps"], grad_clip=v["optim.grad_clip"])
        self.body_train = self.net.body
        self.reset_window()

    def _compile(self, mode):
        self.body_train = torch.compile(self.net.body, mode=mode)
        if self.u_compile_enabled:
            self.u_fn = torch.compile(self.u_net, mode=mode)

    def _extra_files(self):
        return {"net_U.pth": self.u_net.state_dict()}

    def reset_window(self):
        d = self.device
        self.win = {"n": 0, "loss": 0.0, "grad": 0.0, "bin_sum": torch.zeros(N_TBINS, device=d), "bin_cnt": torch.zeros(N_TBINS, device=d)}

    @torch.no_grad()
    def endpoints(self, x, t_img, compiled):
        """x↑・X（余白つき、n）→ 中央の (μ, X′)（V の中の値）。固定した U と明るさの固定はここだけ。"""
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.u_bf16):
            f = (self.u_fn if compiled else self.u_net)(x)
        f = f.float()
        if compiled and self.u_compile_enabled:
            f = f.clone()
        if self.u_residual:
            f = x + f
        mu = center_crop(lock(f, x, self.sigma_px), self.margin)
        target = center_crop(lock(t_img, x, self.sigma_px), self.margin)
        return self.units.z(mu), self.units.z(target)

    def flow_loss(self, net, mu, target, t, sigma_s, eps, body=None, bf16=False, clone=False):
        """サンプルごとの損失 (B,) と、表示用の 1 段での見積もり Z0 + V（t = 0 の行なら X′ の見積もり）。FP32。"""
        s = sigma_s.view(-1, 1, 1, 1)
        tt = t.view(-1, 1, 1, 1)
        z0 = mu + s * eps
        zt = (1.0 - tt) * z0 + tt * target
        v = flow_forward(net, zt, t, sigma_s, body=body, bf16=bf16, channels_last=self.channels_last, clone=clone)
        return (v - (target - z0)).square().mean(dim=(1, 2, 3)), zt + (1.0 - tt) * v

    def train_step(self, batch):
        x = batch["x"].to(self.device, non_blocking=True).float()
        t_img = batch["t"].to(self.device, non_blocking=True).float()
        b = x.shape[0]
        with torch.no_grad():
            mu, target = self.endpoints(x, t_img, self.compiled)
            t = draw_t(b, self.cfg["flow.t_sampling"], self.device)
            sigma_s = draw_sigma_s(b, self.cfg["flow.sigma_s_min_hu"], self.cfg["flow.sigma_s_max_hu"], self.units.unit, self.device)
            eps = torch.randn_like(mu)
        self.net.train()
        per, x1_hat = self.flow_loss(self.net, mu, target, t, sigma_s, eps, body=self.body_train, bf16=self.use_bf16, clone=self.compiled)
        loss_v, grad = self._update(per.mean())
        with torch.no_grad():
            per = per.detach()
            w = self.win
            w["n"] += 1; w["loss"] += loss_v; w["grad"] += grad
            bins = (t * N_TBINS).long().clamp_(0, N_TBINS - 1)
            w["bin_sum"].index_add_(0, bins, per); w["bin_cnt"].index_add_(0, bins, torch.ones_like(per))
            n = self.cfg["log.n_images"]
            nz = self.units.n_from_z
            self.last = {"mu": nz(mu[:n]), "x1_hat": nz(x1_hat[:n].detach()), "target": nz(target[:n]), "t": t[:n], "sigma_s_hu": sigma_s[:n] * self.units.unit}
        return loss_v

    def window_stats(self):
        w = self.win
        n = max(w["n"], 1)
        out = {"loss/flow": w["loss"] / n, "diag/grad_norm": w["grad"] / n}
        s, c = w["bin_sum"].tolist(), w["bin_cnt"].tolist()
        for i in range(N_TBINS):
            if c[i] > 0:
                out[f"loss/flow_t{i}"] = s[i] / c[i]
        return out

    def build_val(self, patches, seed):
        """val/flow の固定の入力（CPU）: パッチ（余白つき）、t（等分）、σs（対数で等分）、ε。毎回同じ。"""
        g = torch.Generator().manual_seed(seed)
        n, p = patches["x"].shape[0], self.cfg["patch.size"]
        cpu = torch.device("cpu")
        return {"x": patches["x"], "t_img": patches["t"], "t": draw_t(n, "stratified_uniform", cpu, g),
                "sigma_s": draw_sigma_s(n, self.cfg["flow.sigma_s_min_hu"], self.cfg["flow.sigma_s_max_hu"], self.units.unit, cpu, g),
                "eps": torch.randn(n, 1, p, p, generator=g)}

    @torch.no_grad()
    def evaluate(self, val):
        """val/flow: 固定パッチの損失（EMA、FP32。U も FP32・compile なし）。"""
        bs, per = self.cfg["optim.batch_size"], []
        u_bf16, self.u_bf16 = self.u_bf16, False
        try:
            for i in range(0, val["x"].shape[0], bs):
                g = {k: v[i:i + bs].to(self.device) for k, v in val.items()}
                mu, target = self.endpoints(g["x"].float(), g["t_img"].float(), compiled=False)
                per.append(self.flow_loss(self.ema, mu, target, g["t"], g["sigma_s"], g["eps"])[0])
        finally:
            self.u_bf16 = u_bf16
        return {"val/flow": float(torch.cat(per).mean())}

    def pipeline(self):
        """画像・確認用: U と V の EMA、FP32、compile なし。"""
        return Pipeline(self.data, self.sigma_px, self.u_net, self.u_residual, self.device, self.ema, self.cfg["flow.unit_hu"], False, self.channels_last)
