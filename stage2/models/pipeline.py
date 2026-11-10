"""stage2/models/pipeline.py — 入力 x↑ から出力までの決まった手順（設計書 C-1・D-1・E-3）。学習中の表示・確認・推論がすべてこれを通る（同じ関数 = 同じ計算）。

  μ  = lock(f, x↑)、f = U(x↑)                              … U の出力（形の絵。明るさの固定つき）
  Z0 = μ + σs·ε                                             … 出発点（ε は画素ごとに独立な正規乱数。V の中の値 = HU ÷ flow.unit_hu）
  Z  ← Z + (1/K)·V(Z, k/K, σs)、k = 0 … K−1                 … Euler（PMRF Algorithm 1。stage2/UPSTREAM.md）
  y  = lock(Z1, x↑)                                         … 最後に明るさを戻す

精度: U・明るさの固定・V の入口と出口は常に FP32。V の本体だけ v_bf16 で bf16 にできる（既定は FP32）。compile はしない。
乱数: スライスごとの種（util/seeds.py）で **CPU・FP32 で引いてから** device へ送る（機械が変わっても同じ乱数。batch の組み方に依らない）。
tensor の値は n（data/ct_io.py。旧 Stage 2 と同じ線形の [−1, 1]、切らない）。
"""

from pathlib import Path

import torch

from data.ct_io import hu_per_unit, hu_tuple
from models.flow_net import build_flow, flow_forward
from models.lock import lock
from models.unet import build_net
from util.seeds import seed_of


def cpu_noise(shape, *seed_parts):
    """(H, W) の正規乱数を CPU・FP32 で引く。種 = sha256('<seed>/<症例>/<スライス名>')。"""
    g = torch.Generator(device="cpu")
    g.manual_seed(seed_of(*seed_parts))
    return torch.randn(shape, generator=g, dtype=torch.float32)


class Units:
    """n ↔ HU ↔ z（V の中の値 = HU ÷ unit_hu）の 1 次式。"""

    def __init__(self, hu, unit_hu=None):
        self.hu_min, self.hpu, self.unit = float(hu[1]), hu_per_unit(hu), (float(unit_hu) if unit_hu else None)

    def hu(self, n):
        return (n + 1.0) * self.hpu + self.hu_min

    def z(self, n):
        return self.hu(n) / self.unit

    def n_from_z(self, z):
        return z * self.unit / self.hpu - self.hu_min / self.hpu - 1.0


class Pipeline:
    def __init__(self, data, sigma_px, u_net, u_residual, device, v_net=None, unit_hu=None, v_bf16=False, v_channels_last=False):
        self.device, self.sigma_px = device, float(sigma_px)
        self.units = Units(hu_tuple(data), unit_hu)
        self.u_net, self.u_residual = u_net, bool(u_residual)
        self.v_net, self.v_bf16, self.v_channels_last = v_net, bool(v_bf16), bool(v_channels_last)

    @torch.no_grad()
    def mu(self, x):
        """x (B, 1, H, W) n → (μ, f)。FP32。"""
        x = x.to(self.device).float()
        f = self.u_net(x).float()
        if self.u_residual:
            f = x + f
        return lock(f, x, self.sigma_px), f

    @torch.no_grad()
    def flow(self, mu, x, eps, sigma_s_hu, steps):
        """μ・x↑ (B, 1, H, W) n、eps (B, 1, H, W) → (y, Z1)（どちらも n。y = lock(Z1, x↑)）。"""
        if self.v_net is None:
            raise RuntimeError("V がありません（U だけの重み）")
        b = mu.shape[0]
        s = torch.full((b,), float(sigma_s_hu) / self.units.unit, device=self.device)
        z = self.units.z(mu.float()) + s.view(-1, 1, 1, 1) * eps.to(self.device).float()
        for k in range(steps):
            t = torch.full((b,), k / steps, device=self.device)
            z = z + (1.0 / steps) * flow_forward(self.v_net, z, t, s, bf16=self.v_bf16, channels_last=self.v_channels_last)
        z1 = self.units.n_from_z(z)
        return lock(z1, x.to(self.device).float(), self.sigma_px), z1


def _load(path, device):
    return torch.load(path, map_location=device, weights_only=True)


def load_pipeline(weight_dir, kind, device, v_bf16=False):
    """重みディレクトリ（U の run か V の run）から Pipeline を組む。戻り値 (pipeline, state の config, step)。
    V の重みディレクトリには U の写し（net_U.pth）と U の実効設定（config['u_config']）が入っているので、それだけで完結する。"""
    weight_dir = Path(weight_dir)
    state = _load(weight_dir / "state.pth", "cpu")
    cfg = state["config"]
    if kind == "u":
        u_cfg, data, shared = cfg["u"], cfg["data"], cfg["shared"]
        u_net = build_net(u_cfg).to(device)
        u_net.load_state_dict(_load(weight_dir / "net_U_ema.pth", device))
        v_net = unit = None
        cl = False
    else:
        u_cfg, data, shared = cfg["u_config"]["u"], cfg["data"], cfg["shared"]
        u_net = build_net(u_cfg).to(device)
        u_net.load_state_dict(_load(weight_dir / "net_U.pth", device))
        v_net = build_flow(cfg["v"]).to(device)
        v_net.load_state_dict(_load(weight_dir / "net_V_ema.pth", device))
        cl = bool(cfg["v"]["optim.channels_last"])
        if cl:
            v_net = v_net.to(memory_format=torch.channels_last)
        v_net.eval().requires_grad_(False)
        unit = cfg["v"]["flow.unit_hu"]
    u_net.eval().requires_grad_(False)
    sigma = float(shared["brightness.sigma_px_512"]) * data["data.upsample_scale"]
    return Pipeline(data, sigma, u_net, u_cfg["net.residual"], device, v_net, unit, v_bf16 and device.type == "cuda", cl), cfg, int(state["step"])
