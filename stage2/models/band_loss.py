"""band_loss.py — [SpicaV5] Stage 2 の MSE の代わり（2026-11-05、ブランチ 018-f_017-stage2_band_loss）。

MSE は「入力から当てられない成分」（Stage 1 で消した細部、PCD のノイズの粒、大きな内腔の中のむら）を平均して平らにする。
そこで帯域で役割を分ける:
  L = λ_lo · MSE(G_σ(ŷ), G_σ(t))            低〜中域は 1 枚ずつ合わせる（入力に情報がある帯域。σ = loss.band_lo_sigma）
    + λ_grad · L1(∇ŷ, ∇t)                    縁の位置と強さ（Sobel。L1 は中央値を取るので縁をなまらせにくい）
    + λ_nps · Σ_b (log P_ŷ(b) − log P_t(b))²  高域は画素ではなくバッチの帯域パワー（NPS）を教師に合わせる（質感。1 枚ずつは当てない）
    + λ_pix · L1(ŷ, t)                       全帯域の弱い錨（ずれ防止）
帯域パワーは Stage 1 の spectral_loss と同じ前処理（σ6 のトレンド除去、鏡の折り返しで窓なし）。Stage 間で import しない約束なので複製。
"""

import torch
import torch.nn.functional as F


def gaussian_blur(x, sigma):
    if sigma <= 0:
        return x
    r = max(1, int(round(3 * sigma)))
    t = torch.arange(-r, r + 1, dtype=torch.float32, device=x.device)
    k = torch.exp(-t ** 2 / (2 * sigma ** 2)); k = (k / k.sum()).to(x.dtype)
    x = F.pad(x, (r, r, r, r), mode="reflect")
    return F.conv2d(F.conv2d(x, k.view(1, 1, 1, -1)), k.view(1, 1, -1, 1))


def sobel(x):
    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=x.device, dtype=x.dtype) / 8.0
    xp = F.pad(x, (1, 1, 1, 1), mode="reflect")
    return F.conv2d(xp, kx.view(1, 1, 3, 3)), F.conv2d(xp, kx.t().contiguous().view(1, 1, 3, 3))


def band_power(z, edges, detrend_sigma=6.0, per_patch=False):
    """(N,1,H,W) → 帯域ごとのパワー（微分可能）。鏡の折り返しで窓なし。per_patch なら (N, B)、でなければバッチ平均 (B,)。"""
    a = (z - gaussian_blur(z, detrend_sigma)).float()
    a = torch.cat([a, a.flip(-1)], -1); a = torch.cat([a, a.flip(-2)], -2)
    h, w = a.shape[-2:]
    fy = torch.fft.fftfreq(h, device=z.device).abs()[:, None]; fx = torch.fft.fftfreq(w, device=z.device).abs()[None, :]
    fr = torch.sqrt(fy ** 2 + fx ** 2)
    s = torch.fft.fft2(a).abs() ** 2
    if per_patch:
        s = s[:, 0]
        return torch.stack([s[:, (fr >= lo) & (fr < hi)].sum(-1) for lo, hi in zip(edges[:-1], edges[1:])], -1)
    s = s.mean(0)[0]
    return torch.stack([s[(fr >= lo) & (fr < hi)].sum() for lo, hi in zip(edges[:-1], edges[1:])])


def band_loss(pred, target, lo_sigma, edges, lam_lo, lam_grad, lam_nps, lam_pix, per_patch=False):
    """合計と、項ごとの値（float）を返す。"""
    l_lo = torch.mean((gaussian_blur(pred, lo_sigma) - gaussian_blur(target, lo_sigma)) ** 2)
    px, py = sobel(pred); tx, ty = sobel(target)
    l_grad = (px - tx).abs().mean() + (py - ty).abs().mean()
    # per_patch: patch ごとに同じ場所の教師 patch と帯域パワーを合わせる（バッチ合計だと強い縁・空気の境界がパワーを占め、
    # 縁を強めるだけで合ってしまい、一様な内腔の粒が平らなまま残った。2026-11-05）。画素の位置は合わせない
    l_nps = ((torch.log(band_power(pred, edges, per_patch=per_patch) + 1e-12) - torch.log(band_power(target, edges, per_patch=per_patch) + 1e-12)) ** 2)
    l_nps = l_nps.sum(-1).mean() if per_patch else l_nps.sum()
    l_pix = (pred - target).abs().mean()
    total = lam_lo * l_lo + lam_grad * l_grad + lam_nps * l_nps + lam_pix * l_pix
    return total, {"lo": float(l_lo.detach()), "grad": float(l_grad.detach()), "nps": float(l_nps.detach()), "pix": float(l_pix.detach())}


def parse_floats(s, name):
    try:
        v = [float(t) for t in str(s).split(",") if t.strip() != ""]
    except ValueError:
        raise ValueError(f"{name} はカンマ区切りの数値: {s!r}")
    if len(v) < 2 or v != sorted(v):
        raise ValueError(f"{name} は 2 つ以上の昇順の数値: {s!r}")
    return v
