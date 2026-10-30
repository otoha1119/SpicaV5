"""stage1_v3/degrade.py — 物理コア P（学習なし・確率的）。PCD512 の HU 画像に、ぼかし（MTF 低下）と NPS 整形ノイズを掛ける。

  mode gaussian      : u = blur(x; σ_b) + σ_n · shape(w; c) · mask
  mode bilateral_blur: u = blur(bilateral(x; σ_color, σ_space); σ_b) + σ_n · shape(w; c) · mask
  mode learned       : データ側では何もしない（u = x を渡し、models/degrade_net.py の P_θ が GPU 上で u を作る。2026-10-30、固定 P からの転換）
                       bilateral（エッジ保存の平滑化）で PCD のノイズを EID 水準まで落とし（iterative recon の役回り）、弱い Gaussian でカーネル差（MTF）を合わせ、最後に NPS 整形ノイズを足す。
                       単一の Gaussian ではノイズ 93 → 22 HU に σ ≈ 1.9 px が要り、中程度のエッジ（肺血管・縦隔境界）が実 EID より潰れる（2026-10-30、1 本目の目視）
    σ_b : Gaussian ぼかしの標準偏差 [px]（鮮鋭度の軸。R-S22 (b)）
    σ_n : 加算ノイズの標準偏差 [HU]（ノイズ量の軸。R-S22 (a)）
    c   : 白色ノイズ w に掛ける Gaussian の標準偏差 [px]（NPS の形。0 = 白色。大きいほど低周波寄り）。整形後は標本標準偏差で 1 に戻してから σ_n を掛ける
    mask: noise_body_only なら blur(x) > body_threshold_hu の画素だけ 1（空気にはノイズを足さない）
  σ_b / σ_n / c はスライスごとに [min, max] の一様分布から引く（R-S23。線量・再構成カーネル差の模擬）。
  範囲は実測（measure_dir.py）を含むまで広げ、少し狭めて同等なら十分（SynthSeg 手順。surveys/20261030 推薦 1）。

学習（data/dataset.py がスライス単位で呼ぶ）と推論（inference_dir.py / degrade_dir.py）で同じ関数を使う。入力・出力とも float32 の HU（正規化前）。
HU 写像・blooming・ビームハードニングのような内容依存の成分は P に無く、学習残差 R の担当（R-S25）。
"""

import cv2
import numpy as np

PARAM_NAMES = ("blur_sigma_px", "noise_sigma_hu", "noise_corr_px", "bilateral_sigma_color_hu", "bilateral_sigma_space_px")


def degrade_config(train):
    """train の平坦 dict から P の設定だけを取り出す（launch.yaml / infer の引数の両方から同じ形で作る）。"""
    return {
        "enabled": bool(train["degrade.enabled"]),
        "mode": str(train["degrade.mode"]),
        "bil_color": (float(train["degrade.bilateral_sigma_color_hu_min"]), float(train["degrade.bilateral_sigma_color_hu_max"])),
        "bil_space": (float(train["degrade.bilateral_sigma_space_px_min"]), float(train["degrade.bilateral_sigma_space_px_max"])),
        "blur": (float(train["degrade.blur_sigma_px_min"]), float(train["degrade.blur_sigma_px_max"])),
        "noise": (float(train["degrade.noise_sigma_hu_min"]), float(train["degrade.noise_sigma_hu_max"])),
        "corr": (float(train["degrade.noise_corr_px_min"]), float(train["degrade.noise_corr_px_max"])),
        "body_only": bool(train["degrade.noise_body_only"]),
        "body_threshold_hu": float(train["degrade.body_threshold_hu"]),
    }


def sample_params(cfg, rng):
    """スライス 1 枚分の (σ_b, σ_n, c) を一様に引く。rng は numpy.random.Generator。enabled=false なら全部 0。"""
    if not cfg["enabled"] or cfg["mode"] == "learned":
        return {k: 0.0 for k in PARAM_NAMES}
    bil = cfg["mode"] == "bilateral_blur"
    return {
        "blur_sigma_px": float(rng.uniform(*cfg["blur"])),
        "noise_sigma_hu": float(rng.uniform(*cfg["noise"])),
        "noise_corr_px": float(rng.uniform(*cfg["corr"])),
        "bilateral_sigma_color_hu": float(rng.uniform(*cfg["bil_color"])) if bil else 0.0,
        "bilateral_sigma_space_px": float(rng.uniform(*cfg["bil_space"])) if bil else 0.0,
    }


def _gaussian_blur(img, sigma):
    if sigma <= 0:
        return img
    k = int(2 * np.ceil(3.0 * sigma) + 1)
    return cv2.GaussianBlur(img, (k, k), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)


def _bilateral(img, sigma_color, sigma_space):
    """エッジ保存の平滑化（cv2.bilateralFilter、float32）。sigma_color [HU]: この差より小さい画素は平均される（ノイズ ≈ 93 HU なら 2〜3 倍が目安）、sigma_space [px]: 空間の広がり。"""
    if sigma_color <= 0 or sigma_space <= 0:
        return img
    d = int(2 * np.ceil(2.0 * sigma_space) + 1)
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space, borderType=cv2.BORDER_REFLECT_101)


def shaped_noise(shape, corr_px, rng):
    """単位分散の空間相関ノイズ。白色 Gaussian に σ = corr_px の Gaussian を掛け、標本標準偏差で 1 に戻す。corr_px = 0 なら白色。"""
    w = rng.standard_normal(shape).astype(np.float32)
    if corr_px <= 0:
        return w
    n = _gaussian_blur(w, corr_px)
    sd = float(n.std())
    if sd <= 0:
        raise RuntimeError("整形ノイズの標準偏差が 0（corr_px が画像に対して大きすぎる）")
    return n / sd


def apply_degrade(hu, params, cfg, rng):
    """hu: float32 (H, W) の HU 画像 → P(x) の HU 画像（float32、同じ形）。params は sample_params の戻り値。"""
    if not cfg["enabled"] or cfg["mode"] == "learned":
        return hu
    x = np.ascontiguousarray(hu, dtype=np.float32)
    if cfg["mode"] == "bilateral_blur":
        x = _bilateral(x, params["bilateral_sigma_color_hu"], params["bilateral_sigma_space_px"])
    elif cfg["mode"] != "gaussian":
        raise ValueError(f"degrade.mode: {cfg['mode']}")
    u = _gaussian_blur(x, params["blur_sigma_px"])
    if params["noise_sigma_hu"] > 0:
        n = shaped_noise(x.shape, params["noise_corr_px"], rng) * np.float32(params["noise_sigma_hu"])
        if cfg["body_only"]:
            n = n * (u > cfg["body_threshold_hu"]).astype(np.float32)
        u = u + n
    return u


def params_to_array(params):
    return np.array([params[k] for k in PARAM_NAMES], dtype=np.float32)


def params_from_array(arr):
    return {k: float(v) for k, v in zip(PARAM_NAMES, arr)}
