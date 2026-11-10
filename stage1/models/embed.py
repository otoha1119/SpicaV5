"""stage1/models/embed.py — 度合いの埋め込み（設計書 C-2）。計算は FP32（autocast の外で呼ぶ）。

  教えるもの        スカラーにする式
  拡散のノイズ σ    s_σ = ln(σ) / 4          （EDM の c_noise と同じ。出典は UPSTREAM.md）
  粗さ σ_b          s_b = σ_b / blur_max_px
  手がかりのノイズ  s_c = ln(σ_c + 0.002) / 4（σ_c は x の単位）
  手がかりの有無    学習する 2 種類のベクトル

各スカラーを、対数で等間隔の emb_freqs 個の周波数（emb_freq_min〜emb_freq_max）の sin / cos にし、それぞれ別の MLP（2F → D → D、SiLU）に通して足す。
手がかり無しのときは s_b・s_c の埋め込みを足さない（「なし」のベクトルだけ）。
角度は 周波数 × スカラー（ラジアン。2π は掛けない）: 設計書は周波数の単位を決めていないので、スカラーの範囲（幅 約 2.5）で一番低い周波数が単調になるほうを採った。
"""

import math

import torch
import torch.nn as nn

NOISE_EPS = 0.002  # s_c = ln(σ_c + 0.002) / 4（σ_c = 0 を有限にする。設計書 C-2）


class ScalarEmbed(nn.Module):
    def __init__(self, n_freqs, f_min, f_max, dim):
        super().__init__()
        self.register_buffer("freqs", torch.exp(torch.linspace(math.log(f_min), math.log(f_max), n_freqs)), persistent=False)
        self.mlp = nn.Sequential(nn.Linear(2 * n_freqs, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, s):
        ang = s.float()[:, None] * self.freqs[None, :]
        return self.mlp(torch.cat([ang.sin(), ang.cos()], dim=1))


class CondEmbed(nn.Module):
    def __init__(self, model, blur_max_px):
        super().__init__()
        args = (model["emb_freqs"], model["emb_freq_min"], model["emb_freq_max"], model["emb_dim"])
        self.sigma, self.blur, self.noise = ScalarEmbed(*args), ScalarEmbed(*args), ScalarEmbed(*args)
        self.has = nn.Embedding(2, model["emb_dim"])  # 0 = 手がかり無し、1 = あり
        self.blur_max_px = float(blur_max_px)

    def forward(self, sigma, blur_px, noise_x, has):
        """sigma / blur_px / noise_x: (B,) FP32。has: (B,) bool。戻り値 (B, D) FP32。"""
        has_f = has.to(torch.float32)[:, None]
        e = self.sigma(sigma.log() / 4.0) + self.has(has.long())
        cue = self.blur(blur_px / self.blur_max_px) + self.noise((noise_x + NOISE_EPS).log() / 4.0)
        return e + has_f * cue
