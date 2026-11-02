"""wavelet_packet_generator.py — [SpicaV5] wavelet packet 型 generator（2026-11-02、ブランチ 014-f_013-stage1_wavelet_packet_G）。

Park et al. 2019 Fig. 3（models/wavelet_generator.py）の欠点への構造対策:
  Park の G は wave-dec の高周波 3 帯（HF）を conv を通さずに同段の wave-rec へ直結する（HF skip）。最細スケールの
  高周波は enc 2 層 + dec 2 層 + 最終 conv の浅い経路しか通らず、PCD ノイズ（95 HU、相関長 6 px）の除去や
  MTF 低下のような高周波の加工を構造的にほぼ持てない。残差化（ResidualGenerator）と fidelity が恒等に引く
  バイアスと重なり、「低解像化が弱い」「恒等落ち」の一因と見る（docs/figures/20261102_stage1_generator_wavelet_packet_u.html）。

この G の構成（図: docs/figures/20261102_stage1_generator_wavelet_packet_u.{html,png}）:
  - 各ノードで wave-dec し、L（LL、c ch）と H（LH + HL + HH、3c ch）の**両方**を子ノードとして再帰的に分解する
    （wavelet packet）。depth 段で葉は 2**depth 個（depth 3 → 8 帯域、64²）。
  - 各ノード: ConvBlock（3×3 conv → BN → LReLU）× n_conv（encoder）→ wave-dec → 子 L / 子 H → wave-rec → ConvBlock × n_conv（decoder）。
    葉は encoder だけ。
  - **skip は無い。** wave-dec は完全再構成なので子が親の情報を全部持ち、恒等は外側の ResidualGenerator（出力 = x + R(x)）が担う。
    どの帯域も必ず葉を通る（素通りなし）。
  - 子が親に返すチャネル数は wave-rec に合わせる: L 側 = c、H 側 = 3c。各ノードの内部幅は ch（64）に揃える。
  - 最後に 3×3 conv（活性化なし）で output_nc へ。Park の (e) "an additional convolution layer" と同じ。
  - Haar / BN / LReLU / zero-padding は Park と同じ（HaarDWT / HaarIDWT / ConvBlock を wavelet_generator.py から流用）。

注意: H 枝の再分解は間引きでスペクトルが折り返るので、葉の名前（H の H …）は周波数順ではない
（「H の H」は高域の下半分）。帯域別の損失や乱数を入れるときは周波数順に読み替える。

入力サイズは 2**depth の倍数が必要（schema の check_values が patch_size を検査する）。
"""

import torch.nn as nn

from .wavelet_generator import ConvBlock, HaarDWT, HaarIDWT


class _PacketNode(nn.Module):
    """1 ノード。in_ch → encoder（ch）→ [wave-dec → 子 L / 子 H → wave-rec → decoder] → out_ch。depth == 0 なら葉（encoder だけ）。"""

    def __init__(self, in_ch: int, ch: int, out_ch: int, depth: int, n_conv: int, slope: float, wavelet: str):
        super().__init__()
        self.depth = depth
        if depth == 0:
            # 葉: encoder の最後の conv で親が要求するチャネル数に合わせる
            chans = [in_ch] + [ch] * (n_conv - 1) + [out_ch]
            self.enc = nn.Sequential(*[ConvBlock(chans[i], chans[i + 1], slope) for i in range(n_conv)])
            return
        chans = [in_ch] + [ch] * n_conv
        self.enc = nn.Sequential(*[ConvBlock(chans[i], chans[i + 1], slope) for i in range(n_conv)])
        self.dwt = HaarDWT(wavelet)
        self.idwt = HaarIDWT(wavelet)
        self.child_L = _PacketNode(ch, ch, ch, depth - 1, n_conv, slope, wavelet)          # L: c ch 入力、c ch を返す
        self.child_H = _PacketNode(3 * ch, ch, 3 * ch, depth - 1, n_conv, slope, wavelet)  # H: 3c ch 入力、3c ch を返す
        chans = [ch] * n_conv + [out_ch]
        self.dec = nn.Sequential(*[ConvBlock(chans[i], chans[i + 1], slope) for i in range(n_conv)])

    def forward(self, x):
        h = self.enc(x)
        if self.depth == 0:
            return h
        ll, hf = self.dwt(h)
        y = self.idwt(self.child_L(ll), self.child_H(hf))
        return self.dec(y)


class WaveletPacketGenerator(nn.Module):
    def __init__(
        self,
        input_nc: int = 1,
        output_nc: int = 1,
        ch: int = 64,            # 各ノードの内部チャネル数
        depth: int = 3,          # 分解の段数（葉は 2**depth 個）
        n_conv: int = 2,         # 各ノードの encoder / decoder の ConvBlock 数
        lrelu_slope: float = 0.2,
        wavelet: str = "ortho",
        final_kernel: int = 3,
        final_norm_act: bool = False,
    ):
        super().__init__()
        if depth < 1 or n_conv < 1 or ch < 1:
            raise ValueError(f"wavelet_packet: depth / n_conv / ch は 1 以上: {depth} / {n_conv} / {ch}")
        self.depth = depth
        self.root = _PacketNode(input_nc, ch, ch, depth, n_conv, lrelu_slope, wavelet)
        final = [nn.Conv2d(ch, output_nc, kernel_size=final_kernel, padding=final_kernel // 2)]
        if final_norm_act:
            final += [nn.BatchNorm2d(output_nc), nn.LeakyReLU(lrelu_slope, inplace=True)]
        self.final = nn.Sequential(*final)

    def forward(self, x):
        m = 2**self.depth
        assert x.shape[-1] % m == 0 and x.shape[-2] % m == 0, f"入力サイズは {m} の倍数が必要: {tuple(x.shape[-2:])}"
        return self.final(self.root(x))
