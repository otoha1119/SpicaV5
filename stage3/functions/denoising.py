import torch


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


def generalized_steps(x, seq, model, b, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')
            et = model(xt, t)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))
            c1 = (
                kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to('cpu'))

    return xs, x0_preds


def ddpm_steps(x, seq, model, b, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        xs = [x]
        x0_preds = []
        betas = b
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(betas, t.long())
            atm1 = compute_alpha(betas, next_t.long())
            beta_t = 1 - at / atm1
            x = xs[-1].to('cuda')

            output = model(x, t.float())
            e = output

            x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e
            x0_from_e = torch.clamp(x0_from_e, -1, 1)
            x0_preds.append(x0_from_e.to('cpu'))
            mean_eps = (
                (atm1.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * x
            ) / (1.0 - at)

            mean = mean_eps
            noise = torch.randn_like(x)
            mask = 1 - (t == 0).float()
            mask = mask.view(-1, 1, 1, 1)
            logvar = beta_t.log()
            sample = mean + mask * torch.exp(0.5 * logvar) * noise
            xs.append(sample.to('cpu'))
    return xs, x0_preds


# =============================================================================
# [SpicaV5 追加 2026-10-13] ここから下は本家に無い。条件付き（入力の channel 連結）のサンプラ。上の本家の関数は変えていない（stage3 では使わない）。
# 出典: IGITUGraz/WeatherDiffusion utils/sampling.py（MIT、commit 5f78729、../LICENSE_WeatherDiffusion。Özdenizci & Legenstein, TPAMI 2023）。
#       WeatherDiffusion 自体が本ファイル（ermongroup/ddim）を写して条件の連結を足したもの。
#   generalized_steps_cond        : WeatherDiffusion の generalized_steps。変更は 'cuda' 固定 → x.device だけ
#   generalized_steps_overlapping : 同名関数（毎ステップ、重なるタイルの ε を画素ごとに平均してから全体で 1 回 DDIM 更新）。変更点:
#       ・'cuda' 固定 → x.device
#       ・x_cond に data_transform（[0,1] → [−1,1]）を掛けない（ここでは呼び出し側で正規化済み [−1, 1]）
#       ・タイルをまとめる数（WeatherDiffusion は manual_batching_size = 64 固定）を引数 tile_batch に。t はそのバッチと同じ長さにする
#       ・batch は 1 枚だけ（WeatherDiffusion の manual_batching も et_output[0] に足すので 1 枚前提）。違えばエラー
#       ・どのタイルにも入らない画素があれば 0 割になるのでエラー（overlapping_grid_indices 参照）
#   overlapping_grid_indices      : WeatherDiffusion models/restoration.py の同名関数（range(0, h − p + 1, r)）。
#       (h − p) が r で割り切れないと右端・下端が、r > p だとタイルの間が被覆されないので、ここではエラーにする（WeatherDiffusion は検査しない）
# 条件 x_cond は model の入力の前、x_t は後ろ（torch.cat([x_cond, x_t], dim=1)。WeatherDiffusion / mirthAI Fast-DDPM と同じ順）。
# =============================================================================

def generalized_steps_cond(x, x_cond, seq, model, b, eta=0.):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to(x.device)

            et = model(torch.cat([x_cond, xt], dim=1), t)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))

            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to('cpu'))
    return xs, x0_preds


def overlapping_grid_indices(h, w, p_size, r):
    if p_size > min(h, w):
        raise ValueError(f"タイル {p_size} が画像 {h}x{w} より大きい")
    if r < 1 or r > p_size or (h - p_size) % r or (w - p_size) % r:
        raise ValueError(f"stride r={r} で {h}x{w} を {p_size} タイルで被覆できません（1 ≤ r ≤ p で、(H − p) と (W − p) が r の倍数であること。1024 / 256 なら r = 16, 32, 64, 128, 256 など）")
    h_list = [i for i in range(0, h - p_size + 1, r)]
    w_list = [i for i in range(0, w - p_size + 1, r)]
    return h_list, w_list


def generalized_steps_overlapping(x, x_cond, seq, model, b, eta=0., corners=None, p_size=None, tile_batch=None):
    with torch.no_grad():
        n = x.size(0)
        if n != 1:
            raise ValueError(f"generalized_steps_overlapping は 1 枚ずつ（batch {n}）")
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]

        x_grid_mask = torch.zeros_like(x_cond, device=x.device)
        for (hi, wi) in corners:
            x_grid_mask[:, :, hi:hi + p_size, wi:wi + p_size] += 1
        if bool((x_grid_mask == 0).any()):
            raise ValueError("どのタイルにも入らない画素があります（corners を確認）")
        x_cond_patch = torch.cat([x_cond[:, :, hi:hi + p_size, wi:wi + p_size] for (hi, wi) in corners], dim=0)

        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to(x.device)
            et_output = torch.zeros_like(x_cond, device=x.device)

            xt_patch = torch.cat([xt[:, :, hi:hi + p_size, wi:wi + p_size] for (hi, wi) in corners], dim=0)
            for k in range(0, len(corners), tile_batch):
                chunk = corners[k:k + tile_batch]
                tk = (torch.ones(len(chunk)) * i).to(x.device)
                outputs = model(torch.cat([x_cond_patch[k:k + tile_batch], xt_patch[k:k + tile_batch]], dim=1), tk)
                for idx, (hi, wi) in enumerate(chunk):
                    et_output[0, :, hi:hi + p_size, wi:wi + p_size] += outputs[idx]

            et = torch.div(et_output, x_grid_mask)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))

            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to('cpu'))
    return xs, x0_preds
