import torch


def noise_estimation_loss(model,
                          x0: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor, keepdim=False):
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(x, t.float())
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)


loss_registry = {
    'simple': noise_estimation_loss,
}


# [SpicaV5 追加 2026-10-13] 条件付き（入力の channel 連結）。上の本家 noise_estimation_loss との違いは model の入力に cond を連結することだけ
# （x_t の作り方・損失 = (C,H,W) の和の batch 平均は同じ）。形は IGITUGraz/WeatherDiffusion models/ddm.py の noise_estimation_loss
# （x0 の前半 channel を条件、後半を目標にして torch.cat([条件, x_t], dim=1)）と同じで、ここでは条件と目標を別引数にした。
def noise_estimation_loss_cond(model,
                               x0: torch.Tensor,
                               cond: torch.Tensor,
                               t: torch.LongTensor,
                               e: torch.Tensor,
                               b: torch.Tensor, keepdim=False):
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(torch.cat([cond, x], dim=1), t.float())
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)
