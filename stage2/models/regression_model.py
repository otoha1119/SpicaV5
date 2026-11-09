"""stage2/models/regression_model.py — Stage 2 の教師あり回帰（U-Net + MSE / L1）の司令塔。junyanz の BaseModel は使わない。

  forward   : pred = net(input)（mode.residual=true なら input + net(input)）
  loss      : mse | l1 を正規化空間 [−1,1] で計算。表示用に HU 換算（1.0 = hu_per_unit HU）の RMSE / MAE も返す
  optimizer : Adam(lr, (beta1, beta2))。lr は一定（論文に減衰の記載なし）→ scheduler は持たない
  checkpoint: 重みディレクトリ（util/run_paths.py）に net_G.pth と state.pth（optimizer / RNG / epoch / total_iters / **保存時の実効設定 config**）を**原子的に**保存
              （<dir>.tmp に書く → 既存 <dir> を .old に → .tmp を <dir> に → .old を消す。Stage 1 の save_networks と同じ手順）
  resume    : state.pth から optimizer と RNG（python / torch / numpy）を復元し、lr / betas は今の実効値を再適用（上書きが最優先）
"""

import copy
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from util.metrics import mse, psnr, ssim
from models.unet import build_net
from util.run_paths import ckpt_dir, write_best_note

BETTER = {"rmse": lambda new, old: new < old, "ssim": lambda new, old: new > old, "psnr": lambda new, old: new > old}  # log.best_metric ごとの「更新」の向き


def _swap_dirs(tmp, final, old, retries=10, wait=3.0):
    """tmp を final に差し替える（Stage 1 と同じ。Windows バインドマウントで os.replace が PermissionError になる対策、2026-11-03）。
    再試行 → ファイル単位の置き換え → それでも駄目なら警告して学習を続ける（次回の保存で再試行）。"""
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


def _to_tuple(x):
    """random.setstate は入れ子の tuple を要求する。torch.load 後に list になっている場合に戻す。"""
    if isinstance(x, list):
        return tuple(_to_tuple(v) for v in x)
    return x


class RegressionModel:
    def __init__(self, train, mode, machine, device):
        self.device = device
        self.config = {"train": dict(train), "mode": dict(mode), "machine": dict(machine)}  # この重みに対応する実効設定（state.pth に入れる。再開の基準）
        self.residual = mode["residual"]
        self.net = build_net(mode).to(device)
        # 高速化（Stage 1 と同じ、2026-11-03）: 学習の forward だけ bf16 autocast + torch.compile。val・監視・推論は compile しない FP32 の self.net（同じ重み）。
        # compile は torch.compile で別の呼び出し口を作る（state_dict のキーは self.net のまま）。CUDA Graph の出力 buffer 再利用に備えて学習の出力は clone する
        self.use_bf16 = train["optim.precision"] == "bf16" and device.type == "cuda"
        self.compile_enabled = bool(train["optim.compile"]) and device.type == "cuda"
        if self.compile_enabled and not (shutil.which(os.environ.get("CC", "cc")) or shutil.which("gcc")):
            raise RuntimeError("torch.compile に必要な C コンパイラがありません。Docker image を再ビルドしてください（down → build）")
        self.net_train = torch.compile(self.net, mode="max-autotune") if self.compile_enabled else self.net
        if mode["loss"] == "mse":
            self.criterion = nn.MSELoss()
        elif mode["loss"] == "l1":
            self.criterion = nn.L1Loss()
        else:
            raise ValueError(mode["loss"])
        self.loss_name = mode["loss"]
        self.lr, self.betas = train["optim.lr"], (train["optim.beta1"], train["optim.beta2"])
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, betas=self.betas)
        # 出力側 D（2026-11-09 ブランチ 023。研究者の設計 2026-10-12「MSE + 実 EID 出力側 D」）。adv.lambda 0 なら無し（main と同じ）
        self.adv_lambda, self.adv_source = float(train["adv.lambda"]), train["adv.source"]
        self.net_D, self.optimizer_D, self.gan = None, None, None
        if self.adv_lambda > 0.0:
            from models.discriminator import GANLoss, PatchDiscriminator
            self.net_D = PatchDiscriminator(1, int(train["adv.ndf"])).to(device)
            self.gan = GANLoss(train["adv.gan_mode"])
            self.gan_thresh = 0.5 if train["adv.gan_mode"] == "lsgan" else 0.0   # D_acc の判定の境界（lsgan は 0.5、vanilla は logit 0）
            self.optimizer_D = torch.optim.Adam(self.net_D.parameters(), lr=float(train["adv.lr_d"]), betas=(0.5, 0.999))
        # G の EMA（2026-11-09 ブランチ 024）: decay > 0 なら学習中の self.net とは別に平均した重み self.net_ema を持ち、val / best / 推論用の net_G.pth はそれを使う
        self.ema_decay = float(train["ema.decay"])
        self.net_ema = None
        if self.ema_decay > 0.0:
            self.net_ema = copy.deepcopy(self.net).eval()
            for q in self.net_ema.parameters():
                q.requires_grad_(False)
        self.cur_epoch, self.epoch_done, self.total_iters = 0, False, 0
        self.best = None  # best/ の記録 {"metric", "value", "epoch", "total_iters"}。state.pth に入れて resume で引き継ぐ（無ければ None = 次の val が必ず best）
        self.last = None  # 直近の学習バッチ (input, pred, target)。optimize が更新
        self.n_params = sum(p.numel() for p in self.net.parameters())

    # --- 前向き ---
    def forward(self, x, net=None):
        net = self.net if net is None else net
        y = net(x)
        return x + y if self.residual else y

    @property
    def net_eval(self):
        """val / 監視 / 推論に使う G。EMA があれば EMA、無ければ学習中の重み。"""
        return self.net_ema if self.net_ema is not None else self.net

    @torch.no_grad()
    def _ema_update(self):
        d = self.ema_decay
        for q_ema, q in zip(self.net_ema.parameters(), self.net.parameters()):
            q_ema.mul_(d).add_(q.detach(), alpha=1.0 - d)
        for b_ema, b in zip(self.net_ema.buffers(), self.net.buffers()):
            b_ema.copy_(b)

    def optimize(self, batch):
        """1 step。戻り値は表示用の dict: loss（正規化空間の損失）、mse（出力 vs 教師。monitor が √ を取って train/rmse に。0 へ）、ssim（1 へ）。
        forward は bf16 autocast（optim.precision）+ compile（optim.compile）。出力は FP32 に戻してから損失を計算する。"""
        x = batch["input"].to(self.device, non_blocking=True)
        t = batch["target"].to(self.device, non_blocking=True)
        self.net.train()
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_bf16):
            y = self.net_train(x)
        y = y.float()
        if self.compile_enabled:
            y = y.clone()
        pred = x + y if self.residual else y
        loss_mse = self.criterion(pred, t)
        loss, extra, fake = loss_mse, {}, None
        if self.net_D is not None:
            # 偽物 = 実 EID patch の出力（adv.source eid）、ペアの出力（eidlike）、両方（both）。本物 = 教師 PCD1024 patch
            fakes = []
            if self.adv_source in ("eid", "both"):
                e = batch["eid"].to(self.device, non_blocking=True)
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_bf16):
                    ye = self.net_train(e)
                ye = ye.float()
                if self.compile_enabled:
                    ye = ye.clone()
                fakes.append(e + ye if self.residual else ye)
            if self.adv_source in ("eidlike", "both"):
                fakes.append(pred)
            fake = torch.cat(fakes, 0)
            self.net_D.train()
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_bf16):
                d_fake_g = self.net_D(fake)
            loss_adv = self.gan(d_fake_g.float(), True)
            loss = loss_mse + self.adv_lambda * loss_adv
            extra["adv_G"] = float(loss_adv.detach())
            extra["mse_only"] = float(loss_mse.detach())
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        if self.net_ema is not None:
            self._ema_update()
        if self.net_D is not None:
            self.optimizer_D.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_bf16):
                d_real = self.net_D(t)
                d_fake = self.net_D(fake.detach())
            d_real, d_fake = d_real.float(), d_fake.float()
            loss_D = 0.5 * (self.gan(d_real, True) + self.gan(d_fake, False))
            loss_D.backward()
            self.optimizer_D.step()
            with torch.no_grad():
                extra["D_loss"] = float(loss_D); extra["D_real"] = float(d_real.mean()); extra["D_fake"] = float(d_fake.mean())
                extra["D_acc"] = 0.5 * (float((d_real > self.gan_thresh).float().mean()) + float((d_fake <= self.gan_thresh).float().mean()))
        with torch.no_grad():
            self.last = (x.detach(), pred.detach(), t.detach())  # TB の images/current 用（util/monitor.py log_images）
            return {"loss": float(loss), "mse": mse(pred, t), "ssim": ssim(pred, t), **extra}

    @torch.no_grad()
    def predict(self, x):
        """監視・検証用の順伝播（eval、勾配なし）。x は (N,1,H,W) 正規化 tensor（CPU 可）。"""
        net = self.net_eval
        net.eval()
        return self.forward(x.to(self.device), net).cpu()

    @torch.no_grad()
    def evaluate(self, dataset, slices):
        """val のフル画像で rmse（正規化空間。0 へ）/ ssim（1 へ）/ psnr（大きいほど良い）をスライス平均で出す。入力そのまま（何もしない場合）の値も *_input として返す（参照線）。"""
        net = self.net_eval
        net.eval()
        acc = {"mse": 0.0, "ssim": 0.0, "psnr": 0.0, "mse_input": 0.0, "ssim_input": 0.0, "psnr_input": 0.0}
        n = 0
        for _case, in_path, pcd_path in slices:
            x = dataset.load_full(in_path).to(self.device)
            t = dataset.load_full(pcd_path).to(self.device)
            y = self.forward(x, net)
            for name, pred in (("", y), ("_input", x)):
                acc["mse" + name] += mse(pred, t); acc["ssim" + name] += ssim(pred, t); acc["psnr" + name] += psnr(pred, t)
            n += 1
        if n == 0:
            raise RuntimeError("検証スライスがありません")
        out = {k: v / n for k, v in acc.items()}
        out = {"rmse": out["mse"] ** 0.5, "ssim": out["ssim"], "psnr": out["psnr"], "rmse_input": out["mse_input"] ** 0.5, "ssim_input": out["ssim_input"], "psnr_input": out["psnr_input"], "n": n}
        return out

    def update_best(self, run_dir, metric, val, epoch, total_iters):
        """epoch 末の val で log.best_metric が最良を更新したら best/ に保存し best.txt を書く。戻り値: 更新したか。
        self.best を先に更新してから保存するので、best/state.pth の best には自分自身の記録が入る。"""
        value = float(val[metric])
        if self.best is not None and self.best.get("metric") == metric and not BETTER[metric](value, float(self.best["value"])):
            return False
        self.best = {"metric": metric, "value": value, "epoch": int(epoch), "total_iters": int(total_iters)}
        self.save(run_dir, "best")
        write_best_note(run_dir, epoch, "auto", metric=metric, value=value, total_iters=total_iters, source=f"val on {val['n']} slices")
        return True

    # --- checkpoint ---
    def _state(self):
        np_state = np.random.get_state()
        return {
            "epoch": int(self.cur_epoch), "epoch_done": bool(self.epoch_done), "total_iters": int(self.total_iters),
            "best": self.best,  # best/ の記録（自動更新の比較対象。resume で引き継ぐ）
            "optimizer": self.optimizer.state_dict(),
            "optimizer_D": self.optimizer_D.state_dict() if self.optimizer_D is not None else None,   # 出力側 D（無ければ None）
            "lr": float(self.optimizer.param_groups[0]["lr"]),
            "config": self.config,  # 保存時の実効設定（train / mode / machine、平坦キー）。resume はこれを基準にする（失敗した起動の launch を引き継がない。レビュー指摘 2026-09-09）
            "rng": {
                "python": random.getstate(),
                "torch": torch.get_rng_state(),
                "numpy": (np_state[0], torch.from_numpy(np_state[1].astype(np.int64)), int(np_state[2]), int(np_state[3]), float(np_state[4])),
            },
        }

    def save(self, run_dir, tag):
        """net_G.pth と state.pth を重みディレクトリに原子的に保存する。"""
        final = ckpt_dir(run_dir, tag)
        tmp, old = final.with_name(final.name + ".tmp"), final.with_name(final.name + ".old")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        torch.save(self.net_eval.state_dict(), tmp / "net_G.pth")   # 推論・評価用（EMA があれば EMA）
        if self.net_ema is not None:
            torch.save(self.net.state_dict(), tmp / "net_G_raw.pth")   # 学習中の重み（resume 用）
        if self.net_D is not None:
            torch.save(self.net_D.state_dict(), tmp / "net_D.pth")
        torch.save(self._state(), tmp / "state.pth")
        _swap_dirs(tmp, final, old)  # Windows バインドマウントの PermissionError 対策

    def load_weights(self, weight_dir):
        g = torch.load(Path(weight_dir) / "net_G.pth", map_location=self.device, weights_only=True)
        raw = Path(weight_dir) / "net_G_raw.pth"
        if self.net_ema is not None:
            self.net_ema.load_state_dict(g)
            if raw.is_file():
                self.net.load_state_dict(torch.load(raw, map_location=self.device, weights_only=True))
            else:
                self.net.load_state_dict(g)   # EMA なしの run から EMA 付きで再開: 学習中の重みも net_G.pth から
                print("[resume] 注意: net_G_raw.pth が無いので学習中の重みも net_G.pth から読む（EMA なしの run からの再開）")
        else:
            self.net.load_state_dict(torch.load(raw, map_location=self.device, weights_only=True) if raw.is_file() else g)   # EMA 付きの run から EMA なしで再開: 学習中の重みを引き継ぐ
        if self.net_D is not None:
            p = Path(weight_dir) / "net_D.pth"
            if p.is_file():
                self.net_D.load_state_dict(torch.load(p, map_location=self.device, weights_only=True))
            else:
                print(f"[resume] 注意: {p} が無いので D は初期化のまま（D なしの run から D 付きで再開）")

    def resume(self, state_path):
        """state.pth から optimizer / RNG / 進捗を復元する。state は CPU に読む（RNG 用 Tensor を CUDA に載せない）。lr / betas は今の実効値を再適用。"""
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        self.optimizer.load_state_dict(state["optimizer"])
        for g in self.optimizer.param_groups:
            g["lr"] = self.lr
            g["betas"] = self.betas
        if self.optimizer_D is not None and state.get("optimizer_D"):
            self.optimizer_D.load_state_dict(state["optimizer_D"])
        rng = state["rng"]
        random.setstate(_to_tuple(rng["python"]))
        torch.set_rng_state(rng["torch"].cpu())
        kind, keys, pos, has_gauss, cached = rng["numpy"]
        np.random.set_state((kind, keys.cpu().numpy().astype(np.uint32), pos, has_gauss, cached))
        self.total_iters = int(state["total_iters"])
        self.best = state.get("best")  # 古い state（best 実装前）には無い → None（次の val が best になる。best/ が既にあれば上書きされる）
        best_msg = f"best {self.best['metric']} {self.best['value']:.6f} @ epoch {self.best['epoch']}" if self.best else "best 記録なし"
        print(f"[resume] {state_path}: epoch {state['epoch']} (done={state['epoch_done']}), total_iters {state['total_iters']}, lr {self.optimizer.param_groups[0]['lr']:.7f}, {best_msg}")
        return state
