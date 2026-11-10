# stage1/UPSTREAM.md — 新 Stage 1（描き手 P）の出典・照合・設計書からの差分の記録

設計の正: `docs/plans/20261111_detail_design/stage1_painter_spec.md`（設計書）、`RECONCILE.md`（裁定）。コードは新規（外部のコードは写していない。式だけ照合した）。

## 1. EDM の式の照合（2026-11-11、一次情報 = 公式実装）

設計書 D-1 は「式は記憶に基づく。原文 Table 1 と照合が必要」としていた。論文 Karras, Aittala, Aila, Laine, "Elucidating the Design Space of Diffusion-Based Generative Models", NeurIPS 2022, arXiv:2206.00364 の**公式実装 NVlabs/edm**（論文 Table 1 の "Ours" 列を実装したもの）の該当行を取得して照合した。論文の PDF / HTML の Table 1 そのものは今回開いていない（照合の相手は公式コード）。

| 項目 | 公式コード（取得した行） | 出どころ | この実装 | 一致 |
|---|---|---|---|---|
| c_skip | `c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)` | `training/networks.py` `EDMPrecond.forward` | `diffusion/edm.py precond` | ○ |
| c_out | `c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()` | 同上 | 同上 | ○ |
| c_in | `c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()` | 同上 | 同上 | ○ |
| c_noise | `c_noise = sigma.log() / 4` | 同上 | `models/embed.py`（s_σ = ln σ / 4） | ○ |
| 合成 | `D_x = c_skip * x + c_out * F_x.to(torch.float32)`（F には `c_in * x` を入れる） | 同上 | `models/painter.py denoise`（合成は FP32） | ○ |
| 損失の重み | `weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2`、`loss = weight * ((D_yn - y) ** 2)` | `training/loss.py` `EDMLoss` | (F − 正解)²、正解 = (x0 − c_skip·x) / c_out | ○（下の検算） |
| ノイズ | `n = torch.randn_like(y) * sigma`、`net(y + n, sigma, ...)` | 同上 | x = x0 + σ·ε | ○ |
| σ の引き方（学習） | `sigma = (rnd_normal * self.P_std + self.P_mean).exp()`（P_mean −1.2、P_std 1.2） | 同上 | **違う（設計書の決定）**: ln σ を [ln σ_min, ln σ_max] で一様 | 設計書どおり |
| σ の並び（描画） | `t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho`、末尾に 0 | `generate.py` `edm_sampler` | `diffusion/edm.py karras_sigmas` | ○ |
| 既定値 | `sigma_min = 0.002`、`sigma_max = 80`、`rho = 7`、`sigma_data = 0.5` | 同上 | σ_min 0.002、**σ_max 40**、ρ 7、σ_data 0.5 | σ_max だけ設計書の値 |
| Euler の 1 段 | `d_cur = (x_hat - denoised) / t_hat`、`x_next = x_hat + (t_next - t_hat) * d_cur` | 同上 | 下の ancestral（η = 0 でこれと同じ） | ○ |

検算（損失の 2 つの形が同じこと）: D = c_skip·x + c_out·F なので D − x0 = c_out·(F − (x0 − c_skip·x)/c_out)。λ = (σ² + σ_d²)/(σ·σ_d)² = 1/c_out² だから λ·(D − x0)² = (F − 正解)²。

検算（端の挙動。`precond` を実際に計算した値、σ_d = 0.5）:

| σ | c_skip | c_out | c_in | 意味 |
|---|---|---|---|---|
| 0.002 | 0.999984 | 0.002000 | 1.999984 | x̂0 ≈ x + σ·F: ほぼ入力そのまま。F はノイズ（の符号反転）を当てる |
| 0.5 | 0.5 | 0.353553 | 1.414214 | 混合 |
| 40 | 0.000156 | 0.499961 | 0.024998 | x̂0 ≈ σ_d·F: F が絵そのもの（÷ σ_d）を当てる。入力は c_in で分散 1 に正規化される |

注意: 設計書は σ_max = 40 を「データの広がりの 80 倍（EDM の標準と同じ比）」と書いているが、公式の既定は σ_max = 80（σ_data 0.5 の 160 倍）。値は設計書の 40 のまま実装した（設定 `diffusion.sigma_max`。再開では変えられない）。

## 2. Euler ancestral の式の照合（2026-11-11）

公式 EDM の sampler は Heun（決定的 + churn）で、祖先サンプリングは無い。設計書 F-1 の式は **crowsonkb/k-diffusion** `k_diffusion/sampling.py` と照合した。

| 項目 | k-diffusion（取得した行） | この実装 |
|---|---|---|
| σ_up | `sigma_up = min(sigma_to, eta * (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2) ** 0.5)` | `diffusion/edm.py ancestral_step`（同じ。min も入れた） |
| σ_down | `sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5` | 同上 |
| 更新 | `d = to_d(x, sigmas[i], denoised)`（= (x − denoised)/σ）、`dt = sigma_down - sigmas[i]`、`x = x + d * dt`、`if sigmas[i + 1] > 0: x = x + noise * s_noise * sigma_up` | `diffusion/sampler.py`: x ← x̂0 + (σ_down/σ_i)·(x − x̂0) + σ_up·ε（代数的に同じ） |
| σ の並び | `sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho`、`append_zero` | `karras_sigmas`（公式 EDM と同じ式） |

違い: この実装は毎段、x̂0 に明るさの錨を掛けてから上の更新をする（設計書 F-2）。描き始めは m + σ_0·ε（m = 元の絵の広い平均。設計書 D-2）。

## 3. 設計書から変えた点

| # | 点 | 設計書 | 実装 | 理由 |
|---|---|---|---|---|
| 1 | ResBlock の中の FiLM の位置 | Conv3×3 → FiLM → GroupNorm → SiLU → Conv3×3 | Conv3×3 → GroupNorm → FiLM（×(1 + scale) + shift）→ SiLU → Conv3×3 | FiLM の直後に GroupNorm を置くと scale / shift の大半が正規化で消える。定番（ADM の scale-shift norm）の順にした |
| 2 | scalar のタグ名 | `train/*`、`check/*`、`val_bin/*`、`time/hours` | `diag/*`、`diag/check_*`、`val/loss_bin/*`、`diag/hours` | 監視ツールが名前に `loss` / `val/` / `diag/` / `sigma` を含むタグだけ要約に出す（RECONCILE） |
| 3 | `loss_cue/*` | none / blur_only / noise_only / both | + `clean`（手がかりあり・σ_b = σ_c = 0） | 4 つに入らない組み合わせが約 2 % ある |
| 4 | 明るさを平均する広さのキー | `generate.yaml` の `anchor.sigma_px` | `configs/shared.yaml` の `brightness.sigma_px_512` 1 か所（上書き `--brightness_sigma_px_512`） | RECONCILE #1 と追加の裁定 |
| 5 | 途中経過のスライス | `log.eid_slices`（相対）、`log.pcd_slice` | `log.eid_slices` / `log.pcd_slices` = コンテナ内の絶対パスの一覧。起動前に有無と「train 側でない」ことを検査 | 追加の裁定（病院 EID は `eid_dir` の外） |
| 6 | 上りの 2 倍拡大 | 最近傍 ×2（`F.interpolate` を想定） | 形の操作（expand + reshape）で同じ計算 | bf16 の最近傍補間は torch の版によって CUDA で未実装だったことがある。結果は同一（CPU で一致を確認） |
| 7 | `run_check.py` / `check.py` / `check.yaml`（設計書 H の段 1・2） | I-1 にある | **未実装** | 今回の範囲は学習・描画・best。学習中の `diag/check_*` とパネルは実装済み |
| 8 | 学習データの平均・標準偏差 | 起動時に 2,000 パッチ測る | 学習の最初の 2,000 パッチ（約 63 step）で測って `dataset_info.yaml` に記録 | 起動を遅らせない |
| 9 | パラメータ数 | 約 4,000〜4,500 万 | 46,904,356（本体 46.37 M） | 設計書の構成どおりに組んだ実数 |

## 4. 設計書に書いていなくて決めた点

| 点 | 決めた内容 | 理由 |
|---|---|---|
| 埋め込みの角度 | 角度 = 周波数 × スカラー（ラジアン。2π を掛けない） | 周波数の単位が未指定。スカラーの幅（約 2.5）で最低周波数 0.25 が単調になる |
| skip の取り方 | ADM と同じ（stem の出力 + 各 ResBlock + 各 Down の出力を、上りの各 ResBlock が 1 つずつ連結） | 「上り 3 個（skip を連結）」を満たす定番 |
| compile の始め方 | `optim.compile_after_steps`（20）: 最初の 20 step は compile せず、21 step 目から。`optim.compile_mode` で mode を選べる | 最初の scalar をすぐ出す。max-autotune が PC で落ちたら yaml / flag だけで下げられる |
| 最初の確認 | `log.preview_first_step`（200）: 起動して 200 step 目に val・パネル・latest の保存を 1 回通す。scalar は 1, 2, 4, …, 64 step 目にも出す | 配線の失敗を 15 分待たずに見つける |
| bf16 の確認 | val のたびに同じ入力を bf16 でも通し、帯ごとの比を `diag/bf16_val_ratio/bin*` に出す | 設計書 C-4 の「必ず見ること」を自動にした |
| 時間の上限 | `optim.max_hours` は再開をまたいだ累計（`state.pth` の `elapsed_sec`） | 「36 時間」を使い切ったら止まる |
| 再開時の machine | データと run の置き場は今の `machines.yaml`（学習の設定ではなく「今どこにあるか」） | 旧コードの 2026-11-11 の扱いに合わせた |
| SIGTERM | いまの step を終えて latest を保存、終了コード 143。DataLoader の worker は SIGTERM を無視し、main が保存後に片付ける | 監視ツールは train.py の全プロセスに SIGTERM を送る |
| NaN | 損失・勾配・val が有限でなければ、重みを更新・保存せず終了コード 3。有限でない scalar は書かない | 監視の条件 |
| 途中経過の絵の失敗 | 例外を loss_log に残して学習は続ける | 夜間に絵 1 枚で学習を止めない |
| 手がかり無しの描画 | 描き始めは σ_0·ε（広い平均なし）、錨なし | 元の絵が無い |
| 疑似 EID の組の名前 | `<run>_step<N>_b<σ_b>_c<σ_c>_a<σ_a>[_<suffix>]`（latest でも step 番号を入れる） | 同じ名前で中身が違う組を作らない |
| 拡大 | 各一覧の 1 つ目だけ（病院 EID は位置が決まっていない） | 位置を推測で決めない |

## 5. 知っておくこと（実装して分かった性質）

- 明るさの錨 x̂0 + (G(y) − G(x̂0)) は、Gaussian が「2 回掛けても同じ」ではないので、**1 回では σ_a の広さの平均を完全には一致させない**（残り = (G − G²)(y − x̂0)。σ_a よりずっと広い成分はほぼ一致）。合成データの smoke（ほとんど学習していない重み、絵と元の差 平均 114 HU）で、出力と入力の広い平均の差は σ_a = 16 で平均 1.7〜1.8 HU、σ = 32 で 0.1 HU。`diag/check_*_wide_mean_diff_hu` はこの残りを見ている
- 画像ごとの乱数源で、batch の組み方を変えても同じ絵になる（CPU の smoke で最大 1 HU = 丸めの差）
