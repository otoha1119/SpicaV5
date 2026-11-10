# stage2/UPSTREAM.md — 新 Stage 2（U + V）の出典・照合・設計書からの差分の記録

設計の正: `docs/plans/20261111_detail_design/stage2_u_flow_spec.md`（設計書）、`RECONCILE.md`（裁定。食い違う所はこちら）。
コード: U の U-Net は `old/stage2/models/unet.py` をそのまま写した。Haar・パネル・症例の列挙・設定の検証・保存と再開は新 Stage 1（`stage1/`）から複製した（Stage 間で import しない）。V・明るさの固定・確認の道具は新規（外部のコードは写していない。式だけ照合した）。

## 1. PMRF の式の照合（2026-11-11）

設計書 C-1 は「調査ノート経由の本文引用。今回は再取得していない」としていた。実装の前に、論文 Ohayon, Michaeli, Elad, "Posterior-Mean Rectified Flow: Towards Minimum MSE Photo-Realistic Image Restoration", ICLR 2025, arXiv:2410.00418 の **arXiv の HTML 版（v3）** と **公式実装 ohayonguy/PMRF**（`lightning_models/mmse_rectified_flow.py`、`train.py`、README）を取得して照合した。
取得は Web 取得用のサブエージェント経由で、下の英文はその報告に載っていた原文の引用（私が論文のページを直接開いたのではない）。「要約のみ」と書いた行は、原文の文字列を取れなかった所。

| 項目 | 原文（引用） | 出どころ | この実装 | 一致 |
|---|---|---|---|---|
| 出発点 | "Here, Z0 ≔ fω∗(Y)+σsϵ, where ϵ ∼ N(0,I) is statistically independent of Y and X, and σs is a hyper-parameter that controls the level of the Gaussian noise added to the posterior mean prediction." | 論文 §3（式 (10) の直後） | `Z0 = μ + σs·ε`（`models/v_model.py flow_loss`、`models/pipeline.py flow`）。μ = 固定した U の出力（MSE で学習 = 事後平均の推定） | ○ |
| 途中 | "where Zt ≔ tX+(1−t)Z0." | 論文 §3 | `zt = (1 − t)·z0 + t·target`（t = 0 が出発点、t = 1 が到達点） | ○ |
| t の分布 | "Zt≔tX+(1−t)(fω∗(Y)+σsϵ), where t is sampled from U[0,1]."／"As in ([Crowson et al., 2024]), we sample t uniformly from U[0,1] using a stratified sampling strategy." | 論文 Algorithm 1／付録 B.1 | `flow.t_sampling: uniform`（設計書どおり、サンプルごとに独立）。`stratified_uniform`（公式実装の既定）も選べる | ○（層別は選択肢） |
| 損失 | "θ∗=argmin_θ ∫₀¹ E[∥(X−Z0)−vθ(Zt,t)∥²] dt"、コード `loss = mse_loss(v_t, x - source_dist_samples)` | 論文 式 (10)・Algorithm 1、公式 `training_step` | `mean((V(Zt, t, σs) − (X′ − Z0))²)`、重み付けなし | ○ |
| V に渡すもの | 論文の vθ(Zt, t)。条件つきの比較対象は別物: "one is conditioned on the degraded measurement Y" / "the other is conditioned on the posterior mean predictor fω∗(Y)"。コードは `--conditional` を付けないとき `return self.model(x_t, t)` | 論文 §5.2、公式 `forward_flow` | `FlowNet.forward(z, t, sigma_s)`。劣化した入力も μ も別には渡さない（設計書 C-3） | ○（σs を足した。下の「変えた点」） |
| 事後平均のモデルの固定 | "The SwinIR model's weights are kept frozen during the vector field's training stage"、コード `param.requires_grad = False`、`self.mmse_model.eval()`、`torch.no_grad()` | 論文 付録 B.1、公式 `__init__`・`training_step` | U は `eval()`・`requires_grad_(False)`・`torch.no_grad()`、optimizer に入れない | ○ |
| 推論 | "Inference (using Euler's method with K steps to solve the ODE)"、"x̂ ← fω∗(y)+σsϵ"、"for i←0,…,K−1 do"、"x̂←x̂+(1/K)vθ∗(x̂,i/K)" | 論文 Algorithm 1 | `z ← z + (1/K)·V(z, k/K, σs)`、k = 0 … K−1（`models/pipeline.py`） | ○ |
| 回数 K | "we use K=25 flow steps in PMRF"（顔の復元）。対照実験は "using K=100 flow steps for each flow algorithm"。"increasing K generally improves the perceptual quality while harming the distortion." | 論文 §5.1.1、§5.2、付録 B.2 | 初期値 25（`flow.steps`、`infer.steps`） | ○ |
| σs の値 | "use σs = 0.1."（顔。画像は [0, 1]）。README: "For denoising, we used `--source_noise_std 0.025`, and for the rest of the tasks we used `--source_noise_std 0.1`." | 論文 付録 B.1、公式 README | 学習は 25〜400 HU の幅、使うときの初期値 100 HU（設計書の決定。論文の値と物差しが違うので比べられない） | 設計書どおり |
| σs を足す理由と加減 | "adding such a noise is critical when the source and target distributions lie on low and high dimensional manifolds, respectively."／"adding noise to fω∗(Y) may harm the MSE of the reconstructions produced by PMRF, and so σs should be taken to be sufficiently small."／"the noise level σs should be carefully tuned, as taking it to be too large or too small may cause the MSE or the perceptual quality of PMRF to degrade, respectively." | 論文 §3、§6 | — | 設計書 C-1 の理解と一致 |
| EMA | 公式実装は評価に EMA（0.9999）を使う（要約のみ） | 公式コード | EMA 0.9999（V）、表示・確認・推論は EMA | ○ |

公式実装にあって、この実装に無いもの（意図して入れていない）:

- 値の切り詰め: 公式は事後平均を `self.mmse_model(y).clip(0, 1)`、出力も `clip(0, 1)` する（論文の式には無い）。この実装は**切らない**（設計書 A-3: 値は線形のまま、上も下も切らない。保存のときだけ uint16 に収める）
- 公式の `--source_noise_std` の既定は 0.0（付け忘れると σs = 0）。この実装は幅を設定に必ず書く
- optimizer は公式が AdamW、この実装は設計書どおり Adam（weight decay 0 なので同じ更新）

取得できなかった所（要約のみ・未確認）: 論文 付録 C.2.2〜C.2.5 と Table 11 の原文、付録 D・E、公式のネットワーク（HDiT）の構成、v3 が ICLR の最終版かどうか。

### PMRF から変えた点（設計書の決定。先例は未確認）

| 点 | 論文 | この実装 | 理由（設計書 C-1・C-3） |
|---|---|---|---|
| σs | 定数（課題ごとに選ぶ） | 学習は対数一様 [25, 400] HU からサンプルごとにくじ引きし、その値を V に埋め込みで教える。使うときに選ぶ | 論文自身が「大きすぎても小さすぎても悪い」と書いている値を先に決め打つと、学習し直しになる。`sigma_s_min_hu = sigma_s_max_hu` にすれば論文どおりの定数 |
| V の条件 | (Zt, t) | (Zt, t, σs)。**形の絵 μ を条件として別に渡さない**（論文の PMRF 本体と同じ。論文が比較対象にした「事後平均で条件づけた flow」にはしない） | μ をきれいなまま見せると、μ に残った手がかりを V が使える |
| 到達点 | X | X′ = X + G∗(x↑ − X)（広い平均を入力のものに差し替えた正解）。出発点の μ も同じ固定つき、最後にも同じ計算 | 明るさは作りで固定（設計書 D） |

## 2. 設計書・RECONCILE から変えた点

| # | 点 | 設計書 | 実装 | 理由 |
|---|---|---|---|---|
| 1 | scalar のタグ名 | `speed/*`、`lock/*`、`leak/*`、`var/*`、`mag/*`、`fake/*`、`train/rmse` | `diag/sec_per_step`、`diag/lock_*`、`diag/leak_*`、`diag/var_gap`、`diag/mag/*`、`diag/fake_*`、`diag/train_rmse` | 監視ツールは名前に `loss` / `val/` / `diag/` / `sigma` を含むタグだけ要約に出す（RECONCILE） |
| 2 | 病院 EID の置き場 | `machines.yaml` に `hospital_eid_dir` を足す | 足さない。`log.eid_slices`（スライス）と `mag.hospital_dirs`（症例フォルダ）にコンテナ内の絶対パスを書く | Stage 1 と同じ流儀。`configs/machines.yaml` に手を入れない（旧 Stage 1 の schema は未知のキーを拒む） |
| 3 | 学習の区切り | U は epoch 数で止まる | U・V とも step で数える（`optim.max_steps` / `steps_per_epoch` / `max_hours`）。U は 40,000 step = 10 epoch、`max_hours` 12 は安全のための上限 | ループを 1 本にした。SIGTERM からの再開が epoch の途中でも同じ |
| 4 | 重みの置き場の名前 | U `weights/epoch_NNN/`、V `weights/step_NNNNNNN/` | どちらも `weights/step_NNNNNNN/` | 同上 |
| 5 | best | 最終 epoch / 最終 step の EMA | 上限（`max_steps` / `max_hours`）で止まったときにだけ、最後の EMA を `best/` に置く（`best.txt` の rule: last）。SIGTERM で止まったときは置かない | 途中で止めた重みを「best」と呼ばない。V の `--u_tag best` が、学習し終えた U だけを指す |
| 6 | 切り抜きの位置 | 4 か所（PCD-002-236、PCD-002-222、EID-049-079、EID-001-093） | 学習中は 2 か所（PCD-002-236、EID-049-079）。確認コマンドは + PCD-002-222。EID-001-093 は外した | EID-001 は P の学習症例（RECONCILE #12）。学習中は表示に出しているスライスだけ切る |
| 7 | 虫眼鏡の imagecas の組 | `[EID-049, EID-001]` | `[EID-049, EID-002]`（test と val） | RECONCILE #12 |
| 8 | 学習中の虫眼鏡の枚数（V） | 症例ごと 20 枚 | 10 枚（`mag.slices_per_case`。確認コマンドは 20） | 1 枚 約 2 秒 × 5 症例 × 30 epoch。学習時間（上限 16 時間）を確認で削りすぎない |
| 9 | `var/gap` | U・V の epoch 末 | U の run だけ。虫眼鏡の対象のスライス（症例ごと 20 枚）で | U だけの性質。V の run で U を 2 回通す時間を省いた |
| 10 | 学習中のパッチの絵（V） | [μ \| U+V \| 正解] | [μ \| 1 段での見積もり `Zt + (1 − t)·V` \| 正解 X′] | パッチごとに 25 回進めると 500 step ごとに学習が止まる。U+V は epoch 末のフルの絵で見る |
| 11 | 補間 | 旧 `upsample` を写す | 同じ `cv2.resize`（bicubic、half-pixel）だが、四捨五入と 0 での切り落としをしない（float のまま） | 設計書 E-2「float のまま」。ファイル化しないので丸める理由が無い |
| 12 | 乱数の種 | `sha1("<seed>/<症例>/<スライス名>")` | sha256（先頭 63 bit）。CPU・FP32 で引いてから GPU へ（ここは設計書どおり） | Stage 1 の `util/seeds.py` と同じ関数にした |
| 13 | 学習パッチ・余白 | U 128 / batch 16、V は 320 四方（余白 32） | U 256 / batch 8、V は 448 四方（余白 96） | RECONCILE #11（明るさの広さが 1024 で σ 32 になったため）。**V の 1 step で U が通す画素が 2 倍になる**（下の 4 節） |
| 14 | 確認コマンドの「旧モデルの出力を横に足す」 | パネルの横に並べる | 表の `old` の段に数字だけ入れる（`check.old_output_dir`。同じ名前のスライスがあるものだけ） | パネルは列の数がそろわない。絵は旧コードの出力フォルダを直接見る |
| 15 | G-2 の手順 5（旧 EID-like で同じ U を学習して見本の値を出す） | 手順にある | 専用のコマンドは無い。旧 EID-like512 を `<pseudo_eid_dir>/<名前>/k00/<症例>/…` に置き、`manifest.yaml`（`k: 1`、`brightness_sigma_px_512`、`image_hw`、`hu_offset: 1400`）を手で書けば、`bash start2.sh u --pseudo_eid_set <名前> --allow_brightness_mismatch true` で同じ U を学習できる | 今回の範囲は学習・推論・確認の道具 |
| 16 | G-4 の `check --flicker` | フラグ | `--what flicker`（`check.what` のリスト） | 設定の流儀（schema のキー） |
| 17 | パラメータ数 | U 約 9.8M、V 約 18M | U 9,803,393、V 17,964,996（本体 17.80M） | 設計書の構成どおりに組んだ実数 |

未実装（今回の範囲の外）: タイル推論（設計書も「作らない」）、DICOM の出力、事前登録（G-5。`docs/experiments/` に書くのは研究者と）。

## 3. 設計書に書いていなくて決めた点

| 点 | 決めた内容 | 理由 |
|---|---|---|
| σs の埋め込みの倍率 | 250 × ln(σs ÷ 100 HU) を正弦波に入れる（25〜400 HU が ±347） | 設計書は「log(σs ÷ 100 HU) を正弦波 64 次元」とだけ書いている。そのままだと ±1.4 で、標準の周波数（1〜1/10000）では高い側しか動かない。t × 1000 と同じ桁にした |
| LayerNorm2d | 画素ごとにチャネル方向で平均 0・分散 1、チャネルごとの重みと偏りつき、eps 1e-5。統計は FP32 で計算 | bf16 は有効 8 ビット。平均を引く計算を丸めない |
| 上りの 2 倍拡大 | 形の操作（expand + reshape） | Stage 1 と同じ（bf16 の最近傍補間の実装の有無に依らない） |
| EMA の序盤 | U・V とも `min(ema_decay, (1 + n) / (10 + n))` | 設計書は V だけに書いている。U も同じ式にした（序盤だけの違い） |
| compile の始め方 | `optim.compile_after_steps`（20）。V は固定した U も同時に compile（`optim.u_compile`） | Stage 1 と同じ。最初の scalar をすぐ出す |
| 最初の確認 | `log.first_check_step`（200）で latest の保存と、症例ごと 2 枚に絞った確認を 1 回通す | 配線の失敗を epoch 末（U 4,000 step、V 5,000 step）まで待たずに見つける |
| epoch 末の順 | latest と weights を保存 → 確認。確認は 1 枚ごとに SIGTERM を見て、受けていたら打ち切る | 監視ツールは SIGTERM の 30 秒後に SIGKILL する。確認（数分）の途中でもその epoch の重みが残る |
| 確認の失敗 | 例外は loss_log に残して学習を続ける。val の値が NaN / Inf なら終了コード 3 | Stage 1 と同じ |
| V の追加の scalar | `val/rmse_mu`・`val/rmse_out`・`val/rmse_input`、`diag/leak_noise_explained` | U の段の値と並べて見るため（どれも見るだけ） |
| 帯域パワーの測り方 | タイル（体内が半分以上）ごとに平均を引いて Hann 窓 → FFT。縁の勾配は中心差分 | 設計書は帯の境と大きさだけ決めている |
| 推論の入力 | `infer.upsample: true`（512 を学習と同じ補間で 1024 に）/ `false`（既に 1024）。大きさからは判定しない | 旧コードの「1 枚目の大きさで判定」は取り違えが起きる |
| 推論・確認の σs | V が学習した幅（`flow.sigma_s_min_hu`〜`max_hu`）の外はエラー | 学習していない強さを黙って使わない |
| V の学習で U と違う組を使う | 注意を出して通す。HU の物差し・補間・症例分割・明るさの広さが U と違えばエラー | 捨て方の違う組を試せるように |
| 再開時の machine | データと run の置き場は今の `machines.yaml` | Stage 1 と同じ |
| 終わり方 | 保存とログを閉じた後、DataLoader の worker を kill して `os._exit` | worker は SIGTERM を無視させている。普通の片付けだと終了が遅れる |

## 4. 知っておくこと（実装して分かった性質。どれも CPU・合成データでの確認）

- **明るさの固定は 1 回では広い平均を完全には一致させない**（Stage 1 の錨と同じ）。残り = (G − G∗G)∗(out − ref)。この式は数値で確かめた（誤差 5e-8）。`diag/lock_out_shift_hu_*` はこの残りを見ている
- σ 32 画素（1024）の Gaussian が通す割合（正弦波で測った）: 周期 400 画素 0.88、200 画素 0.61、100 画素 0.13、50 画素 0.001。設計書 D-2 の数字（σ 16 のとき）の読み替え（RECONCILE #11）
- **U の素の出力 f の広い平均は損失にかからない**（設計書 B-2 のとおり）。学習していない U では f ≈ 0（= 1,348 HU）のままで、μ は「入力をぼかした絵」になる（smoke で確認）。`diag/lock_f_shift_hu_*` が大きいこと自体は異常ではない（流れ続けていないかを見る）
- **V の理論上の視野は片側 約 312 画素（1024 の画素）**（乱数の重みで勾配が届く範囲を測った。U は約 26 画素）。設計書 C-2 は「視野が狭い」と書いているが、4 段・各段 2 ブロックの U-Net なので 256 の学習パッチより広い。学習パッチの中では、どの画素もパッチの端（ゼロ埋め）が視野に入る。1024 一枚の推論では内側の画素に端が見えない。設計書 I-3 (a) の「256 で学習して 1024 に使う」危険にあたる（構成は変えていない。画素ごとの正規化なので、端から離れた所は切り出しても同じ値になることは確かめた: 差 0）
- **速度（未実測。PC で `diag/sec_per_step` を見る）**: RECONCILE #11 で V の学習の U が 448 四方 × 16 枚（設計書の見積もりは 320 四方）になった。設計書 §1 の換算をそのまま使うと、U の順伝播が約 2 倍（0.12〜0.15 → 0.25〜0.3 秒）で、V の 1 step は 0.35〜0.45 秒、15 万 step は 15〜19 時間。**上限 16 時間に先に当たる可能性がある**（そのときは最後の EMA が best になって正常終了する。続きは `resume … --max_hours`）。設定は削っていない
- メモリ（CPU・FP32 の 1024 一枚の forward の最大 RSS。CPU の畳み込みの作業領域込みなので GPU より大きい）: V 6.6 GiB、U 13.1 GiB。GPU では V は特徴 1 枚 67 MB（512²・64ch・FP32）で数 GB 以内の見込み、U は旧コードと同じ（数 GB）
- ゼロ初期化の V は「動かない」: 25 回進めても Z1 = Z0（差 0）。学習の最初の `loss/flow` は (X′ − μ − σs·ε) の 2 乗平均そのもの
- 同じ seed で同じ出力（差 0）、seed を変えると違う絵になる。乱数は CPU で引くので batch の組み方・機械に依らない
