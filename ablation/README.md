# ablation/ — 比較実験（アブレーション）

研究の主系統（Stage 1 / Stage 2 / Stage 3）とは別に、**比較対象として回す手法**を 1 つずつディレクトリに自己完結で置く。
主系統のコード（`stage1/` / `stage2/`）からは import せず、必要なものは複製する（Stage 間の流儀と同じ）。共有するのは Docker（`docker/`）、`configs/machines.yaml`、`configs/splits/` だけ。

## 起動

リポジトリ直下の **`start_ablation.sh`** から。例: `bash start_ablation.sh PC1 CycleGAN`。コンテナは Stage 1 / 2 と同じ `spicav5`（起動済みなら exec だけ。学習中に打っても作り直さない）。

```bash
bash start_ablation.sh [マシン名] <アブレーション名> [infer|build|shell|down|tb] [resume <run> [tag]] [best <run> <epoch>] [--flag value ...]
bash start_ablation.sh PC3 SRCycleGAN                      # 学習
bash start_ablation.sh PC3 SRCycleGAN infer --max_slices 4 --save_panel true
bash start_ablation.sh PC3 SRCycleGAN resume 2026_1027_1200
bash start_ablation.sh PC3 SRCycleGAN tb                   # その ablation の全 run の TensorBoard（machines.yaml の ablation_tb_port = 6008）
```

- 保存先: `machines.yaml` の `ablation_checkpoints_dir`（`/workspace/ablation/checkpoints`）の下に `<名前>/<run>/`。推論の出力は `<repo>/output/ablation/<名前>/`
- 各ディレクトリの契約: `train.sh` / `infer.sh`（コンテナ内の入口。`SPICA_MACHINE` / `SPICA_RESUME` / `SPICA_RESUME_TAG` を受ける）、`mark_best.py`（`--machine --machines --run --epoch`）、`configs/`（設定。既定値なし、`schema.py` が正）。この契約を守れば `start_ablation.sh` は変えずに名前を増やせる

## 一覧

| 名前 | 目的 | 出典 | 追加日 |
|---|---|---|---|
| `SRCycleGAN` | 論文オリジナルの SR-CycleGAN（MMSR 損失あり）で実 EID512 → PCD-like1024 を一発 ×2 SR。SpicaV2 の失敗を同じデータ・表示で定量記録し、Stage 1 `task: sr` / Stage 2 と並べる | Zheng et al., J Med Imaging 9(2) 024003 (2022)。コードは著者公開版（SpicaV2 commit 741924c） | 2026-10-27 |
| `CycleGAN` | 普通の CycleGAN（junyanz 本家の既定）で EID1024（×2 補間済み）→ PCD-like1024 の同解像度・非ペア変換。SRCycleGAN の MMSR 損失（出力を EID 側に引き戻す拘束）を持たない比較対象 | Zhu et al., ICCV 2017。コードは junyanz 本家（stage1/ に vendoring 済み） | 2026-10-27 |

計画・設計: `docs/plans/20261027_stage1-sr-cyclegan-plan.md`。変更履歴: 各ディレクトリの `UPSTREAM.md`。
