#!/bin/bash
# start.sh — 案内だけ（2026-11-11）。main のこの start.sh は学習を起動しない。
#   新しい Stage 1 / Stage 2（案 E）の起動口はブランチ 028-f_main-redraw_create で作る。
#   2026-11-11 以前の Stage 1 / Stage 2 / ablation は old/ に移した（説明は old/README.md）:
#     bash old/start.sh [マシン名] ...            旧 Stage 1
#     bash old/start2.sh [マシン名] ...           旧 Stage 2
#     bash old/start_ablation.sh [マシン名] <名前> ...   旧 ablation
echo "[start] 新しい Stage 1 / 2 はブランチ 028-f_main-redraw_create で実装中。旧版は bash old/start.sh（Stage 2 は bash old/start2.sh、ablation は bash old/start_ablation.sh。説明は old/README.md）" >&2
exit 1
