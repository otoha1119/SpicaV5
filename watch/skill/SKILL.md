---
name: watch-training
description: SpicaV5 の学習を監視し、必要なら止める・直す・回し直すを学習 PC の Codex に依頼し続ける（train-watch）。「学習を監視して」「watch-training」「夜の間見ておいて」で使う。
---

# /watch-training — 学習の監視と自律改良

学習 PC の pc_agent が書く数値（git の watch-data ブランチ）を見て判断し、依頼を書いて学習 PC の Codex に実行させる。
Discord（otoham サーバー #watch-log / #watch-call）に、やり取りを短く残す。

道具は `python3 <watch>/mac.py`（`<watch>` = このスキルの symlink 先の親、つまり `SpicaV5-watch/watch`）。
data リポジトリと作業用 worktree の場所は `<watch>/mac_config.json`（`data_dir` / `work_dir`）。

## 開始
1. `caffeinate -dimsu` を `run_in_background` で起動（スリープ防止。終了時に止める）
2. `mac.py sync` → `mac.py status` で現状を把握し、#watch-log に「監視を始めます（対象 run・見る指標）」を一行
3. 対象 run の EXP 記録（`docs/experiments/`）と、直近の decisions / memory から、この run で何を見るべきかを確認する

## ループ
1. `mac.py wait --timeout 1800` を **run_in_background** で起動して待つ（更新が来たら終わって起こされる。待っている間はトークンを使わない）
2. 起きたら `mac.py sync`。必要に応じて `status/<machine>/<run>.json` の `tags.*.series` を読む
3. 判断して、下の行動のどれかをとる（何もしないなら何もしない。Discord にも書かない）
4. 1 に戻る

## 判断（閾値は固定しない。毎回あなたが決める）
- **止める**: 承認は要らない。発散・崩壊（D→0、高域のゴミ、σ→0、指紋 SD のずれ等の既知の失敗）、改善しない、事前登録の No-Go に当たる、など。
  根拠の数値を添えて `mac.py request --kind stop --run <run> --size none --summary "..."`。止めても latest から resume できるので、迷ったら止めてよい
- **NaN / Inf** は pc_agent が自分で即停止し、応答（kind autostop）が来る。原因を調べて直し、回し直す
- **直して回し直す**: 夜の間も止めない
  - 作業は Mac の `work_dir`（`git worktree add <work_dir> <branch>`、無ければ作る）で行い、ユーザーの作業ツリーには触らない
  - **小**（パラメータ、そのとき見つかったバグの修正）: 同じブランチで直して commit・push → `--size small`
  - **大**（新しい損失・構造・学習方式）: 今のブランチから**新しいブランチを切れば**自分で進めてよい → `--size large`
  - **大を今のブランチに入れる、または Pivot**: しない。`mac.py call "..."` で研究者を呼び、承認を待つ
  - 止める・依頼の `--run` / `--stop_first` は run 名か、同名の run が複数あるときは status の `run_dir`（コンテナ内の絶対パス）
  - 依頼: `mac.py request --machine <PC> --kind run|resume --branch <b> --commit <sha> --command "bash start.sh <PC> ..." --stop_first <run 必要なら> --size small|large --summary "<一行>" --body "<目的・根拠・起動後に見る点>" --note_file <改良内容.md>`
  - 改良したときは必ず改良内容の md（何が問題で、何をどう変えたか、期待する変化）を `--note_file` で渡す（Discord に添付される）
- **呼び出す（#watch-call）**: 大の変更を今のブランチに入れたい、Pivot、候補が尽きた、同じ失敗が 2〜3 回続いた、Codex が needs_user を返した
- **歯止め**: 同じ失敗が 2〜3 回続いたらその方向は打ち切る。1 晩に Codex へ出す依頼は 8 件まで（超えたら呼び出して待つ）。
  候補が尽きたら GPU を空けたまま朝に報告する

## 記録
- 回し直した run は毎回新しい実験として扱い、`docs/experiments/` の EXP 記録に追記する（ネガティブ結果も。work_dir 側でコミット）
- 朝（またはユーザーが戻ったとき）: 夜の間に何で止まり、何を直し、何を回し、結果がどうだったか（ブランチ・commit・数値）を md にまとめ、`mac.py log "朝のまとめ" --file <md>` で投稿
- Discord は簡潔に（「止めました」「lr を 1e-4 にして再開を依頼」程度）。**医用画像とその派生画像は送らない**（数値と md だけ）

## 終了
caffeinate を止め、#watch-log に「監視を終了します」を一行。
