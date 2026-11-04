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
1. `mac.py wait --timeout <秒>` を **run_in_background** で起動して待つ（応答・自動停止・job の終了など status/ 以外の更新が来たら終わって起こされる。数値（status/）だけの更新では起きないので、数値は timeout ごとの定期確認で見る。待っている間はトークンを使わない）。timeout は下の「確認の間隔」で毎回自分で決める
2. 起きたら `mac.py sync`。必要に応じて `status/<machine>/<run>.json` の `tags.*.series` を読む
3. 判断して、下の行動のどれかをとる（何もしないなら何もしない。Discord にも書かない）
4. 1 に戻る

## 確認の間隔
timeout は表や段階を決めずに、その時々の状況を見てあなたが毎回選ぶ。
崩れそうなとき（起動直後、回し直した直後、兆候が出たとき）は短く、落ち着いていれば長く。
数値は pc_config の `interval_s` ごとにしか届かないことだけ頭に入れておく。

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
- **調べる・評価する（job）**: 数値だけで判断できないとき、判断の材料を自分で作る。承認は要らない。
  例: 学習中の重みで実 EID を推論して縁の幅・ノイズ・HU を測る、Stage 2 の序盤の epoch で実 EID がぼけるかを数値にする、ログや run ディレクトリの中身を集計する
  - スクリプトは Mac で書き（置き場は `$CLAUDE_JOB_DIR/tmp` か SpicaV5 の `docs/experiments/<EXP>_scripts/`。**SpicaV5 のリポジトリには commit しない**）、`--script` で添付する
  - `mac.py request --machine <PC> --kind job --size none --summary "<一行>" --script <file> [--script ...] --command '<コンテナ内のコマンド>' [--branch <b> --commit <sha>] --body "<目的・判断に使う数値>"`
  - PC では依頼の commit（省略時は学習中のコードの HEAD）が学習リポジトリ内の別 worktree（`/workspace/.watch_jobs/<id>/`）に展開され、そこを cwd にコンテナ内で裏で走る。学習は止まらないし、学習中の作業ツリーも変わらない。添付は `_job/<file>`、出力先は環境変数 `$SPICA_JOB_OUT`（`--command` は単引用符で囲む）
  - 走っている間は、ログの末尾 20 行が数値の送信と同じ間隔で `status/<PC>/_jobs.json` に届く（`mac.py status` に出る。これでは起きないので定期確認で見る）。終わると応答が来て起こされる。`$SPICA_JOB_OUT` のテキスト（json / csv / md / txt / yaml、1 ファイル 200 KB まで）だけが watch-data の `results/<id>/` に届く。画像はそこに置いても PC の `data_dir/.local/jobs/<id>/` に残るだけで届かない。**判断に要るものは数値か文章にして出す**
  - GPU は学習と共有する。バッチを小さくし、長く占有しない
- **学習コードに測る仕組みを足す**: 毎 epoch 見たい量（例: Stage 2 の実 EID の縁の鋭さ）は、TB の scalar に書くようにすれば exporter がそのまま status に載せる。足し方は「直して回し直す」と同じ（測るだけで学習を変えないなら小）
- **重みを Mac に取ってきて自分で見る**: job で `$SPICA_JOB_OUT/blobs/` に重みなどを置くと（1 ファイル 95 MB まで。画像・ボリュームの拡張子は送られない）、使い捨てブランチ `watch-blob/<id>` で届く。
  `mac.py fetch-blobs <id> --to $CLAUDE_JOB_DIR/tmp/<id>` で取り出す（ブランチは消える）。例: `--command 'mkdir -p "$SPICA_JOB_OUT/blobs" && cp <run_dir>/latest/net_G.pth <run_dir>/latest/state.pth "$SPICA_JOB_OUT/blobs/"'`（`<run_dir>` は status の run_dir）
  Mac 上の実データ（SSD の DataSet、`~/Downloads/PCD-plaque-eval` の評価ペアと手で引いた線）で推論し、数値と画像を**自分で**見て判断する（画像は Mac の中だけ。外に出さない）。前例のスクリプトは SpicaV5 の `docs/experiments/EXP-20261103-01_scripts/`（b_stage2.py = 線の断面の内腔 HU・10–90 % 幅・undershoot）
- **chain（Stage 1 → 推論 → Stage 2）の判定（重要）**: Stage 2 は序盤（epoch 1〜2）で、実 EID がぼけるかがほぼ分かる。Stage 2 の最初の checkpoint ができたらすぐ重みを取ってきて、実 EID と EID-like の出力の縁（10–90 % 幅、PCD1024 の真値との差）を比べる。
  EID-like だけ鋭く実 EID がぼけるなら、Stage 2 を止めて Stage 1 から直し、chain を回し直してよい
- **呼び出す（#watch-call）**: 大の変更を今のブランチに入れたい、Pivot、候補が尽きた、同じ失敗が 2〜3 回続いた、Codex が needs_user を返した
- **歯止め**: 同じ失敗が 2〜3 回続いたらその方向は打ち切る。1 晩に Codex へ出す依頼は 8 件まで（超えたら呼び出して待つ）。
  候補が尽きたら GPU を空けたまま朝に報告する

## 記録
- 回し直した run は毎回新しい実験として扱い、`docs/experiments/` の EXP 記録に追記する（ネガティブ結果も。work_dir 側でコミット）
- 朝（またはユーザーが戻ったとき）: 夜の間に何で止まり、何を直し、何を回し、結果がどうだったか（ブランチ・commit・数値）を md にまとめ、`mac.py log "朝のまとめ" --file <md>` で投稿
- Discord は簡潔に（「止めました」「lr を 1e-4 にして再開を依頼」程度）。**医用画像とその派生画像は送らない**（数値と md だけ）

## 終了
caffeinate を止め、#watch-log に「監視を終了します」を一行。
