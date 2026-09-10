"""stage2/util/run_paths.py — Stage 2 の run ディレクトリ（<stage2_checkpoints_dir>/<run>/）のレイアウトの唯一の正。
stage1/util/run_paths.py（2026-09-07 第 2 版）から 2026-09-08 に複製し、Stage 2 に合わせた（D が無いので checkpoint は 2 ファイル、画像出力は未実装）。

run 名は起動時刻（JST）を yyyy_mmdd_HHMM にしたもの（例 2026_0908_2130）。'/' は使わない。
checkpoint は「重みディレクトリ」単位で扱う（net_G.pth と state.pth の 2 ファイルが 1 組）。

<stage2_checkpoints_dir>/<run>/
  launch.yaml                 解決済み設定（実効値 = yaml + sh の上書き）。resume 時は launch_resume_<日時秒>.yaml が追加され、最新のものが次の再開の基準
  dataset_info.yaml           train.py が起動時に書く: 症例分割の実効リスト、症例ごとの入力 / 教師 / ペア枚数（切り捨て）
  loss_log.txt
  latest/net_G.pth, state.pth            直下に置く重みディレクトリは latest と best だけ
  best/net_G.pth,   state.pth            自動: 毎 epoch の val で train.yaml log.best_metric が最良を更新したとき（state.pth の best に指標と epoch が入る）
  best.txt                               手動: bash start2.sh best <run> <epoch>（weights/epoch_NNN/ をコピー）。どちらも best.txt に epoch・指標・時刻を書く（2026-09-09）
  weights/epoch_NNN/net_G.pth, state.pth save_epoch_freq ごとの checkpoint（保存は <dir>.tmp → rename で原子的。.tmp / .old が残っていたら中断の痕跡）
  output_images/epoch_NNN/                       生 16bit（stored = HU + 1400、HU が読める）。固定 + ランダム + 実 EID テスト（2026-09-09 ユーザー確定）
      <slice>_eidlike.png, _pcd1024.png, _pcdlike.png      固定（train.yaml log.full_slice）とランダム（log.n_full_random 枚、epoch ごとに別）: 入力 / 教師 / 出力
      <eid_slice>_eid1024.png, _pcdlike.png                実 EID テスト（log.eid_slice、eid_dir の 512 を補間したもの / その出力）
  output_images/preview_fixed_<slice>/           表示用（stored 0〜3500 を線形に、log.preview_bits の深度。HU は読めない）
      00_eidlike1024_<slice>.png, 01_pcd1024_<slice>.png, 02_eid1024_<eid_slice>.png   代表（入力 / 教師 / 実 EID 入力）。初回の checkpoint で 1 回だけ
      epoch_NNN_pcdlike.png, epoch_NNN_eid_pcdlike.png     epoch ごとの出力（固定 / 実 EID テスト）
      epoch_NNN_panel.png                                  [EID-like1024 | PCD-like1024 | PCD1024 | 実 EID → PCD-like1024 | 実 EID1024（元）] の 5 列（並びは 2026-09-10 ユーザー確定。util/panel.py。TB の images/full/fixed と同じ絵）
  output_images/preview_random/epoch_NNN_<slice>.png   ランダムスライスの 3 列パネル [EID-like1024 | PCD-like1024 | PCD1024] だけ（TB の images/full/random と同じ絵）
  tb/                         TensorBoard（scalar と images/full/fixed|random）
  tensorboard.log

推論の出力は run の下ではなく **リポジトリ直下の output/**（Stage 1 と同じ根。.gitignore 済み、コンテナでは /workspace/output/。2026-09-10）:
<repo>/output/<run>_<重みディレクトリ名>_<入力フォルダ名>_<実行時刻>/   bash start2.sh infer の出力（inference_dir.py）。実行ごとに別ディレクトリ
    full/<case>/<slice>.png            PCD-like1024（uint16、stored = HU + 1400）
    full_input1024/<case>/<slice>.png  512 入力を補間した 1024 入力（uint16）。512 入力 かつ infer.yaml の save_input1024 のとき
    full_panel/<case>/<slice>.png      表示用パネル。学習の固定パネルと同じ 5 列 [EID-like1024 | PCD-like1024 | PCD1024 | 実 EID → PCD-like1024 | 実 EID1024]（util/panel.py full_labels）。save_panel のとき
                                       input=eidlike なら 1〜3 列目が入力（教師は teacher）で 4・5 列目は eid_slice、input=eid なら 4・5 列目が入力で 1〜3 列目は pcd_slice
    eid/<eid_slice>_{eid1024,pcdlike}.png            input=eidlike: eid_slice の 1024 入力とその出力（uint16）
    ref/<pcd_slice>_{eidlike,pcdlike,pcd1024}.png    input=eid: pcd_slice の入力・出力・教師（uint16）
    metrics.txt                         出力 vs 教師の rmse / ssim / psnr と入力そのままの参照値（teacher のとき）
    infer.yaml                          解決済み設定（run_infer.py が書く）
"""

import re
from pathlib import Path

RUN_NAME_FORMAT = "%Y_%m%d_%H%M"                 # datetime.strftime 用
RUN_NAME_RE = re.compile(r"^\d{4}_\d{4}_\d{4}$")  # 2026_0908_2130
TOP_TAGS = ("latest", "best")                    # run 直下に置く重みディレクトリ。それ以外（epoch 番号、iter_N）は weights/
WEIGHTS_DIR = "weights"
LAUNCH_FILE = "launch.yaml"
DATASET_INFO_FILE = "dataset_info.yaml"
LOSS_LOG_FILE = "loss_log.txt"
TB_DIR = "tb"
OUTPUT_IMAGES_DIR = "output_images"
CKPT_KINDS = ("net_G.pth", "state.pth")          # 重みディレクトリの中身
BEST_NOTE = "best.txt"                           # run 直下。best/ の由来（epoch、指標、auto | manual、時刻）


def write_best_note(run_dir, epoch, how, metric=None, value=None, total_iters=None, source=None):
    """best.txt を書く（auto: 学習中の自動更新 / manual: mark_best.py）。上書き。"""
    import datetime
    jst = datetime.timezone(datetime.timedelta(hours=9), name="JST")
    lines = [f"epoch: {int(epoch)}", f"how: {how}"]
    if metric is not None:
        lines.append(f"metric: val/{metric} = {value:.6f}")
    if total_iters is not None:
        lines.append(f"total_iters: {int(total_iters)}")
    if source is not None:
        lines.append(f"source: {source}")
    lines.append(f"written_at: {datetime.datetime.now(jst).isoformat(timespec='minutes')}")
    with open(Path(run_dir) / BEST_NOTE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run_name(now):
    """起動時刻 → run 名。"""
    return now.strftime(RUN_NAME_FORMAT)


def epoch_dirname(epoch):
    return f"epoch_{int(epoch):03d}"


def ckpt_dir(run_dir, tag):
    """tag（'latest' | 'best' | epoch 番号 | 'iter_N'）→ 重みディレクトリ。
    latest / best は <run>/<tag>/、epoch 番号は <run>/weights/epoch_NNN/、それ以外の文字列は <run>/weights/<tag>/。"""
    tag = str(tag)
    if tag in TOP_TAGS:
        return Path(run_dir) / tag
    if tag.isdigit():
        return Path(run_dir) / WEIGHTS_DIR / epoch_dirname(tag)
    return Path(run_dir) / WEIGHTS_DIR / tag


def ckpt_path(run_dir, tag, kind):
    """重みディレクトリの中のファイル。kind は 'net_G.pth' | 'state.pth'。"""
    return ckpt_dir(run_dir, tag) / kind


def saved_tags(run_dir):
    """その run にある checkpoint の tag 一覧（net_G.pth がある重みディレクトリから）。"""
    run_dir = Path(run_dir)
    tags = [t for t in TOP_TAGS if (run_dir / t / "net_G.pth").is_file()]
    epochs = []
    for d in (run_dir / WEIGHTS_DIR).glob("*"):
        if d.name.endswith((".tmp", ".old")):
            continue
        if (d / "net_G.pth").is_file():
            m = re.fullmatch(r"epoch_(\d+)", d.name)
            epochs.append(str(int(m.group(1))) if m else d.name)
    return tags + sorted(epochs, key=lambda t: (not t.isdigit(), int(t) if t.isdigit() else t))


def latest_launch(run_dir):
    """run の実効設定ファイル。再開のたびに launch_resume_<日時>.yaml が増えるので、あれば最新（名前順の末尾）、無ければ launch.yaml。"""
    run_dir = Path(run_dir)
    resumes = sorted(run_dir.glob("launch_resume_*.yaml"))
    return resumes[-1] if resumes else run_dir / LAUNCH_FILE


def epoch_images_dir(run_dir, epoch):
    return Path(run_dir) / OUTPUT_IMAGES_DIR / epoch_dirname(epoch)


def preview_dir(run_dir, name):
    """表示用のディレクトリ: <run>/output_images/preview_<name>/（name = "fixed_<slice>" か "random"）"""
    return Path(run_dir) / OUTPUT_IMAGES_DIR / f"preview_{name}"


def find_run_dir(weight_dir):
    """重みディレクトリから run ディレクトリ（launch.yaml がある所）を探す。<run>/best → 1 つ上、<run>/weights/epoch_NNN → 2 つ上。見つからなければ None。"""
    p = Path(weight_dir).resolve()
    for _ in range(3):
        p = p.parent
        if (p / LAUNCH_FILE).is_file():
            return p
    return None


# --- 推論の出力（Stage 1 の run_paths.py と同じ規約。根は <repo>/output/ で両 Stage 共用） ---
OUTPUT_DIRNAME = "output"
INFER_STAMP_FORMAT = "%Y_%m%d_%H%M%S"


def repo_root():
    """リポジトリ直下（このファイルは <repo>/stage2/util/run_paths.py）。コンテナでは /workspace。"""
    return Path(__file__).resolve().parents[2]


def output_root(root=None):
    """推論の出力の根。省略時は <repo>/output/。scratch 実行のときだけ root で差し替える。"""
    return Path(root) if root else repo_root() / OUTPUT_DIRNAME


def infer_output_dir(run_dir, weight_dir, input_dir, now, root=None):
    """推論の出力先: <repo>/output/<run>_<重みディレクトリ名>_<入力フォルダ名>_<yyyy_mmdd_HHMMSS>/。名前に run・重み・入力・時刻を全部入れる（実行ごとに別）。"""
    return output_root(root) / f"{Path(run_dir).name}_{Path(weight_dir).name}_{Path(input_dir).name}_{now.strftime(INFER_STAMP_FORMAT)}"
