"""util/run_paths.py — run ディレクトリ（<checkpoints_dir>/<run>/）のレイアウトの唯一の正。[SpicaV5]

run 名は起動時刻（JST）を yyyy_mmdd_HHMM にしたもの（例 2026_0907_1742）。'/' は使わない（引数や TensorBoard で階層に化けるため）。
checkpoint は「重みディレクトリ」単位で扱う（G と D と optimizer state の 3 ファイルが 1 組なので）。

<checkpoints_dir>/<run>/
  launch.yaml                 解決済み設定（実効値 = yaml + sh の上書き）。resume 時は launch_resume_<日時秒>.yaml が追加され、最新のものが次の再開・推論の基準
  train_opt.txt, loss_log.txt
  latest/net_G.pth, net_D.pth, state.pth        直下に置く重みディレクトリは latest と best だけ
  best/net_G.pth,   net_D.pth, state.pth        bash start.sh best <run> <epoch> で weights/epoch_NNN/ をコピー
  best.txt                    best がどの epoch か（判定基準がまだ無いので当面は目視で手動指定）
  weights/epoch_NNN/net_G.pth, net_D.pth, state.pth   save_epoch_freq ごとの checkpoint（保存は <dir>.tmp → rename で原子的。.tmp / .old が残っていたら中断の痕跡）
  output_images/epoch_NNN/<slice>_{pcd,eidlike,R}.png        checkpoint（毎 epoch）ごとのフル 512（uint16、入力と同じ規約 = HU が読める）
  output_images/preview_fixed_<slice>/                       固定スライス（train.yaml log.full_slice）の表示用（表示範囲で線形、log.preview_bits の深度。HU は読めない）
      00_pcd_<slice>.png, 01_eid_<eid_slice>.png             代表: PCD 入力と EID（log.eid_slice）。初回の checkpoint で 1 回だけ（名前順で先頭に来る）
      epoch_NNN_eidlike.png, epoch_NNN_R_color.png           epoch ごとの EID-like（グレー 1ch）と R（カラー RGB: 白 = 0、純青 = −log.diff_range_hu、純赤 = +。util/residual_color.py）
      epoch_NNN_panel.png                                    上を並べた [EID | EID-like | PCD | R + ゲージ]、各列の下にラベル（util/panel.py。TB の images/full/fixed と同じ絵）
  output_images/preview_random/epoch_NNN_<slice>.png         epoch ごとに別のランダムスライス（log.n_full_random 枚）の同じパネル（パネルだけ。RGB 3ch）
  （学習中の 128 patch グリッドは TensorBoard だけ）
  tb/                         TensorBoard

推論とパッチ切り出しの出力は run の下ではなく **リポジトリ直下の output/**（2026-09-09 ユーザー指示。run の下は 6 階層で探しにくい。.gitignore 済み、コンテナでは /workspace/output/）:
<repo>/output/
  <run>_<重みディレクトリ名>_<入力フォルダ名>_<実行時刻>/   bash start.sh infer の出力（inference_dir.py。uint16 PNG / DICOM）。実行ごとに別ディレクトリ
      {full,patch}/<case>/<slice>.png, {full,patch}_dicom/, {full,patch}_R/（16bit、0 HU = 32768）, {full,patch}_R_color/（表示用カラー 8bit RGB）, R_colorbar_pm<range>HU.png, diff_stats.txt, infer.yaml
  <run>_<重みディレクトリ名>_crop_<実行時刻>/case<N>_<pcd>_<eid>/   bash start.sh crop の出力（crop_patches.py。表示用のみ、16bit 生データは無し）
      1_EID_<eid>_x<X>_y<Y>.png, 2_EID-like_<pcd>_x_y.png, 3_PCD_<pcd>_x_y.png（グレー 1ch、patch 四方、等倍）, 4_R_color_<pcd>_x_y.png（RGB）, panel.png（[EID | EID-like | PCD | R + ゲージ] を panel_scale 倍）
"""

import re
from pathlib import Path

RUN_NAME_FORMAT = "%Y_%m%d_%H%M"                 # datetime.strftime 用
RUN_NAME_RE = re.compile(r"^\d{4}_\d{4}_\d{4}$")  # 2026_0907_1742
TOP_TAGS = ("latest", "best")                    # run 直下に置く重みディレクトリ。それ以外（epoch 番号、iter_N）は weights/
WEIGHTS_DIR = "weights"
OUTPUT_IMAGES_DIR = "output_images"
OUTPUT_DIRNAME = "output"  # 推論・crop の出力の根（<repo>/output/。.gitignore 済み）
BEST_NOTE = "best.txt"
LAUNCH_FILE = "launch.yaml"
CKPT_KINDS = ("net_G.pth", "net_D.pth", "state.pth")  # 重みディレクトリの中身


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
    """重みディレクトリの中のファイル。kind は 'net_G.pth' | 'net_D.pth' | 'state.pth'。"""
    return ckpt_dir(run_dir, tag) / kind


def saved_tags(run_dir):
    """その run にある checkpoint の tag 一覧（net_G.pth がある重みディレクトリから）。"""
    run_dir = Path(run_dir)
    tags = [t for t in TOP_TAGS if (run_dir / t / "net_G.pth").is_file()]
    epochs = []
    for d in (run_dir / WEIGHTS_DIR).glob("*"):
        if d.name.endswith((".tmp", ".old")):  # 原子的保存の作業ディレクトリ（F-14）は checkpoint ではない
            continue
        if (d / "net_G.pth").is_file():
            m = re.fullmatch(r"epoch_(\d+)", d.name)
            epochs.append(str(int(m.group(1))) if m else d.name)
    return tags + sorted(epochs, key=lambda t: (not t.isdigit(), int(t) if t.isdigit() else t))


def latest_launch(run_dir):
    """run の実効設定ファイル。再開のたびに launch_resume_<日時>.yaml が増えるので、あれば最新（名前順の末尾）、無ければ launch.yaml（F-10）。"""
    run_dir = Path(run_dir)
    resumes = sorted(run_dir.glob("launch_resume_*.yaml"))
    return resumes[-1] if resumes else run_dir / LAUNCH_FILE


def find_run_dir(weight_dir):
    """重みディレクトリから run ディレクトリ（launch.yaml がある所）を探す。<run>/best → 1 つ上、<run>/weights/epoch_NNN → 2 つ上。
    見つからなければ None。"""
    p = Path(weight_dir).resolve()
    for _ in range(3):
        p = p.parent
        if (p / LAUNCH_FILE).is_file():
            return p
    return None


def epoch_images_dir(run_dir, epoch):
    return Path(run_dir) / OUTPUT_IMAGES_DIR / epoch_dirname(epoch)


def preview_dir(run_dir, name):
    """表示用（パネル [EID | EID-like | PCD | R + ゲージ] と個別画像）のディレクトリ: <run>/output_images/preview_<name>/（name = "fixed_<slice>" か "random"）"""
    return Path(run_dir) / OUTPUT_IMAGES_DIR / f"preview_{name}"


INFER_STAMP_FORMAT = "%Y_%m%d_%H%M%S"


def repo_root():
    """リポジトリ直下（このファイルは <repo>/stage1/util/run_paths.py）。コンテナでは /workspace。"""
    return Path(__file__).resolve().parents[2]


def output_root(root=None):
    """推論・crop の出力の根。省略時は <repo>/output/。scratch 実行のときだけ root で差し替える。"""
    return Path(root) if root else repo_root() / OUTPUT_DIRNAME


def infer_output_dir(run_dir, weight_dir, input_dir, now, root=None):
    """推論の出力先: <repo>/output/<run>_<重みディレクトリ名>_<入力フォルダ名>_<yyyy_mmdd_HHMMSS>/（2026-09-09、run の下の深い階層から移動）。
    名前に run・重み・入力・時刻を全部入れるので、階層を掘らずに何の出力か分かる。実行時刻付きなので再実行が混ざらない（F-13）。"""
    return output_root(root) / f"{Path(run_dir).name}_{Path(weight_dir).name}_{Path(input_dir).name}_{now.strftime(INFER_STAMP_FORMAT)}"


def crop_output_dir(run_dir, weight_dir, now, root=None):
    """パッチ切り出し（bash start.sh crop）の出力先: <repo>/output/<run>_<重みディレクトリ名>_crop_<yyyy_mmdd_HHMMSS>/。"""
    return output_root(root) / f"{Path(run_dir).name}_{Path(weight_dir).name}_crop_{now.strftime(INFER_STAMP_FORMAT)}"
