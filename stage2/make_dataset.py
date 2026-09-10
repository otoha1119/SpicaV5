"""make_dataset.py — Stage 2 の学習データ作成（Docker 内で動く）。[SpicaV5 新規]

INPUT_DIR の <case>/<slice>.png（uint16 1ch、一辺 input_size = 512）を scale 倍（2 → 1024）に補間し、machines.yaml の eidlike1024_dir に同じ <case>/<slice>.png で書く。
値の規約（stored = HU + 1400）は変えない。PCD1024（machines.yaml の pcd1024_dir、別途変換）とはファイル名で対応する。学習時に補間しながら読む方式は取らない（ユーザー決定 2026-09-08）。
変換元（実行ごとに変わる）は sh の変数、出力先 = 学習入力（マシンごとのデータ配置）は machines.yaml、と Stage 1 の infer_stage1.sh / pcd_dir と同じ分担。
設計: docs/plans/20260908_stage2-unet-implementation-spec.md §3

  1. configs/dataset.yaml（scale / interp / input_size）と ../configs/machines.yaml（eidlike1024_dir / num_threads / container_data_root）を configs/schema.py で検証する（既定値なし）
  2. sh からの上書き（DATASET / MACHINE のフラグ、--input_dir）を実効値に反映する。出力先 = 実効値の eidlike1024_dir
  3. 入力を列挙する（Stage 1 の CaseIndex と同じ規則: 直下の症例フォルダを再帰的に、'.' 始まりは無視、PNG 以外は無視して枚数だけ表示）
  4. 出力先は container_data_root 配下で、出力先と出力先.tmp のどちらも存在しないこと（上書きしない。.tmp が残っていれば前回の中断なので手で消す）
  5. 出力先.tmp に 1 枚ずつ書く（num_threads 並列）。1 枚でも失敗すれば止めて .tmp を残す（黙って飛ばさない）
  6. manifest.yaml（由来・方式・症例ごとの枚数・実効設定）を書き、枚数を検算してから出力先に rename する

補間は data/ct_io.py の upsample（cv2.resize、half-pixel 規約: 出力画素 (x+0.5)/scale − 0.5 の位置を入力上で補間。float32 → 四捨五入 → [0, 65535] → uint16）。学習時の実 EID テストスライスも同じ関数。

使い方（通常は ../dataset_stage2.sh 経由）:
  python make_dataset.py --machine PC1 --config configs/dataset.yaml --machines ../configs/machines.yaml \
      --input_dir /workspace/stage1/checkpoints/<run>/infer/<weights>/PCD512_v2/<time>/full -- --interp bicubic [--eidlike1024_dir /workspace/DataSet/EID1024_v1]
"""

import argparse
import datetime
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import yaml
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from configs.schema import DATASET, MACHINE, ConfigError, apply_overrides, check_dataset_values, validate, validate_shared  # noqa: E402
from data.ct_io import read_stored, upsample, write_png  # noqa: E402  補間・読み書きは data/ct_io.py に集約（学習時の実 EID テストスライスと同じ関数）

PATH_FLAGS = ("--input_dir",)  # sh の変数を --flag で上書きできる（start2.sh の引数が最優先）。出力先は machines.yaml のキーなので --eidlike1024_dir（MACHINE のフラグ）
MANIFEST = "manifest.yaml"
EXT = ".png"
ROOT_CASE = "_root"  # 入力直下に直接 PNG がある場合の症例名（Stage 1 CaseIndex と同じ）
SOURCE_INFER_YAML = "infer.yaml"  # Stage 1 推論出力のレイアウト <時刻>/infer.yaml の隣に full/ がある。あれば由来として manifest に写す


def load_yaml(path):
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"設定ファイルが見つかりません: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def split_overrides(tokens):
    """上書き列から sh 変数系のフラグ（--input_dir）を取り出し、残り（DATASET / MACHINE のフラグ）を返す。"""
    paths, rest, i = {}, [], 0
    while i < len(tokens):
        t = tokens[i]
        if t in PATH_FLAGS:
            if i + 1 >= len(tokens) or tokens[i + 1].startswith("--"):
                raise ConfigError(f"{t} には値が必要です")
            paths[t] = tokens[i + 1]
            i += 2
            continue
        rest.append(t)
        i += 1
    return paths, rest


# ---------------------------------------------------------------------------
# 列挙（Stage 1 data/ct_dataset.py CaseIndex._scan と同じ規則。キャッシュは持たない）
# ---------------------------------------------------------------------------
def scan(root):
    """root 直下の症例フォルダを再帰的に列挙する。戻り値 (root からの相対パスのリスト, {症例: 枚数}, PNG 以外のファイル数)。
    '.' 始まりのファイル・フォルダは無視。root 直下に直接ある PNG は症例 "_root" 扱い。"""
    root = Path(root)
    rels, cases, n_other = [], {}, 0
    with os.scandir(root) as it:
        entries = sorted(it, key=lambda e: e.name)
    for e in entries:
        if e.name.startswith("."):
            continue
        if e.is_dir():
            files = []
            for d, dirs, names in os.walk(e.path):
                dirs[:] = sorted(n for n in dirs if not n.startswith("."))
                for f in names:
                    if f.startswith("."):
                        continue
                    if f.lower().endswith(EXT):
                        files.append(os.path.relpath(os.path.join(d, f), root))
                    else:
                        n_other += 1
            files.sort()
            if files:
                cases[e.name] = len(files)
                rels += files
        elif e.is_file():
            if e.name.lower().endswith(EXT):
                cases[ROOT_CASE] = cases.get(ROOT_CASE, 0) + 1
                rels.append(e.name)
            else:
                n_other += 1
    return rels, cases, n_other


# ---------------------------------------------------------------------------
# 1 枚の変換（worker）
# ---------------------------------------------------------------------------
def _init_worker():
    cv2.setNumThreads(1)  # プロセス並列なので OpenCV 内部のスレッドは使わない


def convert_one(args):
    """(in_root, out_root, rel, input_size, scale, interp) → (rel, None) または (rel, エラー文)。"""
    in_root, out_root, rel, input_size, scale, interp = args
    try:
        img = read_stored(Path(in_root) / rel)
        if img.shape != (input_size, input_size):
            raise ValueError(f"入力の大きさが input_size と違います: {img.shape} ≠ ({input_size}, {input_size}): {rel}")
        write_png(Path(out_root) / rel, upsample(img, scale, interp))
        return rel, None
    except Exception as e:  # noqa: BLE001 — worker 内の例外は文字列で親に返し、親が止める
        return rel, f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Stage 2 データセット作成（512 → 1024 補間）")
    p.add_argument("--machine", required=True, help="configs/machines.yaml のエントリ名（eidlike1024_dir = 出力先、num_threads、container_data_root を使う）")
    p.add_argument("--config", required=True, help="方式 YAML（configs/dataset.yaml）")
    p.add_argument("--machines", required=True, help="マシン定義 YAML")
    p.add_argument("--input_dir", required=True, help="変換元（<case>/<slice>.png）")
    p.add_argument("overrides", nargs=argparse.REMAINDER, help="'--' の後に上書き（DATASET / MACHINE のフラグ、--input_dir）")
    a = p.parse_args()
    overrides = a.overrides[1:] if a.overrides and a.overrides[0] == "--" else a.overrides

    try:
        dataset = validate("dataset", load_yaml(a.config), DATASET)
        machines = load_yaml(a.machines)
        if not isinstance(machines, dict) or a.machine not in machines:
            raise ConfigError(f"machines.yaml にエントリ '{a.machine}' がありません。候補: {list(machines) if isinstance(machines, dict) else '(不正な形式)'}")
        machine = validate_shared(f"machines.{a.machine}", machines[a.machine], MACHINE)
        path_over, rest = split_overrides(overrides)
        applied = apply_overrides(rest, {"dataset": (dataset, DATASET), "machine": (machine, MACHINE)})
        check_dataset_values(dataset, machine)
        input_dir = Path(path_over.get("--input_dir", a.input_dir))
        output_dir = Path(machine["eidlike1024_dir"])  # 出力先 = 学習入力（machines.yaml。--eidlike1024_dir で上書き済みならその値）
        if not input_dir.is_dir():
            raise ConfigError(f"入力フォルダがありません: {input_dir}（dataset_stage2.sh の INPUT_DIR か --input_dir を確認）")
        data_root = os.path.normpath(os.path.abspath(machine["container_data_root"]))
        out_abs = os.path.normpath(os.path.abspath(output_dir))
        if os.path.commonpath([data_root, out_abs]) != data_root or out_abs == data_root:
            raise ConfigError(f"出力先（machines.yaml の eidlike1024_dir）は container_data_root（{data_root}）配下のサブフォルダにしてください: {output_dir}")
        tmp_dir = output_dir.with_name(output_dir.name + ".tmp")
        if output_dir.exists():
            raise ConfigError(f"出力先が既に存在します（上書きしない。machines.yaml の eidlike1024_dir を別のバージョンにするか --eidlike1024_dir で指定）: {output_dir}")
        if tmp_dir.exists():
            raise ConfigError(f"前回の中断の痕跡があります。中身を確認して手で消してからやり直してください: {tmp_dir}")
    except ConfigError as e:
        print(f"[make_dataset] 設定エラー: {e}", file=sys.stderr)
        sys.exit(2)

    scale, interp, input_size = dataset["scale"], dataset["interp"], dataset["input_size"]
    output_size = input_size * scale
    n_workers = machine["num_threads"]

    t0 = time.time()
    rels, cases, n_other = scan(input_dir)
    if not rels:
        print(f"[make_dataset] 設定エラー: PNG が見つかりません: {input_dir}", file=sys.stderr)
        sys.exit(2)
    print(f"[make_dataset] {input_dir}: {len(cases)} cases / {len(rels)} files を列挙 ({time.time() - t0:.1f}s)" + (f"、PNG 以外 {n_other} 件は無視" if n_other else ""))
    print(f"[make_dataset] {input_size} → {output_size}（scale {scale}, {interp}）, workers {n_workers} → {tmp_dir} → {output_dir}", flush=True)

    tmp_dir.mkdir(parents=True)
    jobs = [(str(input_dir), str(tmp_dir), rel, input_size, scale, interp) for rel in rels]
    failed = None
    t1 = time.time()
    try:
        if n_workers == 0:
            _init_worker()
            it = map(convert_one, jobs)
            for rel, err in tqdm(it, total=len(jobs), unit="img", dynamic_ncols=True):
                if err:
                    failed = (rel, err)
                    break
        else:
            with Pool(n_workers, initializer=_init_worker) as pool:
                for rel, err in tqdm(pool.imap_unordered(convert_one, jobs, chunksize=8), total=len(jobs), unit="img", dynamic_ncols=True):
                    if err:
                        failed = (rel, err)
                        pool.terminate()
                        break
    except KeyboardInterrupt:
        print(f"\n[make_dataset] 中断しました。途中結果は {tmp_dir} に残っています（消してからやり直す）", file=sys.stderr)
        sys.exit(130)
    if failed:
        print(f"\n[make_dataset] 失敗: {failed[0]}: {failed[1]}\n  途中結果は {tmp_dir} に残っています。原因を直し、.tmp を消してからやり直してください", file=sys.stderr)
        sys.exit(1)
    elapsed = time.time() - t1

    # --- 検算: 出力の枚数が入力と一致し、先頭 1 枚が期待の dtype / 大きさであること ---
    out_rels, out_cases, _ = scan(tmp_dir)
    if sorted(out_rels) != sorted(rels) or out_cases != cases:
        print(f"[make_dataset] 検算エラー: 出力の枚数が入力と一致しません（入力 {len(rels)} / 出力 {len(out_rels)}）。{tmp_dir} を確認してください", file=sys.stderr)
        sys.exit(1)
    first = read_stored(tmp_dir / rels[0])
    if first.shape != (output_size, output_size):
        print(f"[make_dataset] 検算エラー: 出力の大きさが違います {first.shape} ≠ ({output_size}, {output_size}): {rels[0]}", file=sys.stderr)
        sys.exit(1)

    # --- manifest（由来・方式・枚数・実効設定）---
    src_infer_path = input_dir.parent / SOURCE_INFER_YAML
    src_infer = load_yaml(src_infer_path) if src_infer_path.is_file() else None
    jst = datetime.timezone(datetime.timedelta(hours=9), name="JST")
    manifest = {
        "timestamp": datetime.datetime.now(jst).isoformat(timespec="seconds"),
        "machine_name": a.machine,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "output_dir_from": f"machines.yaml [{a.machine}].eidlike1024_dir" + ("（--eidlike1024_dir で上書き）" if "--eidlike1024_dir" in applied else ""),
        "pcd1024_dir": machine["pcd1024_dir"],
        "dataset": {"scale": scale, "interp": interp, "input_size": input_size, "output_size": output_size,
                    "value_convention": "stored = HU + 1400（入力と同じ。補間は float32 → 四捨五入 → clip [0, 65535]）",
                    "pixel_convention": "cv2.resize half-pixel（出力画素 (x+0.5)/scale − 0.5 を入力上で補間）"},
        "n_cases": len(cases), "n_files": len(rels), "n_skipped_non_png": n_other, "cases": cases,
        "elapsed_sec": round(elapsed, 1), "workers": n_workers,
        "overrides": list(overrides), "applied": applied,
        "source_infer_yaml": str(src_infer_path) if src_infer is not None else None,
        "source_infer": src_infer,
        "argv": sys.argv[1:],
    }
    with open(tmp_dir / MANIFEST, "w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, allow_unicode=True, sort_keys=False)

    os.rename(tmp_dir, output_dir)  # 全部揃ってから名前を確定する（途中で止まれば .tmp のまま）
    per_img = elapsed / len(rels)
    print(f"[make_dataset] 完了: {len(cases)} cases / {len(rels)} files → {output_dir}（{elapsed:.0f}s、{per_img * 1000:.0f} ms/枚）。manifest: {output_dir / MANIFEST}")


if __name__ == "__main__":
    main()
