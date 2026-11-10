"""make_split.py — 症例分割（train / val / test）の YAML を作る（bash start.sh split → split_stage1.sh → ここ。Docker 内で動く）。[SpicaV5 新規 2026-10-12]

task: sr（一発超解像）の A 側 = EID（379 症例）を val : test : train = 15 : 15 : 70 に分け、configs/splits/eid_v5.yaml に固定する。
両 Stage が同じファイルを読める形（train / val / test の症例名リスト。configs/schema.check_split の規則）。一度作ったら上書きしない（--overwrite で明示）。

  1. 症例の列挙: --dir 直下のフォルダのうち '.' 始まりでなく PNG を 1 枚以上含むもの（症例名 = フォルダ名。EID_v5 と EID1024_v1 は同じ名前なのでどちらで作っても同じ）
  2. --force_test の症例（例 EID-049 = Stage 2 の目視テスト・crop の代表）を test に固定し、残りを seed 付きで shuffle
  3. test = force_test + 先頭から round(n × test) − len(force_test) 症例、val = 次の round(n × val) 症例、train = 残り。各リストは名前順
  4. 由来（dir / n / 比率 / seed / force_test / 時刻）を header に、check_split で検査してから書く

使い方（通常は ../split_stage1.sh 経由）:
  python make_split.py --machine PC1 --machines ../configs/machines.yaml --out ../configs/splits/eid_v5.yaml --val 0.15 --test 0.15 --seed 0 --force_test EID-049
  --dir /workspace/DataSet/EID_v5 で machines.yaml の eid_dir の代わりに直接指定
"""

import argparse
import datetime
import os
import random
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from configs.schema import MACHINE, ConfigError, check_split, validate  # noqa: E402

EXT = ".png"


def list_cases(root):
    """<root>/<症例>/ のうち PNG を含むもの（サブフォルダも見る。'.' 始まりは無視）。"""
    root = Path(root)
    if not root.is_dir():
        raise ConfigError(f"データディレクトリが存在しません: {root}")
    from data.ct_dataset import load_eval_only  # 重い import なので使うときだけ
    eval_only = load_eval_only()
    cases = []
    with os.scandir(root) as it:
        for e in sorted(it, key=lambda e: e.name):
            if e.name.startswith(".") or not e.is_dir():
                continue
            found = False
            for d, dirs, names in os.walk(e.path):
                dirs[:] = [n for n in dirs if not n.startswith(".")]
                if any(n.lower().endswith(EXT) and not n.startswith(".") for n in names):
                    found = True
                    break
            if found and e.name not in eval_only:  # 評価専用（configs/eval_only.yaml）は分割に入れない
                cases.append(e.name)
    if not cases:
        raise ConfigError(f"症例フォルダ（PNG 入り）が見つかりません: {root}")
    return cases


def make_split(cases, val, test, seed, force_test):
    n = len(cases)
    missing = [c for c in force_test if c not in cases]
    if missing:
        raise ConfigError(f"--force_test の症例がディスクにありません: {missing}")
    n_test, n_val = round(n * test), round(n * val)
    if len(force_test) > n_test:
        raise ConfigError(f"--force_test が {len(force_test)} 症例で、test の枠 round({n} × {test}) = {n_test} を超えています")
    if n_test + n_val >= n:
        raise ConfigError(f"val + test が全体以上です: {n_val} + {n_test} ≥ {n}")
    rest = [c for c in cases if c not in set(force_test)]
    random.Random(seed).shuffle(rest)
    test_cases = list(force_test) + rest[: n_test - len(force_test)]
    val_cases = rest[n_test - len(force_test) : n_test - len(force_test) + n_val]
    train_cases = rest[n_test - len(force_test) + n_val :]
    return {"train": sorted(train_cases), "val": sorted(val_cases), "test": sorted(test_cases)}


def main():
    p = argparse.ArgumentParser(description="症例分割 YAML の生成")
    p.add_argument("--machine", default=None, help="configs/machines.yaml のエントリ名（eid_dir を列挙元にする。--dir と排他）")
    p.add_argument("--machines", default=None, help="マシン定義 YAML（--machine のとき必須）")
    p.add_argument("--dir", default=None, help="列挙元のディレクトリ（直下に症例フォルダ）。--machine の代わり")
    p.add_argument("--out", required=True, help="出力 YAML（例 ../configs/splits/eid_v5.yaml）")
    p.add_argument("--val", type=float, required=True, help="val の比率（0.15）")
    p.add_argument("--test", type=float, required=True, help="test の比率（0.15）。残りが train")
    p.add_argument("--seed", type=int, required=True, help="shuffle の seed")
    p.add_argument("--force_test", action="append", default=[], help="test に固定する症例名（複数可）")
    p.add_argument("--overwrite", action="store_true", help="出力が既にあっても上書きする（既定は止まる。分割は一度決めたら固定）")
    a = p.parse_args()
    try:
        if (a.dir is None) == (a.machine is None):
            raise ConfigError("--dir か --machine のどちらか一方を指定してください")
        if a.machine is not None:
            if not a.machines:
                raise ConfigError("--machine には --machines が必要です")
            with open(a.machines, encoding="utf-8") as f:
                machines = yaml.safe_load(f)
            if not isinstance(machines, dict) or a.machine not in machines:
                raise ConfigError(f"machines.yaml にエントリ '{a.machine}' がありません")
            src = validate(f"machines.{a.machine}", machines[a.machine], MACHINE)["eid_dir"]
        else:
            src = a.dir
        if not (0.0 < a.val < 1.0 and 0.0 < a.test < 1.0):
            raise ConfigError("--val / --test は (0, 1)")
        out = Path(a.out)
        if out.exists() and not a.overwrite:
            raise ConfigError(f"出力が既にあります: {out}（分割は固定。作り直すなら --overwrite）")
        cases = list_cases(src)
        split = make_split(cases, a.val, a.test, a.seed, a.force_test)
        check_split(split, "make_split")
    except ConfigError as e:
        print(f"[make_split] エラー: {e}", file=sys.stderr)
        sys.exit(2)
    jst = datetime.timezone(datetime.timedelta(hours=9), name="JST")
    header = {"source_dir": str(src), "n_cases": len(cases), "ratio": {"val": a.val, "test": a.test, "train": round(1.0 - a.val - a.test, 4)},
              "seed": a.seed, "force_test": list(a.force_test), "generated": datetime.datetime.now(jst).isoformat(timespec="seconds"), "generator": "stage1/make_split.py"}
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("# 症例分割（stage1/make_split.py が生成。一度決めたら固定。train = 学習、val = epoch 末のプレビュー、test = 読まない）\n")
        for line in yaml.safe_dump(header, allow_unicode=True, sort_keys=False, default_flow_style=False).strip().splitlines():
            f.write("# " + line + "\n")
        yaml.safe_dump(split, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
    print(f"[make_split] {src}: {len(cases)} cases → train {len(split['train'])} / val {len(split['val'])} / test {len(split['test'])} (force_test {a.force_test}) → {out}")


if __name__ == "__main__":
    main()
