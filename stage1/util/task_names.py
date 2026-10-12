"""util/task_names.py — task（mode.yaml の task: degrade | sr）ごとの A / G(z) / B の呼び名。[SpicaV5 新規 2026-10-12]

学習（util/monitor.py のファイル名・パネルのラベル）、推論（inference_dir.py のパネル・DICOM の SeriesDescription）、crop（crop_patches.py のファイル名・ラベル）が
同じ表を引く。モデル（models/fidelity_gan_model.py の real_A / fake_B / real_B）と損失は task によらず同じで、変わるのは「どのデータを A / B に入れるか」と表示の名前だけ。

  degrade : A = PCD（z）→ G(z) = EID-like → B = EID（x）。論文どおりの劣化生成（Park 2019）
  sr      : A = EID1024（z）→ G(z) = PCD-like → B = PCD1024（x）。一発超解像（ablation。docs/plans/20261012_stage1-sr-mode-plan.md）

  file   : output_images/epoch_NNN/<slice>_<file.a|gen|b>.png、preview_fixed_*/00_<a>_… 01_<b>_…、crop の 1_<B>_ 2_<GEN>_ 3_<A>_
  label  : パネルのラベル帯（英数のみ。cv2 Hershey）
"""

TASKS = ("degrade", "sr")

_NAMES = {
    "degrade": {"a": "pcd", "gen": "eidlike", "b": "eid", "A": "PCD", "GEN": "EID-like", "B": "EID"},
    "sr":      {"a": "eid", "gen": "pcdlike", "b": "pcd", "A": "EID", "GEN": "PCD-like", "B": "PCD"},
}


def names(task):
    """task → {"a", "gen", "b"}（ファイル名用、小文字）と {"A", "GEN", "B"}（ラベル用）。未知の task はエラー。"""
    if task not in _NAMES:
        raise ValueError(f"task は {TASKS} のどれか: {task!r}")
    return dict(_NAMES[task])


def full_labels(task, epoch_or_tag):
    """フル画像パネルのラベル（並び順 B の代表 → G(z) → A → R）。epoch_or_tag は 'epoch N' か重みディレクトリ名。"""
    n = names(task)
    return [n["B"], f"{n['GEN']}  G(z)   {epoch_or_tag}", f"{n['A']}  z", "R = G(z) - z   [HU]"]
