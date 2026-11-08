"""Extra experiments (after the frozen study): run lists, curves, and the extra student sizes."""

import json

import pytest

from mo433 import config, extras, students


def test_long_study_is_s_b_and_c_at_300_epochs_two_seeds():
    runs = extras.STUDIES["long300"].runs
    assert [(r.row, r.size, r.frac, r.seed, r.epochs) for r in runs] == [
        ("B", "S", 100, 0, 300), ("C", "S", 100, 0, 300), ("B", "S", 100, 1, 300), ("C", "S", 100, 1, 300)]


def test_sizes_study_adds_xs_and_m_with_two_seeds_at_the_frozen_budget():
    runs = extras.STUDIES["sizes"].runs
    assert {(r.size, r.seed) for r in runs} == {("XS", 0), ("XS", 1), ("M", 0), ("M", 1)}
    assert {r.row for r in runs} == {"B", "C"} and {r.epochs for r in runs} == {None}


def test_extra_sizes_follow_the_single_width_rule():
    # Same scaling rule as S/L (spec §4.2): only the base width changes.
    assert students.WIDTHS["XS"] == 12 and students.WIDTHS["M"] == 34
    assert config.SIZES == ("S", "L")  # the frozen study is unchanged
    for size in ("XS", "M"):
        for row in ("B", "C"):
            cfg = config.load_config(row, size)
            assert cfg.student_width == students.WIDTHS[size]
            s_cfg = config.load_config(row, "S")
            assert (cfg.epochs, cfg.optimizer, cfg.head_width) == (s_cfg.epochs, s_cfg.optimizer, s_cfg.head_width)
    params = {s: sum(p.numel() for p in students.build_student(s, 0).parameters()) for s in ("XS", "S", "M", "L")}
    assert params["XS"] < params["S"] < params["M"] < params["L"]


def test_curve_rows_read_metrics_jsonl(tmp_path):
    rd = tmp_path / "B-S-100-s0"
    rd.mkdir()
    lines = [{"epoch": e, "loss": 10.0 - e, "val_knn": 0.1 * e, "lr": 0.001} for e in (1, 2, 3)]
    (rd / "metrics.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    rows = extras.curve_rows(rd, run_id="B-S-100-s0", study="long300", epochs=300)
    assert [r["epoch"] for r in rows] == [1, 2, 3]
    assert rows[2] == {"study": "long300", "run_id": "B-S-100-s0", "epochs": 300, "epoch": 3,
                       "loss": 7.0, "val_knn": pytest.approx(0.3)}


def test_extra_run_ids_never_collide_with_the_main_runs():
    for study in extras.STUDIES.values():
        for r in study.runs:
            assert extras.run_dir(study.name, r).parts[-3:-1] == ("extras", study.name)
