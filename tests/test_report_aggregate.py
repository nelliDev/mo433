"""Aggregation math (spec §8) on a hand-built tiny fixture.

Expected values are worked out by hand in the comments, never recomputed the way the
code does it.
"""

import math

import pytest

from mo433 import report


def _row(encoder_id, frac, draw, acc, probe="linear_svm", feature_set="main"):
    return {
        "encoder_id": encoder_id,
        "feature_set": feature_set,
        "label_frac": frac,
        "draw": draw,
        "probe": probe,
        "accuracy": acc,
    }


# B-S-100: seed 0 draws (0.60, 0.62, 0.64) -> 0.62; seed 1 (0.70, 0.70, 0.70) -> 0.70;
#          seed 2 (0.64, 0.66, 0.68) -> 0.66. Seed means 0.62, 0.66, 0.70: mean 0.66, sd 0.04.
# C-S-100: seed means 0.70, 0.72, 0.74 (constant draws): mean 0.72, sd 0.02.
# Gap C - B = +0.06; reference sd = max(0.04, 0.02) = 0.04 -> beyond spread.
TINY = (
    [_row("B-S-100-s0", 10, d, a) for d, a in enumerate((0.60, 0.62, 0.64))]
    + [_row("B-S-100-s1", 10, d, 0.70) for d in range(3)]
    + [_row("B-S-100-s2", 10, d, a) for d, a in enumerate((0.64, 0.66, 0.68))]
    + [_row(f"C-S-100-s{s}", 10, d, a) for s, a in enumerate((0.70, 0.72, 0.74)) for d in range(3)]
    # D: draws 0.80, 0.84, 0.88 -> mean 0.84, sd over draws 0.04
    + [_row("D", 10, d, a) for d, a in enumerate((0.80, 0.84, 0.88))]
    # L: two seeds, means 0.50 and 0.60 -> mean 0.55, sd 0.0707
    + [_row("B-L-100-s0", 10, d, 0.50) for d in range(3)]
    + [_row("B-L-100-s1", 10, d, 0.60) for d in range(3)]
    + [_row("C-L-100-s0", 10, d, 0.54) for d in range(3)]
    + [_row("C-L-100-s1", 10, d, 0.58) for d in range(3)]
)


def test_parse_encoder_ids():
    assert report.parse_encoder("B-S-100-s0") == report.EncoderKey("B", "S", 100, 0)
    assert report.parse_encoder("C-L-050-s1") == report.EncoderKey("C", "L", 50, 1)
    assert report.parse_encoder("A-L-s1") == report.EncoderKey("A", "L", None, 1)
    assert report.parse_encoder("D") == report.EncoderKey("D", None, None, None)
    assert report.parse_encoder("B-S-100-s0").group == "B-S-100"
    assert report.parse_encoder("A-S-s2").group == "A-S"
    assert report.parse_encoder("D").group == "D"
    with pytest.raises(ValueError):
        report.parse_encoder("X-S-s0")


def test_draws_averaged_within_seed_then_sd_over_seeds():
    cells = report.aggregate(TINY)
    b = cells[("B-S-100", "linear_svm", 10)]
    assert b.mean == pytest.approx(0.66)
    assert b.sd == pytest.approx(0.04)
    assert b.spread == "seeds"
    assert b.n == 3
    assert b.per_seed == pytest.approx({0: 0.62, 1: 0.70, 2: 0.66})


def test_teacher_uses_sd_over_draws():
    d = report.aggregate(TINY)[("D", "linear_svm", 10)]
    assert d.mean == pytest.approx(0.84)
    assert d.sd == pytest.approx(0.04)
    assert d.spread == "draws"
    assert d.n == 3


def test_large_student_keeps_individual_runs():
    b = report.aggregate(TINY)[("B-L-100", "linear_svm", 10)]
    assert b.mean == pytest.approx(0.55)
    assert b.sd == pytest.approx(0.1 / math.sqrt(2))
    assert b.per_seed == pytest.approx({0: 0.50, 1: 0.60})


def test_single_seed_has_undefined_sd():
    rows = [_row("B-S-100-s0", 10, d, 0.6) for d in range(3)]
    cell = report.aggregate(rows)[("B-S-100", "linear_svm", 10)]
    assert cell.mean == pytest.approx(0.6)
    assert math.isnan(cell.sd)


def test_gap_beyond_spread_uses_larger_seed_sd():
    gaps = report.gaps(report.aggregate(TINY))
    g = gaps[("S", 100, "linear_svm", 10)]
    assert g.gap == pytest.approx(0.06)
    assert g.ref_sd == pytest.approx(0.04)
    assert g.beyond is True
    # L: gap 0.56 - 0.55 = 0.01; ref sd = max(0.0707, 0.0283) -> within spread
    gl = gaps[("L", 100, "linear_svm", 10)]
    assert gl.gap == pytest.approx(0.01)
    assert gl.ref_sd == pytest.approx(0.1 / math.sqrt(2))
    assert gl.beyond is False


def test_gap_missing_when_a_side_is_missing():
    rows = [r for r in TINY if not r["encoder_id"].startswith("C-L")]
    assert ("L", 100, "linear_svm", 10) not in report.gaps(report.aggregate(rows))


def test_gap_with_undefined_sd_is_never_beyond():
    rows = [_row("B-S-100-s0", 10, d, 0.5) for d in range(3)] + [_row("C-S-100-s0", 10, d, 0.9) for d in range(3)]
    g = report.gaps(report.aggregate(rows))[("S", 100, "linear_svm", 10)]
    assert g.gap == pytest.approx(0.4)
    assert g.beyond is False


def test_feature_sets_are_kept_apart():
    rows = TINY + [_row("B-S-100-s0", 10, d, 0.1, feature_set="ms_concat") for d in range(3)]
    main = report.aggregate(rows)[("B-S-100", "linear_svm", 10)]
    assert main.mean == pytest.approx(0.66)
    ms = report.aggregate(rows, feature_set="ms_concat")[("B-S-100", "linear_svm", 10)]
    assert ms.mean == pytest.approx(0.1)
