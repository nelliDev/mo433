import numpy as np
import pytest

from mo433 import diagnostics, splits

N, C = 5000, 10


@pytest.fixture
def splits_dir(tmp_path):
    out = tmp_path / "splits"
    out.mkdir()
    for name, arr in splits.generate(_labels(), n_unlabeled=100).items():
        np.save(out / f"{name}.npy", arr)
    return out


def _labels():
    return np.repeat(np.arange(C), N // C)


def _spread(d=64, seed=0, sep=3.0):
    """Class-structured, isotropic-ish features: a good encoder."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(C, d)) * sep
    return (centers[_labels()] + rng.normal(size=(N, d))).astype(np.float32)


def _random_init_like(d=64, seed=1):
    """Row-A stand-in: weak class signal on top of a large shared offset (low spread)."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(C, d)) * 0.3
    return (5.0 + centers[_labels()] + rng.normal(size=(N, d))).astype(np.float32)


def test_constant_embedding_is_flagged(splits_dir):
    anchor = diagnostics.collapse_stats(_random_init_like(), _labels(), splits_dir=splits_dir)
    constant = np.ones((N, 64), np.float32)
    stats = diagnostics.collapse_stats(constant, _labels(), splits_dir=splits_dir)

    result = diagnostics.collapse_check(stats, anchor)

    assert result.flagged
    assert stats.spread == pytest.approx(0.0, abs=1e-6)
    assert result.reasons["spread"]


def test_constant_embedding_is_flagged_even_without_an_anchor(splits_dir):
    stats = diagnostics.collapse_stats(np.ones((N, 64), np.float32), _labels(), splits_dir=splits_dir)
    assert diagnostics.collapse_check(stats, anchor=None).flagged


def test_well_spread_class_structured_embedding_passes(splits_dir):
    anchor = diagnostics.collapse_stats(_random_init_like(), _labels(), splits_dir=splits_dir)
    stats = diagnostics.collapse_stats(_spread(), _labels(), splits_dir=splits_dir)

    result = diagnostics.collapse_check(stats, anchor)

    assert not result.flagged, result.reasons
    assert stats.spread > 0.3
    assert stats.knn > anchor.knn and stats.nmi > anchor.nmi


def test_encoder_no_better_than_its_anchor_is_flagged(splits_dir):
    a = diagnostics.collapse_stats(_random_init_like(), _labels(), splits_dir=splits_dir)
    result = diagnostics.collapse_check(a, anchor=a)  # equal => "<=" holds
    assert result.flagged and result.reasons["knn"] and result.reasons["nmi"]


def test_teacher_check_passes_a_good_embedding_and_fails_a_constant_one(splits_dir):
    good = diagnostics.teacher_check(diagnostics.collapse_stats(_spread(), _labels(), splits_dir=splits_dir))
    bad = diagnostics.teacher_check(
        diagnostics.collapse_stats(np.ones((N, 64), np.float32), _labels(), splits_dir=splits_dir)
    )
    assert good.passed, good.reasons
    assert not bad.passed
    assert good.reference_knn == 0.745


def test_anchor_id_maps_runs_to_row_a_at_the_same_size_and_seed():
    assert diagnostics.anchor_id("B-S-100-s0") == "A-S-s0"
    assert diagnostics.anchor_id("C-L-050-s1") == "A-L-s1"
    assert diagnostics.anchor_id("A-S-s2") is None
    assert diagnostics.anchor_id("D") is None
