"""Splits seam (spec §2, §12.2): committed index files under splits/."""

import numpy as np
import pytest

from mo433 import paths, splits


@pytest.mark.invariant
@pytest.mark.data
def test_regenerating_matches_committed_splits_byte_for_byte(tmp_path):
    splits.write_splits(out_dir=tmp_path)
    committed = sorted(p.name for p in paths.SPLITS_DIR.glob("*.npy"))
    regenerated = sorted(p.name for p in tmp_path.glob("*.npy"))
    assert committed and committed == regenerated
    for name in committed:
        assert (paths.SPLITS_DIR / name).read_bytes() == (tmp_path / name).read_bytes(), name


@pytest.mark.data
def test_labels_match_torchvision_stl10():
    from torchvision.datasets import STL10

    from mo433 import data

    ds = STL10(str(paths.data_root()), split="train", download=False)
    np.testing.assert_array_equal(data.train_labels(), ds.labels)


# The tests below read only the committed files plus the labels, so they need the data.
# STL-10 train has 500 images per class, so per-class counts are exact.


@pytest.fixture(scope="module")
def y():
    from mo433 import data

    return data.train_labels()


@pytest.mark.data
@pytest.mark.parametrize("frac, per_class", [(1, 5), (10, 50), (50, 250), (100, 500)])
def test_label_subsets_are_stratified(y, frac, per_class):
    for draw in (0, 1, 2):
        idx = splits.label_subset(frac, draw)
        assert len(np.unique(idx)) == len(idx) == 10 * per_class
        assert np.bincount(y[idx], minlength=10).tolist() == [per_class] * 10


def test_full_label_subset_is_all_5k_indices():
    for draw in (0, 1, 2):
        np.testing.assert_array_equal(splits.label_subset(100, draw), np.arange(5_000))


def test_label_subset_draws_differ():
    a, b = splits.label_subset(10, 0), splits.label_subset(10, 1)
    assert set(a.tolist()) != set(b.tolist())


@pytest.mark.data
def test_validation_split_is_stratified_tenth_and_bank_is_the_rest(y):
    val, bank = splits.validation_split()
    assert len(val) == 500 and len(bank) == 4_500
    assert np.bincount(y[val], minlength=10).tolist() == [50] * 10
    np.testing.assert_array_equal(np.sort(np.concatenate([val, bank])), np.arange(5_000))


def test_pool_order_is_a_permutation_and_ablation_is_its_first_half():
    order = splits.pool_order()
    assert order.shape == (100_000,)
    np.testing.assert_array_equal(np.sort(order), np.arange(100_000))
    assert not np.array_equal(order, np.arange(100_000))
    np.testing.assert_array_equal(splits.ablation_subset(), order[:50_000])


@pytest.mark.data
def test_cv_folds_partition_labeled_train_stratified(y):
    folds = splits.cv_folds()
    assert len(folds) == 5
    tests = np.concatenate([te for _, te in folds])
    np.testing.assert_array_equal(np.sort(tests), np.arange(5_000))
    for train, test in folds:
        assert np.bincount(y[test], minlength=10).tolist() == [100] * 10
        assert len(np.intersect1d(train, test)) == 0 and len(train) + len(test) == 5_000


def test_unknown_label_subset_is_rejected():
    with pytest.raises(ValueError):
        splits.label_subset(20, 0)
