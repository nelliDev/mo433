"""EuroSAT -> STL-10 binary layout (the second-dataset extra)."""

import numpy as np
from PIL import Image

from mo433 import data, eurosat


def _fake_imagefolder(root, classes=("A", "B"), per_class=10):
    rng = np.random.default_rng(0)
    for c in classes:
        (root / c).mkdir(parents=True)
        for i in range(per_class):
            Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(root / c / f"{c}_{i}.jpg")


def test_materialize_writes_a_stratified_split_in_the_stl10_layout(tmp_path, monkeypatch):
    src, out = tmp_path / "2750", tmp_path / "eurosat"
    _fake_imagefolder(src)
    manifest = eurosat.materialize(src, out, test_frac=0.2, seed=0, n_labeled=8)

    d = out / "stl10_binary"
    train_y = np.fromfile(d / "train_y.bin", dtype=np.uint8)
    test_y = np.fromfile(d / "test_y.bin", dtype=np.uint8)
    assert len(train_y) == 8 and len(test_y) == 4
    assert np.bincount(test_y)[1:].tolist() == [2, 2]  # stratified, 1-based like STL-10
    assert np.bincount(train_y)[1:].tolist() == [4, 4]
    # The pool is the whole train split with labels discarded (brief §4.2); the labeled
    # subset is part of it, as in STL-10 practice.
    assert data.n_unlabeled(out) == 16
    assert set(manifest["train_sources"]) <= set(manifest["unlabeled_sources"])
    assert data.class_names(out) == ("A", "B")
    assert data.read_images(d / "train_X.bin").shape == (8, 96, 96, 3)
    # Pool and test are disjoint sources, together all of them.
    srcs = manifest["unlabeled_sources"] + manifest["test_sources"]
    assert len(set(srcs)) == 20 and not set(manifest["unlabeled_sources"]) & set(manifest["test_sources"])
    unl = data.read_images(d / "unlabeled_X.bin")
    i = manifest["unlabeled_sources"].index(manifest["train_sources"][0])
    assert np.array_equal(unl[i], data.read_images(d / "train_X.bin")[0])


def test_upsampling_matches_pil_bicubic_of_the_source(tmp_path):
    src, out = tmp_path / "2750", tmp_path / "eurosat"
    _fake_imagefolder(src, per_class=5)
    manifest = eurosat.materialize(src, out, test_frac=0.2, seed=0)
    first = src / manifest["test_sources"][0]
    expect = np.asarray(Image.open(first).convert("RGB").resize((96, 96), Image.BICUBIC))
    got = data.read_images(out / "stl10_binary" / "test_X.bin")[0]
    assert np.array_equal(got, expect)


def test_materialize_is_deterministic(tmp_path):
    src = tmp_path / "2750"
    _fake_imagefolder(src)
    a = eurosat.materialize(src, tmp_path / "a", n_labeled=8)
    b = eurosat.materialize(src, tmp_path / "b", n_labeled=8)
    assert a["train_sources"] == b["train_sources"]
    for f in ("train_X.bin", "test_X.bin", "train_y.bin", "test_y.bin", "unlabeled_X.bin"):
        assert (tmp_path / "a" / "stl10_binary" / f).read_bytes() == (tmp_path / "b" / "stl10_binary" / f).read_bytes()
