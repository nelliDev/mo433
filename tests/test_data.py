"""Data seam: STL-10 loading and the deterministic student eval transform (spec §2, §7.1, §12.1)."""

import pytest
import torch

from mo433.data import load_stl10, normalize


def test_normalize_maps_uint8_nhwc_to_imagenet_normalized_nchw():
    u8 = torch.zeros(2, 96, 96, 3, dtype=torch.uint8)
    u8[1] = 255
    x = normalize(u8)
    assert x.shape == (2, 3, 96, 96) and x.dtype == torch.float32
    # ImageNet mean/std (0.485,0.456,0.406)/(0.229,0.224,0.225)
    assert torch.allclose(x[0, :, 0, 0], torch.tensor([-2.1179, -2.0357, -1.8044]), atol=1e-4)
    assert torch.allclose(x[1, :, 0, 0], torch.tensor([2.2489, 2.4286, 2.6400]), atol=1e-4)


@pytest.mark.invariant
def test_normalize_is_deterministic():
    u8 = torch.randint(0, 256, (4, 96, 96, 3), dtype=torch.uint8)
    assert torch.equal(normalize(u8), normalize(u8.clone()))


@pytest.mark.gpu
def test_normalize_runs_on_the_input_device():
    u8 = torch.randint(0, 256, (4, 96, 96, 3), dtype=torch.uint8, device="cuda")
    x = normalize(u8)
    assert x.is_cuda
    assert torch.allclose(x.cpu(), normalize(u8.cpu()), atol=1e-6)


@pytest.mark.data
@pytest.mark.parametrize("split, n", [("train", 5000), ("test", 8000)])
def test_load_stl10_labeled_splits(split, n):
    images, labels = load_stl10(split)
    assert images.shape == (n, 96, 96, 3) and images.dtype == torch.uint8
    assert labels.shape == (n,) and set(labels.tolist()) == set(range(10))
    # loading twice gives identical data (deterministic eval input)
    again, _ = load_stl10(split)
    assert torch.equal(images[:16], again[:16])


def _write_binary_dataset(root, n_train=6, n_test=4, n_unlabeled=8, classes=("a", "b")):
    """A tiny dataset in the STL-10 binary layout (column-major images, 1-based labels)."""
    import numpy as np

    d = root / "stl10_binary"
    d.mkdir(parents=True)
    rng = np.random.default_rng(0)
    nhwc = {}
    for split, n in (("train", n_train), ("test", n_test), ("unlabeled", n_unlabeled)):
        x = rng.integers(0, 256, (n, 96, 96, 3), dtype=np.uint8)
        nhwc[split] = x
        x.transpose(0, 3, 2, 1).tofile(d / f"{split}_X.bin")
        if split != "unlabeled":
            (np.arange(n) % len(classes) + 1).astype(np.uint8).tofile(d / f"{split}_y.bin")
    (d / "class_names.txt").write_text("\n".join(classes) + "\n")
    return nhwc


def test_binary_layout_reader_counts_and_class_names(tmp_path, monkeypatch):
    from mo433 import data

    monkeypatch.setenv("MO433_DATA", str(tmp_path))
    monkeypatch.delenv("MO433_DATASET", raising=False)
    nhwc = _write_binary_dataset(tmp_path)
    images, labels = load_stl10("test")
    assert torch.equal(images, torch.from_numpy(nhwc["test"]))
    assert labels.tolist() == [0, 1, 0, 1]
    assert data.train_labels().tolist() == [0, 1, 0, 1, 0, 1]
    assert data.n_unlabeled() == 8
    assert data.class_names() == ("a", "b")


@pytest.mark.data
def test_binary_reader_is_pixel_identical_to_torchvision_stl10():
    from torchvision.datasets import STL10

    from mo433 import paths

    images, labels = load_stl10("test")
    tv = STL10(str(paths.data_root()), split="test", download=False)
    assert torch.equal(images, torch.from_numpy(tv.data.transpose(0, 2, 3, 1).copy()))
    assert labels.tolist() == tv.labels.tolist()
