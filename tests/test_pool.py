import json
import os
from pathlib import Path

import numpy as np
import pytest

from mo433 import pool

DATA_ROOT = Path(os.environ.get("MO433_DATA", Path(__file__).resolve().parents[1] / "data"))
pytestmark = [
    pytest.mark.data,
    pytest.mark.skipif(
        not (DATA_ROOT / "stl10_binary" / "unlabeled_X.bin").exists(), reason="STL-10 unlabeled_X.bin not found"
    ),
]


def _build(out, workers=0, n=12, k=3, **kw):
    return pool.build_pool(out, n_images=n, k_views=k, data_root=DATA_ROOT, num_workers=workers,
                           rows_per_batch=4, sync_every_rows=4, log=lambda m: None, **kw)


def test_pool_order_is_the_spec_permutation():
    order = pool.pool_order()
    assert np.array_equal(order, np.random.default_rng(0).permutation(100_000))


def test_unlabeled_images_match_torchvision():
    import torchvision

    tv = torchvision.datasets.STL10(str(DATA_ROOT), split="unlabeled", download=False)
    imgs = pool.unlabeled_images(DATA_ROOT)
    for i in (0, 1, 99_999):
        assert np.array_equal(imgs[i], np.asarray(tv[i][0]))


def test_build_writes_memmap_in_pool_order(tmp_path):
    path = _build(tmp_path / "pool")
    views, meta = pool.open_pool(tmp_path / "pool")
    assert path == tmp_path / "pool" / "views.npy"
    assert views.shape == (12, 3, 96, 96, 3) and views.dtype == np.uint8
    assert meta["n_images"] == 12 and meta["k_views"] == 3
    assert meta["order_sha256"] == pool.order_hash(pool.pool_order()[:12])
    assert not (tmp_path / "pool" / "views.npy.partial").exists()
    # Views are augmentations: varied, distinct from each other.
    assert views.std() > 10
    assert not np.array_equal(views[0, 0], views[0, 1])


def test_row_j_holds_image_order_j(tmp_path, monkeypatch):
    import torchvision.transforms as T

    monkeypatch.setattr(pool, "make_transform", T.ToTensor)  # identity augmentation
    _build(tmp_path / "pool", n=10, k=2)
    views, _ = pool.open_pool(tmp_path / "pool")
    imgs, order = pool.unlabeled_images(DATA_ROOT), pool.pool_order()
    for j in range(10):
        assert np.array_equal(views[j, 0], imgs[order[j]])
        assert np.array_equal(views[j, 1], imgs[order[j]])


def test_build_is_reproducible_across_workers_and_resume(tmp_path):
    _build(tmp_path / "a", workers=0)
    _build(tmp_path / "b", workers=2)
    a, _ = pool.open_pool(tmp_path / "a")
    b, _ = pool.open_pool(tmp_path / "b")
    assert np.array_equal(a, b)

    # Simulate a crash after 4 rows: partial file + progress, then resume.
    c = tmp_path / "c"
    _build(c, workers=0)
    meta = json.loads((c / "meta.json").read_text())
    meta.pop("build_seconds")
    os.replace(c / "views.npy", c / "views.npy.partial")
    (c / "meta.json").unlink()
    arr = np.load(c / "views.npy.partial", mmap_mode="r+")
    arr[4:] = 0
    arr.flush()
    del arr
    (c / "progress.json").write_text(json.dumps({"meta": meta, "rows_done": 4}))
    _build(c, workers=0)
    resumed, _ = pool.open_pool(c)
    assert np.array_equal(resumed, a)


def test_different_seed_gives_different_views(tmp_path):
    _build(tmp_path / "a", n=2, k=2)
    _build(tmp_path / "b", n=2, k=2, seed=1)
    assert not np.array_equal(pool.open_pool(tmp_path / "a")[0], pool.open_pool(tmp_path / "b")[0])


def test_complete_pool_is_not_rebuilt(tmp_path):
    _build(tmp_path / "p", n=2, k=2)
    mtime = (tmp_path / "p" / "views.npy").stat().st_mtime_ns
    _build(tmp_path / "p", n=2, k=2)
    assert (tmp_path / "p" / "views.npy").stat().st_mtime_ns == mtime
    with pytest.raises(FileExistsError):
        _build(tmp_path / "p", n=2, k=3)
