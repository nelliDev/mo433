"""PoolLoader: uint8 batches from a pool memmap, checkpointable mid-run."""

import numpy as np
import pytest
import torch

from mo433.sampler import ChunkShuffleSampler, PoolLoader, normalize_gpu


@pytest.fixture
def tiny_pool(tmp_path):
    """[300, 4, 96, 96, 3] where pixel (0,0,0) = row % 256, (0,0,1) = row // 256, (0,0,2) = view."""
    path = tmp_path / "views.npy"
    mm = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=(300, 4, 96, 96, 3))
    rows = np.arange(300)
    mm[:, :, 0, 0, 0] = (rows % 256)[:, None]
    mm[:, :, 0, 0, 1] = (rows // 256)[:, None]
    mm[:, :, 0, 0, 2] = np.arange(4)[None, :]
    mm.flush()
    return path


def _decode(x):
    px = x[..., 0, 0, :].long()
    return px[..., 0] + 256 * px[..., 1], px[..., 2]


@pytest.mark.parametrize("workers", [0, 2])
def test_batches_hold_the_sampled_views(tiny_pool, workers):
    s = ChunkShuffleSampler(n_images=300, k_views=4, views_per_image=2, batch_size=64, chunk_size=16, seed=0)
    loader = PoolLoader(tiny_pool, s, num_workers=workers, pin_memory=False)
    seen = []
    for image_idx, view_idx, x in loader:
        assert x.dtype == torch.uint8 and x.shape == (len(image_idx), 2, 96, 96, 3)
        rows, views = _decode(x)
        assert torch.equal(rows, image_idx[:, None].expand(-1, 2))
        assert torch.equal(views, view_idx)
        seen.append(image_idx)
    assert torch.equal(torch.cat(seen).sort().values, torch.arange(300))
    assert len(seen) == len(loader) == 5


def test_loader_rejects_a_sampler_larger_than_the_pool(tiny_pool):
    s = ChunkShuffleSampler(n_images=301, k_views=4, views_per_image=2)
    with pytest.raises(ValueError):
        PoolLoader(tiny_pool, s)
    s = ChunkShuffleSampler(n_images=300, k_views=5, views_per_image=2)
    with pytest.raises(ValueError):
        PoolLoader(tiny_pool, s)


def test_loader_state_counts_consumed_batches_not_prefetched(tiny_pool):
    def make():
        s = ChunkShuffleSampler(n_images=300, k_views=4, views_per_image=1, batch_size=64, chunk_size=16, seed=5)
        return PoolLoader(tiny_pool, s, num_workers=2, pin_memory=False)

    a = make()
    it = iter(a)
    for _ in range(2):
        next(it)
    state = a.state_dict()
    assert state == {"seed": 5, "epoch": 0, "batch": 2}
    expected = [b[0] for b in it]  # rest of epoch 0
    expected += [b[0] for b in a]  # epoch 1
    assert a.state_dict() == {"seed": 5, "epoch": 2, "batch": 0}

    b = make()
    b.load_state_dict(state)
    got = [x[0] for x in b] + [x[0] for x in b]
    assert len(got) == len(expected)
    assert all(torch.equal(g, e) for g, e in zip(got, expected))


def test_normalize_gpu_matches_imagenet_stats():
    x = torch.randint(0, 256, (2, 96, 96, 3), dtype=torch.uint8)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    y = normalize_gpu(x, dev)
    assert y.shape == (2, 3, 96, 96) and y.dtype == torch.float32
    assert y.is_contiguous(memory_format=torch.channels_last)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    ref = (x.permute(0, 3, 1, 2).float() / 255 - mean) / std
    torch.testing.assert_close(y.cpu(), ref, atol=1e-5, rtol=1e-5)
