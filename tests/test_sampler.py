"""Chunk-shuffle sampler (spec §3): what B and C see each epoch."""

import numpy as np

from mo433.sampler import ChunkShuffleSampler


def _epoch(sampler):
    return [(i.copy(), v.copy()) for i, v in sampler]


def test_every_image_is_visited_exactly_once_per_epoch():
    s = ChunkShuffleSampler(n_images=1000, k_views=20, views_per_image=2, seed=0)
    for _ in range(3):
        idx = np.concatenate([i for i, _ in _epoch(s)])
        assert np.array_equal(np.sort(idx), np.arange(1000))


def test_row_b_gets_two_distinct_views_in_range():
    s = ChunkShuffleSampler(n_images=1000, k_views=20, views_per_image=2, seed=1)
    for _, v in _epoch(s):
        assert v.shape[1] == 2
        assert (v[:, 0] != v[:, 1]).all()
        assert v.min() >= 0 and v.max() < 20


def test_row_c_gets_one_view_and_views_change_across_epochs():
    s = ChunkShuffleSampler(n_images=1000, k_views=20, views_per_image=1, seed=0)
    by_epoch = []
    for _ in range(2):
        idx, v = (np.concatenate(x) for x in zip(*_epoch(s)))
        assert v.shape == (1000, 1)
        by_epoch.append(v[np.argsort(idx), 0])
    # 1000 images, K=20: expected agreement between epochs is 1/20.
    assert (by_epoch[0] != by_epoch[1]).mean() > 0.8
    assert set(np.concatenate(by_epoch)) == set(range(20))


def test_ablation_half_never_reads_beyond_50k():
    s = ChunkShuffleSampler(n_images=50_000, k_views=20, views_per_image=2, seed=3)
    seen = np.concatenate([i for i, _ in _epoch(s)])
    assert seen.max() == 49_999
    assert len(seen) == 50_000


def test_batch_of_256_is_four_contiguous_chunks_of_64_in_shuffled_order():
    s = ChunkShuffleSampler(n_images=100_000, k_views=20, views_per_image=2, seed=0)
    batches = [i for i, _ in _epoch(s)]
    assert len(batches) == 391 and len(batches[-1]) == 100_000 - 390 * 256
    chunks = batches[0].reshape(4, 64)
    assert (chunks[:, 0] % 64 == 0).all()
    assert (np.diff(chunks, axis=1) == 1).all()
    starts = np.concatenate([b[::64] for b in batches[:-1]])
    assert not (np.diff(starts) > 0).all()  # chunk order is shuffled


def test_drop_last_drops_the_partial_batch():
    s = ChunkShuffleSampler(n_images=1000, k_views=20, views_per_image=2, drop_last=True)
    sizes = [len(i) for i, _ in _epoch(s)]
    assert sizes == [256, 256, 256] and len(s) == 3


def test_state_roundtrip_reproduces_the_same_order():
    a = ChunkShuffleSampler(n_images=1000, k_views=20, views_per_image=2, seed=7)
    _epoch(a)
    state = a.state_dict()
    expected = _epoch(a)

    b = ChunkShuffleSampler(n_images=1000, k_views=20, views_per_image=2, seed=0)
    b.load_state_dict(state)
    got = _epoch(b)
    assert len(got) == len(expected)
    assert all(np.array_equal(x[0], y[0]) and np.array_equal(x[1], y[1]) for x, y in zip(got, expected))


def test_mid_epoch_resume_continues_where_it_stopped():
    full = _epoch(ChunkShuffleSampler(n_images=1000, k_views=20, views_per_image=1, seed=2))
    b = ChunkShuffleSampler(n_images=1000, k_views=20, views_per_image=1, seed=2)
    b.load_state_dict({"seed": 2, "epoch": 0, "batch": 2})
    rest = _epoch(b)
    assert len(rest) == 2
    assert all(np.array_equal(x[0], y[0]) for x, y in zip(rest, full[2:]))
    assert b.state_dict() == {"seed": 2, "epoch": 1, "batch": 0}


def test_seeds_give_different_orders():
    a = _epoch(ChunkShuffleSampler(n_images=1000, k_views=20, views_per_image=2, seed=0))
    b = _epoch(ChunkShuffleSampler(n_images=1000, k_views=20, views_per_image=2, seed=1))
    assert not np.array_equal(a[0][0], b[0][0])
