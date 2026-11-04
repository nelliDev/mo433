"""Chunk-shuffle sampler and loader over the view pool (spec §3).

Each epoch the pool rows ``[0, n_images)`` are cut into contiguous chunks of 64 images and the
order of the chunks is shuffled, so a batch of 256 is 4 chunks (4 sequential disk reads instead
of 256 random ones). For every image, ``views_per_image`` distinct view indices out of K are
drawn per epoch: 2 for row B, 1 for row C. The 50% ablation passes ``n_images=50_000``.

Everything an epoch does is a pure function of ``(seed, epoch)``, so the RNG state that goes in
a checkpoint is just ``{"seed", "epoch", "batch"}`` (:meth:`PoolLoader.state_dict`).
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import numpy as np

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ChunkShuffleSampler:
    """Yields ``(image_idx [B], view_idx [B, views_per_image])`` int64 arrays, one epoch per
    ``iter()``. Iterating to the end advances :attr:`epoch`; :attr:`batch` is the number of
    batches of the current epoch to skip (set by :meth:`load_state_dict` for a mid-epoch resume)."""

    def __init__(
        self,
        n_images: int,
        k_views: int,
        views_per_image: int,
        *,
        batch_size: int = 256,
        chunk_size: int = 64,
        seed: int = 0,
        drop_last: bool = False,
    ):
        if not 1 <= views_per_image <= k_views:
            raise ValueError(f"views_per_image={views_per_image} must be in [1, k_views={k_views}]")
        if n_images < 1 or batch_size < 1 or chunk_size < 1:
            raise ValueError("n_images, batch_size and chunk_size must be positive")
        self.n_images, self.k_views, self.views_per_image = n_images, k_views, views_per_image
        self.batch_size, self.chunk_size, self.seed, self.drop_last = batch_size, chunk_size, seed, drop_last
        self.epoch = 0
        self.batch = 0

    def __len__(self) -> int:
        full, rest = divmod(self.n_images, self.batch_size)
        return full + (rest > 0 and not self.drop_last)

    def epoch_plan(self, epoch: int) -> tuple[np.ndarray, np.ndarray]:
        """All of ``epoch``'s ``(image_idx [M], view_idx [M, v])``, in visiting order."""
        rng = np.random.default_rng([self.seed, epoch])
        starts = np.arange(0, self.n_images, self.chunk_size)
        image_idx = np.concatenate(
            [np.arange(s, min(s + self.chunk_size, self.n_images)) for s in rng.permutation(starts)]
        )
        # v distinct views per image: the first v columns of a random permutation of K per row.
        keys = rng.random((self.n_images, self.k_views))
        view_idx = np.argpartition(keys, self.views_per_image - 1, axis=1)[:, : self.views_per_image]
        return image_idx, view_idx[image_idx].astype(np.int64)

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        image_idx, view_idx = self.epoch_plan(self.epoch)
        for b in range(self.batch, len(self)):
            sl = slice(b * self.batch_size, (b + 1) * self.batch_size)
            yield image_idx[sl], view_idx[sl]
        self.epoch += 1
        self.batch = 0

    def state_dict(self) -> dict:
        return {"seed": self.seed, "epoch": self.epoch, "batch": self.batch}

    def load_state_dict(self, state: dict) -> None:
        self.seed, self.epoch, self.batch = int(state["seed"]), int(state["epoch"]), int(state["batch"])


class _BatchReader:
    """Map-style dataset whose "index" is a whole batch spec ``(image_idx, view_idx)``."""

    def __init__(self, path: Path):
        self.path = Path(path)
        mm = np.load(self.path, mmap_mode="r")
        self.k, self.view_shape, self.data_offset = mm.shape[1], mm.shape[2:], mm.offset
        self.view_bytes = int(np.prod(self.view_shape))
        del mm
        self.fd = None

    def __getitem__(self, spec):
        import torch

        if self.fd is None:  # opened lazily, once per worker
            self.fd = os.open(self.path, os.O_RDONLY)
            # The pool (55 GB) exceeds RAM, and a view is 27 KB inside a 553 KB row: kernel
            # readahead would read most of each row to use 1/10 of it. Read exactly each view.
            os.posix_fadvise(self.fd, 0, 0, os.POSIX_FADV_RANDOM)
        image_idx, view_idx = spec
        x = np.empty((*view_idx.shape, *self.view_shape), dtype=np.uint8)  # [B, v, 96, 96, 3]
        flat = x.reshape(-1, self.view_bytes)
        offsets = self.data_offset + (image_idx[:, None] * self.k + view_idx).reshape(-1) * self.view_bytes
        for n in np.argsort(offsets, kind="stable"):  # ascending file order
            if os.preadv(self.fd, [flat[n]], int(offsets[n])) != self.view_bytes:
                raise OSError(f"short read from {self.path}")
        return torch.from_numpy(image_idx), torch.from_numpy(view_idx), torch.from_numpy(x)


class PoolLoader:
    """Iterate one epoch of ``(image_idx [B], view_idx [B, v], views uint8 [B, v, 96, 96, 3])``
    from the pool file at ``path`` (``views.npy``), in ``sampler`` order.

    Each ``iter()`` is one epoch. :meth:`state_dict` counts batches the caller actually
    consumed (not ones prefetched by workers), so saving it in a checkpoint and calling
    :meth:`load_state_dict` on a fresh loader replays exactly the remaining batches."""

    def __init__(
        self,
        path: Path,
        sampler: ChunkShuffleSampler,
        *,
        num_workers: int = 8,
        pin_memory: bool = True,
        prefetch_factor: int = 4,
    ):
        from torch.utils.data import DataLoader

        shape = np.load(path, mmap_mode="r").shape
        if sampler.n_images > shape[0] or sampler.k_views != shape[1]:
            raise ValueError(f"sampler (N={sampler.n_images}, K={sampler.k_views}) does not fit pool {shape}")
        self.sampler = sampler
        self.epoch, self.batch = sampler.epoch, sampler.batch
        self._loader = DataLoader(
            _BatchReader(path),
            sampler=sampler,
            batch_size=None,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            prefetch_factor=prefetch_factor if num_workers else None,
        )

    def __len__(self) -> int:
        return len(self.sampler)

    def __iter__(self):
        self.sampler.epoch, self.sampler.batch = self.epoch, self.batch
        for item in self._loader:
            self.batch += 1
            yield item
        self.epoch, self.batch = self.epoch + 1, 0

    def state_dict(self) -> dict:
        return {"seed": self.sampler.seed, "epoch": self.epoch, "batch": self.batch}

    def load_state_dict(self, state: dict) -> None:
        self.sampler.load_state_dict(state)
        self.epoch, self.batch = self.sampler.epoch, self.sampler.batch


def normalize_gpu(x, device, dtype=None):
    """uint8 NHWC (CPU, ideally pinned) -> float NCHW on ``device``, ImageNet-normalized
    (spec §12.1). The result is channels_last in memory (a free permute of NHWC)."""
    import torch

    x = x.to(device, non_blocking=True).permute(0, 3, 1, 2).float()
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1) * 255
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1) * 255
    x = x.sub_(mean).div_(std)
    return x if dtype is None else x.to(dtype)
