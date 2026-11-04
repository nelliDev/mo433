"""View pool (spec §3): K pre-generated augmented views per unlabeled image.

File format, inside the pool directory (``$MO433_POOL``, default ``$MO433_ROOT/pool``):

- ``views.npy``: a standard ``.npy`` file holding uint8 ``[N, K, 96, 96, 3]`` (HWC views).
  Row ``j`` holds the views of unlabeled image ``order[j]``, where ``order`` is the committed
  pool order (``np.random.default_rng(0).permutation(100_000)``, spec §12). Open it with
  :func:`open_pool` (or ``np.load(path, mmap_mode="r")``).
- ``meta.json``: N, K, shape, dtype, seed, the sha256 of ``order[:N]``, the transform
  arguments and the lightly version. Written last, so its presence means the pool is complete.

While building, the data goes to ``views.npy.partial`` with progress in ``progress.json``; an
interrupted build resumes from the last synced row. Every image's views are generated from an
RNG seeded by ``(seed, row)``, so the pool is bit-reproducible regardless of worker count,
scheduling or resumes.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

N_UNLABELED = 100_000
IMAGE_SHAPE = (96, 96, 3)
VIEW_BYTES = 96 * 96 * 3
TRANSFORM_KWARGS = dict(input_size=96, min_scale=0.2, gaussian_blur=0.5, sigmas=(0.1, 1.0), normalize=None)
POOL_ORDER_SEED = 0

VIEWS_FILE = "views.npy"
META_FILE = "meta.json"
PARTIAL_FILE = "views.npy.partial"
PROGRESS_FILE = "progress.json"


# ---------------------------------------------------------------- inputs


def pool_order() -> np.ndarray:
    """The committed pool order: row j of the pool is unlabeled image ``pool_order()[j]``.

    Read from ``splits/`` when the splits module provides it; otherwise recomputed from the
    spec §12 seed (the two are asserted equal when both exist)."""
    order = np.random.default_rng(POOL_ORDER_SEED).permutation(N_UNLABELED)
    committed = _committed_pool_order()
    if committed is not None and not np.array_equal(committed, order):
        raise ValueError("splits/ pool order differs from default_rng(0).permutation(100_000)")
    return order


def _committed_pool_order() -> np.ndarray | None:
    try:
        from mo433 import splits  # type: ignore[attr-defined]
    except ImportError:
        return None
    loader = getattr(splits, "pool_order", None) or getattr(splits, "load_pool_order", None)
    if loader is None:
        return None
    try:
        return np.asarray(loader())
    except FileNotFoundError:
        return None


def order_hash(order: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(order, dtype="<i8").tobytes()).hexdigest()


def unlabeled_images(data_root: Path) -> np.ndarray:
    """All 100k unlabeled STL-10 images as a read-only uint8 ``[N, 96, 96, 3]`` (HWC) view of
    ``unlabeled_X.bin``; pixel-identical to ``torchvision.datasets.STL10(split="unlabeled")``."""
    raw = np.memmap(Path(data_root) / "stl10_binary" / "unlabeled_X.bin", dtype=np.uint8, mode="r")
    # The binary is column-major per image: [N, C, W, H] -> [N, H, W, C].
    return raw.reshape(-1, 3, 96, 96).transpose(0, 3, 2, 1)


def make_transform():
    from lightly.transforms import VICRegViewTransform

    return VICRegViewTransform(**TRANSFORM_KWARGS)


# ---------------------------------------------------------------- build


class _ViewDataset:
    """Item ``j`` -> uint8 ``[K, 96, 96, 3]``: the K views of image ``order[j]``."""

    def __init__(self, data_root: Path, order: np.ndarray, k_views: int, seed: int):
        self.data_root, self.order, self.k, self.seed = Path(data_root), order, k_views, seed
        self.images = None
        self.transform = None

    def __len__(self) -> int:
        return len(self.order)

    def __getitem__(self, j: int):
        import torch
        from PIL import Image

        if self.images is None:  # lazily, per worker
            torch.set_num_threads(1)
            self.images = unlabeled_images(self.data_root)
            self.transform = make_transform()
        # lightly draws from torch, numpy's global RNG and python's random: seed all three.
        s = np.random.SeedSequence([self.seed, int(j)]).generate_state(2)
        torch.manual_seed(int(s[0]) << 32 | int(s[1]))
        np.random.seed(s)
        random.seed(int(s[0]))
        img = Image.fromarray(np.ascontiguousarray(self.images[self.order[j]]))
        out = np.empty((self.k, *IMAGE_SHAPE), dtype=np.uint8)
        for v in range(self.k):
            x = self.transform(img)  # float CHW in [0, 1], exactly k/255
            out[v] = x.mul(255).round_().to(torch.uint8).permute(1, 2, 0).numpy()
        return out


def _collate(items):
    return np.stack(items)


def set_nocow(directory: Path, log: Callable[[str], None] = print) -> None:
    """Best-effort ``chattr +C`` (btrfs no-copy-on-write; files created later inherit it)."""
    try:
        subprocess.run(["chattr", "+C", str(directory)], check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError) as e:
        log(f"[pool] chattr +C {directory} failed (not btrfs?): {getattr(e, 'stderr', e)!r}")


def build_pool(
    out_dir: Path,
    *,
    n_images: int,
    k_views: int,
    data_root: Path,
    seed: int = 0,
    order: np.ndarray | None = None,
    num_workers: int = 14,
    rows_per_batch: int = 32,
    sync_every_rows: int = 2048,
    log: Callable[[str], None] = print,
) -> Path:
    """Write ``out_dir/views.npy`` (+ ``meta.json``) and return its path. Resumes a matching
    interrupted build; call :func:`remove_pool` first to rebuild from scratch."""
    from torch.utils.data import DataLoader

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_nocow(out_dir, log)
    order = pool_order() if order is None else np.asarray(order)
    if not 0 < n_images <= len(order):
        raise ValueError(f"n_images={n_images} outside (0, {len(order)}]")
    rows = order[:n_images]
    shape = (n_images, k_views, *IMAGE_SHAPE)
    meta = {
        "n_images": n_images,
        "k_views": k_views,
        "shape": list(shape),
        "dtype": "uint8",
        "layout": "row j = unlabeled image order[j]; views HWC",
        "seed": seed,
        "order_sha256": order_hash(rows),
        "transform": "lightly.transforms.VICRegViewTransform",
        "transform_kwargs": {k: (list(v) if isinstance(v, tuple) else v) for k, v in TRANSFORM_KWARGS.items()},
        "lightly_version": _lightly_version(),
    }
    final, partial, progress_path = out_dir / VIEWS_FILE, out_dir / PARTIAL_FILE, out_dir / PROGRESS_FILE
    if final.exists() and (out_dir / META_FILE).exists():
        existing = _read_json(out_dir / META_FILE)
        existing.pop("build_seconds", None)
        if existing == meta:
            log(f"[pool] {final} already complete")
            return final
        raise FileExistsError(f"{final} exists with different metadata; remove it (--force) to rebuild")

    start = 0
    progress = _read_json(progress_path) if progress_path.exists() else None
    if partial.exists() and progress and progress.get("meta") == meta:
        start = int(progress["rows_done"])
        log(f"[pool] resuming {partial} at row {start}/{n_images}")
    else:
        for p in (partial, progress_path):
            p.unlink(missing_ok=True)
        np.lib.format.open_memmap(partial, mode="w+", dtype=np.uint8, shape=shape)._mmap.close()
    offset = _npy_data_offset(partial)
    row_bytes = k_views * VIEW_BYTES

    loader = DataLoader(
        _ViewDataset(data_root, rows, k_views, seed),
        batch_size=rows_per_batch,
        sampler=range(start, n_images),
        num_workers=num_workers,
        collate_fn=_collate,
        prefetch_factor=4 if num_workers else None,
        multiprocessing_context="fork" if num_workers else None,
    )
    t0, done, last_sync, last_log = time.time(), start, start, time.time()
    fd = os.open(partial, os.O_WRONLY)
    try:
        for batch in loader:
            os.pwrite(fd, memoryview(batch).cast("B"), offset + done * row_bytes)
            done += len(batch)
            if done - last_sync >= sync_every_rows or done == n_images:
                os.fsync(fd)
                _write_json_atomic(progress_path, {"meta": meta, "rows_done": done})
                last_sync = done
            if time.time() - last_log > 30 or done == n_images:
                rate = (done - start) / max(time.time() - t0, 1e-9)
                eta = (n_images - done) / max(rate, 1e-9)
                log(f"[pool] {done}/{n_images} images ({rate * k_views:.0f} views/s, eta {eta / 60:.1f} min)")
                last_log = time.time()
    finally:
        os.close(fd)
    if done != n_images:
        raise RuntimeError(f"pool build stopped at {done}/{n_images}")
    os.replace(partial, final)
    _write_json_atomic(out_dir / META_FILE, {**meta, "build_seconds": round(time.time() - t0, 1)})
    progress_path.unlink(missing_ok=True)
    log(f"[pool] wrote {final} {shape} in {time.time() - t0:.0f}s")
    return final


def run(profile: str = "full", force: bool = False, num_workers: int | None = None) -> Path:
    """`mo433 pool`: build the profile's pool (N, K from the config) under ``paths.pool_dir``.

    ``num_workers`` defaults to ``$MO433_NUM_WORKERS`` or 14."""
    import os

    from mo433 import config, paths, stages

    if num_workers is None:
        num_workers = int(os.environ.get("MO433_NUM_WORKERS", "14"))
    cfg = config.load_config(profile=profile)
    out_dir = paths.pool_dir(profile)

    def build() -> None:
        if force:
            remove_pool(out_dir)
        build_pool(out_dir, n_images=cfg.n_images, k_views=cfg.k_views, data_root=paths.data_root(),
                   num_workers=num_workers)

    stages.run_stage("pool", build, profile=profile, force=force)
    return out_dir / VIEWS_FILE


def remove_pool(out_dir: Path) -> None:
    for name in (VIEWS_FILE, META_FILE, PARTIAL_FILE, PROGRESS_FILE):
        (Path(out_dir) / name).unlink(missing_ok=True)


def open_pool(pool_dir: Path) -> tuple[np.ndarray, dict]:
    """The finished pool as a read-only memmap ``[N, K, 96, 96, 3]`` plus its metadata."""
    pool_dir = Path(pool_dir)
    meta_path = pool_dir / META_FILE
    if not meta_path.exists():
        raise FileNotFoundError(f"no finished pool in {pool_dir} (run `mo433 pool`)")
    meta = _read_json(meta_path)
    views = np.load(pool_dir / VIEWS_FILE, mmap_mode="r")
    if list(views.shape) != meta["shape"] or views.dtype != np.uint8:
        raise ValueError(f"{pool_dir}: views {views.shape}/{views.dtype} do not match meta {meta['shape']}")
    return views, meta


# ---------------------------------------------------------------- helpers


def _npy_data_offset(path: Path) -> int:
    with open(path, "rb") as f:
        version = np.lib.format.read_magic(f)
        read_header = {(1, 0): np.lib.format.read_array_header_1_0, (2, 0): np.lib.format.read_array_header_2_0}
        read_header[version](f)
        return f.tell()


def _lightly_version() -> str | None:
    try:
        from importlib.metadata import version

        return version("lightly")
    except Exception:
        return None


def _read_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _write_json_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
