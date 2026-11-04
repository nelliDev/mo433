"""Teacher (row D): the frozen DINO ViT-S/16, its per-view feature cache and sanity check
(spec §3, §4.1, §7.3).

Model: ``timm.create_model("vit_small_patch16_224.dino", pretrained=True, num_classes=0)``,
frozen, ``eval()``, fp16 autocast under ``no_grad``. Input: the whole 96 px image or view,
bicubic-resized to 224 (no crop), ImageNet-normalized. Feature: timm's pooled output (the
CLS token after the final LayerNorm), cast ``.float()``.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mo433 import paths
from mo433.data import IMAGENET_MEAN, IMAGENET_STD
from mo433.encoders import ROW_IDS, Encoder, register

MODEL_NAME = "vit_small_patch16_224.dino"
DIM = 384
INPUT_SIZE = 224


def preprocess(u8: torch.Tensor) -> torch.Tensor:
    """uint8 NHWC ``[B, 96, 96, 3]`` -> float32 ``[B, 3, 224, 224]``, on ``u8``'s device.

    Bicubic resize of the [0, 255] image, clamped to [0, 255] like a resize of a uint8 image
    (PIL), then ImageNet normalization.
    """
    if u8.dtype != torch.uint8 or u8.ndim != 4 or u8.shape[-1] != 3:
        raise ValueError(f"expected uint8 [B, H, W, 3], got {u8.dtype} {tuple(u8.shape)}")
    x = u8.permute(0, 3, 1, 2).float()
    x = F.interpolate(x, size=(INPUT_SIZE, INPUT_SIZE), mode="bicubic", align_corners=False).clamp_(0, 255)
    mean = torch.tensor(IMAGENET_MEAN, device=u8.device).view(1, 3, 1, 1) * 255
    std = torch.tensor(IMAGENET_STD, device=u8.device).view(1, 3, 1, 1) * 255
    return (x - mean) / std


# ---------------------------------------------------------------- row D encoder


def build_teacher() -> torch.nn.Module:
    """The frozen DINO ViT-S/16 (weights from the HF cache); ``forward`` -> pooled CLS [B, 384]."""
    import timm

    return timm.create_model(MODEL_NAME, pretrained=True, num_classes=0).eval().requires_grad_(False)


@register(r"D")
def _row_d(m, profile: str) -> Encoder:
    return teacher_encoder()


def teacher_encoder() -> Encoder:
    return Encoder("D", build_teacher(), dim=DIM, preprocess=preprocess, autocast_dtype=torch.float16)


ROW_IDS["D"] = lambda: ["D"]


# ---------------------------------------------------------------- per-view feature cache
#
# ``$MO433_ROOT/teacher/cache.npy``: a standard ``.npy`` holding float16 ``[N, K, 384]``.
# ``cache[j, v]`` is the teacher feature (raw, not L2-normalized) of pool view ``views[j, v]``,
# i.e. view ``v`` of unlabeled image ``pool_order()[j]``. Open it with :func:`open_cache`
# (a read-only memmap). ``cache.json`` next to it records N, K, the pool's metadata, and
# the wall and GPU time of the build (charged to row C, spec §3, §7.4).

CACHE_FILE = "cache.npy"
CACHE_META_FILE = "cache.json"
PARTIAL_SUFFIX = ".partial"


def cache_path(profile: str = "full") -> Path:
    return paths.teacher_dir(profile) / CACHE_FILE


def open_cache(profile: str = "full") -> np.ndarray:
    """The finished cache as a read-only float16 memmap ``[N, K, 384]``."""
    path = cache_path(profile)
    if not path.exists():
        raise FileNotFoundError(f"no teacher cache at {path} (run `mo433 teacher`)")
    return np.load(path, mmap_mode="r")


def _chunks(views: np.ndarray, rows_per_chunk: int, n_buffers: int) -> Iterator[tuple[int, torch.Tensor]]:
    """Yield ``(row0, pinned uint8 [r, K, 96, 96, 3])`` for consecutive row chunks of ``views``.

    A reader thread copies the memmap (sequential I/O) into a ring of pinned buffers, so disk
    reads overlap the GPU. The consumer must be done with a buffer (its H2D copy finished)
    before it takes the next item: the queue holds at most ``n_buffers - 2`` filled buffers
    while the reader fills another, so the one handed out is never overwritten early.
    """
    n = len(views)
    pin = torch.cuda.is_available()
    bufs = [torch.empty((rows_per_chunk, *views.shape[1:]), dtype=torch.uint8, pin_memory=pin) for _ in range(n_buffers)]
    q: queue.Queue = queue.Queue(maxsize=n_buffers - 2)
    error: list[BaseException] = []

    def reader() -> None:
        try:
            for i, row0 in enumerate(range(0, n, rows_per_chunk)):
                r = min(rows_per_chunk, n - row0)
                buf = bufs[i % n_buffers]
                np.copyto(buf.numpy()[:r], views[row0 : row0 + r])
                q.put((row0, buf[:r]))
        except BaseException as e:  # surfaced in the consumer
            error.append(e)
        finally:
            q.put(None)

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    while (item := q.get()) is not None:
        yield item
    thread.join()
    if error:
        raise error[0]


@torch.no_grad()
def build_cache(
    views: np.ndarray,
    out_path: Path,
    encoder: Encoder,
    *,
    batch_size: int = 256,
    rows_per_chunk: int = 64,
    log: Callable[[str], None] = print,
) -> dict:
    """Write the teacher feature of every ``views[j, v]`` to ``out_path`` (float16 ``[N, K, dim]``
    ``.npy``) atomically (``.partial`` + rename) and return the timing record.

    ``gpu_seconds`` sums CUDA-event time of preprocess + forward per batch; ``wall_seconds``
    is the whole build including I/O.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    device = next(encoder.parameters()).device
    n, k = views.shape[:2]
    partial = out_path.with_name(out_path.name + PARTIAL_SUFFIX)
    out = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float16, shape=(n, k, encoder.dim))
    cuda = device.type == "cuda"
    copy_stream = torch.cuda.Stream(device) if cuda else None
    events = []
    start = last_log = time.time()
    for row0, chunk in _chunks(views, rows_per_chunk, n_buffers=4):
        r = len(chunk)
        flat = chunk.view(r * k, *chunk.shape[2:])
        if cuda:
            with torch.cuda.stream(copy_stream):
                gpu_u8 = flat.to(device, non_blocking=True)
            torch.cuda.current_stream(device).wait_stream(copy_stream)
            gpu_u8.record_stream(torch.cuda.current_stream(device))
        else:
            gpu_u8 = flat.to(device)
        feats = []
        for b in range(0, len(gpu_u8), batch_size):
            if cuda:
                ev = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                ev[0].record()
            with torch.autocast(device.type, dtype=encoder.autocast_dtype or torch.float32, enabled=encoder.autocast_dtype is not None):
                f = encoder(encoder.preprocess(gpu_u8[b : b + batch_size]))
            feats.append(f.float().half())
            if cuda:
                ev[1].record()
                events.append(ev)
        # Blocking D2H; it also means this chunk's H2D copy is done before the next item.
        out[row0 : row0 + r] = torch.cat(feats).view(r, k, encoder.dim).cpu().numpy()
        done = row0 + r
        if time.time() - last_log > 60 or done == n:
            last_log = time.time()
            rate = done * k / (last_log - start)
            log(f"[teacher] {done}/{n} rows, {rate:.0f} views/s, eta {(n - done) * k / rate / 60:.1f} min")
    out.flush()
    del out
    wall = time.time() - start
    if cuda:
        torch.cuda.synchronize(device)
    gpu = sum(a.elapsed_time(b) for a, b in events) / 1000
    os.replace(partial, out_path)
    return {
        "n_images": int(n),
        "k_views": int(k),
        "dim": int(encoder.dim),
        "dtype": "float16",
        "batch_size": batch_size,
        "wall_seconds": round(wall, 2),
        "gpu_seconds": round(gpu, 2),
        "views_per_second": round(n * k / wall, 1),
        "device": torch.cuda.get_device_name(device) if cuda else str(device),
    }


def _open_pool(profile: str) -> tuple[np.ndarray, dict]:
    """The finished view pool and its metadata (``mo433.pool.open_pool`` once #17 is merged)."""
    from mo433 import pool

    if hasattr(pool, "open_pool"):
        return pool.open_pool(paths.pool_dir(profile))
    pool_dir = paths.pool_dir(profile)
    meta_path = pool_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"no finished pool in {pool_dir} (run `mo433 pool`)")
    return np.load(pool_dir / "views.npy", mmap_mode="r"), json.loads(meta_path.read_text())


def build_teacher_cache(profile: str = "full", device: str | None = None, log: Callable[[str], None] = print) -> Path:
    """Stage body of ``mo433 teacher``: cache every pool view, write ``cache.json``."""
    from mo433 import stages

    views, pool_meta = _open_pool(profile)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    enc = teacher_encoder().to(device)
    out = cache_path(profile)
    log(f"[teacher] caching {views.shape[0]}x{views.shape[1]} views -> {out}")
    record = build_cache(views, out, enc, log=log)
    record.update(
        gpu_hours=round(record["gpu_seconds"] / 3600, 4),
        wall_hours=round(record["wall_seconds"] / 3600, 4),
        pool=pool_meta,
        model=MODEL_NAME,
        finished=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **stages.git_state(),
    )
    stages._write_atomic(out.with_name(CACHE_META_FILE), record)
    return out
