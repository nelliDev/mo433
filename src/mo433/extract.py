"""Frozen-feature extraction (spec §7.1): one path for every encoder, never branching on the row.

``extract(encoder_id)`` writes ``$MO433_ROOT/features/<encoder_id>.npz`` with:

==============  ==================  ===========================================================
key             shape / dtype       meaning
==============  ==================  ===========================================================
``train_x``     [5000, d] float32   probed feature of the labeled train split (STL-10 order)
``train_y``     [5000] int64        labels 0-9
``test_x``      [8000, d] float32   probed feature of the test split
``test_y``      [8000] int64        labels 0-9
``encoder_id``  0-d str
``train_x_s3``  [5000, 4w] float32  students only: GAP of stage 3 (multi-scale probe)
``test_x_s3``   [8000, 4w] float32
``train_x_s4``  [5000, 8w] float32  students only: GAP of stage 4 (equal to ``*_x``)
``test_x_s4``   [8000, 8w] float32
==============  ==================  ===========================================================

Rows are in the original STL-10 order, so ``splits/`` indices apply directly. d = 192 (S),
384 (L) or 384 (teacher). Features are raw (not L2-normalized); probes do their own
preprocessing. Extra keys come from ``Encoder.embed``: any key ``k`` it returns is written
as ``train_k`` / ``test_k``.

Encoders are frozen and in ``eval()`` (``Encoder`` enforces both), run under ``no_grad``,
in fp32 with cuDNN deterministic and TF32 off unless the encoder asks for autocast (teacher).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from mo433 import paths
from mo433.data import load_stl10
from mo433.encoders import Encoder, load_encoder

SPLITS = ("train", "test")
SplitLoader = Callable[[str], tuple[torch.Tensor, torch.Tensor]]


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def extract_features(enc: Encoder, images: torch.Tensor, batch_size: int = 500) -> dict[str, np.ndarray]:
    """``enc.embed`` over uint8 NHWC ``images`` (CPU), batched on ``enc``'s device -> float32 arrays."""
    device = next(iter(enc.parameters()), torch.empty(0)).device
    chunks: dict[str, list[torch.Tensor]] = {}
    with torch.backends.cudnn.flags(enabled=True, benchmark=False, deterministic=True, allow_tf32=False):
        for start in range(0, len(images), batch_size):
            u8 = images[start : start + batch_size].to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=enc.autocast_dtype or torch.float32, enabled=enc.autocast_dtype is not None):
                out = enc.embed(enc.preprocess(u8))
            for key, value in out.items():
                chunks.setdefault(key, []).append(value.float().cpu())
    return {key: torch.cat(parts).numpy() for key, parts in chunks.items()}


def features_path(encoder_id: str, profile: str = "full") -> Path:
    return paths.features_dir(profile) / f"{encoder_id}.npz"


def extract(
    encoder_id: str,
    profile: str = "full",
    device: str | None = None,
    batch_size: int = 500,
    load_split: SplitLoader = load_stl10,
) -> Path:
    """Write ``features/<encoder_id>.npz`` (keys in the module docstring) and return its path."""
    enc = load_encoder(encoder_id, profile).to(device or _default_device())
    arrays: dict[str, np.ndarray] = {"encoder_id": np.array(encoder_id)}
    for split in SPLITS:
        images, labels = load_split(split)
        feats = extract_features(enc, images, batch_size)
        if feats["x"].shape != (len(images), enc.dim):
            raise RuntimeError(f"{encoder_id}: feature shape {feats['x'].shape}, expected ({len(images)}, {enc.dim})")
        arrays[f"{split}_y"] = labels.numpy().astype(np.int64)
        arrays.update({f"{split}_{key}": value for key, value in feats.items()})
    out = features_path(encoder_id, profile)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.stem}.tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, out)
    return out
