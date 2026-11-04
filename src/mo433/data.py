"""STL-10 access (spec §2). Images come later (pool / extract); labels are needed by splits."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mo433 import paths

N_LABELED_TRAIN = 5_000
N_UNLABELED = 100_000
N_CLASSES = 10


def train_labels(data_root: Path | None = None) -> np.ndarray:
    """Labels of the 5k labeled train split, 0-based, identical to
    ``torchvision.datasets.STL10(data_root, split="train").labels`` but without reading images."""
    root = Path(data_root) if data_root is not None else paths.data_root()
    raw = np.fromfile(root / "stl10_binary" / "train_y.bin", dtype=np.uint8)
    if raw.shape != (N_LABELED_TRAIN,):
        raise ValueError(f"unexpected train_y.bin shape {raw.shape}")
    return raw.astype(np.int64) - 1
