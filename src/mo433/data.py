"""STL-10 access and the student input normalization (spec §2, §7.1, §12.1).

Images travel as uint8 NHWC ``[N, 96, 96, 3]`` tensors (the view pool's layout) and are
turned into float input only on the device that runs the model, by ``normalize``. The same
helper is the deterministic eval transform for students (the 96 px image as-is) and the
per-step normalization of pool views in training.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from mo433 import paths

N_LABELED_TRAIN = 5_000
N_UNLABELED = 100_000
N_CLASSES = 10

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def train_labels(data_root: Path | None = None) -> np.ndarray:
    """Labels of the 5k labeled train split, 0-based, identical to
    ``torchvision.datasets.STL10(data_root, split="train").labels`` but without reading images."""
    root = Path(data_root) if data_root is not None else paths.data_root()
    raw = np.fromfile(root / "stl10_binary" / "train_y.bin", dtype=np.uint8)
    if raw.shape != (N_LABELED_TRAIN,):
        raise ValueError(f"unexpected train_y.bin shape {raw.shape}")
    return raw.astype(np.int64) - 1


def normalize(u8: torch.Tensor) -> torch.Tensor:
    """uint8 NHWC ``[B, H, W, 3]`` -> float32 ``[B, 3, H, W]`` ImageNet-normalized, on ``u8``'s device.

    The result is NCHW-shaped with channels_last memory (a free permute of the NHWC input),
    which is what the students' bf16/channels_last steps want.
    """
    if u8.dtype != torch.uint8 or u8.ndim != 4 or u8.shape[-1] != 3:
        raise ValueError(f"expected uint8 [B, H, W, 3], got {u8.dtype} {tuple(u8.shape)}")
    mean = torch.tensor(IMAGENET_MEAN, device=u8.device) * 255
    std = torch.tensor(IMAGENET_STD, device=u8.device) * 255
    return ((u8.float() - mean) / std).permute(0, 3, 1, 2)


def load_stl10(split: str) -> tuple[torch.Tensor, torch.Tensor]:
    """A labeled STL-10 split as ``(images uint8 [N, 96, 96, 3], labels int64 [N])`` on CPU.

    ``split`` is ``"train"`` (5k) or ``"test"`` (8k). Reads ``$MO433_DATA/stl10_binary``.
    """
    from torchvision.datasets import STL10

    if split not in ("train", "test"):
        raise ValueError(f"unknown labeled split {split!r}; expected 'train' or 'test'")
    ds = STL10(str(paths.data_root()), split=split, download=False)
    images = torch.from_numpy(np.ascontiguousarray(ds.data.transpose(0, 2, 3, 1)))
    labels = torch.from_numpy(ds.labels.astype(np.int64))
    return images, labels
