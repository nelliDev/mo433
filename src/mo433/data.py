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

N_LABELED_TRAIN = 5_000  # STL-10; other datasets: len(train_labels())
N_UNLABELED = 100_000  # STL-10; other datasets: n_unlabeled()
N_CLASSES = 10
IMAGE_BYTES = 96 * 96 * 3
STL10_CLASSES = ("airplane", "bird", "car", "cat", "deer", "dog", "horse", "monkey", "ship", "truck")
# torchvision.datasets.EuroSAT label order (sorted folder names).
EUROSAT_CLASSES = ("AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial", "Pasture",
                   "PermanentCrop", "Residential", "River", "SeaLake")
DEFAULT_CLASSES = {"stl10": STL10_CLASSES, "eurosat": EUROSAT_CLASSES}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _binary_dir(data_root: Path | None = None) -> Path:
    return (Path(data_root) if data_root is not None else paths.data_root()) / "stl10_binary"


def train_labels(data_root: Path | None = None) -> np.ndarray:
    """Labels of the labeled train split, 0-based, identical to
    ``torchvision.datasets.STL10(data_root, split="train").labels`` but without reading images."""
    d = _binary_dir(data_root)
    raw = np.fromfile(d / "train_y.bin", dtype=np.uint8)
    images = d / "train_X.bin"
    if images.exists() and images.stat().st_size != len(raw) * IMAGE_BYTES:
        raise ValueError(f"train_y.bin has {len(raw)} labels but train_X.bin does not hold that many images")
    return raw.astype(np.int64) - 1


def n_unlabeled(data_root: Path | None = None) -> int:
    """Number of images in ``unlabeled_X.bin`` (100k for STL-10; the train split for EuroSAT)."""
    f = _binary_dir(data_root) / "unlabeled_X.bin"
    if not f.exists() and paths.dataset() == "stl10":
        return N_UNLABELED
    size = f.stat().st_size
    if size % IMAGE_BYTES:
        raise ValueError(f"{f}: {size} bytes is not a whole number of 96x96x3 images")
    return size // IMAGE_BYTES


def class_names(data_root: Path | None = None) -> tuple[str, ...]:
    """Class names in label order, from ``class_names.txt`` (shipped with STL-10, written for
    EuroSAT), else the current dataset's known names (so the report builds without the data)."""
    f = _binary_dir(data_root) / "class_names.txt"
    if not f.exists():
        return DEFAULT_CLASSES[paths.dataset()]
    return tuple(line.strip() for line in f.read_text().splitlines() if line.strip())


def read_images(path: Path) -> np.ndarray:
    """A ``*_X.bin`` file as a read-only uint8 ``[N, 96, 96, 3]`` (HWC) memmap view.
    The binary is column-major per image: ``[N, C, W, H]`` -> ``[N, H, W, C]``."""
    raw = np.memmap(path, dtype=np.uint8, mode="r")
    return raw.reshape(-1, 3, 96, 96).transpose(0, 3, 2, 1)


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
    """A labeled split in the STL-10 binary layout as ``(images uint8 [N, 96, 96, 3], labels int64 [N])``
    on CPU, pixel-identical to ``torchvision.datasets.STL10`` (which also MD5-checks the files and
    so cannot read the EuroSAT copy).

    ``split`` is ``"train"`` or ``"test"``. Reads ``<data root>/stl10_binary`` of the current dataset.
    """
    if split not in ("train", "test"):
        raise ValueError(f"unknown labeled split {split!r}; expected 'train' or 'test'")
    d = _binary_dir()
    images = torch.from_numpy(np.ascontiguousarray(read_images(d / f"{split}_X.bin")))
    labels = torch.from_numpy(np.fromfile(d / f"{split}_y.bin", dtype=np.uint8).astype(np.int64) - 1)
    if len(images) != len(labels):
        raise ValueError(f"{split}: {len(images)} images but {len(labels)} labels")
    return images, labels
