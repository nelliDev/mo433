"""The validation-split k-NN scorer (spec §6 intro, §2): the single score behind every
tuning trial, selection choice and collapse check.

k-NN with k = 20, cosine similarity, uniform majority vote, on L2-normalized encoder
features; queries = the validation split (500 images), bank = the other 90% (4500).
Implemented in torch so training loops can score on the GPU every epoch.

Entry points:

- ``val_knn(features, labels)``: pure function on the [5000, d] features of the labeled
  train split (numpy or torch, any device).
- ``val_knn_encoder(encoder, images, labels)``: convenience for training loops. Runs the
  encoder on the uint8 labeled-train images under ``no_grad`` in ``eval()``, restores its
  previous train/eval mode, and never touches its parameters.
- ``knn_accuracy(bank_x, bank_y, query_x, query_y, k)``: the underlying primitive.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

K_VAL = 20
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _as_tensor(a, device=None, dtype=None) -> torch.Tensor:
    t = a if isinstance(a, torch.Tensor) else torch.from_numpy(np.asarray(a))
    return t.to(device=device if device is not None else t.device, dtype=dtype)


@torch.no_grad()
def knn_predict(bank_x, bank_y, query_x, k: int = K_VAL, chunk: int = 2048) -> torch.Tensor:
    """Cosine k-NN labels for ``query_x`` (uniform vote; ties go to the lowest class id)."""
    bank = F.normalize(_as_tensor(bank_x, dtype=torch.float32), dim=1)
    device = bank.device
    query = F.normalize(_as_tensor(query_x, device=device, dtype=torch.float32), dim=1)
    labels = _as_tensor(bank_y, device=device, dtype=torch.long)
    n_classes = int(labels.max()) + 1
    k = min(k, len(bank))
    out = []
    for start in range(0, len(query), chunk):
        sim = query[start : start + chunk] @ bank.T
        idx = sim.topk(k, dim=1).indices
        votes = F.one_hot(labels[idx], n_classes).sum(dim=1)
        out.append(votes.argmax(dim=1))
    return torch.cat(out) if out else torch.empty(0, dtype=torch.long, device=device)


def knn_accuracy(bank_x, bank_y, query_x, query_y, k: int = K_VAL) -> float:
    pred = knn_predict(bank_x, bank_y, query_x, k)
    truth = _as_tensor(query_y, device=pred.device, dtype=torch.long)
    return float((pred == truth).float().mean())


def val_knn(features, labels, splits_dir: Path | None = None, k: int = K_VAL) -> float:
    """Validation-split k-NN accuracy (0-1) of the labeled-train ``features`` [5000, d]."""
    from mo433 import splits

    val, bank = splits.validation_split(splits_dir)
    x = _as_tensor(features)
    y = _as_tensor(labels, device=x.device)
    val_t = torch.from_numpy(val).to(x.device)
    bank_t = torch.from_numpy(bank).to(x.device)
    return knn_accuracy(x[bank_t], y[bank_t], x[val_t], y[val_t], k)


def normalize_u8(u8: torch.Tensor) -> torch.Tensor:
    """uint8 NHWC [B, 96, 96, 3] -> ImageNet-normalized float32 NCHW (spec §12.1)."""
    if u8.dtype != torch.uint8 or u8.ndim != 4 or u8.shape[-1] != 3:
        raise ValueError(f"expected uint8 [B, H, W, 3], got {u8.dtype} {tuple(u8.shape)}")
    mean = torch.tensor(IMAGENET_MEAN, device=u8.device) * 255
    std = torch.tensor(IMAGENET_STD, device=u8.device) * 255
    return ((u8.float() - mean) / std).permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)


@torch.no_grad()
def encode(encoder: torch.nn.Module, images: torch.Tensor, batch_size: int = 500, device=None) -> torch.Tensor:
    """Pooled features [N, d] of uint8 NHWC ``images``, in ``eval()``; the encoder's mode is restored."""
    if device is None:
        device = next(encoder.parameters()).device
    was_training = encoder.training
    encoder.eval()
    try:
        feats = []
        for start in range(0, len(images), batch_size):
            batch = normalize_u8(images[start : start + batch_size].to(device, non_blocking=True))
            feats.append(encoder(batch).float().flatten(1))
        return torch.cat(feats)
    finally:
        encoder.train(was_training)


def val_knn_encoder(
    encoder: torch.nn.Module,
    images: torch.Tensor,
    labels,
    splits_dir: Path | None = None,
    k: int = K_VAL,
    batch_size: int = 500,
) -> float:
    """``val_knn`` of ``encoder`` on the labeled-train uint8 images [5000, 96, 96, 3].

    Pass the images already on the GPU to avoid a host copy per epoch.
    """
    return val_knn(encode(encoder, images, batch_size), labels, splits_dir, k)
