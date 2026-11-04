"""Row C: distillation from the teacher-feature cache (spec §5.3).

Only the objective lives here; the loop, data, optimizer, checkpointing and logging are the
shared ones in :mod:`mo433.train`. ``mo433 train --row C`` ends up in ``train.train_run``,
which imports this module to register the "C" objective.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

TEACHER_DIM = 384


class DistillObjective:
    """``1 − cosine(projector(student(v)), teacher[i, v])`` averaged over the batch, in fp32.

    ``teacher``: the cache as a tensor ``[N, K, 384]`` (fp16) on the training device;
    ``teacher[j, v]`` is the teacher feature of pool view ``views[j, v]``. The lookup is
    ``teacher[batch.index, batch.view_index[:, 0]]``, a device-side gather (no host sync).
    """

    name = "distill"
    views_per_sample = 1

    def __init__(self, head_width: int, teacher: torch.Tensor):
        self.head_width = head_width
        self.teacher = teacher

    def build_head(self, feature_dim: int) -> nn.Module:
        h = self.head_width
        return nn.Sequential(nn.Linear(feature_dim, h), nn.BatchNorm1d(h), nn.GELU(), nn.Linear(h, TEACHER_DIM))

    def targets(self, batch) -> torch.Tensor:
        """The cached teacher features of the views in ``batch`` (same image, same view)."""
        return self.teacher[batch.index, batch.view_index[:, 0]]

    def loss(self, head: nn.Module, features: list[torch.Tensor], batch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        (f,) = features
        z = head(f).float()
        cos = F.cosine_similarity(z, self.targets(batch).float(), dim=1).mean()
        return 1.0 - cos, {"cosine": cos.detach()}


# ---------------------------------------------------------------- teacher cache on the device

_LOAD_ROWS = 4096  # rows per host->device chunk (4096 x 20 x 384 fp16 = 63 MB of host RAM)


def load_teacher(profile: str, n_used: int, k_views: int, device: str) -> torch.Tensor:
    """Rows ``[0, n_used)`` of the teacher cache as a fp16 tensor ``[n_used, K, 384]`` on ``device``.

    The whole used slice lives on the GPU (1.47 GiB at N=100k, K=20; half for the 50% runs),
    so the per-step lookup is a device gather with no host work and the shared loop needs no
    hook. Measured on the 3060 (#20): 43 µs/step for the GPU gather vs 67 µs for a pinned-CPU
    gather + non_blocking copy, both negligible next to a ~50–100 ms C step; the resident cache
    puts C-S peak at 2.6 GB and C-L at 2.7 GB, within the 6 GB card. It is copied in chunks so
    host RAM never holds more than one chunk. The cache must match the pool the run reads:
    shape-checked always, and the pool metadata recorded in ``cache.json`` is compared with
    the pool's ``meta.json`` when both exist (a rebuilt pool makes the cache stale).
    """
    import json

    from mo433 import paths, teacher

    cache = teacher.open_cache(profile)
    if cache.ndim != 3 or cache.shape[0] < n_used or cache.shape[1] != k_views or cache.shape[2] != TEACHER_DIM:
        raise ValueError(f"teacher cache {cache.shape} does not provide N>={n_used}, K={k_views}, {TEACHER_DIM}")
    meta_path = teacher.cache_path(profile).with_name(teacher.CACHE_META_FILE)
    pool_meta_path = paths.pool_dir(profile) / "meta.json"
    if meta_path.exists() and pool_meta_path.exists():
        cached_pool = json.loads(meta_path.read_text()).get("pool")
        if cached_pool is not None and cached_pool != json.loads(pool_meta_path.read_text()):
            raise ValueError(f"teacher cache {meta_path} was built from a different pool; rerun `mo433 teacher --force`")
    out = torch.empty((n_used, k_views, TEACHER_DIM), dtype=torch.float16, device=device)
    for a in range(0, n_used, _LOAD_ROWS):
        b = min(a + _LOAD_ROWS, n_used)
        out[a:b].copy_(torch.from_numpy(np.array(cache[a:b])))
    return out


def _factory(s) -> DistillObjective:
    return DistillObjective(s.head_width, load_teacher(s.profile, s.n_used, s.k_views, s.device))


def _register() -> None:
    from mo433 import train

    train.register_objective("C", _factory)


_register()
