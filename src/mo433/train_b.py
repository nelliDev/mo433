"""Row B: VICReg on two pool views per image (spec §5.2).

Only the objective lives here; the loop, data, optimizer, checkpointing and logging are the
shared ones in :mod:`mo433.train`. ``mo433 train --row B`` ends up in ``train.train_run``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class VICRegObjective:
    """``lightly.loss.VICRegLoss()`` defaults on ``z.float()``, expander
    ``VICRegProjectionHead(d, h, h, num_layers=3)``.

    The three terms are computed with lightly's own term functions and combined with the
    ``VICRegLoss()`` default coefficients, so the total is exactly ``VICRegLoss()(z1, z2)``
    (tested) while the terms come for free for the per-epoch log.
    """

    name = "vicreg"
    views_per_sample = 2

    def __init__(self, head_width: int):
        from lightly.loss import VICRegLoss

        self.head_width = head_width
        self._ref = VICRegLoss()  # defaults: lambda=25, mu=25, nu=1, eps=1e-4

    def build_head(self, feature_dim: int) -> nn.Module:
        from lightly.models.modules.heads import VICRegProjectionHead

        h = self.head_width
        return VICRegProjectionHead(feature_dim, h, h, num_layers=3)

    def loss(self, head: nn.Module, features: list[torch.Tensor], batch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        from lightly.loss.vicreg_loss import covariance_loss, invariance_loss, variance_loss

        f1, f2 = features
        z1, z2 = head(f1).float(), head(f2).float()
        ref = self._ref
        inv = invariance_loss(z1, z2)
        var = 0.5 * (variance_loss(z1, ref.eps) + variance_loss(z2, ref.eps))
        cov = covariance_loss(z1) + covariance_loss(z2)
        loss = ref.lambda_param * inv + ref.mu_param * var + ref.nu_param * cov
        return loss, {"invariance": inv.detach(), "variance": var.detach(), "covariance": cov.detach()}
