"""Row J (extra, after the freeze): LeJEPA, a direct-training alternative to row B.

LeJEPA (Balestriero & LeCun, 2025, arXiv:2511.08544) replaces VICReg's variance/covariance
heuristics with SIGReg: random 1-D projections of the embeddings are pushed towards N(0, 1) by
the Epps-Pulley characteristic-function test, plus a prediction term pulling every view's
embedding to the mean embedding of the image's views. Reference: github.com/galilai-group/lejepa
(``lejepa/univariate/epps_pulley.py``, ``lejepa/multivariate/slicing.py``, ``MINIMAL.md``).

Settings are the reference defaults, untuned, like every other row: lambda = 0.05, 256 slices
resampled each step, Epps-Pulley on 17 trapezoid knots over t in [0, 3] with weight exp(-t^2/2),
projector ``MLP(d, [2048, 2048, 128])`` with BatchNorm. It uses the same two pool views per
image as row B (so the only difference from B is the objective) and the frozen optimizer.
"""

from __future__ import annotations

import torch
import torch.nn as nn

LAMBDA = 0.05
NUM_SLICES = 256
PROJ_DIM = 128
VIEWS = 2


class EppsPulley(nn.Module):
    """Epps-Pulley statistic of each column of ``x`` [N, S] against N(0, 1); returns [S].

    ``N * integral |ecf(t) - exp(-t^2/2)|^2 exp(-t^2/2) dt`` over the real line, as a trapezoid
    on [0, t_max] with doubled interior weights (the integrand is even).
    """

    def __init__(self, t_max: float = 3.0, n_points: int = 17):
        super().__init__()
        assert n_points % 2 == 1
        t = torch.linspace(0, t_max, n_points)
        dt = t_max / (n_points - 1)
        weights = torch.full((n_points,), 2 * dt)
        weights[[0, -1]] = dt
        phi = (-0.5 * t.square()).exp()
        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", weights * phi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        xt = x.unsqueeze(-1) * self.t  # [N, S, T]
        err = (xt.cos().mean(0) - self.phi).square() + xt.sin().mean(0).square()  # [S, T]
        return (err @ self.weights) * n


class SIGReg(nn.Module):
    """Sliced Epps-Pulley: mean statistic over ``num_slices`` fresh unit directions per call."""

    def __init__(self, num_slices: int = NUM_SLICES):
        super().__init__()
        self.num_slices = num_slices
        self.test = EppsPulley()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        a = torch.randn(z.size(-1), self.num_slices, device=z.device)  # global RNG: checkpointed
        a = a / a.norm(dim=0)
        return self.test(z @ a).mean()


class LeJEPAObjective:
    """``(1 - lambda) * mean ||mean_v z_v - z_v||^2 + lambda * mean_v SIGReg(z_v)`` on ``z.float()``."""

    name = "lejepa"
    views_per_sample = VIEWS

    def __init__(self, head_width: int, lam: float = LAMBDA, proj_dim: int = PROJ_DIM, num_slices: int = NUM_SLICES):
        self.head_width, self.lam, self.proj_dim = head_width, lam, proj_dim
        self.sigreg = SIGReg(num_slices)

    def build_head(self, feature_dim: int) -> nn.Module:
        from torchvision.ops import MLP

        h = self.head_width
        return MLP(feature_dim, [h, h, self.proj_dim], norm_layer=nn.BatchNorm1d)

    def loss(self, head: nn.Module, features: list[torch.Tensor], batch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        z = torch.stack([head(f).float() for f in features])  # [V, B, D]
        self.sigreg.to(z.device)
        # fp32 outside the loop's bf16 autocast: the ECF means need more than bf16's 8 mantissa bits.
        with torch.autocast(device_type=z.device.type, enabled=False):
            pred = (z.mean(0) - z).square().mean()
            sig = torch.stack([self.sigreg(zv) for zv in z]).mean()
        loss = (1 - self.lam) * pred + self.lam * sig
        return loss, {"prediction": pred.detach(), "sigreg": sig.detach()}


def _register() -> None:
    from mo433 import train

    train.register_objective("J", lambda s: LeJEPAObjective(s.head_width))


_register()
