"""The ResNet96 student family (spec §4.2), shared by rows A, B and C.

- Basic residual block (post-activation), 1×1 conv + BN projection shortcut on shape change.
- 3×3 stride-2 stem, no max-pool; stages [2,2,2,2] with widths w/2w/4w/8w; maps 48→24→12→6.
- Global average pool, no head. ``forward`` returns the 8w-d pooled stage-4 feature.
- Kaiming-normal (fan_out) conv init, BN γ=1 / β=0, no zero-init-γ. Only the seed varies.

``build_student(size, seed)`` is the one constructor: seed s gives the initial weights of
``A-{size}-s{s}`` and of the B/C runs with the same seed (spec §12.6).
"""

from __future__ import annotations

import torch
import torch.nn as nn

WIDTHS = {"S": 24, "L": 48, "XS": 12, "M": 34}  # XS and M: the extra-sizes study (mo433.extras) only
DEPTHS = (2, 2, 2, 2)


class BasicBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class ResNet96(nn.Module):
    """``forward(x) -> [B, 8w]``; ``forward_multiscale(x) -> (GAP(stage3) [B, 4w], GAP(stage4) [B, 8w])``."""

    def __init__(self, width: int, depths: tuple[int, ...] = DEPTHS):
        super().__init__()
        self.width = width
        self.stem = nn.Sequential(nn.Conv2d(3, width, 3, 2, 1, bias=False), nn.BatchNorm2d(width), nn.ReLU(inplace=True))
        stages, cin = [], width
        for i, depth in enumerate(depths):
            cout = width * 2**i
            blocks = [BasicBlock(cin if j == 0 else cout, cout, 2 if (i > 0 and j == 0) else 1) for j in range(depth)]
            stages.append(nn.Sequential(*blocks))
            cin = cout
        self.stages = nn.ModuleList(stages)
        self.dim = cin
        self.multiscale_dims = (cin // 2, cin)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _stage_maps(self, x: torch.Tensor) -> list[torch.Tensor]:
        maps, h = [], self.stem(x)
        for stage in self.stages:
            h = stage(h)
            maps.append(h)
        return maps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._stage_maps(x)[-1].mean((2, 3))

    def forward_multiscale(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        maps = self._stage_maps(x)
        return maps[2].mean((2, 3)), maps[3].mean((2, 3))


def build_student(size: str, seed: int) -> ResNet96:
    """A freshly initialized student; the same (size, seed) always gives the same weights."""
    if size not in WIDTHS:
        raise ValueError(f"unknown student size {size!r}; expected one of {sorted(WIDTHS)}")
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        return ResNet96(WIDTHS[size])
