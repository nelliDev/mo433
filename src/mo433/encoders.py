"""Encoder registry: ``load_encoder(encoder_id) -> Encoder`` (spec §7.1).

An **encoder** (GLOSSARY) is any frozen feature extractor that is probed. Every one is an
``Encoder``: an ``nn.Module`` that is frozen and permanently in ``eval()``, and that carries
its own input preprocessing, so ``extract`` treats all rows alike:

    enc = load_encoder("A-S-s0")
    x = enc.preprocess(u8_nhwc_on_device)   # uint8 [B, 96, 96, 3] -> model input
    feats = enc.embed(x)                    # {"x": [B, dim], optional extra features}
    feats["x"] == enc(x)

Encoder ids
-----------
- Row A: ``A-{S,L}-s{seed}``. ``build_student(size, seed)`` with BN running stats at their
  defaults (mean 0, var 1). It equals the initial weights of the B/C runs with that seed.
- Rows B/C: ``{B,C}-{S,L}-{frac:03d}-s{seed}`` (the run id, spec §8). Loads
  ``$MO433_ROOT/runs/<run_id>/final_backbone.pt``, a ResNet96 ``state_dict`` written by the
  trainer with ``save_final_backbone(net, run_dir)``.
- Row D: ``D``, the frozen DINO ViT-S/16, registered by ``mo433.teacher``.

Registering a new kind of encoder
---------------------------------
Decorate a factory with ``@register(regex)``; it gets the ``re.Match`` of the full id and the
profile and returns an ``Encoder``. Modules that register encoders but are not imported here
are listed in ``PROVIDERS`` and imported lazily on the first lookup, e.g. for the teacher::

    # mo433/teacher.py
    @register(r"D")
    def _teacher(m, profile):
        return Encoder("D", vit, dim=384, preprocess=resize224_and_normalize,
                       autocast_dtype=torch.float16)

and adds ``ROW_IDS["D"] = lambda: ["D"]`` so ``mo433 extract --rows D`` finds it.
``extract`` never changes when an encoder kind is added.
"""

from __future__ import annotations

import importlib
import re
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn as nn

from mo433 import config, paths
from mo433.data import normalize
from mo433.students import WIDTHS, ResNet96, build_student

FINAL_BACKBONE = "final_backbone.pt"

# Modules that call ``register`` at import time; imported on the first ``load_encoder``.
PROVIDERS: tuple[str, ...] = ("mo433.teacher",)


class UnknownEncoder(KeyError):
    pass


EmbedFn = Callable[[nn.Module, torch.Tensor], dict[str, torch.Tensor]]


def _plain_embed(net: nn.Module, x: torch.Tensor) -> dict[str, torch.Tensor]:
    return {"x": net(x)}


class Encoder(nn.Module):
    """A frozen feature extractor plus its preprocessing.

    - ``preprocess(u8)``: uint8 NHWC batch, already on the encoder's device -> model input.
    - ``forward(x) -> [B, dim]``: the probed feature (flat vector).
    - ``embed(x) -> dict``: ``"x"`` (= ``forward``) plus any extra features (students add
      ``"x_s3"`` / ``"x_s4"``, the stage-3/4 GAP for the multi-scale probe), in one pass.
    - ``autocast_dtype``: run the forward under autocast at this dtype (teacher: fp16);
      ``None`` means fp32. Outputs are always cast to float32 by ``extract``.
    """

    def __init__(
        self,
        encoder_id: str,
        net: nn.Module,
        dim: int,
        preprocess: Callable[[torch.Tensor], torch.Tensor] = normalize,
        embed: EmbedFn = _plain_embed,
        autocast_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.encoder_id = encoder_id
        self.net = net
        self.dim = dim
        self.preprocess = preprocess
        self._embed = embed
        self.autocast_dtype = autocast_dtype
        self.requires_grad_(False)
        self.train(False)

    def train(self, mode: bool = True) -> Encoder:
        """Always ``eval()``: features are never computed with BN in train mode."""
        return super().train(False)

    def embed(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return self._embed(self.net, x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(x)["x"]


# ---------------------------------------------------------------- registry

_REGISTRY: list[tuple[re.Pattern[str], Callable[[re.Match[str], str], Encoder]]] = []


def register(pattern: str):
    """Register ``factory(match, profile) -> Encoder`` for ids that fully match ``pattern``."""

    def deco(factory):
        _REGISTRY.append((re.compile(pattern), factory))
        return factory

    return deco


def load_encoder(encoder_id: str, profile: str = "full") -> Encoder:
    for module in PROVIDERS:
        importlib.import_module(module)
    for pattern, factory in _REGISTRY:
        m = pattern.fullmatch(encoder_id)
        if m:
            return factory(m, profile)
    raise UnknownEncoder(encoder_id)


def is_registered(encoder_id: str) -> bool:
    """True if some factory accepts ``encoder_id`` (without building it)."""
    for module in PROVIDERS:
        importlib.import_module(module)
    return any(pattern.fullmatch(encoder_id) for pattern, _ in _REGISTRY)


def _row_a_ids() -> list[str]:
    return [f"A-{size}-s{seed}" for size in config.SIZES for seed in config.SEEDS[size]]


def _run_ids(row: str) -> Callable[[], list[str]]:
    return lambda: [config.run_id(*r) for r in config.reported_runs() if r[0] == row]


# Every encoder of a row, for ``mo433 extract --rows``. Row D adds its entry with its factory.
ROW_IDS: dict[str, Callable[[], list[str]]] = {"A": _row_a_ids, "B": _run_ids("B"), "C": _run_ids("C")}


def encoder_ids(rows: list[str]) -> list[str]:
    for module in PROVIDERS:
        importlib.import_module(module)
    return [eid for row in rows for eid in ROW_IDS[row]()]


# ---------------------------------------------------------------- students (rows A, B, C)


def _student_embed(net: nn.Module, x: torch.Tensor) -> dict[str, torch.Tensor]:
    s3, s4 = net.forward_multiscale(x)
    return {"x": s4, "x_s3": s3, "x_s4": s4}


def student_encoder(encoder_id: str, net: ResNet96) -> Encoder:
    return Encoder(encoder_id, net, net.dim, embed=_student_embed)


@register(r"A-(?P<size>[SL])-s(?P<seed>\d+)")
def _row_a(m: re.Match[str], profile: str) -> Encoder:
    return student_encoder(m.group(0), build_student(m["size"], int(m["seed"])))


@register(r"(?P<row>[BC])-(?P<size>[SL])-(?P<frac>\d{3})-s(?P<seed>\d+)")
def _trained_run(m: re.Match[str], profile: str) -> Encoder:
    path = paths.run_dir(m.group(0), profile) / FINAL_BACKBONE
    if not path.exists():
        raise FileNotFoundError(f"{m.group(0)}: no final weights at {path}")
    net = ResNet96(WIDTHS[m["size"]])
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return student_encoder(m.group(0), net)


def save_final_backbone(net: ResNet96, run_dir: Path) -> Path:
    """What a B/C trainer writes at the end of a run so ``load_encoder(run_id)`` finds it."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / FINAL_BACKBONE
    tmp = path.with_suffix(".tmp")
    torch.save({k: v.detach().cpu() for k, v in net.state_dict().items()}, tmp)
    tmp.replace(path)
    return path
