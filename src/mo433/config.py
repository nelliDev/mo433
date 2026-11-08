"""YAML configs -> frozen dataclasses (spec §9 "Configs").

Resolution order, later wins:

1. ``configs/base.yaml`` (shared settings; its ``profiles:`` table is set aside),
2. ``configs/<row>-<size>.yaml`` when a row and size are given,
3. ``configs/datasets/<dataset>.yaml`` when ``$MO433_DATASET`` is not the main ``stl10``,
4. ``profiles.<profile>`` from ``base.yaml`` (``smoke`` shrinks N, K and E).

Every key must name a dataclass field: unknown keys, missing keys and wrong types raise
``ConfigError``. Seed and data fraction are CLI arguments, never config keys.
"""

from __future__ import annotations

import dataclasses
import re
import types
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from mo433 import paths

PROFILES = ("full", "smoke")
ROWS = ("B", "C")
SIZES = ("S", "L")
EXTRA_SIZES = ("XS", "M")  # extra-sizes study (mo433.extras); never in the reported runs
FRACS = (100, 50)  # unlabeled-data percent; 50 = the ablation subset


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class OptimizerConfig:
    name: str
    lr: float
    weight_decay: float
    warmup_epochs: int
    betas: tuple[float, float]


@dataclass(frozen=True)
class ProbeConfig:
    """SVM settings; ``None`` until the CV selection on ``B-S-100-s0`` fills them (spec §7.2)."""

    linear_c: float | None
    rbf_c: float | None
    rbf_gamma: float | str | None


@dataclass(frozen=True)
class Config:
    profile: str
    n_images: int
    k_views: int
    epochs: int
    batch_size: int
    head_width: int
    optimizer: OptimizerConfig
    probe: ProbeConfig
    row: str | None = None
    size: str | None = None
    student_width: int | None = None

    @property
    def feature_dim(self) -> int | None:
        return None if self.student_width is None else 8 * self.student_width


def run_id(row: str, size: str, frac: int, seed: int) -> str:
    """Spec §8 run id, e.g. ``run_id("C", "L", 50, 1) == "C-L-050-s1"``."""
    return f"{row}-{size}-{frac:03d}-s{seed}"


_ENCODER_ID_PATTERNS = (
    re.compile(r"(?P<row>[BC])-(?P<size>[SL])-(?P<frac>\d{3})-s(?P<seed>\d+)"),
    re.compile(r"(?P<row>A)-(?P<size>[SL])-s(?P<seed>\d+)"),
    re.compile(r"(?P<row>D)"),
)


@dataclass(frozen=True)
class EncoderId:
    """A parsed encoder id; fields an encoder does not have are None (A: frac; D: all but row)."""

    row: str
    size: str | None
    frac: int | None
    seed: int | None

    @property
    def group(self) -> str:
        """Encoder group = the id without the seed (A-S, B-S-100, D)."""
        if self.row == "D":
            return "D"
        if self.row == "A":
            return f"A-{self.size}"
        return f"{self.row}-{self.size}-{self.frac:03d}"


def parse_encoder_id(encoder_id: str) -> EncoderId:
    """``B-S-100-s0`` / ``A-S-s0`` / ``D`` -> EncoderId; anything else raises ValueError."""
    for pat in _ENCODER_ID_PATTERNS:
        m = pat.fullmatch(encoder_id)
        if m:
            g = m.groupdict()
            return EncoderId(
                g["row"],
                g.get("size"),
                int(g["frac"]) if g.get("frac") else None,
                int(g["seed"]) if g.get("seed") is not None else None,
            )
    raise ValueError(f"not an encoder id: {encoder_id!r}")


SEEDS = {"S": (0, 1, 2), "L": (0, 1)}


def reported_runs() -> list[tuple[str, str, int, int]]:
    """The 20 reported runs as (row, size, frac, seed) in week-3 order (spec §8):
    main S, main L, ablation S, ablation L; B and C of a seed back to back."""
    return [
        (row, size, frac, seed)
        for frac in FRACS
        for size in SIZES
        for seed in SEEDS[size]
        for row in ROWS
    ]


def load_config(
    row: str | None = None,
    size: str | None = None,
    profile: str = "full",
    configs_dir: Path | None = None,
) -> Config:
    configs_dir = Path(configs_dir) if configs_dir is not None else paths.CONFIGS_DIR
    if profile not in PROFILES:
        raise ConfigError(f"unknown profile {profile!r}; expected one of {PROFILES}")
    if (row is None) != (size is None):
        raise ConfigError("row and size must be given together")

    base = _read_yaml(configs_dir / "base.yaml")
    profiles = base.pop("profiles", {}) or {}
    unknown_profiles = set(profiles) - set(PROFILES)
    if unknown_profiles:
        raise ConfigError(f"base.yaml: unknown profiles {sorted(unknown_profiles)}")

    merged = base
    if row is not None:
        if row not in ROWS or size not in SIZES + EXTRA_SIZES:
            raise ConfigError(f"unknown row/size {row!r}/{size!r}")
        override = _read_yaml(configs_dir / f"{row}-{size}.yaml")
        if (override.get("row"), override.get("size")) != (row, size):
            raise ConfigError(f"{row}-{size}.yaml must declare row: {row} and size: {size}")
        merged = _deep_merge(merged, override)
    if paths.dataset() != paths.DEFAULT_DATASET:
        overlay = _read_yaml(configs_dir / "datasets" / f"{paths.dataset()}.yaml")
        unknown = set(overlay) - {f.name for f in dataclasses.fields(Config)}
        if unknown:
            raise ConfigError(f"datasets/{paths.dataset()}.yaml: unknown keys {sorted(unknown)}")
        merged = _deep_merge(merged, overlay)
    merged = _deep_merge(merged, profiles.get(profile) or {})
    merged["profile"] = profile
    return _build(Config, merged, "config")


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    return data


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _build(cls: type, data: Any, where: str):
    if not isinstance(data, dict):
        raise ConfigError(f"{where}: expected a mapping, got {data!r}")
    hints = typing.get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(data) - set(fields)
    if unknown:
        raise ConfigError(f"{where}: unknown keys {sorted(unknown)}")
    kwargs = {}
    for name, field in fields.items():
        if name not in data:
            if field.default is dataclasses.MISSING:
                raise ConfigError(f"{where}: missing key {name!r}")
            continue
        kwargs[name] = _coerce(hints[name], data[name], f"{where}.{name}")
    return cls(**kwargs)


def _coerce(tp: Any, value: Any, where: str):
    if dataclasses.is_dataclass(tp):
        return _build(tp, value, where)
    origin = typing.get_origin(tp)
    if origin in (typing.Union, types.UnionType):
        errors = []
        for option in typing.get_args(tp):
            try:
                return _coerce(option, value, where)
            except ConfigError as e:
                errors.append(str(e))
        raise ConfigError(f"{where}: {value!r} matches none of {tp}")
    if origin is tuple:
        args = typing.get_args(tp)
        if not isinstance(value, (list, tuple)) or len(value) != len(args):
            raise ConfigError(f"{where}: expected {len(args)} items, got {value!r}")
        return tuple(_coerce(a, v, f"{where}[{i}]") for i, (a, v) in enumerate(zip(args, value)))
    if tp is type(None):
        if value is not None:
            raise ConfigError(f"{where}: expected null")
        return None
    if tp is float:
        # bool is an int subclass; reject it. A string here usually means PyYAML read
        # "1e-3" (no dot) as a string: write 1.0e-3.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{where}: expected a number, got {value!r}")
        return float(value)
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{where}: expected an integer, got {value!r}")
        return value
    if tp is str:
        if not isinstance(value, str):
            raise ConfigError(f"{where}: expected a string, got {value!r}")
        return value
    raise ConfigError(f"{where}: unsupported type {tp}")
