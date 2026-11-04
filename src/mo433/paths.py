"""Single place where every filesystem location is resolved.

Environment variables (all optional):

- ``MO433_ROOT``: artifact root. Default ``<repo>/artifacts`` for the full profile and
  ``<repo>/artifacts-smoke`` for the smoke profile. When set, it is used as-is for
  every profile (point it somewhere else for smoke runs if you need both).
- ``MO433_POOL``: view-pool directory (``chattr +C``). Default ``$MO433_ROOT/pool``.
- ``MO433_DATA``: STL-10 root, the directory that contains ``stl10_binary/``.
  Default ``<repo>/data``. Set it in git worktrees, which lack the gitignored ``data/``.

Committed locations (``configs/``, ``splits/``) always live in the repo.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"
SPLITS_DIR = REPO_ROOT / "splits"

_DEFAULT_ROOT_NAME = {"full": "artifacts", "smoke": "artifacts-smoke"}


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else None


def artifacts_root(profile: str = "full") -> Path:
    """``$MO433_ROOT``, or the profile's default directory under the repo."""
    if profile not in _DEFAULT_ROOT_NAME:
        raise ValueError(f"unknown profile {profile!r}; expected one of {sorted(_DEFAULT_ROOT_NAME)}")
    return _env_path("MO433_ROOT") or REPO_ROOT / _DEFAULT_ROOT_NAME[profile]


def pool_dir(profile: str = "full") -> Path:
    """``$MO433_POOL``, or ``<artifacts_root>/pool``. Smoke never uses ``$MO433_POOL``: it points at
    the real 55 GB pool, which ``mo433 --profile smoke pool --force`` would otherwise delete."""
    env = _env_path("MO433_POOL") if profile != "smoke" else None
    return env or artifacts_root(profile) / "pool"


def data_root() -> Path:
    """``$MO433_DATA``: the directory passed as ``root`` to ``torchvision.datasets.STL10``."""
    return _env_path("MO433_DATA") or REPO_ROOT / "data"


def stl10_binary_dir() -> Path:
    return data_root() / "stl10_binary"


# Gitignored per-stage locations under the artifact root (spec §9 "Storage").
def teacher_dir(profile: str = "full") -> Path:
    return artifacts_root(profile) / "teacher"


def run_dir(run_id: str, profile: str = "full") -> Path:
    return artifacts_root(profile) / "runs" / run_id


def features_dir(profile: str = "full") -> Path:
    return artifacts_root(profile) / "features"


def results_dir(profile: str = "full") -> Path:
    return artifacts_root(profile) / "results"


def tuning_dir(profile: str = "full") -> Path:
    return artifacts_root(profile) / "tuning"


def stages_dir(profile: str = "full") -> Path:
    """Where stage ``.done`` markers live."""
    return artifacts_root(profile) / "stages"
