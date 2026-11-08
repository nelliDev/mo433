"""Single place where every filesystem location is resolved.

Environment variables (all optional):

- ``MO433_ROOT``: artifact root of the full profile. Default ``<repo>/artifacts``.
- ``MO433_SMOKE_ROOT``: artifact root of the smoke profile. Default ``<repo>/artifacts-smoke``.
  Smoke never uses ``$MO433_ROOT``, so a smoke ``--force`` run cannot touch full artifacts.
- ``MO433_POOL``: view-pool directory (``chattr +C``). Default ``$MO433_ROOT/pool``.
- ``MO433_DATA``: STL-10 root, the directory that contains ``stl10_binary/``.
  Default ``<repo>/data``. Set it in git worktrees, which lack the gitignored ``data/``.
- ``MO433_DATASET``: ``stl10`` (default, the main study) or ``eurosat`` (the second-dataset
  extra). EuroSAT is stored in the STL-10 binary layout under ``<repo>/data/eurosat`` (see
  ``mo433.eurosat``) and has its own splits (``splits/eurosat``), artifact roots
  (``artifacts-eurosat[-smoke]``, ignoring ``MO433_ROOT``/``MO433_SMOKE_ROOT``/``MO433_POOL``/``MO433_DATA``,
  which belong to the main study) and report directory (``report/eurosat``).

Committed locations (``configs/``, ``splits/``) always live in the repo.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"
SPLITS_DIR = REPO_ROOT / "splits"

_DEFAULT_ROOT_NAME = {"full": "artifacts", "smoke": "artifacts-smoke"}
_ROOT_ENV = {"full": "MO433_ROOT", "smoke": "MO433_SMOKE_ROOT"}
DATASETS = ("stl10", "eurosat")
DEFAULT_DATASET = "stl10"


def dataset() -> str:
    """``$MO433_DATASET`` (default ``stl10``); anything outside ``DATASETS`` raises ValueError."""
    name = os.environ.get("MO433_DATASET") or DEFAULT_DATASET
    if name not in DATASETS:
        raise ValueError(f"MO433_DATASET={name!r}; expected one of {DATASETS}")
    return name


def _is_main_dataset() -> bool:
    return dataset() == DEFAULT_DATASET


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else None


def artifacts_root(profile: str = "full") -> Path:
    """``$MO433_ROOT`` (full) or ``$MO433_SMOKE_ROOT`` (smoke), else the profile's default
    directory under the repo."""
    if profile not in _DEFAULT_ROOT_NAME:
        raise ValueError(f"unknown profile {profile!r}; expected one of {sorted(_DEFAULT_ROOT_NAME)}")
    if not _is_main_dataset():
        return REPO_ROOT / _DEFAULT_ROOT_NAME[profile].replace("artifacts", f"artifacts-{dataset()}")
    return _env_path(_ROOT_ENV[profile]) or REPO_ROOT / _DEFAULT_ROOT_NAME[profile]


def pool_dir(profile: str = "full") -> Path:
    """``$MO433_POOL``, or ``<artifacts_root>/pool``. Smoke never uses ``$MO433_POOL``: it points at
    the real 55 GB pool, which ``mo433 --profile smoke pool --force`` would otherwise delete."""
    env = _env_path("MO433_POOL") if profile != "smoke" and _is_main_dataset() else None
    return env or artifacts_root(profile) / "pool"


def data_root() -> Path:
    """``$MO433_DATA``: the directory passed as ``root`` to ``torchvision.datasets.STL10``."""
    if not _is_main_dataset():
        return REPO_ROOT / "data" / dataset()
    return _env_path("MO433_DATA") or REPO_ROOT / "data"


def stl10_binary_dir() -> Path:
    return data_root() / "stl10_binary"


def splits_dir() -> Path:
    """Committed index files of the current dataset: ``SPLITS_DIR`` or ``SPLITS_DIR/<dataset>``."""
    return SPLITS_DIR if _is_main_dataset() else SPLITS_DIR / dataset()


def report_dir() -> Path:
    """Committed report inputs and outputs: ``<repo>/report`` or ``<repo>/report/<dataset>``."""
    return REPO_ROOT / "report" if _is_main_dataset() else REPO_ROOT / "report" / dataset()


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
