"""Resumable stage runner (spec §9 "Resume").

A stage is any callable that writes its outputs. ``run_stage`` calls it, and only after
it returns writes ``<artifacts_root>/stages/<name>.done`` atomically (temp file + rename).
A finished stage is skipped; ``force=True`` deletes the marker and rebuilds. A stage that
raises leaves no marker, so the next call reruns it.

Stage names may contain ``/`` to namespace them, e.g. ``"train/B-S-100-s0"``.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from mo433 import paths


def marker_path(name: str, profile: str = "full") -> Path:
    if not name or name.startswith("/") or ".." in Path(name).parts:
        raise ValueError(f"bad stage name {name!r}")
    return paths.stages_dir(profile) / f"{name}.done"


def is_done(name: str, profile: str = "full") -> bool:
    return marker_path(name, profile).exists()


def run_stage(
    name: str,
    build: Callable[[], object],
    *,
    profile: str = "full",
    force: bool = False,
    log: Callable[[str], None] = print,
) -> bool:
    """Run ``build`` unless stage ``name`` is already done. Returns True if it ran."""
    marker = marker_path(name, profile)
    if marker.exists():
        if not force:
            log(f"[{name}] done, skipping (use --force to rebuild)")
            return False
        marker.unlink()
    log(f"[{name}] running")
    start = time.time()
    build()
    _write_atomic(
        marker,
        {
            "stage": name,
            "profile": profile,
            "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "seconds": round(time.time() - start, 3),
            **git_state(),
        },
    )
    log(f"[{name}] done")
    return True


def git_state() -> dict[str, object]:
    """Current commit and dirty flag, recorded in markers and run directories (spec §9)."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=paths.REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=paths.REPO_ROOT,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
        return {"git_commit": commit, "git_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": None}


def _write_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
