import os
from pathlib import Path

# Cap joblib workers before mo433.probe reads it: 16 workers per pytest run OOM-killed the laptop.
os.environ.setdefault("MO433_N_JOBS", "2")

import pytest

from mo433 import paths


@pytest.fixture
def mo433_root(tmp_path, monkeypatch) -> Path:
    """Point $MO433_ROOT at a fresh tmp dir so stages never touch real artifacts."""
    root = tmp_path / "artifacts"
    monkeypatch.setenv("MO433_ROOT", str(root))
    monkeypatch.delenv("MO433_POOL", raising=False)
    return root


def _stl10_present() -> bool:
    return (paths.stl10_binary_dir() / "train_y.bin").exists()


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    skip_data = pytest.mark.skip(reason=f"STL-10 not found under {paths.data_root()} (set MO433_DATA)")
    skip_gpu = pytest.mark.skip(reason="no CUDA GPU")
    has_data, has_gpu = None, None
    for item in items:
        if "data" in item.keywords:
            has_data = _stl10_present() if has_data is None else has_data
            if not has_data:
                item.add_marker(skip_data)
        if "gpu" in item.keywords:
            has_gpu = _cuda_available() if has_gpu is None else has_gpu
            if not has_gpu:
                item.add_marker(skip_gpu)
