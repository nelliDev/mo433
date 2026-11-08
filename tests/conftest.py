import os
import re
from pathlib import Path

# Cap joblib workers before mo433.probe reads it: 16 workers per pytest run OOM-killed the laptop.
os.environ.setdefault("MO433_N_JOBS", "2")

import pytest

from mo433 import paths


@pytest.fixture(autouse=True)
def _main_dataset(monkeypatch):
    """Tests run on the main dataset unless they set $MO433_DATASET themselves: another dataset
    ignores $MO433_ROOT (paths), so an inherited MO433_DATASET=eurosat would let stage tests
    write into the repo's artifacts-eurosat/."""
    monkeypatch.delenv("MO433_DATASET", raising=False)


@pytest.fixture
def mo433_root(tmp_path, monkeypatch) -> Path:
    """Point $MO433_ROOT and $MO433_SMOKE_ROOT at a fresh tmp dir so stages never touch real artifacts."""
    root = tmp_path / "artifacts"
    monkeypatch.setenv("MO433_ROOT", str(root))
    monkeypatch.setenv("MO433_SMOKE_ROOT", str(root))
    monkeypatch.delenv("MO433_POOL", raising=False)
    return root


@pytest.fixture
def null_svm_configs(tmp_path, monkeypatch) -> Path:
    """A copy of configs/ with probe C/gamma back at null, the state before `probe --select-svm`."""
    import shutil

    configs = tmp_path / "configs-null-svm"
    shutil.copytree(paths.CONFIGS_DIR, configs)
    base = configs / "base.yaml"
    base.write_text(re.sub(r"(?m)^(\s+(?:linear_c|rbf_c|rbf_gamma):)\s*\S+", r"\1 null", base.read_text()))
    monkeypatch.setattr(paths, "CONFIGS_DIR", configs)
    return configs


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
