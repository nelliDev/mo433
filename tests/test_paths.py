"""Path resolution: smoke must never resolve to the real view pool."""

from mo433 import paths


def test_smoke_pool_ignores_mo433_pool(tmp_path, monkeypatch):
    monkeypatch.setenv("MO433_ROOT", str(tmp_path / "root"))
    monkeypatch.setenv("MO433_POOL", str(tmp_path / "real-pool"))
    assert paths.pool_dir("full") == tmp_path / "real-pool"
    assert paths.pool_dir("smoke") != tmp_path / "real-pool"
    assert paths.pool_dir("smoke") == paths.artifacts_root("smoke") / "pool"


def test_smoke_root_ignores_mo433_root(tmp_path, monkeypatch):
    # $MO433_ROOT is the full root: a smoke --force run there would delete full artifacts.
    monkeypatch.setenv("MO433_ROOT", str(tmp_path / "root"))
    monkeypatch.delenv("MO433_SMOKE_ROOT", raising=False)
    assert paths.artifacts_root("full") == tmp_path / "root"
    assert paths.artifacts_root("smoke") == paths.REPO_ROOT / "artifacts-smoke"
    monkeypatch.setenv("MO433_SMOKE_ROOT", str(tmp_path / "smoke"))
    assert paths.artifacts_root("smoke") == tmp_path / "smoke"
    assert paths.artifacts_root("full") == tmp_path / "root"


def test_eurosat_dataset_gets_its_own_data_splits_artifacts_and_report(monkeypatch, tmp_path):
    # A second dataset must never read or overwrite the STL-10 study's files.
    monkeypatch.setenv("MO433_ROOT", str(tmp_path / "stl-root"))
    monkeypatch.setenv("MO433_POOL", str(tmp_path / "stl-pool"))
    monkeypatch.setenv("MO433_DATA", str(tmp_path / "stl-data"))
    monkeypatch.setenv("MO433_DATASET", "eurosat")
    assert paths.dataset() == "eurosat"
    assert paths.data_root() == paths.REPO_ROOT / "data" / "eurosat"
    assert paths.splits_dir() == paths.SPLITS_DIR / "eurosat"
    assert paths.artifacts_root("full") == paths.REPO_ROOT / "artifacts-eurosat"
    assert paths.artifacts_root("smoke") == paths.REPO_ROOT / "artifacts-eurosat-smoke"
    assert paths.pool_dir("full") == paths.REPO_ROOT / "artifacts-eurosat" / "pool"
    assert paths.report_dir() == paths.REPO_ROOT / "report" / "eurosat"


def test_default_dataset_is_stl10(monkeypatch):
    monkeypatch.delenv("MO433_DATASET", raising=False)
    monkeypatch.delenv("MO433_DATA", raising=False)
    assert paths.dataset() == "stl10"
    assert paths.splits_dir() == paths.SPLITS_DIR
    assert paths.report_dir() == paths.REPO_ROOT / "report"
    assert paths.data_root() == paths.REPO_ROOT / "data"


def test_unknown_dataset_is_rejected(monkeypatch):
    import pytest

    monkeypatch.setenv("MO433_DATASET", "cifar")
    with pytest.raises(ValueError, match="MO433_DATASET"):
        paths.dataset()
