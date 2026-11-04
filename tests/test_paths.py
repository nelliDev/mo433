"""Path resolution: smoke must never resolve to the real view pool."""

from mo433 import paths


def test_smoke_pool_ignores_mo433_pool(tmp_path, monkeypatch):
    monkeypatch.setenv("MO433_ROOT", str(tmp_path / "root"))
    monkeypatch.setenv("MO433_POOL", str(tmp_path / "real-pool"))
    assert paths.pool_dir("full") == tmp_path / "real-pool"
    assert paths.pool_dir("smoke") != tmp_path / "real-pool"
    assert paths.pool_dir("smoke") == paths.artifacts_root("smoke") / "pool"
