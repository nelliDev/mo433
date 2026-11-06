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
