"""Stage-runner seam: outputs first, then an atomic .done marker; skip when done; --force rebuilds."""

import pytest

from mo433.stages import is_done, run_stage


def test_stage_runs_once_then_is_skipped_until_forced(mo433_root):
    calls = []

    def build():
        calls.append(1)

    assert run_stage("demo", build) is True
    assert is_done("demo")
    assert run_stage("demo", build) is False
    assert len(calls) == 1
    assert run_stage("demo", build, force=True) is True
    assert len(calls) == 2


def test_failed_stage_leaves_no_marker_and_reruns(mo433_root):
    def boom():
        raise RuntimeError("disk full")

    with pytest.raises(RuntimeError):
        run_stage("flaky", boom)
    assert not is_done("flaky")
    assert run_stage("flaky", lambda: None) is True


def test_namespaced_stage_names(mo433_root):
    run_stage("train/B-S-100-s0", lambda: None)
    assert is_done("train/B-S-100-s0")
    assert not is_done("train/B-S-100-s1")


def test_smoke_and_full_markers_are_separate(tmp_path, monkeypatch):
    monkeypatch.delenv("MO433_ROOT", raising=False)
    monkeypatch.delenv("MO433_SMOKE_ROOT", raising=False)
    monkeypatch.setattr("mo433.paths.REPO_ROOT", tmp_path)
    run_stage("demo", lambda: None, profile="smoke")
    assert is_done("demo", profile="smoke")
    assert not is_done("demo", profile="full")
    assert (tmp_path / "artifacts-smoke" / "stages" / "demo.done").exists()
