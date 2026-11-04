"""`mo433 probe` / `mo433 diagnose` end to end on a synthetic features file."""

import json
import shutil

import numpy as np
import pytest

from mo433 import paths, probe, splits
from mo433.cli import main

N_TRAIN, N_TEST, C, W = 5000, 500, 10, 4  # 8w = 32-d main features


@pytest.fixture
def repo(tmp_path, monkeypatch, mo433_root):
    """Synthetic splits and a copy of configs/, so nothing committed is touched."""
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    y = np.repeat(np.arange(C), N_TRAIN // C)
    for name, arr in splits.generate(y, n_unlabeled=100).items():
        np.save(splits_dir / f"{name}.npy", arr)
    configs = tmp_path / "configs"
    shutil.copytree(paths.CONFIGS_DIR, configs)
    monkeypatch.setattr(paths, "SPLITS_DIR", splits_dir)
    monkeypatch.setattr(paths, "CONFIGS_DIR", configs)
    return tmp_path


def _write(encoder_id, seed=0, sep=3.0, multiscale=True):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(C, 12 * W)) * sep
    y_tr = np.repeat(np.arange(C), N_TRAIN // C)
    y_te = np.repeat(np.arange(C), N_TEST // C)
    s34_tr = (centers[y_tr] + rng.normal(size=(N_TRAIN, 12 * W))).astype(np.float32)
    s34_te = (centers[y_te] + rng.normal(size=(N_TEST, 12 * W))).astype(np.float32)
    feats = {"train_x": s34_tr[:, 4 * W :], "train_y": y_tr, "test_x": s34_te[:, 4 * W :], "test_y": y_te}
    if multiscale:
        feats |= {
            "train_x_s3": s34_tr[:, : 4 * W], "train_x_s4": s34_tr[:, 4 * W :],
            "test_x_s3": s34_te[:, : 4 * W], "test_x_s4": s34_te[:, 4 * W :],
        }
    paths.features_dir().mkdir(parents=True, exist_ok=True)
    np.savez(paths.features_dir() / f"{encoder_id}.npz", **feats)


def _rows():
    return probe.read_results(probe.results_csv_path())


def test_probe_runs_knn_only_while_svm_settings_are_null(repo, capsys):
    _write("A-S-s0")
    assert main(["probe", "--encoder", "A-S-s0"]) == 0
    rows = _rows()
    assert len(rows) == 12 and {r["probe"] for r in rows} == {"knn"}
    assert "k-NN only" in capsys.readouterr().out


def test_provisional_svm_rows_are_marked_unfrozen_and_reprobing_replaces_them(repo):
    _write("A-S-s0")
    _write("D", multiscale=False)
    assert main(["probe", "--provisional-svm"]) == 0  # every encoder with features
    assert main(["probe", "--provisional-svm", "--encoder", "A-S-s0"]) == 0
    rows = _rows()
    assert len(rows) == 72
    svm_rows = [r for r in rows if r["probe"] != "knn"]
    assert svm_rows and all(r["svm_params_frozen"] is False for r in svm_rows)


def test_select_svm_then_probe_uses_frozen_settings(repo):
    _write("B-S-100-s0")
    assert main(["probe", "--select-svm"]) == 0
    assert main(["probe", "--encoder", "B-S-100-s0"]) == 0
    svm_rows = [r for r in _rows() if r["probe"] != "knn"]
    assert len(svm_rows) == 24 and all(r["svm_params_frozen"] is True for r in svm_rows)


def test_select_svm_refuses_after_a_c_encoder_was_probed(repo, capsys):
    _write("B-S-100-s0")
    _write("C-S-100-s0")
    assert main(["probe", "--encoder", "C-S-100-s0"]) == 0
    with pytest.raises(SystemExit, match="C-S-100-s0"):
        main(["probe", "--select-svm"])


def test_diagnose_writes_record_tsne_and_multiscale_rows(repo):
    _write("A-S-s0", seed=1, sep=0.2)
    _write("B-S-100-s0", seed=0, sep=3.0)
    assert main(["diagnose", "--encoder", "B-S-100-s0"]) == 0

    out = paths.results_dir() / "diagnostics"
    record = json.loads((out / "B-S-100-s0.json").read_text())
    assert record["collapse_check"]["anchor_id"] == "A-S-s0"
    assert record["collapse_check"]["flagged"] is False
    assert 0 < record["test_characterization"]["test_nmi"] <= 1
    assert np.load(out / "B-S-100-s0_tsne.npy").shape == (N_TEST, 2)
    assert record["multiscale"]["dims"] == {"ms_concat": 12 * W, "ms_randproj": 8 * W}
    assert {r["feature_set"] for r in _rows()} == {"ms_concat", "ms_randproj"}
