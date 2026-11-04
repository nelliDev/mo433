import csv
import json

import numpy as np
import pytest

from mo433 import probe, splits
from mo433.config import ProbeConfig

N_TRAIN, N_TEST, N_CLASSES = 5000, 800, 10


def _features(d=12, seed=0, sep=4.0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(N_CLASSES, d)) * sep
    y_train = np.repeat(np.arange(N_CLASSES), N_TRAIN // N_CLASSES)
    y_test = np.repeat(np.arange(N_CLASSES), N_TEST // N_CLASSES)
    x_train = (centers[y_train] + rng.normal(size=(N_TRAIN, d))).astype(np.float32)
    x_test = (centers[y_test] + rng.normal(size=(N_TEST, d))).astype(np.float32)
    return {"train_x": x_train, "train_y": y_train, "test_x": x_test, "test_y": y_test}


@pytest.fixture
def splits_dir(tmp_path):
    out = tmp_path / "splits"
    out.mkdir()
    y = np.repeat(np.arange(N_CLASSES), N_TRAIN // N_CLASSES)
    for name, arr in splits.generate(y, n_unlabeled=100).items():
        np.save(out / f"{name}.npy", arr)
    return out


def test_knn_only_probe_scores_all_twelve_label_subsets(splits_dir):
    rows = probe.probe_features(_features(), svm=None, splits_dir=splits_dir, n_jobs=1)

    assert [(r["label_frac"], r["draw"]) for r in rows] == [(f, d) for f in (1, 10, 50, 100) for d in (0, 1, 2)]
    assert {r["probe"] for r in rows} == {"knn"}
    assert all(r["accuracy"] > 0.9 for r in rows)  # well-separated blobs
    by_frac = {r["label_frac"]: r for r in rows}
    assert by_frac[1]["k"] == 5 and by_frac[1]["n_train"] == 50  # k = min(20, n_per_class)
    assert by_frac[100]["k"] == 20 and by_frac[100]["n_train"] == 5000


def test_svm_probes_record_their_hyperparameters_and_frozen_flag(splits_dir):
    svm = probe.SvmParams(linear_c=1.0, rbf_c=10.0, rbf_gamma="scale", frozen=False)
    rows = probe.probe_features(_features(), svm=svm, splits_dir=splits_dir, n_jobs=2)

    assert len(rows) == 36
    linear = [r for r in rows if r["probe"] == "linear_svm"]
    rbf = [r for r in rows if r["probe"] == "rbf_svm"]
    assert len(linear) == len(rbf) == 12
    assert all(r["C"] == 1.0 and r["gamma"] == "" and r["svm_params_frozen"] is False for r in linear)
    assert all(r["C"] == 10.0 and r["gamma"] == "scale" for r in rbf)
    assert all(r["accuracy"] > 0.9 for r in rows)


def test_svm_params_from_config_refuse_null_values_unless_provisional():
    null = ProbeConfig(linear_c=None, rbf_c=None, rbf_gamma=None)
    assert probe.svm_params(null, provisional=False) is None
    prov = probe.svm_params(null, provisional=True)
    assert (prov.linear_c, prov.rbf_c, prov.rbf_gamma, prov.frozen) == (1.0, 10.0, "scale", False)

    frozen = ProbeConfig(linear_c=0.1, rbf_c=100.0, rbf_gamma=1.0)
    p = probe.svm_params(frozen, provisional=True)  # frozen values always win
    assert (p.linear_c, p.rbf_c, p.rbf_gamma, p.frozen) == (0.1, 100.0, 1.0, True)


def _row(encoder_id, probe_name="knn", frac=1, draw=0, acc=0.5, feature_set="main"):
    return {
        "encoder_id": encoder_id, "feature_set": feature_set, "label_frac": frac, "draw": draw,
        "probe": probe_name, "accuracy": acc, "n_train": 50, "k": 5, "C": "", "gamma": "",
        "svm_params_frozen": "",
    }


def test_upsert_replaces_an_encoders_rows_and_keeps_the_others(tmp_path):
    path = tmp_path / "results.csv"
    probe.upsert_results([_row("A-S-s0", acc=0.1), _row("A-S-s0", frac=10, acc=0.2), _row("D", acc=0.9)], path)
    probe.upsert_results([_row("A-S-s0", acc=0.3)], path)

    with open(path) as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames[:5] == ["encoder_id", "feature_set", "label_frac", "draw", "probe"]
        assert "accuracy" in reader.fieldnames and "svm_params_frozen" in reader.fieldnames
        got = sorted((r["encoder_id"], r["label_frac"], float(r["accuracy"])) for r in reader)
    assert got == [("A-S-s0", "1", 0.3), ("D", "1", 0.9)]


def test_upsert_replaces_only_the_feature_sets_being_written(tmp_path):
    path = tmp_path / "results.csv"
    probe.upsert_results([_row("B-S-100-s0"), _row("B-S-100-s0", feature_set="ms_concat")], path)
    probe.upsert_results([_row("B-S-100-s0", feature_set="ms_concat", acc=0.7)], path)

    rows = probe.read_results(path)
    assert sorted((r["feature_set"], r["accuracy"]) for r in rows) == [("main", 0.5), ("ms_concat", 0.7)]


@pytest.fixture
def configs_dir(tmp_path):
    import shutil

    from mo433 import paths

    out = tmp_path / "configs"
    shutil.copytree(paths.CONFIGS_DIR, out)
    return out


def _write_features(mo433_root, encoder_id, feats):
    from mo433 import paths

    paths.features_dir().mkdir(parents=True, exist_ok=True)
    np.savez(paths.features_dir() / f"{encoder_id}.npz", **feats)


SMALL_GRID = dict(linear_grid=(1e-3, 1.0), rbf_grid=((1.0, "scale"), (10.0, 0.1)))


def test_select_svm_freezes_values_into_base_yaml_and_writes_a_record(mo433_root, configs_dir, splits_dir):
    from mo433 import config, paths

    _write_features(mo433_root, "B-S-100-s0", _features(seed=3, sep=1.0))
    record = probe.select_svm(configs_dir=configs_dir, splits_dir=splits_dir, n_jobs=2, **SMALL_GRID)

    cfg = config.load_config(configs_dir=configs_dir).probe
    assert (cfg.linear_c, cfg.rbf_c, cfg.rbf_gamma) == (record["linear_c"], record["rbf_c"], record["rbf_gamma"])
    assert cfg.linear_c in SMALL_GRID["linear_grid"]
    assert (cfg.rbf_c, cfg.rbf_gamma) in SMALL_GRID["rbf_grid"]
    text = (configs_dir / "base.yaml").read_text()
    assert "# null until SVM selection" in text  # comments survive the rewrite
    assert "k_views: 20" in text
    saved = json.loads((paths.results_dir() / "svm_selection.json").read_text())
    assert saved["encoder_id"] == "B-S-100-s0"
    assert len(saved["linear_cv"]) == 2 and len(saved["rbf_cv"]) == 2


def test_select_svm_refuses_once_a_c_encoder_has_been_probed(mo433_root, configs_dir, splits_dir):
    _write_features(mo433_root, "B-S-100-s0", _features())
    probe.upsert_results([_row("C-S-100-s0")], probe.results_csv_path())
    before = (configs_dir / "base.yaml").read_text()

    with pytest.raises(probe.SvmSelectionRefused, match="C-S-100-s0"):
        probe.select_svm(configs_dir=configs_dir, splits_dir=splits_dir, n_jobs=1, **SMALL_GRID)
    assert (configs_dir / "base.yaml").read_text() == before


@pytest.mark.invariant
def test_encoder_weights_unchanged_across_extract_and_probe(mo433_root, splits_dir):
    """Spec §9 invariant: a trained run's weights on disk are byte-identical after extract + probe."""
    import hashlib

    import torch

    from mo433 import extract, paths
    from mo433.encoders import save_final_backbone
    from mo433.students import build_student

    rd = paths.run_dir("B-S-100-s0")
    save_final_backbone(build_student("S", seed=3), rd)

    def digest():
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(rd.iterdir())}

    def fake_split(split):
        g = torch.Generator().manual_seed({"train": 0, "test": 1}[split])
        n = {"train": N_TRAIN, "test": N_TEST}[split]
        return torch.randint(0, 256, (n, 96, 96, 3), dtype=torch.uint8, generator=g), torch.arange(n) // (n // N_CLASSES)

    before = digest()
    extract.extract("B-S-100-s0", device="cpu", load_split=fake_split)
    rows = probe.probe_encoder("B-S-100-s0", None, splits_dir=splits_dir, n_jobs=1)
    assert len(rows) == 12
    assert digest() == before
