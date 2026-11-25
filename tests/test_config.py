"""Config seam: YAML files -> frozen, validated Config objects."""

import dataclasses
import json
import re
from pathlib import Path

import pytest

from mo433.config import load_config, run_id
from mo433.paths import REPO_ROOT


def test_row_size_config_has_spec_budget():
    cfg = load_config(row="B", size="S")
    assert (cfg.row, cfg.size, cfg.profile) == ("B", "S", "full")
    assert (cfg.n_images, cfg.k_views, cfg.epochs, cfg.batch_size) == (100_000, 20, 100, 256)
    assert cfg.student_width == 24
    assert cfg.head_width == 2048
    # The optimizer is the committed Optuna winner once `tune optuna --apply` ran, else the fallback.
    summary = REPO_ROOT / "tuning" / "optuna_b.json"
    expected = (json.loads(summary.read_text())["winner"]["params"] if summary.exists()
                else {"lr": 1e-3, "weight_decay": 0.05, "warmup_epochs": 2})
    assert cfg.optimizer.lr == pytest.approx(expected["lr"])
    assert cfg.optimizer.weight_decay == pytest.approx(expected["weight_decay"])
    assert cfg.optimizer.warmup_epochs == expected["warmup_epochs"]
    assert cfg.optimizer.betas == (0.9, 0.999)
    # SVM settings are the committed CV selection once `probe --select-svm` ran, else null.
    svm = REPO_ROOT / "tuning" / "svm_selection.json"
    sel = json.loads(svm.read_text()) if svm.exists() else {"linear_c": None, "rbf_c": None, "rbf_gamma": None}
    assert (cfg.probe.linear_c, cfg.probe.rbf_c, cfg.probe.rbf_gamma) == (sel["linear_c"], sel["rbf_c"], sel["rbf_gamma"])


def test_smoke_profile_shrinks_images_views_and_epochs():
    cfg = load_config(row="C", size="L", profile="smoke")
    assert cfg.profile == "smoke"
    assert (cfg.n_images, cfg.k_views, cfg.epochs) == (2_000, 2, 2)
    assert cfg.student_width == 48 and cfg.batch_size == 256


def test_base_only_config_for_row_agnostic_stages():
    cfg = load_config()
    assert cfg.row is None and cfg.size is None
    assert cfg.n_images == 100_000


def test_config_is_frozen():
    cfg = load_config(row="B", size="S")
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.epochs = 5
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.optimizer.lr = 1.0


def _copy_configs(tmp_path: Path) -> Path:
    src = Path(__file__).resolve().parents[1] / "configs"
    dst = tmp_path / "configs"
    dst.mkdir()
    for f in src.glob("*.yaml"):
        (dst / f.name).write_text(f.read_text())
    return dst


@pytest.mark.parametrize(
    "file, extra",
    [
        ("base.yaml", "batchsize: 128\n"),  # typo of batch_size
        ("base.yaml", "probe:\n  linear_C: 1.0\n"),  # nested typo
        ("B-S.yaml", "lr: 1.0e-3\n"),  # belongs under optimizer
    ],
)
def test_unknown_keys_fail_loudly(tmp_path, file, extra):
    configs = _copy_configs(tmp_path)
    with open(configs / file, "a") as f:
        f.write(extra)
    with pytest.raises(ValueError, match="unknown keys"):
        load_config(row="B", size="S", configs_dir=configs)


def test_float_written_without_dot_is_rejected(tmp_path):
    configs = _copy_configs(tmp_path)
    text = (configs / "base.yaml").read_text()
    text = re.sub(r"(?m)^(\s*lr:).*$", r"\1 1e-3", text, count=1)  # PyYAML reads 1e-3 as a string
    (configs / "base.yaml").write_text(text)
    with pytest.raises(ValueError, match="optimizer.lr"):
        load_config(configs_dir=configs)


def test_reported_runs_follow_week3_order():
    from mo433.config import reported_runs

    ids = [run_id(*r) for r in reported_runs()]
    assert len(ids) == len(set(ids)) == 20
    assert ids[:4] == ["B-S-100-s0", "C-S-100-s0", "B-S-100-s1", "C-S-100-s1"]
    assert ids[6:8] == ["B-L-100-s0", "C-L-100-s0"]
    assert ids[10] == "B-S-050-s0"
    assert ids[-1] == "C-L-050-s1"


def test_run_id_format():
    assert run_id("C", "L", 50, 1) == "C-L-050-s1"


def test_parse_encoder_id_round_trips_run_id_and_reads_a_and_d():
    from mo433.config import EncoderId, parse_encoder_id, reported_runs

    for r in reported_runs():
        assert parse_encoder_id(run_id(*r)) == EncoderId(*r)
    assert parse_encoder_id("A-S-s0") == EncoderId("A", "S", None, 0)
    assert parse_encoder_id("D") == EncoderId("D", None, None, None)
    for bad in ("B-S-100-s0-fv-otf", "B-S-100-s0-otf", "X-S-s0", "B-S-100"):
        with pytest.raises(ValueError):
            parse_encoder_id(bad)


def test_dataset_overlay_applies_after_row_and_before_profile(monkeypatch):
    # EuroSAT pools its whole train split; smoke still shrinks everything.
    monkeypatch.setenv("MO433_DATASET", "eurosat")
    full = load_config("B", "S", "full")
    assert full.n_images == 21_600
    assert full.epochs == 100 and full.student_width == 24
    assert load_config("B", "S", "smoke").n_images == 2_000
    monkeypatch.setenv("MO433_DATASET", "stl10")
    assert load_config("B", "S", "full").n_images == 100_000
