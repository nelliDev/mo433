"""Tuning protocol (spec §6): winner selection, the tie rule, and the smoke Optuna study."""

import json

import numpy as np
import pytest

from mo433 import config as config_mod
from mo433 import paths
from mo433 import tune


# ---------------------------------------------------------------- winner selection


def test_winner_is_the_best_unflagged_complete_trial():
    trials = [
        {"number": 0, "state": "COMPLETE", "value": 0.62, "collapse_flagged": False},
        {"number": 1, "state": "COMPLETE", "value": 0.70, "collapse_flagged": True},  # best, but collapsed
        {"number": 2, "state": "PRUNED", "value": None, "collapse_flagged": None},
        {"number": 3, "state": "COMPLETE", "value": 0.65, "collapse_flagged": False},
    ]
    assert tune.select_winner(trials)["number"] == 3


def test_no_winner_when_every_complete_trial_collapsed():
    trials = [{"number": 0, "state": "COMPLETE", "value": 0.5, "collapse_flagged": True}]
    assert tune.select_winner(trials) is None


# ---------------------------------------------------------------- tie rule


def test_tie_rule_keeps_2048_when_the_margin_is_within_the_seed_spread():
    # 4096 mean beats 2048 by 0.01, but 2048's seeds differ by 0.02 -> not enough.
    out = tune.tie_rule({1024: [0.60, 0.61], 2048: [0.64, 0.66], 4096: [0.655, 0.665]})
    assert out["winner"] == 2048
    assert out["candidates"][4096]["beats_baseline"] is False


def test_tie_rule_switches_when_the_margin_exceeds_both_spreads():
    # 4096 mean 0.70 vs 2048 mean 0.65; spreads 0.01 and 0.02 -> margin 0.05 > 0.02.
    out = tune.tie_rule({1024: [0.60, 0.61], 2048: [0.64, 0.66], 4096: [0.695, 0.705]})
    assert out["winner"] == 4096
    assert out["candidates"][4096]["spread"] == pytest.approx(0.01)
    assert out["candidates"][4096]["threshold"] == pytest.approx(0.02)


def test_tie_rule_picks_the_higher_mean_when_two_widths_qualify():
    out = tune.tie_rule({1024: [0.70, 0.70], 2048: [0.60, 0.60], 4096: [0.75, 0.75]})
    assert out["winner"] == 4096


def test_tie_rule_ignores_widths_with_a_collapsed_seed():
    # 4096 would win, but one of its seeds is collapse-flagged (None) -> ineligible.
    out = tune.tie_rule({1024: [0.60, 0.61], 2048: [0.64, 0.66], 4096: [0.80, None]})
    assert out["winner"] == 2048
    assert out["candidates"][4096]["eligible"] is False


# ---------------------------------------------------------------- the smoke study

TINY = dict(n_images=64, batch_size=16, head_width=32, knn_every=0, num_workers=0, device="cpu",
            checkpoint_every=1)


@pytest.fixture
def tiny_pool(mo433_root):
    out = paths.pool_dir("smoke")
    out.mkdir(parents=True)
    views = np.lib.format.open_memmap(out / "views.npy", mode="w+", dtype=np.uint8, shape=(64, 2, 96, 96, 3))
    views[:] = np.random.default_rng(0).integers(0, 256, views.shape, dtype=np.uint8)
    views.flush()
    (out / "meta.json").write_text(json.dumps({"n_images": 64, "k_views": 2, "shape": [64, 2, 96, 96, 3]}))
    return out


@pytest.mark.data
def test_smoke_optuna_completes_and_resumes(tiny_pool, mo433_root):
    summary = tune.run_optuna(profile="smoke", n_trials=2, epochs=1, overrides=TINY)

    assert [t["state"] for t in summary["trials"]] == ["COMPLETE", "COMPLETE"]
    # trial 0 = the smoke config = the fallback, pinned by spec §6.1 (base.yaml holds the winner after --apply)
    assert summary["trials"][0]["params"] == {"lr": 1e-3, "weight_decay": 0.05, "warmup_epochs": 2}
    assert all(0.0 <= t["value"] <= 1.0 for t in summary["trials"])
    assert summary["winner"]["number"] in (0, 1)
    assert json.loads((mo433_root / "tuning" / "optuna_b.json").read_text())["winner"] == summary["winner"]
    assert (mo433_root / "tuning" / "optuna_b.db").exists()
    trial0 = mo433_root / "tuning" / "optuna" / "trial_0"
    assert (trial0 / "final_backbone.pt").exists() and not (trial0 / "checkpoint.pt").exists()

    again = tune.run_optuna(profile="smoke", n_trials=2, epochs=1, overrides=TINY)  # nothing left to do
    assert len(again["trials"]) == 2

    more = tune.run_optuna(profile="smoke", n_trials=3, epochs=1, overrides=TINY)  # continues the study
    assert [t["number"] for t in more["trials"]] == [0, 1, 2]
    assert more["trials"][0]["value"] == summary["trials"][0]["value"]


class Crash(Exception):
    pass


@pytest.mark.data
def test_interrupted_trial_is_retried_with_the_same_params_and_resumes(tiny_pool, mo433_root):
    def crash_in_epoch_2(msg):
        if "epoch=2 " in msg and "resumed" not in msg:
            raise Crash

    with pytest.raises(Crash):
        tune.run_optuna(profile="smoke", n_trials=1, epochs=2, overrides=TINY, log=crash_in_epoch_2)

    lines = []
    summary = tune.run_optuna(profile="smoke", n_trials=1, epochs=2, overrides=TINY, log=lines.append)
    done = [t for t in summary["trials"] if t["state"] == "COMPLETE"]
    assert len(done) == 1 and done[0]["number"] == 1 and done[0]["retry_of"] == 0
    assert done[0]["params"] == summary["trials"][0]["params"]
    assert any("resumed at epoch 1" in line for line in lines)
    assert isinstance(done[0]["collapse_flagged"], bool)


# ---------------------------------------------------------------- grid, LARS, fixed-view, --apply

GRID_TINY = {k: v for k, v in TINY.items() if k != "head_width"}


def _settings(run_dir):
    return json.loads((run_dir / "config.json").read_text())["settings"]


@pytest.mark.data
def test_smoke_grid_uses_the_optuna_winner_and_applies_the_tie_rule(tiny_pool, mo433_root):
    winner = {"number": 3, "params": {"lr": 2e-3, "weight_decay": 0.1, "warmup_epochs": 1}, "value": 0.7}
    (mo433_root / "tuning").mkdir(parents=True)
    (mo433_root / "tuning" / "optuna_b.json").write_text(json.dumps({"winner": winner}))

    out = tune.run_grid(profile="smoke", rows=["B"], epochs=1, overrides=GRID_TINY)["B"]

    assert sorted((p["width"], p["seed"]) for p in out["points"]) == [
        (w, s) for w in (1024, 2048, 4096) for s in (0, 1)
    ]
    s = _settings(mo433_root / "tuning" / "grid" / "B-w4096-s1")
    assert (s["head_width"], s["seed"], s["lr"], s["weight_decay"], s["warmup_epochs"]) == (4096, 1, 2e-3, 0.1, 1)
    assert out["optimizer_source"] == "optuna winner (trial 3)"
    assert out["tie_rule"]["winner"] == out["winner"] in (1024, 2048, 4096)
    assert json.loads((mo433_root / "tuning" / "grid_B.json").read_text())["winner"] == out["winner"]


@pytest.mark.data
def test_smoke_lars_runs_six_configs_with_linearly_scaled_lr(tiny_pool, mo433_root):
    out = tune.run_lars(profile="smoke", epochs=1, overrides=GRID_TINY)

    assert sorted((t["base_lr"], t["weight_decay"]) for t in out["trials"]) == [
        (lr, wd) for lr in (0.2, 1.0, 2.0) for wd in (1e-6, 1e-4)
    ]
    t = next(t for t in out["trials"] if t["base_lr"] == 1.0 and t["weight_decay"] == 1e-4)
    s = _settings(mo433_root / "tuning" / "lars" / t["name"])
    assert s["optimizer"] == "lars" and s["lr"] == pytest.approx(1.0 * 16 / 256) and s["head_width"] == 2048


@pytest.mark.data
def test_smoke_fixed_view_scores_both_twins_on_validation_and_keeps_the_test_delta(tiny_pool, mo433_root):
    out = tune.run_fixed_view(profile="smoke", epochs=1, overrides=TINY)

    pool, otf = out["pool"], out["on_the_fly"]
    assert 0 <= pool["val_knn"] <= 1 and 0 <= otf["val_knn"] <= 1
    assert out["delta_val_pp"] == pytest.approx(100 * (otf["val_knn"] - pool["val_knn"]))
    assert out["switch"] == (out["delta_val_pp"] > 2.0)
    assert "knn" in out["delta_test_pp"]
    assert _settings(mo433_root / "tuning" / "fixed_view" / "otf")["on_the_fly"] is True
    assert out["pool_twin_is_main_run"] is False  # overrides given: never the main run


def test_apply_writes_the_winners_into_the_configs(tmp_path, mo433_root):
    import shutil

    cfg = tmp_path / "configs"
    shutil.copytree(paths.CONFIGS_DIR, cfg)
    t = mo433_root / "tuning"
    t.mkdir(parents=True)
    (t / "optuna_b.json").write_text(json.dumps(
        {"winner": {"number": 5, "params": {"lr": 0.0021, "weight_decay": 0.031, "warmup_epochs": 4}, "value": 0.7}}))
    (t / "grid_B.json").write_text(json.dumps({"winner": 4096}))
    (t / "grid_C.json").write_text(json.dumps({"winner": 1024}))

    tune.apply_winners("optuna", profile="smoke", configs_dir=cfg)
    tune.apply_winners("grid", profile="smoke", configs_dir=cfg)

    for size in ("S", "L"):
        b = config_mod.load_config("B", size, configs_dir=cfg)
        c = config_mod.load_config("C", size, configs_dir=cfg)
        assert (b.optimizer.lr, b.optimizer.weight_decay, b.optimizer.warmup_epochs) == (0.0021, 0.031, 4)
        assert (b.head_width, c.head_width) == (4096, 1024)
    assert "# Shared settings" in (cfg / "base.yaml").read_text()  # comments kept
