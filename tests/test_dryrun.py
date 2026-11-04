"""Dry-run overhead factor (spec §8 slip rule, §10 W1) from run directories."""

import json

import pytest

from mo433 import dryrun


def _fake_run(path, run_id, *, wall, epochs, n_used=100000, setup=0.0):
    """epochs: list of (epoch_seconds, gpu_seconds, loader_wait_frac, knn_seconds)."""
    path.mkdir(parents=True)
    with open(path / "metrics.jsonl", "w") as f:
        for i, (es, gs, lw, ks) in enumerate(epochs, 1):
            f.write(json.dumps({"epoch": i, "epoch_seconds": es, "gpu_seconds": gs,
                                "loader_wait_frac": lw, "knn_seconds": ks}) + "\n")
    row, size, frac = run_id.split("-")[:3]
    (path / "summary.json").write_text(json.dumps({
        "run_id": run_id, "wall_seconds": wall, "gpu_seconds": sum(e[1] for e in epochs),
        "epochs_done": len(epochs),
        "settings": {"row": row, "size": size, "frac": int(frac), "n_images": n_used * 100 // int(frac),
                     "epochs": len(epochs), "batch_size": 256},
    }))
    return path


def test_overhead_of_one_run_is_wall_over_summed_gpu_step_time(tmp_path):
    # 2 epochs: 40 s and 30 s of GPU steps (70 s), 100 s wall in total -> 100 / 70.
    rd = _fake_run(tmp_path / "B-S-100-s0", "B-S-100-s0", wall=100.0,
                   epochs=[(45.0, 40.0, 0.10, 5.0), (35.0, 30.0, 0.20, 5.0)])
    o = dryrun.run_overhead(rd)
    assert o["run_id"] == "B-S-100-s0"
    assert o["gpu_seconds"] == pytest.approx(70.0)
    assert o["wall_seconds"] == pytest.approx(100.0)
    assert o["overhead"] == pytest.approx(100 / 70)
    # Epoch-loop overhead only (data waiting, logging): (45 + 35) / 70.
    assert o["epoch_overhead"] == pytest.approx(80 / 70)
    # Time-weighted loader wait: (0.10*45 + 0.20*35) / 80.
    assert o["loader_wait_frac"] == pytest.approx(11.5 / 80)
    assert o["knn_seconds"] == pytest.approx(10.0)
    # Setup / checkpoints / anything outside epochs and k-NN: 100 - 80 - 10.
    assert o["other_seconds"] == pytest.approx(10.0)
    # Steady-state GPU time per epoch excludes the warm-up first epoch.
    assert o["gpu_seconds_per_epoch"] == pytest.approx(30.0)


def test_summary_gives_overall_factor_and_verdict_against_the_slip_threshold(tmp_path):
    b = _fake_run(tmp_path / "b", "B-S-100-s0", wall=120.0, epochs=[(50.0, 50.0, 0.0, 0.0)] * 2)  # 1.2x
    c = _fake_run(tmp_path / "c", "C-S-100-s0", wall=80.0, epochs=[(25.0, 25.0, 0.3, 0.0)] * 2)  # 1.6x
    s = dryrun.summarize([dryrun.run_overhead(b), dryrun.run_overhead(c)])
    assert s["overall_overhead"] == pytest.approx(200 / 150)
    assert s["threshold"] == 1.3
    assert s["exceeds_threshold"] is True
    assert "1.33" in s["verdict"] and "slip" in s["verdict"]

    ok = dryrun.summarize([dryrun.run_overhead(b)])
    assert ok["exceeds_threshold"] is False


def test_projection_of_the_20_runs_from_measured_gpu_time_per_epoch(tmp_path):
    # B-S: 100 s GPU/epoch, C-S: 50 s GPU/epoch on 100k images; 1.0x overhead.
    b = _fake_run(tmp_path / "b", "B-S-100-s0", wall=300.0, epochs=[(100.0, 100.0, 0, 0)] * 3)
    c = _fake_run(tmp_path / "c", "C-S-100-s0", wall=150.0, epochs=[(50.0, 50.0, 0, 0)] * 3)
    s = dryrun.summarize([dryrun.run_overhead(b), dryrun.run_overhead(c)], epochs=100, n_images=100000,
                         l_over_s=2.0)
    p = s["projection"]
    # Per seed and fraction: S = (100 + 50) * 100 epochs; L = 2x that. Seeds: 3 S, 2 L;
    # fractions 100% + 50% = 1.5x.  S: 15000 * 3 * 1.5 = 67500 s; L: 30000 * 2 * 1.5 = 90000 s.
    assert p["gpu_hours"] == pytest.approx((67500 + 90000) / 3600)
    assert p["wall_hours"] == pytest.approx(p["gpu_hours"] * s["overall_overhead"])
    assert p["l_source"].startswith("S x 2.0")
    assert p["spec_gpu_hours"] == 18.7
    assert len(p["runs"]) == 20
