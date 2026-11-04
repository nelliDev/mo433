"""Training-loop seam: ``train_run`` end to end on a tiny CPU setup (spec §5.1, §9).

A fake view pool (random uint8, the real on-disk format) stands in for ``mo433 pool``.
"""

import json

import numpy as np
import pytest
import torch

from mo433 import paths
from mo433.config import load_config
from mo433.train import train_run

TINY = dict(
    n_images=64, batch_size=16, epochs=4, warmup_epochs=1, head_width=32,
    knn_every=0, num_workers=0, device="cpu", checkpoint_every=2,
)  # 4 steps per epoch, 16 steps in all


@pytest.fixture
def tiny_pool(mo433_root):
    out = paths.pool_dir("smoke")
    out.mkdir(parents=True)
    rng = np.random.default_rng(0)
    views = np.lib.format.open_memmap(out / "views.npy", mode="w+", dtype=np.uint8, shape=(64, 2, 96, 96, 3))
    views[:] = rng.integers(0, 256, views.shape, dtype=np.uint8)
    views.flush()
    (out / "meta.json").write_text(json.dumps({"n_images": 64, "k_views": 2, "shape": [64, 2, 96, 96, 3], "dtype": "uint8"}))
    return out


def _jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


class Stop(Exception):
    pass


def test_smoke_run_completes_and_writes_the_run_directory(tiny_pool):
    summary = train_run("B", "S", 100, 0, profile="smoke", overrides=TINY)
    rd = paths.run_dir("B-S-100-s0", "smoke")

    assert summary["epochs_done"] == 4 and summary["steps_done"] == 16
    assert {"config.json", "metrics.jsonl", "checkpoint.pt", "final_backbone.pt", "summary.json"} <= {
        p.name for p in rd.iterdir()
    }
    cfg = json.loads((rd / "config.json").read_text())
    assert cfg["run_id"] == "B-S-100-s0" and cfg["settings"]["lr"] == load_config("B", "S", "smoke").optimizer.lr and cfg["settings"]["head_width"] == 32
    assert "git_commit" in cfg and "git_dirty" in cfg

    log = _jsonl(rd / "metrics.jsonl")
    assert [r["epoch"] for r in log] == [1, 2, 3, 4]
    for key in ("loss", "invariance", "variance", "covariance", "emb_std", "lr", "step", "epoch_seconds", "gpu_seconds"):
        assert key in log[0], key
    assert all(np.isfinite(r["loss"]) for r in log)
    assert log[-1]["lr"] == pytest.approx(0.0, abs=1e-12)  # cosine reached 0 at the last step

    saved = json.loads((rd / "summary.json").read_text())
    assert saved["wall_seconds"] > 0 and saved["gpu_seconds"] > 0

    from mo433.students import build_student

    net = build_student("S", 0)
    net.load_state_dict(torch.load(rd / "final_backbone.pt", weights_only=True))  # backbone only, loads cleanly


def test_killed_then_resumed_run_continues_the_schedule_and_trajectory(tiny_pool, tmp_path):
    ref_dir, cut_dir = tmp_path / "ref", tmp_path / "cut"
    train_run("B", "S", 100, 0, profile="smoke", overrides=TINY, run_dir=ref_dir)

    def kill_at_3(epoch, metrics):
        if epoch == 3:
            raise Stop

    with pytest.raises(Stop):  # an epoch_callback exception (e.g. an Optuna prune) propagates
        train_run("B", "S", 100, 0, profile="smoke", overrides=TINY, run_dir=cut_dir, epoch_callback=kill_at_3)
    assert not (cut_dir / "final_backbone.pt").exists()

    seen = []
    summary = train_run("B", "S", 100, 0, profile="smoke", overrides=TINY, run_dir=cut_dir,
                        epoch_callback=lambda e, m: seen.append(e))
    assert seen == [3, 4]  # resumed from the epoch-2 checkpoint
    assert summary["epochs_done"] == 4 and summary["steps_done"] == 16

    ref, cut = _jsonl(ref_dir / "metrics.jsonl"), _jsonl(cut_dir / "metrics.jsonl")
    assert [r["epoch"] for r in cut] == [1, 2, 3, 4]
    assert [r["step"] for r in cut] == [r["step"] for r in ref]
    assert [r["lr"] for r in cut] == pytest.approx([r["lr"] for r in ref], rel=1e-12)
    assert [r["loss"] for r in cut] == pytest.approx([r["loss"] for r in ref], rel=1e-4)

    a = torch.load(ref_dir / "final_backbone.pt", weights_only=True)
    b = torch.load(cut_dir / "final_backbone.pt", weights_only=True)
    assert all(torch.allclose(a[k].float(), b[k].float(), rtol=1e-3, atol=1e-5) for k in a)


def test_overrides_and_epochs_argument_shape_the_run(tiny_pool, tmp_path):
    seen = []
    summary = train_run("B", "S", 50, 1, profile="smoke", overrides={**TINY, "lr": 3e-3}, epochs=2,
                        run_dir=tmp_path / "r", epoch_callback=lambda e, m: seen.append(m))
    assert summary["epochs_done"] == 2
    assert summary["steps_done"] == 2 * (32 // 16)  # frac 50 -> pool images [0, 32)
    assert json.loads((tmp_path / "r" / "config.json").read_text())["settings"]["lr"] == 3e-3
    assert seen[-1]["lr"] == pytest.approx(0.0, abs=1e-12)


@pytest.fixture
def smoke_pool(mo433_root):
    """A fake pool of the smoke profile's size (N=2000, K=2)."""
    out = paths.pool_dir("smoke")
    out.mkdir(parents=True)
    views = np.lib.format.open_memmap(out / "views.npy", mode="w+", dtype=np.uint8, shape=(2000, 2, 96, 96, 3))
    views[:] = np.random.default_rng(0).integers(0, 256, views.shape, dtype=np.uint8)
    views.flush()
    (out / "meta.json").write_text(json.dumps({"n_images": 2000, "k_views": 2, "shape": [2000, 2, 96, 96, 3]}))
    return out


@pytest.mark.gpu
@pytest.mark.data
def test_cli_smoke_profile_trains_with_validation_knn_then_skips_until_forced(smoke_pool, capsys):
    from mo433.cli import main
    from mo433.encoders import load_encoder

    argv = ["--profile", "smoke", "train", "--row", "B", "--size", "S", "--frac", "100", "--seed", "0"]
    assert main(argv) == 0
    log = _jsonl(paths.run_dir("B-S-100-s0", "smoke") / "metrics.jsonl")
    assert [r["epoch"] for r in log] == [1, 2]
    assert all(0.0 <= r["val_knn"] <= 1.0 and r["val_emb_std"] > 0 and r["peak_mem_gb"] > 0 for r in log)
    assert load_encoder("B-S-100-s0", "smoke").dim == 192  # the registry finds the final weights

    assert main(argv) == 0
    assert "skipping" in capsys.readouterr().out
    assert main(argv + ["--force"]) == 0
    assert len(_jsonl(paths.run_dir("B-S-100-s0", "smoke") / "metrics.jsonl")) == 2  # fresh, not appended
