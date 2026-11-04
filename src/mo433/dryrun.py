"""Week-1 full-scale dry run (spec §8 slip rule, §10 W1): ``mo433 tune dryrun``.

Trains B-S and C-S (frac 100, seed 0) for 5 epochs on the real pool (N=100k, K=20) with the
optimizer currently in ``configs/`` (the fallback, or the Optuna winner once ``--apply``-ed)
and the train-all defaults (validation k-NN every epoch, default loader workers), so the
measured overhead is the one the 20 reported runs will pay.

**Overhead factor** = run wall time / summed GPU step time (CUDA events around every
forward + backward + optimizer step, ``gpu_seconds`` in ``metrics.jsonl``), per run and
overall (Σ wall / Σ GPU). The verdict compares the overall factor with the 1.3× of the spec
§8 budget (≈18.7 GPU-h ≈ 24 wall-h): above it, the slip-rule decision is raised now instead
of at the 15 Nov gate. Both are only meaningful on an otherwise idle GPU.

The breakdown says where the overhead goes: ``epoch_overhead`` (epoch loop wall / GPU: data
waiting, host-side logging), ``loader_wait_frac`` (share of epoch wall spent waiting for a
batch), ``knn_seconds`` (per-epoch validation k-NN) and ``other_seconds`` (setup,
checkpoints, the final backbone save). Setup is a one-time cost, so the 5-epoch factor
slightly overstates a 100-epoch run's; ``amortized_overhead`` projects it to E epochs.

**Projection** of the 20 runs: steady-state GPU seconds per epoch (median over epochs 2..,
the first epoch carries cudnn autotuning and worker start-up) per image, scaled to each run's
images (N × frac) and E epochs. Sizes not measured (by default L) use S × ``L_OVER_S``, the
b256 train-throughput ratio of ResNet96 w=24 vs w=48 measured in
docs/student-design/README.md §3 (6,936 / 2,765 img/s).

Outputs: run dirs under ``$MO433_ROOT/tuning/dryrun/<run_id>/`` and the summary
``tuning/dryrun.json`` (also copied into the committed ``tuning/`` for the full profile).
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from mo433 import config as config_mod

THRESHOLD = 1.3  # spec §8: 18.7 GPU-h -> about 24 wall-h at 1.3x
SPEC_GPU_HOURS = 18.7
SPEC_WALL_HOURS = 24.0
DRYRUN_EPOCHS = 5
L_OVER_S = round(6936 / 2765, 3)  # docs/student-design/README.md §3, b256 train img/s
SUMMARY_NAME = "dryrun.json"


def _read_metrics(run_dir: Path) -> list[dict]:
    lines = (Path(run_dir) / "metrics.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def run_overhead(run_dir: Path | str) -> dict[str, Any]:
    """Overhead breakdown of one finished run directory (``summary.json`` + ``metrics.jsonl``)."""
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text())
    metrics = _read_metrics(run_dir)
    if not metrics:
        raise ValueError(f"{run_dir}: empty metrics.jsonl")
    st = summary["settings"]
    gpu = sum(m["gpu_seconds"] for m in metrics)
    epoch_wall = sum(m["epoch_seconds"] for m in metrics)
    knn = sum(m.get("knn_seconds", 0.0) for m in metrics)
    wall = summary["wall_seconds"]
    steady = [m["gpu_seconds"] for m in (metrics[1:] if len(metrics) > 1 else metrics)]
    n_used = st["n_images"] * st["frac"] // 100
    return {
        "run_id": summary["run_id"],
        "row": st["row"],
        "size": st["size"],
        "epochs": len(metrics),
        "n_used": n_used,
        "wall_seconds": wall,
        "gpu_seconds": gpu,
        "overhead": wall / gpu,
        "epoch_overhead": epoch_wall / gpu,
        "loader_wait_frac": sum(m["loader_wait_frac"] * m["epoch_seconds"] for m in metrics) / epoch_wall,
        "knn_seconds": knn,
        "other_seconds": wall - epoch_wall - knn,
        "gpu_seconds_per_epoch": statistics.median(steady),
        "peak_mem_gb": summary.get("peak_mem_gb"),
    }


def _amortized(o: dict, epochs: int) -> float:
    """The run's factor at ``epochs`` epochs: one-time ``other_seconds``, the rest per epoch."""
    per_epoch_wall = (o["wall_seconds"] - o["other_seconds"]) / o["epochs"]
    per_epoch_gpu = o["gpu_seconds"] / o["epochs"]
    return (o["other_seconds"] + epochs * per_epoch_wall) / (epochs * per_epoch_gpu)


def project(per_run: list[dict], *, epochs: int, n_images: int, overhead: float,
            l_over_s: float = L_OVER_S) -> dict[str, Any]:
    """GPU and wall hours of the 20 reported runs from measured GPU seconds per epoch-image."""
    per_image = {(o["row"], o["size"]): o["gpu_seconds_per_epoch"] / o["n_used"] for o in per_run}
    sources = {}
    for row in config_mod.ROWS:
        for size in config_mod.SIZES:
            if (row, size) in per_image:
                sources[(row, size)] = "measured"
            elif (row, "S") in per_image and size == "L":
                per_image[(row, "L")] = per_image[(row, "S")] * l_over_s
                sources[(row, size)] = f"S x {l_over_s}"
    runs = []
    for row, size, frac, seed in config_mod.reported_runs():
        if (row, size) not in per_image:
            raise ValueError(f"no measurement for row {row} (size {size}) to project from")
        gpu_s = per_image[(row, size)] * (n_images * frac // 100) * epochs
        runs.append({"run_id": config_mod.run_id(row, size, frac, seed), "gpu_hours": gpu_s / 3600,
                     "wall_hours": gpu_s * overhead / 3600, "source": sources[(row, size)]})
    gpu_h = sum(r["gpu_hours"] for r in runs)
    l_sources = sorted({sources[(r, "L")] for r in config_mod.ROWS})
    return {
        "epochs": epochs,
        "n_images": n_images,
        "overhead": overhead,
        "gpu_hours": gpu_h,
        "wall_hours": gpu_h * overhead,
        "spec_gpu_hours": SPEC_GPU_HOURS,
        "spec_wall_hours": SPEC_WALL_HOURS,
        "l_source": ", ".join(l_sources),
        "longest_run": max(runs, key=lambda r: r["gpu_hours"]),
        "runs": runs,
    }


def summarize(per_run: list[dict], *, epochs: int | None = None, n_images: int | None = None,
              l_over_s: float = L_OVER_S, threshold: float = THRESHOLD) -> dict[str, Any]:
    """Overall factor (Σ wall / Σ GPU), verdict vs ``threshold`` and, with ``epochs`` and
    ``n_images``, the projection of the 20 runs."""
    wall = sum(o["wall_seconds"] for o in per_run)
    gpu = sum(o["gpu_seconds"] for o in per_run)
    factor = wall / gpu
    exceeds = factor > threshold
    out: dict[str, Any] = {
        "runs": per_run,
        "overall_overhead": factor,
        "overall_loader_wait_frac": (
            sum(o["loader_wait_frac"] * o["epoch_overhead"] * o["gpu_seconds"] for o in per_run)
            / sum(o["epoch_overhead"] * o["gpu_seconds"] for o in per_run)
        ),
        "threshold": threshold,
        "exceeds_threshold": exceeds,
        "verdict": (
            f"overhead {factor:.2f}x > {threshold}x: raise the slip-rule decision now (spec §8), "
            "not at the 15 Nov gate" if exceeds else
            f"overhead {factor:.2f}x <= {threshold}x: within the spec §8 budget; no early slip decision"
        ),
    }
    if epochs is not None and n_images is not None:
        out["amortized_overhead"] = {o["run_id"]: _amortized(o, epochs) for o in per_run}
        out["projection"] = project(per_run, epochs=epochs, n_images=n_images, overhead=factor, l_over_s=l_over_s)
    return out


def format_report(s: dict) -> str:
    lines = ["[dryrun] run           epochs  wall s   GPU s  overhead  epoch-loop  loader-wait  kNN s  other s"]
    for o in s["runs"]:
        lines.append(
            f"[dryrun] {o['run_id']:12s} {o['epochs']:6d} {o['wall_seconds']:7.0f} {o['gpu_seconds']:7.0f}"
            f"  {o['overhead']:7.3f}x  {o['epoch_overhead']:9.3f}x  {o['loader_wait_frac']:10.1%}"
            f"  {o['knn_seconds']:5.0f}  {o['other_seconds']:7.0f}"
        )
    lines.append(f"[dryrun] overall overhead {s['overall_overhead']:.3f}x "
                 f"(loader wait {s['overall_loader_wait_frac']:.1%})")
    if "amortized_overhead" in s:
        p = s["projection"]
        am = ", ".join(f"{k} {v:.3f}x" for k, v in s["amortized_overhead"].items())
        lines.append(f"[dryrun] amortized to {p['epochs']} epochs: {am}")
        lines.append(
            f"[dryrun] projected 20 runs: {p['gpu_hours']:.1f} GPU-h, {p['wall_hours']:.1f} wall-h "
            f"(spec: {p['spec_gpu_hours']} GPU-h, {p['spec_wall_hours']:.0f} wall-h); L from {p['l_source']}; "
            f"longest {p['longest_run']['run_id']} {p['longest_run']['wall_hours']:.1f} wall-h"
        )
    lines.append(f"[dryrun] VERDICT: {s['verdict']}")
    return "\n".join(lines)


def run_dryrun(profile: str = "full", *, epochs: int | None = None, sizes: Iterable[str] = ("S",),
               overrides: dict[str, Any] | None = None, force: bool = False,
               log: Callable[[str], None] = print) -> dict[str, Any]:
    """Train (or resume) the dry-run runs, then write and print the overhead summary."""
    from mo433 import paths, train, tune

    full_cfg = config_mod.load_config("B", "S", profile)
    epochs = epochs or (DRYRUN_EPOCHS if profile == "full" else full_cfg.epochs)
    root = paths.tuning_dir(profile) / "dryrun"
    per_run = []
    for size in sizes:
        for row in config_mod.ROWS:
            rid = config_mod.run_id(row, size, 100, 0)
            rd = root / rid
            train.train_run(row, size, 100, 0, profile=profile, overrides=overrides, run_dir=rd,
                            epochs=epochs, fresh=force, log=log)
            per_run.append(run_overhead(rd))
    s = summarize(per_run, epochs=full_cfg.epochs, n_images=full_cfg.n_images)
    s |= {"profile": profile, "dryrun_epochs": epochs, "optimizer": tune._asdict(full_cfg.optimizer),
          "overrides": overrides or {}, "gpu": _gpu_name()}
    tune._publish(profile, SUMMARY_NAME, s)
    log(format_report(s))
    return s


def _gpu_name() -> str | None:
    import torch

    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
