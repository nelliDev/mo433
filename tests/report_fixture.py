"""Synthetic report inputs: every encoder of the design, every probe, fraction and draw."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from mo433 import config, report

BASE = {"A": 0.30, "B": 0.55, "C": 0.62, "D": 0.85}
FRAC_BONUS = {1: -0.15, 10: -0.05, 50: 0.0, 100: 0.02}
PROBE_BONUS = {"knn": -0.02, "linear_svm": 0.0, "rbf_svm": 0.01}


def write_csv(path: Path, columns, rows) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return path


def results_rows(encoders=None, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    encoders = encoders or report.expected_encoders()
    rows = []
    for eid in encoders:
        key = report.parse_encoder(eid)
        base = BASE[key.row] + (0.03 if key.size == "L" else 0.0) - (0.02 if key.frac == 50 else 0.0)
        feature_sets = ("main",) if key.row == "D" else report.FEATURE_SETS
        for fs in feature_sets:
            for lf in report.LABEL_FRACS:
                for draw in range(3):
                    for probe in report.PROBES:
                        acc = base + FRAC_BONUS[lf] + PROBE_BONUS[probe] + rng.normal(0, 0.01)
                        rows.append(
                            {
                                "encoder_id": eid,
                                "feature_set": fs,
                                "label_frac": lf,
                                "draw": draw,
                                "probe": probe,
                                "accuracy": round(float(np.clip(acc, 0, 1)), 4),
                                "n_train": 50 * lf,
                                "k": 5 if probe == "knn" else "",
                                "C": "" if probe == "knn" else 1.0,
                                "gamma": "scale" if probe == "rbf_svm" else "",
                                "svm_params_frozen": "" if probe == "knn" else True,
                                "git_commit": "abc",
                                "timestamp": "2026-11-20T00:00:00",
                            }
                        )
    return rows


RESULTS_HEADER = (
    "encoder_id", "feature_set", "label_frac", "draw", "probe", "accuracy", "n_train", "k", "C",
    "gamma", "svm_params_frozen", "git_commit", "timestamp",
)  # the #16 header


def write_inputs(root: Path, encoders=None, *, costs=True, runs_cost=True, diagnostics=True, tsne=True):
    inputs = report.Inputs.committed(root)
    write_csv(inputs.results, RESULTS_HEADER, results_rows(encoders))
    if costs:
        write_csv(
            inputs.costs,
            report.COSTS_COLUMNS,
            [
                dict(encoder_id="S", params=1_570_000, flops=2.0e9, macs=1.0e9, feature_dim=192,
                     gpu_latency_ms=40.0, cpu_latency_ms=6.0),
                dict(encoder_id="L", params=6_280_000, flops=8.0e9, macs=4.0e9, feature_dim=384,
                     gpu_latency_ms=120.0, cpu_latency_ms=15.0),
                dict(encoder_id="D", params=21_670_000, flops=8.6e9, macs=4.3e9, feature_dim=384,
                     gpu_latency_ms=900.0, cpu_latency_ms=40.0),
            ],
        )
    if runs_cost:
        rows = []
        for row, size, frac, seed in config.reported_runs():
            gpu = (0.8 if size == "S" else 2.5) * (0.5 if frac == 50 else 1.0) * (1.0 if row == "B" else 0.7)
            rows.append(dict(run_id=config.run_id(row, size, frac, seed), wall_h=gpu * 1.3, gpu_h=gpu,
                             teacher_cache_gpu_h_charged=0.5 if row == "C" else 0.0))
        write_csv(inputs.runs_cost, report.RUNS_COST_COLUMNS, rows)
    if diagnostics:
        rows = []
        for eid in encoders or report.expected_encoders():
            key = report.parse_encoder(eid)
            rows.append(dict(
                encoder_id=eid, val_knn=BASE[key.row], val_spread=0.8, val_nmi=BASE[key.row] - 0.2,
                test_nmi=BASE[key.row] - 0.2, test_ari=BASE[key.row] - 0.3,
                collapse_flagged=(eid == "B-L-050-s1") if key.row in "BC" else "",
                teacher_passed=True if key.row == "D" else "",
            ))
        write_csv(inputs.diagnostics, report.DIAGNOSTICS_COLUMNS, rows)
    if tsne:
        rng = np.random.default_rng(0)
        arrays = {}
        for eid in report.TSNE_PANELS:
            arrays[f"{eid}__xy"] = rng.normal(size=(200, 2)).astype(np.float32)
            arrays[f"{eid}__y"] = rng.integers(0, 10, 200).astype(np.uint8)
        inputs.tsne.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(inputs.tsne, **arrays)
    return inputs
