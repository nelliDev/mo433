"""Extra experiments run after the frozen study (``configs-frozen``), reported apart from it.

Each study trains B/C runs with the frozen settings plus one declared change, then probes the
final backbones with the unchanged probes and frozen SVMs (spec §7). Nothing here touches
the 20 reported runs, ``results.csv`` or ``artifacts/runs``:

- ``long300``: the S student, B and C, seeds 0-1, at E = 300 instead of 100. Tests the brief's
  "direct training catches up as the epochs grow" (Q6) directly.
- ``sizes``: two more student sizes, XS (w=12) and M (w=34), by the same width rule as S/L,
  B and C, seeds 0-1, 100% data, E = 100. Adds points to the gap-vs-parameters plot (Q5).

Layout: run dirs and features under ``<artifacts root>/extras/<study>/``; committed outputs
``report/extras/<study>.csv`` (probe rows, ``RESULTS_COLUMNS`` plus ``study``/``epochs``) and
``report/extras/<study>_curves.csv`` (per-epoch loss and validation k-NN, including the
matching main runs as the 100-epoch reference).
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mo433 import config, paths

CURVE_COLUMNS = ("study", "run_id", "epochs", "epoch", "loss", "val_knn")


@dataclass(frozen=True)
class ExtraRun:
    row: str
    size: str
    frac: int
    seed: int
    epochs: int | None = None  # None = the frozen E

    @property
    def run_id(self) -> str:
        return config.run_id(self.row, self.size, self.frac, self.seed)


@dataclass(frozen=True)
class Study:
    name: str
    runs: tuple[ExtraRun, ...]
    reference_runs: tuple[str, ...] = ()  # main runs whose curves are the baseline


STUDIES = {
    s.name: s
    for s in (
        Study("long300", tuple(ExtraRun(row, "S", 100, seed, 300) for seed in (0, 1) for row in config.ROWS),
              reference_runs=tuple(config.run_id(row, "S", 100, seed) for seed in (0, 1) for row in config.ROWS)),
        Study("sizes", tuple(ExtraRun(row, size, 100, seed)
                             for size in config.EXTRA_SIZES for seed in (0, 1) for row in config.ROWS)),
    )
}


def study_dir(study: str, profile: str = "full") -> Path:
    return paths.artifacts_root(profile) / "extras" / study


def run_dir(study: str, run: ExtraRun, profile: str = "full") -> Path:
    return study_dir(study, profile) / run.run_id


def output_dir() -> Path:
    return paths.report_dir() / "extras"


# ---------------------------------------------------------------- curves


def curve_rows(rd: Path, *, run_id: str, study: str, epochs: int) -> list[dict]:
    """Per-epoch ``loss`` and ``val_knn`` from ``rd/metrics.jsonl``."""
    rows = []
    for line in (Path(rd) / "metrics.jsonl").read_text().splitlines():
        if line.strip():
            m = json.loads(line)
            rows.append({"study": study, "run_id": run_id, "epochs": epochs, "epoch": m["epoch"],
                         "loss": m.get("loss"), "val_knn": m.get("val_knn")})
    return rows


# ---------------------------------------------------------------- evaluation


def _features(study: str, run: ExtraRun, profile: str) -> dict[str, np.ndarray]:
    """Spec §7.1 features of the run's final backbone on the labeled train and test splits."""
    import torch

    from mo433 import extract
    from mo433.data import load_stl10
    from mo433.encoders import FINAL_BACKBONE, student_encoder
    from mo433.students import WIDTHS, ResNet96

    out = study_dir(study, profile) / "features" / f"{run.run_id}.npz"
    if not out.exists():
        net = ResNet96(WIDTHS[run.size])
        net.load_state_dict(torch.load(run_dir(study, run, profile) / FINAL_BACKBONE, map_location="cpu",
                                       weights_only=True))
        enc = student_encoder(run.run_id, net).to("cuda" if torch.cuda.is_available() else "cpu")
        arrays = {"encoder_id": np.array(run.run_id)}
        for split in extract.SPLITS:
            images, labels = load_stl10(split)
            arrays[f"{split}_y"] = labels.numpy().astype(np.int64)
            arrays.update({f"{split}_{k}": v for k, v in extract.extract_features(enc, images).items()})
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(f".{out.stem}.tmp.npz")
        np.savez(tmp, **arrays)
        tmp.replace(out)
    with np.load(out) as z:
        return {k: z[k] for k in z.files}


def evaluate(study: str, run: ExtraRun, profile: str = "full") -> list[dict]:
    """Every probe on every label subset of one extra run, with the frozen SVMs."""
    from mo433 import probe

    svm = probe.svm_params(config.load_config(profile=profile).probe)
    if svm is None:
        raise RuntimeError("extras need the frozen SVM settings in configs/base.yaml")
    feats = _features(study, run, profile)
    epochs = run.epochs or config.load_config(run.row, run.size, profile).epochs
    return [{**r, "study": study, "epochs": epochs} for r in probe.probe_features(feats, svm, encoder_id=run.run_id)]


def _write_csv(path: Path, rows: list[dict], columns: list[str] | tuple[str, ...]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)
    return path


def run(study: str, profile: str = "full", train_only: bool = False, log: Callable[[str], None] = print) -> dict:
    """``mo433 extra <study>``: train what is missing, then probe everything and write the CSVs."""
    from mo433 import train

    st = STUDIES[study]
    for r in st.runs:
        rd = run_dir(study, r, profile)
        if not (rd / "summary.json").exists():
            log(f"[extra {study}] training {r.run_id} (epochs={r.epochs or 'frozen'})")
            train.train_run(r.row, r.size, r.frac, r.seed, profile=profile, run_dir=rd, epochs=r.epochs, log=log)
    if train_only:
        return {}

    rows, curves = [], []
    for r in st.runs:
        log(f"[extra {study}] probing {r.run_id}")
        rows += evaluate(study, r, profile)
        epochs = r.epochs or config.load_config(r.row, r.size, profile).epochs
        curves += curve_rows(run_dir(study, r, profile), run_id=r.run_id, study=study, epochs=epochs)
    for rid in st.reference_runs:
        ref = paths.run_dir(rid, profile)
        if (ref / "metrics.jsonl").exists():
            size = config.parse_encoder_id(rid).size
            curves += curve_rows(ref, run_id=rid, study="main", epochs=config.load_config("B", size, profile).epochs)

    from mo433.report import RESULTS_COLUMNS

    out = output_dir() if profile == "full" else study_dir(study, profile) / "report"
    cols = [*RESULTS_COLUMNS, "study", "epochs"]
    extra_cols = [k for k in (rows[0] if rows else {}) if k not in cols]
    res = _write_csv(out / f"{study}.csv", rows, cols + extra_cols)
    cur = _write_csv(out / f"{study}_curves.csv", curves, CURVE_COLUMNS)
    log(f"[extra {study}] wrote {res} ({len(rows)} rows) and {cur}")
    return {"results": res, "curves": cur}
