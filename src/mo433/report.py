"""Stage ``report``: results -> LaTeX tables, macros and figures (spec §8, §9, §11).

Two steps, so the report rebuilds from committed files without a GPU or artifacts:

1. ``sync_inputs`` copies whatever the GPU stages wrote under ``$MO433_ROOT/results/``
   into committed locations, when the artifact copy exists and is newer:

   - ``results/results.csv``        -> ``<repo>/results.csv`` (spec §9 "Storage")
   - ``results/costs.csv``          -> ``report/data/costs.csv``
   - ``results/runs_cost.csv``      -> ``report/data/runs_cost.csv``
   - ``results/diagnostics/*.json`` -> ``report/data/diagnostics.csv`` (flattened)
   - the four t-SNE panels          -> ``report/data/tsne.npz`` (coords + test labels)

2. ``build_report`` reads only those committed inputs and writes
   ``report/generated/*.tex`` and ``report/figures/*.pdf``. Missing encoders, runs or
   input files are skipped with a warning; a figure whose inputs are all missing is not
   rewritten, so an older committed PDF survives.

Input schemas are defined once, below (``RESULTS_COLUMNS``, ``COSTS_COLUMNS``,
``RUNS_COST_COLUMNS``, ``DIAGNOSTICS_COLUMNS``).

Aggregation (spec §8)
---------------------
- Rows A/B/C: the 3 label draws are averaged within each seed; the cell is mean ± sd
  (``ddof=1``) over those seed means. One seed -> sd undefined (NaN, shown as "--").
- Row D (one checkpoint): mean ± sd over the draws, footnoted in every table.
- C − B gap per (size, data fraction, probe, label fraction): difference of the two
  cell means. It is **beyond spread** iff ``|gap| > max(sd_B, sd_C)``, the larger of
  the two seed sds; an undefined sd never counts as beyond spread. The same reference sd
  is the gap's error bar in the figures.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
import warnings
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from mo433 import config, paths

# ---------------------------------------------------------------- input schemas

# results.csv, written by `probe` / `diagnose` (#16). One row per
# (encoder_id, feature_set, label_frac, draw, probe). The report needs the first six.
RESULTS_COLUMNS = (
    "encoder_id",  # A-S-s0, B-S-100-s0, D, ...
    "feature_set",  # main | ms_concat | ms_randproj (missing column = main)
    "label_frac",  # 1, 10, 50, 100 (percent of the 5k labeled train split)
    "draw",  # 0, 1, 2
    "probe",  # knn | linear_svm | rbf_svm
    "accuracy",  # test accuracy in [0, 1]
    "svm_params_frozen",  # SVM rows: True / False (provisional); optional
)

# costs.csv, written by `measure` (#23). One row per architecture; ``encoder_id`` may be
# an architecture key (S, L, D) or any encoder id of that architecture (A-S-s0, ...).
COSTS_COLUMNS = (
    "encoder_id",
    "params",  # count
    "flops",  # per image at the native input size (96 px students, 224 teacher)
    "macs",  # flops / 2
    "feature_dim",
    "gpu_latency_ms",  # RTX 3060, batch 256, fp32: ms per batch
    "cpu_latency_ms",  # batch 1, 16 threads: ms per image
)

# runs_cost.csv, written by `measure` (#23). One row per reported run.
# ``teacher_cache_gpu_h_charged`` is the full one-off teacher-cache GPU time, repeated on
# every C run (0 for B): per-run C totals include it, grand totals count it once.
RUNS_COST_COLUMNS = ("run_id", "wall_h", "gpu_h", "teacher_cache_gpu_h_charged")

# report/data/diagnostics.csv, flattened by ``sync_inputs`` from
# ``results/diagnostics/<encoder_id>.json`` (#16).
DIAGNOSTICS_COLUMNS = (
    "encoder_id",
    "val_knn",
    "val_spread",
    "val_nmi",
    "test_nmi",
    "test_ari",
    "collapse_flagged",  # True / False / "" (row A and D: not applicable)
    "teacher_passed",  # row D only
)

PROBES = ("knn", "linear_svm", "rbf_svm")
LABEL_FRACS = (1, 10, 50, 100)
FEATURE_SETS = ("main", "ms_concat", "ms_randproj")
# Display names of the rows in every generated table and figure (the letters stay internal).
ROW_NAME = {"A": "Untrained", "B": "VICReg", "C": "Distilled", "D": "Teacher", "J": "LeJEPA"}
GAP_NAME = "Distilled $-$ VICReg"
TSNE_PANELS = ("A-S-s0", "B-S-100-s0", "C-S-100-s0", "D")  # spec §7.3: one per row

# Teacher pretraining estimate (spec §4.1): 1.16k GPU-h / 300 ep x 800 ep.
TEACHER_PRETRAIN_GPU_H = 3100.0
# Fallback student parameter counts (spec §4.2) when costs.csv is missing.
SPEC_PARAMS = {"S": 1.57e6, "L": 6.28e6, "D": 21.67e6}

# ---------------------------------------------------------------- committed locations


def report_dir() -> Path:
    return paths.report_dir()


@dataclass(frozen=True)
class Inputs:
    """Committed report inputs (``sync_inputs`` writes them, ``build_report`` reads them)."""

    results: Path
    costs: Path
    runs_cost: Path
    diagnostics: Path
    tsne: Path

    @classmethod
    def committed(cls, repo_root: Path | None = None) -> Inputs:
        root = Path(repo_root) if repo_root is not None else paths.REPO_ROOT
        # The main study keeps results.csv at the repo root; another dataset keeps it beside its data.
        rel = paths.report_dir().relative_to(paths.REPO_ROOT)
        data = root / rel / "data"
        return cls(
            results=root / "results.csv" if paths.dataset() == paths.DEFAULT_DATASET else root / rel / "results.csv",
            costs=data / "costs.csv",
            runs_cost=data / "runs_cost.csv",
            diagnostics=data / "diagnostics.csv",
            tsne=data / "tsne.npz",
        )


# ---------------------------------------------------------------- encoder ids

# One parser for every module (config.parse_encoder_id); the report's historical names.
EncoderKey = config.EncoderId
parse_encoder = config.parse_encoder_id


def parse_group(group: str) -> EncoderKey:
    """``B-S-100`` -> EncoderKey(B, S, 100, None)."""
    if group == "D":
        return EncoderKey("D", None, None, None)
    parts = group.split("-")
    return EncoderKey(parts[0], parts[1], int(parts[2]) if len(parts) > 2 else None, None)


def arch_key(encoder_id: str) -> str:
    """S, L or D: the architecture an encoder id (or a bare S/L/D) refers to."""
    if encoder_id in ("S", "L", "D"):
        return encoder_id
    key = parse_encoder(encoder_id)
    return "D" if key.row == "D" else key.size


def expected_groups() -> list[str]:
    """Every encoder group of the design, in table order."""
    out = []
    for size in config.SIZES:
        out.append(f"A-{size}")
        for frac in config.FRACS:
            out += [f"{row}-{size}-{frac:03d}" for row in config.ROWS]
    return out + ["D"]


def expected_encoders() -> list[str]:
    ids = [f"A-{s}-s{seed}" for s in config.SIZES for seed in config.SEEDS[s]]
    ids += [config.run_id(*r) for r in config.reported_runs()]
    return ids + ["D"]


# ---------------------------------------------------------------- aggregation


@dataclass(frozen=True)
class Cell:
    """One aggregated (group, probe, label_frac) cell."""

    group: str
    probe: str
    label_frac: int
    mean: float
    sd: float  # over seeds (A/B/C) or over draws (D); NaN if fewer than 2
    n: int  # number of seeds (or draws for D)
    spread: str  # "seeds" | "draws"
    per_seed: dict[int, float] = field(default_factory=dict)  # draw-averaged, A/B/C only


def _sd(values) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")


def aggregate(rows: Iterable[Mapping], feature_set: str = "main") -> dict[tuple[str, str, int], Cell]:
    """``{(group, probe, label_frac): Cell}`` for one feature set (spec §8)."""
    by_seed: dict[tuple, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if (r.get("feature_set") or "main") != feature_set:
            continue
        key = parse_encoder(r["encoder_id"])
        seed_or_draw = int(r["draw"]) if key.row == "D" else key.seed
        by_seed[(key.group, r["probe"], int(r["label_frac"]))][seed_or_draw].append(float(r["accuracy"]))

    cells = {}
    for (group, probe, frac), groups in by_seed.items():
        if group == "D":
            draws = [a for accs in groups.values() for a in accs]
            cells[(group, probe, frac)] = Cell(group, probe, frac, float(np.mean(draws)), _sd(draws), len(draws), "draws")
        else:
            per_seed = {s: float(np.mean(a)) for s, a in sorted(groups.items())}
            vals = list(per_seed.values())
            cells[(group, probe, frac)] = Cell(
                group, probe, frac, float(np.mean(vals)), _sd(vals), len(vals), "seeds", per_seed
            )
    return cells


@dataclass(frozen=True)
class Gap:
    size: str
    frac: int  # unlabeled-data percent (100 main, 50 ablation)
    probe: str
    label_frac: int
    gap: float  # C - B
    ref_sd: float  # max(sd_B, sd_C); NaN if either is undefined
    beyond: bool


def gaps(cells: Mapping[tuple[str, str, int], Cell]) -> dict[tuple[str, int, str, int], Gap]:
    """C − B per (size, data frac, probe, label frac) where both cells exist."""
    out = {}
    for (group, probe, lf), b in cells.items():
        key = parse_group(group)
        if key.row != "B":
            continue
        c = cells.get((f"C-{key.size}-{key.frac:03d}", probe, lf))
        if c is None:
            continue
        ref = max(b.sd, c.sd) if not (math.isnan(b.sd) or math.isnan(c.sd)) else float("nan")
        gap = c.mean - b.mean
        out[(key.size, key.frac, probe, lf)] = Gap(
            key.size, key.frac, probe, lf, gap, ref, bool(not math.isnan(ref) and abs(gap) > ref)
        )
    return out


# ---------------------------------------------------------------- reading inputs


def _read_csv(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def read_results(path: Path) -> list[dict]:
    """Rows of ``results.csv`` as parsed by ``probe.read_results`` (which owns the schema);
    unknown encoder ids are dropped with a warning."""
    from mo433 import probe

    rows, bad = [], set()
    for r in probe.read_results(path):
        try:
            parse_encoder(r["encoder_id"])
        except ValueError:
            bad.add(r["encoder_id"])
            continue
        r["feature_set"] = r.get("feature_set") or "main"
        rows.append(r)
    if bad:
        warnings.warn(f"results.csv: ignoring unknown encoder ids {sorted(bad)}", stacklevel=2)
    return rows


def _float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def read_costs(path: Path) -> dict[str, dict[str, float]]:
    """``{S|L|D: {column: value}}`` from ``costs.csv``."""
    out = {}
    for r in _read_csv(path):
        try:
            key = arch_key(r["encoder_id"])
        except ValueError:
            continue
        out.setdefault(key, {c: _float(r.get(c)) for c in COSTS_COLUMNS[1:]})
    return out


def read_runs_cost(path: Path) -> dict[str, dict[str, float]]:
    return {r["run_id"]: {c: _float(r.get(c)) for c in RUNS_COST_COLUMNS[1:]} for r in _read_csv(path)}


def read_diagnostics(path: Path) -> dict[str, dict]:
    out = {}
    for r in _read_csv(path):
        rec = {c: _float(r.get(c)) for c in DIAGNOSTICS_COLUMNS[1:6]}
        for c in ("collapse_flagged", "teacher_passed"):
            v = r.get(c, "")
            rec[c] = None if v in ("", None) else v == "True"
        out[r["encoder_id"]] = rec
    return out


# ---------------------------------------------------------------- sync (artifacts -> committed)


def _copy_if_newer(src: Path, dest: Path, log) -> bool:
    if not src.exists():
        return False
    if dest.exists() and dest.stat().st_mtime >= src.stat().st_mtime:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    log(f"report: copied {src} -> {dest}")
    return True


def flatten_diagnostics(record: Mapping) -> dict:
    """One ``diagnostics/<id>.json`` record (#16 schema) -> a ``DIAGNOSTICS_COLUMNS`` row."""
    val = record.get("val_stats") or {}
    test = record.get("test_characterization") or {}
    collapse = record.get("collapse_check") or {}
    teacher = record.get("teacher_check") or {}
    return {
        "encoder_id": record["encoder_id"],
        "val_knn": val.get("knn", ""),
        "val_spread": val.get("spread", ""),
        "val_nmi": val.get("nmi", ""),
        "test_nmi": test.get("test_nmi", ""),
        "test_ari": test.get("test_ari", ""),
        "collapse_flagged": collapse.get("flagged", ""),
        "teacher_passed": teacher.get("passed", ""),
    }


def _test_labels(encoder_id: str, profile: str) -> np.ndarray | None:
    feats = paths.features_dir(profile) / f"{encoder_id}.npz"
    if feats.exists():
        with np.load(feats) as z:
            if "test_y" in z.files:
                return np.asarray(z["test_y"], dtype=np.int64)
    raw = paths.stl10_binary_dir() / "test_y.bin"
    if raw.exists():
        return np.fromfile(raw, dtype=np.uint8).astype(np.int64) - 1  # STL-10 labels are 1..10
    return None


def sync_inputs(profile: str = "full", inputs: Inputs | None = None, log=print) -> Inputs:
    """Copy the GPU stages' outputs into the committed report inputs (see module doc)."""
    inputs = inputs or Inputs.committed()
    res = paths.results_dir(profile)
    _copy_if_newer(res / "results.csv", inputs.results, log)
    _copy_if_newer(res / "costs.csv", inputs.costs, log)
    _copy_if_newer(res / "runs_cost.csv", inputs.runs_cost, log)

    diag_dir = res / "diagnostics"
    jsons = sorted(diag_dir.glob("*.json")) if diag_dir.exists() else []
    if jsons:
        rows = [flatten_diagnostics(json.loads(p.read_text())) for p in jsons]
        inputs.diagnostics.parent.mkdir(parents=True, exist_ok=True)
        with open(inputs.diagnostics, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=DIAGNOSTICS_COLUMNS)
            w.writeheader()
            w.writerows(rows)
        log(f"report: flattened {len(rows)} diagnostics records -> {inputs.diagnostics}")

        arrays = {}
        for eid in TSNE_PANELS:
            coords = diag_dir / f"{eid}_tsne.npy"
            labels = _test_labels(eid, profile)
            if coords.exists() and labels is not None:
                xy = np.load(coords).astype(np.float32)
                if len(xy) == len(labels):
                    arrays[f"{eid}__xy"] = xy
                    arrays[f"{eid}__y"] = labels.astype(np.uint8)
        if arrays:
            np.savez_compressed(inputs.tsne, **arrays)
            log(f"report: t-SNE panels {sorted({k.split('__')[0] for k in arrays})} -> {inputs.tsne}")
    return inputs


def read_tsne(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    if not Path(path).exists():
        return {}
    with np.load(path) as z:
        ids = sorted({k.split("__")[0] for k in z.files})
        return {eid: (z[f"{eid}__xy"], z[f"{eid}__y"]) for eid in ids if f"{eid}__y" in z.files}


# ---------------------------------------------------------------- build


@dataclass
class ReportData:
    """Everything the tables, macros and figures are made from."""

    rows: list[dict]
    cells: dict[str, dict[tuple[str, str, int], Cell]]  # feature_set -> cells
    gaps: dict[tuple[str, int, str, int], Gap]
    costs: dict[str, dict[str, float]]
    runs_cost: dict[str, dict[str, float]]
    diagnostics: dict[str, dict]
    tsne: dict[str, tuple[np.ndarray, np.ndarray]]
    provisional_svm: bool
    missing_encoders: list[str]

    @property
    def main(self) -> dict[tuple[str, str, int], Cell]:
        return self.cells["main"]

    def params(self, arch: str) -> float:
        p = self.costs.get(arch, {}).get("params", float("nan"))
        return SPEC_PARAMS[arch] if math.isnan(p) else p


def load_data(inputs: Inputs) -> ReportData:
    rows = read_results(inputs.results)
    if not rows:
        warnings.warn(f"no results in {inputs.results}: tables will be empty", stacklevel=2)
    present = {r["encoder_id"] for r in rows}
    missing = [e for e in expected_encoders() if e not in present]
    if missing and rows:
        warnings.warn(f"results.csv lacks {len(missing)} encoders: {', '.join(missing)}", stacklevel=2)
    provisional = any(str(r.get("svm_params_frozen", "")) == "False" for r in rows)
    if provisional:
        warnings.warn("results.csv has provisional (non-frozen) SVM rows; tables are marked", stacklevel=2)
    cells = {fs: aggregate(rows, fs) for fs in FEATURE_SETS}
    data = ReportData(
        rows=rows,
        cells=cells,
        gaps=gaps(cells["main"]),
        costs=read_costs(inputs.costs),
        runs_cost=read_runs_cost(inputs.runs_cost),
        diagnostics=read_diagnostics(inputs.diagnostics),
        tsne=read_tsne(inputs.tsne),
        provisional_svm=provisional,
        missing_encoders=missing,
    )
    for name, value in (("costs", data.costs), ("runs_cost", data.runs_cost), ("diagnostics", data.diagnostics)):
        if not value:
            warnings.warn(f"no {name} input ({getattr(inputs, name)}): its table is left with placeholders", stacklevel=2)
    return data


def build_report(inputs: Inputs | None = None, out_dir: Path | None = None, log=print) -> dict[str, list[Path]]:
    """Write ``<out_dir>/generated/*.tex`` and ``<out_dir>/figures/*.pdf`` from committed inputs."""
    from mo433 import report_figures, report_tex

    inputs = inputs or Inputs.committed()
    out_dir = Path(out_dir) if out_dir is not None else report_dir()
    data = load_data(inputs)
    tex = report_tex.write_all(data, out_dir / "generated")
    figs = report_figures.write_all(data, out_dir / "figures")
    log(f"report: wrote {len(tex)} tex files and {len(figs)} figures under {out_dir}")
    return {"tex": tex, "figures": figs}


def run(profile: str = "full", log=print) -> dict[str, list[Path]]:
    """``mo433 report``: sync the artifacts' outputs into committed inputs, then build."""
    inputs = sync_inputs(profile, log=log)
    return build_report(inputs, log=log)
