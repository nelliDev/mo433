"""Probes (spec §7.2) and the master results table ``results.csv`` (spec §9).

One evaluation function for every encoder: ``probe_features`` takes cached features
(the ``features/<encoder_id>.npz`` arrays written by ``extract``) and never sees a model,
so probing cannot change encoder parameters.

Per label subset (4 fractions x 3 committed draws), fitted on the subset and scored on test:

- preprocessing: L2-normalize, then ``StandardScaler`` fitted on the subset only;
- ``knn``: cosine k-NN, k = min(20, n_per_class), uniform vote;
- ``linear_svm``: ``LinearSVC(C=linear_c)``;
- ``rbf_svm``: ``SVC(kernel="rbf", C=rbf_c, gamma=rbf_gamma)``.

SVM hyperparameters come from ``configs/base.yaml`` (``probe:``), frozen by
``select_svm`` on ``B-S-100-s0``. While they are null, ``probe`` runs k-NN only unless
``provisional=True`` (CLI ``--provisional-svm``), which uses C=1 (linear) and
C=10, gamma="scale" (RBF) and marks the rows ``svm_params_frozen=False``.

``results.csv`` lives at ``$MO433_ROOT/results/results.csv`` (gitignored). The report
reads the committed copy at the repo root: ``export_results()`` copies it there.
Rows are keyed by (encoder_id, feature_set, label_frac, draw, probe); re-probing an
(encoder, feature set) replaces all its rows.
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import tempfile
import time
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from mo433 import paths

PROBES = ("knn", "linear_svm", "rbf_svm")
FEATURE_SETS = ("main", "ms_concat", "ms_randproj")
K_MAX = 20
# joblib workers each hold ~600 MB; 16 of them per pytest run OOM-killed the 15 GB laptop.
N_JOBS = int(os.environ.get("MO433_N_JOBS", "6"))

RESULTS_COLUMNS = (
    "encoder_id",  # e.g. A-S-s0, B-S-100-s0, D
    "feature_set",  # main | ms_concat | ms_randproj (multi-scale probe and its control)
    "label_frac",  # percent of the 5k labeled train split: 1, 10, 50, 100
    "draw",  # 0, 1, 2
    "probe",  # knn | linear_svm | rbf_svm
    "accuracy",  # test accuracy in [0, 1]
    "n_train",  # size of the label subset
    "k",  # k-NN only
    "C",  # SVMs only
    "gamma",  # RBF only ("scale" or a number)
    "svm_params_frozen",  # SVMs only: True = selected values from config, False = provisional
    "git_commit",
    "timestamp",
)
KEY_COLUMNS = RESULTS_COLUMNS[:5]
_INT_COLUMNS = ("label_frac", "draw", "n_train", "k")

PROVISIONAL_LINEAR_C = 1.0
PROVISIONAL_RBF_C = 10.0
PROVISIONAL_RBF_GAMMA = "scale"


# ---------------------------------------------------------------- SVM settings


@dataclass(frozen=True)
class SvmParams:
    linear_c: float
    rbf_c: float
    rbf_gamma: float | str
    frozen: bool


def svm_params(probe_cfg, provisional: bool = False) -> SvmParams | None:
    """SVM settings from a ``config.ProbeConfig``.

    Frozen (all set) -> those values. Any null -> ``None`` (k-NN only), or the provisional
    fallback when ``provisional`` is true.
    """
    values = (probe_cfg.linear_c, probe_cfg.rbf_c, probe_cfg.rbf_gamma)
    if all(v is not None for v in values):
        return SvmParams(float(values[0]), float(values[1]), values[2], frozen=True)
    if provisional:
        return SvmParams(PROVISIONAL_LINEAR_C, PROVISIONAL_RBF_C, PROVISIONAL_RBF_GAMMA, frozen=False)
    return None


# ---------------------------------------------------------------- the probe function


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def preprocess(train_x: np.ndarray, *others: np.ndarray) -> list[np.ndarray]:
    """L2-normalize everything, then standardize with a scaler fitted on ``train_x`` only."""
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(l2_normalize(train_x))
    return [scaler.transform(l2_normalize(a)) for a in (train_x, *others)]


def knn_k(train_y: np.ndarray) -> int:
    return int(min(K_MAX, np.bincount(np.asarray(train_y)).min()))


def make_classifier(name: str, svm: SvmParams | None, train_y: np.ndarray):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC, LinearSVC

    if name == "knn":
        return KNeighborsClassifier(n_neighbors=knn_k(train_y), metric="cosine", algorithm="brute")
    if svm is None:
        raise ValueError(f"{name} needs SVM parameters")
    if name == "linear_svm":
        return LinearSVC(C=svm.linear_c, max_iter=10_000)
    if name == "rbf_svm":
        return SVC(kernel="rbf", C=svm.rbf_c, gamma=svm.rbf_gamma)
    raise ValueError(f"unknown probe {name!r}")


def fit_score(name: str, svm: SvmParams | None, train_x, train_y, test_x, test_y) -> float:
    """Preprocess, fit probe ``name`` on (train_x, train_y), return test accuracy."""
    import warnings

    from sklearn.exceptions import ConvergenceWarning

    tr, te = preprocess(train_x, test_x)
    clf = make_classifier(name, svm, train_y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        clf.fit(tr, train_y)
    return float((clf.predict(te) == np.asarray(test_y)).mean())


def _row_params(name: str, svm: SvmParams | None, train_y: np.ndarray) -> dict:
    blank = {"k": "", "C": "", "gamma": "", "svm_params_frozen": ""}
    if name == "knn":
        return {**blank, "k": knn_k(train_y)}
    if name == "linear_svm":
        return {**blank, "C": svm.linear_c, "svm_params_frozen": svm.frozen}
    return {**blank, "C": svm.rbf_c, "gamma": svm.rbf_gamma, "svm_params_frozen": svm.frozen}


def probe_features(
    feats: Mapping[str, np.ndarray],
    svm: SvmParams | None,
    *,
    encoder_id: str = "",
    feature_set: str = "main",
    train_key: str = "train_x",
    test_key: str = "test_x",
    splits_dir: Path | None = None,
    n_jobs: int = N_JOBS,
) -> list[dict]:
    """Every probe on every label subset -> result rows (see ``RESULTS_COLUMNS``).

    ``feats`` holds ``train_x`` [5000, d], ``train_y``, ``test_x``, ``test_y`` (other keys
    can be chosen with ``train_key``/``test_key``). ``svm=None`` runs k-NN only.
    """
    from joblib import Parallel, delayed

    from mo433 import splits  # imported here: it pulls in torch, and report reads results.csv via this module

    train_x, train_y = np.asarray(feats[train_key]), np.asarray(feats["train_y"])
    test_x, test_y = np.asarray(feats[test_key]), np.asarray(feats["test_y"])
    probes = PROBES if svm is not None else ("knn",)
    jobs = [
        (frac, draw, name, splits.label_subset(frac, draw, splits_dir))
        for frac in splits.LABEL_FRACS
        for draw in splits.DRAWS
        for name in probes
    ]
    accs = Parallel(n_jobs=n_jobs)(
        delayed(fit_score)(name, svm, train_x[idx], train_y[idx], test_x, test_y) for _, _, name, idx in jobs
    )
    stamp = _provenance()
    return [
        {
            "encoder_id": encoder_id,
            "feature_set": feature_set,
            "label_frac": frac,
            "draw": draw,
            "probe": name,
            "accuracy": acc,
            "n_train": len(idx),
            **_row_params(name, svm, train_y[idx]),
            **stamp,
        }
        for (frac, draw, name, idx), acc in zip(jobs, accs)
    ]


def _provenance() -> dict:
    from mo433.stages import git_state

    state = git_state()
    commit = state["git_commit"] or ""
    if commit and state["git_dirty"]:
        commit += "-dirty"
    return {"git_commit": commit, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")}


# ---------------------------------------------------------------- results.csv


def results_csv_path(profile: str = "full") -> Path:
    return paths.results_dir(profile) / "results.csv"


def committed_results_path() -> Path:
    """Where the report reads ``results.csv`` (spec §9 "Storage": committed at the repo root)."""
    return paths.REPO_ROOT / "results.csv"


def export_results(profile: str = "full", dest: Path | None = None) -> Path:
    """Copy ``$MO433_ROOT/results/results.csv`` to the committed location (repo root)."""
    dest = Path(dest) if dest is not None else committed_results_path()
    shutil.copyfile(results_csv_path(profile), dest)
    return dest


def read_results(path: Path) -> list[dict]:
    """Rows of ``results.csv`` with ints, floats and booleans parsed back."""
    if not Path(path).exists():
        return []
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for col in _INT_COLUMNS:
            if r.get(col, "") != "":
                r[col] = int(r[col])
        r["accuracy"] = float(r["accuracy"])
        if r.get("C", "") != "":
            r["C"] = float(r["C"])
        if r.get("svm_params_frozen", "") != "":
            r["svm_params_frozen"] = r["svm_params_frozen"] == "True"
    return rows


def upsert_results(rows: Iterable[dict], path: Path) -> Path:
    """Write ``rows``, dropping every existing row of the same (encoder_id, feature_set)."""
    rows = list(rows)
    replaced = {(r["encoder_id"], r["feature_set"]) for r in rows}
    kept = [r for r in read_results(path) if (r["encoder_id"], r["feature_set"]) not in replaced]
    merged = sorted(
        kept + rows,
        key=lambda r: (r["encoder_id"], r["feature_set"], int(r["label_frac"]), int(r["draw"]), PROBES.index(r["probe"])),
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".results.", suffix=".csv")
    try:
        with os.fdopen(fd, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=RESULTS_COLUMNS, extrasaction="ignore", restval="")
            writer.writeheader()
            writer.writerows(merged)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


# ---------------------------------------------------------------- SVM selection (spec §7.2)

SELECTION_ENCODER = "B-S-100-s0"
LINEAR_C_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
RBF_GRID = tuple((c, g) for c in (0.1, 1.0, 10.0, 100.0) for g in (0.1, 1.0, 10.0, "scale"))


class SvmSelectionRefused(RuntimeError):
    pass


def cv_select(
    train_x: np.ndarray,
    train_y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    *,
    linear_grid=LINEAR_C_GRID,
    rbf_grid=RBF_GRID,
    n_jobs: int = N_JOBS,
) -> dict:
    """5-fold CV (the committed folds) over both grids; the best mean accuracy wins.

    Ties go to the first grid entry (smallest C, then the gamma order of the grid).
    Preprocessing is refitted on each fold's training part, exactly as in the probe.
    """
    from joblib import Parallel, delayed

    train_x, train_y = np.asarray(train_x), np.asarray(train_y)
    cands = [("linear_svm", SvmParams(c, 1.0, "scale", True)) for c in linear_grid]
    cands += [("rbf_svm", SvmParams(1.0, c, g, True)) for c, g in rbf_grid]
    jobs = [(i, tr, te) for i in range(len(cands)) for tr, te in folds]
    accs = Parallel(n_jobs=n_jobs)(
        delayed(fit_score)(cands[i][0], cands[i][1], train_x[tr], train_y[tr], train_x[te], train_y[te])
        for i, tr, te in jobs
    )
    per_cand: dict[int, list[float]] = {}
    for (i, _, _), acc in zip(jobs, accs):
        per_cand.setdefault(i, []).append(acc)

    linear_cv, rbf_cv = [], []
    for i, (name, p) in enumerate(cands):
        entry = {"mean": float(np.mean(per_cand[i])), "folds": per_cand[i]}
        if name == "linear_svm":
            linear_cv.append({"C": p.linear_c, **entry})
        else:
            rbf_cv.append({"C": p.rbf_c, "gamma": p.rbf_gamma, **entry})
    best_lin = max(linear_cv, key=lambda e: e["mean"])  # max keeps the first of equals
    best_rbf = max(rbf_cv, key=lambda e: e["mean"])
    return {
        "linear_c": best_lin["C"],
        "rbf_c": best_rbf["C"],
        "rbf_gamma": best_rbf["gamma"],
        "linear_cv_acc": best_lin["mean"],
        "rbf_cv_acc": best_rbf["mean"],
        "linear_cv": linear_cv,
        "rbf_cv": rbf_cv,
    }


def _yaml_scalar(v) -> str:
    if isinstance(v, str):
        return v
    s = repr(float(v))
    if "e" in s and "." not in s.split("e")[0]:  # PyYAML needs a dot: 1e-05 -> 1.0e-05
        mant, exp = s.split("e")
        s = f"{mant}.0e{exp}"
    return s


def write_probe_config(values: Mapping[str, object], base_yaml: Path) -> None:
    """Set ``probe.{linear_c, rbf_c, rbf_gamma}`` in ``base.yaml`` in place, keeping layout and comments."""
    lines = Path(base_yaml).read_text().splitlines(keepends=True)
    in_probe, done = False, set()
    for i, line in enumerate(lines):
        if re.match(r"^probe:", line):
            in_probe = True
            continue
        if in_probe and re.match(r"^\S", line):
            break
        m = re.match(r"^(\s+)(linear_c|rbf_c|rbf_gamma):(\s*)(\S+)(.*)$", line.rstrip("\n")) if in_probe else None
        if m and m.group(2) in values:
            indent, key, sp, _old, rest = m.groups()
            lines[i] = f"{indent}{key}:{sp or ' '}{_yaml_scalar(values[key])}{rest}\n"
            done.add(key)
    missing = set(values) - done
    if missing:
        raise ValueError(f"{base_yaml}: probe keys not found: {sorted(missing)}")
    Path(base_yaml).write_text("".join(lines))


def probed_c_encoders(profile: str = "full") -> list[str]:
    rows = read_results(results_csv_path(profile))
    return sorted({r["encoder_id"] for r in rows if r["encoder_id"].startswith("C-")})


def load_features(encoder_id: str, profile: str = "full") -> dict[str, np.ndarray]:
    path = paths.features_dir(profile) / f"{encoder_id}.npz"
    if not path.exists():
        raise FileNotFoundError(f"no cached features for {encoder_id} at {path} (run `mo433 extract`)")
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def available_encoders(profile: str = "full") -> list[str]:
    """Encoder ids with cached features, sorted."""
    return sorted(p.stem for p in paths.features_dir(profile).glob("*.npz"))


def probe_encoder(
    encoder_id: str,
    svm: SvmParams | None,
    *,
    profile: str = "full",
    splits_dir: Path | None = None,
    n_jobs: int = N_JOBS,
) -> list[dict]:
    """Probe ``features/<encoder_id>.npz`` (main feature set) and upsert into ``results.csv``."""
    rows = probe_features(
        load_features(encoder_id, profile), svm, encoder_id=encoder_id, splits_dir=splits_dir, n_jobs=n_jobs
    )
    upsert_results(rows, results_csv_path(profile))
    return rows


def select_svm(
    encoder_id: str = SELECTION_ENCODER,
    *,
    profile: str = "full",
    configs_dir: Path | None = None,
    splits_dir: Path | None = None,
    n_jobs: int = N_JOBS,
    linear_grid=LINEAR_C_GRID,
    rbf_grid=RBF_GRID,
) -> dict:
    """CV-select the SVM settings on ``encoder_id`` and freeze them into ``base.yaml``.

    Refuses (``SvmSelectionRefused``) once any C encoder has rows in ``results.csv``.
    Writes the full CV record to ``$MO433_ROOT/results/svm_selection.json``.
    """
    from mo433 import splits

    probed = probed_c_encoders(profile)
    if probed:
        raise SvmSelectionRefused(
            f"SVM settings must be frozen before any C encoder is probed (spec §7.2); "
            f"results.csv already has rows for {', '.join(probed)}"
        )
    feats = load_features(encoder_id, profile)
    result = cv_select(
        feats["train_x"],
        feats["train_y"],
        splits.cv_folds(splits_dir),
        linear_grid=linear_grid,
        rbf_grid=rbf_grid,
        n_jobs=n_jobs,
    )
    base_yaml = Path(configs_dir if configs_dir is not None else paths.CONFIGS_DIR) / "base.yaml"
    write_probe_config({k: result[k] for k in ("linear_c", "rbf_c", "rbf_gamma")}, base_yaml)
    record = {"encoder_id": encoder_id, "n_folds": splits.N_FOLDS, **result, **_provenance()}
    out = paths.results_dir(profile) / "svm_selection.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2))
    return record
