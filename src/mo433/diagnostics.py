"""Diagnostics (spec §7.3, §4.2, §12.5): collapse check, teacher sanity check,
characterization (k-means NMI/ARI, t-SNE) and the eval-only multi-scale probe.

Collapse check, on the validation split (queries = val, bank = the other 90%):

- ``knn``: ``knn.val_knn`` accuracy (k = 20, cosine);
- ``spread``: mean per-dimension std of the L2-normalized val embeddings x sqrt(d)
  (about 1 for an isotropic embedding, 0 for a constant one);
- ``nmi``: NMI of k-means (k = 10, ``n_init=10``, ``random_state=0``) on the
  L2-normalized val embeddings against the labels.

An encoder is flagged if any of ``knn <= A.knn``, ``spread < 0.3``, ``nmi <= A.nmi``
holds, A being row A (random init) at the same size and seed. Without an anchor only the
spread criterion can be evaluated; the result says so (``anchor_id`` is None).

Teacher sanity check (row D, no anchor), thresholds chosen here (spec says "well above 0"):
``spread > 0.3`` (same bar as the collapse check), ``nmi > 0.1``, and val k-NN at least
the published 74.5% minus a 5-point tolerance for protocol differences (``>= 0.695``).

Outputs of ``diagnose_encoder``: ``$MO433_ROOT/results/diagnostics/<encoder_id>.json``
and ``<encoder_id>_tsne.npy`` (float32 [8000, 2], test-split order, labels = ``test_y``
in the features file); multi-scale probe rows go into ``results.csv`` with
``feature_set`` = ``ms_concat`` / ``ms_randproj``.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from mo433 import knn, paths, probe

SPREAD_MIN = 0.3
TEACHER_REFERENCE_KNN = 0.745
TEACHER_KNN_TOLERANCE = 0.05
TEACHER_NMI_MIN = 0.1
N_CLUSTERS = 10
TSNE_PERPLEXITY = 30.0
PROJECTION_SEED = 0


# ---------------------------------------------------------------- statistics


def spread(x: np.ndarray) -> float:
    """Mean per-dimension std of the L2-normalized rows, times sqrt(d)."""
    z = probe.l2_normalize(x)
    return float(z.std(axis=0).mean() * np.sqrt(z.shape[1]))


def kmeans_scores(x: np.ndarray, y: np.ndarray, seed: int = 0) -> tuple[float, float]:
    """(NMI, ARI) of k-means (k = 10) on ``x`` against ``y``."""
    import warnings

    from sklearn.cluster import KMeans
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)  # constant embeddings: < 10 distinct points
        pred = KMeans(N_CLUSTERS, n_init=10, random_state=seed).fit_predict(np.asarray(x, dtype=np.float64))
    return float(normalized_mutual_info_score(y, pred)), float(adjusted_rand_score(y, pred))


@dataclass(frozen=True)
class CollapseStats:
    knn: float
    spread: float
    nmi: float


def collapse_stats(features, labels, splits_dir: Path | None = None) -> CollapseStats:
    """Validation-split statistics of the labeled-train ``features`` [5000, d]."""
    from mo433 import splits

    features, labels = np.asarray(features), np.asarray(labels)
    val, _ = splits.validation_split(splits_dir)
    nmi, _ = kmeans_scores(probe.l2_normalize(features[val]), labels[val])
    return CollapseStats(
        knn=knn.val_knn(features, labels, splits_dir),
        spread=spread(features[val]),
        nmi=nmi,
    )


@dataclass(frozen=True)
class CollapseResult:
    flagged: bool
    reasons: dict[str, bool]  # criterion -> True if it fired
    stats: CollapseStats
    anchor: CollapseStats | None
    thresholds: dict[str, float] = field(default_factory=lambda: {"spread_min": SPREAD_MIN})


def collapse_check(stats: CollapseStats, anchor: CollapseStats | None) -> CollapseResult:
    """Flag if k-NN <= anchor's, spread < 0.3, or NMI <= anchor's (spec §7.3).

    ``anchor=None`` evaluates the spread criterion only.
    """
    reasons = {"spread": stats.spread < SPREAD_MIN}
    if anchor is not None:
        reasons["knn"] = stats.knn <= anchor.knn
        reasons["nmi"] = stats.nmi <= anchor.nmi
    return CollapseResult(flagged=any(reasons.values()), reasons=reasons, stats=stats, anchor=anchor)


@dataclass(frozen=True)
class TeacherResult:
    passed: bool
    reasons: dict[str, bool]  # criterion -> True if it failed
    stats: CollapseStats
    reference_knn: float = TEACHER_REFERENCE_KNN
    thresholds: dict[str, float] = field(
        default_factory=lambda: {
            "knn_min": TEACHER_REFERENCE_KNN - TEACHER_KNN_TOLERANCE,
            "spread_min": SPREAD_MIN,
            "nmi_min": TEACHER_NMI_MIN,
        }
    )


def teacher_check(stats: CollapseStats) -> TeacherResult:
    """Row D sanity check: k-NN near the published 74.5%, spread and NMI well above 0."""
    reasons = {
        "knn": stats.knn < TEACHER_REFERENCE_KNN - TEACHER_KNN_TOLERANCE,
        "spread": stats.spread <= SPREAD_MIN,
        "nmi": stats.nmi <= TEACHER_NMI_MIN,
    }
    return TeacherResult(passed=not any(reasons.values()), reasons=reasons, stats=stats)


_RUN_ID = re.compile(r"^[BC]-([SL])-\d{3}-s(\d+)$")


def anchor_id(encoder_id: str) -> str | None:
    """Row-A encoder at the same size and seed as run ``encoder_id`` (None for A and D)."""
    m = _RUN_ID.match(encoder_id)
    return f"A-{m.group(1)}-s{m.group(2)}" if m else None


def is_teacher(encoder_id: str) -> bool:
    return encoder_id.startswith("D")


# ---------------------------------------------------------------- characterization (test split)


def characterize(feats: dict, *, tsne: bool = True, n_jobs: int = 16) -> dict:
    """k-means NMI/ARI and t-SNE on the test split after probe preprocessing
    (scaler fitted on the full labeled train set, spec §12.5)."""
    _, test = probe.preprocess(feats["train_x"], feats["test_x"])
    nmi, ari = kmeans_scores(test, feats["test_y"])
    out: dict = {"test_nmi": nmi, "test_ari": ari}
    if tsne:
        from sklearn.manifold import TSNE

        out["tsne"] = (
            TSNE(2, perplexity=TSNE_PERPLEXITY, random_state=0, n_jobs=n_jobs).fit_transform(test).astype(np.float32)
        )
    return out


# ---------------------------------------------------------------- multi-scale probe (spec §4.2)


def random_projection(in_dim: int, out_dim: int, seed: int = PROJECTION_SEED) -> np.ndarray:
    """Fixed Gaussian projection [in_dim, out_dim], entries N(0, 1/out_dim)."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((in_dim, out_dim)) / np.sqrt(out_dim)).astype(np.float32)


def multiscale_feature_sets(feats: dict) -> dict[str, dict] | None:
    """``ms_concat`` = [GAP(s3), GAP(s4)] (12w-d) and its control ``ms_randproj`` = that concat
    through a fixed random projection to 8w (the main feature dimension). None if the
    features file has no stage-3/4 arrays (e.g. the teacher)."""
    keys = ("train_x_s3", "test_x_s3", "train_x_s4", "test_x_s4")
    if not all(k in feats for k in keys):
        return None
    labels = {"train_y": feats["train_y"], "test_y": feats["test_y"]}
    train = np.concatenate([feats["train_x_s3"], feats["train_x_s4"]], axis=1)
    test = np.concatenate([feats["test_x_s3"], feats["test_x_s4"]], axis=1)
    proj = random_projection(train.shape[1], feats["train_x"].shape[1])
    return {
        "ms_concat": {"train_x": train, "test_x": test, **labels},
        "ms_randproj": {"train_x": train @ proj, "test_x": test @ proj, **labels},
    }


# ---------------------------------------------------------------- stage entry point


def diagnostics_dir(profile: str = "full") -> Path:
    return paths.results_dir(profile) / "diagnostics"


def diagnose_encoder(
    encoder_id: str,
    *,
    profile: str = "full",
    svm: probe.SvmParams | None = None,
    anchor: str | None = None,
    tsne: bool = True,
    splits_dir: Path | None = None,
    n_jobs: int = 16,
) -> dict:
    """Collapse/teacher check, characterization and multi-scale probe for one encoder.

    ``anchor`` overrides the row-A encoder id (default ``anchor_id(encoder_id)``).
    ``svm=None`` runs the multi-scale probe with k-NN only (see ``probe.svm_params``).
    """
    feats = probe.load_features(encoder_id, profile)
    stats = collapse_stats(feats["train_x"], feats["train_y"], splits_dir)
    record: dict = {"encoder_id": encoder_id, "val_stats": asdict(stats)}

    if is_teacher(encoder_id):
        record["teacher_check"] = _jsonable(asdict(teacher_check(stats)))
    elif (aid := anchor or anchor_id(encoder_id)) is not None:
        anchor_path = paths.features_dir(profile) / f"{aid}.npz"
        anchor_stats = None
        if anchor_path.exists():
            a = probe.load_features(aid, profile)
            anchor_stats = collapse_stats(a["train_x"], a["train_y"], splits_dir)
        result = collapse_check(stats, anchor_stats)
        record["collapse_check"] = {
            **_jsonable(asdict(result)),
            "anchor_id": aid if anchor_stats is not None else None,
            "note": "" if anchor_stats is not None else f"anchor {aid} features missing: spread criterion only",
        }
    else:
        record["collapse_check"] = None  # row A is the anchor itself

    char = characterize(feats, tsne=tsne, n_jobs=n_jobs)
    out_dir = diagnostics_dir(profile)
    out_dir.mkdir(parents=True, exist_ok=True)
    if tsne:
        np.save(out_dir / f"{encoder_id}_tsne.npy", char.pop("tsne"))
        record["tsne_file"] = f"{encoder_id}_tsne.npy"
        record["tsne_params"] = {"perplexity": TSNE_PERPLEXITY, "random_state": 0}
    record["test_characterization"] = char

    ms = multiscale_feature_sets(feats)
    if ms is not None:
        rows = []
        for name, fs in ms.items():
            rows += probe.probe_features(
                fs, svm, encoder_id=encoder_id, feature_set=name, splits_dir=splits_dir, n_jobs=n_jobs
            )
        probe.upsert_results(rows, probe.results_csv_path(profile))
        record["multiscale"] = {
            "dims": {name: int(fs["train_x"].shape[1]) for name, fs in ms.items()},
            "projection_seed": PROJECTION_SEED,
            "svm_params_frozen": None if svm is None else svm.frozen,
        }

    (out_dir / f"{encoder_id}.json").write_text(json.dumps(record, indent=2))
    return record


def _jsonable(d):
    if isinstance(d, dict):
        return {k: _jsonable(v) for k, v in d.items()}
    if isinstance(d, (np.bool_, bool)):
        return bool(d)
    if isinstance(d, np.floating):
        return float(d)
    return d
