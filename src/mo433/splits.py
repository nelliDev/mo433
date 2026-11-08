"""Committed index files under ``splits/`` (spec §2, §12.2).

Generated once by ``mo433 splits`` and committed, so every encoder sees the same subsets
by construction. All files are int64 ``.npy`` arrays:

========================  ===============================================================
``labels_{PPP}_d{D}.npy`` label subset, PPP in {001, 010, 050, 100} percent, draw D in {0,1,2};
                          indices into the 5k labeled train split (brief §3.1)
``val.npy``               validation split: stratified 10% (500) of labeled train
``bank.npy``              the other 90% (4500): the k-NN neighbour bank for tuning scores
``pool_order.npy``        ``default_rng(0).permutation(100_000)``: pool position p holds
                          unlabeled image ``pool_order[p]``
``cv_folds.npy``          fold id (0-4) of each labeled train image for probe CV
========================  ===============================================================

Use the accessors below rather than reading files by hand.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from mo433 import data, paths

LABEL_FRACS = (1, 10, 50, 100)  # percent of the 5k labeled train split
DRAWS = (0, 1, 2)
N_FOLDS = 5
ABLATION_SIZE = 50_000  # STL-10; in general half the pool order (ablation_subset)


def label_subset_name(frac: int, draw: int) -> str:
    return f"labels_{frac:03d}_d{draw}"


def generate(y_train: np.ndarray, n_unlabeled: int = data.N_UNLABELED) -> dict[str, np.ndarray]:
    """Every index array, keyed by file stem. Pure: depends only on the labels and ``n_unlabeled``."""
    y_train = np.asarray(y_train)
    all_idx = np.arange(len(y_train))
    out: dict[str, np.ndarray] = {}

    # Brief §3.1, verbatim apart from percent keys.
    for frac in LABEL_FRACS:
        for draw in DRAWS:
            if frac == 100:
                idx = all_idx
            else:
                idx, _ = train_test_split(all_idx, train_size=frac / 100, stratify=y_train, random_state=draw)
            out[label_subset_name(frac, draw)] = idx

    bank, val = train_test_split(all_idx, test_size=0.1, stratify=y_train, random_state=0)
    out["val"], out["bank"] = val, bank

    out["pool_order"] = np.random.default_rng(0).permutation(n_unlabeled)

    folds = np.empty(len(y_train), dtype=np.int64)
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=0)
    for k, (_, test) in enumerate(skf.split(all_idx, y_train)):
        folds[test] = k
    out["cv_folds"] = folds

    return {k: np.ascontiguousarray(v, dtype=np.int64) for k, v in out.items()}


def write_splits(out_dir: Path | None = None, data_root: Path | None = None) -> Path:
    out_dir = Path(out_dir) if out_dir is not None else paths.splits_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, arr in generate(data.train_labels(data_root), data.n_unlabeled(data_root)).items():
        np.save(out_dir / f"{name}.npy", arr, allow_pickle=False)
    print(f"wrote splits to {out_dir}")
    return out_dir


# ---------------------------------------------------------------- read API for later stages


def load_split(name: str, splits_dir: Path | None = None) -> np.ndarray:
    """Load ``splits/<name>.npy`` (e.g. ``"val"``, ``"labels_010_d2"``)."""
    splits_dir = Path(splits_dir) if splits_dir is not None else paths.splits_dir()
    return np.load(splits_dir / f"{name}.npy", allow_pickle=False)


def label_subset(frac: int, draw: int, splits_dir: Path | None = None) -> np.ndarray:
    if frac not in LABEL_FRACS or draw not in DRAWS:
        raise ValueError(f"no label subset for frac={frac}%, draw={draw}")
    return load_split(label_subset_name(frac, draw), splits_dir)


def validation_split(splits_dir: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(val, bank): k-NN queries and neighbour bank for every tuning/selection score."""
    return load_split("val", splits_dir), load_split("bank", splits_dir)


def pool_order(splits_dir: Path | None = None) -> np.ndarray:
    return load_split("pool_order", splits_dir)


def ablation_subset(splits_dir: Path | None = None) -> np.ndarray:
    """Unlabeled image indices of the ablation subset = the first half of the pool positions
    ([0, 50k) for STL-10)."""
    order = pool_order(splits_dir)
    return order[: len(order) // 2]


def cv_folds(splits_dir: Path | None = None) -> list[tuple[np.ndarray, np.ndarray]]:
    """[(train_idx, test_idx)] for the 5 probe-CV folds, in StratifiedKFold order."""
    fold_of = load_split("cv_folds", splits_dir)
    return [(np.flatnonzero(fold_of != k), np.flatnonzero(fold_of == k)) for k in range(N_FOLDS)]
