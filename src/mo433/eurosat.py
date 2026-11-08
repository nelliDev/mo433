"""EuroSAT (RGB, 27k images, 64 px, 10 classes) in the STL-10 binary layout: the second-dataset extra.

The brief (§4.2) asks STL-10 studies to "preferably add a non-ImageNet dataset as a second
experiment", because the DINO teacher may have seen STL-10's ImageNet-derived images.
EuroSAT has no unlabeled split and no official test split, so ``materialize``:

- splits the 27k images 80/20, stratified by class (``train_test_split``, ``random_state=0``):
  21.6k train, 5.4k test;
- takes a stratified 5k subset of the train split as the labeled train split (``random_state=0``),
  mirroring STL-10's 5k labeled images: the label fractions mean the same number of labels and
  the RBF probes stay affordable (a 21.6k-row RBF SVM fit takes tens of minutes per encoder);
- upsamples every 64 px image to 96 px with PIL bicubic, so the students, the view pool, the
  augmentation and the 96->224 teacher resize stay exactly those of the STL-10 study (brief §4.3,
  first option);
- writes ``<out>/stl10_binary/{train,test}_{X,y}.bin``, ``unlabeled_X.bin`` (= all 21.6k train
  images, labels discarded: brief §4.2, so the labeled 5k are also in the pool, as in that section) and ``class_names.txt``, plus ``manifest.json`` with
  the source file of every row.

Then ``MO433_DATASET=eurosat`` runs the unchanged pipeline on it (see ``mo433.paths``).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from mo433 import paths

SIZE = 96
TEST_FRAC = 0.2
SPLIT_SEED = 0
N_LABELED = 5_000
MANIFEST = "manifest.json"
SUFFIXES = {".jpg", ".jpeg", ".png", ".tif"}


def download(raw_root: Path) -> Path:
    """torchvision's EuroSAT download (zip from its Hugging Face mirror, MD5-checked) -> the
    ImageFolder directory ``<raw_root>/eurosat/2750``."""
    from torchvision.datasets import EuroSAT

    EuroSAT(str(raw_root), download=True)
    return Path(raw_root) / "eurosat" / "2750"


def _sources(src: Path) -> tuple[list[str], np.ndarray, tuple[str, ...]]:
    classes = tuple(sorted(p.name for p in src.iterdir() if p.is_dir()))
    files, labels = [], []
    for k, c in enumerate(classes):
        for f in sorted((src / c).iterdir()):
            if f.suffix.lower() in SUFFIXES:
                files.append(f"{c}/{f.name}")
                labels.append(k)
    return files, np.asarray(labels, dtype=np.int64), classes


def _write_images(src: Path, files: list[str], path: Path) -> None:
    from PIL import Image

    out = np.empty((len(files), SIZE, SIZE, 3), dtype=np.uint8)
    for i, f in enumerate(files):
        with Image.open(src / f) as im:
            out[i] = np.asarray(im.convert("RGB").resize((SIZE, SIZE), Image.BICUBIC))
    # STL-10 binaries are column-major per image: [N, H, W, C] -> [N, C, W, H].
    out.transpose(0, 3, 2, 1).tofile(path)


def materialize(src: Path, out_root: Path, test_frac: float = TEST_FRAC, seed: int = SPLIT_SEED,
                n_labeled: int = N_LABELED) -> dict:
    """Write the STL-10-layout copy of the ImageFolder ``src`` under ``out_root``; return the manifest."""
    from sklearn.model_selection import train_test_split

    src, out_root = Path(src), Path(out_root)
    files, y, classes = _sources(src)
    train_idx, test_idx = train_test_split(np.arange(len(files)), test_size=test_frac, stratify=y,
                                           random_state=seed)
    train_idx, test_idx = np.sort(train_idx), np.sort(test_idx)
    if n_labeled < len(train_idx):
        labeled_idx, _ = train_test_split(train_idx, train_size=n_labeled, stratify=y[train_idx], random_state=seed)
        labeled_idx = np.sort(labeled_idx)
    else:
        labeled_idx = train_idx
    d = out_root / "stl10_binary"
    d.mkdir(parents=True, exist_ok=True)
    _write_images(src, [files[i] for i in train_idx], d / "unlabeled_X.bin")
    for split, idx in (("train", labeled_idx), ("test", test_idx)):
        _write_images(src, [files[i] for i in idx], d / f"{split}_X.bin")
        (y[idx] + 1).astype(np.uint8).tofile(d / f"{split}_y.bin")  # 1-based, like STL-10
    (d / "class_names.txt").write_text("\n".join(classes) + "\n")
    manifest = {
        "source": str(src), "classes": list(classes), "size_px": SIZE, "resize": "PIL bicubic",
        "test_frac": test_frac, "split_seed": seed,
        "n_unlabeled": len(train_idx), "n_train": len(labeled_idx), "n_test": len(test_idx),
        "unlabeled_sources": [files[i] for i in train_idx],
        "train_sources": [files[i] for i in labeled_idx], "test_sources": [files[i] for i in test_idx],
        "md5": {f.name: hashlib.md5(f.read_bytes()).hexdigest() for f in sorted(d.glob("*.bin"))},
    }
    (out_root / MANIFEST).write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def prepare(raw_root: Path | None = None, out_root: Path | None = None, log=print) -> dict:
    """``mo433 prepare-eurosat``: download (if needed) and materialize into ``data/eurosat``."""
    out_root = Path(out_root) if out_root is not None else paths.REPO_ROOT / "data" / "eurosat"
    raw_root = Path(raw_root) if raw_root is not None else paths.REPO_ROOT / "data" / "eurosat-raw"
    src = download(raw_root)
    manifest = materialize(src, out_root)
    log(f"eurosat: {manifest['n_unlabeled']} unlabeled / {manifest['n_train']} labeled train / "
        f"{manifest['n_test']} test images at {SIZE} px in {out_root}")
    return manifest
