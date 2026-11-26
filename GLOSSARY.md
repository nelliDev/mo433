# MO433 distillation study

A controlled, budget-matched comparison of a small student trained directly with self-supervision (row B) against the same student distilled from a frozen teacher (row C).

## Rows (what is A, B, C, D, J)

Every probed encoder belongs to a row. All student rows use the same student architecture
(ResNet96, below); only how its weights were obtained differs. The letters are internal (run ids,
configs, macro names such as `\AccBSFullLinTen`); every generated table and figure uses the
names below (`report.ROW_NAME`), and a gap is written "Distilled − VICReg".

| Row | Name in the report | Weights come from | Role |
|---|---|---|---|
| **A** | Untrained | Nothing: Kaiming init, never trained; BatchNorm in `eval()` with default running stats | Baseline: tells "distillation helped" apart from "any pretraining helped" |
| **B** | VICReg | Trained from scratch on the unlabeled images with VICReg (lightly defaults, 2 views per image, 3-layer expander of width 2048, discarded) | The comparison: the small convnet trained on its own |
| **C** | Distilled | Trained to match the frozen teacher's feature of the same view: projector Linear(d,2048)-BN-GELU-Linear(2048,384), loss 1 − cosine, projector discarded | The proposed pipeline |
| **D** | Teacher | DINO ViT-S/16 (`vit_small_patch16_224.dino`, 21.7M params, 384-d CLS token), frozen, never trained here; sees 224 px | Ceiling and cost reference |
| **J** | LeJEPA (extra) | Like B but with the LeJEPA objective (SIGReg + view prediction), same views and optimizer | Second direct method: is the gap VICReg's or direct training's? |

**The gap** = C − B (same size, same data, same budget). "Beyond the spread" = |gap| larger
than the larger of the two seed standard deviations.

## Student architecture and sizes

**ResNet96**: our student family, a basic-block ResNet for 96 px inputs: 3×3 stride-2 stem
(no max-pool), four stages of two basic blocks with widths w / 2w / 4w / 8w (feature maps
48→24→12→6), global average pooling, no head. Feature dimension = 8w. BatchNorm + ReLU.

Sizes differ by the base width w only (single scaling rule):

| Size | w | Params | Feature dim | In |
|---|---|---|---|---|
| **XS** | 12 | 0.39M | 96 | extras (`sizes`) |
| **S** | 24 | 1.57M | 192 | main study (3 seeds) |
| **M** | 34 | 3.15M | 272 | extras (`sizes`) |
| **L** | 48 | 6.28M | 384 | main study (2 seeds) |

## Run and encoder ids

- `B-S-100-s0` = row B, size S, 100% of the unlabeled pool, seed 0. `050` = the 50% ablation subset.
- `A-S-s0` = random-init S student with seed 0 (no data fraction: never trained).
- `D` = the teacher (one checkpoint, no seed).
- Groups drop the seed: `C-L-100` = all seeds of C at size L, 100% data.

## Probes, fractions and study names

- **Probes**: k-NN (cosine, k = min(20, examples per class)), linear SVM (C=0.01), RBF SVM
  (C=100, γ=scale), all on L2-normalized + standardized frozen features. The linear SVM is the main one.
- **Label fraction**: 1 / 10 / 50 / 100% of the labeled training set (3 stratified draws each).
- **Data fraction**: 100% or 50% of the unlabeled images used for training (the ablation).
- **E**: training epochs (100 in the main study; 300 in the `long300` extra).
- **Extras** (after the freeze, `mo433 extra <study>`): `long300` (E=300), `sizes` (XS, M),
  `lejepa` (row J); EuroSAT is the whole study rerun with `MO433_DATASET=eurosat`.
- **Datasets**: STL-10 (main; 100k unlabeled, 5k labeled train, 8k test, from ImageNet) and
  EuroSAT (extra; satellite, 21.6k train used as the unlabeled pool, 5k labeled subset, 5.4k test).

## Language

**View pool**:
The fixed set of K pre-generated augmented views per unlabeled image, shared by rows B and C and the only views either row trains on.
_Avoid_: view cache, augmentation pool

**Ablation subset**:
The fixed random half of the unlabeled images used for the 50%-data ablation, identical across rows, student sizes and seeds.
_Avoid_: half split, reduced set

**Fixed-view control**:
A one-seed matched pair of row-B trainings, one on the view pool and one on fresh augmentations drawn every step, that measures what training on fixed views costs B.
_Avoid_: on-the-fly ablation, augmentation control

**Run**:
One training of row B or row C for a given student size, unlabeled-data fraction and seed.
_Avoid_: experiment, job

**Encoder**:
Any frozen feature extractor that is probed: a run's final weights, a random-init row-A student, or the teacher.
_Avoid_: model, backbone (when meaning the probed object)

**Validation split**:
The fixed, class-stratified tenth of the labeled training set on which every tuning and selection choice is scored; the test split never informs a choice.
_Avoid_: dev set, holdout

**Tuning grid**:
The equal-sized set of configurations, varying only a row's own discarded head, given to B and to C; effort is counted in configurations, not compute.
_Avoid_: sweep, search

**Collapse check**:
The test that flags an encoder whose embedding is no better spread or no more class-structured than the same architecture at random initialization.
_Avoid_: sanity check (reserved for the teacher check against its published accuracy)
