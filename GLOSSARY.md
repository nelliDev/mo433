# MO433 distillation study

A controlled, budget-matched comparison of a small student trained directly with self-supervision (row B) against the same student distilled from a frozen teacher (row C).

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
