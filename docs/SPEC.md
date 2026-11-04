# MO433 experimental-design spec (locked)

The single reference for implementing "Distilling Unsupervised Representations into Small Models" (`mo433_project.pdf`, the brief). Every line traces to a closed ticket of the [design map](https://github.com/nelliDev/mo433/issues/1); the brief is the source of truth for anything this spec does not override. Domain terms (view pool, ablation subset, fixed-view control, run, encoder, validation split, tuning grid, collapse check) are used as defined in [`GLOSSARY.md`](../GLOSSARY.md).

Status: locked 2026-11-03. Configs freeze at the git tag `configs-frozen` on **Sun 15 Nov 2026**; submission **Mon 30 Nov 2026** (deadline 1 Dec).

Changing anything here after the freeze requires a new ticket and a note in the report.

## 1. Question and rows

Does distilling a frozen self-supervised ViT into a small convnet (C) beat training the same convnet directly with self-supervision (B), as student size and unlabeled data vary, under a budget that charges the teacher's cost? C − B is the result; a difference counts only if it exceeds the seed-to-seed spread.

| Row | Encoder | Seeds |
|---|---|---|
| A | ResNet96 student, random init, BN in `eval()` with default running stats | 3 (S), 2 (L) |
| B | ResNet96 trained from scratch with VICReg on the **view pool** | 3 (S), 2 (L) |
| C | Same ResNet96, distilled from the frozen teacher on the **view pool** | 3 (S), 2 (L) |
| D | DINO ViT-S/16 teacher, frozen | 1 checkpoint |

**Stated confounds** (each gets a sentence in the report):
- C vs B differs in supervision source *and* objective (DINO-trained teacher vs VICReg in B).
- STL-10 images come from ImageNet, which the teacher saw unlabeled; this favours C and D. Handled by discussion only (no second dataset).
- Optimizer tuned on B and copied to C: a bias toward B, so C > B is conservative.
- Fixed views (§6.4).
- The ablation halves data *and* steps (§3).

Sources: [#3](https://github.com/nelliDev/mo433/issues/3), [#5](https://github.com/nelliDev/mo433/issues/5), [#8](https://github.com/nelliDev/mo433/issues/8), [#10](https://github.com/nelliDev/mo433/issues/10), brief §2.

## 2. Data

- **Dataset:** STL-10 from `data/stl10_binary/` (HF mirror `xingslong/stl10_binary`, MD5 matches torchvision), loaded with `torchvision.datasets.STL10('data', download=False)`. 100k unlabeled / 5k labeled train / 8k test, 96 px, 10 classes.
- **Unlabeled set:** all N = 100k images, written to the view pool in a seeded random order.
- **Ablation subset:** the first 50k images of that order. It is shared by B, C, S, L and all seeds.
- **Validation split:** a fixed, class-stratified 10% of the labeled train set (500 images). Every tuning and selection choice is scored on it, with the other 90% as the k-NN neighbour bank. The test split informs no choice.
- **Label subsets** for the reported probes: fractions {1%, 10%, 50%, 100%} × 3 draws of the *full* 5k labeled train set, generated exactly as in brief §3.1 (`train_test_split(..., stratify=y_train, random_state=draw)`; 100% = all indices).
- **Probe-CV folds:** 5-fold stratified on the full labeled train set.
- All index files (label subsets, validation split, ablation order, CV folds) are generated once by a seeded script and **committed under `splits/`**, so "same subsets for every encoder" holds by construction.

Sources: [#6](https://github.com/nelliDev/mo433/issues/6), [#7](https://github.com/nelliDev/mo433/issues/7), [#8](https://github.com/nelliDev/mo433/issues/8), [#9](https://github.com/nelliDev/mo433/issues/9).

## 3. Views, teacher cache and sampler

- **View pool:** K = 20 views per image, made with lightly 1.6.0
  `VICRegViewTransform(input_size=96, min_scale=0.2, gaussian_blur=0.5, sigmas=(0.1, 1.0), normalize=None)`;
  everything else at defaults (jitter 0.4/0.4/0.2/0.1 at p=0.8, grayscale 0.2, solarize 0.1, flip 0.5).
- **Storage:** a uint8 memmap `[100_000, 20, 96, 96, 3]` (≈55 GB) in a `chattr +C` directory (`$MO433_POOL`, default `$MO433_ROOT/pool/`). Normalization happens on the GPU.
- **Teacher cache:** the teacher's feature for every (image, view): 2M × 384 fp16 (≈1.5 GB). It is computed once and reused by every C run, seed, size and data fraction. Its GPU time is recorded and charged to C.
- **Sampler: chunk shuffle.** Each epoch the order of 64-image chunks is shuffled, so a batch of 256 is 4 chunks. View indices are drawn at random per epoch: 2 distinct views for B, 1 for C. Measured at 6–11k pairs/s, above every GPU step.
- **Exposure at E = 100:** B draws from C(20, 2) = 190 pairs per image and sees each view about 10 times. C sees each view about 5 times.
- **Ablation:** the 50% runs read only images [0, 50k). Epochs stay equal, so they get half the optimizer steps. The ablation is therefore a joint data-and-compute budget axis, and the report says so.

Sources: [#3](https://github.com/nelliDev/mo433/issues/3), [#6](https://github.com/nelliDev/mo433/issues/6), [#7](https://github.com/nelliDev/mo433/issues/7), [#11](https://github.com/nelliDev/mo433/issues/11).

## 4. Encoders

### 4.1 Teacher (row D)
- `timm.create_model("vit_small_patch16_224.dino", pretrained=True, num_classes=0).eval().requires_grad_(False)`. It is byte-identical to FAIR `dino_deitsmall16_pretrain.pth` / hub `dino_vits16`.
- 21.67M params, 384-d, 4.3 GMACs at 224.
- **Input:** the whole 96 px image or view, bicubic-resized to 224 with no crop, normalized with ImageNet mean/std `(0.485,0.456,0.406)/(0.229,0.224,0.225)`. fp16 autocast under `no_grad`; features cast `.float()` before L2-norm or caching.
- **Feature:** the CLS token after the final LayerNorm (timm's pooled output).
- **Sanity check:** k-NN on its features vs the published **74.5%** (ImageNet, L2-normalized CLS), plus the collapse check (§7.3). Fair linear reference: **76.1%** (single CLS).
- **Pretraining cost for the budget analysis:** ImageNet-1k unlabeled, 800 epochs. The published figure is 1.16k GPU-h for 300 epochs (DINO Table 8); linear extrapolation gives **≈3.1k GPU-h**, which the report labels an estimate.

### 4.2 Student family "ResNet96" (rows A, B, C)

| Choice | Decision |
|---|---|
| Block | Basic residual block: 3×3 conv–BN–ReLU–3×3 conv–BN, add shortcut, ReLU. 1×1 conv + BN projection shortcut on shape change |
| Stem | 3×3 stride-2 conv, no max-pool |
| Stages | [2,2,2,2], widths w/2w/4w/8w, maps 48→24→12→6, global average pool, no head/dropout/drop-path |
| Norm / act | BatchNorm + ReLU; features always extracted in `eval()` |
| Init | Kaiming-normal fan_out, BN γ=1 / β=0, no zero-init-γ; only the seed varies across A/B/C |
| Scaling rule | width only |
| **S** | w = 24: 1.57M params, 192-d |
| **L** | w = 48: 6.28M params, 384-d (teacher ≈ 3.5× L) |

- Training uses bf16 autocast and channels_last.
- **Rejected:** depthwise/ConvNeXt-style blocks (weak under stock SSL, 3.1× slower) and a trained multi-scale fusion head.
- **Eval-only multi-scale probe** (pre-registered): concat GAP(stage 3) + GAP(stage 4) of the trained net for A/B/C, with a fixed random projection to 8w as a dimension-matched control. It uses the same probes and frozen SVM settings as the main probes.

Full design record: [`docs/student-design/`](student-design/README.md). Its "Unverified claims" must be checked before they are cited.

Sources: [#2](https://github.com/nelliDev/mo433/issues/2), [#4](https://github.com/nelliDev/mo433/issues/4), [#5](https://github.com/nelliDev/mo433/issues/5).

## 5. Training (rows B and C)

### 5.1 Shared budget (identical for B and C, both sizes)

| Setting | Value |
|---|---|
| Epochs E | 100 |
| Batch | 256 (pairs for B, images for C) |
| Optimizer | AdamW, betas (0.9, 0.999), weight decay on all params (backbone + head, no BN/bias exclusion) |
| Schedule | linear warmup, then per-step cosine to 0 over the whole run |
| lr / wd / warmup | from the Optuna study (§6.1). Fallback: 1e-3 / 0.05 / 2 epochs |
| Checkpointing | resumable every 10 epochs (model, optimizer, scheduler, sampler RNG). Probed from **final-epoch** weights only |
| Monitoring | per-epoch JSONL: loss (B: the three VICReg terms separately), embedding std, validation k-NN. Monitoring only, never used for selection |

If the batch ever has to change, LR is rescaled linearly and the report says so. Gradient accumulation is not allowed for B, because VICReg's variance and covariance terms and BN act per micro-batch.

### 5.2 Row B: VICReg
- `lightly.loss.VICRegLoss()` defaults (λ=25, μ=25, ν=1, eps=1e-4), computed on `z.float()`.
- **Expander:** `VICRegProjectionHead(d, h, h, num_layers=3)`, with h from the tuning grid (default 2048). It is discarded after training.
- Two distinct pool views per image per step.

### 5.3 Row C: distillation
- **Projector:** `Linear(d, h) – BatchNorm1d(h) – GELU – Linear(h, 384)` (brief §3.5), with h from the tuning grid (default 2048). It is discarded after training.
- **Loss:** `1 − cosine(projector(student(v)), teacher_cache[i, v])`, averaged over the batch. It stays fixed for the main comparison.
- The student and the teacher see the same view v: the student at 96 px from the pool, the teacher at 224 via the cache. No labels, logits or classification head.

Sources: [#3](https://github.com/nelliDev/mo433/issues/3), [#7](https://github.com/nelliDev/mo433/issues/7), [#10](https://github.com/nelliDev/mo433/issues/10), [#12](https://github.com/nelliDev/mo433/issues/12), brief §3.4–3.5.

## 6. Tuning protocol (week 2, all on S, all scored on the validation split)

Every tuning score uses the same function: k-NN (k = 20, cosine) on L2-normalized encoder features (the student's pooled output, never the head) from the deterministic transform, with the validation split as queries and the other 90% as the bank. Heads are never scored.

### 6.1 Optimizer: one Optuna study on B
- **Search space:** lr log-uniform [3e-4, 3e-3], wd log-uniform [0.01, 0.2], warmup ∈ {1, 2, 4} epochs.
- B, S, expander 2048, 20 epochs, one seed per trial. TPE sampler, 16 trials, MedianPruner, SQLite storage.
- Trial 0 is queued as the smoke config (1e-3 / 0.05 / 2), which is also the fallback.
- The winner is copied unchanged to C and to L. Budget ≈ 2.5–4.5 GPU-h.
- Report caveat: the study's 20-epoch trials are a proxy for the 100-epoch runs.

### 6.2 Equal-effort tuning grid (after 6.1, using its winner)
- B expander width ∈ {1024, 2048, 4096}; C projector hidden width ∈ {1024, 2048, 4096}.
- 20 epochs, **2 seeds per point**, so 6 trials per method (≈1.7 GPU-h B + 0.6 GPU-h C).
- Effort is counted in configurations, not GPU-hours.
- **Tie rule:** a width replaces 2048 only if its 2-seed mean beats 2048's by more than the larger of the two seed spreads. Each method picks its own winner. L reuses S's winners.
- Collapse-flagged trials (§7.3) are excluded from selection.

### 6.3 LARS check (report evidence only, Kaggle)
- 6 trials: B, S, 20 epochs, base lr ∈ {0.2, 1.0, 2.0} × wd ∈ {1e-6, 1e-4}, lr scaled by bs/256.
- If the best run beats the Optuna winner, the optimizer decision reopens. Otherwise it becomes a footnote ("tuned over 6 configs, X% vs Y%, one seed each").

### 6.4 Fixed-view control
- One matched pair: B, S, 100% data, E = 100, seed 0, tuned settings. One run on the pool, one on fresh `VICRegViewTransform` augmentations every step. ≈2.8 GPU-h. Probed like the main runs.
- **Pre-registered switch:** if on-the-fly beats the pool by **> 2 pp k-NN on the validation split**, every main B run goes on-the-fly and C stays on the pool. The switch is a choice, so it never reads the test split. The test-split probes of the pair give the Δ quoted in the report.
- If the frozen config is byte-identical, the pool twin is reused as main run `B-S-100-s0`.
- **Report:**
  - Methods gets one line: B and C share the K = 20 pool.
  - Limitations gets one paragraph with:
    - the exposure numbers (§3);
    - the expected direction, citing EMC²;
    - the measured Δ, labelled one seed and one size;
    - the consequence: C's margin may be inflated by up to Δ in that configuration, with no extrapolation to L or 50% data;
    - the 2 pp rule, stated as fixed before the run. If it fired, the exposure and consequence items are replaced by the input mismatch it creates (B on fresh views, C on the pool).
  - If Δ < 0, report it as-is with no extra runs.

Envelopes: the tuning grid has ≤ 20 GPU-h (a ceiling, not a quota) and the control ≤ 4 GPU-h. The LARS check runs on Kaggle, and every other run is on the laptop.

Sources: [#8](https://github.com/nelliDev/mo433/issues/8), [#10](https://github.com/nelliDev/mo433/issues/10), [#11](https://github.com/nelliDev/mo433/issues/11).

## 7. Evaluation (identical for every encoder)

### 7.1 Feature extraction
- One `extract` function behind a `load_encoder(encoder_id) -> nn.Module` registry. It never branches on the row.
- **Deterministic transform:** the 96 px image as-is (no resize or crop needed at native resolution) for students; §4.1 for the teacher.
- Encoders are frozen and in `eval()`, features computed under `no_grad`. Cached to `features/<encoder_id>.npz` for the train and test splits.

### 7.2 Probes
- **Preprocessing:** L2-normalize, then a `StandardScaler` fitted on the training subset only.
- **k-NN:** cosine, k = min(20, n_per_class), untuned.
- **Linear SVM:** `LinearSVC`, C ∈ {1e-3, 1e-2, 1e-1, 1, 10, 100}.
- **RBF SVM:** `SVC(kernel="rbf")`, C ∈ {0.1, 1, 10, 100} × γ ∈ {0.1, 1, 10, `scale`}. No subsampling (train is 5k).
- **SVM hyperparameters** are chosen once by 5-fold CV (committed folds) on the full labeled train set, using the **first reported B** run (`B-S-100-s0`, 100 epochs). They are frozen before any C encoder is probed and used unchanged for every encoder, label fraction, draw and the multi-scale probe.
- **Fit/score:** each probe is fitted on every label subset (4 fractions × 3 draws) and scored on the test split.

### 7.3 Diagnostics
- **Collapse check:** an encoder is flagged if **any** of these holds on the validation split:
  1. its k-NN ≤ row A's k-NN (same architecture);
  2. per-dimension std of L2-normalized embeddings × √d < 0.3;
  3. k-means (k = 10) NMI ≤ row A's NMI.
- When it runs: at the end of every tuning trial and every reported run. Rows A and B/C are compared at the same size (S or L) and seed.
- **Teacher sanity check** (row D, no row A anchor): k-NN vs the published 74.5%; per-dimension std × √d well above 0; k-means NMI well above 0 (brief §3.3). It must pass before any C run.
- A flagged reported run is reported with its flag and never silently re-seeded. If a bug is found, the whole set is rerun.
- **Characterization** (test split, every encoder):
  - k-means (k = 10) NMI and ARI;
  - a 2-D projection with one method and one parameter set for all encoders, shown as a 4-panel figure (A/B/C/D).

### 7.4 Cost measurement (`measure`)
- **Params** and **FLOPs** at the actual input size: 96 px for the students, 224 for the teacher.
- **GPU latency:** the 3060, batch 256, fp32, `eval()` + `no_grad`, 10 warm-up batches, mean of 100 timed batches with `cuda.synchronize()`.
- **CPU latency:** batch 1, 16 threads.
- **GPU-hours per run:** wall and GPU time, plus the teacher-cache GPU time charged to C.

Sources: [#5](https://github.com/nelliDev/mo433/issues/5), [#8](https://github.com/nelliDev/mo433/issues/8), [#9](https://github.com/nelliDev/mo433/issues/9), brief §3.6–3.8.

## 8. Reported runs and aggregation

- **20 runs:** {B, C} × {S: seeds 0–2, L: seeds 0–1} × {100%, 50%} = 2 × 5 × 2.
- **Run id:** `{row}-{size}-{frac}-s{seed}`, e.g. `C-L-050-s1`.
- **Total budget:** ≈18.7 GPU-h, about 24 wall-h at 1.3× overhead. The longest run is B-L at ≈3.3 h.
- **Week-3 run order:**
  1. main S;
  2. main L;
  3. ablation S;
  4. ablation L.
  
  B and C for the same seed run back to back. If time runs out, the ablation is cut, never the main comparison.
- **Slip rule:**
  - Decided at the 15 Nov gate, or as soon as the dry run shows more than 1.3× overhead.
  - Applied identically to all 20 runs, never partway through a set.
  - Cut order: images 100k → 75k (the ablation becomes 37.5k), then epochs 100 → 80 → 60, never seeds.
- **Aggregation:**
  - Average the 3 label draws within each seed, then report mean ± sd over seeds.
  - D (one checkpoint) gets mean ± sd over draws, with a footnote.
  - L (2 seeds) also shows its individual runs.
  - A C − B gap is "beyond spread" only if it exceeds the seed sd.
- **Non-determinism:** runs seed python, numpy, torch and the sampler, but are not bitwise deterministic. The report says so.

Sources: [#7](https://github.com/nelliDev/mo433/issues/7), [#9](https://github.com/nelliDev/mo433/issues/9).

## 9. Repository and pipeline

- **Package:** `src/mo433/` with one module per stage: `data`, `pool`, `teacher`, `students`, `train_b`, `train_c`, `extract`, `probe`, `diagnostics`, `measure`, `report`, `cli`. `bench/` stays outside the package. `pyproject.toml` and `uv.lock` move to the repo root from `task/env-throughput`.
- **Environment:** uv, Python 3.12, torch 2.11.0+cu128, torchvision 0.26, timm 1.0.31, lightly 1.6.0, scikit-learn 1.9.1, optuna.
- **CLI:** `uv run mo433 <subcommand>`, no Makefile.
  - Setup: `splits`, `pool`, `teacher`.
  - Training: `train --row {B,C} --size {S,L} --frac {100,50} --seed N` and `train-all` (week-3 order).
  - Evaluation and reporting: `extract`, `probe`, `diagnose`, `measure`, `report`.
  - Tuning (opt-in): `tune {optuna,grid,fixed-view,lars}`. Its study DBs and summary JSON are committed.
  - `all` runs `splits` → `report` from the frozen configs.
- **Resume:** each stage writes its outputs, then an atomic `.done` marker. Finished stages are skipped; `--force` rebuilds.
- **Smoke profile:** `--profile smoke` uses 2k images, K = 2, 2 epochs and writes to `artifacts-smoke/`. It is the W1 dry run and the pre-flight check before every long run.
- **Configs:**
  - YAML is loaded into frozen dataclasses. `configs/base.yaml` holds the shared settings (N, K, E, batch, AdamW, probe C/γ), with one small file per row × size.
  - Seed and data fraction are CLI arguments.
  - The freeze is the tag `configs-frozen`.
- **Storage:**
  - Committed: `splits/`, `results.csv`, `report/generated/*.tex`, `report/figures/*.pdf`, tuning summaries.
  - Gitignored under `$MO433_ROOT` (default `artifacts/`): `pool/`, `teacher/`, `runs/<run_id>/`, `features/`, `results/`, `tuning/`.
  - Raw data stays in `data/`. The 20 final checkpoints (≈300 MB) go on a GitHub release.
- **Run directory:** the resolved config, git commit and dirty flag, wall and GPU time, and a per-epoch JSONL log. No W&B.
- **Results → LaTeX:**
  - `results.csv` is the master table, with one row per (encoder_id, label_frac, draw, probe). The report reads only this file.
  - `report` writes booktabs tables and a `\newcommand` macro for every number quoted in prose.
  - Figures:
    - the label-efficiency curve (linear-SVM accuracy vs log label fraction, one line per encoder);
    - the 4-panel projection;
    - the ablation grid (setting × size, C − B gap per cell);
    - C − B gap vs student params.
- **README:** states the commands that reproduce every table and figure (brief §8).
- **Invariant tests** (pytest, run inside `smoke`):
  - the eval transform is deterministic;
  - `splits/` matches the committed indices;
  - `load_encoder` returns a flat vector of the stated dimension;
  - encoder parameters are unchanged across `extract` and `probe`.

Source: [#9](https://github.com/nelliDev/mo433/issues/9).

## 10. Calendar

| Week | Laptop GPU | Implementation and writing |
|---|---|---|
| W1, 3–8 Nov | Pool + teacher cache (≈30 min), teacher sanity check vs 74.5%, row A features, smoke profile, then a 5-epoch full-scale dry run on the real pool that measures the overhead factor for the slip rule | Implement the pipeline |
| W2, 9–15 Nov | Optuna (≤ 4.5 h) → head grid → fixed-view control (≤ 4 h); LARS check on Kaggle | **Freeze configs Sun 15 Nov**; LaTeX skeleton, related work |
| W3, 16–22 Nov | All 20 runs (done ≈ Wed 18); rerun slot Thu 19 – Sun 22 | SVM CV after `B-S-100-s0`; probes as runs land; write methods |
| W4, 23–30 Nov | Probe and figure regeneration only; from-scratch reproducibility check (`mo433 all` on a clean clone, frozen configs, reusing the release checkpoints for training stages) | Results and discussion, README; **submit Mon 30 Nov** |

Source: [#7](https://github.com/nelliDev/mo433/issues/7).

## 11. Report obligations

The report:
- is written in LaTeX; the template and length have not been announced yet;
- states the hardware (RTX 3060 Laptop 6 GB, 16 threads, 15 GB RAM), dataset, splits and their uses, seeds, K, the teacher checkpoint and pooling, and BN-eval for row A;
- reports params, FLOPs, latency and feature dimension per encoder, with probe capacity varying by dimension as a caveat;
- includes a related-work section citing the 16 verified entries ([#4](https://github.com/nelliDev/mo433/issues/4)). It positions the work as locating the distil-vs-direct-SSL crossover across student size and data under a teacher-inclusive budget, with Farhat & Chen 2024 as the closest prior work and Shi et al. 2022 as the justification for full tuning effort on B;
- answers brief §6 Q1–6, with the "worth it" judgement in accuracy per total GPU-hour (teacher pretraining estimate + teacher-cache GPU time + distillation GPU time);
- covers every confound listed in §1 and the limitations in §6.3–6.4;
- reports the throughput measurements behind the choice of N, K and E (loader, B and C steps, teacher forward; [#6](https://github.com/nelliDev/mo433/issues/6), [#12](https://github.com/nelliDev/mo433/issues/12)), as brief §3.9 asks;
- presents the ablation as a grid (setting × size, C − B gap per cell), stating what it tests, what was expected and what was found, and whether the gap moves in the same direction for both sizes (brief §5);
- frames results as undertrained at this scale and labels a negative result honestly.

## 12. Implementation defaults filled in by this spec

These gaps were left open by the tickets. They are fixed here so implementation needs no further decision; any of them can be overridden by ticket before the freeze.

1. **Student input normalization:** ImageNet mean/std, the same as the teacher, applied on the GPU to both pool views and eval images.
2. **Index seeds:**
   - Pool image order: `numpy.random.default_rng(0).permutation(100_000)`.
   - Validation split: `train_test_split(..., test_size=0.1, stratify=y, random_state=0)`.
   - CV folds: `StratifiedKFold(5, shuffle=True, random_state=0)`.
   - Label subsets: `random_state=draw`, as in the brief.
3. **FLOP counter:** `torch.utils.flop_counter`, which needs no extra dependency. It follows #5 and resolves #9's "fvcore or thop", and the report quotes MACs = FLOPs/2.
4. **Latency precision:** fp32 for the reported latency table ([#9](https://github.com/nelliDev/mo433/issues/9)). This supersedes #5's bf16 numbers, which stay design-time evidence only.
5. **2-D projection:** t-SNE (scikit-learn), perplexity 30, `random_state=0`, on test-split features after the probe preprocessing (scaler fitted on the 100% labeled train set). No extra dependency.
6. **Seed mapping:** seed s initializes the student (row A uses the same seed s as B/C, so A-S-s0 is B-S-100-s0's initial weights) and seeds the sampler.

## 13. Known departures from the brief

These decisions were taken in tickets and are kept. The report justifies each one.

1. **Label-draw variance in the spread** ([#9](https://github.com/nelliDev/mo433/issues/9)). Brief §2 asks for the label-subset draws to be included in the spread; this spec averages draws within each seed before taking the seed sd.
2. **Ablation runs last** ([#7](https://github.com/nelliDev/mo433/issues/7)). Brief §5 says to run reduced-data settings first because they are cheaper. Here the main comparison runs first, so that a slip cuts the ablation and never the result.

## 14. Out of scope

- A second dataset.
- A second teacher.
- More than two student sizes.
- Patch-token or feature-map distillation.
- DINO as B's method.
- A supervised ImageNet-pretrained reference row.
- The trained multi-scale fusion head.
