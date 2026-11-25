# MO433: Distilling Unsupervised Representations into Small Models

Does distilling a frozen self-supervised ViT (DINO ViT-S/16) into a small convnet beat
training the same convnet directly with self-supervision (VICReg), as student size and
unlabeled data vary, under a budget that charges the teacher's cost? The study compares
four rows on STL-10, all evaluated with the same frozen-feature probes:

| Row | Encoder |
|---|---|
| A | ResNet96 student at random init (BN in `eval()`), the floor |
| B | ResNet96 trained from scratch with VICReg on the view pool |
| C | the same ResNet96 distilled from the frozen teacher on the view pool |
| D | the frozen DINO ViT-S/16 teacher, the ceiling |

Two student sizes (S: 1.57M params, L: 6.28M), 100% and 50% unlabeled data, 3 seeds for S
and 2 for L: 20 training runs. The full experimental design is locked in
[`docs/SPEC.md`](docs/SPEC.md); the domain terms are in [`GLOSSARY.md`](GLOSSARY.md); the
course brief is `mo433_project.pdf`.

## Environment

[uv](https://docs.astral.sh/uv/), Python 3.12, CUDA 12.8 wheels of torch 2.11 (pinned in
`uv.lock`). Reference hardware: RTX 3060 Laptop (6 GB), 16 CPU threads, 15 GB RAM.

```bash
uv sync                      # creates .venv from uv.lock
uv run mo433 --help          # every subcommand
uv run pytest                # tests; GPU/data tests skip when unavailable
```

## Data

STL-10 binaries (100k unlabeled, 5k labeled train, 8k test, 96 px). The Stanford server is
often unreachable, so download the identical tarball from the Hugging Face mirror
[`xingslong/stl10_binary`](https://huggingface.co/datasets/xingslong/stl10_binary) and
check it against torchvision's MD5:

```bash
mkdir -p data
curl -L -o data/stl10_binary.tar.gz \
  https://huggingface.co/datasets/xingslong/stl10_binary/resolve/main/stl10_binary.tar.gz
echo "91f7769df0f17e558f3565bffb0c7dfb  data/stl10_binary.tar.gz" | md5sum -c -
tar -xzf data/stl10_binary.tar.gz -C data      # -> data/stl10_binary/*.bin (2.9 GB)
```

Per-file MD5s (as checked by `torchvision.datasets.STL10`): `train_X.bin`
`918c2871b30a85fa023e0c44e0bee87f`, `train_y.bin` `5a34089d4802c674881badbb80307741`,
`unlabeled_X.bin` `5242ba1fed5e4be9e1e742405eb56ca4`, `test_X.bin`
`7f263ba9f9e0b06b93213547f721ac82`, `test_y.bin` `36f9794fa4beb8a2c72628de14fa638e`.
The code loads it with `torchvision.datasets.STL10(<data root>, download=False)`.

## Environment variables

| Variable | Meaning | Default |
|---|---|---|
| `MO433_ROOT` | artifact root of the full profile (pool, teacher cache, runs, features, results) | `artifacts/` |
| `MO433_SMOKE_ROOT` | artifact root of `--profile smoke`; smoke never uses `MO433_ROOT` | `artifacts-smoke/` |
| `MO433_POOL` | view-pool directory; put it on a `chattr +C` directory (≈55 GB memmap) | `$MO433_ROOT/pool` |
| `MO433_DATA` | directory that contains `stl10_binary/` | `data/` |
| `MO433_NUM_WORKERS` | data-loader workers for `pool` and training runs (cap it on a shared machine) | 14 (`pool`), 8 (training) |
| `MO433_N_JOBS` | joblib workers for probes and diagnostics | 6 |
| `MO433_DATASET` | `stl10` (the main study) or `eurosat` (second-dataset extra, below) | `stl10` |

## Reproducing every table and figure

Every stage writes its outputs and then an atomic `.done` marker under
`$MO433_ROOT/stages/`; finished stages are skipped, `--force` rebuilds. `--profile smoke`
(2k images, K = 2, 2 epochs, writes to `$MO433_SMOKE_ROOT`, default `artifacts-smoke/`) runs the same pipeline in
minutes and is the pre-flight check before every long run.

Everything in one command, from the frozen configs (git tag `configs-frozen`):

```bash
uv run mo433 all                     # splits -> invariants -> pool -> teacher -> train-all -> extract -> probe -> diagnose -> measure -> report
uv run mo433 --profile smoke all     # the same, smoke-sized
```

`invariants` runs the spec §9 invariant tests (`pytest -m invariant`) before any GPU stage.
SVM C/gamma: the full `all` probes only with the frozen values in `configs/base.yaml`. While
they are null it stops after `extract` and says so: run `mo433 probe --select-svm`
(5-fold CV on B-S-100-s0), commit `configs/base.yaml`, rerun `mo433 all` (finished
stages are skipped). The smoke `all` probes and diagnoses with the provisional SVM settings
(`--provisional-svm`) and only one encoder per row (`A-S-s0`, `B-S-100-s0`, `C-S-100-s0`,
`D`); its report goes to `$MO433_SMOKE_ROOT/report-repo/`, never to the committed files.

Or stage by stage:

```bash
# setup
uv run mo433 splits                  # committed index files under splits/ (label subsets, validation split, CV folds, pool order)
uv run mo433 pool                    # K = 20 view pool memmap (spec §3)
uv run mo433 teacher                 # teacher sanity check vs 74.5% k-NN + teacher-feature cache

# the 20 reported runs (spec §8), in week-3 order, each extracted as it lands ...
uv run mo433 train-all --dry-list    # print the order with done/todo
uv run mo433 train-all               # (--only S|L for one size; a diverged run is skipped and reported)
# ... or one at a time
uv run mo433 train --row B --size S --frac 100 --seed 0
uv run mo433 train --row C --size L --frac 50 --seed 1

# evaluation
uv run mo433 extract                 # frozen features of every encoder (rows A, B, C, D)
uv run mo433 probe --select-svm      # once: SVM C/gamma by 5-fold CV on B-S-100-s0, written to configs/base.yaml
uv run mo433 probe                   # k-NN / linear SVM / RBF SVM on every label subset -> results.csv
uv run mo433 diagnose                # collapse check, k-means NMI/ARI, t-SNE, multi-scale probe
uv run mo433 measure                 # params, FLOPs, latency, GPU-hours per run

# report (no GPU needed)
uv run mo433 report                  # -> results.csv, report/data/, report/generated/*.tex, report/figures/*.pdf
cd report && latexmk -pdf main.tex   # optional: compile the skeleton with every generated file
```

Tuning studies are opt-in (`uv run mo433 tune {optuna,grid,fixed-view,lars}`); their study
databases and summaries are committed.

The week-1 dry run (`uv run mo433 tune dryrun`, spec §8, §10) trains B-S and C-S for 5 epochs
at full scale (N = 100k, K = 20) with the current configs, under `$MO433_ROOT/tuning/dryrun/`.
It reports the overhead factor (wall time / summed GPU step time) per run and overall, the
loader-wait share, a projection of the 20 runs' GPU- and wall-hours, and a verdict against
the 1.3× slip-rule threshold, and writes `tuning/dryrun.json`. Run it on an idle GPU.

### What `mo433 report` produces

`report` never touches the GPU (it does not import torch). It first copies the GPU stages'
outputs from `$MO433_ROOT/results/` into committed files when those are newer
(`--no-sync` skips this), then builds everything from the committed files only:

| Committed input | From |
|---|---|
| `results.csv` (repo root) | `results/results.csv`: one row per (encoder, feature set, label fraction, draw, probe) |
| `report/data/costs.csv` | `results/costs.csv`: params, FLOPs, MACs, feature dim, latencies per architecture |
| `report/data/runs_cost.csv` | `results/runs_cost.csv`: wall and GPU hours per run, teacher-cache time charged to C |
| `report/data/diagnostics.csv` | `results/diagnostics/*.json`, flattened |
| `report/data/tsne.npz` | the four t-SNE panels (A-S-s0, B-S-100-s0, C-S-100-s0, D) with test labels |

| Output | Content |
|---|---|
| `report/generated/main_probes.tex` | Table: every probe × label fraction for A/B/C (S, L, 100% data) and D, with the C − B gap |
| `report/generated/l_runs.tex` | Table: the individual large-student encoders |
| `report/generated/ablation.tex` | Table: ablation grid, C − B per data setting × size, all probes |
| `report/generated/multiscale.tex` | Table: eval-only multi-scale probe and its random-projection control |
| `report/generated/costs.tex` | Table: params, GMACs, feature dim, GPU and CPU latency |
| `report/generated/budget.tex` | Table: GPU-hours per run group, teacher cache and pretraining estimate, accuracy per GPU-hour |
| `report/generated/diagnostics.tex` | Table: val k-NN, spread, NMI, test NMI/ARI, collapse flags, teacher check |
| `report/generated/macros.tex` | a `\newcommand` for every number quoted in the prose (naming scheme in `src/mo433/report_tex.py`) |
| `report/figures/label_efficiency.pdf` | linear-SVM accuracy vs label fraction (log x), one line per encoder group |
| `report/figures/tsne.pdf` | 4-panel t-SNE (A/B/C/D) |
| `report/figures/ablation_grid.pdf` | C − B gap per data setting × size, one panel per label fraction |
| `report/figures/gap_vs_params.pdf` | C − B gap vs student parameters, one line per label fraction |

Aggregation (spec §8): the 3 label draws are averaged within each seed, then mean ± sd over
seeds; D (one checkpoint) shows mean ± sd over the draws, with a footnote. A C − B gap is
"beyond spread" only if it exceeds the larger of B's and C's seed sd. Missing encoders,
runs or inputs are skipped with a warning (tables show `--`, macros a bold `??`), so the
report can be regenerated as runs land.

## Extra experiments (after the freeze)

Run after the 20 reported runs, with the frozen settings plus one declared change each, and
reported apart from the main study. None of them touches `results.csv`, `artifacts/runs` or
the frozen configs.

**Second dataset, EuroSAT** (brief §4.2: a non-ImageNet dataset as a second experiment).
`mo433 prepare-eurosat` downloads EuroSAT RGB (torchvision's MD5-checked mirror) and writes a
96 px copy in the STL-10 binary layout to `data/eurosat/`: a stratified 80/20 split (21.6k
images, all of them the unlabeled pool, labels discarded; 5.4k test), and a stratified 5k
subset of the 21.6k as the labeled train split, mirroring STL-10's 5k. Then the unchanged
pipeline runs on it with every setting of the STL-10 study (untuned; only `n_images` differs,
`configs/datasets/eurosat.yaml`):

```bash
uv run mo433 prepare-eurosat
MO433_DATASET=eurosat uv run mo433 all   # splits/eurosat, artifacts-eurosat/, report/eurosat/
```

`MO433_DATASET=eurosat` ignores `MO433_ROOT`, `MO433_SMOKE_ROOT`, `MO433_POOL` and `MO433_DATA`,
which belong to the main study. Its generated macros, table labels and captions are prefixed
(`\EurosatAccCSFullLinTen`, `eurosat:tab:main-probes`, "EuroSAT: ...") so they can be input
into the same `main.tex`.

**Longer training and more student sizes** (`mo433 extra <study>`, `src/mo433/extras.py`):

| Study | Runs | Change |
|---|---|---|
| `long300` | B-S, C-S, seeds 0–1 | E = 300 instead of 100 |
| `sizes` | B and C at XS (w=12) and M (w=34), seeds 0–1 | two more widths, same scaling rule |

Each trains what is missing under `artifacts/extras/<study>/`, probes with the frozen SVMs
and writes `report/extras/<study>.csv` (probe rows) and `report/extras/<study>_curves.csv`
(per-epoch loss and validation k-NN, with the matching main runs as the 100-epoch reference).

## Storage layout

```
configs/            YAML configs (frozen at tag configs-frozen)          committed
splits/             label subsets, validation split, CV folds, pool order committed
results.csv         master results table                                  committed
report/             main.tex, data/, generated/*.tex, figures/*.pdf      committed
data/stl10_binary/  raw STL-10                                            gitignored
$MO433_ROOT/        (default artifacts/)                                  gitignored
  pool/               view pool memmap (or $MO433_POOL)
  teacher/            teacher-feature cache
  runs/<run_id>/      config, git commit, timings, per-epoch JSONL, checkpoints, final_backbone.pt
  features/           <encoder_id>.npz
  results/            results.csv, costs.csv, runs_cost.csv, diagnostics/
  tuning/             tuning studies
  stages/             .done markers
```

Run ids are `{row}-{size}-{frac}-s{seed}` (e.g. `C-L-050-s1`); row-A encoders are
`A-{S,L}-s{seed}` and the teacher is `D`.

## Checkpoints

The 20 final student checkpoints (≈300 MB) are attached to a GitHub release of this
repository instead of being committed. To evaluate them without retraining, put each one
at `$MO433_ROOT/runs/<run_id>/final_backbone.pt` and run `extract`, `probe`, `diagnose`
and `report` directly instead of `train-all`. The teacher downloads from timm (`vit_small_patch16_224.dino`).

Runs seed python, numpy, torch and the sampler but are not bitwise deterministic on GPU.
