# Student throughput and optimizer smoke run on the 3060

Measured 2026-11-03/04 for the wayfinder ticket [#12](https://github.com/nelliDev/mo433/issues/12). Script: `bench/student.py`; raw JSON in `bench/results/` (`vicreg`, `distill`, `feed`, `smoke_adamw`, `smoke_lars`), smoke-run logs in `bench/logs/`. Charts: [dashboard](https://claude.ai/artifact/3cAVc8hRWx5zQ99ZJcbtTu) (private).

Setup for all measurements: RTX 3060 Laptop 6 GB (desktop holds ~0.85 GB), torch 2.11 + cu128, bf16 autocast, channels_last, cudnn.benchmark, no `torch.compile`. Students are the decided `ResNet96` (S w=24, L w=48) from `docs/student-design/bench_throughput.py`. uint8 NHWC input, normalised on the GPU inside the timed step.

## (i) One VICReg step (two views)

lightly `VICRegLoss()` on `z.float()`, `VICRegProjectionHead(D, 2048, 2048, num_layers=3)`, LARS. Separate forward per view (as in the smoke run); a single concatenated forward is within 1–7 % (it changes BN batch statistics, so it is a design choice, not a speed one).

| student | pairs/step | img/s | 100k-pair epoch | peak alloc GB |
|---|---|---|---|---|
| S | 128 | 3,358 | 60 s | 0.87 |
| S | 256 | 4,566 | 44 s | 1.39 |
| S | 512 | 5,257 | 38 s | 2.38 |
| S | 1024 | 5,834 | 34 s | 4.50 |
| L | 128 | 1,885 | 106 s | 1.41 |
| L | 256 | 2,179 | 92 s | 2.38 |
| L | 512 | 2,440 | 82 s | 4.32 |
| L | 1024 | OOM | | |

- The 3-layer 2048 expander + loss cost ~35 % vs the student-design benchmark's single Linear head (S 7.1k → 4.6k img/s at 512 images).
- Largest feasible batch: S 1024 pairs, L 512 pairs. **Gradient accumulation does not help B**: VICReg's variance and covariance terms (and BN) are computed per micro-batch, so accumulating does not emulate a larger batch. Batch 256 pairs (the official small-batch setting) fits both sizes with room to spare, so no accumulation is needed.

## (ii) One distillation step from cached teacher features

The brief's projector (`Linear(D,2048)–BN–GELU–Linear(2048,384)`), cosine loss against fp16 cached CLS features, one image per sample.

| student | batch | img/s | 100k-image epoch | peak alloc GB |
|---|---|---|---|---|
| S | 256 | 5,666 | 17.6 s | 0.68 |
| S | 512 | 6,387 | 15.7 s | 1.30 |
| S | 1024 | 6,618 | 15.1 s | 2.56 |
| S | 2048 | 6,593 | 15.2 s | 4.48 |
| L | 256 | 2,541 | 39.3 s | 1.26 |
| L | 512 | 2,619 | 38.2 s | 2.28 |
| L | 1024 | 2,588 | 38.6 s | 4.48 |
| L | 2048 | OOM | | |

A C step sees one view per sample, a B step two. At equal batch size and steps (the brief's "same budget"), a C epoch costs ~40 % of a B epoch. Whether C should also see two views per step is for the budget/tuning tickets.

## Loader ceiling with the pool in RAM

Random 2-of-K pairs gathered from an in-RAM uint8 pool (20k × K=4), pinned, copied to GPU and normalised, main process only: **~90k pairs/s**. Once the pool is in RAM the loader is never the limit.

## Bottleneck

Pairs/s at batch 256 pairs:

| stage | pairs/s | source |
|---|---|---|
| pool in RAM → GPU | 90,060 | this ticket |
| memmap, chunk-shuffled (128-runs) | 11,388 | env ticket |
| on-the-fly `VICRegViewTransform`, 14 workers | ~3,300 | env ticket |
| **VICReg GPU step, S** | **2,283** | this ticket |
| **VICReg GPU step, L** | **1,089** | this ticket |
| memmap, per-image random, pool > RAM | 518 | env ticket |
| smoke run end-to-end, S, on-the-fly, 12 workers | 1,951 (AdamW) / 1,790 (LARS) | this ticket |
| teacher forward (one-off caching), img/s | ~1,450 | env ticket |

- **The student GPU step is the bottleneck** for both sizes with any loader except per-image random reads from a pool larger than RAM (518 pairs/s, which would starve even L).
- **On-the-fly augmentation keeps up with S**: the smoke run measured 1.95k pairs/s end to end vs 2.28k GPU-only, with the loop waiting on the loader only 2–5 % of the time; the remaining ~15 % is CPU contention and host→device copies. For L (1.1k pairs/s) on-the-fly augmentation is effectively free. This is relevant to [Cost of fixed views for B](https://github.com/nelliDev/mo433/issues/11): B on-the-fly is cheap; C still needs a pool because the online teacher (1.45k img/s) would be the bottleneck.
- Per-run GPU time at batch 256 pairs, 100k images: B-S ≈ 44–51 s/epoch, B-L ≈ 92 s/epoch; C-S ≈ 18 s, C-L ≈ 39 s (one view per sample). 20 epochs: B-S ≈ 17 min, B-L ≈ 31 min, C-S ≈ 6 min, C-L ≈ 13 min, plus teacher caching once (≈ 9 min for 800k views).

## B smoke run: AdamW vs LARS

S student, all 100k unlabeled images, views augmented on the fly (not the K-pool), batch 256 pairs, bf16, 2-epoch linear warmup + cosine to 0, 20 epochs, seed 0. Evaluation: backbone features, `KNeighborsClassifier(k=20, metric="cosine")` fitted on the 5k train split, scored on the 8k test split (smoke-run only; tuning must not use test), and `std_of_l2_normalized` (healthy ≈ 1/√192 = 0.072).

- AdamW: lr 1e-3, weight decay 0.05 (brief's suggestion).
- LARS (`lightly.utils.lars.LARS`): lr 0.5 × batch/256 = 0.5, momentum 0.9, weight decay 1e-6 (official VICReg small-batch base LR).

| epoch | AdamW k-NN | LARS k-NN | AdamW feat std | LARS feat std |
|---|---|---|---|---|
| 0 (random init) | 29.8 % | 29.8 % | 0.009 | 0.009 |
| 5 | 63.0 % | 56.0 % | 0.041 | 0.039 |
| 10 | 67.7 % | 61.4 % | 0.043 | 0.040 |
| 15 | 70.3 % | 63.4 % | 0.045 | 0.041 |
| 20 | **71.1 %** | **63.4 %** | 0.045 | 0.041 |

Final loss terms (epoch 20): AdamW total 16.98 (inv 0.092, var 0.512, cov 1.90); LARS 18.12 (inv 0.098, var 0.568, cov 1.47).

- **Neither diverged or collapsed.** lightly's "training diverges without LARS" comment did not reproduce here with AdamW at 1e-3 and warmup.
- AdamW is ahead at every checkpoint, by ~7.7 points at epoch 20, and LARS had plateaued by epoch 15. AdamW is also ~8 % faster per epoch (LARS computes per-layer trust ratios).
- Caveats: one seed each; LARS's base LR and weight decay come from the official 1000-epoch ImageNet recipe and were not tuned for a 20-epoch, 1.6M-param, 96 px setting; the two runs also differ in weight decay by design (each optimizer's standard value). This is evidence that AdamW is a safe default, not proof that it beats a tuned LARS.

The GPU was shared with the desktop throughout (~0.85 GB held), so the numbers are slightly pessimistic.
