# Environment, data and teacher throughput on the 3060

Measured 2026-11-03 for the wayfinder ticket [#6](https://github.com/nelliDev/mo433/issues/6). Script: `bench/throughput.py`; raw JSON in `bench/results/`.

## Environment

- uv project, Python 3.12 (torch has no 3.15 wheels; system Python is 3.15), `.python-version` pinned.
- torch 2.11.0+cu128 / torchvision 0.26.0+cu128 (explicit `pytorch-cu128` index), timm 1.0.31, lightly 1.6.0, scikit-learn 1.9.1. Driver reports CUDA 13.4; cu128 wheels work.
- Machine: RTX 3060 Laptop 6 GB (~1.1 GB already held by the desktop/browser, GPU shared with them during runs), 16 threads, 15 GB RAM, `/home` is btrfs with `compress=zstd:3`, 128 GB free.

## Data

- STL-10 binary: `ai.stanford.edu` timed out, so the identical tarball came from the HF mirror `xingslong/stl10_binary`. MD5 `91f7769df0f17e558f3565bffb0c7dfb` matches torchvision's `tgz_md5`. Extracted to `data/stl10_binary/` (2.9 GB): 100k unlabeled, 5k train (10 folds), 8k test. Loads with `torchvision.datasets.STL10('data', download=False)`.

## (i) VICRegViewTransform at 96 px (CPU, PIL, uint8 output)

`VICRegViewTransform(input_size=96, min_scale=0.2, gaussian_blur=0.5, sigmas=(0.1,1.0), normalize=None)`, batch 256, 20k images.

| workers | views/s |
|---|---|
| 4 | 3,102 |
| 8 | 5,370 |
| 14 | 6,614 |

1M views (100k × K=10) ≈ 2.5 min CPU, plus the disk write. On-the-fly augmentation would supply about 3.3k **pairs**/s with 14 workers, but those workers compete with the training process for CPU.

## (ii) Loading view pairs from a uint8 memmap `[N, K, 96, 96, 3]`

K=8 over 100k images is a **22.1 GB** file, larger than RAM (15 GB, about 6–11 GB free for page cache). The test file held tiled real images. Real augmented views may compress differently on zstd btrfs.

| sampler | images | workers | pairs/s |
|---|---|---|---|
| per-image random (`shuffle=True`) | 100k | 4 / 8 / 14 | 471 / 510 / 518 |
| per-image random | 50k | 4 / 8 / 14 | 2,144 / 24,354 / 8,280 (page-cache warm-up; noisy) |
| chunk-shuffled, runs of 32 images | 100k | 4 / 8 | 5,966 / 7,703 |
| chunk-shuffled, runs of 128 images | 100k | 4 / 8 | 11,388 / 10,325 |

Sequential read: about 1.0 GB/s. Building the 22 GB file took 261 s.

**Takeaway:** with fully random access, a pool larger than RAM is disk-bound at about 500 pairs/s, which would starve training. Either shuffle chunks (shuffle the order of contiguous runs of about 128 images, then pick 2 of the K views per image), or keep the pool small enough to stay in page cache (≲ 6 GB, e.g. 50k × K=4). Put the pool in a `chattr +C` / uncompressed directory.

## (iii) Teacher forward: timm `vit_small_patch16_224.dino`, fp16, no_grad

uint8 96 px on GPU → bicubic 224 → ImageNet normalise → `.half()` model → 384-d CLS.

| batch | img/s | peak alloc MiB | peak reserved MiB |
|---|---|---|---|
| 64 | 1,503 | 194 | 250 |
| 128 | 1,528 | 334 | 506 |
| 256 | 1,448 | 615 | 802 |
| 512 | 1,423 | 1,175 | 1,738 |
| 1024 | 1,393 | 2,298 | 3,860 |
| 2048 | OOM | | |

Throughput levels off from batch 128 at about 1.45k img/s, so feature caching is one-off and cheap:

- 100k × K=8 = 800k views ≈ **9 min**; K=10 ≈ 11.5 min; plus 5k train + 8k test (unaugmented) ≈ 10 s.
- The feature cache is small: 100k × 8 × 384 × fp16 = **0.6 GB**, which fits in RAM or VRAM.

## For the budget decision

The pre-generation steps (transform plus teacher) cost minutes, not hours. The real constraint is **disk I/O of a view pool larger than RAM**, so the sampler and pool size have to be designed around it. Student-side step throughput (the next ticket, "Student throughput and optimizer smoke run") decides whether 6–11k pairs/s from a chunk-shuffled memmap is enough.
