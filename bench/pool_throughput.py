"""Throughput of the real chunk-shuffle loader on the built view pool (issue #17 acceptance).

  uv run python bench/pool_throughput.py [--frac 100 50] [--workers 8] [--epochs 1]

Each measured epoch runs the full training-side path: PoolLoader -> pinned uint8 batch ->
H2D -> normalize_gpu, for B (2 views) and C (1 view). Reports pairs/s (B) and images/s (C).
"""

import argparse
import json
import time

import torch

from mo433 import paths
from mo433.pool import open_pool
from mo433.sampler import ChunkShuffleSampler, PoolLoader, normalize_gpu


def measure(pool_path, n_images, k, views, workers, epochs, max_batches, seed):
    s = ChunkShuffleSampler(n_images, k, views, seed=seed, drop_last=True)
    loader = PoolLoader(pool_path, s, num_workers=workers)
    results = []
    for _ in range(epochs):
        t0, seen = time.perf_counter(), 0
        for b, (_, _, x) in enumerate(loader):
            for v in range(views):
                normalize_gpu(x[:, v], "cuda")
            seen += len(x)
            if max_batches and b + 1 >= max_batches:
                break
        torch.cuda.synchronize()
        results.append(round(seen / (time.perf_counter() - t0)))
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default="full")
    p.add_argument("--frac", type=int, nargs="+", default=[100, 50])
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-batches", type=int, default=0)
    args = p.parse_args()
    pool_dir = paths.pool_dir(args.profile)
    views, meta = open_pool(pool_dir)
    n, k = views.shape[:2]
    out = []
    for frac in args.frac:
        for row, nv in (("B", 2), ("C", 1)):
            r = measure(pool_dir / "views.npy", n * frac // 100, k, nv, args.workers, args.epochs, args.max_batches, 0)
            res = {"frac": frac, "row": row, "views_per_image": nv, "workers": args.workers,
                   "per_s_by_epoch": r, "unit": "pairs/s" if nv == 2 else "images/s"}
            out.append(res)
            print(json.dumps(res), flush=True)
    return out


if __name__ == "__main__":
    main()
