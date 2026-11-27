"""Throughput benchmarks for the wayfinder ticket
"Environment, data and teacher throughput on the 3060".

  uv run python bench/throughput.py transform   # (i)  VICRegViewTransform at 96 px, CPU workers
  uv run python bench/throughput.py memmap      # (ii) pair loading from a uint8 view memmap
  uv run python bench/throughput.py teacher     # (iii) DINO ViT-S/16 fp16 forward, 96->224

Throwaway: lives on a task branch, never merged.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

DATA = "data/stl10_binary"
N_UNLABELED = 100_000
VIEW_BYTES = 96 * 96 * 3


def load_unlabeled(n=None):
    # STL-10 binary: N x 3 x 96 x 96, column-major per image -> transpose to HWC
    raw = np.fromfile(f"{DATA}/unlabeled_X.bin", dtype=np.uint8, count=-1 if n is None else n * VIEW_BYTES)
    return raw.reshape(-1, 3, 96, 96).transpose(0, 3, 2, 1)


def gpu_mem():
    return {
        "peak_alloc_MiB": round(torch.cuda.max_memory_allocated() / 2**20),
        "peak_reserved_MiB": round(torch.cuda.max_memory_reserved() / 2**20),
    }


# ---------------------------------------------------------------- (i) transform


class TransformDataset(Dataset):
    def __init__(self, images):
        from lightly.transforms import VICRegViewTransform

        self.images = images
        self.t = VICRegViewTransform(input_size=96, min_scale=0.2, gaussian_blur=0.5, sigmas=(0.1, 1.0), normalize=None)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        from PIL import Image

        x = self.t(Image.fromarray(self.images[i]))  # float CHW in [0,1]
        return (x * 255).round().to(torch.uint8).permute(1, 2, 0)  # HWC uint8, the stored format


def bench_transform(args):
    images = load_unlabeled(args.n)
    out = []
    for workers in args.workers:
        dl = DataLoader(TransformDataset(images), batch_size=256, shuffle=True, num_workers=workers,
                        persistent_workers=False, drop_last=True)
        it = iter(dl)
        next(it)  # warm-up: worker spawn
        t0, seen = time.perf_counter(), 0
        for b in it:
            seen += len(b)
        dt = time.perf_counter() - t0
        r = {"workers": workers, "views_per_s": round(seen / dt)}
        out.append(r)
        print(r, flush=True)
    return out


# ---------------------------------------------------------------- (ii) memmap


def build_memmap(path, n, k):
    """Fill an [n, k, 96, 96, 3] uint8 memmap with tiled real images (content doesn't affect read speed)."""
    if os.path.exists(path) and os.path.getsize(path) == n * k * VIEW_BYTES:
        return
    src = load_unlabeled(20_000)
    mm = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=(n, k, 96, 96, 3))
    t0 = time.perf_counter()
    for s in range(0, n, 5000):
        e = min(n, s + 5000)
        idx = np.arange(s, e) % len(src)
        mm[s:e] = src[idx][:, None]
    mm.flush()
    del mm
    print(f"built {path} ({n * k * VIEW_BYTES / 1e9:.1f} GB) in {time.perf_counter() - t0:.0f}s", flush=True)


class PairDataset(Dataset):
    def __init__(self, path, n):
        self.path, self.n, self.mm = path, n, None

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if self.mm is None:  # open lazily, per worker
            self.mm = np.load(self.path, mmap_mode="r")
        k = self.mm.shape[1]
        a, b = np.random.choice(k, 2, replace=False)
        return torch.from_numpy(np.array(self.mm[i, a])), torch.from_numpy(np.array(self.mm[i, b]))


def bench_memmap(args):
    build_memmap(args.memmap, N_UNLABELED, args.k)
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1) * 255
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1) * 255
    out = []
    for n in args.subsets:
        for workers in args.workers:
            dl = DataLoader(PairDataset(args.memmap, n), batch_size=256, shuffle=True, num_workers=workers,
                            pin_memory=True, drop_last=True)
            it = iter(dl)
            next(it)
            t0, seen = time.perf_counter(), 0
            for step, (a, b) in enumerate(it):
                # what the training step will do: H2D + normalise on GPU
                for v in (a, b):
                    v = v.cuda(non_blocking=True).permute(0, 3, 1, 2).float().sub_(mean).div_(std)
                seen += len(a)
                if step >= args.max_steps:
                    break
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            r = {"images": n, "workers": workers, "pairs_per_s": round(seen / dt)}
            out.append(r)
            print(r, flush=True)
    return out


# ---------------------------------------------------------------- (iii) teacher


def bench_teacher(args):
    import timm

    model = timm.create_model("vit_small_patch16_224.dino", pretrained=True, num_classes=0).cuda().eval().half()
    cfg = model.pretrained_cfg
    mean = torch.tensor(cfg["mean"], device="cuda").view(1, 3, 1, 1)
    std = torch.tensor(cfg["std"], device="cuda").view(1, 3, 1, 1)
    out = []
    for bs in args.batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        x8 = torch.randint(0, 256, (bs, 3, 96, 96), dtype=torch.uint8, device="cuda")
        try:
            with torch.no_grad():
                def step():
                    x = F.interpolate(x8.float().div_(255), size=224, mode="bicubic", align_corners=False)
                    x = ((x - mean) / std).half()
                    return model(x)  # [bs, 384] CLS after final norm
                for _ in range(3):
                    step()
                torch.cuda.synchronize()
                iters = max(5, 4096 // bs)
                t0 = time.perf_counter()
                for _ in range(iters):
                    f = step()
                torch.cuda.synchronize()
                dt = time.perf_counter() - t0
        except torch.OutOfMemoryError:
            print({"batch": bs, "oom": True}, flush=True)
            out.append({"batch": bs, "oom": True})
            break
        r = {"batch": bs, "imgs_per_s": round(bs * iters / dt), "feat": list(f.shape), **gpu_mem()}
        out.append(r)
        print(r, flush=True)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("which", choices=["transform", "memmap", "teacher"])
    p.add_argument("--workers", type=int, nargs="+", default=[4, 8, 14])
    p.add_argument("--n", type=int, default=20_000, help="images for the transform bench")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--memmap", default="data/bench_views.npy")
    p.add_argument("--subsets", type=int, nargs="+", default=[50_000, 100_000])
    p.add_argument("--max-steps", type=int, default=150)
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[64, 128, 256, 512, 1024, 2048])
    p.add_argument("--out", default=None)
    args = p.parse_args()
    res = {"transform": bench_transform, "memmap": bench_memmap, "teacher": bench_teacher}[args.which](args)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=2)
