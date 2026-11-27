"""Student-side throughput and optimizer smoke run for the wayfinder ticket
"Student throughput and optimizer smoke run".

  .venv/bin/python bench/student.py vicreg    # (i)  one VICReg step (two views) per student size
  .venv/bin/python bench/student.py distill   # (ii) one distillation step from cached teacher features
  .venv/bin/python bench/student.py feed      # host->GPU feed of uint8 pairs from RAM (no disk, no aug)
  .venv/bin/python bench/student.py smoke --opt adamw   # B smoke run, S student, on-the-fly views
  .venv/bin/python bench/student.py smoke --opt lars

Throwaway: lives on a task branch, never merged.
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "docs", "student-design"))
from bench_throughput import ResNet96  # noqa: E402  (the decided student family)

DATA = "data/stl10_binary"
SIZES = {"S": 24, "L": 48}
TEACHER_DIM = 384
DEV = "cuda"
AMP = dict(device_type="cuda", dtype=torch.bfloat16)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255
torch.backends.cudnn.benchmark = True


def load_split(name, n=None):
    # STL-10 binary: N x 3 x 96 x 96, column-major per image -> NHWC uint8
    raw = np.fromfile(f"{DATA}/{name}_X.bin", dtype=np.uint8, count=-1 if n is None else n * 96 * 96 * 3)
    return raw.reshape(-1, 3, 96, 96).transpose(0, 3, 2, 1)


def to_input(u8):
    """uint8 NHWC on GPU -> normalized float NCHW, channels_last (what every training step does)."""
    x = u8.permute(0, 3, 1, 2).float()
    x = (x - MEAN.to(x.device)) / STD.to(x.device)
    return x.contiguous(memory_format=torch.channels_last)


def mem():
    return {"peak_alloc_GB": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "peak_reserved_GB": round(torch.cuda.max_memory_reserved() / 2**30, 2)}


def timed(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters


def make_lars(params, lr):
    from lightly.utils.lars import LARS

    return LARS(params, lr=lr, momentum=0.9, weight_decay=1e-6)


# ---------------------------------------------------------------- (i) VICReg step


class VICRegModel(nn.Module):
    def __init__(self, w):
        super().__init__()
        from lightly.models.modules.heads import VICRegProjectionHead

        self.backbone = ResNet96(w)
        self.head = VICRegProjectionHead(self.backbone.dim, 2048, 2048, num_layers=3)

    def forward(self, x):
        return self.head(self.backbone(x))


def vicreg_step_fn(model, opt, loss_fn, v1, v2):
    def step():
        with torch.autocast(**AMP):
            # one forward over both views: same BN batch statistics as lightly's examples use per view
            z = model(torch.cat([to_input(v1), to_input(v2)]))
        z1, z2 = z.float().chunk(2)
        loss = loss_fn(z1, z2)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    return step


def vicreg_step_fn_sep(model, opt, loss_fn, v1, v2):
    def step():
        with torch.autocast(**AMP):
            z1, z2 = model(to_input(v1)), model(to_input(v2))
        loss = loss_fn(z1.float(), z2.float())
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    return step


def bench_vicreg(args):
    from lightly.loss import VICRegLoss

    out = []
    for size, w in SIZES.items():
        for pairs in args.pairs:
            for mode, fn in (("separate", vicreg_step_fn_sep), ("concat", vicreg_step_fn)):
                if mode == "concat" and pairs > 256:
                    continue  # BN-batch variant only checked at the default batch
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                model = VICRegModel(w).to(DEV, memory_format=torch.channels_last).train()
                opt = make_lars(model.parameters(), 0.1)
                v1 = torch.randint(0, 256, (pairs, 96, 96, 3), dtype=torch.uint8, device=DEV)
                v2 = torch.randint(0, 256, (pairs, 96, 96, 3), dtype=torch.uint8, device=DEV)
                r = {"size": size, "w": w, "pairs": pairs, "images": 2 * pairs, "forward": mode}
                try:
                    dt = timed(fn(model, opt, VICRegLoss(), v1, v2), args.iters, warmup=5)
                    r |= {"pairs_per_s": round(pairs / dt), "img_per_s": round(2 * pairs / dt),
                          "epoch_100k_s": round(100_000 / (pairs / dt), 1)} | mem()
                except torch.OutOfMemoryError:
                    r |= {"oom": True}
                del model, opt, v1, v2
                out.append(r)
                print(r, flush=True)
    return out


# ---------------------------------------------------------------- (ii) distillation step


class DistillModel(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.backbone = ResNet96(w)
        d = self.backbone.dim
        # the brief's projector, verbatim
        self.head = nn.Sequential(nn.Linear(d, 2048), nn.BatchNorm1d(2048), nn.GELU(), nn.Linear(2048, TEACHER_DIM))

    def forward(self, x):
        return self.head(self.backbone(x))


def bench_distill(args):
    out = []
    for size, w in SIZES.items():
        for batch in args.batches:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            model = DistillModel(w).to(DEV, memory_format=torch.channels_last).train()
            opt = make_lars(model.parameters(), 0.1)
            x = torch.randint(0, 256, (batch, 96, 96, 3), dtype=torch.uint8, device=DEV)
            t = torch.randn(batch, TEACHER_DIM, device=DEV).half()  # cached fp16 teacher CLS

            def step():
                with torch.autocast(**AMP):
                    s = model(to_input(x))
                loss = 1 - F.cosine_similarity(s.float(), t.float(), dim=-1).mean()
                loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)

            r = {"size": size, "w": w, "batch": batch}
            try:
                dt = timed(step, args.iters, warmup=5)
                r |= {"img_per_s": round(batch / dt), "epoch_100k_s": round(100_000 / (batch / dt), 1)} | mem()
            except torch.OutOfMemoryError:
                r |= {"oom": True}
            del model, opt, x, t
            out.append(r)
            print(r, flush=True)
    return out


# ---------------------------------------------------------------- feed: RAM pool -> GPU


def bench_feed(args):
    """Ceiling of a loader once disk is out of the picture: random pairs gathered from an
    in-RAM uint8 pool [N, K, 96, 96, 3], pinned, copied to GPU, normalized."""
    n, k, pairs = args.n, args.k, 256
    src = load_split("unlabeled", n)
    pool = torch.from_numpy(np.ascontiguousarray(np.repeat(src[:, None], k, axis=1)))
    out = []
    for workers in (0,):
        pin1 = torch.empty((pairs, 96, 96, 3), dtype=torch.uint8).pin_memory()
        pin2 = torch.empty_like(pin1).pin_memory()
        g = torch.Generator().manual_seed(0)

        def batch():
            i = torch.randint(0, n, (pairs,), generator=g)
            a = torch.randint(0, k, (pairs,), generator=g)
            b = (a + torch.randint(1, k, (pairs,), generator=g)) % k  # distinct second view
            torch.index_select(pool.view(n * k, 96, 96, 3), 0, i * k + a, out=pin1)
            torch.index_select(pool.view(n * k, 96, 96, 3), 0, i * k + b, out=pin2)
            return to_input(pin1.to(DEV, non_blocking=True)), to_input(pin2.to(DEV, non_blocking=True))

        dt = timed(batch, 200, warmup=10)
        r = {"pool": f"{n}x{k}", "pool_GB": round(pool.numel() / 1e9, 2), "workers": workers,
             "pairs_per_s": round(pairs / dt)}
        out.append(r)
        print(r, flush=True)
    return out


# ---------------------------------------------------------------- smoke run


class OnTheFlyPairs(Dataset):
    def __init__(self, images):
        from lightly.transforms import VICRegViewTransform

        self.images = images
        self.t = VICRegViewTransform(input_size=96, min_scale=0.2, gaussian_blur=0.5, sigmas=(0.1, 1.0), normalize=None)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        from PIL import Image

        img = Image.fromarray(self.images[i])
        u8 = lambda v: (v * 255).round().to(torch.uint8).permute(1, 2, 0)  # noqa: E731
        return u8(self.t(img)), u8(self.t(img))


@torch.no_grad()
def embed(backbone, images, bs=1024):
    backbone.eval()
    feats = []
    for s in range(0, len(images), bs):
        u8 = torch.from_numpy(np.ascontiguousarray(images[s:s + bs])).to(DEV)
        with torch.autocast(**AMP):
            feats.append(backbone(to_input(u8)).float().cpu())
    backbone.train()
    return torch.cat(feats).numpy()


def evaluate(backbone, tr, ytr, te, yte):
    from lightly.utils.debug import std_of_l2_normalized
    from sklearn.neighbors import KNeighborsClassifier

    ftr, fte = embed(backbone, tr), embed(backbone, te)
    knn = KNeighborsClassifier(n_neighbors=20, metric="cosine").fit(ftr, ytr)  # k = min(20, 500/class)
    return {"knn_top1": round(float(knn.score(fte, yte)), 4),
            "feat_std_l2n": round(float(std_of_l2_normalized(torch.from_numpy(fte))), 4),
            "feat_std_l2n_healthy": round(1 / math.sqrt(fte.shape[1]), 4)}


def smoke(args):
    from lightly.loss import VICRegLoss
    from lightly.loss.vicreg_loss import covariance_loss, invariance_loss, variance_loss

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    unl = load_split("unlabeled", args.n)
    tr, te = load_split("train"), load_split("test")
    ytr = np.fromfile(f"{DATA}/train_y.bin", dtype=np.uint8)
    yte = np.fromfile(f"{DATA}/test_y.bin", dtype=np.uint8)

    model = VICRegModel(SIZES[args.size]).to(DEV, memory_format=torch.channels_last).train()
    steps_per_epoch = len(unl) // args.pairs
    total = steps_per_epoch * args.epochs
    warm = steps_per_epoch * args.warmup
    if args.opt == "adamw":
        base_lr = args.lr or 1e-3
        opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=args.wd if args.wd is not None else 0.05)
    else:
        base_lr = (args.lr or 0.5) * args.pairs / 256  # official VICReg: base 0.5 @ batch 256
        opt = make_lars(model.parameters(), base_lr)
        if args.wd is not None:
            for g in opt.param_groups:
                g["weight_decay"] = args.wd
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / (total - warm))))
    loss_fn = VICRegLoss()
    dl = DataLoader(OnTheFlyPairs(unl), batch_size=args.pairs, shuffle=True, num_workers=args.workers,
                    pin_memory=True, drop_last=True, persistent_workers=True, prefetch_factor=4)

    log = {"args": vars(args), "base_lr": base_lr, "init": evaluate(model.backbone, tr, ytr, te, yte), "epochs": []}
    print("init", log["init"], flush=True)
    for ep in range(args.epochs):
        t0, sums, n, wait = time.perf_counter(), np.zeros(4), 0, 0.0
        tw = time.perf_counter()
        for v1, v2 in dl:
            wait += time.perf_counter() - tw
            v1, v2 = v1.to(DEV, non_blocking=True), v2.to(DEV, non_blocking=True)
            with torch.autocast(**AMP):
                z1, z2 = model(to_input(v1)), model(to_input(v2))
            z1, z2 = z1.float(), z2.float()
            loss = loss_fn(z1, z2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            with torch.no_grad():
                sums += [loss.item(), invariance_loss(z1, z2).item(),
                         0.5 * (variance_loss(z1, 1e-4) + variance_loss(z2, 1e-4)).item(),
                         covariance_loss(z1).item() + covariance_loss(z2).item()]
            n += 1
            tw = time.perf_counter()
        dt = time.perf_counter() - t0
        m = sums / n
        e = {"epoch": ep + 1, "loss": round(m[0], 4), "inv": round(m[1], 4), "var": round(m[2], 4),
             "cov": round(m[3], 4), "lr": round(sched.get_last_lr()[0], 6), "epoch_s": round(dt, 1),
             "pairs_per_s": round(n * args.pairs / dt), "loader_wait_frac": round(wait / dt, 3),
             "nan": bool(not np.isfinite(m[0]))}
        if (ep + 1) % args.eval_every == 0 or ep + 1 == args.epochs:
            e |= evaluate(model.backbone, tr, ytr, te, yte)
        log["epochs"].append(e)
        print(e, flush=True)
        if e["nan"]:
            break
    return log


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("vicreg")
    s.add_argument("--pairs", type=int, nargs="+", default=[128, 256, 512, 1024])
    s.add_argument("--iters", type=int, default=20)
    s = sub.add_parser("distill")
    s.add_argument("--batches", type=int, nargs="+", default=[256, 512, 1024, 2048])
    s.add_argument("--iters", type=int, default=20)
    s = sub.add_parser("feed")
    s.add_argument("--n", type=int, default=20_000)
    s.add_argument("--k", type=int, default=4)
    s = sub.add_parser("smoke")
    s.add_argument("--opt", choices=["adamw", "lars"], required=True)
    s.add_argument("--size", default="S")
    s.add_argument("--n", type=int, default=100_000)
    s.add_argument("--epochs", type=int, default=20)
    s.add_argument("--warmup", type=int, default=2)
    s.add_argument("--pairs", type=int, default=256)
    s.add_argument("--lr", type=float, default=None)
    s.add_argument("--wd", type=float, default=None)
    s.add_argument("--workers", type=int, default=12)
    s.add_argument("--eval-every", type=int, default=5)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--tag", default="")
    args = p.parse_args()

    res = {"vicreg": bench_vicreg, "distill": bench_distill, "feed": bench_feed, "smoke": smoke}[args.cmd](args)
    os.makedirs("bench/results", exist_ok=True)
    name = args.cmd if args.cmd != "smoke" else f"smoke_{args.opt}{args.tag}"
    with open(f"bench/results/{name}.json", "w") as f:
        json.dump({"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "results": res}, f, indent=1)


if __name__ == "__main__":
    main()
