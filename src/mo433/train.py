"""Row-agnostic training loop shared by rows B and C (spec §5.1).

Entry point: :func:`train_run` (programmatic, used by the CLI and the tuning studies).

What the loop owns, identically for every row:

- the student (``build_student(size, seed)``, the same init as row A seed s) plus the row's
  discarded head, bf16 autocast on CUDA, channels_last, GPU normalization of uint8 views;
- AdamW (betas from the config, weight decay on all params), linear warmup then per-step
  cosine to 0 over the whole run (:func:`lr_factor`);
- the data: a :class:`BatchSource` yielding :class:`Batch` es of pool views (or fresh views
  for the fixed-view control), a pure function of (seed, epoch) so resume is exact;
- checkpoints every ``checkpoint_every`` epochs and at the end, exact resume, final
  backbone weights for ``load_encoder``, and the run directory (layout below).

What a row supplies is an :class:`Objective` (``train_b.VICRegObjective``; row C adds its
own and registers it with :func:`register_objective`).

Run directory (``paths.run_dir(run_id, profile)`` unless ``run_dir=`` is given)::

    config.json         resolved settings, the loaded Config, pool/data info, git commit +
                        dirty flag, torch/GPU versions, start time
    metrics.jsonl       one JSON object per epoch (keys: see ``_epoch_record``)
    checkpoint.pt       latest resumable state (model, head, optimizer, scheduler, sampler
                        state, epoch, global step, python/numpy/torch/CUDA RNG, timers)
    final_backbone.pt   backbone state_dict only, written when the last epoch finishes
    summary.json        written at the end: epochs/steps, wall + GPU seconds, peak memory,
                        last epoch's metrics

``emb_std`` in the log is the spec §7.3 statistic: per-dimension std of the L2-normalized
backbone features × √d (healthy ≈ 1, collapsed → 0). ``emb_std`` is averaged over the
epoch's training batches (first view, train-mode BN); ``val_emb_std`` is on the 5k labeled
train images in eval mode whenever the validation k-NN runs.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import random
import shutil
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mo433 import config as config_mod
from mo433 import paths, stages

CHUNK = 64  # images per chunk of the chunk-shuffle sampler (spec §3)
CHECKPOINT = "checkpoint.pt"
METRICS = "metrics.jsonl"
SUMMARY = "summary.json"
CONFIG = "config.json"

# Settings that may differ on resume without breaking exactness of the schedule.
_RESUME_FREE = {"knn_every", "num_workers", "checkpoint_every"}
OVERRIDABLE = {
    "lr", "weight_decay", "warmup_epochs", "head_width", "epochs", "batch_size", "n_images",
    "knn_every", "checkpoint_every", "num_workers", "device", "optimizer",
}
OPTIMIZERS = ("adamw", "lars")
LARS_MOMENTUM = 0.9


# ---------------------------------------------------------------- integration points
# One place for every API owned by another module.


class _deps:
    from mo433.data import load_stl10, normalize
    from mo433.encoders import save_final_backbone
    from mo433.knn import val_knn
    from mo433.pool import open_pool
    from mo433.sampler import ChunkShuffleSampler, PoolLoader
    from mo433.students import build_student


# ---------------------------------------------------------------- schedule


def lr_factor(step: int, warmup_steps: int, total_steps: int) -> float:
    """Multiplier of the base LR for optimizer update ``step`` (0-based).

    Linear warmup reaching 1 at the last warmup step, then cosine decay reaching exactly 0
    at the last step of the run (``total_steps - 1``).
    """
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps + 1) / max(total_steps - warmup_steps, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


# ---------------------------------------------------------------- settings


@dataclass(frozen=True)
class RunSettings:
    """Everything that defines a run, resolved from the config, CLI arguments and overrides."""

    row: str
    size: str
    frac: int
    seed: int
    profile: str
    run_id: str
    n_images: int  # N of the full pool; the run reads pool images [0, n_images * frac / 100)
    k_views: int
    epochs: int
    batch_size: int
    head_width: int
    lr: float
    weight_decay: float
    warmup_epochs: int
    betas: tuple[float, float]
    optimizer: str = "adamw"  # "lars": LARS check (spec §6.3) only; see build_optimizer
    on_the_fly: bool = False
    knn_every: int = 1  # validation k-NN every N epochs (and at the last); 0 = never
    checkpoint_every: int = 10
    num_workers: int = 8
    device: str = "cuda"

    @property
    def n_used(self) -> int:
        return self.n_images * self.frac // 100

    @property
    def steps_per_epoch(self) -> int:
        return self.n_used // self.batch_size

    @property
    def total_steps(self) -> int:
        return self.steps_per_epoch * self.epochs

    @property
    def warmup_steps(self) -> int:
        # Capped at half the run so short runs (smoke: E=2, warmup 2) still decay to 0;
        # never binds for E > 2 * warmup_epochs (every real run and tuning trial).
        return min(self.steps_per_epoch * self.warmup_epochs, self.total_steps // 2)


def resolve_settings(
    row: str,
    size: str,
    frac: int,
    seed: int,
    profile: str = "full",
    overrides: dict[str, Any] | None = None,
    epochs: int | None = None,
    on_the_fly: bool = False,
) -> RunSettings:
    cfg = config_mod.load_config(row, size, profile)
    overrides = dict(overrides or {})
    unknown = set(overrides) - OVERRIDABLE
    if unknown:
        raise ValueError(f"unknown overrides {sorted(unknown)}; allowed: {sorted(OVERRIDABLE)}")
    if epochs is not None:
        overrides["epochs"] = epochs
    if frac not in config_mod.FRACS:
        raise ValueError(f"frac must be one of {config_mod.FRACS}")
    if on_the_fly and row != "B":
        raise ValueError("on-the-fly views apply to row B only")
    base = dict(
        row=row, size=size, frac=frac, seed=seed, profile=profile,
        run_id=config_mod.run_id(row, size, frac, seed),
        n_images=cfg.n_images, k_views=cfg.k_views, epochs=cfg.epochs, batch_size=cfg.batch_size,
        head_width=cfg.head_width, lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay,
        warmup_epochs=cfg.optimizer.warmup_epochs, betas=tuple(cfg.optimizer.betas), on_the_fly=on_the_fly,
        device="cuda" if torch.cuda.is_available() else "cpu",
        # $MO433_NUM_WORKERS caps loader workers (tests, `all --profile smoke` on a shared box).
        num_workers=int(os.environ.get("MO433_NUM_WORKERS", RunSettings.num_workers)),
    )
    s = RunSettings(**{**base, **overrides})
    if s.steps_per_epoch < 1:
        raise ValueError(f"{s.n_used} images < one batch of {s.batch_size}")
    return s


# ---------------------------------------------------------------- objectives


@dataclass
class Batch:
    """One step's data. On the training device when it reaches an objective.

    ``index``: pool positions [B] (row j of the pool = unlabeled image ``pool_order[j]``).
    ``view_index``: pool view ids [B, V] (-1 for on-the-fly views, which are not in the pool).
    ``views``: uint8 NHWC views [B, V, 96, 96, 3] (the pool loader's layout); ``views[i, v]`` is
    view ``view_index[i, v]`` of image ``index[i]``.
    """

    index: torch.Tensor
    view_index: torch.Tensor
    views: torch.Tensor

    def pin_memory(self) -> Batch:  # used by the DataLoader's pin-memory thread
        return Batch(self.index.pin_memory(), self.view_index.pin_memory(), self.views.pin_memory())

    def to(self, device: str) -> Batch:
        nb = device != "cpu"
        return Batch(*(t.to(device, non_blocking=nb) for t in (self.index, self.view_index, self.views)))


class Objective(Protocol):
    """What a row plugs into the loop.

    ``views_per_sample``: distinct pool views drawn per image per step (B: 2, C: 1).
    ``build_head(d)``: the discarded head on top of the d-dim backbone feature.
    ``loss(head, features, batch)``: ``features[v]`` is the backbone output for ``batch.views[v]``
    (``batch.views[:, v]``, computed under the loop's autocast); returns the scalar loss and a dict of detached
    scalar tensors logged per epoch (epoch means). Must not call ``.item()`` (no host syncs).
    """

    name: str
    views_per_sample: int

    def build_head(self, feature_dim: int) -> nn.Module: ...

    def loss(self, head: nn.Module, features: list[torch.Tensor], batch: Batch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]: ...


ObjectiveFactory = Callable[[RunSettings], Objective]
_OBJECTIVES: dict[str, ObjectiveFactory] = {}


def register_objective(row: str, factory: ObjectiveFactory) -> None:
    _OBJECTIVES[row] = factory


def _objective_for(s: RunSettings) -> Objective:
    if s.row not in _OBJECTIVES:
        if s.row == "B":
            from mo433.train_b import VICRegObjective

            register_objective("B", lambda st: VICRegObjective(st.head_width))
        elif s.row == "C":
            import mo433.train_c  # noqa: F401  (registers "C" on import)
    return _OBJECTIVES[s.row](s)


# ---------------------------------------------------------------- data


class BatchSource(Protocol):
    steps_per_epoch: int

    def batches(self, epoch: int) -> Iterator[Batch]:
        """The batches of 0-based ``epoch``; a pure function of (seed, epoch)."""

    def state_dict(self) -> dict: ...


def _iterate(dataset, num_workers: int, pin: bool) -> Iterator[Batch]:
    """One epoch's batches. Workers are forked per epoch (datasets hold the epoch's plan);
    they only read the memmap / run PIL transforms, never touch CUDA."""
    if num_workers == 0:
        for i in range(len(dataset)):
            yield dataset[i].pin_memory() if pin else dataset[i]
        return
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=None, shuffle=False, num_workers=num_workers, collate_fn=_identity,
        pin_memory=pin, prefetch_factor=4, multiprocessing_context="fork", worker_init_fn=_seed_worker,
    )
    yield from loader


def _identity(b):
    return b


def _seed_worker(worker_id: int) -> None:
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)
    torch.set_num_threads(1)


class PoolSource:
    """Chunk-shuffled batches of V distinct pool views per image (``mo433.sampler``, spec §3):
    pool rows [0, n_used), drop-last, a pure function of (seed, epoch)."""

    def __init__(self, s: RunSettings, views_per_sample: int):
        views, _ = _deps.open_pool(paths.pool_dir(s.profile))
        if views.shape[1] != s.k_views or views.shape[0] < s.n_used:
            raise ValueError(f"pool {views.shape[:2]} does not provide N>={s.n_used}, K={s.k_views}")
        self.sampler = _deps.ChunkShuffleSampler(
            s.n_used, s.k_views, views_per_sample, batch_size=s.batch_size, chunk_size=CHUNK, seed=s.seed,
            drop_last=True,
        )
        self.loader = _deps.PoolLoader(
            Path(views.filename), self.sampler, num_workers=s.num_workers, pin_memory=s.device.startswith("cuda")
        )
        self.steps_per_epoch = len(self.loader)
        assert self.steps_per_epoch == s.steps_per_epoch

    def batches(self, epoch: int) -> Iterator[Batch]:
        self.loader.load_state_dict({"seed": self.sampler.seed, "epoch": epoch, "batch": 0})
        for index, view_index, views in self.loader:
            yield Batch(index, view_index, views)

    def state_dict(self) -> dict:
        return {"kind": "pool-chunk-shuffle", **self.loader.state_dict()}


class _OnTheFlyEpoch(torch.utils.data.Dataset):
    """Item ``step`` -> a Batch of two fresh VICRegViewTransform views of each image."""

    def __init__(self, rows: np.ndarray, image_ids: np.ndarray, batch_size: int):
        self.rows, self.image_ids, self.bs = rows, image_ids, batch_size
        self.images = self.transform = None

    def __len__(self):
        return len(self.rows) // self.bs

    def __getitem__(self, step: int) -> Batch:
        from PIL import Image

        if self.images is None:
            self.images = _unlabeled_images()
            self.transform = _view_transform()
        rows = self.rows[step * self.bs : (step + 1) * self.bs]
        out = np.empty((len(rows), 2, 96, 96, 3), dtype=np.uint8)
        for i, r in enumerate(rows):
            img = Image.fromarray(np.ascontiguousarray(self.images[self.image_ids[r]]))
            for v in range(2):
                out[i, v] = self.transform(img).mul(255).round_().to(torch.uint8).permute(1, 2, 0).numpy()
        return Batch(torch.from_numpy(rows), torch.full((len(rows), 2), -1), torch.from_numpy(out))


def _unlabeled_images() -> np.ndarray:
    from mo433.data import read_images

    return read_images(paths.stl10_binary_dir() / "unlabeled_X.bin")


def _view_transform():
    from lightly.transforms import VICRegViewTransform

    return VICRegViewTransform(input_size=96, min_scale=0.2, gaussian_blur=0.5, sigmas=(0.1, 1.0), normalize=None)


class OnTheFlySource:
    """Fixed-view control (spec §6.4): fresh views every step from the raw unlabeled images,
    same images as the pool run (pool positions [0, n_used), i.e. ``pool_order[:n_used]``),
    reshuffled per epoch. Views themselves are not reproducible across resumes."""

    def __init__(self, s: RunSettings, views_per_sample: int):
        from mo433 import splits

        if views_per_sample != 2:
            raise ValueError("on-the-fly views are for row B (2 views)")
        self.s = s
        self.image_ids = splits.pool_order()
        self.steps_per_epoch = s.steps_per_epoch
        self.pin = s.device.startswith("cuda")

    def batches(self, epoch: int) -> Iterator[Batch]:
        rows = np.random.default_rng([self.s.seed, epoch]).permutation(self.s.n_used)
        ds = _OnTheFlyEpoch(rows[: self.steps_per_epoch * self.s.batch_size], self.image_ids, self.s.batch_size)
        return _iterate(ds, self.s.num_workers, self.pin)

    def state_dict(self) -> dict:
        return {"kind": "on-the-fly", "seed": self.s.seed}


# ---------------------------------------------------------------- model


class Model(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone, self.head = backbone, head


def emb_std(features: torch.Tensor) -> torch.Tensor:
    """Spec §7.3: per-dimension std of L2-normalized features × √d (1 = spread, 0 = collapsed)."""
    f = F.normalize(features.float(), dim=1)
    return f.std(dim=0).mean() * math.sqrt(f.shape[1])


class _Validator:
    """Validation k-NN on the labeled train split (monitoring only, spec §5.1)."""

    def __init__(self, device: str):
        images, labels = _deps.load_stl10("train")
        self.images = images.to(device)
        self.labels = labels.to(device)

    @torch.no_grad()
    def __call__(self, backbone: nn.Module, amp: bool) -> dict[str, float]:
        was = backbone.training
        backbone.eval()
        feats = []
        for start in range(0, len(self.images), 1000):
            with torch.autocast(self.images.device.type, dtype=torch.bfloat16, enabled=amp):
                feats.append(backbone(_input(self.images[start : start + 1000])).float())
        backbone.train(was)
        f = torch.cat(feats)
        return {"val_knn": float(_deps.val_knn(f, self.labels)), "val_emb_std": float(emb_std(f))}


def _input(u8: torch.Tensor) -> torch.Tensor:
    return _deps.normalize(u8).contiguous(memory_format=torch.channels_last)


# ---------------------------------------------------------------- the loop


class TrainingDiverged(RuntimeError):
    pass


def train_run(
    row: str,
    size: str,
    frac: int,
    seed: int,
    profile: str = "full",
    overrides: dict[str, Any] | None = None,
    run_dir: Path | str | None = None,
    epochs: int | None = None,
    epoch_callback: Callable[[int, dict[str, Any]], None] | None = None,
    on_the_fly: bool = False,
    fresh: bool = False,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Train one run (resuming from ``run_dir/checkpoint.pt`` if present) and return its summary.

    ``overrides``: any of ``OVERRIDABLE`` (lr, weight_decay, warmup_epochs, head_width,
    epochs, batch_size, n_images, knn_every, checkpoint_every, num_workers, device).
    ``epoch_callback(epoch, metrics)`` runs after each epoch is logged and checkpointed
    (1-based epoch, the same dict as the metrics.jsonl line); an exception it raises (e.g.
    ``optuna.TrialPruned``) propagates and leaves the run resumable from its last checkpoint.
    ``fresh=True`` deletes ``run_dir`` first. Raises :class:`TrainingDiverged` on a
    non-finite epoch loss (after logging the epoch).
    """
    wall_start = time.perf_counter()
    s = resolve_settings(row, size, frac, seed, profile, overrides, epochs, on_the_fly)
    rd = Path(run_dir) if run_dir is not None else paths.run_dir(s.run_id, profile)
    if fresh and rd.exists():
        shutil.rmtree(rd)
    rd.mkdir(parents=True, exist_ok=True)
    device = s.device
    cuda = device.startswith("cuda")
    if cuda:
        torch.backends.cudnn.benchmark = True
        torch.cuda.reset_peak_memory_stats()

    random.seed(s.seed)
    np.random.seed(s.seed)
    torch.manual_seed(s.seed)
    objective = _objective_for(s)
    backbone = _deps.build_student(s.size, s.seed)
    model = Model(backbone, objective.build_head(backbone.dim)).to(device, memory_format=torch.channels_last)
    model.train()
    opt = build_optimizer(model, s)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: lr_factor(step, s.warmup_steps, s.total_steps))
    source: BatchSource = (OnTheFlySource if s.on_the_fly else PoolSource)(s, objective.views_per_sample)

    start_epoch, step, wall_prev, gpu_total = 0, 0, 0.0, 0.0
    ckpt_path = rd / CHECKPOINT
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        _check_resumable(ck["settings"], s)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        start_epoch, step = ck["epoch"], ck["step"]
        wall_prev, gpu_total = ck["wall_seconds"], ck["gpu_seconds"]
        _set_rng(ck["rng"])
        _truncate_metrics(rd / METRICS, start_epoch)
        log(f"[train {s.run_id}] resumed at epoch {start_epoch}/{s.epochs}, step {step}")
    else:
        (rd / METRICS).unlink(missing_ok=True)
    _write_json(rd / CONFIG, _config_record(s, source, objective))

    validator = _Validator(device) if s.knn_every > 0 and start_epoch < s.epochs else None
    record: dict[str, Any] = {}
    for epoch in range(start_epoch, s.epochs):
        t0 = time.perf_counter()
        sums: dict[str, torch.Tensor] = {}
        events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        cpu_step_seconds, wait = 0.0, 0.0
        tw = time.perf_counter()
        for batch in source.batches(epoch):
            wait += time.perf_counter() - tw
            if cuda:
                ev = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                ev[0].record()
            else:
                ts = time.perf_counter()
            batch = batch.to(device)
            with torch.autocast(device_type="cuda" if cuda else "cpu", dtype=torch.bfloat16, enabled=cuda):
                feats = [model.backbone(_input(batch.views[:, v])) for v in range(batch.views.shape[1])]
                loss, terms = objective.loss(model.head, feats, batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            with torch.no_grad():
                for k, v in {"loss": loss.detach().float(), **terms, "emb_std": emb_std(feats[0].detach())}.items():
                    sums[k] = sums[k] + v if k in sums else v.clone()
            if cuda:
                ev[1].record()
                events.append(ev)
            else:
                cpu_step_seconds += time.perf_counter() - ts
            tw = time.perf_counter()
        if cuda:
            torch.cuda.synchronize()
            gpu_epoch = sum(a.elapsed_time(b) for a, b in events) / 1000
        else:
            gpu_epoch = cpu_step_seconds
        gpu_total += gpu_epoch
        epoch_seconds = time.perf_counter() - t0
        n = source.steps_per_epoch
        record = {"epoch": epoch + 1, "step": step, "lr": opt.param_groups[0]["lr"]}
        record |= {k: float(v) / n for k, v in sums.items()}
        record |= {
            "epoch_seconds": round(epoch_seconds, 3),
            "gpu_seconds": round(gpu_epoch, 3),
            "samples_per_s": round(n * s.batch_size / epoch_seconds, 1),
            "loader_wait_frac": round(wait / epoch_seconds, 4),
        }
        if validator is not None and ((epoch + 1) % s.knn_every == 0 or epoch + 1 == s.epochs):
            tk = time.perf_counter()
            record |= validator(model.backbone, cuda)
            record["knn_seconds"] = round(time.perf_counter() - tk, 3)
        if cuda:
            record["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
        with open(rd / METRICS, "a") as f:
            f.write(json.dumps(record) + "\n")
        log(f"[train {s.run_id}] " + " ".join(f"{k}={_fmt(v)}" for k, v in record.items()))
        if not math.isfinite(record["loss"]):
            raise TrainingDiverged(f"{s.run_id}: non-finite loss at epoch {epoch + 1}")

        done = epoch + 1 == s.epochs
        if done or (epoch + 1) % s.checkpoint_every == 0:
            wall = wall_prev + time.perf_counter() - wall_start
            _save_checkpoint(ckpt_path, model, opt, sched, source, s, epoch + 1, step, wall, gpu_total)
        if done:
            _deps.save_final_backbone(model.backbone, rd)
        if epoch_callback is not None:
            epoch_callback(epoch + 1, record)

    if not record:  # already finished: report the stored summary
        if (rd / SUMMARY).exists():
            return json.loads((rd / SUMMARY).read_text())
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        wall_prev, gpu_total, step = ck["wall_seconds"], ck["gpu_seconds"], ck["step"]
    summary = {
        "run_id": s.run_id,
        "run_dir": str(rd),
        "epochs_done": s.epochs,
        "steps_done": step,
        "wall_seconds": round(wall_prev + time.perf_counter() - wall_start, 3) if record else wall_prev,
        "gpu_seconds": round(gpu_total, 3),
        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3) if cuda else None,
        "final": record,
        "settings": _settings_dict(s),
        **stages.git_state(),
    }
    _write_json(rd / SUMMARY, summary)
    return summary


# ---------------------------------------------------------------- helpers


def _fmt(v):
    return f"{v:.4g}" if isinstance(v, float) else v


def _settings_dict(s: RunSettings) -> dict:
    d = dataclasses.asdict(s)
    d["betas"] = list(d["betas"])
    return d


def build_optimizer(model: nn.Module, s: RunSettings) -> torch.optim.Optimizer:
    """AdamW with weight decay on all params (spec §5.1), or, for the LARS check (spec §6.3),
    ``lightly`` LARS (momentum 0.9, trust 0.001) with biases and BN params excluded from weight
    decay and from the LARS adaptation, as in the VICReg reference implementation."""
    if s.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=s.lr, betas=s.betas, weight_decay=s.weight_decay)
    if s.optimizer == "lars":
        from lightly.utils.lars import LARS

        params = [p for p in model.parameters() if p.requires_grad]
        groups = [{"params": [p for p in params if p.ndim > 1]},
                  {"params": [p for p in params if p.ndim <= 1], "weight_decay": 0.0}]
        return LARS(groups, lr=s.lr, momentum=LARS_MOMENTUM, weight_decay=s.weight_decay)
    raise ValueError(f"optimizer must be one of {OPTIMIZERS}, got {s.optimizer!r}")


_FIELD_DEFAULTS = {f.name: f.default for f in dataclasses.fields(RunSettings) if f.default is not dataclasses.MISSING}


def _check_resumable(saved: dict, s: RunSettings) -> None:
    now = _settings_dict(s)
    # Settings added after a checkpoint was written resume as their default.
    diff = {k: (saved.get(k, _FIELD_DEFAULTS.get(k)), now[k]) for k in now
            if k not in _RESUME_FREE and saved.get(k, _FIELD_DEFAULTS.get(k)) != now[k]}
    if diff:
        raise ValueError(f"checkpoint was made with different settings {diff}; use a fresh run dir or --force")


def _config_record(s: RunSettings, source, objective) -> dict:
    cfg = config_mod.load_config(s.row, s.size, s.profile)
    rec = {
        "run_id": s.run_id,
        "settings": _settings_dict(s),
        "derived": {"n_used": s.n_used, "steps_per_epoch": s.steps_per_epoch, "total_steps": s.total_steps,
                    "warmup_steps": s.warmup_steps},
        "objective": objective.name,
        "source": source.state_dict(),
        "config": dataclasses.asdict(cfg),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if s.device.startswith("cuda") else None,
        "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **stages.git_state(),
    }
    return rec


def _rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _set_rng(st: dict) -> None:
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"])
    if st.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


def _save_checkpoint(path: Path, model, opt, sched, source, s, epoch, step, wall, gpu) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(),
        "sampler": {**source.state_dict(), "next_epoch": epoch},
        "epoch": epoch,
        "step": step,
        "rng": _rng_state(),
        "settings": _settings_dict(s),
        "wall_seconds": wall,
        "gpu_seconds": gpu,
    }
    tmp = path.with_name(f".{path.name}.tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _truncate_metrics(path: Path, epochs: int) -> None:
    if not path.exists():
        return
    keep = [line for line in path.read_text().splitlines() if line and json.loads(line)["epoch"] <= epochs]
    path.write_text("".join(line + "\n" for line in keep))


def _write_json(path: Path, payload: dict) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    os.replace(tmp, path)
