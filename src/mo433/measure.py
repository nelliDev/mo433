"""Stage module: ``mo433 measure``, the cost table (spec §7.4, §11, §12.3–12.4).

Costs are per *architecture*, not per trained encoder: every A/B/C encoder of a size shares
the ResNet96 architecture, so they share params, FLOPs and latency. The report joins on
``rows`` / size. Outputs, under ``results_dir(profile)`` (``$MO433_ROOT/results/``):

``costs.csv``: one row per architecture (``ResNet96-S``, ``ResNet96-L``, ``D``)
    encoder_id            ``ResNet96-S`` | ``ResNet96-L`` | ``D``
    kind                  ``student`` | ``teacher``
    rows                  the encoders it covers: ``A/B/C-S``, ``A/B/C-L``, ``D``
    params                trainable + frozen parameter count of the probed network
    flops                 ``torch.utils.flop_counter`` FLOPs of one image at input_px
    macs                  flops // 2 (what the report quotes, spec §12.3)
    feature_dim           probed feature dimension
    input_px              96 for the students, 224 for the teacher (its 96->224 resize and
                          normalization are not counted or timed)
    gpu_batch             256
    gpu_latency_ms_mean   ms per fp32 batch, ``eval()`` + ``no_grad``, 10 warm-up batches,
    gpu_latency_ms_sd     mean / sd over 100 timed batches, each closed by ``cuda.synchronize()``
    gpu_throughput_img_s  gpu_batch / mean latency
    cpu_threads           16
    cpu_latency_ms_mean   ms per batch-1 forward on the CPU (10 warm-up, 100 timed)
    cpu_latency_ms_sd
    gpu_name              ``torch.cuda.get_device_name(0)``
    gpu_other_processes   other CUDA compute processes seen by nvidia-smi right before or
                          after the GPU timing ("pid name memory; ...")
    cpu_load1             max 1-min load average around the CPU timing
    provisional           True if the latency may be inflated: other GPU processes,
                          nvidia-smi failed, no CUDA, or cpu_load1 > CPU_LOAD_LIMIT
    provisional_reason    why
    measured_at           local timestamp of this ``measure`` run

``runs_cost.csv``: one row per directory under ``$MO433_ROOT/runs/`` with timing data
    run_id, row, size, frac, seed   parsed from ``{B,C}-{S,L}-{frac:03d}-s{seed}`` (empty if
                                    the id has another form, e.g. a tuning/control run)
    complete              True if ``summary.json`` exists; otherwise the times are summed
                          from ``metrics.jsonl`` (epoch_seconds + knn_seconds, gpu_seconds)
    wall_h, gpu_h         from ``summary.json`` ``wall_seconds`` / ``gpu_seconds``
    teacher_cache_gpu_h_charged   the full one-off teacher-cache GPU time, charged to every
                          C run (spec §3), 0 for B
    total_gpu_h           gpu_h + teacher_cache_gpu_h_charged

``costs_summary.json``: constants and totals for the budget analysis (spec §11)
    teacher_pretrain_gpu_h_estimate  3100 (≈3.1k GPU-h, an *estimate*, spec §4.1)
    teacher_cache_gpu_h   the one-off cache cost (amortized view: count it once per study)
    run_gpu_h_by_row, n_runs, n_c_runs, any_provisional, gpu_name, torch, measured_at

The teacher-cache GPU time is any JSON object with a ``gpu_seconds`` key under
``teacher_dir`` (files named ``*cache*`` first). Missing run dirs or cache just omit rows /
leave the value empty. The stage is cheap (a few GPU minutes): re-run it with
``mo433 measure --force`` on an idle GPU to replace provisional latencies.
"""

from __future__ import annotations

import csv
import json
import os
import re
import statistics
import subprocess
import time
from pathlib import Path

import torch
import torch.nn as nn

from mo433 import paths
from mo433.students import build_student

STUDENT_PX = 96
TEACHER_PX = 224
GPU_BATCH = 256
CPU_THREADS = 16
# Spec §4.1: 1.16k GPU-h for 300 epochs (DINO Table 8) linearly extrapolated to 800 epochs.
TEACHER_PRETRAIN_GPU_H_ESTIMATE = 3100
CPU_LOAD_LIMIT = 4.0  # 1-min load average above which CPU latency is called provisional

COSTS_CSV = "costs.csv"
RUNS_COST_CSV = "runs_cost.csv"
SUMMARY_JSON = "costs_summary.json"

COST_COLUMNS = (
    "encoder_id", "kind", "rows", "params", "flops", "macs", "feature_dim", "input_px",
    "gpu_batch", "gpu_latency_ms_mean", "gpu_latency_ms_sd", "gpu_throughput_img_s",
    "cpu_threads", "cpu_latency_ms_mean", "cpu_latency_ms_sd",
    "gpu_name", "gpu_other_processes", "cpu_load1", "provisional", "provisional_reason", "measured_at",
)  # fmt: skip
RUN_COLUMNS = (
    "run_id", "row", "size", "frac", "seed", "complete",
    "wall_h", "gpu_h", "teacher_cache_gpu_h_charged", "total_gpu_h",
)  # fmt: skip


def count_flops(net: nn.Module, x: torch.Tensor) -> int:
    """FLOPs of one ``net(x)`` under ``no_grad`` (``torch.utils.flop_counter``; MACs = FLOPs/2)."""
    from torch.utils.flop_counter import FlopCounterMode

    counter = FlopCounterMode(display=False)
    with torch.no_grad(), counter:
        net(x)
    return int(counter.get_total_flops())


def count_params(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())


def _targets(sizes, teacher: bool):
    """(metadata, frozen eval net) per measured architecture, built lazily."""
    for size in sizes:
        net = build_student(size, 0).eval().requires_grad_(False)
        meta = {"encoder_id": f"ResNet96-{size}", "kind": "student", "rows": f"A/B/C-{size}"}
        yield meta | {"feature_dim": net.dim, "input_px": STUDENT_PX}, net
    if teacher:
        yield {"encoder_id": "D", "kind": "teacher", "rows": "D", "feature_dim": 384, "input_px": TEACHER_PX}, (
            load_teacher()
        )


def _arch_costs(net: nn.Module, px: int) -> dict:
    flops = count_flops(net, torch.zeros(1, 3, px, px))
    return {"params": count_params(net), "flops": flops, "macs": flops // 2}


def architecture_costs(sizes=("S", "L"), teacher: bool = True) -> list[dict]:
    """Params / FLOPs / MACs (batch 1, actual input size) per architecture."""
    return [meta | _arch_costs(net, meta["input_px"]) for meta, net in _targets(sizes, teacher)]


# ---------------------------------------------------------------- latency


def _timed(fn, warmup: int, timed: int, sync) -> tuple[float, float]:
    """Mean and sd (ms) of ``timed`` calls of ``fn`` after ``warmup`` untimed ones."""
    for _ in range(warmup):
        fn()
    sync()
    times = []
    for _ in range(timed):
        t0 = time.perf_counter()
        fn()
        sync()
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.fmean(times), (statistics.stdev(times) if len(times) > 1 else 0.0)


def gpu_latency(net: nn.Module, px: int, batch: int = GPU_BATCH, warmup: int = 10, timed: int = 100):
    """(mean ms, sd ms) per fp32 batch on CUDA, ``eval()`` + ``no_grad``; None without CUDA."""
    if not torch.cuda.is_available():
        return None
    net = net.cuda()
    x = torch.randn(batch, 3, px, px, device="cuda")
    try:
        with torch.no_grad():
            return _timed(lambda: net(x), warmup, timed, torch.cuda.synchronize)
    finally:
        net.cpu()
        del x
        torch.cuda.empty_cache()


def cpu_latency(net: nn.Module, px: int, threads: int = CPU_THREADS, warmup: int = 10, timed: int = 100):
    """(mean ms, sd ms) per batch-1 forward on the CPU with ``threads`` intra-op threads."""
    prev = torch.get_num_threads()
    torch.set_num_threads(threads)
    x = torch.randn(1, 3, px, px)
    try:
        with torch.no_grad():
            return _timed(lambda: net(x), warmup, timed, lambda: None)
    finally:
        torch.set_num_threads(prev)


def other_gpu_processes(smi_csv: str, own_pid: int) -> list[str]:
    """Parse ``nvidia-smi --query-compute-apps=pid,process_name,used_memory`` CSV output."""
    out = []
    for line in smi_csv.splitlines():
        fields = [f.strip() for f in line.split(",")]
        if len(fields) < 3 or not fields[0].isdigit() or int(fields[0]) == own_pid:
            continue
        out.append(" ".join(fields[:3]))
    return out


def query_gpu_processes() -> list[str] | None:
    """Other CUDA compute processes on the GPU right now; None if nvidia-smi fails."""
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return other_gpu_processes(res.stdout, os.getpid())


_RUN_ID = re.compile(r"(?P<row>[BC])-(?P<size>[SL])-(?P<frac>\d{3})-s(?P<seed>\d+)")


def _read_json(path: Path) -> dict | None:
    try:
        obj = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def teacher_cache_gpu_seconds(profile: str = "full") -> float | None:
    """GPU seconds of the one-off teacher-feature cache, or None if no record exists.

    Looks for a JSON object with a ``gpu_seconds`` key under ``teacher_dir`` (files whose
    name contains "cache" first), so it does not depend on the exact file name #18 picks.
    """
    tdir = paths.teacher_dir(profile)
    if not tdir.is_dir():
        return None
    files = sorted(tdir.rglob("*.json"), key=lambda p: ("cache" not in p.name, len(p.parts), str(p)))
    for f in files:
        obj = _read_json(f)
        if obj and isinstance(obj.get("gpu_seconds"), int | float):
            return float(obj["gpu_seconds"])
    return None


def _run_seconds(rd: Path) -> tuple[float, float, bool] | None:
    """(wall s, GPU s, complete) from ``summary.json``, else summed from ``metrics.jsonl``."""
    summary = _read_json(rd / "summary.json")
    if summary and "gpu_seconds" in summary:
        return float(summary.get("wall_seconds") or 0.0), float(summary["gpu_seconds"]), True
    log = rd / "metrics.jsonl"
    if not log.exists():
        return None
    wall = gpu = 0.0
    for line in log.read_text().splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        wall += float(rec.get("epoch_seconds", 0.0)) + float(rec.get("knn_seconds", 0.0))
        gpu += float(rec.get("gpu_seconds", 0.0))
    return wall, gpu, False


def run_costs(profile: str = "full") -> list[dict]:
    """One row per run directory under ``runs/`` (see the module docstring for the columns)."""
    runs = paths.artifacts_root(profile) / "runs"
    if not runs.is_dir():
        return []
    cache_s = teacher_cache_gpu_seconds(profile) or 0.0
    out = []
    for rd in sorted(p for p in runs.iterdir() if p.is_dir()):
        secs = _run_seconds(rd)
        if secs is None:
            continue
        wall, gpu, complete = secs
        m = _RUN_ID.match(rd.name)
        row = m["row"] if m else None
        charged = cache_s if row == "C" else 0.0
        out.append(
            {
                "run_id": rd.name,
                "row": row,
                "size": m["size"] if m else None,
                "frac": int(m["frac"]) if m else None,
                "seed": int(m["seed"]) if m else None,
                "complete": complete,
                "wall_h": round(wall / 3600, 4),
                "gpu_h": round(gpu / 3600, 4),
                "teacher_cache_gpu_h_charged": round(charged / 3600, 4),
                "total_gpu_h": round((gpu + charged) / 3600, 4),
            }
        )
    return out


TIMM_TEACHER ="vit_small_patch16_224.dino"


def load_teacher() -> nn.Module:
    """The row-D network that takes a 224 px normalized input.

    Uses ``load_encoder("D")`` once the teacher module registers it (#18); until then builds
    the same timm model directly (spec §4.1). Either way the module is frozen and in eval.
    """
    from mo433 import encoders

    if encoders.is_registered("D"):
        return encoders.load_encoder("D")
    import timm

    return timm.create_model(TIMM_TEACHER, pretrained=True, num_classes=0).eval().requires_grad_(False)


# ---------------------------------------------------------------- the stage


def _r(x: float | None, nd: int = 3) -> float | None:
    return None if x is None else round(x, nd)


def _load1() -> float | None:
    try:
        return os.getloadavg()[0]
    except OSError:
        return None


def measure(
    profile: str = "full",
    sizes=("S", "L"),
    teacher: bool = True,
    warmup: int = 10,
    timed: int = 100,
    log=print,
) -> dict:
    """Write ``costs.csv``, ``runs_cost.csv`` and ``costs_summary.json`` under ``results_dir``."""
    out_dir = paths.results_dir(profile)
    out_dir.mkdir(parents=True, exist_ok=True)
    cuda = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda else None
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    rows = []
    for meta, net in _targets(sizes, teacher):
        px = meta["input_px"]
        row = meta | _arch_costs(net, px)
        before = query_gpu_processes() if cuda else []
        gpu = gpu_latency(net, px, warmup=warmup, timed=timed) if cuda else None
        after = query_gpu_processes() if cuda else []
        load = _load1()
        cpu = cpu_latency(net, px, warmup=warmup, timed=timed)
        load = max(x for x in (load, _load1(), 0.0) if x is not None)

        reasons = []
        if not cuda:
            reasons.append("no CUDA")
        elif before is None or after is None:
            reasons.append("nvidia-smi query failed")
        others = sorted(set(before or []) | set(after or []))
        if others:
            reasons.append("other GPU processes running")
        if load > CPU_LOAD_LIMIT:
            reasons.append(f"CPU load {load:.1f} > {CPU_LOAD_LIMIT:g}")
        row |= {
            "gpu_batch": GPU_BATCH if gpu else None,
            "gpu_latency_ms_mean": _r(gpu[0]) if gpu else None,
            "gpu_latency_ms_sd": _r(gpu[1]) if gpu else None,
            "gpu_throughput_img_s": round(GPU_BATCH / gpu[0] * 1000, 1) if gpu else None,
            "cpu_threads": CPU_THREADS,
            "cpu_latency_ms_mean": _r(cpu[0]),
            "cpu_latency_ms_sd": _r(cpu[1]),
            "gpu_name": gpu_name,
            "gpu_other_processes": "; ".join(others),
            "cpu_load1": _r(load, 2),
            "provisional": bool(reasons),
            "provisional_reason": "; ".join(reasons),
            "measured_at": stamp,
        }
        log(
            f"[measure] {row['encoder_id']}: {row['params'] / 1e6:.2f}M params, {row['macs'] / 1e9:.3f} GMACs, "
            f"GPU {row['gpu_latency_ms_mean']} ms/b{GPU_BATCH}, CPU {row['cpu_latency_ms_mean']} ms/b1"
            + (f" (provisional: {row['provisional_reason']})" if reasons else "")
        )
        rows.append(row)
        del net
    _write_csv(out_dir / COSTS_CSV, COST_COLUMNS, rows)

    runs = run_costs(profile)
    _write_csv(out_dir / RUNS_COST_CSV, RUN_COLUMNS, runs)

    cache_s = teacher_cache_gpu_seconds(profile)
    by_row: dict[str, float] = {}
    for r in runs:
        by_row[r["row"] or "other"] = by_row.get(r["row"] or "other", 0.0) + r["gpu_h"]
    summary = {
        "measured_at": stamp,
        "gpu_name": gpu_name,
        "torch": torch.__version__,
        "teacher_pretrain_gpu_h_estimate": TEACHER_PRETRAIN_GPU_H_ESTIMATE,
        "teacher_pretrain_note": "estimate: DINO ViT-S/16 1.16k GPU-h @300 ep (Table 8) x 800/300 (spec §4.1)",
        "teacher_cache_gpu_h": None if cache_s is None else round(cache_s / 3600, 4),
        "n_runs": len(runs),
        "n_c_runs": sum(r["row"] == "C" for r in runs),
        "run_gpu_h_by_row": {k: round(v, 4) for k, v in sorted(by_row.items())},
        "any_provisional": any(r["provisional"] for r in rows),
    }
    (out_dir / SUMMARY_JSON).write_text(json.dumps(summary, indent=2) + "\n")
    log(f"[measure] wrote {out_dir / COSTS_CSV}, {out_dir / RUNS_COST_CSV} ({len(runs)} runs), {SUMMARY_JSON}")
    return summary


def _write_csv(path: Path, columns, rows: list[dict]) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(columns))
        w.writeheader()
        for r in rows:
            w.writerow({c: ("" if r.get(c) is None else r[c]) for c in columns})
    tmp.replace(path)
