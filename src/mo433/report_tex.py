"""LaTeX outputs of ``mo433 report``: booktabs tables and ``macros.tex`` (spec §9).

Every number is generated from ``ReportData``; nothing is typed by hand. Missing cells
print ``--`` in tables; a missing macro value prints a bold ``??`` so it is visible in the
PDF but the document still compiles.

Macro names (letters only)
--------------------------
Accuracies are in percent with one decimal; gaps in percentage points, signed.

- ``\\Acc<Group><Probe><Frac>`` / ``\\AccSd...``: cell mean / sd (spec §8 aggregation).
  Group: ``AS``, ``AL``, ``BSFull`` (B-S-100), ``BSHalf`` (B-S-050), ..., ``D``.
  Probe: ``Knn``, ``Lin``, ``Rbf``. Frac (label fraction): ``One``, ``Ten``, ``Fifty``,
  ``Hundred``. Example: ``\\AccCLFullLinTen``.
- ``\\Acc<Group><Probe><Frac>Seed<N>``: one seed's draw-averaged accuracy (A/B/C).
- ``\\AccMsCat...`` / ``\\AccMsProj...``: the multi-scale probe and its random-projection
  control (same suffixes as ``\\Acc``).
- ``\\Gap<Size><Data><Probe><Frac>``: C − B; ``\\GapSd...`` the reference sd (larger of
  the two seed sds); ``\\GapBeyond...`` ``yes``/``no``. Size ``S``/``L``, Data
  ``Full``/``Half``. Example: ``\\GapSFullLinTen``.
- ``\\AblShift<Size><Probe><Frac>``: gap(50% data) − gap(100% data);
  ``\\AblSameDir<Probe><Frac>``: ``yes`` if that shift has the same sign for S and L.
- Costs, per architecture ``S``/``L``/``D``: ``\\Params`` (M), ``\\GMacs``, ``\\FeatDim``,
  ``\\GpuLat`` (ms per batch of 256), ``\\CpuLat`` (ms per image).
- Budget: ``\\GpuH<Group>`` mean GPU-h per run, ``\\GpuHTotalB``, ``\\GpuHTotalC``
  (distillation only), ``\\GpuHTeacherCache``, ``\\GpuHTeacherPretrain`` (estimate),
  ``\\GpuHTotalCAllIn``, ``\\WallHTotal``, ``\\AccPerGpuH<Group>`` (linear-SVM accuracy at
  100% labels per GPU-hour of one run; C includes cache + pretraining) and
  ``\\AccPerGpuH<Group>NoPretrain``.
- Diagnostics per group: ``\\ValKnn``, ``\\Spread``, ``\\NmiVal``, ``\\NmiTest``,
  ``\\AriTest``, ``\\NFlagged`` (``k/n``); ``\\TeacherPassed``.
- Bookkeeping: ``\\NEncodersReported``, ``\\NEncodersMissing``, ``\\SvmProvisional``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from mo433 import config, paths
from mo433.report import (
    FEATURE_SETS,
    LABEL_FRACS,
    PROBES,
    ROW_NAME,
    TEACHER_PRETRAIN_GPU_H,
    ReportData,
    expected_encoders,
    expected_groups,
    parse_group,
)

MISSING = "--"
MISSING_MACRO = r"\textbf{??}"
PROBE_CODE = {"knn": "Knn", "linear_svm": "Lin", "rbf_svm": "Rbf"}
PROBE_LABEL = {"knn": "$k$-NN", "linear_svm": "Linear SVM", "rbf_svm": "RBF SVM"}
FRAC_CODE = {1: "One", 10: "Ten", 50: "Fifty", 100: "Hundred"}
DATA_CODE = {100: "Full", 50: "Half"}
SEED_CODE = {0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four"}
FS_CODE = {"main": "", "ms_concat": "MsCat", "ms_randproj": "MsProj"}
FS_LABEL = {"main": "main", "ms_concat": "s3+s4", "ms_randproj": "s3+s4 proj."}
D_FOOTNOTE = r"$^\dagger$The teacher is one checkpoint: $\pm$ is the sd over the 3 label draws."
GAP_FOOTNOTE = (
    r"$\Delta$ = Distilled $-$ VICReg in pp (reference sd = larger of the two seed sds); "
    r"$^*$ beyond spread, $|\Delta|$ $>$ reference sd."
)


# ---------------------------------------------------------------- formatting


def _nan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def pct(x) -> str:
    return MISSING if _nan(x) else f"{100 * x:.1f}"


def signed_pp(x) -> str:
    if _nan(x):
        return MISSING
    v = round(100 * x, 1)
    return f"+{v:.1f}" if v >= 0 else rf"\ensuremath{{-}}{abs(v):.1f}"


def num(x, fmt: str) -> str:
    return MISSING if _nan(x) else format(x, fmt)


def pm(cell) -> str:
    if cell is None:
        return MISSING
    if _nan(cell.sd):
        return pct(cell.mean)
    return rf"{pct(cell.mean)} $\pm$ {pct(cell.sd)}"


def gap_cell(g) -> str:
    if g is None:
        return MISSING
    sd = "" if _nan(g.ref_sd) else f" ({pct(g.ref_sd)})"
    star = "$^*$" if g.beyond else ""
    return f"{signed_pp(g.gap)}{sd}{star}"


def group_code(group: str) -> str:
    k = parse_group(group)
    if k.row == "D":
        return "D"
    return f"{k.row}{k.size}" + ("" if k.frac is None else DATA_CODE[k.frac])


def group_label(group: str) -> str:
    k = parse_group(group)
    if k.row == "D":
        return r"Teacher$^\dagger$"
    data = "" if k.frac is None else f", {k.frac}\\%"
    return f"{ROW_NAME[k.row]} {k.size}{data}"


def _seed_mean(values: Iterable[float]) -> tuple[float, float]:
    vals = [v for v in values if not _nan(v)]
    if not vals:
        return float("nan"), float("nan")
    return float(np.mean(vals)), (float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan"))


# ---------------------------------------------------------------- macros


DATASET_TITLES = {"eurosat": "EuroSAT"}


def _namespace() -> tuple[str, str, str]:
    """(macro prefix, label prefix, caption prefix) of the current dataset; empty for the main
    study, so a second dataset's tables and macros can be input into the same main.tex."""
    ds = paths.dataset()
    if ds == paths.DEFAULT_DATASET:
        return "", "", ""
    return ds.capitalize(), f"{ds}:", f"{DATASET_TITLES.get(ds, ds)}: "


class Macros:
    def __init__(self):
        self.values: dict[str, str] = {}

    def add(self, name: str, value: str) -> None:
        if not name.isalpha() or not name.isascii():
            raise ValueError(f"macro name must be letters only: {name!r}")
        if name in self.values:
            raise ValueError(f"duplicate macro {name!r}")
        self.values[name] = MISSING_MACRO if value in (MISSING, "") else value

    def render(self) -> str:
        lines = ["% Generated by `mo433 report` from results.csv and report/data/. Do not edit."]
        prefix = _namespace()[0]
        lines += [f"\\newcommand{{\\{prefix}{k}}}{{{v}}}" for k, v in self.values.items()]
        return "\n".join(lines) + "\n"


def build_macros(data: ReportData) -> Macros:
    m = Macros()
    for fs in FEATURE_SETS:
        cells = data.cells[fs]
        for group in expected_groups():
            if fs != "main" and group == "D":
                continue
            for probe in PROBES:
                for lf in LABEL_FRACS:
                    stem = f"{FS_CODE[fs]}{group_code(group)}{PROBE_CODE[probe]}{FRAC_CODE[lf]}"
                    c = cells.get((group, probe, lf))
                    m.add(f"Acc{stem}", pct(c.mean) if c else MISSING)
                    m.add(f"AccSd{stem}", pct(c.sd) if c else MISSING)
                    if fs == "main" and group != "D":
                        size = parse_group(group).size
                        for seed in config.SEEDS[size]:
                            v = c.per_seed.get(seed) if c else None
                            m.add(f"Acc{stem}Seed{SEED_CODE[seed]}", pct(v))

    for size in config.SIZES:
        for frac in config.FRACS:
            for probe in PROBES:
                for lf in LABEL_FRACS:
                    stem = f"{size}{DATA_CODE[frac]}{PROBE_CODE[probe]}{FRAC_CODE[lf]}"
                    g = data.gaps.get((size, frac, probe, lf))
                    m.add(f"Gap{stem}", signed_pp(g.gap) if g else MISSING)
                    m.add(f"GapSd{stem}", pct(g.ref_sd) if g else MISSING)
                    m.add(f"GapBeyond{stem}", ("yes" if g.beyond else "no") if g else MISSING)
    for probe in PROBES:
        for lf in LABEL_FRACS:
            shifts = {}
            for size in config.SIZES:
                full, half = data.gaps.get((size, 100, probe, lf)), data.gaps.get((size, 50, probe, lf))
                shifts[size] = half.gap - full.gap if full and half else float("nan")
                m.add(f"AblShift{size}{PROBE_CODE[probe]}{FRAC_CODE[lf]}", signed_pp(shifts[size]))
            same = MISSING if any(_nan(v) for v in shifts.values()) else (
                "yes" if np.sign(shifts["S"]) == np.sign(shifts["L"]) else "no")
            m.add(f"AblSameDir{PROBE_CODE[probe]}{FRAC_CODE[lf]}", same)

    for arch in ("S", "L", "D"):
        c = data.costs.get(arch, {})
        m.add(f"Params{arch}", num(c.get("params", float("nan")) / 1e6, ".2f"))
        m.add(f"GMacs{arch}", num(c.get("macs", float("nan")) / 1e9, ".2f"))
        m.add(f"FeatDim{arch}", num(c.get("feature_dim"), ".0f"))
        m.add(f"GpuLat{arch}", num(c.get("gpu_latency_ms"), ".1f"))
        m.add(f"CpuLat{arch}", num(c.get("cpu_latency_ms"), ".1f"))

    budget = budget_summary(data)
    for group, b in budget["groups"].items():
        code = group_code(group)
        m.add(f"GpuH{code}", num(b["gpu_h"], ".2f"))
        m.add(f"AccPerGpuH{code}", num(b["acc_per_gpu_h"], ".3g"))
        if parse_group(group).row == "C":
            m.add(f"AccPerGpuH{code}NoPretrain", num(b["acc_per_gpu_h_no_pretrain"], ".3g"))
    m.add("GpuHTotalB", num(budget["total_b"], ".1f"))
    m.add("GpuHTotalC", num(budget["total_c"], ".1f"))
    m.add("GpuHTeacherCache", num(budget["teacher_cache"], ".2f"))
    m.add("GpuHTeacherPretrain", f"{TEACHER_PRETRAIN_GPU_H:,.0f}".replace(",", "{,}"))
    m.add("GpuHTotalCAllIn", num(budget["total_c_all_in"], ",.0f").replace(",", "{,}"))
    m.add("WallHTotal", num(budget["wall_total"], ".1f"))

    for group, d in diagnostics_summary(data).items():
        code = group_code(group)
        for key, name in (("val_knn", "ValKnn"), ("val_spread", "Spread"), ("val_nmi", "NmiVal"),
                          ("test_nmi", "NmiTest"), ("test_ari", "AriTest")):
            fmt = (lambda v: num(v, ".2f")) if key == "val_spread" else pct if key == "val_knn" else (
                lambda v: num(v, ".3f"))
            m.add(f"{name}{code}", fmt(d[key][0]))
        m.add(f"NFlagged{code}", d["flagged"])
    teacher = data.diagnostics.get("D", {}).get("teacher_passed")
    m.add("TeacherPassed", MISSING if teacher is None else ("yes" if teacher else "no"))

    present = {r["encoder_id"] for r in data.rows}
    m.add("NEncodersReported", str(sum(e in present for e in expected_encoders())))
    m.add("NEncodersMissing", str(len(data.missing_encoders)))
    m.add("SvmProvisional", "yes" if data.provisional_svm else "no")
    return m


# ---------------------------------------------------------------- derived summaries


def _runs_of(group: str) -> list[str]:
    k = parse_group(group)
    return [config.run_id(k.row, k.size, k.frac, s) for s in config.SEEDS[k.size]]


def budget_summary(data: ReportData) -> dict:
    groups, total_b, total_c, wall, cache = {}, 0.0, 0.0, 0.0, 0.0
    any_runs = False
    for group in expected_groups():
        k = parse_group(group)
        if k.row not in config.ROWS:
            continue
        runs = [data.runs_cost[r] for r in _runs_of(group) if r in data.runs_cost]
        gpu = float(np.mean([r["gpu_h"] for r in runs])) if runs else float("nan")
        wall_h = float(np.mean([r["wall_h"] for r in runs])) if runs else float("nan")
        group_cache = max((r["teacher_cache_gpu_h_charged"] for r in runs), default=0.0)
        if runs:
            any_runs = True
            s = sum(r["gpu_h"] for r in runs)
            total_b += s if k.row == "B" else 0.0
            total_c += s if k.row == "C" else 0.0
            wall += sum(r["wall_h"] for r in runs)
            cache = max(cache, group_cache)
        acc = data.main.get((group, "linear_svm", 100))
        acc = acc.mean if acc else float("nan")
        per_run = gpu + (group_cache if k.row == "C" else 0.0)
        groups[group] = {
            "n": len(runs),
            "gpu_h": gpu,
            "wall_h": wall_h,
            "acc": acc,
            "acc_per_gpu_h": 100 * acc / (per_run + (TEACHER_PRETRAIN_GPU_H if k.row == "C" else 0.0)),
            "acc_per_gpu_h_no_pretrain": 100 * acc / per_run,
        }
    nan = float("nan")
    return {
        "groups": groups,
        "total_b": total_b if any_runs else nan,
        "total_c": total_c if any_runs else nan,
        "teacher_cache": cache if any_runs else nan,
        "total_c_all_in": total_c + cache + TEACHER_PRETRAIN_GPU_H if any_runs else nan,
        "wall_total": wall if any_runs else nan,
    }


def _encoders_of(group: str) -> list[str]:
    k = parse_group(group)
    if k.row == "D":
        return ["D"]
    if k.row == "A":
        return [f"A-{k.size}-s{s}" for s in config.SEEDS[k.size]]
    return _runs_of(group)


def diagnostics_summary(data: ReportData) -> dict[str, dict]:
    out = {}
    for group in expected_groups():
        recs = [data.diagnostics[e] for e in _encoders_of(group) if e in data.diagnostics]
        d = {key: _seed_mean(r[key] for r in recs) for key in ("val_knn", "val_spread", "val_nmi", "test_nmi", "test_ari")}
        flags = [r["collapse_flagged"] for r in recs if r["collapse_flagged"] is not None]
        d["flagged"] = f"{sum(flags)}/{len(flags)}" if flags else MISSING
        d["n"] = len(recs)
        out[group] = d
    return out


# ---------------------------------------------------------------- tables


def _table(caption: str, label: str, colspec: str, header: list[str], body: list[str], notes: list[str],
           size: str = r"\small") -> str:
    _, label_prefix, caption_prefix = _namespace()
    caption, label = caption_prefix + caption, label_prefix + label
    lines = [
        "% Generated by `mo433 report`. Do not edit.",
        r"\begin{table}[t]",
        r"\centering",
        size,
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
    ]
    if notes:
        lines.append(r"\par\smallskip{\footnotesize " + r" \\ ".join(notes) + "}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def _provisional_note(data: ReportData) -> list[str]:
    return [r"\textbf{Provisional:} some SVM rows use non-frozen settings."] if data.provisional_svm else []


FRAC_HEADER = [f"{lf}\\%" for lf in LABEL_FRACS]


def main_probes_table(data: ReportData) -> str:
    body = []
    for i, probe in enumerate(PROBES):
        if i:
            body.append(r"\midrule")
        body.append(rf"\multicolumn{{5}}{{l}}{{\textit{{{PROBE_LABEL[probe]}}}}} \\")
        for size in config.SIZES:
            for group in (f"A-{size}", f"B-{size}-100", f"C-{size}-100"):
                body.append(" & ".join([group_label(group)] + [pm(data.main.get((group, probe, lf))) for lf in LABEL_FRACS]) + r" \\")
            body.append(" & ".join([rf"$\Delta$ ({size})"] + [gap_cell(data.gaps.get((size, 100, probe, lf))) for lf in LABEL_FRACS]) + r" \\")
        body.append(" & ".join([group_label("D")] + [pm(data.main.get(("D", probe, lf))) for lf in LABEL_FRACS]) + r" \\")
    return _table(
        "Test accuracy (\\%) by probe and label fraction, 100\\% unlabeled data. Mean $\\pm$ sd over seeds "
        "(draws averaged within each seed; S: 3 seeds, L: 2 seeds).",
        "tab:main-probes", "lcccc", ["Encoder"] + FRAC_HEADER, body,
        [GAP_FOOTNOTE, D_FOOTNOTE] + _provisional_note(data),
    )


def l_runs_table(data: ReportData) -> str:
    body = []
    for i, probe in enumerate(PROBES):
        if i:
            body.append(r"\midrule")
        body.append(rf"\multicolumn{{5}}{{l}}{{\textit{{{PROBE_LABEL[probe]}}}}} \\")
        for group in [g for g in expected_groups() if parse_group(g).size == "L"]:
            for seed in config.SEEDS["L"]:
                c = data.main.get((group, probe, LABEL_FRACS[0]))
                name = f"{group_label(group)}, seed {seed}"
                vals = []
                for lf in LABEL_FRACS:
                    c = data.main.get((group, probe, lf))
                    vals.append(pct(c.per_seed.get(seed)) if c else MISSING)
                body.append(" & ".join([name] + vals) + r" \\")
    return _table(
        "Individual large-student (L) encoders: test accuracy (\\%), mean over the 3 label draws.",
        "tab:l-runs", "lcccc", ["Encoder"] + FRAC_HEADER, body, _provisional_note(data), size=r"\footnotesize",
    )


def ablation_table(data: ReportData) -> str:
    cols = [(s, f) for s in config.SIZES for f in config.FRACS]
    body = []
    for i, probe in enumerate(PROBES):
        if i:
            body.append(r"\midrule")
        for lf in LABEL_FRACS:
            first = PROBE_LABEL[probe] if lf == LABEL_FRACS[0] else ""
            cells = [gap_cell(data.gaps.get((s, f, probe, lf))) for s, f in cols]
            body.append(" & ".join([first, f"{lf}\\%"] + cells) + r" \\")
    header = ["Probe", "Labels"] + [f"{s}, {f}\\% data" for s, f in cols]
    return _table(
        "Ablation grid: Distilled $-$ VICReg gap (pp) per unlabeled-data setting $\\times$ student size. "
        "The 50\\% setting halves both the data and the optimizer steps.",
        "tab:ablation", "llcccc", header, body, [GAP_FOOTNOTE] + _provisional_note(data), size=r"\footnotesize",
    )


def multiscale_table(data: ReportData) -> str:
    lfs = (10, 100)
    body = []
    for group in [g for g in expected_groups() if g != "D"]:
        vals = [pm(data.cells[fs].get((group, "linear_svm", lf))) for lf in lfs for fs in FEATURE_SETS]
        body.append(" & ".join([group_label(group)] + vals) + r" \\")
    header = ["Encoder"] + [f"{FS_LABEL[fs]}, {lf}\\%" for lf in lfs for fs in FEATURE_SETS]
    return _table(
        "Eval-only multi-scale probe (linear SVM, test accuracy \\%): main feature, concatenated "
        "GAP(stage 3)+GAP(stage 4), and its fixed random projection to the main dimension.",
        "tab:multiscale", "l" + "c" * len(header[1:]), header, body, _provisional_note(data), size=r"\scriptsize",
    )


def costs_table(data: ReportData) -> str:
    names = {"S": "ResNet96-S (students)", "L": "ResNet96-L (students)", "D": "ViT-S/16 (teacher)"}
    px = {"S": 96, "L": 96, "D": 224}
    body = []
    for arch in ("S", "L", "D"):
        c = data.costs.get(arch, {})
        body.append(" & ".join([
            names[arch], str(px[arch]),
            num(c.get("params", float("nan")) / 1e6, ".2f"),
            num(c.get("macs", float("nan")) / 1e9, ".2f"),
            num(c.get("feature_dim"), ".0f"),
            num(c.get("gpu_latency_ms"), ".1f"),
            num(c.get("cpu_latency_ms"), ".1f"),
        ]) + r" \\")
    return _table(
        "Encoder cost at the native input size. GPU latency: RTX 3060 Laptop, batch 256, fp32 (ms per batch); "
        "CPU latency: batch 1, 16 threads (ms per image). MACs = FLOPs/2.",
        "tab:costs", "lrrrrrr", ["Encoder", "Input px", "Params (M)", "GMACs", "Dim", "GPU ms", "CPU ms"], body,
        ["Probe capacity grows with the feature dimension, which differs between S and L/D."],
    )


def budget_table(data: ReportData) -> str:
    b = budget_summary(data)
    body = []
    for group, g in b["groups"].items():
        body.append(" & ".join([
            group_label(group), str(g["n"]), num(g["gpu_h"], ".2f"), num(g["wall_h"], ".2f"),
            pct(g["acc"]), num(g["acc_per_gpu_h"], ".3g"),
        ]) + r" \\")
    body.append(r"\midrule")
    for label, key, fmt in (("Total VICReg (GPU-h)", "total_b", ".1f"), ("Total distillation (GPU-h)", "total_c", ".1f"),
                            ("Teacher cache, once (GPU-h)", "teacher_cache", ".2f"),
                            ("Distillation all-in incl.\\ pretraining (GPU-h)", "total_c_all_in", ",.0f")):
        body.append(rf"\multicolumn{{2}}{{l}}{{{label}}} & {num(b[key], fmt).replace(',', '{,}')} & & & \\")
    pretrain = f"{TEACHER_PRETRAIN_GPU_H:,.0f}".replace(",", "{,}")
    body.append(rf"\multicolumn{{2}}{{l}}{{Teacher pretraining (estimate)}} & {pretrain} & & & \\")
    return _table(
        "Training budget per run group (mean per run) and accuracy per GPU-hour "
        "(linear SVM, 100\\% labels; Distilled is charged the teacher cache and the teacher's pretraining estimate).",
        "tab:budget", "lrrrrr", ["Group", "Runs", "GPU-h", "Wall-h", "Acc.", "pp/GPU-h"], body,
        [r"Teacher pretraining: 1.16k GPU-h for 300 epochs (DINO Table~8), extrapolated linearly to 800 epochs."],
        size=r"\footnotesize",
    )


def diagnostics_table(data: ReportData) -> str:
    summary = diagnostics_summary(data)
    body = []
    for group, d in summary.items():
        flag = d["flagged"]
        if group == "D":
            t = data.diagnostics.get("D", {}).get("teacher_passed")
            flag = MISSING if t is None else ("pass" if t else "FAIL")
        elif parse_group(group).row == "A":
            flag = "anchor" if d["n"] else MISSING

        def f(key, scale=1.0, fmt=".3f"):
            mean, sd = d[key]
            if _nan(mean):
                return MISSING
            s = format(mean * scale, fmt)
            return s if _nan(sd) else rf"{s} $\pm$ {format(sd * scale, fmt)}"

        body.append(" & ".join([
            "Teacher" if group == "D" else group_label(group), f("val_knn", 100, ".1f"), f("val_spread", 1, ".2f"), f("val_nmi"),
            f("test_nmi"), f("test_ari"), flag,
        ]) + r" \\")
    return _table(
        "Unsupervised diagnostics, mean $\\pm$ sd over seeds. Validation split: $k$-NN accuracy (\\%), "
        "spread = per-dimension std $\\times\\sqrt{d}$, $k$-means NMI; test split: $k$-means NMI and ARI. "
        "Flagged: collapse check fired (count/encoders); Teacher: sanity check.",
        "tab:diagnostics", "lcccccc",
        ["Encoder", "Val $k$-NN", "Spread", "Val NMI", "Test NMI", "Test ARI", "Flagged"], body, [],
        size=r"\scriptsize",
    )


TABLES = {
    "main_probes": main_probes_table,
    "l_runs": l_runs_table,
    "ablation": ablation_table,
    "multiscale": multiscale_table,
    "costs": costs_table,
    "budget": budget_table,
    "diagnostics": diagnostics_table,
}


def write_all(data: ReportData, out_dir: Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, fn in TABLES.items():
        p = out_dir / f"{name}.tex"
        p.write_text(fn(data))
        written.append(p)
    p = out_dir / "macros.tex"
    p.write_text(build_macros(data).render())
    return written + [p]
