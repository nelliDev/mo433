"""Figures of ``mo433 report`` (spec §9), matplotlib only.

- ``label_efficiency.pdf``: linear-SVM test accuracy vs label fraction (log x), one line
  per encoder group at 100% unlabeled data; error bars = seed sd (D: draw sd).
- ``tsne.pdf``: 4 panels A/B/C/D (``report.TSNE_PANELS``), coloured by test class.
- ``ablation_grid.pdf``: C − B gap (linear SVM) per unlabeled-data setting × student size,
  one panel per label fraction, annotated with gap, reference sd and the beyond-spread star.
- ``gap_vs_params.pdf``: C − B gap (linear SVM) vs student parameters, one line per label
  fraction; solid = 100% data, dashed/hollow = 50% ablation; error bars = reference sd.

Colours: rows have fixed colours in every figure (A neutral grey, B Okabe-Ito blue, C
Okabe-Ito vermillion, D black), with a fixed marker per row and a line style per size, so
identity never relies on colour alone. Label fractions use four Okabe-Ito hues.
A figure whose inputs are all missing is skipped with a warning and not rewritten.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np

from mo433 import config
from mo433.report import LABEL_FRACS, TSNE_PANELS, ReportData, parse_encoder

ROW_COLOR = {"A": "#8C8C8C", "B": "#0072B2", "C": "#D55E00", "D": "#000000"}
ROW_MARKER = {"A": "x", "B": "o", "C": "s", "D": "D"}
SIZE_STYLE = {"S": "--", "L": "-", None: ":"}
FRAC_COLOR = {1: "#56B4E9", 10: "#009E73", 50: "#E69F00", 100: "#CC79A7"}
CLASS_COLORS = ["#CC6677", "#332288", "#DDCC77", "#117733", "#88CCEE",
                "#882255", "#44AA99", "#999933", "#AA4499", "#555555"]  # Tol muted + dark grey
PANEL_TITLE = {"A": "A: random init", "B": "B: VICReg", "C": "C: distilled", "D": "D: DINO ViT-S/16"}


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": "#e6e6e6", "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
        "pdf.fonttype": 42, "savefig.bbox": "tight", "lines.linewidth": 1.5,
    })
    return plt


def _nan_to_zero(x: float) -> float:
    return 0.0 if math.isnan(x) else x


def _save(fig, path: Path) -> Path:
    fig.savefig(path, metadata={"CreationDate": None})
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path


def label_efficiency(data: ReportData, path: Path) -> Path | None:
    groups = ["A-S", "A-L", "B-S-100", "B-L-100", "C-S-100", "C-L-100", "D"]
    present = [g for g in groups if any((g, "linear_svm", lf) in data.main for lf in LABEL_FRACS)]
    if not present:
        warnings.warn("label-efficiency figure: no linear-SVM results; skipped", stacklevel=2)
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for i, g in enumerate(present):
        key = parse_encoder(g + "-s0") if g != "D" else parse_encoder("D")
        cells = [data.main.get((g, "linear_svm", lf)) for lf in LABEL_FRACS]
        xs = [lf * 1.04 ** (i - len(present) / 2) for lf, c in zip(LABEL_FRACS, cells) if c]  # jitter
        ys = [100 * c.mean for c in cells if c]
        es = [100 * _nan_to_zero(c.sd) for c in cells if c]
        label = "D (draw sd)" if g == "D" else g.replace("-100", "")
        ax.errorbar(xs, ys, yerr=es, color=ROW_COLOR[key.row], marker=ROW_MARKER[key.row], ms=5,
                    linestyle=SIZE_STYLE[key.size], capsize=2, label=label)
    ax.set_xscale("log")
    ax.set_xticks(LABEL_FRACS, [f"{lf}%" for lf in LABEL_FRACS])
    ax.minorticks_off()
    ax.set_xlabel("labeled training images (fraction of 5k)")
    ax.set_ylabel("linear-SVM test accuracy (%)")
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5))
    return _save(fig, path)


def tsne_panels(data: ReportData, path: Path) -> Path | None:
    if not any(e in data.tsne for e in TSNE_PANELS):
        warnings.warn("t-SNE figure: no coordinates in report/data/tsne.npz; skipped", stacklevel=2)
        return None
    from mo433.data import class_names

    classes = class_names()
    plt = _plt()
    fig, axes = plt.subplots(1, 4, figsize=(8.0, 2.4))
    for ax, eid in zip(axes, TSNE_PANELS):
        row = parse_encoder(eid).row
        ax.set_title(PANEL_TITLE[row] + ("" if row == "D" else f" ({eid})"), fontsize=7)
        ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        if eid not in data.tsne:
            warnings.warn(f"t-SNE figure: {eid} missing; empty panel", stacklevel=2)
            ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes, color="#777777")
            continue
        xy, y = data.tsne[eid]
        for k in range(10):
            sel = y == k
            ax.scatter(xy[sel, 0], xy[sel, 1], s=1.0, color=CLASS_COLORS[k], linewidths=0, rasterized=True,
                       label=classes[k])
    handles = [plt.Line2D([], [], marker="o", ls="", color=CLASS_COLORS[k], ms=4) for k in range(10)]
    fig.legend(handles, classes, loc="lower center", ncol=10, fontsize=6.5, bbox_to_anchor=(0.5, -0.06))
    return _save(fig, path)


def ablation_grid(data: ReportData, path: Path) -> Path | None:
    grid = {lf: np.full((len(config.FRACS), len(config.SIZES)), np.nan) for lf in LABEL_FRACS}
    gaps = {}
    for lf in LABEL_FRACS:
        for i, frac in enumerate(config.FRACS):
            for j, size in enumerate(config.SIZES):
                g = data.gaps.get((size, frac, "linear_svm", lf))
                if g:
                    grid[lf][i, j] = 100 * g.gap
                    gaps[(lf, i, j)] = g
    if not gaps:
        warnings.warn("ablation figure: no C - B gaps; skipped", stacklevel=2)
        return None
    plt = _plt()
    vmax = max(1.0, float(np.nanmax([np.nanmax(np.abs(v)) for v in grid.values() if np.isfinite(v).any()])))
    fig, axes = plt.subplots(1, len(LABEL_FRACS), figsize=(8.0, 2.3), constrained_layout=True)
    cmap = plt.get_cmap("PuOr_r").with_extremes(bad="#eeeeee")
    for ax, lf in zip(axes, LABEL_FRACS):
        im = ax.imshow(np.ma.masked_invalid(grid[lf]), cmap=cmap, vmin=-vmax, vmax=vmax)
        ax.grid(False)
        ax.set_xticks(range(len(config.SIZES)), [f"{s}" for s in config.SIZES])
        ax.set_yticks(range(len(config.FRACS)), [f"{f}% data" for f in config.FRACS])
        ax.set_title(f"{lf}% labels", fontsize=8)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for i in range(len(config.FRACS)):
            for j in range(len(config.SIZES)):
                g = gaps.get((lf, i, j))
                if g is None:
                    ax.text(j, i, "n/a", ha="center", va="center", fontsize=7, color="#555555")
                    continue
                sd = "" if math.isnan(g.ref_sd) else f"\n±{100 * g.ref_sd:.1f}"
                text = f"{100 * g.gap:+.1f}{'*' if g.beyond else ''}{sd}"
                dark = abs(100 * g.gap) > 0.6 * vmax
                ax.text(j, i, text, ha="center", va="center", fontsize=7, color="white" if dark else "#111111")
    fig.colorbar(im, ax=axes, shrink=0.8, label="C − B (pp), linear SVM")
    return _save(fig, path)


def gap_vs_params(data: ReportData, path: Path) -> Path | None:
    if not any(k[2] == "linear_svm" for k in data.gaps):
        warnings.warn("gap-vs-params figure: no C - B gaps; skipped", stacklevel=2)
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.axhline(0, color="#555555", lw=0.8)
    for i, lf in enumerate(LABEL_FRACS):
        for frac in config.FRACS:
            pts = [(data.params(s), data.gaps[(s, frac, "linear_svm", lf)]) for s in config.SIZES
                   if (s, frac, "linear_svm", lf) in data.gaps]
            if not pts:
                continue
            jitter = 1.03 ** (i - 1.5 + (0.25 if frac == 50 else 0))
            ax.errorbar(
                [p * jitter for p, _ in pts], [100 * g.gap for _, g in pts],
                yerr=[100 * _nan_to_zero(g.ref_sd) for _, g in pts], color=FRAC_COLOR[lf], marker="o", ms=5,
                linestyle="-" if frac == 100 else "--", mfc=FRAC_COLOR[lf] if frac == 100 else "white", capsize=2,
                label=f"{lf}% labels" + ("" if frac == 100 else ", 50% data"),
            )
    ax.set_xscale("log")
    ticks = [data.params(s) for s in config.SIZES]
    ax.set_xticks(ticks, [f"{s}\n{t / 1e6:.2f}M" for s, t in zip(config.SIZES, ticks)])
    ax.minorticks_off()
    ax.set_xlim(ticks[0] / 1.6, ticks[-1] * 1.6)
    ax.set_xlabel("student parameters")
    ax.set_ylabel("C − B, linear-SVM accuracy (pp)")
    ax.legend(fontsize=6.5, ncol=2, loc="best")
    return _save(fig, path)


FIGURES = {
    "label_efficiency": label_efficiency,
    "tsne": tsne_panels,
    "ablation_grid": ablation_grid,
    "gap_vs_params": gap_vs_params,
}


def write_all(data: ReportData, out_dir: Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, fn in FIGURES.items():
        p = fn(data, out_dir / f"{name}.pdf")
        if p is not None:
            written.append(p)
    return written
