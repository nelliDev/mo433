"""``build_report`` and ``mo433 report`` end to end on synthetic inputs."""

import json
import os
import re
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from mo433 import paths, report
from report_fixture import RESULTS_HEADER, results_rows, write_csv, write_inputs

TABLES = ("main_probes", "l_runs", "ablation", "multiscale", "costs", "budget", "diagnostics")
FIGURES = ("label_efficiency", "tsne", "ablation_grid", "gap_vs_params")
MACRO = re.compile(r"\\newcommand\{\\([^}]*)\}\{(.*)\}$")


def macros(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text().splitlines():
        m = MACRO.match(line)
        if m:
            assert m.group(1) not in out, f"duplicate macro {m.group(1)}"
            out[m.group(1)] = m.group(2)
    return out


@pytest.fixture
def full(tmp_path):
    inputs = write_inputs(tmp_path / "repo")
    out = tmp_path / "repo" / "report"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        written = report.build_report(inputs, out, log=lambda s: None)
    return inputs, out, written


def test_full_fixture_produces_every_table_and_figure(full):
    _, out, _ = full
    for name in TABLES + ("macros",):
        text = (out / "generated" / f"{name}.tex").read_text()
        assert text.strip(), name
    for name in TABLES:
        assert "\\toprule" in (out / "generated" / f"{name}.tex").read_text(), name
    for name in FIGURES:
        assert (out / "figures" / f"{name}.pdf").read_bytes()[:4] == b"%PDF", name


def test_full_fixture_has_no_placeholders_or_missing_cells(full):
    _, out, _ = full
    for name in ("main_probes", "ablation", "costs", "budget", "diagnostics"):
        text = (out / "generated" / f"{name}.tex").read_text()
        assert "--" not in text.replace("---", ""), name


def test_macro_names_are_unique_letters_only(full):
    _, out, _ = full
    names = macros(out / "generated" / "macros.tex")
    assert len(names) > 100
    for name in names:
        assert re.fullmatch(r"[A-Za-z]+", name), name
    # one macro per quoted cell family, e.g. the main comparison at 10% labels
    for name in ("AccBSFullLinTen", "AccSdBSFullLinTen", "GapSFullLinTen", "GapSdSFullLinTen",
                 "GapBeyondSFullLinTen", "AccDLinHundred", "AccBLFullLinTenSeedOne", "ParamsS",
                 "GpuHTotalC", "GpuHTeacherPretrain", "NmiTestCSFull"):
        assert name in names, name


def test_macro_values_match_hand_computed_tiny_fixture(tmp_path):
    from test_report_aggregate import TINY

    inputs = report.Inputs.committed(tmp_path)
    write_csv(inputs.results, RESULTS_HEADER, TINY)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report.build_report(inputs, tmp_path / "report", log=lambda s: None)
    m = macros(tmp_path / "report" / "generated" / "macros.tex")
    assert m["AccBSFullLinTen"] == "66.0"
    assert m["AccSdBSFullLinTen"] == "4.0"
    assert m["AccDLinTen"] == "84.0"
    assert m["AccSdDLinTen"] == "4.0"
    assert m["GapSFullLinTen"] == "+6.0"
    assert m["GapSdSFullLinTen"] == "4.0"
    assert m["GapBeyondSFullLinTen"] == "yes"
    assert m["GapLFullLinTen"] == "+1.0"
    assert m["GapBeyondLFullLinTen"] == "no"
    assert m["AccBLFullLinTenSeedZero"] == "50.0"


def test_missing_encoders_and_inputs_degrade_with_warnings(tmp_path):
    only_s = [e for e in report.expected_encoders() if "-S-" in e and "-050-" not in e]
    inputs = write_inputs(tmp_path, only_s, costs=False, runs_cost=False, diagnostics=False, tsne=False)
    with pytest.warns(UserWarning):
        written = report.build_report(inputs, tmp_path / "report", log=lambda s: None)
    gen = tmp_path / "report" / "generated"
    for name in TABLES + ("macros",):
        assert (gen / f"{name}.tex").exists(), name
    figs = {p.stem for p in written["figures"]}
    assert {"label_efficiency", "gap_vs_params"} <= figs
    assert "tsne" not in figs  # no coordinates at all: not drawn
    assert not (tmp_path / "report" / "figures" / "tsne.pdf").exists()
    m = macros(gen / "macros.tex")
    assert m["GapSFullLinTen"].startswith(("+", "\\ensuremath"))
    assert "GapLFullLinTen" in m  # still defined (as a visible placeholder) so prose compiles


def test_existing_figure_survives_when_its_inputs_are_missing(full, tmp_path):
    inputs, out, _ = full
    before = (out / "figures" / "tsne.pdf").read_bytes()
    inputs.tsne.unlink()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        written = report.build_report(inputs, out, log=lambda s: None)
    assert "tsne" not in {p.stem for p in written["figures"]}
    assert (out / "figures" / "tsne.pdf").read_bytes() == before


def test_sync_copies_newer_artifacts_and_flattens_diagnostics(tmp_path, mo433_root, monkeypatch):
    res = paths.results_dir()
    write_csv(res / "results.csv", RESULTS_HEADER, results_rows(["D"]))
    (res / "diagnostics").mkdir(parents=True)
    record = {
        "encoder_id": "B-S-100-s0",
        "val_stats": {"knn": 0.5, "spread": 0.9, "nmi": 0.3},
        "collapse_check": {"flagged": True, "reasons": {"spread": False}},
        "test_characterization": {"test_nmi": 0.31, "test_ari": 0.2},
    }
    (res / "diagnostics" / "B-S-100-s0.json").write_text(json.dumps(record))
    np.save(res / "diagnostics" / "D_tsne.npy", np.zeros((8000, 2), np.float32))
    stl = tmp_path / "data" / "stl10_binary"
    stl.mkdir(parents=True)
    (np.arange(8000) % 10 + 1).astype(np.uint8).tofile(stl / "test_y.bin")
    monkeypatch.setenv("MO433_DATA", str(tmp_path / "data"))
    inputs = report.sync_inputs(inputs=report.Inputs.committed(tmp_path / "repo"), log=lambda s: None)
    assert inputs.results.read_bytes() == (res / "results.csv").read_bytes()
    diag = report.read_diagnostics(inputs.diagnostics)
    assert diag["B-S-100-s0"]["collapse_flagged"] is True
    assert diag["B-S-100-s0"]["test_nmi"] == pytest.approx(0.31)
    xy, y = report.read_tsne(inputs.tsne)["D"]
    assert xy.shape == (8000, 2) and y[:3].tolist() == [0, 1, 2]

    # an older artifact copy never overwrites a newer committed one
    inputs.results.write_text("edited\n")
    os.utime(res / "results.csv", (1, 1))
    report.sync_inputs(inputs=inputs, log=lambda s: None)
    assert inputs.results.read_text() == "edited\n"


def test_cli_report_runs_without_gpu_or_torch(tmp_path):
    """`mo433 report` with CUDA hidden: copies results.csv to the repo root, builds, never imports torch."""
    root = tmp_path / "artifacts"
    write_csv(root / "results" / "results.csv", RESULTS_HEADER, results_rows())
    repo = tmp_path / "repo"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "MO433_ROOT": str(root),
           "PYTHONPATH": str(paths.REPO_ROOT / "src")}
    code = (
        "import sys; from mo433.cli import main; rc = main(['report', '--repo-root', sys.argv[1]]);"
        "assert 'torch' not in sys.modules, 'report imported torch'; sys.exit(rc)"
    )
    proc = subprocess.run([sys.executable, "-c", code, str(repo)], env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert (repo / "results.csv").exists()
    assert (repo / "report" / "generated" / "main_probes.tex").exists()
    assert (repo / "report" / "figures" / "label_efficiency.pdf").exists()


@pytest.mark.skipif(shutil.which("latexmk") is None, reason="latexmk not installed")
def test_latex_skeleton_compiles_with_generated_files(full, tmp_path):
    _, out, _ = full
    shutil.copy(paths.REPO_ROOT / "report" / "main.tex", out / "main.tex")
    proc = subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
        cwd=out, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stdout[-3000:]
    assert (out / "main.pdf").exists()


def test_second_dataset_namespaces_macros_labels_and_captions(tmp_path, monkeypatch):
    # The EuroSAT tables go into the same main.tex as the STL-10 ones: nothing may collide.
    monkeypatch.setenv("MO433_DATASET", "eurosat")
    inputs = write_inputs(tmp_path / "repo")
    out = tmp_path / "repo" / "report" / "eurosat"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report.build_report(inputs, out, log=lambda s: None)
    names = macros(out / "generated" / "macros.tex")
    assert names and all(n.startswith("Eurosat") for n in names)
    assert "EurosatAccBSFullLinTen" in names
    main = (out / "generated" / "main_probes.tex").read_text()
    assert r"\label{eurosat:tab:" in main and r"\caption{EuroSAT: " in main
