"""``mo433 report-extras``: committed extra CSVs -> tables, macros, figures; never torch."""

import os
import re
import subprocess
import sys
import warnings

from mo433 import paths, report_extras, students


def test_hardcoded_parameter_counts_match_the_student_family():
    for size, n in report_extras.PARAMS.items():
        assert sum(p.numel() for p in students.build_student(size, 0).parameters()) == n


def test_build_on_the_committed_csvs(tmp_path):
    inp = report_extras.Inputs(paths.REPO_ROOT)
    inp.out = tmp_path
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        written = report_extras.build(inp, log=lambda s: None)
    names = {p.name for p in written}
    assert {"extras_datasets.tex", "extras_long300.tex", "extras_sizes.tex", "extras_macros.tex"} <= names
    macros = (tmp_path / "generated" / "extras_macros.tex").read_text()
    for name in re.findall(r"\\newcommand\{\\([^}]*)\}", macros):
        assert name.startswith("Ext") and name.isalpha(), name
    assert r"\newcommand{\ExtParamsXS}{0.39}" in macros
    sizes = (tmp_path / "generated" / "extras_sizes.tex").read_text()
    assert r"\label{tab:extras-sizes}" in sizes and "XS & 0.39" in sizes


def test_report_extras_never_imports_torch(tmp_path):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(paths.REPO_ROOT / "src")}
    code = ("import sys, warnings; warnings.simplefilter('ignore'); from mo433 import report_extras as r;"
            "i = r.Inputs(); i.out = __import__('pathlib').Path(sys.argv[1]); r.build(i, log=lambda s: None);"
            "assert 'torch' not in sys.modules")
    proc = subprocess.run([sys.executable, "-c", code, str(tmp_path)], env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
