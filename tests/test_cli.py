"""CLI seam: `mo433 <subcommand>` as the user invokes it."""

import shutil
import subprocess

import pytest

from mo433.cli import main

SPEC_SUBCOMMANDS = [
    "splits", "pool", "teacher", "train", "train-all", "extract", "probe",
    "diagnose", "measure", "report", "tune", "all",
]


def test_help_lists_every_spec_subcommand(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for name in SPEC_SUBCOMMANDS:
        assert name in out, name
    assert "--profile" in out and "--force" in out


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv not on PATH")
def test_console_script_is_installed():
    proc = subprocess.run(["uv", "run", "mo433", "--help"], capture_output=True, text=True)
    assert proc.returncode == 0
    assert "train-all" in proc.stdout


def test_unimplemented_stage_exits_nonzero_with_message(capsys):
    assert main(["measure"]) == 3
    assert "not implemented" in capsys.readouterr().err


def test_train_rejects_values_outside_the_design(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["train", "--row", "A", "--size", "S", "--frac", "100", "--seed", "0"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit):
        main(["train", "--row", "B", "--size", "S", "--frac", "75", "--seed", "0"])


def test_train_on_the_fly_is_row_b_only():
    with pytest.raises(SystemExit, match="row B only"):
        main(["train", "--row", "C", "--size", "S", "--frac", "100", "--seed", "0", "--on-the-fly"])


@pytest.mark.parametrize(
    "argv",
    [
        ["--profile", "smoke", "--force", "tune", "lars"],
        ["tune", "lars", "--profile", "smoke", "--force"],
    ],
)
def test_global_flags_work_before_or_after_subcommand(argv, monkeypatch):
    from mo433 import cli

    seen = {}

    def spy(ctx, args):
        seen["ctx"] = ctx

    monkeypatch.setitem(cli.COMMANDS, "tune", cli.COMMANDS["tune"].__class__("tune", "", spy, cli._tune_args))
    assert main(argv) == 0
    assert seen["ctx"] == cli.Context(profile="smoke", force=True)


def _spy_extract(monkeypatch):
    from mo433 import extract

    calls = []
    monkeypatch.setattr(extract, "extract", lambda encoder_id, profile="full", **kw: calls.append((encoder_id, profile)))
    return calls


def test_extract_rows_runs_one_stage_per_encoder(mo433_root, monkeypatch):
    from mo433.stages import is_done

    calls = _spy_extract(monkeypatch)
    assert main(["extract", "--rows", "A"]) == 0
    assert [c[0] for c in calls] == ["A-S-s0", "A-S-s1", "A-S-s2", "A-L-s0", "A-L-s1"]
    assert is_done("extract/A-L-s1")
    assert main(["extract", "A-S-s0", "A-L-s1"]) == 0  # done: skipped
    assert len(calls) == 5
    assert main(["extract", "A-S-s0", "--force", "--profile", "smoke"]) == 0
    assert calls[-1] == ("A-S-s0", "smoke")


def test_extract_rejects_unknown_ids(mo433_root, monkeypatch):
    calls = _spy_extract(monkeypatch)
    with pytest.raises(SystemExit):
        main(["extract", "A-M-s0"])
    assert calls == []
