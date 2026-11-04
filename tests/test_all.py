"""`mo433 all` (spec §9): the pipeline order, the SVM-parameter policy, and the smoke run end to end."""

import pytest

from mo433 import cli
from mo433.cli import main


@pytest.fixture
def recorded(monkeypatch, mo433_root):
    """Replace every pipeline handler with a recorder of (name, parsed args)."""
    calls = []
    for name in cli.PIPELINE:
        cmd = cli.COMMANDS[name]

        def rec(ctx, args, name=name):
            calls.append((name, ctx.profile, args))

        monkeypatch.setitem(cli.COMMANDS, name, cmd.__class__(name, cmd.help, rec, cmd.add_args))
    return calls


def test_pipeline_runs_splits_to_report_with_invariants_before_any_gpu_stage():
    assert cli.PIPELINE == (
        "splits", "invariants", "pool", "teacher", "train-all", "extract", "probe", "diagnose", "measure", "report",
    )


def test_smoke_all_runs_every_stage_and_probes_with_provisional_svms(recorded):
    assert main(["all", "--profile", "smoke"]) == 0
    assert [c[0] for c in recorded] == list(cli.PIPELINE)
    assert {c[1] for c in recorded} == {"smoke"}
    args = {name: a for name, _, a in recorded}
    for stage in ("probe", "diagnose"):
        assert args[stage].provisional_svm is True
        assert args[stage].encoder == ["A-S-s0", "B-S-100-s0", "C-S-100-s0", "D"]
    assert args["probe"].select_svm is False
    assert args["train-all"].only is None and args["train-all"].dry_list is False


def test_full_all_stops_before_probing_while_svm_params_are_null(recorded, capsys):
    from mo433 import config

    assert config.load_config().probe.linear_c is None  # the committed state until --select-svm
    with pytest.raises(SystemExit, match="--select-svm"):
        main(["all"])
    assert [c[0] for c in recorded] == ["splits", "invariants", "pool", "teacher", "train-all", "extract"]


@pytest.mark.slow
@pytest.mark.data
@pytest.mark.gpu
@pytest.mark.weights
def test_smoke_all_end_to_end(tmp_path, monkeypatch):
    """The real thing: every stage on the smoke profile, invariant tests included (minutes)."""
    from mo433 import config, paths, probe, stages, train

    root = tmp_path / "artifacts-smoke"
    monkeypatch.setenv("MO433_ROOT", str(root))
    monkeypatch.setenv("MO433_NUM_WORKERS", "2")
    assert main(["all", "--profile", "smoke"]) == 0

    for r in config.reported_runs():
        rid = config.run_id(*r)
        assert stages.is_done(f"train/{rid}", "smoke") and stages.is_done(f"extract/{rid}", "smoke")
    rows = probe.read_results(paths.results_dir("smoke") / "results.csv")
    assert {r["encoder_id"] for r in rows} == set(cli.SMOKE_EVAL_ENCODERS)
    assert {r["probe"] for r in rows} == {"knn", "linear_svm", "rbf_svm"}
    assert (root / "report-repo" / "report" / "generated" / "macros.tex").exists()

    # A rerun of train-all skips every finished run.
    def boom(*a, **k):
        raise AssertionError("finished run retrained")

    monkeypatch.setattr(train, "train_run", boom)
    assert main(["train-all", "--profile", "smoke"]) == 0
