"""`mo433 train-all` (spec §8 week-3 order, §9 resume) with training and extraction spied."""

import pytest

from mo433.cli import main

# Spec §8: main S, main L, ablation S, ablation L; B and C of a seed back to back.
SPEC_ORDER = [
    "B-S-100-s0", "C-S-100-s0", "B-S-100-s1", "C-S-100-s1", "B-S-100-s2", "C-S-100-s2",
    "B-L-100-s0", "C-L-100-s0", "B-L-100-s1", "C-L-100-s1",
    "B-S-050-s0", "C-S-050-s0", "B-S-050-s1", "C-S-050-s1", "B-S-050-s2", "C-S-050-s2",
    "B-L-050-s0", "C-L-050-s0", "B-L-050-s1", "C-L-050-s1",
]


@pytest.fixture
def spies(mo433_root, monkeypatch):
    from mo433 import extract, train

    calls = []

    def fake_train(row, size, frac, seed, profile="full", run_dir=None, fresh=False, **kw):
        calls.append(("train", f"{row}-{size}-{frac:03d}-s{seed}", profile, fresh))
        return {}

    def fake_extract(encoder_id, profile="full", **kw):
        calls.append(("extract", encoder_id, profile))

    monkeypatch.setattr(train, "train_run", fake_train)
    monkeypatch.setattr(extract, "extract", fake_extract)
    return calls


def test_dry_list_prints_the_spec_order_and_trains_nothing(spies, capsys):
    assert main(["train-all", "--dry-list"]) == 0
    listed = [line.split()[1] for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert listed == SPEC_ORDER
    assert spies == []


def test_trains_every_run_in_order_and_extracts_each_right_after_it(spies):
    assert main(["train-all"]) == 0
    expected = []
    for rid in SPEC_ORDER:
        expected += [("train", rid, "full", False), ("extract", rid, "full")]
    assert spies == expected


def test_finished_runs_are_skipped_on_rerun(spies):
    from mo433.stages import is_done

    assert main(["train-all", "--only", "L"]) == 0
    assert is_done("train/C-L-050-s1") and is_done("extract/C-L-050-s1")
    spies.clear()
    assert main(["train-all"]) == 0
    trained = [c[1] for c in spies if c[0] == "train"]
    assert trained == [r for r in SPEC_ORDER if "-S-" in r]


def test_only_restricts_to_one_size(spies):
    assert main(["train-all", "--only", "S"]) == 0
    trained = [c[1] for c in spies if c[0] == "train"]
    assert trained == [r for r in SPEC_ORDER if "-S-" in r]


def test_force_retrains_finished_runs_from_scratch(spies):
    assert main(["train-all", "--only", "S"]) == 0
    spies.clear()
    assert main(["train-all", "--only", "S", "--force"]) == 0
    trained = [c for c in spies if c[0] == "train"]
    assert len(trained) == 12 and all(c[3] is True for c in trained)
