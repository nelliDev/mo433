import json

import pytest
import torch
import torch.nn as nn

from mo433 import measure


def test_flops_of_linear_layer_match_analytic_count():
    # y = xW^T + b on [8, 32] -> [8, 16]: 8*16*32 multiply-adds = 2*4096 FLOPs
    net = nn.Linear(32, 16)
    assert measure.count_flops(net, torch.zeros(8, 32)) == 2 * 8 * 16 * 32


def test_student_costs_match_spec():
    # spec §4.2 params; MACs from the analytic counts in docs/student-design/debate/residual.md
    rows = {r["encoder_id"]: r for r in measure.architecture_costs(sizes=("S", "L"), teacher=False)}
    s, l = rows["ResNet96-S"], rows["ResNet96-L"]
    assert round(s["params"] / 1e6, 2) == 1.57 and round(l["params"] / 1e6, 2) == 6.28
    assert s["macs"] / 1e6 == pytest.approx(177, abs=2)
    assert l["macs"] / 1e6 == pytest.approx(704, abs=3)
    assert s["flops"] == 2 * s["macs"]
    assert (s["feature_dim"], l["feature_dim"]) == (192, 384)
    assert s["input_px"] == l["input_px"] == 96
    assert s["kind"] == "student"


@pytest.mark.weights
def test_teacher_costs_match_spec():
    # spec §4.1: 21.67M params, 384-d, 4.3 GMACs at 224
    (d,) = measure.architecture_costs(sizes=(), teacher=True)
    assert (d["encoder_id"], d["kind"], d["feature_dim"], d["input_px"]) == ("D", "teacher", 384, 224)
    assert round(d["params"] / 1e6, 2) == 21.67
    assert d["macs"] / 1e9 == pytest.approx(4.3, abs=0.1)


def _write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def test_run_costs_read_run_dirs_and_charge_teacher_cache_to_c(mo433_root):
    _write_json(mo433_root / "runs/B-S-100-s0/summary.json", {"wall_seconds": 7200, "gpu_seconds": 5400})
    _write_json(mo433_root / "runs/C-L-050-s1/summary.json", {"wall_seconds": 3600, "gpu_seconds": 1800})
    _write_json(mo433_root / "teacher/cache_timing.json", {"gpu_seconds": 900, "wall_seconds": 1000})

    rows = {r["run_id"]: r for r in measure.run_costs()}

    b, c = rows["B-S-100-s0"], rows["C-L-050-s1"]
    assert (b["wall_h"], b["gpu_h"], b["teacher_cache_gpu_h_charged"], b["total_gpu_h"]) == (2.0, 1.5, 0.0, 1.5)
    assert (c["wall_h"], c["gpu_h"], c["teacher_cache_gpu_h_charged"], c["total_gpu_h"]) == (1.0, 0.5, 0.25, 0.75)
    assert (c["row"], c["size"], c["frac"], c["seed"], c["complete"]) == ("C", "L", 50, 1, True)


def test_unfinished_run_sums_its_epoch_log_and_is_marked_incomplete(mo433_root):
    log = mo433_root / "runs/B-L-100-s0/metrics.jsonl"
    log.parent.mkdir(parents=True)
    log.write_text(
        '{"epoch": 1, "epoch_seconds": 1800, "gpu_seconds": 1440}\n'
        '{"epoch": 2, "epoch_seconds": 1800, "gpu_seconds": 1440}\n'
    )

    (r,) = measure.run_costs()

    assert (r["run_id"], r["complete"], r["wall_h"], r["gpu_h"]) == ("B-L-100-s0", False, 1.0, 0.8)


def test_other_gpu_processes_ignore_our_own_pid():
    smi = "123, /usr/bin/python, 900 MiB\n456, .venv/bin/python, 178 MiB\n"
    assert measure.other_gpu_processes(smi, own_pid=456) == ["123 /usr/bin/python 900 MiB"]
    assert measure.other_gpu_processes("", own_pid=1) == []


def test_measure_stage_writes_cost_tables(mo433_root, monkeypatch):
    import csv

    _write_json(mo433_root / "runs/C-S-100-s0/summary.json", {"wall_seconds": 3600, "gpu_seconds": 3600})
    _write_json(mo433_root / "teacher/cache_timing.json", {"gpu_seconds": 360})
    monkeypatch.setattr(measure, "query_gpu_processes", lambda: ["999 trainer 3000 MiB"])

    measure.measure(teacher=False, timed=2, warmup=1)

    costs = list(csv.DictReader(open(mo433_root / "results/costs.csv")))
    assert [r["encoder_id"] for r in costs] == ["ResNet96-S", "ResNet96-L"]
    assert {"params", "macs", "flops", "feature_dim", "gpu_latency_ms_mean", "gpu_latency_ms_sd",
            "cpu_latency_ms_mean", "provisional"} <= set(costs[0])
    assert float(costs[0]["cpu_latency_ms_mean"]) > 0
    if torch.cuda.is_available():
        assert costs[0]["provisional"] == "True" and "999 trainer" in costs[0]["gpu_other_processes"]
    runs = list(csv.DictReader(open(mo433_root / "results/runs_cost.csv")))
    assert runs[0]["run_id"] == "C-S-100-s0" and float(runs[0]["total_gpu_h"]) == 1.1
    summary = json.loads((mo433_root / "results/costs_summary.json").read_text())
    assert summary["teacher_pretrain_gpu_h_estimate"] == 3100
    assert summary["teacher_cache_gpu_h"] == 0.1


def test_cli_measure_runs_as_a_stage(mo433_root, monkeypatch):
    from mo433 import cli

    calls = []
    monkeypatch.setattr(measure, "measure", lambda profile="full", **kw: calls.append(profile))
    assert cli.main(["measure"]) == 0
    assert cli.main(["measure"]) == 0  # done marker: skipped
    assert cli.main(["measure", "--force"]) == 0
    assert calls == ["full", "full"]


def test_missing_runs_and_cache_give_no_rows(mo433_root):
    assert measure.run_costs() == []
    assert measure.teacher_cache_gpu_seconds() is None
