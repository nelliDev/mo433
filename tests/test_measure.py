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
    s, l = rows["S"], rows["L"]
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


def test_epoch_gpu_time_above_its_wall_time_is_clamped_and_counted(mo433_root):
    # #27: one bad CUDA-event reading (B-L-050-s0, epoch 85) made summary gpu_seconds 3x the wall time.
    _write_json(mo433_root / "runs/B-L-050-s0/summary.json", {"wall_seconds": 130, "gpu_seconds": 10_090})
    (mo433_root / "runs/B-L-050-s0/metrics.jsonl").write_text(
        '{"epoch": 1, "epoch_seconds": 60, "gpu_seconds": 45}\n'
        '{"epoch": 2, "epoch_seconds": 60, "gpu_seconds": 10045}\n'
    )

    (r,) = measure.run_costs()

    assert r["gpu_h"] == round((45 + 60) / 3600, 4)
    assert r["gpu_epochs_clamped"] == 1 and r["complete"] is True and r["wall_h"] == round(130 / 3600, 4)


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
    # A shared GPU, faked so the test never touches the real one.
    monkeypatch.setattr(measure, "query_gpu_processes", lambda: ["999 trainer 3000 MiB"])
    monkeypatch.setattr(measure, "gpu_name", lambda: "Fake 3060")
    monkeypatch.setattr(measure, "gpu_latency", lambda net, px, **kw: (64.0, 2.0))

    measure.measure(teacher=False, timed=2, warmup=1, cuda=True)

    costs = list(csv.DictReader(open(mo433_root / "results/costs.csv")))
    assert [r["encoder_id"] for r in costs] == ["S", "L"]
    s = costs[0]
    assert (s["gpu_latency_ms"], s["gpu_latency_ms_sd"], s["gpu_throughput_img_s"]) == ("64.0", "2.0", "4000.0")
    assert float(s["cpu_latency_ms"]) > 0 and s["gpu_name"] == "Fake 3060"
    assert s["provisional"] == "True" and "999 trainer" in s["gpu_other_processes"]
    runs = list(csv.DictReader(open(mo433_root / "results/runs_cost.csv")))
    assert runs[0]["run_id"] == "C-S-100-s0" and float(runs[0]["total_gpu_h"]) == 1.1
    summary = json.loads((mo433_root / "results/costs_summary.json").read_text())
    assert summary["teacher_pretrain_gpu_h_estimate"] == 3100
    assert summary["teacher_cache_gpu_h"] == 0.1


def test_report_reads_measure_outputs(mo433_root, monkeypatch):
    from mo433 import report

    _write_json(mo433_root / "runs/B-S-100-s0/summary.json", {"wall_seconds": 7200, "gpu_seconds": 3600})
    _write_json(mo433_root / "runs/C-S-100-s0/summary.json", {"wall_seconds": 3600, "gpu_seconds": 3600})
    _write_json(mo433_root / "teacher/cache_timing.json", {"gpu_seconds": 360})
    measure.measure(teacher=False, timed=2, warmup=1, cuda=False)  # CPU only

    costs = report.read_costs(mo433_root / "results/costs.csv")
    assert set(costs) == {"S", "L"}
    assert round(costs["S"]["params"] / 1e6, 2) == 1.57 and costs["L"]["feature_dim"] == 384
    assert costs["S"]["macs"] * 2 == costs["S"]["flops"] and costs["S"]["cpu_latency_ms"] > 0
    runs = report.read_runs_cost(mo433_root / "results/runs_cost.csv")
    assert runs["B-S-100-s0"] == {"wall_h": 2.0, "gpu_h": 1.0, "teacher_cache_gpu_h_charged": 0.0}
    assert runs["C-S-100-s0"] == {"wall_h": 1.0, "gpu_h": 1.0, "teacher_cache_gpu_h_charged": 0.1}


@pytest.mark.gpu
def test_gpu_latency_of_small_student_is_positive():
    if measure.query_gpu_processes():
        pytest.skip("GPU busy with other processes")
    from mo433.students import build_student

    mean, sd = measure.gpu_latency(build_student("S", 0).eval(), 96, batch=8, warmup=1, timed=3)
    assert mean > 0 and sd >= 0


def test_cli_measure_runs_as_a_stage(mo433_root, monkeypatch):
    from mo433 import cli

    calls = []
    monkeypatch.setattr(measure, "measure", lambda profile="full", **kw: calls.append(profile))
    assert cli.main(["measure"]) == 0
    assert cli.main(["measure"]) == 0  # done marker: skipped
    assert cli.main(["measure", "--force"]) == 0
    assert calls == ["full", "full"]


def test_teacher_cache_time_is_read_from_cache_json_in_hours(mo433_root):
    # the record `mo433 teacher` (#18) writes next to the cache
    _write_json(mo433_root / "teacher/cache.json", {"gpu_hours": 0.5, "wall_hours": 0.6, "spot_check": {}})
    assert measure.teacher_cache_gpu_seconds() == 1800.0


def test_missing_runs_and_cache_give_no_rows(mo433_root):
    assert measure.run_costs() == []
    assert measure.teacher_cache_gpu_seconds() is None
