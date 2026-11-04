"""Week-2 tuning protocol (spec §6): ``mo433 tune {optuna,grid,fixed-view,lars}``.

Every score is the spec §6 validation k-NN (k = 20, cosine) of the encoder's pooled output,
computed in fp32 on the **final** weights of each trial (:func:`score_run`); the per-epoch bf16
``val_knn`` of the training log only feeds the Optuna pruner. Every trial also gets the
spec §7.3 collapse check against row A at the same size and seed (spread-only when the
anchor's features are missing); flagged trials are recorded but never selected.

Outputs (``$MO433_ROOT/tuning/``, gitignored)::

    optuna_b.db            Optuna SQLite storage (study "optuna_b")
    optuna_b.json          study summary (trials, intermediate values, collapse flags, winner)
    optuna/trial_<n>/      run dir of each trial (checkpoint deleted once finished)
    grid_<row>.json, grid/<row>-w<width>-s<seed>/
    fixed_view.json, lars.json, lars/...

With the full profile the summaries (and the study DB) are also copied into the repo's
committed ``tuning/`` directory (spec §9). ``--apply`` writes the winners into ``configs/``;
the ``configs-frozen`` tag stays a manual step.

Decisions fixed here (documented in the summaries):

- MedianPruner(n_startup_trials=4, n_warmup_steps=5): no pruning until 4 trials have
  finished, and never before epoch 6 of a trial (warmup ≤ 4 epochs plus one epoch of decay).
- A trial that dies (crash, kill) is retried once per failure with the same params as a
  new trial (user attr ``retry_of``); its run dir is moved over and resumed from its last
  checkpoint (trials checkpoint every 5 epochs). Diverged trials (non-finite loss) are
  recorded as pruned with ``diverged=True``.
- Seed spread for the tie rule = |s0 − s1| (the range of the 2 seeds; the sample sd of two
  values is that over √2, so the range is the stricter bar).
"""

from __future__ import annotations

import json
import re
import math
import shutil
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mo433 import config as config_mod
from mo433 import paths

STUDY_NAME = "optuna_b"
N_TRIALS = 16
TRIAL_EPOCHS = 20
TRIAL_WIDTH = 2048
SAMPLER_SEED = 0
PRUNER = {"n_startup_trials": 4, "n_warmup_steps": 5}
TRIAL_CHECKPOINT_EVERY = 5
SPACE = {"lr": (3e-4, 3e-3), "weight_decay": (0.01, 0.2), "warmup_epochs": (1, 2, 4)}
GRID_WIDTHS = (1024, 2048, 4096)
GRID_SEEDS = (0, 1)
BASELINE_WIDTH = 2048
SWITCH_PP = 2.0  # fixed-view switch: on-the-fly must beat the pool by > 2 pp validation k-NN
LARS_LRS = (0.2, 1.0, 2.0)
LARS_WDS = (1e-6, 1e-4)

REPO_TUNING_DIR = paths.REPO_ROOT / "tuning"


# Spec §6.1: trial 0 is the smoke config, which is also the fallback. Pinned here rather than read
# from configs/base.yaml, because ``tune optuna --apply`` overwrites that file with the winner.
TRIAL0 = {"lr": 1e-3, "weight_decay": 0.05, "warmup_epochs": 2}


def trial0_params() -> dict[str, Any]:
    """The smoke config = the fallback (1e-3 / 0.05 / 2)."""
    return dict(TRIAL0)


# ---------------------------------------------------------------- selection rules


def select_winner(trials: list[dict]) -> dict | None:
    """Best ``value`` among COMPLETE trials that are not collapse-flagged (None if none)."""
    ok = [t for t in trials if t["state"] == "COMPLETE" and t.get("value") is not None and not t.get("collapse_flagged")]
    return max(ok, key=lambda t: t["value"]) if ok else None


def seed_spread(values: list[float]) -> float:
    return max(values) - min(values)


def tie_rule(scores: dict[int, list[float | None]], baseline: int = BASELINE_WIDTH) -> dict:
    """Spec §6.2: a width replaces ``baseline`` only if its seed mean beats the baseline's by
    more than the larger of the two seed spreads (|s0 − s1|). A width with a missing or
    collapse-flagged seed (``None``) is ineligible. Among qualifying widths the highest
    mean wins. If the baseline itself is ineligible, the best eligible mean wins outright.
    """
    cands: dict[int, dict] = {}
    for w, vals in scores.items():
        eligible = len(vals) > 0 and all(v is not None for v in vals)
        cands[w] = {
            "values": vals,
            "eligible": eligible,
            "mean": sum(vals) / len(vals) if eligible else None,
            "spread": seed_spread(vals) if eligible else None,
        }
    base = cands.get(baseline)
    if base is None or not base["eligible"]:
        elig = {w: c for w, c in cands.items() if c["eligible"]}
        winner = max(elig, key=lambda w: elig[w]["mean"]) if elig else baseline
        return {"baseline": baseline, "baseline_eligible": False, "winner": winner, "candidates": cands,
                "outcome": "baseline ineligible: best eligible mean" if elig else "no eligible width: keep baseline"}
    qualifying = []
    for w, c in cands.items():
        if w == baseline:
            continue
        if c["eligible"]:
            c["margin"] = c["mean"] - base["mean"]
            c["threshold"] = max(c["spread"], base["spread"])
            c["beats_baseline"] = c["margin"] > c["threshold"]
            if c["beats_baseline"]:
                qualifying.append(w)
        else:
            c["beats_baseline"] = False
    winner = max(qualifying, key=lambda w: cands[w]["mean"]) if qualifying else baseline
    return {"baseline": baseline, "baseline_eligible": True, "winner": winner, "candidates": cands,
            "outcome": f"{winner} replaces {baseline}" if winner != baseline else f"keep {baseline}"}


# ---------------------------------------------------------------- scoring a finished run


class _Scorer:
    """fp32 validation k-NN + collapse check of a run's final backbone (spec §6, §7.3)."""

    def __init__(self, profile: str, device: str):
        from mo433.data import load_stl10

        self.profile, self.device = profile, device
        images, labels = load_stl10("train")
        self.images = images.to(device)
        self.labels = labels
        self._anchors: dict[str, Any] = {}

    def anchor_stats(self, size: str, seed: int):
        from mo433 import diagnostics, probe

        aid = f"A-{size}-s{seed}"
        if aid not in self._anchors:
            stats = None
            if (paths.features_dir(self.profile) / f"{aid}.npz").exists():
                a = probe.load_features(aid, self.profile)
                stats = diagnostics.collapse_stats(a["train_x"], a["train_y"])
            self._anchors[aid] = stats
        return aid, self._anchors[aid]

    def __call__(self, run_dir: Path, size: str, seed: int) -> dict:
        import torch

        from mo433 import diagnostics, knn
        from mo433.encoders import FINAL_BACKBONE
        from mo433.students import build_student

        net = build_student(size, seed)
        net.load_state_dict(torch.load(Path(run_dir) / FINAL_BACKBONE, map_location="cpu", weights_only=True))
        net.to(self.device)
        feats = knn.encode(net, self.images).cpu().numpy()
        labels = self.labels.numpy()
        value = knn.val_knn(feats, labels)
        stats = diagnostics.collapse_stats(feats, labels)
        aid, anchor = self.anchor_stats(size, seed)
        check = diagnostics.collapse_check(stats, anchor)
        del net
        return {
            "val_knn": value,
            "collapse_flagged": bool(check.flagged),
            "collapse": {
                "reasons": check.reasons,
                "stats": _asdict(stats),
                "anchor_id": aid if anchor is not None else None,
                "anchor": _asdict(anchor) if anchor is not None else None,
            },
        }


def _asdict(x):
    import dataclasses

    return dataclasses.asdict(x)


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _drop_checkpoint(run_dir: Path) -> None:
    from mo433.train import CHECKPOINT

    (Path(run_dir) / CHECKPOINT).unlink(missing_ok=True)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    tmp.replace(path)


def _publish(profile: str, name: str, summary: dict, db: Path | None = None) -> None:
    """Write the summary under ``$MO433_ROOT/tuning`` and, for the full profile, into the repo."""
    _write_json(paths.tuning_dir(profile) / name, summary)
    if profile == "full":
        _write_json(REPO_TUNING_DIR / name, summary)
        if db is not None and db.exists():
            _sqlite_copy(db, REPO_TUNING_DIR / db.name)


def _sqlite_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    s, d = sqlite3.connect(src), sqlite3.connect(dst)
    try:
        s.backup(d)
    finally:
        s.close()
        d.close()


# ---------------------------------------------------------------- 6.1 Optuna study


def _study(db: Path, create: bool = True):
    import optuna

    return optuna.create_study(
        study_name=STUDY_NAME,
        storage=f"sqlite:///{db}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SAMPLER_SEED),
        pruner=optuna.pruners.MedianPruner(**PRUNER),
        load_if_exists=True,
    )


def _requeue_dead_trials(study, log) -> None:
    """Fail trials left RUNNING by a killed process; retry every failed trial once (same params)."""
    from optuna.trial import TrialState

    trials = study.get_trials(deepcopy=False)
    retried = {t.user_attrs.get("retry_of") for t in trials}
    for t in trials:
        if t.state == TrialState.RUNNING:  # one process per study: anything RUNNING now is dead
            study._storage.set_trial_state_values(t._trial_id, TrialState.FAIL)
        if t.state in (TrialState.RUNNING, TrialState.FAIL) and t.number not in retried:
            log(f"[tune optuna] retrying dead trial {t.number} with the same params {t.params}")
            study.enqueue_trial(t.params, user_attrs={"retry_of": t.number}, skip_if_exists=False)


def _finished(study) -> int:
    from optuna.trial import TrialState

    return len(study.get_trials(deepcopy=False, states=(TrialState.COMPLETE, TrialState.PRUNED)))


def run_optuna(
    profile: str = "full",
    n_trials: int | None = None,
    epochs: int | None = None,
    overrides: dict[str, Any] | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    """Run (or continue) the spec §6.1 study until ``n_trials`` trials have finished
    (complete or pruned), then write the summary. Rerunning a finished study only rewrites
    the summary."""
    import optuna

    from mo433 import train

    n_trials = n_trials if n_trials is not None else (N_TRIALS if profile == "full" else 2)
    if epochs is None:
        epochs = TRIAL_EPOCHS if profile == "full" else config_mod.load_config("B", "S", profile).epochs
    root = paths.tuning_dir(profile)
    root.mkdir(parents=True, exist_ok=True)
    db = root / f"{STUDY_NAME}.db"
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = _study(db)
    if not study.get_trials(deepcopy=False):
        study.enqueue_trial(trial0_params())
    _requeue_dead_trials(study, log)
    scorer: _Scorer | None = None
    extra = dict(overrides or {})

    def objective(trial: optuna.Trial) -> float:
        nonlocal scorer
        params = {
            "lr": trial.suggest_float("lr", *SPACE["lr"], log=True),
            "weight_decay": trial.suggest_float("weight_decay", *SPACE["weight_decay"], log=True),
            "warmup_epochs": trial.suggest_categorical("warmup_epochs", list(SPACE["warmup_epochs"])),
        }
        rd = root / "optuna" / f"trial_{trial.number}"
        prev = trial.user_attrs.get("retry_of")
        if prev is not None and (root / "optuna" / f"trial_{prev}").exists() and not rd.exists():
            (root / "optuna" / f"trial_{prev}").rename(rd)
        # Re-report the epochs a resumed run already logged, so the pruner sees them.
        metrics = rd / train.METRICS
        if metrics.exists():
            for line in metrics.read_text().splitlines():
                rec = json.loads(line)
                if "val_knn" in rec:
                    trial.report(rec["val_knn"], rec["epoch"])
        log(f"[tune optuna] trial {trial.number}: {params}" + (f" (retry of {prev})" if prev is not None else ""))

        def on_epoch(epoch: int, m: dict) -> None:
            if "val_knn" in m and math.isfinite(m["val_knn"]):
                trial.report(m["val_knn"], epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned(f"pruned at epoch {epoch} (val_knn {m['val_knn']:.4f})")

        ov = {"checkpoint_every": TRIAL_CHECKPOINT_EVERY, **extra, **params, "head_width": extra.get("head_width", TRIAL_WIDTH)}
        try:
            summary = train.train_run("B", "S", 100, 0, profile=profile, overrides=ov, run_dir=rd, epochs=epochs,
                                      epoch_callback=on_epoch, log=log)
        except optuna.TrialPruned:
            _drop_checkpoint(rd)
            raise
        except train.TrainingDiverged as e:
            trial.set_user_attr("diverged", True)
            _drop_checkpoint(rd)
            raise optuna.TrialPruned(str(e)) from e
        if scorer is None:
            scorer = _Scorer(profile, summary["settings"]["device"])
        res = scorer(rd, "S", 0)
        trial.set_user_attr("collapse_flagged", res["collapse_flagged"])
        trial.set_user_attr("collapse", res["collapse"])
        trial.set_user_attr("gpu_seconds", summary["gpu_seconds"])
        trial.set_user_attr("bf16_val_knn_last", summary["final"].get("val_knn"))
        _drop_checkpoint(rd)
        log(f"[tune optuna] trial {trial.number}: val_knn={res['val_knn']:.4f} flagged={res['collapse_flagged']}")
        return res["val_knn"]

    while _finished(study) < n_trials:
        study.optimize(objective, n_trials=1, gc_after_trial=True)

    summary = optuna_summary(study, n_trials=n_trials, epochs=epochs, profile=profile)
    _publish(profile, f"{STUDY_NAME}.json", summary, db)
    w = summary["winner"]
    log(f"[tune optuna] {summary['n_finished']} trials finished; winner: "
        + (f"trial {w['number']} {w['params']} val_knn={w['value']:.4f}" if w else "none (all collapsed)"))
    return summary


def optuna_summary(study, *, n_trials: int, epochs: int, profile: str) -> dict:
    trials = []
    for t in study.get_trials(deepcopy=False):
        trials.append({
            "number": t.number,
            "state": t.state.name,
            "params": t.params,
            "value": t.value,
            "intermediate_val_knn_bf16": {int(k): v for k, v in sorted(t.intermediate_values.items())},
            "collapse_flagged": t.user_attrs.get("collapse_flagged"),
            "collapse": t.user_attrs.get("collapse"),
            "diverged": t.user_attrs.get("diverged", False),
            "retry_of": t.user_attrs.get("retry_of"),
            "gpu_seconds": t.user_attrs.get("gpu_seconds"),
            "duration_seconds": t.duration.total_seconds() if t.duration else None,
        })
    w = select_winner(trials)
    return {
        "study": STUDY_NAME,
        "profile": profile,
        "protocol": {
            "row": "B", "size": "S", "frac": 100, "seed": 0, "head_width": TRIAL_WIDTH, "epochs": epochs,
            "n_trials": n_trials, "sampler": f"TPESampler(seed={SAMPLER_SEED})", "pruner": {"MedianPruner": PRUNER},
            "space": {"lr": ["log-uniform", *SPACE["lr"]], "weight_decay": ["log-uniform", *SPACE["weight_decay"]],
                      "warmup_epochs": list(SPACE["warmup_epochs"])},
            "trial0": trial0_params(),
            "objective": "fp32 validation k-NN (k=20, cosine) of the final backbone; pruner sees per-epoch bf16 val_knn",
            "selection": "best COMPLETE trial that is not collapse-flagged (spec §7.3)",
        },
        "n_finished": sum(t["state"] in ("COMPLETE", "PRUNED") for t in trials),
        "trials": trials,
        "winner": None if w is None else {"number": w["number"], "params": w["params"], "value": w["value"]},
    }


# ---------------------------------------------------------------- shared: one scored 20-epoch run

SCORE = "score.json"


def _default_epochs(profile: str, full: int = TRIAL_EPOCHS) -> int:
    return full if profile == "full" else config_mod.load_config("B", "S", profile).epochs


def _scored_run(row: str, seed: int, run_dir: Path, overrides: dict, epochs: int, profile: str,
                scorer_box: list, log) -> dict:
    """Train (or resume) one S run at frac 100 and score its final weights; cached in ``score.json``."""
    from mo433 import train

    run_dir = Path(run_dir)
    cached = run_dir / SCORE
    if cached.exists():
        return json.loads(cached.read_text())
    try:
        summary = train.train_run(row, "S", 100, seed, profile=profile, overrides=overrides, run_dir=run_dir,
                                  epochs=epochs, log=log)
    except train.TrainingDiverged as e:
        res = {"val_knn": None, "collapse_flagged": True, "diverged": True, "error": str(e)}
    else:
        if not scorer_box:
            scorer_box.append(_Scorer(profile, summary["settings"]["device"]))
        res = {**scorer_box[0](run_dir, "S", seed), "diverged": False, "gpu_seconds": summary["gpu_seconds"],
               "bf16_val_knn_last": summary["final"].get("val_knn")}
    _drop_checkpoint(run_dir)
    _write_json(cached, res)
    return res


def _optuna_optimizer(profile: str) -> tuple[dict, str]:
    """The Optuna winner (lr, wd, warmup) if its summary exists, else the configs' optimizer."""
    for path in (paths.tuning_dir(profile) / f"{STUDY_NAME}.json", REPO_TUNING_DIR / f"{STUDY_NAME}.json"):
        if path.exists() and (profile == "full" or path.parent != REPO_TUNING_DIR):
            w = json.loads(path.read_text()).get("winner")
            if w:
                return dict(w["params"]), f"optuna winner (trial {w['number']})"
    return trial0_params(), "configs/base.yaml optimizer (no optuna winner found)"


# ---------------------------------------------------------------- 6.2 equal-effort grid


def run_grid(profile: str = "full", rows: list[str] | None = None, epochs: int | None = None,
             overrides: dict[str, Any] | None = None, log: Callable[[str], None] = print) -> dict[str, dict]:
    """B expander / C projector width ∈ {1024, 2048, 4096} × seeds {0, 1}, 20 epochs, with the
    Optuna winner's optimizer; the tie rule picks each row's width (spec §6.2)."""
    rows = list(rows or config_mod.ROWS)
    epochs = epochs if epochs is not None else _default_epochs(profile)
    opt, source = _optuna_optimizer(profile)
    scorer_box: list = []
    out = {}
    for row in rows:
        points = []
        for width in GRID_WIDTHS:
            for seed in GRID_SEEDS:
                name = f"{row}-w{width}-s{seed}"
                rd = paths.tuning_dir(profile) / "grid" / name
                log(f"[tune grid] {name}: {opt}")
                ov = {"checkpoint_every": TRIAL_CHECKPOINT_EVERY, **(overrides or {}), **opt, "head_width": width}
                res = _scored_run(row, seed, rd, ov, epochs, profile, scorer_box, log)
                points.append({"name": name, "width": width, "seed": seed, **res})
                log(f"[tune grid] {name}: val_knn={res['val_knn']} flagged={res['collapse_flagged']}")
        scores = {w: [p["val_knn"] if not p["collapse_flagged"] else None for p in points if p["width"] == w]
                  for w in GRID_WIDTHS}
        rule = tie_rule(scores)
        summary = {
            "row": row, "size": "S", "profile": profile, "epochs": epochs, "seeds": list(GRID_SEEDS),
            "widths": list(GRID_WIDTHS), "optimizer": opt, "optimizer_source": source,
            "score": "fp32 validation k-NN (k=20, cosine) of the final backbone",
            "spread_definition": "|s0 - s1| (range of the two seeds)",
            "collapse_rule": "a width with any collapse-flagged (or diverged) seed is ineligible",
            "points": points,
            "seed_spreads": {w: c["spread"] for w, c in rule["candidates"].items()},
            "tie_rule": rule,
            "winner": rule["winner"],
        }
        _publish(profile, f"grid_{row}.json", summary)
        log(f"[tune grid] {row}: {rule['outcome']} -> head_width {rule['winner']}")
        out[row] = summary
    return out


# ---------------------------------------------------------------- 6.3 LARS check


def run_lars(profile: str = "full", epochs: int | None = None, overrides: dict[str, Any] | None = None,
             log: Callable[[str], None] = print) -> dict:
    """6 runs: B, S, seed 0, 20 epochs, LARS with base lr ∈ {0.2, 1, 2} × wd ∈ {1e-6, 1e-4},
    lr = base × batch/256, warmup from the Optuna winner (spec §6.3). Meant for Kaggle."""
    epochs = epochs if epochs is not None else _default_epochs(profile)
    opt, source = _optuna_optimizer(profile)
    batch = (overrides or {}).get("batch_size", config_mod.load_config("B", "S", profile).batch_size)
    scorer_box: list = []
    trials = []
    for base_lr in LARS_LRS:
        for wd in LARS_WDS:
            name = f"lr{base_lr:g}-wd{wd:g}"
            ov = {"checkpoint_every": TRIAL_CHECKPOINT_EVERY, **(overrides or {}), "optimizer": "lars",
                  "lr": base_lr * batch / 256, "weight_decay": wd, "warmup_epochs": opt["warmup_epochs"],
                  "head_width": TRIAL_WIDTH}
            log(f"[tune lars] {name}: lr={ov['lr']:g}")
            res = _scored_run("B", 0, paths.tuning_dir(profile) / "lars" / name, ov, epochs, profile, scorer_box, log)
            trials.append({"name": name, "base_lr": base_lr, "lr": ov["lr"], "weight_decay": wd, **res})
            log(f"[tune lars] {name}: val_knn={res['val_knn']} flagged={res['collapse_flagged']}")
    best = max((t for t in trials if not t["collapse_flagged"] and t["val_knn"] is not None),
               key=lambda t: t["val_knn"], default=None)
    optuna_value = None
    p = paths.tuning_dir(profile) / f"{STUDY_NAME}.json"
    if p.exists() and json.loads(p.read_text()).get("winner"):
        optuna_value = json.loads(p.read_text())["winner"]["value"]
    summary = {
        "profile": profile, "epochs": epochs, "batch_size": batch, "lr_scaling": "lr = base_lr * batch / 256",
        "optimizer": "lightly LARS, momentum 0.9, trust 0.001; bias/BN excluded from wd and adaptation",
        "warmup_epochs": opt["warmup_epochs"], "warmup_source": source,
        "trials": trials,
        "best": best and {k: best[k] for k in ("name", "base_lr", "weight_decay", "val_knn")},
        "optuna_winner_value": optuna_value,
        "reopens_optimizer_decision": None if best is None or optuna_value is None else best["val_knn"] > optuna_value,
    }
    _publish(profile, "lars.json", summary)
    return summary


# ---------------------------------------------------------------- 6.4 fixed-view control

MAIN_POOL_RUN = "B-S-100-s0"
FROZEN_TAG = "configs-frozen"
_FV_CONFIGS = ("base.yaml", "B-S.yaml")


def configs_hash() -> str:
    import hashlib

    h = hashlib.sha256()
    for name in _FV_CONFIGS:
        h.update((paths.CONFIGS_DIR / name).read_bytes())
    return h.hexdigest()


def frozen_configs_identical() -> bool:
    """True if the ``configs-frozen`` tag exists and B-S's configs are byte-identical to it."""
    import subprocess

    for name in _FV_CONFIGS:
        r = subprocess.run(["git", "show", f"{FROZEN_TAG}:configs/{name}"], cwd=paths.REPO_ROOT, capture_output=True)
        if r.returncode != 0 or r.stdout != (paths.CONFIGS_DIR / name).read_bytes():
            return False
    return True


def _twin_features(run_dir: Path, encoder_id: str, profile: str) -> dict:
    """Spec §7.1 features of a twin's final backbone (train + test splits), cached as an npz."""
    import numpy as np
    import torch

    from mo433 import extract
    from mo433.data import load_stl10
    from mo433.encoders import FINAL_BACKBONE, student_encoder
    from mo433.students import build_student

    out = paths.features_dir(profile) / f"{encoder_id}.npz"
    if not out.exists():
        net = build_student("S", 0)
        net.load_state_dict(torch.load(run_dir / FINAL_BACKBONE, map_location="cpu", weights_only=True))
        enc = student_encoder(encoder_id, net).to(_device())
        arrays = {"encoder_id": np.array(encoder_id)}
        for split in extract.SPLITS:
            images, labels = load_stl10(split)
            arrays[f"{split}_y"] = labels.numpy().astype(np.int64)
            arrays.update({f"{split}_{k}": v for k, v in extract.extract_features(enc, images).items()})
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(f".{out.stem}.tmp.npz")
        np.savez(tmp, **arrays)
        tmp.replace(out)
    with np.load(out) as z:
        return {k: z[k] for k in z.files}


def _evaluate_twin(feats: dict, encoder_id: str, profile: str) -> dict:
    from mo433 import diagnostics, knn, probe

    svm = probe.svm_params(config_mod.load_config(profile=profile).probe)
    stats = diagnostics.collapse_stats(feats["train_x"], feats["train_y"])
    anchor = None
    if (paths.features_dir(profile) / "A-S-s0.npz").exists():
        a = probe.load_features("A-S-s0", profile)
        anchor = diagnostics.collapse_stats(a["train_x"], a["train_y"])
    check = diagnostics.collapse_check(stats, anchor)
    rows = probe.probe_features(feats, svm, encoder_id=encoder_id)
    test = {r["probe"]: r["accuracy"] for r in rows if r["label_frac"] == 100 and r["draw"] == 0}
    return {
        "encoder_id": encoder_id,
        "val_knn": knn.val_knn(feats["train_x"], feats["train_y"]),
        "collapse_flagged": bool(check.flagged), "collapse_reasons": check.reasons,
        "test_100pct_draw0": test, "svm_frozen": None if svm is None else svm.frozen,
        "probe_rows": rows,
    }


def run_fixed_view(profile: str = "full", epochs: int | None = None, overrides: dict[str, Any] | None = None,
                   log: Callable[[str], None] = print) -> dict:
    """Spec §6.4: B-S-100-s0 with the configured (tuned) settings at E = 100, on the pool and
    on fresh views. The switch reads the validation k-NN only; the test probes give the Δ."""
    from mo433 import extract, probe, stages, train

    cfg_epochs = config_mod.load_config("B", "S", profile).epochs
    epochs = epochs if epochs is not None else cfg_epochs
    ov = dict(overrides or {})
    main_eligible = not ov and epochs == cfg_epochs and frozen_configs_identical()
    root = paths.tuning_dir(profile) / "fixed_view"
    tuning_pool = root / "pool"
    main_dir = paths.run_dir(MAIN_POOL_RUN, profile)
    stage = f"train/{MAIN_POOL_RUN}"

    # A pool twin trained before the freeze becomes the main run once the frozen configs match it.
    hash_file = tuning_pool / "configs.sha256"
    if (main_eligible and (tuning_pool / SUMMARY_NAME).exists() and not main_dir.exists()
            and hash_file.exists() and hash_file.read_text().strip() == configs_hash()):
        log(f"[tune fixed-view] promoting the pool twin to main run {MAIN_POOL_RUN}")
        shutil.copytree(tuning_pool, main_dir)
        stages.run_stage(stage, lambda: None, profile=profile)

    twins = {}
    for kind, otf in (("pool", False), ("on_the_fly", True)):
        if kind == "pool" and main_eligible:
            rd, eid = main_dir, MAIN_POOL_RUN
            stages.run_stage(stage, lambda: train.train_run("B", "S", 100, 0, profile=profile, run_dir=rd, log=log),
                             profile=profile)
            stages.run_stage(f"extract/{eid}", lambda: extract.extract(eid, profile), profile=profile)
            feats = probe.load_features(eid, profile)
        else:
            rd = tuning_pool if kind == "pool" else root / "otf"
            eid = f"{MAIN_POOL_RUN}-fv-{'pool' if kind == 'pool' else 'otf'}"
            if not (rd / SUMMARY_NAME).exists():
                (paths.features_dir(profile) / f"{eid}.npz").unlink(missing_ok=True)
            train.train_run("B", "S", 100, 0, profile=profile, overrides=ov or None, run_dir=rd, epochs=epochs,
                            on_the_fly=otf, log=log)
            if kind == "pool":
                hash_file.write_text(configs_hash() + "\n")
            feats = _twin_features(rd, eid, profile)
        twins[kind] = {**_evaluate_twin(feats, eid, profile), "run_dir": str(rd)}
        log(f"[tune fixed-view] {kind}: val_knn={twins[kind]['val_knn']:.4f} test={twins[kind]['test_100pct_draw0']}")

    pool, otf_ = twins["pool"], twins["on_the_fly"]
    delta_val = 100 * (otf_["val_knn"] - pool["val_knn"])
    summary = {
        "profile": profile, "epochs": epochs, "overrides": ov, "configs_sha256": configs_hash(),
        "pool_twin_is_main_run": main_eligible,
        "pool": pool, "on_the_fly": otf_,
        "delta_val_pp": delta_val,
        "switch_rule": f"on-the-fly beats the pool by > {SWITCH_PP} pp validation k-NN (pre-registered)",
        "switch": delta_val > SWITCH_PP,
        "delta_test_pp": {p: 100 * (otf_["test_100pct_draw0"][p] - pool["test_100pct_draw0"][p])
                          for p in pool["test_100pct_draw0"]},
    }
    _publish(profile, "fixed_view.json", summary)
    log(f"[tune fixed-view] Δval={delta_val:+.2f} pp -> switch={summary['switch']}; Δtest={summary['delta_test_pp']}")
    return summary


SUMMARY_NAME = "summary.json"


# ---------------------------------------------------------------- --apply


def _set_yaml_keys(path: Path, values: dict[str, Any], section: str | None = None) -> None:
    """Set top-level (or ``section:``-nested) scalar keys in place, keeping comments and layout."""
    from mo433.probe import _yaml_scalar

    lines = path.read_text().splitlines(keepends=True)
    inside, done = section is None, set()
    for i, line in enumerate(lines):
        if section is not None:
            if re.match(rf"^{section}:", line):
                inside = True
                continue
            if inside and re.match(r"^\S", line):
                break
        if not inside:
            continue
        pat = r"^(\s+)(\w+):(\s*)(\S+)(.*)$" if section else r"^()(\w+):(\s*)(\S+)(.*)$"
        m = re.match(pat, line.rstrip("\n"))
        if m and m.group(2) in values:
            indent, key, sp, _old, rest = m.groups()
            v = values[key]
            lines[i] = f"{indent}{key}:{sp or ' '}{v if isinstance(v, int) else _yaml_scalar(v)}{rest}\n"
            done.add(key)
    if set(values) - done:
        raise ValueError(f"{path}: keys not found: {sorted(set(values) - done)}")
    path.write_text("".join(lines))


def _summary(profile: str, name: str) -> dict:
    for p in (paths.tuning_dir(profile) / name, REPO_TUNING_DIR / name):
        if p.exists():
            return json.loads(p.read_text())
    raise SystemExit(f"no {name} under {paths.tuning_dir(profile)} or {REPO_TUNING_DIR}; run the study first")


def apply_winners(study: str, profile: str = "full", configs_dir: Path | None = None,
                  log: Callable[[str], None] = print) -> dict:
    """Write a study's winner into ``configs/`` (never tags ``configs-frozen``)."""
    cfg = Path(configs_dir) if configs_dir is not None else paths.CONFIGS_DIR
    if study == "optuna":
        w = _summary(profile, f"{STUDY_NAME}.json").get("winner")
        if not w:
            raise SystemExit("optuna study has no winner (every complete trial collapsed?)")
        p = w["params"]
        values = {"lr": float(p["lr"]), "weight_decay": float(p["weight_decay"]), "warmup_epochs": int(p["warmup_epochs"])}
        _set_yaml_keys(cfg / "base.yaml", values, section="optimizer")
        log(f"[tune optuna --apply] configs/base.yaml optimizer <- {values}")
        return values
    if study == "grid":
        applied = {}
        for row in config_mod.ROWS:
            try:
                width = int(_summary(profile, f"grid_{row}.json")["winner"])
            except SystemExit:
                log(f"[tune grid --apply] no grid_{row}.json: {row} unchanged")
                continue
            for size in config_mod.SIZES:  # L reuses S's winner (spec §6.2)
                _set_yaml_keys(cfg / f"{row}-{size}.yaml", {"head_width": width})
            applied[row] = width
            log(f"[tune grid --apply] configs/{row}-{{S,L}}.yaml head_width <- {width}")
        return applied
    raise SystemExit(f"tune {study} --apply: nothing to write into configs/ (see tuning/{study.replace('-', '_')}.json)")


# ---------------------------------------------------------------- CLI


def run_cli(study: str, *, profile: str, n_trials: int | None = None, epochs: int | None = None,
            rows: list[str] | None = None, apply: bool = False, force: bool = False,
            overrides: dict[str, Any] | None = None) -> None:
    if apply:
        apply_winners(study, profile=profile)
    elif study == "optuna":
        run_optuna(profile=profile, n_trials=n_trials, epochs=epochs, overrides=overrides)
    elif study == "grid":
        run_grid(profile=profile, rows=rows, epochs=epochs, overrides=overrides)
    elif study == "lars":
        run_lars(profile=profile, epochs=epochs, overrides=overrides)
    elif study == "fixed-view":
        run_fixed_view(profile=profile, epochs=epochs, overrides=overrides)
    else:
        raise NotImplementedError(f"tune {study}")
