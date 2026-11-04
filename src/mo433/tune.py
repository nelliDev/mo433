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


def trial0_params() -> dict[str, Any]:
    """The smoke config = the fallback in ``configs/base.yaml`` (1e-3 / 0.05 / 2)."""
    opt = config_mod.load_config("B", "S").optimizer
    return {"lr": opt.lr, "weight_decay": opt.weight_decay, "warmup_epochs": opt.warmup_epochs}


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


# ---------------------------------------------------------------- CLI


def run_cli(study: str, *, profile: str, n_trials: int | None = None, epochs: int | None = None,
            rows: list[str] | None = None, apply: bool = False, force: bool = False,
            overrides: dict[str, Any] | None = None) -> None:
    if apply:
        apply_winners(study, profile=profile)
        return
    if study == "optuna":
        run_optuna(profile=profile, n_trials=n_trials, epochs=epochs, overrides=overrides)
    else:
        raise NotImplementedError(f"tune {study}")


def apply_winners(study: str, profile: str = "full", configs_dir: Path | None = None) -> dict:
    raise NotImplementedError(f"tune {study} --apply")
