"""``uv run mo433 <subcommand>`` (spec §9 "CLI").

Adding a stage: write a handler ``def _name(ctx: Context, args) -> None`` decorated with
``@command("name", help=..., add_args=...)``. Import heavy modules (torch, ...) inside the
handler so ``--help`` stays fast. Wrap the stage's work in ``ctx.run_stage(stage_name, fn)``
to get ``.done`` skipping and ``--force``. Raise ``NotImplementedError`` for stubs.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass

from mo433 import config as config_mod
from mo433 import stages


@dataclass(frozen=True)
class Context:
    """What every handler gets: the profile and --force, plus helpers bound to them."""

    profile: str
    force: bool

    def config(self, row: str | None = None, size: str | None = None) -> config_mod.Config:
        return config_mod.load_config(row=row, size=size, profile=self.profile)

    def run_stage(self, name: str, build: Callable[[], object]) -> bool:
        return stages.run_stage(name, build, profile=self.profile, force=self.force)

    def is_done(self, name: str) -> bool:
        return stages.is_done(name, profile=self.profile)


@dataclass(frozen=True)
class _Command:
    name: str
    help: str
    handler: Callable[[Context, argparse.Namespace], object]
    add_args: Callable[[argparse.ArgumentParser], None] | None


COMMANDS: dict[str, _Command] = {}


def command(name: str, *, help: str, add_args: Callable[[argparse.ArgumentParser], None] | None = None):
    def register(fn):
        COMMANDS[name] = _Command(name, help, fn, add_args)
        return fn

    return register


# ---------------------------------------------------------------- setup


@command("splits", help="generate the committed index files under splits/ (spec §2)")
def _splits(ctx: Context, args: argparse.Namespace) -> None:
    from mo433 import splits

    splits.write_splits()  # committed outputs: always regenerated, no .done marker


@command("pool", help="write the K-view pool memmap (spec §3)")
def _pool(ctx: Context, args: argparse.Namespace) -> None:
    from mo433 import pool

    pool.run(profile=ctx.profile, force=ctx.force)


@command("teacher", help="teacher sanity check and teacher-feature cache (spec §3, §4.1)")
def _teacher(ctx: Context, args: argparse.Namespace) -> None:
    from mo433 import extract, teacher

    # Sanity first (cheap; spec §7.3: it must pass before any C run), then the cache. A failed
    # sanity check raises, leaving no teacher/sanity marker and building no cache.
    ctx.run_stage("extract/D", lambda: extract.extract("D", ctx.profile))
    ctx.run_stage("teacher/sanity", lambda: teacher.run_sanity(ctx.profile))
    ctx.run_stage("teacher/cache", lambda: teacher.build_teacher_cache(ctx.profile))


# ---------------------------------------------------------------- training


def _train_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--row", required=True, choices=config_mod.ROWS)
    p.add_argument("--size", required=True, choices=config_mod.SIZES)
    p.add_argument("--frac", required=True, type=int, choices=config_mod.FRACS, help="unlabeled-data percent")
    p.add_argument("--seed", required=True, type=int)
    p.add_argument(
        "--on-the-fly",
        action="store_true",
        help="row B only: fresh VICRegViewTransform views every step instead of the pool (spec §6.4)",
    )
    p.add_argument("--knn-every", type=int, default=None,
                   help="validation k-NN every N epochs (monitoring only; default 1, 0 = never)")


@command("train", help="one run of row B or C", add_args=_train_args)
def _train(ctx: Context, args: argparse.Namespace) -> None:
    if args.on_the_fly and args.row != "B":
        raise SystemExit("--on-the-fly applies to row B only")
    from mo433 import paths, train

    rid = config_mod.run_id(args.row, args.size, args.frac, args.seed)
    # The fixed-view control's on-the-fly twin gets its own run dir and stage name, so it
    # never collides with (or resumes from) the pool run of the same id.
    name = f"{rid}-otf" if args.on_the_fly else rid
    overrides = {"knn_every": args.knn_every} if args.knn_every is not None else None

    def build() -> None:
        train.train_run(
            args.row, args.size, args.frac, args.seed, profile=ctx.profile, overrides=overrides,
            run_dir=paths.run_dir(name, ctx.profile), on_the_fly=args.on_the_fly, fresh=ctx.force,
        )

    ctx.run_stage(f"train/{name}", build)


@command("train-all", help="all 20 reported runs in week-3 order (spec §8)")
def _train_all(ctx: Context, args: argparse.Namespace) -> None:
    raise NotImplementedError("train-all")


# ---------------------------------------------------------------- evaluation and reporting


def _eval_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--encoder",
        action="append",
        default=None,
        metavar="ID",
        help="encoder id (repeatable); default: every features/<id>.npz",
    )
    p.add_argument(
        "--provisional-svm",
        action="store_true",
        help="while probe C/gamma are null in config, run SVMs with C=1 / C=10,gamma=scale "
        "(rows marked svm_params_frozen=False) instead of k-NN only",
    )
    p.add_argument("--n-jobs", type=int, default=None, help="joblib workers (default: $MO433_N_JOBS or 6)")


def _probe_args(p: argparse.ArgumentParser) -> None:
    _eval_args(p)
    p.add_argument(
        "--select-svm",
        action="store_true",
        help="choose SVM C/gamma by 5-fold CV on --encoder (default B-S-100-s0, spec §7.2), "
        "write them into configs/base.yaml, and do not probe",
    )
    p.add_argument(
        "--export",
        action="store_true",
        help="afterwards copy $MO433_ROOT/results/results.csv to the committed results.csv at the repo root",
    )


def _diagnose_args(p: argparse.ArgumentParser) -> None:
    _eval_args(p)
    p.add_argument("--anchor", default=None, help="row-A encoder id to compare against (default: same size+seed)")
    p.add_argument("--no-tsne", action="store_true", help="skip the t-SNE projection")


def _encoders(ctx: Context, args: argparse.Namespace) -> list[str]:
    from mo433 import probe

    ids = getattr(args, "encoder", None) or probe.available_encoders(ctx.profile)
    if not ids:
        raise SystemExit("no cached features found; run `mo433 extract` first")
    return ids


def _svm(ctx: Context, args: argparse.Namespace):
    from mo433 import probe

    svm = probe.svm_params(ctx.config().probe, provisional=getattr(args, "provisional_svm", False))
    if svm is None:
        print("probe C/gamma are null in configs/base.yaml: k-NN only (run `probe --select-svm` or pass --provisional-svm)")
    elif not svm.frozen:
        print(f"provisional SVM settings (not frozen): {svm}")
    return svm


EXTRACT_ROWS = ("A", "B", "C", "D")


def _extract_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("encoder_ids", nargs="*", metavar="ENCODER_ID", help="e.g. A-S-s0 B-L-050-s1")
    p.add_argument("--rows", nargs="+", choices=EXTRACT_ROWS, help="every encoder of these rows")


@command("extract", help="frozen features for every encoder (spec §7.1)", add_args=_extract_args)
def _extract(ctx: Context, args: argparse.Namespace) -> None:
    from mo433 import encoders, extract

    ids = list(getattr(args, "encoder_ids", None) or [])
    rows = getattr(args, "rows", None)
    if not ids and not rows:  # `mo433 all`, or bare `mo433 extract`: every encoder
        rows = [r for r in EXTRACT_ROWS if r in encoders.ROW_IDS]
    missing_rows = [r for r in rows or [] if r not in encoders.ROW_IDS]
    if missing_rows:
        raise NotImplementedError(f"extract: row {', '.join(missing_rows)} has no registered encoders yet")
    ids += encoders.encoder_ids(rows or [])
    unknown = [i for i in ids if not encoders.is_registered(i)]
    if unknown:
        raise SystemExit(f"mo433 extract: unknown encoder id(s): {', '.join(unknown)}")
    for encoder_id in dict.fromkeys(ids):
        ctx.run_stage(f"extract/{encoder_id}", lambda e=encoder_id: extract.extract(e, ctx.profile))


@command("probe", help="k-NN / linear SVM / RBF SVM probes (spec §7.2)", add_args=_probe_args)
def _probe(ctx: Context, args: argparse.Namespace) -> None:
    from mo433 import probe

    n_jobs = getattr(args, "n_jobs", None) or probe.N_JOBS
    if args.select_svm:
        encoders = getattr(args, "encoder", None) or [probe.SELECTION_ENCODER]
        if len(encoders) != 1:
            raise SystemExit("--select-svm takes exactly one --encoder")
        try:
            rec = probe.select_svm(encoders[0], profile=ctx.profile, n_jobs=n_jobs)
        except probe.SvmSelectionRefused as e:
            raise SystemExit(f"mo433 probe --select-svm: {e}") from e
        print(
            f"selected linear C={rec['linear_c']} (cv {rec['linear_cv_acc']:.4f}), "
            f"rbf C={rec['rbf_c']} gamma={rec['rbf_gamma']} (cv {rec['rbf_cv_acc']:.4f}); "
            "written to configs/base.yaml"
        )
        return

    svm = _svm(ctx, args)
    for eid in _encoders(ctx, args):

        def build(eid=eid):
            rows = probe.probe_encoder(eid, svm, profile=ctx.profile, n_jobs=n_jobs)
            summary = {(r["probe"]): r["accuracy"] for r in rows if r["label_frac"] == 100 and r["draw"] == 0}
            print(f"{eid}: 100% labels " + ", ".join(f"{k}={v:.4f}" for k, v in summary.items()))

        # Only frozen-setting results are final; k-NN-only/provisional probes always rerun.
        if svm is not None and svm.frozen:
            ctx.run_stage(f"probe/{eid}", build)
        else:
            build()
    if getattr(args, "export", False):
        print(f"exported results.csv to {probe.export_results(ctx.profile)}")


@command("diagnose", help="collapse check and characterization (spec §7.3)", add_args=_diagnose_args)
def _diagnose(ctx: Context, args: argparse.Namespace) -> None:
    from mo433 import diagnostics, probe

    svm = _svm(ctx, args)
    for eid in _encoders(ctx, args):

        def build(eid=eid):
            rec = diagnostics.diagnose_encoder(
                eid,
                profile=ctx.profile,
                svm=svm,
                anchor=getattr(args, "anchor", None),
                tsne=not getattr(args, "no_tsne", False),
                n_jobs=getattr(args, "n_jobs", None) or probe.N_JOBS,
            )
            check = rec.get("teacher_check") or rec.get("collapse_check")
            verdict = "n/a (row A)" if check is None else (
                ("PASS" if check["passed"] else "FAIL") if "passed" in check else
                ("FLAGGED" if check["flagged"] else "ok")
            )
            print(f"{eid}: val {rec['val_stats']}, check {verdict}, test {rec['test_characterization']}")

        if svm is not None and svm.frozen:
            ctx.run_stage(f"diagnose/{eid}", build)
        else:
            build()


@command("measure", help="params, FLOPs, latency, GPU-hours (spec §7.4)")
def _measure(ctx: Context, args: argparse.Namespace) -> None:
    from mo433 import measure

    # Cheap (a few GPU minutes); re-run with --force on an idle GPU to replace provisional latency.
    ctx.run_stage("measure", lambda: measure.measure(ctx.profile))


def _report_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--repo-root",
        default=None,
        metavar="DIR",
        help="where the committed inputs (results.csv, report/data/) and outputs (report/) live; default: this repo",
    )
    p.add_argument(
        "--no-sync",
        action="store_true",
        help="do not copy $MO433_ROOT/results/* into the committed inputs; build from committed files only",
    )


@command("report", help="results.csv -> LaTeX tables, macros and figures (spec §9)", add_args=_report_args)
def _report(ctx: Context, args: argparse.Namespace) -> None:
    # Committed outputs: always regenerated, no .done marker (like `splits`). Never imports torch.
    from pathlib import Path

    from mo433 import report

    root = Path(args.repo_root) if getattr(args, "repo_root", None) else None
    inputs = report.Inputs.committed(root)
    if not getattr(args, "no_sync", False):
        report.sync_inputs(ctx.profile, inputs)
    report.build_report(inputs, (root / "report") if root else None)


# ---------------------------------------------------------------- tuning


TUNE_STUDIES = ("optuna", "grid", "fixed-view", "lars")


def _tune_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("study", choices=TUNE_STUDIES)


@command("tune", help="opt-in tuning studies (spec §6)", add_args=_tune_args)
def _tune(ctx: Context, args: argparse.Namespace) -> None:
    raise NotImplementedError(f"tune {args.study}")


# ---------------------------------------------------------------- everything

PIPELINE = ("splits", "pool", "teacher", "train-all", "extract", "probe", "diagnose", "measure", "report")


@command("all", help="splits -> report from the frozen configs")
def _all(ctx: Context, args: argparse.Namespace) -> None:
    for name in PIPELINE:
        COMMANDS[name].handler(ctx, argparse.Namespace(select_svm=False))


# ---------------------------------------------------------------- entry point


def _global_args(p: argparse.ArgumentParser, *, suppress: bool) -> None:
    # Accepted both before and after the subcommand; the subparser copy only overrides
    # when given (SUPPRESS), so `mo433 --profile smoke train ...` also works.
    d = (lambda v: argparse.SUPPRESS) if suppress else (lambda v: v)
    p.add_argument("--profile", choices=config_mod.PROFILES, default=d("full"), help="full (default) or smoke")
    p.add_argument("--force", action="store_true", default=d(False), help="rebuild finished stages")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mo433", description="MO433 distillation study pipeline (docs/SPEC.md)")
    _global_args(parser, suppress=False)
    sub = parser.add_subparsers(dest="command", required=True, metavar="<subcommand>")
    for cmd in COMMANDS.values():
        p = sub.add_parser(cmd.name, help=cmd.help, description=cmd.help)
        _global_args(p, suppress=True)
        if cmd.add_args:
            cmd.add_args(p)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ctx = Context(profile=args.profile, force=args.force)
    try:
        COMMANDS[args.command].handler(ctx, args)
    except NotImplementedError as e:
        print(f"mo433: {e or args.command}: not implemented yet", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
