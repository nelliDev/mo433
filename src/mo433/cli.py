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
    raise NotImplementedError("pool")


@command("teacher", help="teacher sanity check and teacher-feature cache (spec §3, §4.1)")
def _teacher(ctx: Context, args: argparse.Namespace) -> None:
    raise NotImplementedError("teacher")


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


@command("train", help="one run of row B or C", add_args=_train_args)
def _train(ctx: Context, args: argparse.Namespace) -> None:
    if args.on_the_fly and args.row != "B":
        raise SystemExit("--on-the-fly applies to row B only")
    raise NotImplementedError(f"train {config_mod.run_id(args.row, args.size, args.frac, args.seed)}")


@command("train-all", help="all 20 reported runs in week-3 order (spec §8)")
def _train_all(ctx: Context, args: argparse.Namespace) -> None:
    raise NotImplementedError("train-all")


# ---------------------------------------------------------------- evaluation and reporting


def _probe_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--select-svm",
        action="store_true",
        help="choose SVM C/gamma by 5-fold CV on B-S-100-s0 (spec §7.2) instead of probing",
    )


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
    raise NotImplementedError("probe --select-svm" if args.select_svm else "probe")


@command("diagnose", help="collapse check and characterization (spec §7.3)")
def _diagnose(ctx: Context, args: argparse.Namespace) -> None:
    raise NotImplementedError("diagnose")


@command("measure", help="params, FLOPs, latency, GPU-hours (spec §7.4)")
def _measure(ctx: Context, args: argparse.Namespace) -> None:
    raise NotImplementedError("measure")


@command("report", help="results.csv -> LaTeX tables, macros and figures (spec §9)")
def _report(ctx: Context, args: argparse.Namespace) -> None:
    raise NotImplementedError("report")


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
