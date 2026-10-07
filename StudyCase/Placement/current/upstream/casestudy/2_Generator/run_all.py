"""Run one country's Generator DAG against an explicit or profile results root."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from sglib.core.infra.paths import find_repo_root
from sglib.dataoverview.handoff import load_bundle as load_dataoverview
from sglib.generator import stage
from sglib.generator.registry import topological_order, unit_status


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--country", required=True)
    parser.add_argument("--step")
    parser.add_argument("--family")
    parser.add_argument("--group")
    parser.add_argument("--sweep")
    parser.add_argument("--unit", help="execute one exact unit without expanding dependencies")
    parser.add_argument("--profile", choices=("smoke", "preflight", "formal"), default="formal")
    parser.add_argument("--backend", choices=("local", "hpc"), default="local")
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--list", action="store_true", dest="list_only")
    parser.add_argument("--show-config", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    return parser


def _select(args, units) -> set[str]:
    if args.unit:
        if any((args.step, args.family, args.group, args.sweep)):
            raise ValueError("--unit cannot be combined with step/family/group/sweep filters")
        if args.unit not in units:
            raise ValueError(f"unknown Generator unit: {args.unit}")
        return {args.unit}
    selected = set()
    for unit in units.values():
        if args.step and unit.step != args.step:
            continue
        if args.family and not (unit.step == "materialize" and unit.member == args.family):
            continue
        if args.group and not (unit.step in {"train", "verify", "infer"} and unit.member == args.group):
            continue
        if args.sweep and not (unit.step == "sweeps" and unit.member == args.sweep):
            continue
        selected.add(unit.id)
    if not selected:
        raise ValueError("filters select no Generator unit; use --list")
    return selected


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = _parser().parse_args(argv)
    repo = find_repo_root(__file__)
    ctx = stage.country_context(repo, args.country, profile=args.profile,
        results_root=args.results_root, backend=args.backend, input_loader=load_dataoverview)
    if args.show_config:
        for key, value, source in ctx.loaded.explain():
            print(f"{key} = {value!r}  # {source.relative_to(repo)}")
        return 0
    selected = _select(args, ctx.units)
    if args.list_only:
        print(f"backend={ctx.backend} results_root={ctx.results} profile={ctx.profile} read_only={ctx.read_only}")
        order = [ctx.units[args.unit]] if args.unit else topological_order(ctx.units, selected)
        for unit in order:
            print(f"{unit.id}\t{unit_status(unit, ctx.root)}\tdepends={','.join(unit.depends_on) or '-'}")
        return 0
    stage.run_units(ctx, selected, expand=args.unit is None, refresh=args.refresh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
