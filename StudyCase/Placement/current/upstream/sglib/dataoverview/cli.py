from __future__ import annotations

import argparse
from pathlib import Path
import sys

from sglib.core.infra.config import LoadedConfig, load_dataoverview_config
from sglib.core.infra.paths import find_repo_root

from .processing.registry import build_registry, status, topological_order
from .processing.unit import UnitContext


def _load_countries(repo_root: Path) -> dict[str, LoadedConfig]:
    stage = repo_root / "casestudy" / "1_DataOverview"
    profile_root = repo_root / "casestudy" / "config" / "countries"
    loaded: dict[str, LoadedConfig] = {}
    for profile_path in sorted(profile_root.glob("*.toml")):
        code = profile_path.stem.lower()
        from sglib.core.infra.terms import load_country_profile

        profile = load_country_profile(profile_path)
        overlay = stage / profile.directory / f"{code}.toml"
        if overlay.is_file():
            loaded[code] = load_dataoverview_config(
                stage / "general" / "general.toml", profile_path, overlay
            )
    if not loaded:
        raise RuntimeError("no DataOverview country overlays were discovered")
    return loaded


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the DataOverview unit DAG")
    parser.add_argument("--phase", help="processing or overview")
    parser.add_argument("--country")
    parser.add_argument("--category")
    parser.add_argument("--dataset", help="dataset or product id")
    parser.add_argument("--step", help="download, derive, grid, features, or overview")
    parser.add_argument("--list", action="store_true", dest="list_only")
    parser.add_argument("--show-config", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    return parser


def _selected(args: argparse.Namespace, units: dict) -> set[str]:
    selected: set[str] = set()
    for unit in units.values():
        if args.phase and unit.phase != args.phase:
            continue
        if args.country and unit.country != args.country:
            continue
        if args.category and unit.category != args.category:
            continue
        parts = unit.id.split(".")
        member = parts[1] if len(parts) > 2 else ""
        if args.dataset and member != args.dataset:
            continue
        if args.step and args.step not in {unit.step, member}:
            continue
        selected.add(unit.id)
    if not selected:
        raise ValueError("filters select no DataOverview units; use --list to inspect the registry")
    return selected


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    args = _parser().parse_args(argv)
    repo_root = find_repo_root(__file__)
    loaded = _load_countries(repo_root)
    if args.country and args.country not in loaded:
        raise ValueError(f"unknown country {args.country!r}; discovered={sorted(loaded)}")
    if args.show_config:
        if not args.country:
            raise ValueError("--show-config requires --country")
        for key, value, source in loaded[args.country].explain():
            print(f"{key} = {value!r}  # {source.relative_to(repo_root)}")
        return 0
    configurations = {code: dict(item.values) for code, item in loaded.items()}
    units = build_registry(repo_root, configurations)
    selected = _selected(args, units)
    ordered = topological_order(units, selected)
    if args.list_only:
        for unit in ordered:
            dependency = "-" if not unit.depends_on else ",".join(unit.depends_on)
            print(f"{unit.id}\t{unit.phase}\t{unit.category}\t{status(unit)}\tdepends={dependency}")
        return 0
    for unit in ordered:
        refresh = bool(args.refresh and unit.id in selected)
        state = status(unit)
        if state.startswith("BLOCKED") and not unit.done():
            raise RuntimeError(f"{unit.id}: {state}")
        if unit.done() and not refresh:
            print(f"SKIP {unit.id}")
            continue
        print(f"RUN  {unit.id}")
        config = configurations.get(unit.country, {})
        unit.run(UnitContext(repo_root=repo_root, config=config, refresh=refresh), unit)
        if not unit.done():
            raise RuntimeError(f"{unit.id}: runner returned without satisfying its output contract")
        print(f"PASS {unit.id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
