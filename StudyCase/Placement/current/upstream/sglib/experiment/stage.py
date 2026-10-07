"""Shared execution for the numbered Experiment scripts and notebooks (plan 006a).

Same shape as ``sglib.generator.stage``: a country context resolves the results
root, builds the unit registry and reports read-only (closed) roots; ``run_step``
executes selected units in DAG order, skipping DONE units with a reason and
refusing to run when predecessors are not DONE. Upstream handoffs are injected
by the orchestration entrypoint (``data_loader``, ``generator_loader``) so this
package never imports another stage.
"""

from __future__ import annotations

from functools import partial
from sglib.core.chain import stage as chain_stage
from sglib.core.chain.stage import StageOrderError, StepReport, resolve_results_root

from dataclasses import dataclass, field, replace
import inspect
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from sglib.core.infra.artifacts import atomic_json
from sglib.core.infra.content_chain import code_projection, verify_chain

from .config import LoadedExperimentConfig, load_experiment_config
from .registry import (PREPARED_MARKER, ExperimentUnit, build_registry, planning_progress, topological_order,
                       unit_output_path, unit_status)


COUNTRIES = ("uk", "au", "nl", "nz")
CLOSURE_SCHEMAS = {"sg_content_chain_closure_v1", "sg_006_planning_completion_v1"}




@dataclass(frozen=True)
class StageContext:
    repo: Path
    results: Path
    root: Path
    country: str
    profile: str
    backend: str
    loaded: LoadedExperimentConfig
    candidates: dict
    units: dict
    read_only: bool
    upstream: Mapping[str, Any] | None = None
    regions: tuple[str, ...] | None = None
    limit: int | None = None
    handoffs: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)






def with_upstream(ctx: StageContext, **dependencies) -> StageContext:
    """Create an independent execution context with explicit upstream loaders/objects.

    No cached dependency is carried across contexts. Objects can be provided
    directly when an orchestration has already built an extended handoff.
    """
    return replace(ctx, upstream={**dict(ctx.upstream or {}), **dependencies}, handoffs={})


def load_upstream(ctx: StageContext, name: str):
    """Resolve an upstream dependency once within this exact execution context."""
    if name not in (ctx.upstream or {}):
        raise ValueError(f"the orchestration entrypoint must provide upstream['{name}']")
    if name not in ctx.handoffs:
        dependency = ctx.upstream[name]
        ctx.handoffs[name] = dependency(ctx.repo, ctx.country) if callable(dependency) else dependency
    return ctx.handoffs[name]


def _closed_results(results: Path) -> bool:
    paths = sorted((results / "3_Experiment/_closures").glob("*.json"))
    for path in paths:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            schema = value.get("schema_version")
            if schema == "sg_content_chain_closure_v1":
                verify_chain(value)
            elif schema not in CLOSURE_SCHEMAS:
                raise ValueError(f"unrecognized closure schema: {schema}")
        except (OSError, ValueError, KeyError, AttributeError, TypeError) as exc:
            raise StageOrderError(f"unrecognized or invalid closure: {path}") from exc
    return bool(paths)


def country_context(repo_root, country, *, profile="formal", results_root=None, backend="local",
                    upstream=None, regions=None, limit=None) -> StageContext:
    """``upstream`` maps 'data' and 'generator' to the handoff loaders
    supplied by the orchestration entrypoint (see casestudy/3_Experiment/<dir>)."""

    if backend not in {"local", "hpc"}:
        raise ValueError(f"unknown backend: {backend}")
    from sglib.core.infra.setup_snapshot import config_root as setup_config_root

    repo = Path(repo_root).resolve()
    loaded = load_experiment_config(repo, country)
    results = resolve_results_root(repo, profile, results_root)
    root = Path(str(loaded.values["paths"][f"{profile}_root"]).format(
        results_root=results.as_posix(), directory=loaded.country_profile.directory)).resolve()
    if root != results / "3_Experiment" / loaded.country_profile.directory:
        raise ValueError("country output root must be inside the selected results root")
    read_only = _closed_results(results)
    if read_only and setup_config_root(root) is not None:
        # A closed root is judged by the registrations it recorded, not by the evolving checkout.
        loaded = load_experiment_config(repo, country, config_root=setup_config_root(root))
    candidates = json.loads(loaded.sources["candidate_registry"].read_text(encoding="utf-8"))
    units = build_registry(country, loaded, candidates)
    if regions is not None:
        unknown = set(regions) - set(loaded.values["regions"])
        if unknown:
            raise ValueError(f"{country}: unknown regions {sorted(unknown)}")
    return StageContext(repo, results, root, country, profile, backend, loaded, candidates, units,
                        read_only, dict(upstream or {}),
                        None if regions is None else tuple(regions), limit)


def setup_config_files(ctx: StageContext) -> list[str]:
    """Repository-relative configuration files an Experiment country actually reads."""

    files = {Path(path).resolve() for path in ctx.loaded.sources.values() if Path(path).is_file()}
    files.add(Path(ctx.loaded.country_profile.source_path).resolve())
    base = Path(ctx.repo).resolve()
    return sorted(path.relative_to(base).as_posix() for path in files if path.is_relative_to(base))


def code_checks(ctx: StageContext) -> list[dict]:
    """Compare every content-chain receipt of the root with the current Experiment code projections."""

    from sglib.core.infra.setup_snapshot import receipt_code_checks
    from . import planning_identity

    planning = planning_identity.numerical_code()
    numerical = numerical_code()
    return receipt_code_checks(ctx.root, lambda node_id, receipt: planning if ".Planning." in node_id else numerical)


select_units = partial(chain_stage.select_units, label='Experiment', sort=True)


def skip_reason(unit: ExperimentUnit, root: Path, loaded: LoadedExperimentConfig) -> str:
    marker = unit_output_path(unit, root)
    if unit.step == "planning":
        done, expected = planning_progress(unit, root, loaded)
        return f"all {done}/{expected} solver coordinates hold verified receipts under {marker}"
    return f"verified receipt of the registered node: {marker}"


def require_done(repo_root, country, members, **kwargs) -> list[str]:
    ctx = country_context(repo_root, country, **kwargs)
    selected = select_units(ctx.units, members)
    missing = [f"{key} ({unit_status(ctx.units[key], ctx.root, ctx.loaded)})" for key in selected
               if unit_status(ctx.units[key], ctx.root, ctx.loaded) != "DONE"]
    if missing:
        raise StageOrderError("required units are not DONE: " + ", ".join(missing))
    return selected


def figures_root(ctx: StageContext) -> Path:
    """Views mirror the results root under results/_views (formal: results/_views/3_Experiment/<dir>)."""

    base = ctx.repo / "results"
    try:
        relative = ctx.results.relative_to(base)
    except ValueError:
        relative = Path(ctx.results.name)
    path = base / "_views" / relative / "3_Experiment" / ctx.loaded.country_profile.directory
    path.mkdir(parents=True, exist_ok=True)
    return path


def immutable(document: dict, path: Path) -> Path:
    """Write once; an existing file must hold exactly the same document."""

    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != document:
            raise ValueError(f"refusing to rewrite an existing registration or receipt: {path}")
        return path
    return atomic_json(document, path)


def numerical_code() -> str:
    """Code projection of every numerical function of the Experiment package."""

    from . import (allocator_observations, boundary_diagnostics, conditional_bounds, connection,
                   connection_observations, control_fields, correction, defenses, planning_pool,
                   planning_tasks, reconstruction, sweep_observations)

    symbols = {}
    for module in (allocator_observations, boundary_diagnostics, conditional_bounds, connection,
                   connection_observations, control_fields, correction, defenses, planning_pool,
                   planning_tasks, reconstruction, sweep_observations):
        for name, value in inspect.getmembers(module, inspect.isfunction):
            if value.__module__ == module.__name__:
                symbols[f"{module.__name__}.{name}"] = value
    return code_projection(symbols)["code_sha256"]


def _selected_region(ctx: StageContext, unit: ExperimentUnit) -> bool:
    return ctx.regions is None or unit.step not in {"observe", "planning"} or unit.member in ctx.regions


def _execute(ctx: StageContext, unit: ExperimentUnit) -> bool:
    from . import production

    if unit.step == "preflight":
        production.run_preflight(ctx, unit)
    elif unit.step == "observe":
        if ctx.backend == "hpc":
            production.prepare_unit(ctx, unit)
            return False
        production.run_observe(ctx, unit)
    elif unit.step == "planning":
        if ctx.backend == "hpc":
            production.prepare_unit(ctx, unit)
            return False
        production.run_planning(ctx, unit)
    elif unit.step == "bounds":
        production.run_bounds(ctx, unit)
    elif unit.step == "defense":
        production.run_defense(ctx, unit)
    elif unit.step == "audit":
        production.run_audit(ctx, unit)
    else:
        raise StageOrderError(f"unknown Experiment step {unit.step}")
    return True


def run_units(ctx, selected, *, expand=False, refresh=False):
    selected = tuple(selected)
    return chain_stage.run_units(ctx, selected, _hooks(ctx),
        ordered=topological_order(ctx.units, set(selected)), expand=expand, refresh=refresh, snapshot_status=True)


def run_step(repo_root, country, members, *, profile="formal", results_root=None, backend="local",
             refresh=False, upstream=None, regions=None, limit=None) -> StepReport:
    ctx = country_context(repo_root, country, profile=profile, results_root=results_root, backend=backend,
                          upstream=upstream, regions=regions, limit=limit)
    return run_units(ctx, select_units(ctx.units, members), refresh=refresh)


def status_table(ctx: StageContext) -> list[dict[str, Any]]:
    rows = []
    for unit in topological_order(ctx.units):
        state = unit_status(unit, ctx.root, ctx.loaded)
        row = {"unit": unit.id, "step": unit.step, "member": unit.member, "state": state}
        if unit.step == "planning":
            done, expected = planning_progress(unit, ctx.root, ctx.loaded)
            row["coordinates"] = f"{done}/{expected}"
        rows.append(row)
    return rows


def _partial(ctx, unit):
    if unit.step == "planning" and ctx.limit is not None and unit_status(unit, ctx.root, ctx.loaded) == "PENDING":
        done, expected = planning_progress(unit, ctx.root, ctx.loaded)
        print(f"PARTIAL {unit.id}: {done}/{expected} solver coordinates (limit={ctx.limit})", flush=True)
        return True
    return False


def _read_only_audit(ctx, unit):
    if unit.step != "audit" or unit_status(unit, ctx.root, ctx.loaded) != "PENDING":
        return False
    from . import production
    document = production.coverage(ctx)
    atomic_json(document, figures_root(ctx) / "audit.json")
    print(f"GENERATED {unit.id}: {document['status']} audit written to {figures_root(ctx) / 'audit.json'}; "
          "place it into the closed root after confirmation", flush=True)
    return True


def _hooks(ctx):
    return chain_stage.RunHooks(
        status=lambda unit: unit_status(unit, ctx.root, ctx.loaded),
        execute=lambda unit: _execute(ctx, unit),
        skip_reason=lambda unit: skip_reason(unit, ctx.root, ctx.loaded),
        selected=lambda unit: _selected_region(ctx, unit),
        partial=lambda unit: _partial(ctx, unit),
        read_only_audit=lambda unit: _read_only_audit(ctx, unit))
