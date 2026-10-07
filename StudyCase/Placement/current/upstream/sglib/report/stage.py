"""Numbered mechanical rendering; formal roots are read-only."""

from functools import partial
from sglib.core.chain import stage as chain_stage
from sglib.core.chain.stage import StageOrderError, StepReport, resolve_results_root
from dataclasses import dataclass, field
import json
from pathlib import Path

from sglib.core.infra.content_chain import verify_chain

from .config import load_report_config
from .registry import build_registry, topological_order, unit_status




@dataclass(frozen=True)
class StageContext:
    repo: Path
    results: Path
    root: Path
    country: str
    profile: str
    loaded: object
    units: dict
    read_only: bool
    upstream: dict
    cache: dict = field(default_factory=dict, compare=False)




def country_context(repo_root, country='cross', *, profile='formal', results_root=None, upstream=None):
    if country != 'cross':
        raise ValueError('Report only has cross-country units')
    if profile not in ('formal', 'smoke', 'preflight'):
        raise ValueError(f'unknown Report profile: {profile}')
    from sglib.core.infra.setup_snapshot import config_root as setup_config_root

    repo = Path(repo_root).resolve()
    results = resolve_results_root(repo, profile, results_root or None)
    loaded = load_report_config(repo)
    root = results / '5_Report'
    closures = sorted((root / '_closures').glob('*.json'))
    for path in closures:
        verify_chain(json.loads(path.read_text(encoding='utf-8')))
    read_only = profile == 'formal' or results == repo / 'results' or bool(closures)
    if read_only and setup_config_root(root) is not None:
        # A closed root is judged by the registrations it recorded, not by the evolving checkout.
        loaded = load_report_config(repo, config_root=setup_config_root(root))
    return StageContext(repo, results, root, country, profile, loaded,
                        build_registry(loaded), read_only, dict(upstream or {}))


def setup_config_files(ctx):
    """Repository-relative configuration files the Report stage actually reads."""
    from .config import REPORT_CONFIG
    return [REPORT_CONFIG]


def code_checks(ctx):
    """Compare every content-chain receipt of the root with the current Report code projections."""
    from sglib.core.infra.setup_snapshot import receipt_code_checks
    from .production import numerical_code

    def current(node_id, receipt):
        parts = node_id.split('.')
        member = parts[2] if len(parts) > 2 else ''
        return numerical_code(member) if member in ctx.loaded.registrations else None

    return receipt_code_checks(ctx.root, current)


def figures_root(ctx):
    try:
        relative = ctx.results.relative_to(ctx.repo / 'results')
    except ValueError:
        relative = Path(ctx.results.name)
    path = ctx.repo / 'results/_views' / relative / '5_Report'
    path.mkdir(parents=True, exist_ok=True)
    return path


select_units = partial(chain_stage.select_units, label='Report', sort=False)


def dependency_states(ctx, dependencies):
    external = [key for key in dependencies if key not in ctx.units]
    states = {}
    if external:
        if 'status' not in ctx.upstream:
            raise StageOrderError("entrypoint must inject upstream['status'] for Analysis dependencies")
        injected = ctx.upstream['status'](ctx.repo)
        states.update({key: injected.get(key, 'PENDING') for key in external})
    states.update({key: unit_status(ctx.units[key], ctx.root, ctx.loaded) for key in dependencies if key in ctx.units})
    return states


def _execute(ctx, unit):
    from .production import execute
    return execute(ctx, unit)


def run_units(ctx, selected):
    selected = tuple(selected)
    return chain_stage.run_units(ctx, selected, _hooks(ctx), ordered=topological_order(ctx.units, selected),
        precheck=False, dependency_first=True, sort_selected=False)


def run_step(repo_root, country, members, *, profile='formal', results_root=None, upstream=None):
    ctx = country_context(repo_root, country, profile=profile, results_root=results_root, upstream=upstream)
    return run_units(ctx, select_units(ctx.units, members))


def status_table(ctx):
    return [{'unit': u.id, 'step': u.step, 'member': u.member, 'state': unit_status(u, ctx.root, ctx.loaded)}
            for u in topological_order(ctx.units)]


def _status(ctx, unit):
    if isinstance(unit, str):
        return dependency_states(ctx, (unit,))[unit]
    return unit_status(unit, ctx.root, ctx.loaded)


def _read_only_audit(ctx, unit):
    if unit.step != 'audit':
        return False
    from . import production
    production.write_audit(ctx, figures_root(ctx))
    print(f'PASS {unit.id}: audit generated into views', flush=True)
    return True


def _skip_reason(ctx, unit):
    from . import production
    if unit.step == 'audit' and production.coverage(ctx)['status'] != 'PASS':
        raise StageOrderError('incomplete Report audit')
    return 'registered receipt and outputs present'


def _hooks(ctx):
    return chain_stage.RunHooks(
        status=lambda unit: _status(ctx, unit),
        execute=lambda unit: _execute(ctx, unit),
        skip_reason=lambda unit: _skip_reason(ctx, unit),
        read_only_audit=lambda unit: _read_only_audit(ctx, unit))
