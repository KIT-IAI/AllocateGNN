"""Rebuild the CIVD downstream dependency closure in a fresh inherited result root.

The caller copies unaffected products and omits ``regenerated_paths()`` before
updating Experiment observations. This module never removes or replaces files.
"""
from functools import partial
import json
from pathlib import Path

import pandas as pd

from sglib.analysis import downstream as analysis_inputs
from sglib.analysis import manifest as analysis_manifest
from sglib.analysis import production as analysis_production
from sglib.analysis import stage as analysis_stage
from sglib.analysis.config import COUNTRIES, DIRECTORIES
from sglib.core.infra.artifacts import atomic_json
from sglib.core.infra.paths import portable_path
from sglib.core.infra.content_chain import verify_chain
from sglib.core.infra.hashing import sha256_file
from sglib.experiment import downstream as experiment_inputs
from sglib.report import production as report_production
from sglib.report import stage as report_stage
from sglib.report import tables as report_tables


from sglib.analysis.civd_downstream import (
    CHANGED_COUNTRIES, REPORT_MEMBERS, CORRECTION_SCOPE, fresh_root,
    regenerated_paths, scientific_invariants, verify_links,
)


def check_links(repo, results):
    ctx = report_stage.country_context(repo, profile="smoke", results_root=results)
    return verify_links(repo, results, report_context=ctx, report_requests=report_tables.requested)

def run_downstream(repo: Path, results: Path, *, verify_only: bool = False):
    """Recompute affected units and return/write their audits and invariant checks.

    Every input root is supplied explicitly. The formal root is comparison-only.
    ``results`` must be a fresh child of ``repo/results/_staging`` or ``_releases`` with the paths
    returned by ``regenerated_paths()`` omitted from its inherited products.
    ``verify_only`` checks completed outputs and writes their validation record
    after an interrupted validation; it never reruns producers.
    """
    if verify_only:
        repo, results = Path(repo).resolve(), Path(results).resolve()
        containers = [repo / 'results' / name for name in ('_staging', '_releases')]
        if not any(results != base and results.is_relative_to(base) for base in containers):
            raise ValueError('downstream verification requires a dedicated candidate root')
        if list(results.glob('*/_closures/*')):
            raise ValueError('sealed results refuse downstream validation writes')
    else:
        repo, results = fresh_root(repo, results)
    upstream = {'status': partial(experiment_inputs.country_status, results_root=results),
                'tables': partial(experiment_inputs.load_country_tables, results_root=results)}
    production = []
    if not verify_only:
        for cc in CHANGED_COUNTRIES:
            ctx = analysis_stage.country_context(repo, cc, profile='smoke', results_root=results, upstream=upstream)
            for member in ('C3', 'support'):
                ctx.loaded.registrations[member]['correction_scope'] = CORRECTION_SCOPE
                ctx.loaded.registrations[member]['scope_override'] = {
                    'inherited_protocol': 'UK_AU_historical_defense_only',
                    'effective_CIVD_scope': CORRECTION_SCOPE,
                    'countries': list(CHANGED_COUNTRIES),
                    'comparison_status': 'posthoc_not_preregistered',
                    'original_C3_contrasts': 'unchanged_preregistered_five_contrasts',
                }
            result = analysis_stage.run_units(ctx, [f'{cc}.core.C3', f'{cc}.support.support', f'{cc}.audit.audit'])
            production.extend(result.ran)
            ctx.cache.clear()
        ctx = analysis_stage.country_context(repo, 'cross', profile='smoke', results_root=results, upstream=upstream)
        ctx.loaded.registrations['synthesis']['correction_scope'] = CORRECTION_SCOPE
        result = analysis_stage.run_units(ctx, ['cross.synthesis.synthesis', 'cross.audit.audit'])
        production.extend(result.ran)
        ctx.cache.clear()
    gates = {}
    for cc in (*COUNTRIES, 'cross'):
        ctx = analysis_stage.country_context(repo, cc, profile='smoke', results_root=results, upstream=upstream)
        gate = analysis_manifest.run_gate(ctx)
        if gate['status'] == 'GENERATED':
            gate = analysis_manifest.run_gate(ctx)
        if gate['status'] != 'PASS' or gate['unclaimed']:
            raise ValueError(f'Analysis leaf gate failed: {cc}')
        gates[cc] = gate
    report_upstream = {'status': partial(analysis_inputs.status, results_root=results),
                       'tables': partial(analysis_inputs.load, results_root=results)}
    ctx = report_stage.country_context(repo, profile='smoke', results_root=results, upstream=report_upstream)
    if not verify_only:
        result = report_stage.run_units(ctx, [*(f'cross.render.{name}' for name in REPORT_MEMBERS), 'cross.audit.audit'])
        production.extend(result.ran)
    audit = report_production.coverage(ctx)
    if audit['status'] != 'PASS' or len(audit['stable_ids']) != 37 or len(audit['experiments']) != 30:
        raise ValueError('Report audit does not cover 37 IDs and 30 experiments')
    if json.loads((ctx.root / 'audit.json').read_text(encoding='utf-8')) != audit:
        raise ValueError('Report audit does not describe the regenerated receipts')
    checks = scientific_invariants(repo, results)
    links = check_links(repo, results)
    previous_validation = results / 'civd_downstream_validation.json'
    if verify_only and previous_validation.is_file():
        production = json.loads(previous_validation.read_text(encoding='utf-8')).get('production', [])
    document = {'status': 'PASS', 'results_root': portable_path(results, repo), 'production': production,
                'validation_only': verify_only, 'analysis_gates': gates, 'report_audit': audit, 'scientific_invariants': checks,
                'verified_parent_links': links, 'C3_main_contrasts': 'exactly_equal_to_formal'}
    atomic_json(document, results / 'civd_downstream_validation.json')
    return document
