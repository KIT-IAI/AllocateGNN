"""Rebuild the CIVD downstream dependency closure in a fresh inherited result root.

The caller copies unaffected products and omits ``regenerated_paths()`` before
updating Experiment observations. This module never removes or replaces files.
"""
from functools import partial
import json
from pathlib import Path

import numpy as np
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


CHANGED_COUNTRIES = COUNTRIES
REPORT_MEMBERS = ('C2', 'C3', 'C4', 'C5', 'C6', 'SYN')
CORRECTION_SCOPE = 'four_country_posthoc_bugfix_comparison'


def regenerated_paths():
    """Repository-result-relative paths the caller must omit when inheriting."""
    paths = [f'4_Analysis/{DIRECTORIES[cc]}/{name}'
             for cc in CHANGED_COUNTRIES for name in ('C3', 'support', 'audit.json', 'manifest.json')]
    paths += [f'4_Analysis/9_CrossCountry/{name}' for name in ('synthesis', 'audit.json', 'manifest.json')]
    paths += [f'5_Report/{name}' for name in (*REPORT_MEMBERS, 'audit.json', 'index.md')]
    return tuple(paths)


def fresh_root(repo, results):
    repo, results = Path(repo).resolve(), Path(results).resolve()
    formal = (repo / 'results').resolve()
    containers = tuple((formal / name).resolve() for name in ('_staging', '_releases'))
    if results == formal or not any(results != base and results.is_relative_to(base) for base in containers):
        raise ValueError('CIVD downstream rerun requires a dedicated results/_staging or _releases child')
    if not results.is_dir():
        raise ValueError('inherit Experiment/Analysis/Report before running downstream')
    for relative in regenerated_paths():
        target = (results / relative).resolve()
        if not target.is_relative_to(results) or target == results or target == formal:
            raise ValueError(f'unsafe downstream output path: {relative}')
        if target.exists():
            raise ValueError(f'omit regenerated output when inheriting: {relative}')
    for stage in ('4_Analysis', '5_Report'):
        target = (results / stage).resolve()
        if not target.is_relative_to(results):
            raise ValueError(f'downstream stage escapes rerun root: {stage}')
        if list((target / '_closures').glob('*.json')):
            raise ValueError('create rerun closures only after downstream production')
    return repo, results


def _receipt(root):
    document = verify_chain(json.loads((root / 'receipt.json').read_text(encoding='utf-8')))
    for relative, digest in document['outputs'].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f'downstream output differs from receipt: {path}')
    return document


def _target_context(repo, results, prior_synthesis, current_synthesis):
    """Read the exact inherited context bytes bound by both synthesis receipts."""
    sources = {key: digest for key, digest in prior_synthesis["commitment"]["inputs"].items()
               if key.endswith(":context/regions.csv")}
    current = {key: digest for key, digest in current_synthesis["commitment"]["inputs"].items()
               if key.endswith(":context/regions.csv")}
    if not sources or current != sources:
        raise ValueError("target reconciliation changed its inherited context inputs")
    rows = []
    for key, digest in sources.items():
        country, region, relative = key.split(":", 2)
        if country not in DIRECTORIES or relative != "context/regions.csv":
            raise ValueError(f"invalid target context coordinate: {key}")
        pair = []
        for base in (repo / "results", results):
            root = base / "3_Experiment" / DIRECTORIES[country] / "observations" / region
            path = (root / relative).resolve()
            receipt = verify_chain(json.loads((root / "receipt.json").read_text(encoding="utf-8")))
            if (not path.is_relative_to(root.resolve()) or receipt["outputs"].get(relative) != digest
                    or sha256_file(path) != digest):
                raise ValueError(f"target context differs from its observation/synthesis receipts: {key}")
            pair.append(analysis_production.frame(path))
        pd.testing.assert_frame_equal(pair[0], pair[1], check_dtype=False, check_exact=True)
        table = pair[1]
        if len(table) != 1 or not table.country.eq(country).all() or not table.region.eq(region).all():
            raise ValueError(f"target context country/region identity differs: {key}")
        rows.append(table)
    return pd.concat(rows, ignore_index=True)


def _target_schema_transition(left, right, context):
    """Validate the recorded-count to live-context schema change without dropping evidence."""
    common = ["country", "region", "n_source", "n_target", "country_target_count_from_regions"]
    previous = common + ["registered_target_count", "target_count_matches_current_frozen_specification"]
    added = ["granularity_ratio", "source_total", "target_total", "region_nonempty",
             "granularity_ratio_consistent", "mass_reconciled"]
    if set(left.columns) != set(previous) or set(right.columns) != set(common + added):
        raise ValueError("unrecognized target_count_audit schema transition")
    pd.testing.assert_frame_equal(left[common], right[common], check_dtype=False, check_exact=True)
    keys = ["country", "region"]
    if left.duplicated(keys).any() or right.duplicated(keys).any() or context.duplicated(keys).any():
        raise ValueError("duplicate target reconciliation coordinate")
    prior_total = left.groupby("country")["n_target"].transform("sum")
    if (not left.registered_target_count.eq(prior_total).all()
            or not left.country_target_count_from_regions.eq(prior_total).all()
            or not left.target_count_matches_current_frozen_specification.eq(True).all()):
        raise ValueError("prior registered target counts fail their recorded reconciliation")
    columns = ["n_source", "n_target", "granularity_ratio", "source_total", "target_total"]
    expected = right[keys].merge(context[keys + columns], on=keys, how="outer", validate="one_to_one", indicator=True)
    if not expected._merge.eq("both").all():
        raise ValueError("target reconciliation context coverage differs")
    # Restore the recorded row order explicitly; values themselves remain exact.
    expected = right[keys].merge(context[keys + columns], on=keys, how="left", validate="one_to_one", sort=False)
    pd.testing.assert_frame_equal(right[keys + columns], expected, check_dtype=False, check_exact=True)
    source, target = expected.n_source.to_numpy(float), expected.n_target.to_numpy(float)
    source_total, target_total = expected.source_total.to_numpy(float), expected.target_total.to_numpy(float)
    checks = {
        "region_nonempty": (source > 0) & (target > 0),
        "granularity_ratio_consistent": np.isclose(expected.granularity_ratio.to_numpy(float), target / source, rtol=1e-12, atol=0.0),
        "mass_reconciled": np.isfinite(source_total) & (source_total > 0) & np.isclose(source_total, target_total, rtol=1e-9, atol=1e-9),
    }
    for name, expected_values in checks.items():
        if not expected_values.all() or not np.array_equal(right[name].to_numpy(), expected_values):
            raise ValueError(f"target reconciliation failed independent {name} check")
    total = right.groupby("country")["n_target"].transform("sum")
    if not right.country_target_count_from_regions.eq(total).all():
        raise ValueError("target reconciliation country totals differ")
    return {"comparison": "exact_inherited_counts_and_receipted_context_reconciliation",
            "inherited_columns": common, "independently_validated_columns": added,
            "retired_columns": ["registered_target_count", "target_count_matches_current_frozen_specification"]}


def _same_table(old, new, repo, *, exact=False, exclude_c3=False, target_context=None):
    left, right = analysis_production.frame(old), analysis_production.frame(new)
    if exclude_c3:
        left = left[left.claim_id.ne('C3')].reset_index(drop=True)
        right = right[right.claim_id.ne('C3')].reset_index(drop=True)
    if target_context is not None and set(left.columns) != set(right.columns):
        transition = _target_schema_transition(left, right, target_context)
        return {"path": portable_path(new, repo), "rows": len(right), **transition}
    pd.testing.assert_frame_equal(left, right, check_dtype=False, check_exact=exact,
                                  rtol=1e-12, atol=1e-12)
    return {'path': portable_path(new, repo), 'rows': len(right), 'comparison': 'exact' if exact else 'rtol_atol_1e-12'}


def scientific_invariants(repo, results):
    """Check preserved claims, including regenerated mixed support/synthesis tables."""
    formal, checks = repo / 'results', []
    for cc in COUNTRIES:
        directory = DIRECTORIES[cc]
        for member in ('C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'support'):
            relative = Path('4_Analysis') / directory / member
            old, new = formal / relative, results / relative
            baseline, current = _receipt(old), _receipt(new)
            if set(baseline['outputs']) != set(current['outputs']):
                raise ValueError(f'Analysis output inventory changed: {relative}')
            if cc not in CHANGED_COUNTRIES or member not in ('C3', 'support'):
                if baseline != current:
                    raise ValueError(f'unaffected Analysis receipt changed: {relative}')
                continue
            if member == 'C3':
                regional = analysis_production.frame(new / 'region_metrics.csv')
                expected = analysis_production.frame(old / 'region_metrics.csv')
                keys = ['region', 'candidate', 'metric']
                coordinates = set(expected[expected.allocator.eq('VD')][keys].itertuples(index=False, name=None))
                civd = regional[regional.allocator.eq('CIVD')]
                if set(civd[keys].itertuples(index=False, name=None)) != coordinates or civd.duplicated(keys).any():
                    raise ValueError(f'CIVD regional coordinates do not match the inherited VD comparison: {cc}')
                if set(regional.allocator) != {'VD', 'IDR-fixed', 'IDR-matched', 'CIVD'}:
                    raise ValueError(f'four allocator comparison is incomplete: {cc}')
            if member == 'support':
                summary = analysis_production.frame(new / 'C3_CIVD_summary.csv')
                if summary.empty or set(summary.evidence) != {CORRECTION_SCOPE}:
                    raise ValueError(f'CIVD support is missing its effective four-country scope: {cc}')
            for name in baseline['outputs']:
                if member == 'C3' and name in ('region_metrics.csv', 'gates.csv', 'maps.csv'):
                    continue
                if member == 'support' and name.startswith('C3_'):
                    continue
                checks.append(_same_table(old / name, new / name, repo, exact=member == 'C3',
                                          exclude_c3=name == 'claim_coordinate_audit.csv'))
    relative = Path('4_Analysis/9_CrossCountry/synthesis')
    baseline, current = _receipt(formal / relative), _receipt(results / relative)
    if set(baseline['outputs']) != set(current['outputs']):
        raise ValueError('synthesis output inventory changed')
    target_context = _target_context(repo, results, baseline, current)
    for name in baseline['outputs']:
        checks.append(_same_table(formal / relative / name, results / relative / name, repo,
                                  exclude_c3=name == 'claim_coordinate_audit.csv',
                                  target_context=target_context if name == 'target_count_audit.csv' else None))
    for member in ('C1', *REPORT_MEMBERS):
        relative = Path('5_Report') / member
        baseline, current = _receipt(formal / relative), _receipt(results / relative)
        if set(baseline['outputs']) != set(current['outputs']):
            raise ValueError(f'Report output inventory changed: {member}')
        if member == 'C1' and baseline != current:
            raise ValueError('inherited C1 Report receipt changed')
        for name in baseline['outputs']:
            if not name.endswith('.csv') or member == 'C3':
                continue
            checks.append(_same_table(formal / relative / name, results / relative / name, repo,
                                      exclude_c3=name == 'claim_coordinate_audit.csv',
                                      target_context=target_context if member == 'SYN' and name == 'target_count_audit.csv' else None))
    return checks


def verify_links(repo, results, *, report_context, report_requests):
    links = []
    for cc in COUNTRIES:
        root = results / '4_Analysis' / DIRECTORIES[cc]
        support = _receipt(root / 'support')
        for claim in ('C1', 'C2', 'C3', 'C4', 'C5', 'C6'):
            parent = _receipt(root / claim)
            if support['commitment']['inputs'][f'analysis:{claim}'] != parent['receipt_sha256']:
                raise ValueError(f'stale support parent: {cc}:{claim}')
            links.append(f'{cc}:support <- {claim}')
    synthesis = _receipt(results / '4_Analysis/9_CrossCountry/synthesis')
    for cc in COUNTRIES:
        for member in ('C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'support'):
            parent = _receipt(results / '4_Analysis' / DIRECTORIES[cc] / member)
            if synthesis['commitment']['inputs'][f'{cc}:{member}'] != parent['receipt_sha256']:
                raise ValueError(f'stale synthesis parent: {cc}:{member}')
            links.append(f'cross:synthesis <- {cc}:{member}')
    ctx = report_context
    for member in ctx.loaded.registrations:
        requested = report_requests(member, COUNTRIES)
        actual = _receipt(ctx.root / member)['commitment']['inputs']
        expected = analysis_inputs.load(repo, requested, results_root=results)['inputs']
        if actual != expected:
            raise ValueError(f'stale Report input receipts: {member}')
        links.extend(f'Report:{member} <- {key}' for key in actual)
    return links
