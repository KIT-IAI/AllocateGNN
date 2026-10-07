"""Analysis output contracts and coverage audit; statistical producers are unchanged."""
import inspect
import json
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits
from sglib.core.infra.artifacts import atomic_json
from sglib.core.infra.content_chain import code_projection, derive_chain_commitment, derive_chain_receipt, verify_chain
from sglib.core.infra.hashing import sha256_file
from sglib.core.infra.paths import case_study_root
from .registry import AUDIT_SCHEMA, expected_node_id, unit_output_path


def frame(path):
    try:
        return pd.read_csv(path, dtype={'country': str, 'region': str, 'target_id': str,
                                       'source_id': str, 'station_id': str},
                           float_precision='round_trip', low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def load_tables(root):
    receipt = verify_chain(json.loads((root / 'receipt.json').read_text(encoding='utf-8')))
    tables = {}
    for name, digest in receipt['outputs'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f'Analysis output differs from receipt: {path}')
        if path.suffix == '.csv':
            tables[path.stem] = frame(path)
    return tables, receipt


def experiment_tables(ctx, country=None):
    country = country or ctx.country
    if 'tables' not in ctx.upstream:
        raise ValueError("entrypoint must inject upstream['tables']")
    key = ('experiment', country)
    if key not in ctx.cache:
        ctx.cache[key] = ctx.upstream['tables'](ctx.repo, country)
    return ctx.cache[key]


def numerical_code(kind):
    from . import c1, c2, c3, c4, c5, c6, paired_inference, linear_contrasts, support, synthesis
    modules = {'C1': (c1, paired_inference), 'C2': (c2, paired_inference),
               'C3': (c3, linear_contrasts, paired_inference), 'C4': (c4, linear_contrasts, paired_inference),
               'C5': (c5, linear_contrasts, paired_inference), 'C6': (c6,),
               'support': (support,), 'synthesis': (synthesis,)}
    symbols = {f'{module.__name__}.{name}': fn for module in modules[kind]
               for name, fn in inspect.getmembers(module, inspect.isfunction) if fn.__module__ == module.__name__}
    if kind == 'support':
        symbols['assembly'] = support_tables
    return code_projection(symbols)['code_sha256']


def publish(ctx, unit, inputs, producer):
    target = unit_output_path(unit, ctx.root).parent
    if target.exists() and any(target.iterdir()):
        raise ValueError(f'will not overwrite existing Analysis outputs: {target}')
    parameters = ctx.loaded.registrations[unit.member]
    commitment = derive_chain_commitment(expected_node_id(unit), inputs=inputs,
        scientific_parameters=parameters, code_sha256=numerical_code(unit.member))
    with threadpool_limits(limits=int(parameters.get('compute_threads', 1))):
        tables = producer()
    target.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name, table in tables.items():
        if not isinstance(table, pd.DataFrame):
            raise TypeError(f'Analysis output is not a DataFrame: {name}')
        path = target / f'{name}.csv'
        table.to_csv(path, index=False, float_format='%.17g', lineterminator='\n')
        outputs[path.name] = sha256_file(path)
    receipt = derive_chain_receipt(commitment, outputs=outputs, observations={'backend': 'local_cpu'})
    atomic_json(receipt, target / 'receipt.json')
    ctx.cache[unit.member] = (tables, receipt)


def core_tables(ctx, claim):
    if claim not in ctx.cache:
        ctx.cache[claim] = load_tables(ctx.root / claim)
    return ctx.cache[claim]


def run_core(ctx, unit):
    from . import c1, c2, c3, c4, c5, c6
    kind, spec = unit.member, ctx.loaded.specification
    if kind == 'C2':
        tables, receipt = core_tables(ctx, 'C1')
        return publish(ctx, unit, {'region_metrics': receipt['outputs']['region_metrics.csv']},
                       lambda: c2.analyze(tables['region_metrics'], spec, ctx.loaded.registrations[kind]['contrasts']))
    upstream = experiment_tables(ctx)
    obs = upstream['observations']
    if kind == 'C1':
        wanted = {'reconstruction/' + name + '.csv' for name in ('metrics', 'equal_route_regions', 'equal_route', 'map_grid', 'map_sources')}
        producer = lambda: c1.analyze(obs['reconstruction'], spec)
    elif kind == 'C3':
        wanted = {'allocator/' + name + '.csv' for name in ('metrics', 'gates', 'maps')}
        producer = lambda: c3.analyze(obs['allocator'], spec)
    elif kind == 'C4':
        wanted = {'reconstruction/metrics.csv', 'connection/connection_metrics.csv'}
        producer = lambda: c4.analyze(obs['reconstruction']['metrics'], upstream['planning'], obs['connection']['connection_metrics'], spec)
    elif kind == 'C5':
        wanted = {'connection/panel_metrics.csv'}
        producer = lambda: c5.analyze(obs['connection']['panel_metrics'], spec)
    elif kind == 'C6':
        wanted = {'bounds.csv'}
        producer = lambda: c6.analyze(upstream['bounds']['bounds'], spec)
    else:
        raise ValueError(kind)
    inputs = {key: digest for key, digest in upstream['inputs'].items()
              if key.split(':', 1)[-1] in wanted or (kind == 'C4' and key.startswith('planning:'))}
    publish(ctx, unit, inputs, producer)


def support_tables(country, upstream, core, candidate_order, *, correction_scope=None):
    """Original support assembly, retaining the mixed audit table's column schema."""
    from . import support
    obs = upstream['observations']
    connection = obs['connection']
    t1 = obs.get('T1', {})
    outputs = {}
    outputs.update(support.c2_support(obs['correction']['coordinates'], obs['sweeps']['metrics']))
    c3_options = {'correction_scope': correction_scope} if correction_scope is not None else {}
    outputs.update(support.c3_support(obs['allocator']['gates'], t1.get('region_metrics', pd.DataFrame()),
                                     t1.get('station_metrics', pd.DataFrame()), core['C3']['region_metrics'], **c3_options))
    outputs.update(support.c4_support(obs['reconstruction']['metrics'], upstream['planning'],
                                     connection['connection_metrics'], upstream['defense']['C4_fixed_300']))
    outputs['C4_regional_levels'] = outputs['C4_regional_levels'].sort_values(
        ['task_coordinate', 'region', 'candidate', 'metric'], kind='stable').reset_index(drop=True)
    matching = outputs['C4_matching_summary']
    rank = {candidate: ordinal for ordinal, candidate in enumerate(candidate_order)}
    if not matching.candidate.isin(rank).all():
        raise ValueError('support matching candidate is absent from the candidate registry')
    outputs['C4_matching_summary'] = matching.sort_values(
        ['candidate', 'metric', 'matching', 'unit'],
        key=lambda col: col.map(rank) if col.name == 'candidate' else col,
        kind='stable').reset_index(drop=True)
    outputs.update(support.c5_support(connection['panel_metrics'], core['C5']['panel_associations'], connection['scale_metrics']))
    outputs.update(support.c6_support(core['C6']['bounds_observations'], core['C6']['bounds_audit'], core['C6']['tolerance_curves']))
    outputs['claim_evidence_status'] = support.evidence_status(country, core, outputs)
    resolution = []
    for claim in ('C1', 'C2', 'C3', 'C4', 'C5'):
        table = core[claim]['inference_resolution_audit'].copy()
        table['claim_id'] = claim
        resolution.append(table)
    outputs['inference_resolution_audit'] = pd.concat(resolution, ignore_index=True, sort=False)
    coordinate = []
    for claim, name in [('C1', 'coordinate_audit'), ('C2', 'coordinate_audit'), ('C3', 'region_metrics'),
                        ('C4', 'region_metrics'), ('C5', 'region_differences'), ('C6', 'bounds_observations')]:
        table = core[claim][name].copy()
        table['claim_id'] = claim
        if claim == 'C5':
            table['status'] = 'VALID'
        coordinate.append(table)
    audit = pd.concat(coordinate, ignore_index=True, sort=False)
    # 007 §7.3 permits deleting rows only: preserve the existing empty metadata columns.
    metadata = ['step_m', 'n_cells', 'active_cells', 'branch'] + (
        ['variant_id', 'n'] if country == 'nl' else ['fixed_predictions', 'source_or_assignment_changed'])
    outputs['claim_coordinate_audit'] = audit.reindex(columns=[*audit.columns, *metadata])
    return outputs


def run_support(ctx, unit):
    from .config import CLAIMS
    upstream = experiment_tables(ctx)
    core, inputs = {}, dict(upstream['inputs'])
    for claim in CLAIMS:
        tables, receipt = core_tables(ctx, claim)
        core[claim] = tables
        inputs[f'analysis:{claim}'] = receipt['receipt_sha256']
    registry = json.loads((case_study_root(ctx.repo) / '2_Generator/general/candidate_registry.json').read_text(encoding='utf-8'))
    candidate_order = [candidate['label'] for candidate in registry['candidates']]
    scope = ctx.loaded.registrations[unit.member].get('correction_scope')
    publish(ctx, unit, inputs, lambda: support_tables(ctx.country, upstream, core, candidate_order, correction_scope=scope))


def run_synthesis(ctx, unit):
    from . import synthesis
    from .config import CLAIMS, COUNTRIES, DIRECTORIES
    data, inputs = {}, {}
    for country in COUNTRIES:
        root = ctx.results / '4_Analysis' / DIRECTORIES[country]
        core = {}
        for claim in CLAIMS:
            core[claim], receipt = load_tables(root / claim)
            inputs[f'{country}:{claim}'] = receipt['receipt_sha256']
        support, receipt = load_tables(root / 'support')
        inputs[f'{country}:support'] = receipt['receipt_sha256']
        upstream = experiment_tables(ctx, country)
        data[country] = {'core': core, 'support': support, 'defense': upstream['defense'],
                         'context': upstream['observations']['context']['regions']}
        inputs.update({f'{country}:{key}': digest for key, digest in upstream['inputs'].items()
                       if key.endswith(':context/regions.csv') or key.startswith('defense:')})
        del ctx.cache[('experiment', country)]
    publish(ctx, unit, inputs, lambda: synthesis.assemble(data))


def coverage(ctx):
    from .stage import status_table
    rows = [row for row in status_table(ctx) if row['step'] != 'audit']
    produced = sum(row['state'] == 'DONE' for row in rows)
    return {'schema_version': AUDIT_SCHEMA, 'country': ctx.country,
            'status': 'PASS' if produced == len(rows) else 'INCOMPLETE',
            'expected': len(rows), 'produced': produced, 'units': rows}


def execute(ctx, unit):
    if unit.step == 'core':
        return run_core(ctx, unit)
    if unit.step == 'support':
        return run_support(ctx, unit)
    if unit.step == 'synthesis':
        return run_synthesis(ctx, unit)
    if unit.step == 'audit':
        document = coverage(ctx)
        if document['status'] != 'PASS':
            raise ValueError('incomplete Analysis units')
        atomic_json(document, unit_output_path(unit, ctx.root))
        return
    raise ValueError(f'Analysis producer not yet installed: {unit.id}')
