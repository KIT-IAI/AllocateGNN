"""Receipted mechanical rendering and a two-table, non-narrative audit index."""
import importlib
import inspect
import json
import re

import pandas as pd

from sglib.core.infra.artifacts import atomic_json, atomic_text
from sglib.core.infra.content_chain import code_projection, derive_chain_commitment, derive_chain_receipt, verify_chain
from sglib.core.infra.hashing import sha256_file
from . import tables
from .registry import AUDIT_SCHEMA, expected_node_id, unit_output_path, unit_status


def renderer(member):
    name = 'synthesis' if member == 'SYN' else member.lower()
    return importlib.import_module(f'sglib.report.figures.{name}')


def numerical_code(member):
    from .figures import common
    modules = (renderer(member), common, tables)
    symbols = {f'{module.__name__}.{name}': fn for module in modules
               for name, fn in inspect.getmembers(module, inspect.isfunction) if fn.__module__ == module.__name__}
    return code_projection(symbols)['code_sha256']


def publish(ctx, unit, inputs, producer):
    if ctx.read_only:
        raise ValueError('read-only Report root refuses rendering')
    target = unit_output_path(unit, ctx.root).parent
    if target.exists() and any(target.iterdir()):
        raise ValueError(f'will not overwrite existing Report outputs: {target}')
    commitment = derive_chain_commitment(expected_node_id(unit), inputs=inputs,
        scientific_parameters=ctx.loaded.registrations[unit.member], code_sha256=numerical_code(unit.member))
    producer(target)
    outputs = {path.relative_to(target).as_posix(): sha256_file(path)
               for path in sorted(target.rglob('*')) if path.is_file()}
    receipt = derive_chain_receipt(commitment, outputs=outputs, observations={'shared': ctx.loaded.spec})
    atomic_json(receipt, target / 'receipt.json')


def execute(ctx, unit):
    if ctx.read_only:
        raise ValueError('read-only Report root refuses production')
    if unit.step == 'audit':
        write_audit(ctx, ctx.root)
        return True
    if 'tables' not in ctx.upstream:
        raise ValueError("entrypoint must inject upstream['tables']")
    requested = tables.requested(unit.member, ctx.loaded.spec['countries'])
    payload = ctx.upstream['tables'](ctx.repo, requested)
    if set(payload['inputs']) != set(requested) or set(payload['tables']) != set(requested):
        raise ValueError('Analysis input keys differ from the actual table request')
    registration = ctx.loaded.registrations[unit.member]
    spec = {**registration['spec'], 'shared': ctx.loaded.spec, 'figures': registration['figures']}

    def produce(target):
        renderer(unit.member).render(payload['tables'], spec, target)
        tables.assemble(unit.member, payload['tables'], {**registration, 'shared': ctx.loaded.spec}, target)

    publish(ctx, unit, payload['inputs'], produce)
    return True


def expand_identifiers(value, default_prefix=None):
    result = []
    for token in str(value).split(','):
        match = re.fullmatch(r'([A-Za-z][A-Za-z0-9_-]*?)(\d+)(?:\.\.(\d+))?', token.strip())
        if match is None:
            raise ValueError(f'invalid registered ID range: {value}')
        prefix, start, stop = match.groups()
        if prefix == 'E':
            prefix = default_prefix
        if prefix is None:
            raise ValueError(f'missing registered ID prefix: {value}')
        result.extend(f'{prefix}{number:0{len(start)}d}' for number in range(int(start), int(stop or start)+1))
    return result


def experiment_index(ctx):
    path = ctx.root / 'SYN/tables/T-SYN-04.csv'
    if not path.is_file():
        return []
    rows = []
    for row in pd.read_csv(path).to_dict('records'):
        identifiers = [*expand_identifiers(row['figure_ids']), *expand_identifiers(row['table_ids'])]
        experiments = expand_identifiers(row['experiments'], f"{row['claim_id']}-E")
        for experiment in experiments:
            rows.append({'claim': row['claim_id'], 'experiment': experiment, 'stable_ids': identifiers})
    return rows


def coverage(ctx):
    states, receipts = {}, {}
    for key, unit in ctx.units.items():
        if unit.step != 'render':
            continue
        states[key] = unit_status(unit, ctx.root, ctx.loaded)
        if states[key] == 'DONE':
            receipt = verify_chain(json.loads(unit_output_path(unit, ctx.root).read_text(encoding='utf-8')))
            receipts[key] = receipt['receipt_sha256']
    ids = []
    for identifier, record in ctx.loaded.ids.items():
        member, kind = record['member'], record['kind']
        paths = ([f'{member}/figures/{identifier}.png', f'{member}/figures/{identifier}.pdf',
                  f'{member}/sources/{identifier}.csv'] if kind == 'figure' else [f'{member}/tables/{identifier}.csv'])
        key = f'cross.render.{member}'
        ids.append({'stable_id': identifier, 'kind': kind, 'paths': paths, 'unit': key,
                    'receipt_sha256': receipts.get(key), 'exists': all((ctx.root / path).is_file() for path in paths)})
    experiments = experiment_index(ctx)
    valid_coverage = bool(experiments) and all(row['stable_ids'] and set(row['stable_ids']).issubset(ctx.loaded.ids)
                                             for row in experiments)
    passed = all(value == 'DONE' for value in states.values()) and all(row['exists'] for row in ids) and valid_coverage
    return {'schema_version': AUDIT_SCHEMA, 'status': 'PASS' if passed else 'FAIL',
            'units': states, 'stable_ids': ids, 'receipt_sha256': receipts, 'experiments': experiments}


def markdown_table(columns, rows):
    def escaped(value):
        return str(value).replace('|', '\\|').replace('\n', ' ')
    return '\n'.join(['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join('---' for _ in columns) + ' |',
                      *['| ' + ' | '.join(escaped(value) for value in row) + ' |' for row in rows]])


def index_text(document):
    artifacts = [[row['stable_id'], row['kind'], ', '.join(row['paths']), row['unit'], row['receipt_sha256']]
                 for row in document['stable_ids']]
    experiments = [[row['claim'], row['experiment'], ', '.join(row['stable_ids'])] for row in document['experiments']]
    return (markdown_table(['stable_id', 'kind', 'paths', 'unit', 'receipt_sha256'], artifacts) + '\n\n'
            + markdown_table(['claim', 'experiment', 'stable_ids'], experiments) + '\n')


def write_setup(ctx):
    """Record the run setup of a writable Report root once; a closed root keeps what it carries."""
    from sglib.core.infra.setup_snapshot import dirty_allowed, verify_setup, write_setup as snapshot
    from .stage import code_checks, setup_config_files

    existing = verify_setup(ctx.root)
    if existing['status'] != 'ABSENT' or ctx.read_only:
        print(f"setup {existing['status']}: {ctx.root / 'setup'}", flush=True)
        return existing
    snapshot(ctx.root, repo_root=ctx.repo, stage='5_Report', country=ctx.country, profile=ctx.profile,
             config_files=setup_config_files(ctx), code_checks=code_checks(ctx),
             allow_dirty=dirty_allowed(ctx.repo, ctx.results, ctx.profile))
    report = verify_setup(ctx.root)
    print(f"setup WRITTEN ({report['status']}): {ctx.root / 'setup'}", flush=True)
    return report


def write_audit(ctx, target):
    if ctx.read_only:
        from .stage import figures_root
        if target.resolve() != figures_root(ctx).resolve():
            raise ValueError('read-only Report audit must go into views')
    document = coverage(ctx)
    if document['status'] != 'PASS':
        raise ValueError('incomplete Report audit')
    atomic_json(document, target / 'audit.json')
    atomic_text(target / 'index.md', index_text(document))
    return document
