"""Verified country tables for downstream entrypoint injection (007 §5)."""
from pathlib import Path
import json

import pandas as pd
from sglib.core.infra.content_chain import verify_chain
from sglib.core.infra.hashing import sha256_file
from . import stage
from .registry import coordinate_root, unit_output_path, unit_status


def country_status(repo, country, *, results_root=None):
    ctx = stage.country_context(repo, country, results_root=results_root)
    return {key: unit_status(unit, ctx.root, ctx.loaded) for key, unit in ctx.units.items()}


def _frame(path):
    try:
        return pd.read_csv(path, dtype={'country': str, 'region': str, 'target_id': str,
                                       'source_id': str, 'station_id': str},
                           float_precision='round_trip', low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _tables(root):
    receipt = verify_chain(json.loads((root / 'receipt.json').read_text(encoding='utf-8')))
    tables = {}
    for relative, digest in receipt['outputs'].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f'Experiment output differs from receipt: {path}')
        if path.suffix == '.csv':
            tables[relative] = _frame(path)
    return tables, receipt


def load_country_tables(repo, country, *, results_root=None):
    """Read seven observation families, optional T1, planning metrics, bounds and defense.

    Inputs default to the sealed Experiment root; an explicit results_root
    selects a separately verified rerun. Output profiles never redirect reads.
    Preserve the original
    country/region/field order; legacy planning directory ordinals sort as text
    in the original worker's JSON registry. No statistical operation occurs here.
    """
    ctx = stage.country_context(repo, country, results_root=results_root)
    states = country_status(repo, country, results_root=results_root)
    required = [key for key, unit in ctx.units.items() if unit.step in ('observe', 'planning', 'bounds', 'defense')]
    if any(states[key] != 'DONE' for key in required):
        raise ValueError('Experiment country tables require DONE input units')
    parts, inputs, receipts = {}, {}, {}
    for region in ctx.loaded.values['regions']:
        tables, receipt = _tables(ctx.root / 'observations' / region)
        receipts[receipt['node_id']] = receipt['receipt_sha256']
        for relative, table in tables.items():
            family, name = relative.split('/')
            parts.setdefault(family, {}).setdefault(Path(name).stem, []).append(table)
            inputs[f'{region}:{relative}'] = receipt['outputs'][relative]
    observations = {family: {name: pd.concat([t for t in frames if len(t.columns)], ignore_index=True)
                              if any(len(t.columns) for t in frames) else pd.DataFrame()
                              for name, frames in tables.items()} for family, tables in parts.items()}
    if not set(ctx.loaded.values['observe']['families']).issubset(observations):
        raise ValueError('Experiment observation families incomplete')
    planning_paths, ordinal = [], 0
    for cc in stage.COUNTRIES:
        other = ctx if cc == country else stage.country_context(repo, cc, results_root=results_root)
        for unit in other.units.values():
            if unit.step != 'planning':
                continue
            for field, seed in unit.coordinates:
                if cc == country:
                    planning_paths.append((str(ordinal), coordinate_root(unit, other.root, other.loaded, field, seed)))
                ordinal += 1
    planning = []
    for index, path in sorted(planning_paths):
        tables, receipt = _tables(path)
        planning.append(tables['metrics.csv'])
        inputs[f'planning:{index}'] = receipt['outputs']['metrics.csv']
        receipts[receipt['node_id']] = receipt['receipt_sha256']
    extras = {}
    for member in ('bounds', 'defense'):
        tables, receipt = _tables(ctx.root / member)
        extras[member] = {Path(name).stem: table for name, table in tables.items()}
        inputs.update({f'{member}:{name}': digest for name, digest in receipt['outputs'].items()})
        receipts[receipt['node_id']] = receipt['receipt_sha256']
    return {'observations': observations, 'planning': pd.concat(planning, ignore_index=True),
            **extras, 'inputs': inputs, 'receipt_sha256': receipts, 'states': states}
