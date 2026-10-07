"""Read-only verified Analysis tables for the overview notebooks."""
import json
import pandas as pd

from .. import production, stage
from ..config import CLAIMS


def status(ctx):
    return pd.DataFrame(stage.status_table(ctx))


def contrasts(ctx):
    frames = []
    for claim in CLAIMS[:5]:
        table = production.load_tables(ctx.root / claim)[0]['contrasts'].copy()
        table.insert(0, 'claim_id', claim)
        frames.append(table)
    return pd.concat(frames, ignore_index=True, sort=False)


def evidence(ctx):
    member = 'synthesis' if ctx.country == 'cross' else 'support'
    return production.load_tables(ctx.root / member)[0]['claim_evidence_status']


def bounds(ctx):
    source = production.load_tables(ctx.root / 'C6')[0]
    return {name: source[name] for name in ('bounds_audit', 'tolerance_curves')}


def synthesis(ctx):
    return production.load_tables(ctx.root / 'synthesis')[0]


def audit(ctx):
    document = json.loads((ctx.root / 'audit.json').read_text(encoding='utf-8'))
    return pd.DataFrame(document['units'])
