"""On-demand verified sealed Analysis tables for entrypoint injection."""
import json

from sglib.core.infra.content_chain import verify_chain
from sglib.core.infra.hashing import sha256_file
from .config import COUNTRIES
from .production import frame
from .registry import unit_output_path, unit_status
from .stage import country_context


def status(repo, *, results_root=None):
    states = {}
    for country in (*COUNTRIES, 'cross'):
        ctx = country_context(repo, country, results_root=results_root)
        states.update({key: unit_status(unit, ctx.root, ctx.loaded) for key, unit in ctx.units.items()})
    return states


def load(repo, keys, *, results_root=None):
    """keys maps '<country>:<member>' to table stems; input identities are receipt SHAs.

    The requested tables alone are decoded, and every requested file is verified
    using the same path and SHA checks as production.load_tables. Inputs default
    to sealed Analysis; only an explicit results_root selects a verified rerun.
    Output profiles and environment variables never redirect these reads.
    """
    tables, inputs = {}, {}
    for key, names in keys.items():
        country, member = key.split(':')
        ctx = country_context(repo, country, results_root=results_root)
        step = 'synthesis' if country == 'cross' else 'support' if member == 'support' else 'core'
        unit = ctx.units[f'{country}.{step}.{member}']
        if unit_status(unit, ctx.root, ctx.loaded) != 'DONE':
            raise ValueError(f'Analysis input is not DONE: {unit.id}')
        root = unit_output_path(unit, ctx.root).parent
        receipt = verify_chain(json.loads((root / 'receipt.json').read_text(encoding='utf-8')))
        part = {}
        for name in names:
            relative = f'{name}.csv'
            path = (root / relative).resolve()
            if (not path.is_relative_to(root.resolve()) or not path.is_file()
                    or receipt['outputs'].get(relative) != sha256_file(path)):
                raise ValueError(f'Analysis output differs from receipt: {path}')
            part[name] = frame(path)
        if not part:
            raise ValueError(f'empty Analysis table request: {key}')
        tables[key] = part
        inputs[key] = receipt['receipt_sha256']
    return {'tables': tables, 'inputs': inputs}
