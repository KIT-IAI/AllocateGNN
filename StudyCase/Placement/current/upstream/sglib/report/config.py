"""Registered mechanical rendering choices; no statistical configuration."""
from dataclasses import dataclass
from pathlib import Path
import tomllib

from sglib.core.infra.hashing import sha256_json
from sglib.core.infra.paths import resolve_case_path


@dataclass(frozen=True)
class LoadedReportConfig:
    spec: dict
    registrations: dict
    ids: dict


REPORT_CONFIG = 'casestudy/5_Report/general/report.toml'


def load_report_config(repo, *, config_root=None):
    """``config_root`` (a closed root's setup/config) replaces the checkout as the configuration tree."""
    path = resolve_case_path(Path(config_root if config_root is not None else repo) / REPORT_CONFIG)
    document = tomllib.loads(path.read_text(encoding='utf-8'))
    shared, units = document['shared'], document['units']
    if list(units) != [*shared['claims'], 'SYN']:
        raise ValueError('Report units must match the registered claim order and SYN')
    if not shared['countries'] or len(set(shared['countries'])) != len(shared['countries']):
        raise ValueError('Report countries must be nonempty and unique')
    if set(shared['colors']) != set(shared['countries']):
        raise ValueError('Report country colors are incomplete')
    ids = {}
    for member, registration in units.items():
        for kind in ('figures', 'tables'):
            for identifier in registration[kind]:
                if identifier in ids or '/' in identifier or '\\' in identifier:
                    raise ValueError(f'duplicate or unsafe Report ID: {identifier}')
                ids[identifier] = {'member': member, 'kind': kind[:-1]}
    return LoadedReportConfig(shared, units, ids)


def registered_values_match(member, config, parameters):
    """Analysis registration comparison with an empty projection rule."""
    expected = config.registrations[member]
    return sorted(key for key in set(expected) | set(parameters)
                  if key not in expected or key not in parameters
                  or sha256_json(expected[key]) != sha256_json(parameters[key]))
