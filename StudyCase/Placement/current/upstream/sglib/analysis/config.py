"""Analysis authority extracted from the 29 sealed receipts (007 §4)."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path

from sglib.core.infra.hashing import sha256_json
from sglib.core.infra.terms import load_country_profile

from sglib.core.infra.paths import case_study_root

COUNTRIES = ('uk', 'au', 'nl', 'nz')
DIRECTORIES = dict(zip(COUNTRIES, ('1_UK', '2_AU', '4_NL', '5_NZ')))
CLAIMS = tuple(f'C{i}' for i in range(1, 7))


@dataclass(frozen=True)
class LoadedAnalysisConfig:
    country: str
    directory: str
    specification: dict
    registrations: dict


def registration_projection(parameters):
    projected = deepcopy(dict(parameters))
    spec = projected.get('specification')
    if spec is not None and 'expected_coordinates' in spec:
        digest = sha256_json(spec.pop('expected_coordinates'))
        if spec.get('expected_coordinates_sha256') not in (None, digest):
            raise ValueError('Analysis expected coordinate digest differs')
        spec['expected_coordinates_sha256'] = digest
    return projected


def load_analysis_config(repo_root, country, *, config_root=None):
    """``config_root`` (a closed root's setup/config) replaces the checkout as the configuration tree."""
    repo = Path(config_root if config_root is not None else repo_root).resolve()
    base = case_study_root(repo) / '4_Analysis'
    general = json.loads((base / 'general/registrations.json').read_text(encoding='utf-8'))
    if country == 'cross':
        return LoadedAnalysisConfig(country, '9_CrossCountry', {}, {'synthesis': general['synthesis']})
    profile = load_country_profile(repo / 'casestudy/config/countries' / f'{country}.toml')
    local = json.loads((base / profile.directory / 'registrations.json').read_text(encoding='utf-8'))
    common_spec, local_spec = general['specification'], local['specification']
    if set(common_spec) & set(local_spec):
        raise ValueError('Analysis specification registered twice')
    spec = {**deepcopy(common_spec), **deepcopy(local_spec), 'country': country,
            'unit': profile.units['demand']}
    spec['expected_coordinates'] = [[region, method, seed] for region in spec['regions']
                                    for method in spec['method_order'] for seed in spec['methods'][method]['seeds']]
    # The large coordinate list is derived; its original registered digest remains authoritative.
    registration_projection({'specification': spec})
    registrations = {}
    for kind in (*CLAIMS, 'support'):
        shared, overlay = general['units'][kind], local['units'].get(kind, {})
        if set(shared) & set(overlay):
            raise ValueError(f'{country}/{kind}: parameters registered twice')
        value = {**deepcopy(shared), **deepcopy(overlay), 'specification': deepcopy(spec)}
        if kind == 'support':
            value['country'] = country
        registrations[kind] = value
    return LoadedAnalysisConfig(country, profile.directory, spec, registrations)


def registered_values_match(kind, config, parameters):
    expected = registration_projection(config.registrations[kind])
    actual = registration_projection(parameters)
    return sorted(key for key, value in expected.items()
                  if key not in actual or sha256_json(actual[key]) != sha256_json(value))
