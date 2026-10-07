"""Analysis units; cross-stage dependencies are names answered by the entrypoint."""

from functools import partial
from sglib.core.chain.stage import topological_order as _chain_topological_order
from dataclasses import dataclass
import json

from sglib.core.infra.content_chain import ContentChainError, verify_chain
from .config import CLAIMS, COUNTRIES, registered_values_match

AUDIT_SCHEMA = 'sg_analysis_audit_v1'


@dataclass(frozen=True)
class AnalysisUnit:
    id: str
    country: str
    step: str
    member: str
    depends_on: tuple[str, ...]


def build_registry(config):
    country = config.country
    units = {}

    def add(step, member, dependencies):
        key = f'{country}.{step}.{member}'
        units[key] = AnalysisUnit(key, country, step, member, tuple(dependencies))
        return key

    if country == 'cross':
        dependencies = [f'{cc}.core.{claim}' for cc in COUNTRIES for claim in CLAIMS]
        dependencies += [f'{cc}.support.support' for cc in COUNTRIES]
        synthesis = add('synthesis', 'synthesis', dependencies)
        add('audit', 'audit', [synthesis])
        return units
    observed = [f'{country}.observe.{region}' for region in config.specification['regions']]
    planning = [f'{country}.planning.{region}' for region in config.specification['regions']]
    for claim in CLAIMS:
        dependencies = ([f'{country}.core.C1'] if claim == 'C2' else
                        [f'{country}.bounds.c6'] if claim == 'C6' else
                        observed + planning if claim == 'C4' else observed)
        add('core', claim, dependencies)
    add('support', 'support', [*(f'{country}.core.{claim}' for claim in CLAIMS), f'{country}.defense.support'])
    add('audit', 'audit', list(units))
    return units


topological_order = partial(_chain_topological_order, label='Analysis')


def expected_node_id(unit):
    if unit.step == 'core':
        return f'analysis.{unit.member}.{unit.country}'
    if unit.step == 'support':
        return f'analysis.support006.{unit.country}.v2'
    if unit.step == 'synthesis':
        return 'analysis.plan006.cross_country.v2'
    raise ValueError('audit is not a content-chain node')


def unit_output_path(unit, root):
    return root / 'audit.json' if unit.step == 'audit' else root / unit.member / 'receipt.json'


def unit_status(unit, root, config):
    path = unit_output_path(unit, root)
    if not path.exists():
        return 'PENDING'
    try:
        doc = json.loads(path.read_text(encoding='utf-8'))
        if unit.step == 'audit':
            return 'DONE' if doc.get('schema_version') == AUDIT_SCHEMA and doc.get('country') == unit.country and doc.get('status') == 'PASS' else 'INVALID'
        verify_chain(doc)
        if doc.get('node_id') != expected_node_id(unit) or registered_values_match(unit.member, config, doc['commitment']['scientific_parameters']):
            return 'INVALID'
        if not doc.get('outputs'):
            return 'INVALID'
        for name in doc['outputs']:
            output = (path.parent / name).resolve()
            if not output.is_relative_to(path.parent.resolve()) or not output.is_file():
                return 'INVALID'
        return 'DONE'
    except (OSError, ValueError, KeyError, TypeError, AttributeError, ContentChainError):
        return 'INVALID'
