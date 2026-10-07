"""Seven rendering units and their injected Analysis dependencies."""

from functools import partial
from sglib.core.chain.stage import topological_order as _chain_topological_order
from dataclasses import dataclass
import json

from sglib.core.infra.content_chain import ContentChainError, verify_chain
from .config import registered_values_match

AUDIT_SCHEMA = 'sg_report_audit_v1'


@dataclass(frozen=True)
class ReportUnit:
    id: str
    country: str
    step: str
    member: str
    depends_on: tuple[str, ...]


def build_registry(config):
    units = {}
    for member in config.registrations:
        dependencies = []
        if member != 'SYN':
            for cc in config.spec['countries']:
                if member != 'C6':
                    dependencies.append(f'{cc}.core.{member}')
                if member != 'C1':
                    dependencies.append(f'{cc}.support.support')
        if member in ('C4', 'SYN'):
            dependencies.append('cross.synthesis.synthesis')
        key = f'cross.render.{member}'
        units[key] = ReportUnit(key, 'cross', 'render', member, tuple(dependencies))
    key = 'cross.audit.audit'
    units[key] = ReportUnit(key, 'cross', 'audit', 'audit', tuple(units))
    return units


topological_order = partial(_chain_topological_order, label='Report')


def expected_node_id(unit):
    if unit.step != 'render':
        raise ValueError('audit is not a content-chain node')
    return f'report.render.{unit.member}.v1'


def unit_output_path(unit, root):
    return root / 'audit.json' if unit.step == 'audit' else root / unit.member / 'receipt.json'


def unit_status(unit, root, config):
    path = unit_output_path(unit, root)
    if not path.exists():
        return 'PENDING'
    try:
        document = json.loads(path.read_text(encoding='utf-8'))
        if unit.step == 'audit':
            return ('DONE' if document.get('schema_version') == AUDIT_SCHEMA
                    and document.get('status') == 'PASS' and (root / 'index.md').is_file() else 'INVALID')
        verify_chain(document)
        if document['node_id'] != expected_node_id(unit) or registered_values_match(
                unit.member, config, document['commitment']['scientific_parameters']):
            return 'INVALID'
        for name in document['outputs']:
            output = (path.parent / name).resolve()
            if not output.is_relative_to(path.parent.resolve()) or not output.is_file():
                return 'INVALID'
        return 'DONE'
    except (OSError, ValueError, KeyError, TypeError, AttributeError, ContentChainError):
        return 'INVALID'
