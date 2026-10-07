"""008 gates: upstream injection, registered drawing, receipts, read-only audit."""
import ast
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from sglib.core.infra.artifacts import atomic_json
from sglib.core.infra.content_chain import derive_chain_commitment, derive_chain_receipt
from sglib.core.infra.hashing import sha256_file
from sglib.report import production, stage, tables
from sglib.report.config import load_report_config
from sglib.report.registry import build_registry, expected_node_id, unit_output_path, unit_status

pytestmark = pytest.mark.gate

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def ctx(tmp_path):
    return stage.country_context(ROOT, profile='smoke', results_root=tmp_path)


def make_receipt(ctx, member='C1', parameters=None, node=None):
    unit = ctx.units[f'cross.render.{member}']
    target = unit_output_path(unit, ctx.root).parent
    outputs = {}
    registration = ctx.loaded.registrations[member]
    paths = [f'figures/{identifier}.{suffix}' for identifier in registration['figures'] for suffix in ('png', 'pdf')]
    paths += [f'sources/{identifier}.csv' for identifier in registration['figures']]
    paths += [f'tables/{identifier}.csv' for identifier in registration['tables']]
    for name in paths:
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'identity,value\na,1\n')
        outputs[name] = sha256_file(path)
    if member == 'SYN':
        rows = []
        for claim, experiments in zip(ctx.loaded.spec['claims'],
                ('C1-E01,E04,E05', 'C2-E01..06,E08', 'C3-E01..05', 'C4-E01..05', 'C5-E01..05', 'C6-E01..05'), strict=True):
            rows.append({'claim_id': claim, 'experiments': experiments,
                         'figure_ids': ','.join(ctx.loaded.registrations[claim]['figures']),
                         'table_ids': ','.join(ctx.loaded.registrations[claim]['tables'])})
        path = target / 'tables/T-SYN-04.csv'
        pd.DataFrame(rows).to_csv(path, index=False, lineterminator='\n')
        outputs['tables/T-SYN-04.csv'] = sha256_file(path)
    commitment = derive_chain_commitment(node or expected_node_id(unit), inputs={},
        scientific_parameters=parameters or registration, code_sha256='b'*64)
    document = derive_chain_receipt(commitment, outputs=outputs)
    atomic_json(document, target / 'receipt.json')
    return target / 'receipt.json', document


def complete(ctx):
    for member in ctx.loaded.registrations:
        make_receipt(ctx, member)


def test_registry_dependencies_nodes_and_stable_ids(ctx):
    assert len(ctx.units) == 8 and len(ctx.loaded.ids) == 37
    assert sum(record['kind'] == 'figure' for record in ctx.loaded.ids.values()) == 19
    for member in ctx.loaded.registrations:
        unit = ctx.units[f'cross.render.{member}']
        expected = []
        if member != 'SYN':
            for cc in ctx.loaded.spec['countries']:
                if member != 'C6': expected.append(f'{cc}.core.{member}')
                if member != 'C1': expected.append(f'{cc}.support.support')
        if member in ('C4', 'SYN'): expected.append('cross.synthesis.synthesis')
        assert unit.depends_on == tuple(expected)
        assert expected_node_id(unit) == f'report.render.{member}.v1'
        assert unit_output_path(unit, ctx.root) == ctx.root / member / 'receipt.json'
        assert set(tables.requested(member, ctx.loaded.spec['countries'])) == {
            f'{key.split(".")[0]}:{key.split(".")[-1]}' for key in expected}
    assert ctx.units['cross.audit.audit'].depends_on == tuple(f'cross.render.{member}' for member in ctx.loaded.registrations)


def test_missing_malformed_and_wrong_node_receipts(ctx):
    unit = ctx.units['cross.render.C1']
    assert unit_status(unit, ctx.root, ctx.loaded) == 'PENDING'
    path, document = make_receipt(ctx)
    assert unit_status(unit, ctx.root, ctx.loaded) == 'DONE'
    document['receipt_sha256'] = '0'*64
    atomic_json(document, path)
    assert unit_status(unit, ctx.root, ctx.loaded) == 'INVALID'
    make_receipt(ctx, node='report.render.C2.v1')
    assert unit_status(unit, ctx.root, ctx.loaded) == 'INVALID'
    path.write_text('[]', encoding='utf-8', newline='\n')
    assert unit_status(unit, ctx.root, ctx.loaded) == 'INVALID'


def test_registration_mismatch_invalidates(ctx):
    changed = deepcopy(ctx.loaded.registrations['C1'])
    changed['spec']['metrics'] = ['rmse']
    make_receipt(ctx, parameters=changed)
    assert unit_status(ctx.units['cross.render.C1'], ctx.root, ctx.loaded) == 'INVALID'


def test_done_uses_existence_without_a_hash_gate(ctx):
    make_receipt(ctx)
    output = ctx.root / 'C1/figures/F-C1-01.png'
    output.write_bytes(b'new drawing bytes')
    unit = ctx.units['cross.render.C1']
    assert unit_status(unit, ctx.root, ctx.loaded) == 'DONE'
    output.unlink()
    assert unit_status(unit, ctx.root, ctx.loaded) == 'INVALID'


def test_predecessors_and_injection_are_required(ctx):
    with pytest.raises(stage.StageOrderError, match='earlier numbered'):
        stage.run_units(ctx, ['cross.audit.audit'])
    with pytest.raises(stage.StageOrderError, match='inject'):
        stage.run_units(ctx, ['cross.render.C1'])
    ctx.upstream['status'] = lambda repo: {}
    with pytest.raises(stage.StageOrderError, match='predecessors'):
        stage.run_units(ctx, ['cross.render.C1'])
    assert not ctx.root.exists()


def test_done_skips_production_but_requires_done_inputs(ctx, monkeypatch):
    make_receipt(ctx)
    ctx.upstream['status'] = lambda repo: {key: 'DONE' for key in ctx.units['cross.render.C1'].depends_on}
    monkeypatch.setattr(stage, '_execute', lambda *args: pytest.fail('DONE recomputed'))
    report = stage.run_units(ctx, ['cross.render.C1'])
    assert report.skipped == ('cross.render.C1',) and not report.ran
    ctx.upstream['status'] = lambda repo: {}
    with pytest.raises(stage.StageOrderError, match='predecessors'):
        stage.run_units(ctx, ['cross.render.C1'])


def test_formal_refuses_write_even_when_execute_is_called_directly(ctx):
    formal = replace(ctx, read_only=True, profile='formal')
    formal.upstream['status'] = lambda repo: {key: 'DONE' for key in formal.units['cross.render.C1'].depends_on}
    with pytest.raises(stage.StageOrderError, match='unfinished'):
        stage.run_units(formal, ['cross.render.C1'])
    with pytest.raises(ValueError, match='read-only'):
        production.execute(formal, formal.units['cross.render.C1'])
    with pytest.raises(ValueError, match='read-only'):
        production.publish(formal, formal.units['cross.render.C1'], {}, lambda target: pytest.fail('producer called'))
    assert not formal.root.exists()


def test_formal_audit_goes_to_views_and_index_has_two_tables(ctx, monkeypatch, tmp_path):
    complete(ctx)
    formal = replace(ctx, read_only=True, profile='formal')
    view = tmp_path / 'views'
    monkeypatch.setattr(stage, 'figures_root', lambda unused: view)
    result = stage.run_units(formal, ['cross.audit.audit'])
    assert result.prepared == ('cross.audit.audit',)
    assert not (ctx.root / 'audit.json').exists()
    document = json.loads((view / 'audit.json').read_text(encoding='utf-8'))
    assert document['status'] == 'PASS' and len(document['stable_ids']) == 37
    assert len(document['experiments']) == 30
    assert {row['experiment'] for row in document['experiments']} == {
        'C1-E01', 'C1-E04', 'C1-E05', 'C2-E08',
        *(f'C2-E{number:02d}' for number in range(1, 7)),
        *(f'C{claim}-E{number:02d}' for claim in range(3, 7) for number in range(1, 6)),
    }
    assert all(row['stable_ids'] for row in document['experiments'])
    text = (view / 'index.md').read_text(encoding='utf-8')
    assert len(text.strip().split('\n\n')) == 2
    assert all(line.startswith('|') for line in text.splitlines() if line)
    with pytest.raises(ValueError, match='views'):
        production.write_audit(formal, ctx.root)


def test_audit_detects_missing_stable_ids_and_incomplete_coverage(ctx):
    complete(ctx)
    assert production.coverage(ctx)['status'] == 'PASS'
    (ctx.root / 'C2/figures/F-C2-03.pdf').unlink()
    assert production.coverage(ctx)['status'] == 'FAIL'


def test_entrypoint_injects_only_analysis_adapter_and_notebook_is_clean():
    base = ROOT / 'casestudy/5_Report'
    script = (base / '01_render.py').read_text(encoding='utf-8')
    assert "UPSTREAM = {'status': status, 'tables': load}" in script
    assert 'from sglib.analysis.downstream import status, load' in script
    assert all(option in script for option in ('--claim', '--profile', '--results-root'))
    notebook = json.loads((base / '02_audit.ipynb').read_text(encoding='utf-8'))
    cells = [cell for cell in notebook['cells'] if cell['cell_type'] == 'code']
    assert cells and all(cell['outputs'] == [] and cell['execution_count'] is None for cell in cells)


def test_report_has_no_other_stage_imports():
    banned = ('sglib.analysis', 'sglib.experiment', 'sglib.generator', 'sglib.dataoverview')
    for path in (ROOT / 'sglib/report').rglob('*.py'):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(not alias.name.startswith(banned) for alias in node.names), path
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or '').startswith(banned), path
                assert not (node.level >= 2 and (node.module or '').startswith(('analysis', 'experiment', 'generator', 'dataoverview'))), path


def test_figure_filters_have_no_literal_selection_values():
    methods = {'eq', 'ne', 'lt', 'le', 'gt', 'ge', 'isin', 'query'}
    for path in (ROOT / 'sglib/report/figures').glob('*.py'):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in methods:
                assert all(not isinstance(arg, (ast.Constant, ast.List, ast.Tuple, ast.Set)) for arg in node.args), (path, node.lineno)
    renderer_source = '\n'.join(path.read_text(encoding='utf-8') for path in (ROOT / 'sglib/report/figures').glob('*.py'))
    for key in ('metrics', 'candidate_order', 'scan_metric', 'fixed_order_max', 'program_order_min',
                'map_candidate', 'secondary_status', 'scale_metric', 'scale_candidates',
                'tolerance_curve', 'budget_realization', 'saturation'):
        assert key in renderer_source


def test_c6_sources_keep_full_table_while_plotting_uses_registration(tmp_path, monkeypatch):
    from sglib.report.figures import common
    frame = pd.DataFrame({'country': ['nz', 'nz'], 'radius_km': [10, 20], 'x': [1, 2], 'y': [3, 4], 'candidate': ['a', 'a']})
    monkeypatch.setattr(common, 'save_at', lambda root, identifier, fig: common.plt.close(fig))
    common.line_facets(tmp_path, 'F-C6-01', frame, 'curve', 'x', 'y', 'candidate',
                       shared={'countries': ['nz']}, selections={'radius_km': 20})
    assert pd.read_csv(tmp_path / 'sources/F-C6-01.csv').radius_km.tolist() == [10, 20]
    assert common.select(frame, {'radius_km': 20}).x.tolist() == [2]


def test_heatmap_dimensions_follow_registration(tmp_path, monkeypatch):
    from sglib.report.figures import synthesis
    config = load_report_config(ROOT)
    spec = {**config.registrations['SYN']['spec'], 'figures': ['F-SYN-02'],
            'shared': {'claims': ['C2', 'C5'], 'countries': ['nz']}}
    rows = [{**dict.fromkeys(spec['status_columns'], ''), 'country': 'nz', 'claim_id': claim, 'valid': 1, 'expected': 2}
            for claim in spec['shared']['claims']]
    shapes = []
    def saved(root, identifier, fig):
        shapes.append(fig.axes[0].images[0].get_array().shape)
        synthesis.plt.close(fig)
    monkeypatch.setattr(synthesis, 'save_at', saved)
    synthesis.render({'cross:synthesis': {'claim_evidence_status': pd.DataFrame(rows)}}, spec, tmp_path)
    assert shapes == [(2, 1)]


def test_downstream_verifies_requested_bytes_and_receipt_sha(tmp_path, monkeypatch):
    from sglib.analysis import downstream
    root = tmp_path / 'C1'
    root.mkdir()
    output = root / 'chosen.csv'
    output.write_bytes(b'value\n1\n')
    commitment = derive_chain_commitment('analysis.C1.nz', inputs={}, scientific_parameters={}, code_sha256='a'*64)
    document = derive_chain_receipt(commitment, outputs={'chosen.csv': sha256_file(output), 'unrequested.csv': 'b'*64})
    atomic_json(document, root / 'receipt.json')
    unit = SimpleNamespace(id='nz.core.C1', step='core', member='C1')
    fake = SimpleNamespace(root=tmp_path, loaded=None, units={'nz.core.C1': unit})
    monkeypatch.setattr(downstream, 'country_context', lambda repo, country, **kwargs: fake)
    monkeypatch.setattr(downstream, 'unit_status', lambda *args: 'DONE')
    result = downstream.load(ROOT, {'nz:C1': ['chosen']})
    assert result['inputs'] == {'nz:C1': document['receipt_sha256']}
    assert list(result['tables']['nz:C1']) == ['chosen']
    output.write_bytes(b'value\n2\n')
    with pytest.raises(ValueError, match='differs from receipt'):
        downstream.load(ROOT, {'nz:C1': ['chosen']})


def test_downstream_explicit_input_root_reads_only_its_verified_tables(tmp_path):
    from sglib.analysis import downstream, stage as analysis_stage
    documents = []
    roots = [tmp_path / 'first', tmp_path / 'second']
    for ordinal, results in enumerate(roots, 1):
        ctx = analysis_stage.country_context(ROOT, 'nz', profile='smoke', results_root=results)
        target = ctx.root / 'C1'
        target.mkdir(parents=True)
        output = target / 'chosen.csv'
        output.write_text(f'value\n{ordinal}\n', encoding='utf-8')
        commitment = derive_chain_commitment('analysis.C1.nz', inputs={},
            scientific_parameters=ctx.loaded.registrations['C1'], code_sha256='a'*64)
        document = derive_chain_receipt(commitment, outputs={'chosen.csv': sha256_file(output)})
        atomic_json(document, target / 'receipt.json')
        documents.append(document)
    for ordinal, results in enumerate(roots, 1):
        payload = downstream.load(ROOT, {'nz:C1': ['chosen']}, results_root=results)
        assert payload['tables']['nz:C1']['chosen'].value.tolist() == [ordinal]
        assert payload['inputs']['nz:C1'] == documents[ordinal-1]['receipt_sha256']
        assert downstream.status(ROOT, results_root=results)['nz.core.C1'] == 'DONE'
    (roots[0] / '4_Analysis/5_NZ/C1/chosen.csv').write_bytes(b'value\n999\n')
    with pytest.raises(ValueError, match='differs from receipt'):
        downstream.load(ROOT, {'nz:C1': ['chosen']}, results_root=roots[0])
    assert downstream.load(ROOT, {'nz:C1': ['chosen']}, results_root=roots[1])['tables']['nz:C1']['chosen'].value.tolist() == [2]


def test_report_closure_makes_a_smoke_profile_read_only(ctx):
    from sglib.core.infra.content_chain import derive_chain_closure
    _, receipt = make_receipt(ctx)
    atomic_json(derive_chain_closure([receipt]), ctx.root / "_closures/release.json")
    closed = stage.country_context(ROOT, profile="smoke", results_root=ctx.results)
    assert closed.read_only
    with pytest.raises(ValueError, match="read-only"):
        production.execute(closed, closed.units["cross.render.C2"])
