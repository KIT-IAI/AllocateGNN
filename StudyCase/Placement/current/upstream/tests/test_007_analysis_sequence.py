"""007 contracts: immutable roots, dependencies, registration, full leaf gate and context."""
from copy import deepcopy
import json
from pathlib import Path

import pandas as pd
import pytest

from sglib.analysis import stage, manifest, synthesis
from sglib.analysis.config import CLAIMS, COUNTRIES, DIRECTORIES, load_analysis_config, registration_projection
from sglib.analysis.registry import AUDIT_SCHEMA, build_registry, expected_node_id, unit_output_path, unit_status
from sglib.analysis.production import support_tables
from sglib.core.infra.artifacts import atomic_json
from sglib.core.infra.content_chain import derive_chain_commitment, derive_chain_receipt, derive_chain_closure
from sglib.core.infra.hashing import sha256_file

pytestmark = pytest.mark.gate

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    # Views normally live under the repository's results/_views; keep test views in tmp_path.
    def isolated_figures_root(context):
        path = tmp_path / '_views' / '4_Analysis' / context.loaded.directory
        path.mkdir(parents=True, exist_ok=True)
        return path
    monkeypatch.setattr(stage, 'figures_root', isolated_figures_root)
    return stage.country_context(ROOT, 'nz', profile='smoke', results_root=tmp_path)


def receipt(ctx, member='C1', parameters=None):
    key = f'nz.core.{member}' if member in CLAIMS else 'nz.support.support'
    unit = ctx.units[key]
    path = unit_output_path(unit, ctx.root)
    path.parent.mkdir(parents=True, exist_ok=True)
    output = path.parent / 'table.csv'
    output.write_bytes(b'identity,value\na,1\n')
    commitment = derive_chain_commitment(expected_node_id(unit), inputs={},
        scientific_parameters=parameters or ctx.loaded.registrations[member], code_sha256='b'*64)
    document = derive_chain_receipt(commitment, outputs={'table.csv': sha256_file(output)})
    atomic_json(document, path)
    return path, document


def test_registry_matches_country_and_cross_dependencies():
    for cc in COUNTRIES:
        config = load_analysis_config(ROOT, cc)
        units = build_registry(config)
        assert len(units) == 8
        assert units[f'{cc}.core.C2'].depends_on == (f'{cc}.core.C1',)
        assert units[f'{cc}.core.C6'].depends_on == (f'{cc}.bounds.c6',)
        assert len(units[f'{cc}.core.C4'].depends_on) == 2*len(config.specification['regions'])
        assert units[f'{cc}.support.support'].depends_on[-1] == f'{cc}.defense.support'
    units = build_registry(load_analysis_config(ROOT, 'cross'))
    assert len(units) == 2 and len(units['cross.synthesis.synthesis'].depends_on) == 28


def test_large_coordinate_projection_keeps_registered_digest(ctx):
    parameters = ctx.loaded.registrations['C1']
    projected = registration_projection(parameters)
    assert 'expected_coordinates' not in projected['specification']
    assert 'expected_coordinates' in parameters['specification']
    modified = deepcopy(parameters)
    modified['specification']['expected_coordinates'].pop()
    with pytest.raises(ValueError, match='digest'):
        registration_projection(modified)


def test_receipt_state_valid_missing_and_malformed(ctx):
    unit = ctx.units['nz.core.C1']
    assert unit_status(unit,ctx.root,ctx.loaded)=='PENDING'
    path, document = receipt(ctx)
    assert unit_status(unit,ctx.root,ctx.loaded)=='DONE'
    document['receipt_sha256']='0'*64
    atomic_json(document,path)
    assert unit_status(unit,ctx.root,ctx.loaded)=='INVALID'
    receipt(ctx)
    (path.parent/'table.csv').unlink()
    assert unit_status(unit,ctx.root,ctx.loaded)=='INVALID'
    path.write_text('[]',encoding='utf-8')
    assert unit_status(unit,ctx.root,ctx.loaded)=='INVALID'


def test_registered_statistics_and_node_identity_are_checked(ctx):
    changed=deepcopy(ctx.loaded.registrations['C1'])
    changed['bootstrap_seed']+=1
    receipt(ctx,parameters=changed)
    assert unit_status(ctx.units['nz.core.C1'],ctx.root,ctx.loaded)=='INVALID'
    path, doc=receipt(ctx)
    wrong=derive_chain_commitment('analysis.C1.uk', inputs={},scientific_parameters=ctx.loaded.registrations['C1'],code_sha256='b'*64)
    atomic_json(derive_chain_receipt(wrong,outputs=doc['outputs']),path)
    assert unit_status(ctx.units['nz.core.C1'],ctx.root,ctx.loaded)=='INVALID'


def test_numbered_predecessor_is_required(ctx):
    with pytest.raises(stage.StageOrderError,match='earlier numbered'):
        stage.run_units(ctx,['nz.core.C2'])


def test_cross_requires_all_four_country_chains(tmp_path):
    cross=stage.country_context(ROOT,'cross',profile='smoke',results_root=tmp_path)
    with pytest.raises(stage.StageOrderError,match='predecessors'):
        stage.run_units(cross,['cross.synthesis.synthesis'])


def test_injected_experiment_status_must_be_done(ctx):
    receipt(ctx)
    with pytest.raises(stage.StageOrderError,match='inject'):
        stage.run_units(ctx,['nz.core.C1'])
    ctx.upstream['status']=lambda repo,country: {}
    with pytest.raises(stage.StageOrderError,match='predecessors'):
        stage.run_units(ctx,['nz.core.C1'])


def test_done_skips_production(ctx,monkeypatch):
    receipt(ctx)
    ctx.upstream['status']=lambda repo,country: {key:'DONE' for key in ctx.units['nz.core.C1'].depends_on}
    monkeypatch.setattr('sglib.analysis.production.execute',lambda *args: pytest.fail('DONE recomputed'))
    report=stage.run_units(ctx,['nz.core.C1'])
    assert report.skipped==('nz.core.C1',) and not report.ran


def test_formal_root_refuses_missing_production(tmp_path):
    formal=stage.country_context(ROOT,'nz',profile='formal',results_root=tmp_path,
        upstream={'status':lambda repo,country: {f'nz.observe.{r}':'DONE' for r in load_analysis_config(ROOT,'nz').specification['regions']}})
    with pytest.raises(stage.StageOrderError,match='unfinished'):
        stage.run_units(formal,['nz.core.C1'])
    assert not formal.root.exists()


def test_closure_also_makes_smoke_read_only(ctx):
    _,doc=receipt(ctx)
    atomic_json(derive_chain_closure([doc]),ctx.results/'4_Analysis/_closures/core.json')
    reopened=stage.country_context(ROOT,'nz',profile='smoke',results_root=ctx.results)
    assert reopened.read_only


def test_manifest_requires_placed_audit_and_detects_content_or_extras(ctx):
    for kind in (*CLAIMS,'support'):
        receipt(ctx,kind)
    with pytest.raises(RuntimeError,match='audit.json'):
        manifest.run_gate(ctx)
    atomic_json({'schema_version':AUDIT_SCHEMA,'country':'nz','status':'PASS'},ctx.root/'audit.json')
    assert manifest.run_gate(ctx)['status']=='GENERATED'
    assert manifest.run_gate(ctx)['status']=='PASS'
    (ctx.root/'C1/table.csv').write_bytes(b'identity,value\na,2\n')
    report=manifest.run_gate(ctx)
    assert report['status']=='FAIL' and report['leaves']['C1']['changed']==['C1/table.csv']
    (ctx.root/'unexpected.csv').write_bytes(b'x\n1\n')
    report=manifest.run_gate(ctx)
    assert report['status']=='FAIL' and report['unclaimed']==['unexpected.csv']


def test_c4_support_tables_follow_data_keys_and_candidate_registry(monkeypatch):
    from sglib.analysis import support
    levels=pd.DataFrame([
        {'region':region,'candidate':candidate,'task_coordinate':task,'metric':metric,'value':float(i)}
        for i,(task,region,candidate,metric) in enumerate([
            ('sizing','b','GPM','RSD'), ('siting','a','GPM','WSD'),
            ('sizing','a','GNN','RSD'), ('sizing','a','GPM','RSD')])])
    matching=pd.DataFrame({'candidate':['GNN','GPM','GNN'],'metric':['b','b','a'],
                           'matching':['one','one','many'],'unit':['u']*3,'value':[1.,2.,3.]})
    for name in ('c2_support','c3_support','c5_support','c6_support'):
        monkeypatch.setattr(support,name,lambda *args:{})
    monkeypatch.setattr(support,'evidence_status',lambda *args:pd.DataFrame())
    empty=pd.DataFrame()
    obs={family:{name:empty for name in names} for family,names in {
        'correction':['coordinates'],'sweeps':['metrics'],'allocator':['gates'],
        'reconstruction':['metrics'],'connection':['connection_metrics','panel_metrics','scale_metrics']}.items()}
    upstream={'observations':obs,'planning':empty,'defense':{'C4_fixed_300':empty}}
    core={claim:{name:empty for name in ('inference_resolution_audit','coordinate_audit','region_metrics',
          'region_differences','panel_associations','bounds_observations','bounds_audit','tolerance_curves')} for claim in CLAIMS}
    registry=json.loads((ROOT/'casestudy/2_Generator/general/candidate_registry.json').read_text(encoding='utf-8'))
    order=[candidate['label'] for candidate in registry['candidates']]
    outputs=[]
    for seed in (17,31):
        monkeypatch.setattr(support,'c4_support',lambda *args: {
            'C4_regional_levels':levels.sample(frac=1,random_state=seed),
            'C4_matching_summary':matching.sample(frac=1,random_state=seed)})
        outputs.append(support_tables('nz',upstream,core,order))
    pd.testing.assert_frame_equal(outputs[0]['C4_regional_levels'],outputs[1]['C4_regional_levels'])
    pd.testing.assert_frame_equal(outputs[0]['C4_matching_summary'],outputs[1]['C4_matching_summary'])
    assert outputs[0]['C4_regional_levels'].value.tolist()==[1.,2.,3.,0.]
    candidate_ranks=[order.index(candidate) for candidate in outputs[0]['C4_matching_summary'].candidate]
    assert candidate_ranks==sorted(candidate_ranks)
    assert sorted(outputs[0]['C4_matching_summary'].value)==[1.,2.,3.]


def synthesis_inputs():
    data={}
    for cc,total in zip(COUNTRIES,(3,5,7,11),strict=True):
        core={'C1':{'shared_inventory':pd.DataFrame({'country':[cc]}),
                    'region_metrics':pd.DataFrame({'region':['r','r'],'metric':['rmse','rmse'],'candidate':['GPM','Uni'],'value':[1.,2.]}),
                    'contrasts':pd.DataFrame({'left':['GPM'],'right':['Uni'],'contrast_id':['C1-01'],'unit':['u']})}}
        for claim in CLAIMS[1:5]:
            core[claim]={'region_differences':pd.DataFrame({'region':['r'],'contrast_id':[claim+'-01'],'value':[1.],'unit':['u']})}
        data[cc]={'core':core, 'support':{
            'claim_evidence_status':pd.DataFrame({'country':[cc]*6,'claim_id':list(CLAIMS)}),
            'claim_coordinate_audit':pd.DataFrame({'claim_id':list(CLAIMS)}),
            'inference_resolution_audit':pd.DataFrame({'claim_id':['C1']})},
            'defense':{'C4_connection_map':pd.DataFrame({'region':['r']})},
            'context':pd.DataFrame({'country':[cc],'region':['r'],'n_source':[2],'n_target':[total],
                                    'granularity_ratio':[float(total)/2],'capacity_basis':['declared'],'unit':['u'],
                                    'source_total':[10.],'target_total':[10.]})}
    return data


def test_synthesis_reconciles_region_context_and_full_matrix():
    data=synthesis_inputs()
    output=synthesis.assemble(data)
    assert len(output['claim_evidence_status'])==24
    audit=output['target_count_audit']
    # Target counts are read from the region context and reconciled; no historical count is registered.
    assert audit.country_target_count_from_regions.tolist()==[3,5,7,11]
    assert audit[['region_nonempty','granularity_ratio_consistent','mass_reconciled']].all().all()
    assert 'registered_target_count' not in audit
    assert output['coverage_corrections'].empty and len(output['coverage_corrections'].columns)==7
    for column,value in (('n_target',0),('granularity_ratio',99.),('target_total',9.)):
        broken=synthesis_inputs(); broken['nz']['context'].loc[0,column]=value
        with pytest.raises(ValueError,match='target'):
            synthesis.assemble(broken)
    broken=synthesis_inputs(); broken['nz']['context'].loc[0,'region']='other'
    with pytest.raises(ValueError,match='上下文'):
        synthesis.assemble(broken)


def test_synthesis_rejects_duplicate_or_missing_claim():
    data=synthesis_inputs()
    data['nz']['support']['claim_evidence_status'].loc[5,'claim_id']='C1'
    with pytest.raises(ValueError,match='矩阵'):
        synthesis.assemble(data)


def test_numbered_entrypoints_and_notebooks_are_complete():
    for cc,directory in DIRECTORIES.items():
        base=ROOT/'casestudy/4_Analysis'/directory
        assert (base/'01_core.py').is_file() and (base/'02_support.py').is_file()
        assert '--claim' in (base/'01_core.py').read_text(encoding='utf-8')
        doc=json.loads((base/'03_overview.ipynb').read_text(encoding='utf-8'))
        cells=[cell for cell in doc['cells'] if cell['cell_type']=='code']
        assert all(cell['outputs']==[] and cell['execution_count'] is None for cell in cells)
        assert 'tables.bounds(ctx)' in ''.join(''.join(cell['source']) for cell in cells)
    base=ROOT/'casestudy/4_Analysis/9_CrossCountry'
    assert (base/'01_synthesis.py').is_file() and (base/'02_overview.ipynb').is_file()
    doc=json.loads((base/'02_overview.ipynb').read_text(encoding='utf-8'))
    assert all(cell['outputs']==[] and cell['execution_count'] is None for cell in doc['cells'] if cell['cell_type']=='code')
