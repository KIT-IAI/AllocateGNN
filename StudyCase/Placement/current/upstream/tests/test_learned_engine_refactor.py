"""Temporary-root CPU model execution through the public prepared-task APIs."""
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import torch
from shapely.geometry import Point
from torch_geometric.data import HeteroData

from sglib.core.infra.artifacts import atomic_json
from sglib.core.infra.hashing import sha256_file
from sglib.generator.weighter.learned.inputs import save_graph_cache
from sglib.generator.weighter.learned.models.gnn.landuse import INDEXED_LANDUSE_REPRESENTATION
from sglib.generator.weighter.learned.training.tasks import TrainingTask
from sglib.generator.weighter.learned.training.preparation import prepare_tasks, load_prepared_task
from sglib.generator.weighter.learned.training.engine import run_training_task
from sglib.generator.weighter.learned.training.verify import verify_task_artifacts
from sglib.generator.weighter.learned.inference.preparation import prepare_inference_tasks, load_inference_task
from sglib.generator.weighter.learned.inference.engine import run_inference_task
from sglib.generator.weighter.learned.inference.verify import verify_inference_artifacts



def _synthetic_inputs(root):
    repo, results = root / 'repo', root / 'repo/results'
    results.mkdir(parents=True)
    params = {
        'regions': ['R1', 'R2'], 'n_folds': 2,
        'training_weighting_policy': 'uniform_region_cyclic_sgd__source_mean_v1',
        'training_batch_size': 1,
        'config_map': {'baseline': {'epochs': 2, 'objective_weights': {'landuse_prediction_loss': 1.0}}},
        'gnn': {'hidden_dim': 8, 'embedding_dim': 4, 'num_layers': 1, 'conv_type': 'sage',
                'allocation_temperature_start': 2.0, 'learning_rate': .001,
                'weight_decay': .0001, 'warmup_epochs': 0, 'decay_epochs': 0, 'cosine_eta_min': .000001},
        'mlp': {'learning_rate': .001, 'weight_decay': .0001, 'cosine_eta_min': .000001},
    }
    atomic_json(params, repo / 'params.json')
    graph = HeteroData()
    graph['source'].x = torch.tensor([[.1], [.9]], dtype=torch.float32)
    graph['agent'].x = torch.tensor([[.2], [.4], [.6], [.8]], dtype=torch.float32)
    edges = torch.tensor([[0, 0, 1, 1], [0, 1, 2, 3]])
    graph['source', 'connects_to', 'agent'].edge_index = edges
    graph['agent', 'rev_connects_to', 'source'].edge_index = edges.flip(0)
    graph.landuse_flat_index = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    graph.landuse_ratio = torch.tensor([[.3, .7], [.7, .3]], dtype=torch.float32)
    graph.landuse_supervision_representation = INDEXED_LANDUSE_REPRESENTATION
    graph.agent_index_map = pd.Series(range(4))
    graph.source_index_map = pd.Series(range(2))
    grid = gpd.GeoDataFrame({'source': ['A','A','B','B'], 'covered_mask': [True]*4,
        'unknown_mask': [False]*4, 'zero_mask': [False]*4, 'built_fraction': [1.]*4},
        geometry=[Point(x,0) for x in range(4)], crs='EPSG:3857')
    source = gpd.GeoDataFrame({'source': ['A','B'], 'Demand (MVA)': [3.,7.]},
        geometry=[Point(0,0), Point(3,0)], crs='EPSG:3857')
    regions = tuple(params['regions'])
    graphs = {r: graph.clone() for r in regions}
    bundle = SimpleNamespace(country='uk', regions=regions, source_column='source',
        demand_column='Demand (MVA)', relation_column='source',
        grids={r:(grid.copy(),) for r in regions}, ntl={r:np.ones(4) for r in regions},
        source_regions={r:source.copy() for r in regions}, stations={r:source.copy() for r in regions})
    cache = save_graph_cache(results/'graphs/lu5.pkl', bundle, graphs,
        feature_set='lu5', input_fingerprints={'synthetic': 'a'*64})
    return repo, results, cache


@pytest.mark.consume
@pytest.mark.parametrize('family', ['mlp','gnn'])
def test_cpu_training_inference_and_relocation(tmp_path, monkeypatch, family):
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    repo, results, cache = _synthetic_inputs(tmp_path)
    task = TrainingTask(group=f'B-UK-{family.upper()}', country='uk', family=family,
        config='baseline', signal='none', parameter='fixed', value='default', seed=42,
        fold=1, output_relative=f'training/{family}', feature_set='lu5', compute_threads=1)
    identity = {'run_identity': {'schema_version': 'sg_generator_execution_generation_v3',
        'run_fingerprint': 'a'*64, 'formal_reuse_allowed': False}}
    prepared = prepare_tasks([task], frozen_params_path=repo/'params.json',
        graph_cache_by_feature_set={'lu5':cache}, repo_root=repo, results_root=results,
        config_fingerprint=sha256_file(repo/'params.json'), input_fingerprints={'synthetic': 'a'*64},
        execution_backend='local', execution_identity=identity)[0]
    task_path = atomic_json(prepared.to_dict(), results/'task.json')
    loaded = load_prepared_task(task_path, repo_root=repo, results_root=results)
    output = run_training_task(loaded, backend='local', device='cpu')
    completion = verify_task_artifacts(loaded)
    assert completion['epochs_observed'] == 2
    checkpoint = torch.load(output/'model.pth', map_location='cpu', weights_only=False)
    assert ({'encoder','edge_layer'} if family=='mlp' else
        {'encoder_state_dict','edge_weighting_layer_state_dict'}) <= set(checkpoint)
    inference = prepare_inference_tasks([task_path], output_root=results/'inference',
        repo_root=repo, results_root=results, execution_backend='local')[0]
    inference_path = atomic_json(inference.to_dict(), results/'inference_task.json')
    loaded_inference = load_inference_task(inference_path, repo_root=repo, results_root=results)
    inferred = run_inference_task(loaded_inference, backend='local')
    inference_completion = verify_inference_artifacts(loaded_inference)
    assert inference_completion['chain']['commitment'] == inference.chain_commitment
    for region in ('R1','R2'):
        field = np.load(inferred/'fields'/f'{region}.npz')['data']
        np.testing.assert_allclose([field[:2].sum(),field[2:].sum()], [3.,7.], rtol=1e-8, atol=1e-8)
    # Moving a tree preserves historical task identity and checkpoint receipt.
    moved = tmp_path/'relocated'
    repo.rename(moved)
    relocated = load_prepared_task(moved/'results/task.json', repo_root=moved, results_root=moved/'results')
    assert verify_task_artifacts(relocated) == completion
    relocated_inference = load_inference_task(moved/'results/inference_task.json', repo_root=moved, results_root=moved/'results')
    assert verify_inference_artifacts(relocated_inference) == inference_completion
    assert run_training_task(relocated, backend='local', device='cpu') == Path(relocated.output_path)
    assert run_inference_task(relocated_inference, backend='local') == Path(relocated_inference.output_path)


@pytest.mark.gate
def test_recorded_parameter_locator_follows_promotion_and_case_rename(tmp_path):
    from sglib.generator.weighter.learned.paths import join_relative
    from sglib.generator.weighter.learned.errors import TrainingTaskError
    params = tmp_path / 'results/2_Generator/1_UK/inputs/worker_params.json'
    params.parent.mkdir(parents=True)
    params.write_bytes(b'promoted frozen parameters')
    historical = 'results/v3_final/2_Generator/1_UK/inputs/worker_params.json'
    assert join_relative(tmp_path, historical, 'params') == params
    original = tmp_path / historical
    original.parent.mkdir(parents=True)
    original.write_bytes(b'existing historical parameters')
    assert join_relative(tmp_path, historical, 'params') == original
    current = tmp_path / 'casestudy/config.json'
    current.parent.mkdir()
    current.write_bytes(b'case resource')
    # Historical recorded locator must remain unchanged while its target moved.
    assert join_relative(tmp_path, 'casestudy2/config.json', 'config') == current
    with pytest.raises(TrainingTaskError, match='safe relative path'):
        join_relative(tmp_path, '../outside.json', 'params')


@pytest.mark.formal_results
@pytest.mark.consume
@pytest.mark.parametrize('family', ['gnn', 'mlp'])
def test_inherited_checkpoint_loads_with_unchanged_weight_keys(family):
    """Read an original checkpoint into fresh modules without training or writing."""
    import json
    from sglib.generator.weighter.learned.inference.checkpoint import load_gnn_cpu
    from sglib.generator.weighter.learned.models.mlp import load_mlp_checkpoint
    root = Path(__file__).resolve().parents[1] / 'results/2_Generator/1_UK'
    checkpoint = root / f'2_Weighter/base/{family}/seed_123/fold1/model.pth'
    params_path = root / 'inputs/worker_params.json'
    if not checkpoint.is_file() or not params_path.is_file():
        pytest.skip('requires inherited UK GNN/MLP checkpoint and frozen worker parameters')
    params = json.loads(params_path.read_text(encoding='utf-8'))
    checkpoint_hash = sha256_file(checkpoint)
    graph = HeteroData()
    graph['source'].x = torch.zeros((2, len(params['source_feature_cols'])))
    graph['agent'].x = torch.zeros((4, len(params['agent_feature_cols'])))
    edge = torch.tensor([[0, 0, 1, 1], [0, 1, 2, 3]])
    graph['source', 'connects_to', 'agent'].edge_index = edge
    graph['agent', 'rev_connects_to', 'source'].edge_index = edge.flip(0)
    stored = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if family == 'gnn':
        model = load_gnn_cpu(SimpleNamespace(config='baseline', checkpoint_path=str(checkpoint)), {'toy':graph}, params, None)
        actual = {'encoder_state_dict': model.encoder.state_dict(),
                  'edge_weighting_layer_state_dict': model.edge_weighting_layer.state_dict()}
    else:
        encoder, edge_layer = load_mlp_checkpoint(checkpoint.parent, {'toy':graph}, params, device='cpu')
        actual = {'encoder': encoder.state_dict(), 'edge_layer': edge_layer.state_dict()}
    for component, tensors in actual.items():
        assert set(tensors) == set(stored[component])
        for name, tensor in tensors.items():
            torch.testing.assert_close(tensor.cpu(), stored[component][name].cpu(), rtol=0, atol=0)
    assert sha256_file(checkpoint) == checkpoint_hash
