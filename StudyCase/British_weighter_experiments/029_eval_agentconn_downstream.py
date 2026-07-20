# -*- coding: utf-8 -*-
"""
Input-channel disclosure table, plus an optional AgentConn downstream
evaluation step (029_eval_agentconn_downstream.py).

Two parts:

[Part 1: disclosure table (always run)] Machine-generates an "input-channel
disclosure table" that documents exactly which input channels the model
consumes -- so the paper can state explicitly what the model does and does
not "see". **Every entry is read from the source of truth via importlib /
static source-code scanning / graph-cache inspection; nothing is
hand-transcribed** (the companion test tests/test_r214.py asserts that the
reported fields match the re-verified facts, guarding against manual
transcription errors). Entries cover:

- 005's AGENT_FEATURE_COLS (5-dimensional land-use features; no NTL, no
  coordinates)
- AGENT_CONNECTIVITY = None (the main experiment has no agent-agent edges)
- The loss sets for the four CONFIG_MAP configurations (NTL/Proximity only
  enter training through the prior loss terms)
- tau (allocation_temperature_start) = 0.01, which appears at two locations
  in 005 (the resume branch and the normal branch); both occurrences are
  located and recorded via static regex scanning
- GraphBuilder static facts: geometry is dropped before featurization
  (in preprocess_features); agent.x holds only the processed features,
  with coordinates stored as a separate coords attribute
- EdgeWeightSolver static facts: the encoder forward pass consumes only
  x_dict and edge_index_dict (coords / ntl_values / proximity_scores never
  reach the forward pass; the latter two are used only by the prior loss)
- Empirical inspection of the main graph_cache and the AgentConn
  graph_cache: node/edge types, agent feature dimensionality, near-edge
  counts (0 in the main cache, > 0 in AgentConn), star-topology structure,
  and the list of auxiliary attributes

Output: results/exp_r214/input_channel_disclosure.json

[Part 2: AgentConn downstream evaluation (optional, run in this invocation)]
exp0_kfold_prior_AgentConn has only 48 fully trained models (trained on an
HPC cluster; at the validation-loss level they show "no difference" from
the main experiment -- see
StudyCase/docs/paper-antagonism-ntl-proximity/agent_conn_ablation_report.md)
and **has never been evaluated on downstream RMSE/MAE/Corr**. This script
uses its own graph_cache (which includes near-edges) to run the same
inference procedure as 005 on CPU, plus Voronoi aggregation (the inference
logic reuses 005's functions via the same importlib file-path import
technique used elsewhere in this suite; aggregation reuses the assignment-
cache split from shared_correction_utils). It produces kfold_test csv files
for the three metrics, structured identically to exp0, and compares them
against the main results using the standard significance-testing protocol
(AgentConn baseline vs. main baseline).

Outputs (results/exp_r214/agentconn_downstream/):
- seed_{s}/{config}/kfold_test_{rmse,mae,corr}.csv (36 files, structured
  identically to exp0, containing only the 6 voronoi_* methods -- CIVD is
  skipped, see below)
- agentconn_vs_main.csv        (long-format per-region comparison table:
                                 config x method x metric x region)
- agentconn_vs_main_test.json  (significance test: exact permutation p +
                                 bootstrap CI + per-seed p + Cauchy
                                 combination for the 3 baseline-arm metrics,
                                 plus a machine-generated conclusion field)
- run_meta.json                (run metadata + split verification + record
                                 of any failed models)

Intentional differences from 005's predict_and_evaluate_location (following
the same approach used elsewhere in this suite, e.g. 020):
- CIVD assignment evaluation is skipped (it requires an hdbscan clustering
  + Pyomo solver chain; the comparison test only needs the voronoi_*
  metrics, which line up directly with the voronoi_* rows in the main
  results csv);
- The grid_demands pickle is not written to disk (only the metrics are
  needed here; this avoids regenerating 768 derived artifacts);
- Voronoi assignment and proximity scores are cached per region (they are
  identical across all 48 models for a given region, so this is a pure
  performance cache and does not change any numeric result).

This script only reads pre-existing frozen inputs (models, caches, the main
csv) and writes only to results/exp_r214/. It runs entirely on CPU
(device='cpu', with torch.load's map_location following device) because the
GPU is concurrently occupied training other model configurations.

Usage::

    python 029_eval_agentconn_downstream.py                    # disclosure table + downstream eval
    python 029_eval_agentconn_downstream.py --disclosure-only  # disclosure table only
"""

import argparse
import gc
import importlib.util
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path

# Guard against an OpenMP conflict between torch (libiomp) and scipy (MKL)
# in the same process (a known issue in this project; same workaround used
# in 020)
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold

# ════════════════════════════════════════════════════════════
# Paths (repo root located via the same walk-up approach as 020; 005's
# fixed parent-count approach is not reliable here)
# ════════════════════════════════════════════════════════════

SCRIPT_DIR = Path(__file__).resolve().parent
_p = SCRIPT_DIR
while not (_p / 'SpatialAllocation').exists():
    if _p.parent == _p:
        raise RuntimeError('Could not locate the repository root (the SpatialAllocation package)')
    _p = _p.parent
PROJECT_ROOT = _p
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from SpatialAllocation.GNN.core.EdgeWeightSolver import EdgeWeightSolver  # noqa: E402
from SpatialAllocation.GNN.core.ModelConfig import ModelConfig  # noqa: E402

import shared_correction_utils as scu  # noqa: E402
from revision_statistics import (  # noqa: E402
    exact_sign_flip_permutation, paired_region_bootstrap, cauchy_combination,
)

# ─── Source files targeted by the disclosure-table static scanner (read-only) ───
SRC_005 = SCRIPT_DIR / '005_kfold_prior_training.py'
SRC_GRAPHBUILDER = PROJECT_ROOT / 'SpatialAllocation' / 'GNN' / 'utils' / 'GraphBuilder.py'
SRC_SOLVER = PROJECT_ROOT / 'SpatialAllocation' / 'GNN' / 'core' / 'EdgeWeightSolver.py'
SRC_NTL_LOSS = PROJECT_ROOT / 'SpatialAllocation' / 'GNN' / 'Layer' / 'LossFunction' / 'NTLPriorLoss.py'
SRC_PROX_LOSS = PROJECT_ROOT / 'SpatialAllocation' / 'GNN' / 'Layer' / 'LossFunction' / 'ProximityPriorLoss.py'

# ─── Frozen inputs ───
RESULTS_DIR = SCRIPT_DIR / 'results'
EXP0_DIR = RESULTS_DIR / 'exp0_kfold_prior'
AGENTCONN_DIR = RESULTS_DIR / 'exp0_kfold_prior_AgentConn'
MAIN_CACHE = EXP0_DIR / 'graph_cache' / 'cached_graphs.pickle'
AGENTCONN_CACHE = AGENTCONN_DIR / 'graph_cache' / 'cached_graphs.pickle'

# ─── Outputs ───
OUT_DIR = RESULTS_DIR / 'exp_r214'
DISCLOSURE_JSON = OUT_DIR / 'input_channel_disclosure.json'
DOWNSTREAM_DIR = OUT_DIR / 'agentconn_downstream'

SEEDS = [42, 123, 456]
CONFIGS = ['baseline', 'ntl', 'proximity', 'ntl_prox']
N_FOLDS = 4
METRICS = ['rmse', 'mae', 'corr']

# The downstream evaluation covers only the voronoi_* metrics (CIVD is
# skipped, see the module docstring)
VORONOI_METHODS = {
    'voronoi_GNN': 'gnn_demand',
    'voronoi_wc_GNN': 'wc_gnn_demand',
    'voronoi_ntl_GNN': 'ntl_gnn_demand',
    'voronoi_prox_GNN': 'prox_gnn_demand',
    'voronoi_ntl_prox_GNN': 'ntl_prox_gnn_demand',
    'voronoi_wc_ntl_prox_GNN': 'wc_ntl_prox_gnn_demand',
}

# Fixed bootstrap seeds (same per-metric offset convention as used in the
# r216 experiment)
BOOT_SEEDS = {'rmse': 214001, 'mae': 214002, 'corr': 214003}
NEAR_ET = ('agent', 'near', 'agent')


# ════════════════════════════════════════════════════════════
# Utility: import 005 via importlib (005 has a numeric filename prefix, so
# it must be loaded by file path rather than a normal import statement)
# ════════════════════════════════════════════════════════════

def import_kfold005():
    """Import the 005 module by file path.

    At module level, 005 contains only imports and constant definitions
    (all the heavy lifting lives inside __main__ / function bodies), so
    calling exec_module here does not trigger any data loading or training.
    """
    spec = importlib.util.spec_from_file_location('kfold005_r214', SRC_005)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ════════════════════════════════════════════════════════════
# Utility: static source-code scanning (the disclosure table's 'source'
# field is file:line, with line numbers located dynamically)
# ════════════════════════════════════════════════════════════

def _read_lines(path: Path):
    """Read a source file into a list of lines (use lines[i-1] for 1-based indexing)."""
    return path.read_text(encoding='utf-8').splitlines()


def _find_lines(lines, pattern: str):
    """Return all 1-based line numbers matching pattern."""
    rx = re.compile(pattern)
    return [i + 1 for i, ln in enumerate(lines) if rx.search(ln)]


def _find_block_span(lines, start_pattern: str, end_pattern: str):
    """Locate the line-number span of a block-style constant: search
    forward from start_pattern for the first line matching end_pattern."""
    starts = _find_lines(lines, start_pattern)
    if len(starts) != 1:
        raise RuntimeError(f'Block start line is not unique: {start_pattern} -> {starts}')
    start = starts[0]
    rx_end = re.compile(end_pattern)
    for i in range(start, len(lines)):
        if rx_end.search(lines[i]):
            return start, i + 1
    raise RuntimeError(f'Could not find block end line: {end_pattern} (starting from line {start})')


def _enclosing_def(lines, lineno: int):
    """Return the name of the nearest enclosing def for 1-based line number lineno."""
    rx = re.compile(r'^\s*def\s+(\w+)\s*\(')
    for i in range(lineno - 1, -1, -1):
        mt = rx.match(lines[i])
        if mt:
            return mt.group(1)
    return None


def _src(path: Path, spec: str) -> str:
    """Build the 'source' field: path relative to the repo root plus a line-number annotation."""
    return f'{path.relative_to(PROJECT_ROOT).as_posix()}:{spec}'


# ════════════════════════════════════════════════════════════
# Utility: empirical graph-cache inspection
# ════════════════════════════════════════════════════════════

def _inspect_graph_cache(cache_file: Path, agent_feature_cols) -> dict:
    """Load a graph cache and extract every empirical fact needed by the
    disclosure table.

    Returns (facts_dict, cached). `cached` is returned so the caller can
    reuse it (for the AgentConn downstream evaluation); when it is not
    needed, the caller is responsible for deleting it and calling
    gc.collect().
    """
    with open(cache_file, 'rb') as f:
        cached = pickle.load(f)
    graphs = cached['graphs']

    locs = list(graphs.keys())
    g0 = graphs[locs[0]]
    node_types = list(g0.node_types)
    edge_types = [list(et) for et in g0.edge_types]
    agent_dim = int(g0['agent'].x.shape[1])

    # Agent feature names: read directly from the graph object's
    # feature_mapping_a (the mapping recorded at graph-construction time)
    mapping_a = getattr(g0, 'feature_mapping_a', None)
    feature_names_in_x = list(mapping_a.keys()) if mapping_a else None

    # Auxiliary attributes (agent attributes stored separately from x)
    agent_attrs = sorted(g0['agent'].keys())
    aux_attrs = [a for a in agent_attrs if a != 'x']

    near_by_graph = {}
    star_ok = True
    n_agents_total = 0
    for loc in locs:
        g = graphs[loc]
        near_by_graph[loc] = (int(g[NEAR_ET].edge_index.shape[1])
                              if NEAR_ET in g.edge_types else 0)
        # Star-topology membership structure: number of source->agent edges
        # == number of agents, and each agent appears exactly once
        ei = g['source', 'connects_to', 'agent'].edge_index
        n_a = int(g['agent'].num_nodes)
        n_agents_total += n_a
        agents_in_edges = ei[1].numpy()
        if not (ei.shape[1] == n_a and len(np.unique(agents_in_edges)) == n_a):
            star_ok = False
        # All graphs should share an identical schema
        if [list(et) for et in g.edge_types] != edge_types:
            raise RuntimeError(f'{cache_file.name}: {loc} has edge types inconsistent with the first graph')
        if int(g['agent'].x.shape[1]) != agent_dim:
            raise RuntimeError(f'{cache_file.name}: {loc} has an agent feature dimensionality inconsistent with the first graph')

    facts = {
        'n_graphs': len(locs),
        'locations': locs,
        'node_types': node_types,
        'edge_types': edge_types,
        'agent_feature_dim': agent_dim,
        'agent_feature_names_in_x': feature_names_in_x,
        'agent_feature_names_match_005_cols': feature_names_in_x == list(agent_feature_cols),
        'agent_aux_attrs_not_in_x': aux_attrs,
        'aux_attr_presence': {
            a: all(a in graphs[loc]['agent'].keys() for loc in locs)
            for a in ['coords', 'ntl_values', 'proximity_scores', 'rci_mask']
        },
        'near_edge_total': int(sum(near_by_graph.values())),
        'near_edge_by_graph': near_by_graph,
        'star_membership_structure_all_graphs': bool(star_ok),
        'n_agents_total': int(n_agents_total),
    }
    return facts, cached


# ════════════════════════════════════════════════════════════
# Part 1: disclosure table
# ════════════════════════════════════════════════════════════

def build_disclosure(m005, keep_agentconn_cache: bool):
    """Machine-generate the input-channel disclosure table. Each entry is
    a {value, source} pair.

    Returns (disclosure_dict, agentconn_cached_or_None).
    """
    lines_005 = _read_lines(SRC_005)
    lines_gb = _read_lines(SRC_GRAPHBUILDER)
    lines_sv = _read_lines(SRC_SOLVER)
    lines_ntl = _read_lines(SRC_NTL_LOSS)
    lines_prox = _read_lines(SRC_PROX_LOSS)

    items = {}

    # ── 1. AGENT_FEATURE_COLS (value read via importlib + line span located statically) ──
    span = _find_block_span(lines_005, r'^AGENT_FEATURE_COLS\s*=\s*\[', r'\]')
    items['agent_feature_cols'] = {
        'value': {
            'columns': list(m005.AGENT_FEATURE_COLS),
            'count': len(m005.AGENT_FEATURE_COLS),
            'all_landuse_proportions': all(
                c.startswith('lu_') and c.endswith('_prop')
                for c in m005.AGENT_FEATURE_COLS),
            'contains_ntl': any('ntl' in c.lower() for c in m005.AGENT_FEATURE_COLS),
            'contains_coordinates': any(
                k in c.lower() for c in m005.AGENT_FEATURE_COLS
                for k in ('coord', '_x', '_y', 'lat', 'lon')),
        },
        'source': _src(SRC_005, f'L{span[0]}-{span[1]} (importlib module attribute + regex-located line span)'),
    }

    # ── 2. AGENT_CONNECTIVITY ──
    ln = _find_lines(lines_005, r'^AGENT_CONNECTIVITY\s*=')
    if len(ln) != 1:
        raise RuntimeError(f'AGENT_CONNECTIVITY definition line is not unique: {ln}')
    items['agent_connectivity'] = {
        'value': m005.AGENT_CONNECTIVITY,
        'source': _src(SRC_005, f'L{ln[0]} (importlib module attribute)'),
    }

    # ── 3. Loss sets for each CONFIG_MAP configuration ──
    span = _find_block_span(lines_005, r'^CONFIG_MAP\s*=\s*\{', r'^\}')
    items['config_map_losses'] = {
        'value': {
            name: {
                'losses': sorted(cfg['objective_weights'].keys()),
                'objective_weights': dict(cfg['objective_weights']),
                'epochs': cfg['epochs'],
            }
            for name, cfg in m005.CONFIG_MAP.items()
        },
        'source': _src(SRC_005, f'L{span[0]}-{span[1]} (importlib module attribute + regex-located line span)'),
    }

    # ── 4. Two occurrences of the initial tau value (located via static
    #      regex scanning, since values inside function bodies cannot be
    #      read as module attributes) ──
    rx_tau = re.compile(r'allocation_temperature_start\s*=\s*([0-9eE.+-]+)')
    occurrences = []
    for i, text in enumerate(lines_005):
        mt = rx_tau.search(text)
        if mt:
            occurrences.append({'line': i + 1, 'value': float(mt.group(1))})
    tau_values = sorted({o['value'] for o in occurrences})
    items['tau_initial'] = {
        'value': {
            'occurrences': occurrences,
            'n_occurrences': len(occurrences),
            'all_equal': len(tau_values) == 1,
            'unique_value': tau_values[0] if len(tau_values) == 1 else tau_values,
        },
        'source': _src(SRC_005, 'L' + ','.join(str(o['line']) for o in occurrences)
                       + ' (static regex scan for allocation_temperature_start)'),
    }

    # ── 5. GraphBuilder: geometry is dropped before featurization ──
    ln = _find_lines(lines_gb, r"drop\(columns='geometry'")
    if len(ln) != 1:
        raise RuntimeError(f'GraphBuilder geometry-drop line is not unique: {ln}')
    fn = _enclosing_def(lines_gb, ln[0])
    if fn != 'preprocess_features':
        raise RuntimeError(f'geometry drop is not inside preprocess_features (found inside {fn} instead)')
    items['graphbuilder_geometry_drop'] = {
        'value': {
            'line_text': lines_gb[ln[0] - 1].strip(),
            'function': fn,
            'meaning': 'The geometry column is dropped in the very first featurization step -- coordinates cannot possibly enter the node feature matrix x',
        },
        'source': _src(SRC_GRAPHBUILDER, f'L{ln[0]} (static regex scan + enclosing-function verification)'),
    }

    # ── 6. GraphBuilder: agent.x comes from processed features; coords is a separate attribute ──
    ln_x = _find_lines(lines_gb, r"data\['agent'\]\.x\s*=")
    ln_c = _find_lines(lines_gb, r"data\['agent'\]\.coords\s*=")
    if len(ln_x) != 1 or len(ln_c) != 1:
        raise RuntimeError(f'agent.x / agent.coords assignment lines are not unique: {ln_x}, {ln_c}')
    items['agent_coords_separate_attribute'] = {
        'value': {
            'x_line_text': lines_gb[ln_x[0] - 1].strip(),
            'coords_line_text': lines_gb[ln_c[0] - 1].strip(),
            'meaning': 'agent.x holds only the output of preprocess_features (geometry has already been '
                       'dropped); coordinates are stored separately as the agent.coords auxiliary attribute '
                       'and are not part of the feature matrix',
        },
        'source': _src(SRC_GRAPHBUILDER, f'L{ln_x[0]},L{ln_c[0]}'),
    }

    # ── 7. EdgeWeightSolver: the encoder forward pass consumes only x_dict + edge_index_dict ──
    ln_enc = _find_lines(lines_sv, r'self\.encoder\(\s*\w+\.x_dict,\s*\w+\.edge_index_dict\s*\)')
    if len(ln_enc) < 2:
        raise RuntimeError(f'Fewer than 2 encoder forward-call lines found (expected training + inference): {ln_enc}')
    items['encoder_forward_inputs'] = {
        'value': {
            'call_lines': ln_enc,
            'call_texts': [lines_sv[i - 1].strip() for i in ln_enc],
            'meaning': 'The encoder forward pass (both training and inference) receives only x_dict and '
                       'edge_index_dict; coords / ntl_values / proximity_scores / rci_mask never reach the '
                       'forward pass',
        },
        'source': _src(SRC_SOLVER, 'L' + ','.join(map(str, ln_enc)) + ' (static regex scan)'),
    }

    # ── 8./9. The only entry point for NTL and Proximity is the prior loss ──
    def _prior_item(lines_loss, src_path, registry_key):
        ln_reg = _find_lines(lines_loss, rf'"{registry_key}"')
        configs_with = sorted(
            name for name, cfg in m005.CONFIG_MAP.items()
            if registry_key in cfg['objective_weights'])
        return {
            'value': {
                'in_agent_feature_cols': any(
                    registry_key.split('_')[0] in c.lower()
                    for c in m005.AGENT_FEATURE_COLS),
                'loss_registry_key': registry_key,
                'configs_with_this_prior': configs_with,
                'meaning': f'The {registry_key} signal is not part of the input features; it only affects '
                           f'training gradients via the prior loss under the {configs_with} configuration(s), '
                           f'and is not used at all during inference',
            },
            'source': (_src(src_path, f'L{ln_reg[0]} (loss_registry registration key)')
                       + ' + ' + _src(SRC_005, 'CONFIG_MAP (importlib)')),
        }

    items['ntl_pathway'] = _prior_item(lines_ntl, SRC_NTL_LOSS, 'ntl_prior')
    items['proximity_pathway'] = _prior_item(lines_prox, SRC_PROX_LOSS, 'proximity_prior')

    # ── 10. Empirical inspection of the main graph_cache (load -> read -> release immediately) ──
    print('Loading the main graph_cache (~437MB)...')
    main_facts, main_cached = _inspect_graph_cache(MAIN_CACHE, m005.AGENT_FEATURE_COLS)
    del main_cached
    gc.collect()
    items['main_graph_cache'] = {
        'value': main_facts,
        'source': str(MAIN_CACHE.relative_to(PROJECT_ROOT).as_posix()) + ' (inspected graph-by-graph)',
    }

    # ── 11. Empirical inspection of the AgentConn graph_cache (optionally
    #        kept in memory for reuse by the downstream evaluation) ──
    print('Loading the AgentConn graph_cache (~440MB)...')
    ac_facts, ac_cached = _inspect_graph_cache(AGENTCONN_CACHE, m005.AGENT_FEATURE_COLS)
    items['agentconn_graph_cache'] = {
        'value': ac_facts,
        'source': str(AGENTCONN_CACHE.relative_to(PROJECT_ROOT).as_posix()) + ' (inspected graph-by-graph)',
    }
    if not keep_agentconn_cache:
        del ac_cached
        gc.collect()
        ac_cached = None

    disclosure = {
        'meta': {
            'script': '029_eval_agentconn_downstream.py',
            'purpose': 'Machine-generated input-channel disclosure table (every fact is read '
                       'programmatically from source; nothing is hand-transcribed)',
            'generated_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            'python': sys.version.split()[0],
            'torch': torch.__version__,
            'numpy': np.__version__,
            'generation_methods': [
                'Import 005 via importlib and read its module attributes',
                'Static regex scanning of source code to locate line numbers (computed dynamically, never hand-transcribed)',
                'Empirical, graph-by-graph inspection of the cached pickle files',
            ],
        },
        'items': items,
    }
    return disclosure, ac_cached


# ════════════════════════════════════════════════════════════
# Part 2: AgentConn downstream evaluation
# ════════════════════════════════════════════════════════════

def _make_solver(model_path: Path, config_name: str, m005, graphs, train_locs):
    """Initialize the model architecture on CPU using the same
    hyperparameters as 005, and load the frozen checkpoint."""
    epochs = m005.CONFIG_MAP[config_name]['epochs']
    objective_weights = m005.CONFIG_MAP[config_name]['objective_weights']
    warmup_epochs, decay_epochs = 20, 20
    config = ModelConfig(
        epochs=epochs,
        hidden_dim=256,
        embedding_dim=128,
        num_layers=3,
        conv_type='hgt',
        allocation_temperature_start=0.01,
        learning_rate=1e-3,
        weight_decay=1e-4,
        use_scheduler=True,
        warmup_epochs=warmup_epochs,
        decay_epochs=decay_epochs,
        cosine_epochs=epochs - warmup_epochs - decay_epochs,
        cosine_eta_min=1e-5,
        learnable=False,
        save_path=str(model_path),
        device='cpu',                      # Force CPU (the GPU is concurrently occupied training other model configurations)
    )
    solver = EdgeWeightSolver(config)
    dl = DataLoader([graphs[loc] for loc in train_locs], batch_size=1, shuffle=False)
    solver.init_model(dl, objective_weights)
    solver._load_checkpoint()
    return solver


def _predict_region_metrics(loc, solver, m005, cached,
                            assignment_cache, prox_cache):
    """Single region: inference -> seven demand columns (matching 005) ->
    Voronoi aggregation across three metrics.

    Demand-column construction mirrors the first half of
    005.predict_and_evaluate_location line for line (following the same
    approach used elsewhere in this suite); NTL/Proximity correction calls
    005's functions directly (borrowed via importlib); aggregation uses the
    shared module's assignment-cache split (compute_voronoi_assignment +
    aggregate_by_assignment).
    """
    graphs = cached['graphs']
    grids = cached['grids']
    ntl_dict = cached['ntl_dict']
    region_dict = cached['region_dict']
    subs_dict = cached['subs_dict']

    grid_gdf, _step = grids[loc]
    region_sub = region_dict[loc]
    subs_sub = subs_dict[loc]
    graph = graphs[loc]
    ntl_values = ntl_dict[loc]

    edge_weights_df = solver.predict_edge_weights(graph)

    region_info = region_sub.set_index('ITL3')
    source_index_map = graph.source_index_map

    grid_gdf = grid_gdf.copy()
    grid_gdf['gnn_demand'] = 0.0
    for _, row in edge_weights_df.iterrows():
        s_idx = int(row['source_node_idx'])
        a_orig_idx = int(row['agent_original_idx'])
        w = row['predicted_weight']
        s_orig_idx = source_index_map.iloc[s_idx]
        itl3 = region_sub.loc[s_orig_idx, 'ITL3']
        total_demand = region_info.loc[itl3, 'Demand (MVA)']
        grid_gdf.loc[a_orig_idx, 'gnn_demand'] += w * total_demand

    # WC correction (same as 005)
    grid_gdf['wc_gnn_demand'] = grid_gdf['gnn_demand'].copy()
    wc_weight = 1.0 - grid_gdf['wc_others_ratio'].values
    grid_gdf['wc_gnn_demand'] *= wc_weight
    for itl3, group in grid_gdf.groupby('ITL3'):
        if itl3 not in region_info.index:
            continue
        idx = group.index
        total_demand = region_info.loc[itl3, 'Demand (MVA)']
        wc_sum = grid_gdf.loc[idx, 'wc_gnn_demand'].sum()
        if wc_sum > 0:
            grid_gdf.loc[idx, 'wc_gnn_demand'] *= total_demand / wc_sum
        else:
            grid_gdf.loc[idx, 'wc_gnn_demand'] = total_demand / len(group)

    # NTL / Proximity / stacked correction (reuses 005's functions
    # directly; proximity scores are cached per region)
    if loc not in prox_cache:
        prox_cache[loc] = m005.compute_proximity_scores(grid_gdf, subs_sub)
    prox_scores = prox_cache[loc]

    m005.compute_ntl_corrected_demand(
        grid_gdf, region_sub, 'gnn_demand', ntl_values, 'ntl_gnn_demand')
    m005.compute_proximity_corrected_demand(
        grid_gdf, region_sub, 'gnn_demand', prox_scores, 'prox_gnn_demand')
    m005.compute_proximity_corrected_demand(
        grid_gdf, region_sub, 'ntl_gnn_demand', prox_scores, 'ntl_prox_gnn_demand')
    m005.compute_ntl_corrected_demand(
        grid_gdf, region_sub, 'wc_gnn_demand', ntl_values, 'wc_ntl_gnn_demand')
    m005.compute_proximity_corrected_demand(
        grid_gdf, region_sub, 'wc_ntl_gnn_demand', prox_scores, 'wc_ntl_prox_gnn_demand')

    # Voronoi aggregation (shared-module assignment-cache split -- computed once per region)
    assignment = scu.compute_voronoi_assignment(
        grid_gdf, subs_sub, cache=assignment_cache, cache_key=loc)

    loc_metrics = {}
    for method, col in VORONOI_METHODS.items():
        subs_res = scu.aggregate_by_assignment(subs_sub, assignment,
                                               grid_gdf[col].values)
        loc_metrics[method] = scu.evaluate_allocation(subs_res)
    return loc_metrics


def _verify_splits_against_exp0(all_locations):
    """The KFold split (random_state=seed) must match exp0's frozen kfold_splits.json."""
    location_array = np.array(all_locations)
    verification = {}
    splits_by_seed = {}
    for seed in SEEDS:
        kf_splits = list(KFold(n_splits=N_FOLDS, shuffle=True,
                               random_state=seed).split(location_array))
        computed = {
            f'fold_{i + 1}': {
                'train': location_array[tr].tolist(),
                'test': location_array[te].tolist(),
            } for i, (tr, te) in enumerate(kf_splits)
        }
        frozen_path = EXP0_DIR / f'seed_{seed}' / 'baseline' / 'kfold_splits.json'
        with open(frozen_path, encoding='utf-8') as f:
            frozen = json.load(f)
        verification[str(seed)] = (computed == frozen)
        splits_by_seed[seed] = computed
    if not all(verification.values()):
        raise RuntimeError(f"KFold split does not match exp0's frozen splits: {verification}")
    return splits_by_seed, verification


def run_agentconn_downstream(m005, cached):
    """Downstream kfold_test evaluation for the 48 AgentConn models,
    compared against the main results, with significance testing."""
    DOWNSTREAM_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    all_locations = list(m005.ALL_LOCATIONS)
    splits_by_seed, splits_verified = _verify_splits_against_exp0(all_locations)
    print('KFold split verification passed (all 3 seeds match exp0\'s frozen splits)')

    assignment_cache = {}
    prox_cache = {}
    failed_models = []

    # ── Run inference over test regions for each seed x config x fold ──
    for seed in SEEDS:
        for config_name in CONFIGS:
            combo_t0 = time.time()
            test_by_metric = {m: {mname: {} for mname in VORONOI_METHODS}
                              for m in METRICS}
            col_order = []                                            # Column order follows 005's fold ordering

            for fold_i in range(1, N_FOLDS + 1):
                fold_key = f'fold_{fold_i}'
                train_locs = splits_by_seed[seed][fold_key]['train']
                test_locs = splits_by_seed[seed][fold_key]['test']
                model_path = (AGENTCONN_DIR / f'seed_{seed}' / config_name
                              / f'fold{fold_i}' / 'model.pth')
                try:
                    solver = _make_solver(model_path, config_name, m005,
                                          cached['graphs'], train_locs)
                except Exception as exc:  # noqa: BLE001 -- record failed models honestly rather than masking the error
                    failed_models.append({
                        'model': str(model_path.relative_to(PROJECT_ROOT)),
                        'error': f'{type(exc).__name__}: {exc}',
                    })
                    print(f'  !! Failed to load model {model_path}: {exc}')
                    for loc in test_locs:
                        col_order.append(loc)
                    continue

                for loc in test_locs:
                    loc_metrics = _predict_region_metrics(
                        loc, solver, m005, cached, assignment_cache, prox_cache)
                    col_order.append(loc)
                    for mname, mm in loc_metrics.items():
                        # Round to 4 decimal places, matching 005 (consistent with the frozen csv precision)
                        test_by_metric['rmse'][mname][loc] = round(mm['rmse'], 4)
                        test_by_metric['mae'][mname][loc] = round(mm['mae'], 4)
                        test_by_metric['corr'][mname][loc] = round(float(mm['corr']), 4)
                del solver
                gc.collect()

            # Write to disk (structured identically to exp0: rows =
            # methods in lexicographic order, columns = fold-ordered
            # regions + mean)
            out_seed_dir = DOWNSTREAM_DIR / f'seed_{seed}' / config_name
            out_seed_dir.mkdir(parents=True, exist_ok=True)
            for metric in METRICS:
                df = pd.DataFrame(test_by_metric[metric]).T
                # reindex: for regions whose model failed to load, leave the column as NaN honestly (already recorded in run_meta)
                df = df.reindex(index=sorted(VORONOI_METHODS.keys()), columns=col_order)
                df['mean'] = df.mean(axis=1).round(4)
                df.to_csv(out_seed_dir / f'kfold_test_{metric}.csv')
            print(f'seed_{seed}/{config_name}: completed 16 regions x 6 methods '
                  f'({time.time() - combo_t0:.0f}s)')

    # ── Build the long-format side-by-side comparison table against the main results ──
    rows = []
    for seed in SEEDS:
        for config_name in CONFIGS:
            for metric in METRICS:
                ac_df = pd.read_csv(DOWNSTREAM_DIR / f'seed_{seed}' / config_name
                                    / f'kfold_test_{metric}.csv', index_col=0)
                main_df = pd.read_csv(EXP0_DIR / f'seed_{seed}' / config_name
                                      / f'kfold_test_{metric}.csv', index_col=0)
                for method in sorted(VORONOI_METHODS.keys()):
                    for loc in all_locations:
                        rows.append({
                            'seed': seed, 'config': config_name, 'metric': metric,
                            'method': method, 'location': loc,
                            'agentconn': float(ac_df.loc[method, loc]),
                            'main': float(main_df.loc[method, loc]),
                        })
    long_df = pd.DataFrame(rows)
    long_df['diff'] = long_df['agentconn'] - long_df['main']

    # Seed-averaged comparison table (average over seeds first, then compare by region)
    agg = (long_df
           .groupby(['config', 'metric', 'method', 'location'], as_index=False)
           .agg(agentconn_mean=('agentconn', 'mean'),
                main_mean=('main', 'mean'),
                diff_mean=('diff', 'mean')))
    for seed in SEEDS:
        sub = long_df[long_df['seed'] == seed][
            ['config', 'metric', 'method', 'location', 'diff']]
        agg = agg.merge(sub.rename(columns={'diff': f'diff_seed{seed}'}),
                        on=['config', 'metric', 'method', 'location'], how='left')
    agg.to_csv(DOWNSTREAM_DIR / 'agentconn_vs_main.csv', index=False)

    # ── Significance testing: AgentConn baseline vs. main baseline (baseline config / voronoi_GNN) ──
    test_json = {'meta': {
        'protocol': 'Average over seeds -> compute paired differences across the 16 regions -> exact '
                    'sign-flip permutation test (2^16 exhaustive); CI is a paired, region-level bootstrap '
                    '(outer layer only); per-seed p-values are all reported as-is, plus a Cauchy combination '
                    '(the median is not used)',
        'comparison': 'AgentConn baseline (voronoi_GNN) vs. main baseline exp0 (voronoi_GNN)',
        'diff_convention': 'diff = AgentConn - main (a positive value means AgentConn is worse for '
                           'RMSE/MAE; a positive value means AgentConn is better for Corr)',
        'config': 'baseline', 'method': 'voronoi_GNN',
        'n_regions': len(all_locations),
    }, 'comparisons': {}}

    for metric in METRICS:
        sub = agg[(agg['config'] == 'baseline') & (agg['metric'] == metric)
                  & (agg['method'] == 'voronoi_GNN')].set_index('location')
        sub = sub.loc[all_locations]                    # Fix the region ordering
        diffs = sub['diff_mean'].to_numpy(dtype=float)  # 16 seed-averaged paired differences

        if not np.all(np.isfinite(diffs)):
            # NaN values caused by failed models -- honestly record the degraded status rather than papering over it
            test_json['comparisons'][metric] = {
                'status': 'skipped_due_to_failed_models',
                'n_nan_regions': int(np.sum(~np.isfinite(diffs))),
            }
            continue

        perm = exact_sign_flip_permutation(diffs)
        boot = paired_region_bootstrap(diffs, B=10000, seed=BOOT_SEEDS[metric])

        per_seed = {}
        for seed in SEEDS:
            d_s = sub[f'diff_seed{seed}'].to_numpy(dtype=float)
            per_seed[str(seed)] = exact_sign_flip_permutation(d_s)
        cauchy_p = cauchy_combination([v['p'] for v in per_seed.values()])

        test_json['comparisons'][metric] = {
            'mean_diff': float(np.mean(diffs)),
            'perm_p': perm['p'],
            'perm_t_obs': perm['t_obs'],
            'perm_min_attainable_p': perm['min_attainable_p'],
            'ci_lo': boot['ci_lo'], 'ci_hi': boot['ci_hi'],
            'boot_B': boot['B'], 'boot_seed': boot['seed'],
            'per_seed_perm_p': {s: v['p'] for s, v in per_seed.items()},
            'cauchy_combined_p': cauchy_p,
            'significant_at_0.05': bool(perm['p'] < 0.05),
        }

    # All conclusion fields are generated by deterministic rules from the numbers (never hand-written)
    rmse_c = test_json['comparisons']['rmse']
    degraded = 'mean_diff' not in rmse_c
    if degraded:
        test_json['conclusion'] = {'status': 'degraded_test_skipped',
                                   'reason': 'One or more models failed to load, so the significance test was skipped (see run_meta)'}
    else:
        direction = ('AgentConn downstream RMSE is worse (diff > 0)' if rmse_c['mean_diff'] > 0
                     else 'AgentConn downstream RMSE is better (diff < 0)')
        any_sig = any(test_json['comparisons'][m].get('significant_at_0.05', False)
                      for m in METRICS)
        test_json['conclusion'] = {
            'headline_metric': 'rmse',
            'direction': direction,
            'rmse_mean_diff': rmse_c['mean_diff'],
            'rmse_perm_p': rmse_c['perm_p'],
            'any_metric_significant_at_0.05': any_sig,
            'val_loss_reference': 'agent_conn_ablation_report.md: no significant difference in '
                                  'validation loss during training (all p > 0.3)',
            'consistent_with_val_loss_no_difference': not any_sig,
            'paper_action': ('reportable' if not any_sig else 'internal_record_only'),
            'paper_action_rule': 'Decision rule: if the AgentConn downstream result contradicts the '
                                 'val-loss finding (i.e., a significant difference is found), it is kept '
                                 'as an internal record only and not included in the paper',
        }
    with open(DOWNSTREAM_DIR / 'agentconn_vs_main_test.json', 'w', encoding='utf-8') as f:
        json.dump(test_json, f, ensure_ascii=False, indent=2)

    # ── Run metadata ──
    run_meta = {
        'script': '029_eval_agentconn_downstream.py',
        'generated_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'device': 'cpu',
        'cpu_only': True,
        'torch': torch.__version__,
        'n_models_expected': len(SEEDS) * len(CONFIGS) * N_FOLDS,
        'n_models_failed': len(failed_models),
        'failed_models': failed_models,
        'splits_verified_against_exp0': splits_verified,
        'civd_skipped': 'Skipped, following the same precedent as 020 -- only the voronoi_* metrics '
                        'are evaluated (see the module docstring for the rationale)',
        'grid_demands_not_saved': True,
        'elapsed_seconds': round(time.time() - t_start, 1),
    }
    with open(DOWNSTREAM_DIR / 'run_meta.json', 'w', encoding='utf-8') as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)

    print(f'\nAgentConn downstream evaluation complete: {DOWNSTREAM_DIR}')
    print(f'  Failed models: {len(failed_models)} | Total time: {run_meta["elapsed_seconds"]}s')
    if degraded:
        print('  Warning: significance test skipped (failed models), see run_meta.json for details')
        return run_meta
    for metric in METRICS:
        c = test_json['comparisons'][metric]
        print(f'  [{metric}] mean_diff={c["mean_diff"]:+.4f}  perm_p={c["perm_p"]:.4f}  '
              f'CI=[{c["ci_lo"]:+.4f},{c["ci_hi"]:+.4f}]  '
              f'per-seed p={list(c["per_seed_perm_p"].values())}')
    print(f'  Conclusion: {test_json["conclusion"]["direction"]} | '
          f'Consistent with the no-difference val-loss finding: '
          f'{test_json["conclusion"]["consistent_with_val_loss_no_difference"]} | '
          f'paper_action={test_json["conclusion"]["paper_action"]}')
    return run_meta


# ════════════════════════════════════════════════════════════
# Main entry point
# ════════════════════════════════════════════════════════════

def main() -> int:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    parser = argparse.ArgumentParser(description='Input-channel disclosure table + AgentConn downstream evaluation')
    parser.add_argument('--disclosure-only', action='store_true',
                        help='Generate only the disclosure table, skipping the AgentConn downstream evaluation')
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print('Importing the 005 module (via importlib, following the same approach as 020)...')
    m005 = import_kfold005()

    # Part 1: disclosure table (always run)
    disclosure, ac_cached = build_disclosure(
        m005, keep_agentconn_cache=not args.disclosure_only)
    with open(DISCLOSURE_JSON, 'w', encoding='utf-8') as f:
        json.dump(disclosure, f, ensure_ascii=False, indent=2)
    print(f'\nDisclosure table written to: {DISCLOSURE_JSON}')
    for name, item in disclosure['items'].items():
        print(f'  {name:<36s} source={item["source"][:80]}')

    # Part 2: AgentConn downstream evaluation (optional)
    if not args.disclosure_only:
        print('\nStarting AgentConn downstream evaluation (48 models x their respective fold test regions, CPU only)...')
        run_agentconn_downstream(m005, ac_cached)

    print(f'\nAll done, total elapsed time {time.time() - t0:.0f}s')
    return 0


if __name__ == '__main__':
    sys.exit(main())
