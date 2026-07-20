"""
032 - Training entry point for the tau initialization sweep (optional, intended
for idle overnight GPU windows)

Copied from 027_train_feature_fusion.py (027 already carries the walk-up
repo-root resolution fix and the Windows/Agg compatibility handling, so this
does not start from 005), adapted into a **tau (allocation_temperature_start)
sweep entry point for the original paper configuration**:

=== Scientific difference from 027 (this sweep is not the fusion arm) ===
This script sweeps tau on the **original 5-dim paper configuration**, not the
fusion arm:
- AGENT_FEATURE_COLS is restored to the original UK 5-dim land-use columns
  (no ntl_feat/prox_feat);
- the graph cache **directly reuses the exp0 primary cache**
  (results/exp0_kfold_prior/graph_cache/cached_graphs.pickle, 5-dim agent
  features) -- read-only; this script **does not build any new graph
  cache** and contains no build_graphs/load_data path (a missing cache
  raises immediately, pointing back to the upstream cache-presence check);
- apart from tau, the training configuration is identical to 005/exp0
  baseline (hgt / hidden 256 / embedding 128 / 3 layers / epochs 200 /
  lr 1e-3 / learnable=False).

=== tau parameterization (guarding against the two ModelConfig sites drifting apart) ===
tau appears in two separate ModelConfig constructions copied from 005 (the
resume/inference-only branch and the normal-training branch) -- in this
script, both are driven by the single tau_start parameter of
run_kfold_training, which in turn is supplied only by the CLI
--tau-start flag, preventing the two sites from drifting apart.

=== Tau values and output layout ===
tau in {0.005, 0.01, 0.05, 0.5, 2.0}, each with its own output directory:
    results/exp_r213_tau_sweep/tau_{tag}/seed_{seed}/baseline/fold*/...
tag = the tau value with its decimal point replaced by an underscore
(following the existing repo naming convention seen in
exp0_kfold_prior_0_01):
    0.005->tau_0_005  0.01->tau_0_01  0.05->tau_0_05  0.5->tau_0_5  2.0->tau_2_0
The output structure within each directory matches a single exp0
(seed, config) run exactly (fold*/{model.pth,rmse.csv,mae.csv,corr.csv,
grid_demands/} + kfold_test_*.csv + kfold_splits.json), plus a
run_manifest.json recording the tau/seed/cache path/anchor convention.

=== Tolerance anchor for the tau=0.01 rung (GPU retraining is not bit-for-bit
reproducible, hence a tolerance range rather than an exact match) ===
Once the tau=0.01 rung (= the exp0 training value) finishes, the mean
voronoi_GNN value in that seed's kfold_test_rmse.csv should fall within the
range spanned by the original exp0 baseline 3-seed run:
    exp0 baseline voronoi_GNN mean RMSE (seed 42/123/456) = 9.1647 / 9.4921 / 9.1500
    -> tolerance-anchor range [9.1500, 9.4921] (read from the frozen
       kfold_test_rmse.csv on 2026-07-14)
Falling inside the range means the training pipeline is consistent with
exp0, so differences seen at the other four tau values can be attributed to
tau; falling outside the range means the pipeline should be re-checked
(cache, fold split, config) before trusting the sweep results.

=== Budget and execution notes ===
5 tau values x 1 seed (default 42) x 4 folds = 20 training runs, roughly
7 GPU-hours -- **only run this during an idle overnight GPU window** (this
sweep is lower priority than the fusion-arm training and should not be
started during the day). The headless environment uses MPLBACKEND=Agg
(inherited from 027's compatibility handling). Recommended execution order
(one tau value per command, run sequentially):
    1) tau 0.01   -- run the anchor rung first, to validate the pipeline against exp0
    2) tau 2.0    -- the most flattened extreme (the most informative contrast direction)
    3) tau 0.005  -- the sharpest extreme
    4) tau 0.05   -- intermediate value
    5) tau 0.5    -- intermediate value

Usage:
    python 032_train_tau_sweep.py --help
    python 032_train_tau_sweep.py --tau-start 0.01 --dry-run     # dry run: validates
                                                                  # args/paths and exits before loading the cache
    python 032_train_tau_sweep.py --tau-start 0.01               # seed defaults to 42
    python 032_train_tau_sweep.py --tau-start 2.0 --seed 42

Config mapping:
    baseline:   {'landuse_prediction_loss': 1.0}   (this sweep only covers the baseline config)
"""

import sys
import argparse
import pickle
import warnings
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Repository root: walk up until a directory containing SpatialAllocation is
# found (the same approach used in 020/027; counting parents like 005 does
# would point outside the repo)
SCRIPT_DIR = Path(__file__).resolve().parent
_p = SCRIPT_DIR
while not (_p / 'SpatialAllocation').exists():
    if _p.parent == _p:
        raise RuntimeError('Could not find the repository root (the SpatialAllocation package)')
    _p = _p.parent
PROJECT_ROOT = _p
sys.path.insert(0, str(PROJECT_ROOT))

from SpatialAllocation.GNN.core.EdgeWeightSolver import EdgeWeightSolver
from SpatialAllocation.GNN.core.ModelConfig import ModelConfig
from SpatialAllocation.Allocator import allocator_registry
from SpatialAllocation.Allocator.clustering.do_clustering import do_clustering

warnings.filterwarnings('ignore', category=FutureWarning)

# The default Windows console codepage (cp1252) cannot encode non-ASCII text --
# force UTF-8 (does not affect file outputs)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ════════════════════════════════════════════════════════════
# Path constants
# ════════════════════════════════════════════════════════════

DATA_DIR = SCRIPT_DIR / 'results' / 'intermediate'
STATIC_DIR = SCRIPT_DIR / 'results' / 'static_allocation'

# exp0 primary graph cache (frozen, read-only; its presence and 5-dim agent
# features have already been verified upstream) -- this script does not
# build a new cache; a missing cache raises immediately
FROZEN_EXP0_CACHE = (SCRIPT_DIR / 'results' / 'exp0_kfold_prior'
                     / 'graph_cache' / 'cached_graphs.pickle')

# tau sweep results root (each tau value gets its own subdirectory)
SWEEP_ROOT = SCRIPT_DIR / 'results' / 'exp_r213_tau_sweep'

N_FOLDS = 4

# --- tau values and their directory tags (decimal point -> underscore, existing repo naming convention) ---
TAU_CHOICES = [0.005, 0.01, 0.05, 0.5, 2.0]
TAU_TAGS = {0.005: '0_005', 0.01: '0_01', 0.05: '0_05', 0.5: '0_5', 2.0: '2_0'}

# tolerance anchor for the tau=0.01 rung (details in the module docstring;
# exp0 baseline voronoi_GNN mean, range across 3 seeds)
ANCHOR_TAU = 0.01
ANCHOR_RMSE_RANGE = (9.1500, 9.4921)

# --- region configuration ---
ALL_LOCATIONS = [
    'London',
    'TLH2', 'TLH3', 'TLJ1', 'TLF1', 'TLF2',
    'TLC1', 'TLC2', 'TLD6', 'TLG1', 'TLG2', 'TLE4',
    'TLH1', 'TLE3', 'TLD3', 'TLD4',
]

# --- feature configuration ---
# This sweep restores the original UK 5-dim land-use features (matching
# 005/exp0, no fusion columns)
AGENT_FEATURE_COLS = [
    'lu_residential_prop', 'lu_commercial_prop', 'lu_industrial_prop',
    'lu_agricultural_prop', 'lu_others_prop',
]

SOURCE_FEATURE_COLS = [
    'residential_percent', 'commercial_percent', 'industrial_percent',
    'agricultural_percent', 'others_percent',
]

# --- post-correction constants ---
RCI_THRESHOLD = 0.5
PROXIMITY_GAMMA = 2.0
DIST_CLAMP_KM = 0.01
TARGET_CRS = 'EPSG:27700'

# --- land-use mapping ---
LANDUSE_PERCENT_MAP = {
    'lu_residential_prop': 'residential_percent',
    'lu_commercial_prop': 'commercial_percent',
    'lu_industrial_prop': 'industrial_percent',
    'lu_agricultural_prop': 'agricultural_percent',
    'lu_others_prop': 'others_percent',
}
LU_COLS = list(LANDUSE_PERCENT_MAP.keys())
PCT_COLS = list(LANDUSE_PERCENT_MAP.values())

# --- config mapping ---
# This sweep only covers baseline (tau is swept against the original paper's
# main configuration; interactions between tau and the prior-loss configs
# are out of scope here)
CONFIG_MAP = {
    'baseline': {
        'objective_weights': {'landuse_prediction_loss': 1.0},
        'epochs': 200,
    },
}

AGENT_CONNECTIVITY = None


# ════════════════════════════════════════════════════════════
# Utility functions (evaluation and post-correction chain, identical to 027/005)
# ════════════════════════════════════════════════════════════

def compute_ntl_corrected_demand(grid_gdf, region_sub, base_demand_col,
                                  ntl_values, corrected_col):
    """NTL post-correction: base_demand x ntl_factor, renormalized per ITL3 to
    preserve demand conservation."""
    rci_sum = (grid_gdf['lu_residential_prop'].values
              + grid_gdf['lu_commercial_prop'].values
              + grid_gdf['lu_industrial_prop'].values)
    rci_mask = rci_sum > RCI_THRESHOLD

    grid_gdf[corrected_col] = 0.0
    region_info = region_sub.set_index('ITL3')

    for itl3, group in grid_gdf.groupby('ITL3'):
        if itl3 not in region_info.index:
            continue

        total_demand = region_info.loc[itl3, 'Demand (MVA)']
        idx = group.index
        ntl_group = ntl_values[idx]
        rci_group = rci_mask[idx]

        rci_nonzero_ntl = ntl_group[rci_group & (ntl_group > 0)]
        if len(rci_nonzero_ntl) > 0:
            epsilon = np.percentile(rci_nonzero_ntl, 5)
        else:
            nonzero_ntl = ntl_group[ntl_group > 0]
            epsilon = np.percentile(nonzero_ntl, 5) if len(nonzero_ntl) > 0 else 0.1

        rci_ntl = ntl_group[rci_group]
        if len(rci_ntl) > 0:
            ntl_median = np.median(rci_ntl)
        else:
            ntl_median = np.median(ntl_group)

        if ntl_median <= 0:
            ntl_median = epsilon

        ntl_factor = np.log(1 + ntl_group + epsilon) / np.log(1 + ntl_median)

        base = grid_gdf.loc[idx, base_demand_col].values
        raw = base * ntl_factor
        raw_sum = raw.sum()
        if raw_sum > 0:
            grid_gdf.loc[idx, corrected_col] = total_demand * raw / raw_sum
        else:
            grid_gdf.loc[idx, corrected_col] = total_demand / len(group)


def compute_proximity_scores(grid_gdf, subs_sub, gamma=PROXIMITY_GAMMA):
    """Compute the substation-proximity score for each grid point."""
    grid_proj = grid_gdf.to_crs(TARGET_CRS)
    subs_proj = subs_sub.to_crs(TARGET_CRS)

    grid_coords = np.column_stack([grid_proj.geometry.x.values,
                                    grid_proj.geometry.y.values])
    subs_coords = np.column_stack([subs_proj.geometry.x.values,
                                    subs_proj.geometry.y.values])

    dist_km = cdist(grid_coords, subs_coords, metric='euclidean') / 1000.0
    dist_km = np.maximum(dist_km, DIST_CLAMP_KM)
    proximity = np.sum(dist_km ** (-gamma), axis=1)
    return proximity


def compute_proximity_corrected_demand(grid_gdf, region_sub, base_demand_col,
                                        proximity_scores, corrected_col):
    """Proximity post-correction: base_demand x prox_factor, renormalized per ITL3
    to preserve demand conservation."""
    rci_sum = (grid_gdf['lu_residential_prop'].values
              + grid_gdf['lu_commercial_prop'].values
              + grid_gdf['lu_industrial_prop'].values)
    rci_mask = rci_sum > RCI_THRESHOLD

    grid_gdf[corrected_col] = 0.0
    region_info = region_sub.set_index('ITL3')

    for itl3, group in grid_gdf.groupby('ITL3'):
        if itl3 not in region_info.index:
            continue

        total_demand = region_info.loc[itl3, 'Demand (MVA)']
        idx = group.index
        prox_group = proximity_scores[idx]
        rci_group = rci_mask[idx]

        rci_prox = prox_group[rci_group]
        if len(rci_prox) > 0:
            prox_median = np.median(rci_prox)
        else:
            prox_median = np.median(prox_group)

        if prox_median <= 0:
            prox_median = 1e-6

        prox_factor = np.log(1 + prox_group) / np.log(1 + prox_median)

        base = grid_gdf.loc[idx, base_demand_col].values
        raw = base * prox_factor
        raw_sum = raw.sum()
        if raw_sum > 0:
            grid_gdf.loc[idx, corrected_col] = total_demand * raw / raw_sum
        else:
            grid_gdf.loc[idx, corrected_col] = total_demand / len(group)


def evaluate_allocation(subs_result, actual_col='Demand (MVA)', alloc_col='allocated_demand'):
    """Compute allocation quality metrics."""
    actual = subs_result[actual_col].values
    allocated = subs_result[alloc_col].values
    corr, _ = pearsonr(actual, allocated)
    rmse = np.sqrt(mean_squared_error(actual, allocated))
    mae = mean_absolute_error(actual, allocated)
    conservation_error = abs(allocated.sum() - actual.sum()) / actual.sum()
    return {
        'corr': corr, 'rmse': rmse, 'mae': mae,
        'conservation_error': conservation_error,
        'total_allocated': allocated.sum(),
        'total_actual': actual.sum(),
    }


# ════════════════════════════════════════════════════════════
# Frozen graph cache loading (read-only -- this script does not build a new cache)
# ════════════════════════════════════════════════════════════

def load_frozen_exp0_cache():
    """Read-only load of the exp0 primary graph cache, asserting all 16 regions are
    present and agent features are 5-dim.

    A missing cache raises immediately (its presence and checksum are verified by
    an upstream check; this script contains no build_graphs/load_data path, to
    prevent accidentally rebuilding and contaminating the frozen output).
    """
    if not FROZEN_EXP0_CACHE.exists():
        raise FileNotFoundError(
            f'exp0 primary graph cache not found: {FROZEN_EXP0_CACHE}\n'
            f'This script only reads that cache and does not build it -- please run '
            f'the upstream cache-presence check first.')

    print(f'Loading exp0 primary graph cache (read-only): {FROZEN_EXP0_CACHE}')
    with open(FROZEN_EXP0_CACHE, 'rb') as f:
        cached = pickle.load(f)

    graphs = cached['graphs']
    expected_dim = len(AGENT_FEATURE_COLS)
    assert set(graphs) >= set(ALL_LOCATIONS), 'Cache regions incomplete (!= 16 study regions)'
    for loc in ALL_LOCATIONS:
        dim = graphs[loc]['agent'].x.shape[1]
        assert dim == expected_dim, (
            f'{loc}: agent feature dim {dim} != {expected_dim} -- '
            f'the exp0 primary cache is corrupted or points at the fusion cache '
            f'(7-dim) instead; stop immediately')

    return (graphs, cached['grids'], cached['ntl_dict'],
            cached['region_dict'], cached['subs_dict'])


# ════════════════════════════════════════════════════════════
# Prediction + post-correction + aggregation (single region, identical to 027/005)
# ════════════════════════════════════════════════════════════

def predict_and_evaluate_location(loc, solver, graphs, grids, ntl_dict,
                                   region_dict, subs_dict,
                                   save_grid_demands_dir=None):
    """Run prediction -> post-correction -> aggregation -> evaluation for a single
    region, returning {method: metrics_dict}.

    save_grid_demands_dir: if not None, saves the grid_demands pickle to this directory.
    """
    grid_gdf, step_size_m = grids[loc]
    region_sub = region_dict[loc]
    subs_sub = subs_dict[loc]
    graph = graphs[loc]
    ntl_values = ntl_dict[loc]

    # predict edge weights
    edge_weights_df = solver.predict_edge_weights(graph)

    # map to grid_gdf -> gnn_demand
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

    # WC correction
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

    # NTL post-correction
    compute_ntl_corrected_demand(
        grid_gdf, region_sub, 'gnn_demand', ntl_values, 'ntl_gnn_demand')

    # Proximity post-correction
    prox_scores = compute_proximity_scores(grid_gdf, subs_sub)
    compute_proximity_corrected_demand(
        grid_gdf, region_sub, 'gnn_demand', prox_scores, 'prox_gnn_demand')

    # NTL -> Proximity stacked
    compute_proximity_corrected_demand(
        grid_gdf, region_sub, 'ntl_gnn_demand', prox_scores, 'ntl_prox_gnn_demand')

    # WC + NTL -> Proximity full stack
    compute_ntl_corrected_demand(
        grid_gdf, region_sub, 'wc_gnn_demand', ntl_values, 'wc_ntl_gnn_demand')
    compute_proximity_corrected_demand(
        grid_gdf, region_sub, 'wc_ntl_gnn_demand', prox_scores, 'wc_ntl_prox_gnn_demand')

    # save grid_demands (reused by downstream processing and concentration analysis)
    if save_grid_demands_dir is not None:
        save_grid_demands_dir.mkdir(parents=True, exist_ok=True)
        demand_cols_to_save = [
            'gnn_demand', 'wc_gnn_demand', 'ntl_gnn_demand',
            'prox_gnn_demand', 'ntl_prox_gnn_demand',
            'wc_ntl_gnn_demand', 'wc_ntl_prox_gnn_demand',
        ]
        grid_demands = {col: grid_gdf[col].values.copy() for col in demand_cols_to_save}
        with open(save_grid_demands_dir / f'{loc}_grid_demands.pickle', 'wb') as f:
            pickle.dump(grid_demands, f)

    # -- Voronoi allocation --
    alloc_voronoi = allocator_registry.create('voronoi')
    voronoi_res = alloc_voronoi.allocate(grid_gdf, subs_sub)

    demand_cols = {
        'voronoi_GNN': 'gnn_demand',
        'voronoi_wc_GNN': 'wc_gnn_demand',
        'voronoi_ntl_GNN': 'ntl_gnn_demand',
        'voronoi_prox_GNN': 'prox_gnn_demand',
        'voronoi_ntl_prox_GNN': 'ntl_prox_gnn_demand',
        'voronoi_wc_ntl_prox_GNN': 'wc_ntl_prox_gnn_demand',
    }

    loc_metrics = {}
    for method_name, demand_col in demand_cols.items():
        demand_arr = grid_gdf[demand_col].values
        subs_result = subs_sub.copy()
        subs_result['allocated_demand'] = 0.0
        for target_idx in range(len(subs_sub)):
            mask = voronoi_res.assignment == target_idx
            subs_result.loc[target_idx, 'allocated_demand'] = demand_arr[mask].sum()
        loc_metrics[method_name] = evaluate_allocation(subs_result)

    # -- CIVD allocation --
    coords_4326 = np.column_stack([subs_sub.geometry.x.values, subs_sub.geometry.y.values])
    cluster_gdf, centroid_gdf = do_clustering(coords_4326, method='hdbscan', min_cluster_size=2)
    centroid_gdf = centroid_gdf.reset_index(drop=True)

    target_civd = subs_sub.copy()
    target_civd['cluster_label'] = cluster_gdf['cluster_label']

    # cache the CIVD allocation (STATIC_DIR reused read-only, identical to 005/027)
    civd_cache_path = STATIC_DIR / f'{loc}_civd_cache.pickle'
    if civd_cache_path.exists():
        with open(civd_cache_path, 'rb') as f:
            cache = pickle.load(f)
        civd_assignment = cache['assignment']
    else:
        civd_config = {
            'solver': 'scip',
            'method': 'civd',
            'cluster_label_column': 'cluster_label',
            'n_jobs': -1,
        }
        alloc_civd = allocator_registry.create('civd', config=civd_config)
        civd_res = alloc_civd.allocate(grid_gdf, target_civd)
        civd_assignment = civd_res.assignment

    civd_demand_cols = {
        'civd_GNN': 'gnn_demand',
        'civd_wc_GNN': 'wc_gnn_demand',
        'civd_ntl_GNN': 'ntl_gnn_demand',
        'civd_prox_GNN': 'prox_gnn_demand',
        'civd_ntl_prox_GNN': 'ntl_prox_gnn_demand',
        'civd_wc_ntl_prox_GNN': 'wc_ntl_prox_gnn_demand',
    }

    for method_name, demand_col in civd_demand_cols.items():
        demand_arr = grid_gdf[demand_col].values
        subs_result = subs_sub.copy()
        subs_result['allocated_demand'] = 0.0
        cluster_demands = {}
        for label in np.unique(civd_assignment):
            mask = civd_assignment == label
            cluster_demands[label] = demand_arr[mask].sum()
        for label, total_d in cluster_demands.items():
            members = cluster_gdf[cluster_gdf['cluster_label'] == label].index
            n_members = len(members)
            if n_members > 0:
                for idx in members:
                    if idx < len(subs_result):
                        subs_result.loc[idx, 'allocated_demand'] += total_d / n_members
        loc_metrics[method_name] = evaluate_allocation(subs_result)

    return loc_metrics


# ════════════════════════════════════════════════════════════
# K-Fold training main loop (tau parameterization: both ModelConfig sites are driven by tau_start)
# ════════════════════════════════════════════════════════════

def assert_training_complete(model_path: Path, expected_epochs: int) -> None:
    """Hardened check that a resumed training run actually completed.

    model.pth is a best-checkpoint snapshot that can be written to disk at any
    point during training, so a hard interruption (crash/kill) can leave a
    partial result behind; *_training_log.json is only written once the full
    epoch loop has finished. So "model.pth exists" alone is not sufficient to
    conclude training completed -- the log must also exist with a full epoch
    count, otherwise this refuses to proceed and tells the caller how to fix
    it.
    """
    log_path = model_path.with_name(model_path.name[:-4] + '_training_log.json')
    if not log_path.exists():
        raise SystemExit(f'Truncated training run: {model_path} exists but {log_path.name} does not -- '
                         'move this directory aside for inspection and rerun (do not resume directly).')
    with open(log_path, encoding='utf-8') as f:
        _log = json.load(f)
    n_logged = len(_log.get('train_losses', {}).get('total', []))
    if n_logged != expected_epochs:
        raise SystemExit(f'Incomplete training log: {log_path} ({n_logged}/{expected_epochs}) -- '
                         'move this directory aside for inspection and rerun (do not resume directly).')


def run_kfold_training(config_name: str, seed: int, tau_start: float, out_root: Path,
                       graphs, grids, ntl_dict, region_dict, subs_dict):
    """Run K-Fold cross-validation training for a single config x seed x tau value.

    tau_start: allocation_temperature_start -- **both the resume branch and the
    normal-training branch's ModelConfig are driven by this parameter** (guarding
    against the two sites drifting apart).
    out_root: this tau value's independent output directory
    (results/exp_r213_tau_sweep/tau_{tag}/).
    """

    exp_config = CONFIG_MAP[config_name]
    objective_weights = exp_config['objective_weights']
    epochs = exp_config['epochs']

    # set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # output directory
    seed_dir = out_root / f'seed_{seed}' / config_name
    seed_dir.mkdir(parents=True, exist_ok=True)

    print(f'\n{"=" * 60}')
    print(f'config: {config_name} | seed: {seed} | epochs: {epochs} | tau_start: {tau_start}')
    print(f'loss function: {objective_weights}')
    print(f'output directory: {seed_dir}')
    print(f'{"=" * 60}\n')

    # K-Fold split (random_state follows seed, identical to 005/exp0)
    location_array = np.array(ALL_LOCATIONS)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

    fold_rmse_tables = []
    fold_mae_tables = []
    fold_corr_tables = []

    for fold_idx, (train_indices, test_indices) in enumerate(kf.split(location_array)):
        train_locs = location_array[train_indices].tolist()
        test_locs = location_array[test_indices].tolist()

        print(f'\n{"=" * 60}')
        print(f'Fold {fold_idx + 1}/{N_FOLDS}')
        print(f'  train regions ({len(train_locs)}): {train_locs}')
        print(f'  test regions ({len(test_locs)}): {test_locs}')
        print(f'{"=" * 60}')

        fold_dir = seed_dir / f'fold{fold_idx + 1}'
        fold_dir.mkdir(parents=True, exist_ok=True)
        model_path = fold_dir / 'model.pth'

        # -- check whether this fold is already fully complete (both model and metric CSVs present) --
        fold_csvs = [fold_dir / f'{name}.csv' for name in ('rmse', 'mae', 'corr')]
        if model_path.exists() and all(f.exists() for f in fold_csvs):
            assert_training_complete(model_path, epochs)
            # check whether grid_demands is missing -- if so, load the model and rerun inference
            grid_demands_dir = fold_dir / 'grid_demands'
            expected_gd = [grid_demands_dir / f'{loc}_grid_demands.pickle'
                           for loc in ALL_LOCATIONS]
            missing_gd = [p for p in expected_gd if not p.exists()]

            if missing_gd:
                print(f'Fold {fold_idx + 1} metrics already complete, but {len(missing_gd)} grid_demands are missing; rerunning inference...')

                warmup_epochs = 20
                decay_epochs = 20
                cosine_epochs = epochs - warmup_epochs - decay_epochs
                # tau parameterization site 1/2 (resume/inference-only branch) -- driven by the tau_start parameter
                config = ModelConfig(
                    epochs=epochs,
                    hidden_dim=256,
                    embedding_dim=128,
                    num_layers=3,
                    conv_type='hgt',
                    allocation_temperature_start=tau_start,
                    learning_rate=1e-3,
                    weight_decay=1e-4,
                    use_scheduler=True,
                    warmup_epochs=warmup_epochs,
                    decay_epochs=decay_epochs,
                    cosine_epochs=cosine_epochs,
                    cosine_eta_min=1e-5,
                    learnable=False,
                    save_path=str(model_path),
                    device='cuda' if torch.cuda.is_available() else 'cpu',
                )
                solver = EdgeWeightSolver(config)
                train_graphs_tmp = [graphs[loc] for loc in train_locs]
                train_dl_tmp = DataLoader(train_graphs_tmp, batch_size=1, shuffle=False)
                solver.init_model(train_dl_tmp, objective_weights)
                solver._load_checkpoint()

                for loc in ALL_LOCATIONS:
                    gd_file = grid_demands_dir / f'{loc}_grid_demands.pickle'
                    if gd_file.exists():
                        continue
                    print(f'  regenerating grid_demands: {loc}')
                    predict_and_evaluate_location(
                        loc, solver, graphs, grids, ntl_dict, region_dict, subs_dict,
                        save_grid_demands_dir=grid_demands_dir,
                    )
                print(f'Fold {fold_idx + 1} grid_demands regeneration complete')
            else:
                print(f'Fold {fold_idx + 1} already complete (including grid_demands), skipping')

            saved_rmse = pd.read_csv(fold_dir / 'rmse.csv', index_col=0)
            saved_mae = pd.read_csv(fold_dir / 'mae.csv', index_col=0)
            saved_corr = pd.read_csv(fold_dir / 'corr.csv', index_col=0)
            fold_rmse_tables.append(saved_rmse.T.to_dict())
            fold_mae_tables.append(saved_mae.T.to_dict())
            fold_corr_tables.append(saved_corr.T.to_dict())
            continue

        # -- normal training / load model + evaluate --
        # scale the scheduler to the number of epochs
        warmup_epochs = 20
        decay_epochs = 20
        cosine_epochs = epochs - warmup_epochs - decay_epochs

        # tau parameterization site 2/2 (normal-training branch) -- driven by the tau_start parameter
        config = ModelConfig(
            epochs=epochs,
            hidden_dim=256,
            embedding_dim=128,
            num_layers=3,
            conv_type='hgt',
            allocation_temperature_start=tau_start,
            learning_rate=1e-3,
            weight_decay=1e-4,
            use_scheduler=True,
            warmup_epochs=warmup_epochs,
            decay_epochs=decay_epochs,
            cosine_epochs=cosine_epochs,
            cosine_eta_min=1e-5,
            learnable=False,
            save_path=str(model_path),
            device='cuda' if torch.cuda.is_available() else 'cpu',
        )

        solver = EdgeWeightSolver(config)

        train_graphs = [graphs[loc] for loc in train_locs]
        test_graphs = [graphs[loc] for loc in test_locs]
        train_dl = DataLoader(train_graphs, batch_size=1, shuffle=False)
        test_dl = DataLoader(test_graphs, batch_size=1, shuffle=False)

        # check whether a trained model already exists -> skip training
        if model_path.exists():
            assert_training_complete(model_path, epochs)
            print(f'Model already exists at {model_path}, skipping training; building model structure and loading checkpoint')
            solver.init_model(train_dl, objective_weights)
            solver._load_checkpoint()
        else:
            print(f'Training config: {config.conv_type}, hidden={config.hidden_dim}, epochs={config.epochs}')
            print(f'device: {config.device}')
            solver.train_multi_graph(train_dl, test_dataloader=test_dl,
                                     objective_weights=objective_weights)
            print(f'Fold {fold_idx + 1} training complete')

        # predict & evaluate on all regions
        fold_rmse = {}
        fold_mae = {}
        fold_corr = {}

        grid_demands_dir = fold_dir / 'grid_demands'

        for loc in ALL_LOCATIONS:
            role = 'TRAIN' if loc in train_locs else 'TEST'
            print(f'\n  evaluating {loc} ({role})...')

            loc_metrics = predict_and_evaluate_location(
                loc, solver, graphs, grids, ntl_dict, region_dict, subs_dict,
                save_grid_demands_dir=grid_demands_dir,
            )

            for method_name, m in loc_metrics.items():
                if method_name not in fold_rmse:
                    fold_rmse[method_name] = {}
                    fold_mae[method_name] = {}
                    fold_corr[method_name] = {}
                fold_rmse[method_name][loc] = round(m['rmse'], 4)
                fold_mae[method_name][loc] = round(m['mae'], 4)
                fold_corr[method_name][loc] = round(m['corr'], 4)

                print(f'    {method_name}: corr={m["corr"]:.4f}, '
                      f'RMSE={m["rmse"]:.4f}, MAE={m["mae"]:.4f}')

        fold_rmse_tables.append(fold_rmse)
        fold_mae_tables.append(fold_mae)
        fold_corr_tables.append(fold_corr)

        # save this fold's tables
        pd.DataFrame(fold_rmse).T.to_csv(fold_dir / 'rmse.csv')
        pd.DataFrame(fold_mae).T.to_csv(fold_dir / 'mae.csv')
        pd.DataFrame(fold_corr).T.to_csv(fold_dir / 'corr.csv')

    # ════════════════════════════════════════════════════════════
    # Aggregate K-Fold results
    # ════════════════════════════════════════════════════════════

    print(f'\n{"=" * 60}')
    print('Aggregating K-Fold results...')
    print(f'{"=" * 60}')

    all_methods = sorted(fold_rmse_tables[0].keys())
    kf_splits = list(KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed).split(location_array))

    # only keep the result from the fold where the region was in the test set
    test_rmse = {m: {} for m in all_methods}
    test_mae = {m: {} for m in all_methods}
    test_corr = {m: {} for m in all_methods}

    for fold_idx, (_, test_indices) in enumerate(kf_splits):
        test_locs = location_array[test_indices].tolist()
        for loc in test_locs:
            for method in all_methods:
                test_rmse[method][loc] = fold_rmse_tables[fold_idx][method][loc]
                test_mae[method][loc] = fold_mae_tables[fold_idx][method][loc]
                test_corr[method][loc] = fold_corr_tables[fold_idx][method][loc]

    test_rmse_df = pd.DataFrame(test_rmse).T
    test_mae_df = pd.DataFrame(test_mae).T
    test_corr_df = pd.DataFrame(test_corr).T

    test_rmse_df['mean'] = test_rmse_df.mean(axis=1).round(4)
    test_mae_df['mean'] = test_mae_df.mean(axis=1).round(4)
    test_corr_df['mean'] = test_corr_df.mean(axis=1).round(4)

    test_rmse_df.to_csv(seed_dir / 'kfold_test_rmse.csv')
    test_mae_df.to_csv(seed_dir / 'kfold_test_mae.csv')
    test_corr_df.to_csv(seed_dir / 'kfold_test_corr.csv')

    print(f'\n=== K-Fold test-set RMSE ===')
    print(test_rmse_df.to_string())

    print(f'\n=== K-Fold test-set MAE ===')
    print(test_mae_df.to_string())

    print(f'\n=== K-Fold test-set Correlation ===')
    print(test_corr_df.to_string())

    # save split info
    split_info = {}
    for fold_idx, (train_indices, test_indices) in enumerate(kf_splits):
        split_info[f'fold_{fold_idx + 1}'] = {
            'train': location_array[train_indices].tolist(),
            'test': location_array[test_indices].tolist(),
        }
    with open(seed_dir / 'kfold_splits.json', 'w', encoding='utf-8') as f:
        json.dump(split_info, f, ensure_ascii=False, indent=2)

    # -- self-check the tolerance anchor for the tau=0.01 rung (informational only, does not abort -- GPU retraining is not bit-for-bit reproducible) --
    if abs(tau_start - ANCHOR_TAU) < 1e-12 and 'voronoi_GNN' in test_rmse_df.index:
        anchor_val = float(test_rmse_df.loc['voronoi_GNN', 'mean'])
        lo, hi = ANCHOR_RMSE_RANGE
        in_range = lo <= anchor_val <= hi
        print(f'\n[anchor self-check] tau=0.01 rung voronoi_GNN mean RMSE = {anchor_val:.4f}, '
              f'exp0 3-seed range [{lo}, {hi}] -> {"within range, OK" if in_range else "out of range, check the pipeline first!"}')

    print(f'\nAll results saved to: {seed_dir}')


# ════════════════════════════════════════════════════════════
# Main entry point
# ════════════════════════════════════════════════════════════

ALL_CONFIGS = list(CONFIG_MAP.keys())
DEFAULT_SEED = 42


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Tau initialization sweep training entry point -- each invocation runs one tau value x one seed x 4 folds')
    parser.add_argument('--tau-start', type=float, required=True,
                        choices=TAU_CHOICES, dest='tau_start',
                        help=f'allocation_temperature_start value to sweep ({TAU_CHOICES})')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                        help=f'random seed (this sweep uses 1 seed per tau value, default {DEFAULT_SEED})')
    parser.add_argument('--dry-run', action='store_true',
                        help='dry run: validates args, output directory, and frozen-input '
                             'presence, then exits before loading the graph cache or training')
    args = parser.parse_args()

    tau_tag = TAU_TAGS[args.tau_start]
    out_root = SWEEP_ROOT / f'tau_{tau_tag}'

    print(f'PyTorch: {torch.__version__}')
    print(f'CUDA: {torch.cuda.is_available()}')
    print(f'tau value: {args.tau_start} (tag={tau_tag}) | seed: {args.seed}')
    print(f'output directory: {out_root}')
    print(f'frozen graph cache (read-only): {FROZEN_EXP0_CACHE}')

    # -- --dry-run: validate args/paths then exit before loading the cache --
    if args.dry_run:
        print('\n' + '=' * 60)
        print('--dry-run: validation only, no cache loading or training')
        print('=' * 60)
        problems = []
        if not FROZEN_EXP0_CACHE.exists():
            problems.append(f'exp0 primary graph cache not found: {FROZEN_EXP0_CACHE}')
        civd_missing = [loc for loc in ALL_LOCATIONS
                        if not (STATIC_DIR / f'{loc}_civd_cache.pickle').exists()]
        if civd_missing:
            problems.append(f'{len(civd_missing)} CIVD static cache(s) missing: {civd_missing}'
                            ' (missing entries trigger an SCIP re-solve, not fatal but slow)')
        print(f'  exp0 primary cache present: {FROZEN_EXP0_CACHE.exists()}')
        print(f'  CIVD static caches: {len(ALL_LOCATIONS) - len(civd_missing)}/{len(ALL_LOCATIONS)} present')
        print(f'  tau parameterization: both the resume branch and the normal-training branch '
              f'ModelConfig are driven by tau_start={args.tau_start}')
        print(f'  tau=0.01 tolerance anchor: voronoi_GNN mean RMSE in [{ANCHOR_RMSE_RANGE[0]}, {ANCHOR_RMSE_RANGE[1]}]')
        if problems:
            print('\n[dry-run found issues]')
            for p in problems:
                print(f'  - {p}')
            sys.exit(1 if not FROZEN_EXP0_CACHE.exists() else 0)
        print('\n[dry-run passed] Frozen inputs are present; ready to start during an idle overnight GPU window.')
        sys.exit(0)

    # read-only load of the frozen graph cache (no rebuild path)
    graphs, grids, ntl_dict, region_dict, subs_dict = load_frozen_exp0_cache()

    # write the run manifest (records tau/seed/cache source/anchor convention, for
    # notebook and test traceability)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        'experiment': 'tau_sweep',
        'tau_start': args.tau_start,
        'tau_tag': tau_tag,
        'seed': args.seed,
        'config': 'baseline',
        'n_folds': N_FOLDS,
        'agent_feature_cols': AGENT_FEATURE_COLS,
        'graph_cache': str(FROZEN_EXP0_CACHE),
        'graph_cache_readonly': True,
        'anchor': {
            'tau': ANCHOR_TAU,
            'metric': 'kfold_test_rmse.csv :: voronoi_GNN :: mean',
            'range_exp0_3seed': list(ANCHOR_RMSE_RANGE),
            'kind': 'tolerance_anchor (GPU retraining is not bit-for-bit reproducible)',
        },
        'started_at': datetime.now().isoformat(timespec='seconds'),
    }
    with open(out_root / f'run_manifest_seed_{args.seed}.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    run_kfold_training('baseline', args.seed, args.tau_start, out_root,
                       graphs, grids, ntl_dict, region_dict, subs_dict)

    print('\n' + '=' * 60)
    print(f'Tau value {args.tau_start} (seed {args.seed}) complete')
    print('=' * 60)
