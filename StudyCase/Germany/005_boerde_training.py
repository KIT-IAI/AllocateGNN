"""
005 Börde single-graph training script

Single sample (1 NUTS3 region, 34 Gemeinden, 13 substations, 51619 grid points).
No k-fold; stability is estimated using multiple seeds. Training = evaluation (same graph).

Usage:
    python 005_boerde_training.py --config baseline --seed 42
    python 005_boerde_training.py --config ntl --seed 123
    python 005_boerde_training.py --all
    python 005_boerde_training.py --all --seeds 42 123
    python 005_boerde_training.py --all --configs baseline ntl

Config map:
    baseline:   {'landuse_prediction_loss': 1.0}
    ntl:        {'landuse_prediction_loss': 1.0, 'ntl_prior': 0.1}
    proximity:  {'landuse_prediction_loss': 1.0, 'proximity_prior': 0.1}
    ntl_prox:   {'landuse_prediction_loss': 1.0, 'ntl_prior': 0.1, 'proximity_prior': 0.1}
"""

import sys
import argparse
import pickle
import warnings
import json
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import torch
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from shapely.geometry import Point

# Project root directory
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from SpatialAllocation.GNN.utils.GraphBuilder import (
    preprocess_features, prepare_hetero_graph_from_processed
)
from SpatialAllocation.GNN.core.EdgeWeightSolver import EdgeWeightSolver
from SpatialAllocation.GNN.core.ModelConfig import ModelConfig
from SpatialAllocation.Allocator import allocator_registry
from SpatialAllocation.Allocator.clustering.do_clustering import do_clustering
from SpatialAllocation.Weighter import weighter_registry
from SpatialAllocation.FeatureExtractor.correctors.proximity_corrector import ProximityCorrector

warnings.filterwarnings('ignore', category=FutureWarning)

# ════════════════════════════════════════════════════════════
# Path constants
# ════════════════════════════════════════════════════════════

DATA_DIR = SCRIPT_DIR / 'results' / 'intermediate'
ASSEMBLED_DIR = DATA_DIR / 'features' / 'assembled'
EXTRACTED_DIR = DATA_DIR / 'features' / 'extracted'
EXP_DIR = SCRIPT_DIR / 'results' / 'models'

# ─── Column name constants ───
RELATION_COL = 'Name'                  # Sub-region grouping column (Gemeinde name)
DEMAND_COL = 'p_mw'                    # Substation actual load column
DERIVED_DEMAND_COL = 'Demand (MVA)'    # Derived Gemeinde-level load (for GraphBuilder compatibility)
TARGET_CRS = 'EPSG:25832'

# ─── Feature control ───
AGENT_FEATURE_COLS = [
    'lu_residential_prop', 'lu_commercial_prop', 'lu_industrial_prop',
    'lu_agricultural_prop', 'lu_others_prop',
]

SOURCE_FEATURE_COLS = [
    'residential_percent', 'commercial_percent', 'industrial_percent',
    'agricultural_percent', 'others_percent',
]

# ─── Post-correction constants ───
RCI_THRESHOLD = 0.5
PROXIMITY_GAMMA = 2.0
DIST_CLAMP_KM = 0.01

# ─── Landuse mapping ───
LANDUSE_PERCENT_MAP = {
    'lu_residential_prop': 'residential_percent',
    'lu_commercial_prop': 'commercial_percent',
    'lu_industrial_prop': 'industrial_percent',
    'lu_agricultural_prop': 'agricultural_percent',
    'lu_others_prop': 'others_percent',
}
LU_COLS = list(LANDUSE_PERCENT_MAP.keys())
PCT_COLS = list(LANDUSE_PERCENT_MAP.values())

LU_PROP_TO_CATEGORY = {
    'lu_residential_prop': 'residential',
    'lu_commercial_prop': 'commercial',
    'lu_industrial_prop': 'industrial',
    'lu_agricultural_prop': 'agricultural',
    'lu_others_prop': 'others',
}

# ─── Config map ───
CONFIG_MAP = {
    'baseline': {
        'objective_weights': {'landuse_prediction_loss': 1.0},
        'epochs': 200,
    },
    'ntl': {
        'objective_weights': {'landuse_prediction_loss': 1.0, 'ntl_prior': 0.1},
        'epochs': 200,
    },
    'proximity': {
        'objective_weights': {'landuse_prediction_loss': 1.0, 'proximity_prior': 0.1},
        'epochs': 200,
    },
    'ntl_prox': {
        'objective_weights': {
            'landuse_prediction_loss': 1.0,
            'ntl_prior': 0.1,
            'proximity_prior': 0.1,
        },
        'epochs': 400,
    },
}

AGENT_CONNECTIVITY = None


# ════════════════════════════════════════════════════════════
# Utility functions
# ════════════════════════════════════════════════════════════

def derive_gemeinde_demand(region_gdf, subs_gdf):
    """Derive Gemeinde-level load from substation p_mw, adding a 'Demand (MVA)' column to source_regions."""
    gemeinde_demand = subs_gdf.groupby('Gemeinde')[DEMAND_COL].sum()
    region_gdf = region_gdf.copy()
    region_gdf[DERIVED_DEMAND_COL] = (
        region_gdf[RELATION_COL].map(gemeinde_demand).fillna(0.0)
    )
    n_with = (region_gdf[DERIVED_DEMAND_COL] > 0).sum()
    print(f'Gemeinde load derivation: {n_with}/{len(region_gdf)} have load, '
          f'total {region_gdf[DERIVED_DEMAND_COL].sum():.2f} MW')
    return region_gdf


def compute_demand(grid_gdf, region_sub, weighter_result, demand_col='demand'):
    """Combine the weighter's output weights with regional percentages to compute grid-level demand."""
    W = weighter_result.weights
    gdf = grid_gdf.copy()
    gdf[demand_col] = 0.0
    region_info = region_sub.set_index(RELATION_COL)

    for name, group in gdf.groupby(RELATION_COL):
        if name not in region_info.index:
            continue
        total_demand = region_info.loc[name, DERIVED_DEMAND_COL]
        idx = group.index

        if W.ndim == 2:
            pcts = np.array([region_info.loc[name, c] for c in PCT_COLS])
            score = W[idx] @ pcts
        else:
            score = W[idx]

        score_sum = score.sum()
        if score_sum > 0:
            gdf.loc[idx, demand_col] = total_demand * score / score_sum
        else:
            gdf.loc[idx, demand_col] = total_demand / len(group)

    return gdf


def compute_ntl_corrected_demand(grid_gdf, region_sub, base_demand_col,
                                  ntl_values, corrected_col):
    """NTL post-correction: base_demand x ntl_factor, renormalized per Gemeinde to preserve demand conservation."""
    rci_sum = (grid_gdf['lu_residential_prop'].values
               + grid_gdf['lu_commercial_prop'].values
               + grid_gdf['lu_industrial_prop'].values)
    rci_mask = rci_sum > RCI_THRESHOLD

    grid_gdf[corrected_col] = 0.0
    region_info = region_sub.set_index(RELATION_COL)

    for name, group in grid_gdf.groupby(RELATION_COL):
        if name not in region_info.index:
            continue

        total_demand = region_info.loc[name, DERIVED_DEMAND_COL]
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
    """Compute the substation proximity score for each grid point."""
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
    """Proximity post-correction: base_demand x prox_factor, renormalized per Gemeinde to preserve demand conservation."""
    rci_sum = (grid_gdf['lu_residential_prop'].values
               + grid_gdf['lu_commercial_prop'].values
               + grid_gdf['lu_industrial_prop'].values)
    rci_mask = rci_sum > RCI_THRESHOLD

    grid_gdf[corrected_col] = 0.0
    region_info = region_sub.set_index(RELATION_COL)

    for name, group in grid_gdf.groupby(RELATION_COL):
        if name not in region_info.index:
            continue

        total_demand = region_info.loc[name, DERIVED_DEMAND_COL]
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


def evaluate_allocation(subs_result, actual_col=DEMAND_COL, alloc_col='allocated_demand'):
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
# Data loading
# ════════════════════════════════════════════════════════════

def load_data():
    """Load Börde data and precompute NTL/Proximity/RCI features."""
    print('=' * 60)
    print('Loading Börde data and precomputing prior features...')
    print('=' * 60)

    region_gdf = gpd.read_file(str(DATA_DIR / 'source_regions.gpkg'))
    subs_gdf = gpd.read_file(str(DATA_DIR / 'substations.gpkg'))
    print(f'Gemeinden: {region_gdf.shape[0]} rows')
    print(f'Substations: {subs_gdf.shape[0]} rows')

    # Derive Gemeinde-level load
    region_gdf = derive_gemeinde_demand(region_gdf, subs_gdf)

    # Load grid points
    path = ASSEMBLED_DIR / 'boerde_grid_points.pickle'
    with open(path, 'rb') as f:
        grid_gdf, step_size_m = pickle.load(f)
    print(f'Grid points: {grid_gdf.shape[0]} rows, step={step_size_m}m')

    # NTL
    ntl_path = EXTRACTED_DIR / 'boerde_ntl.npz'
    ntl_npz = np.load(ntl_path, allow_pickle=True)
    ntl_values = ntl_npz['data'][:, 0]
    assert len(ntl_values) == len(grid_gdf), \
        f'NTL length {len(ntl_values)} != grid points {len(grid_gdf)}'

    # Proximity (gamma=1.0, used for prior loss injection)
    prox_scores = ProximityCorrector.compute_scores(
        grid_gdf, subs_gdf, gamma=1.0,
        target_crs=TARGET_CRS, clamp_km=DIST_CLAMP_KM)

    # RCI mask
    rci = (grid_gdf['lu_residential_prop'] + grid_gdf['lu_commercial_prop']
           + grid_gdf['lu_industrial_prop']).values
    rci_mask = rci > RCI_THRESHOLD

    # Compute landuse_demand (used for landuse_mapping_matrix)
    gpm = weighter_registry.create(
        'gpm', config={'mode': 'categorical', 'proportion_columns': LU_COLS})
    gpm_res = gpm.compute(grid_gdf, target_gdf=subs_gdf)
    grid_gdf = compute_demand(grid_gdf, region_gdf, gpm_res, demand_col='landuse_demand')

    print('Data loading complete')
    return grid_gdf, step_size_m, region_gdf, subs_gdf, ntl_values, prox_scores, rci_mask


# ════════════════════════════════════════════════════════════
# Graph construction (with NTL/Proximity/RCI injection)
# ════════════════════════════════════════════════════════════

def build_graph(grid_gdf, region_gdf, subs_gdf,
                ntl_values, prox_scores, rci_mask,
                inject_priors=True):
    """Build a single HeteroData graph."""
    print('\n' + '=' * 60)
    print('Building HeteroData graph...')
    print('=' * 60)

    # Derive landuse category column
    lu_prop_cols = [c for c in LU_PROP_TO_CATEGORY if c in grid_gdf.columns]
    if 'landuse' not in grid_gdf.columns and lu_prop_cols:
        categories = [LU_PROP_TO_CATEGORY[c] for c in lu_prop_cols]
        max_idx = grid_gdf[lu_prop_cols].values.argmax(axis=1)
        grid_gdf = grid_gdf.copy()
        grid_gdf['landuse'] = [categories[i] for i in max_idx]

    # Coordinate projection + normalization
    gdf_a = grid_gdf.copy().to_crs('EPSG:3857')
    gdf_t = subs_gdf.copy().to_crs('EPSG:3857')
    gdf_s = region_gdf.copy()
    gdf_s['geometry'] = gdf_s.geometry.centroid.to_crs('EPSG:3857')

    coords_a = np.column_stack([gdf_a.geometry.x, gdf_a.geometry.y])
    coords_t = np.column_stack([gdf_t.geometry.x, gdf_t.geometry.y])
    coords_s = np.column_stack([gdf_s.geometry.x, gdf_s.geometry.y])

    scaler = StandardScaler().fit(np.vstack([coords_a, coords_t, coords_s]))

    coords_a_scaled = scaler.transform(coords_a)
    coords_s_scaled = scaler.transform(coords_s)

    gdf_a_scaled = gdf_a.copy()
    gdf_a_scaled['geometry'] = [Point(x, y) for x, y in coords_a_scaled]

    gdf_s_scaled = gdf_s.copy()
    gdf_s_scaled['geometry'] = [Point(x, y) for x, y in coords_s_scaled]

    agent_cols = [c for c in AGENT_FEATURE_COLS if c in gdf_a_scaled.columns]
    source_cols = [c for c in SOURCE_FEATURE_COLS if c in gdf_s_scaled.columns]

    features_a = preprocess_features(gdf_a_scaled[agent_cols + ['geometry']])
    features_s = preprocess_features(gdf_s_scaled[source_cols + ['geometry']])

    hetero_data = prepare_hetero_graph_from_processed(
        gdf_s_scaled, gdf_a_scaled,
        processed_features_s=features_s,
        processed_features_a=features_a,
        relation_column=RELATION_COL,
        agent_connectivity=AGENT_CONNECTIVITY,
    )

    # Inject NTL / Proximity / RCI (needed for prior losses)
    if inject_priors:
        hetero_data['agent'].ntl_values = torch.tensor(
            ntl_values, dtype=torch.float32)
        hetero_data['agent'].proximity_scores = torch.tensor(
            prox_scores, dtype=torch.float32)
        hetero_data['agent'].rci_mask = torch.tensor(
            rci_mask, dtype=torch.bool)

    print(f'  boerde: agent={hetero_data["agent"].num_nodes}, '
          f'source={hetero_data["source"].num_nodes}, '
          f'edges={hetero_data["source", "connects_to", "agent"].edge_index.shape[1]}')

    return hetero_data


# ════════════════════════════════════════════════════════════
# Prediction + post-correction + aggregation (single region)
# ════════════════════════════════════════════════════════════

def predict_and_evaluate(solver, graph, grid_gdf, step_size_m,
                          region_gdf, subs_gdf, ntl_values,
                          save_dir=None):
    """Predict -> post-correct -> aggregate -> evaluate; returns (metrics_dict, grid_gdf_with_demands)."""
    # Predict edge weights
    edge_weights_df = solver.predict_edge_weights(graph)

    # Map back to grid_gdf -> gnn_demand
    region_info = region_gdf.set_index(RELATION_COL)
    source_index_map = graph.source_index_map

    grid_gdf = grid_gdf.copy()
    grid_gdf['gnn_demand'] = 0.0

    for _, row in edge_weights_df.iterrows():
        s_idx = int(row['source_node_idx'])
        a_orig_idx = int(row['agent_original_idx'])
        w = row['predicted_weight']

        s_orig_idx = source_index_map.iloc[s_idx]
        name = region_gdf.loc[s_orig_idx, RELATION_COL]
        total_demand = region_info.loc[name, DERIVED_DEMAND_COL]
        grid_gdf.loc[a_orig_idx, 'gnn_demand'] += w * total_demand

    # WC correction
    grid_gdf['wc_gnn_demand'] = grid_gdf['gnn_demand'].copy()
    wc_weight = 1.0 - grid_gdf['wc_others_ratio'].values
    grid_gdf['wc_gnn_demand'] *= wc_weight

    for name, group in grid_gdf.groupby(RELATION_COL):
        if name not in region_info.index:
            continue
        idx = group.index
        total_demand = region_info.loc[name, DERIVED_DEMAND_COL]
        wc_sum = grid_gdf.loc[idx, 'wc_gnn_demand'].sum()
        if wc_sum > 0:
            grid_gdf.loc[idx, 'wc_gnn_demand'] *= total_demand / wc_sum
        else:
            grid_gdf.loc[idx, 'wc_gnn_demand'] = total_demand / len(group)

    # NTL post-correction
    compute_ntl_corrected_demand(
        grid_gdf, region_gdf, 'gnn_demand', ntl_values, 'ntl_gnn_demand')

    # Proximity post-correction
    prox_scores = compute_proximity_scores(grid_gdf, subs_gdf)
    compute_proximity_corrected_demand(
        grid_gdf, region_gdf, 'gnn_demand', prox_scores, 'prox_gnn_demand')

    # NTL -> Proximity stacking
    compute_proximity_corrected_demand(
        grid_gdf, region_gdf, 'ntl_gnn_demand', prox_scores, 'ntl_prox_gnn_demand')

    # WC + NTL -> Proximity full stack
    compute_ntl_corrected_demand(
        grid_gdf, region_gdf, 'wc_gnn_demand', ntl_values, 'wc_ntl_gnn_demand')
    compute_proximity_corrected_demand(
        grid_gdf, region_gdf, 'wc_ntl_gnn_demand', prox_scores, 'wc_ntl_prox_gnn_demand')

    # Save grid_demands
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        demand_cols_to_save = [
            'gnn_demand', 'wc_gnn_demand', 'ntl_gnn_demand',
            'prox_gnn_demand', 'ntl_prox_gnn_demand',
            'wc_ntl_gnn_demand', 'wc_ntl_prox_gnn_demand',
        ]
        grid_demands = {col: grid_gdf[col].values.copy() for col in demand_cols_to_save}
        with open(save_dir / 'boerde_grid_demands.pickle', 'wb') as f:
            pickle.dump(grid_demands, f)

    # ── Voronoi allocation ──
    alloc_voronoi = allocator_registry.create('voronoi')
    voronoi_res = alloc_voronoi.allocate(grid_gdf, subs_gdf)

    demand_cols = {
        'voronoi_GNN': 'gnn_demand',
        'voronoi_wc_GNN': 'wc_gnn_demand',
        'voronoi_ntl_GNN': 'ntl_gnn_demand',
        'voronoi_prox_GNN': 'prox_gnn_demand',
        'voronoi_ntl_prox_GNN': 'ntl_prox_gnn_demand',
        'voronoi_wc_ntl_prox_GNN': 'wc_ntl_prox_gnn_demand',
    }

    metrics = {}
    for method_name, demand_col in demand_cols.items():
        demand_arr = grid_gdf[demand_col].values
        subs_result = subs_gdf.copy()
        subs_result['allocated_demand'] = 0.0
        for target_idx in range(len(subs_gdf)):
            mask = voronoi_res.assignment == target_idx
            subs_result.loc[target_idx, 'allocated_demand'] = demand_arr[mask].sum()
        metrics[method_name] = evaluate_allocation(subs_result)

    # ── CIVD allocation ──
    coords_4326 = np.column_stack([
        subs_gdf.geometry.x.values, subs_gdf.geometry.y.values])
    cluster_gdf, centroid_gdf = do_clustering(
        coords_4326, method='hdbscan', min_cluster_size=2)

    target_civd = subs_gdf.copy()
    target_civd['cluster_label'] = cluster_gdf['cluster_label']

    civd_config = {
        'solver': 'scip', 'method': 'civd',
        'cluster_label_column': 'cluster_label', 'n_jobs': -1,
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
        subs_result = subs_gdf.copy()
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
        metrics[method_name] = evaluate_allocation(subs_result)

    return metrics, grid_gdf


# ════════════════════════════════════════════════════════════
# Main training loop
# ════════════════════════════════════════════════════════════

def assert_training_complete(model_path: Path, expected_epochs: int) -> None:
    """Guard against treating an interrupted run as a completed training run.

    model.pth is the best checkpoint saved at any point during the training loop,
    so a hard interruption (crash/kill) can leave a partial checkpoint on disk;
    *_training_log.json is only written after the full epoch loop finishes.
    Therefore the presence of model.pth alone does not prove training completed --
    the log must exist and report the expected number of epochs, otherwise the
    run is rejected with guidance on how to recover.
    """
    log_path = model_path.with_name(model_path.name[:-4] + '_training_log.json')
    if not log_path.exists():
        raise SystemExit(f'Truncated training run: {model_path} exists but {log_path.name} is missing -- '
                         'move this directory aside for inspection and rerun (do not resume in place).')
    with open(log_path, encoding='utf-8') as f:
        _log = json.load(f)
    n_logged = len(_log.get('train_losses', {}).get('total', []))
    if n_logged != expected_epochs:
        raise SystemExit(f'Incomplete training log: {log_path} ({n_logged}/{expected_epochs}) -- '
                         'move this directory aside for inspection and rerun (do not resume in place).')


def run_training(config_name, seed, graph, grid_gdf, step_size_m,
                  region_gdf, subs_gdf, ntl_values):
    """Run training + evaluation for a single config x seed combination."""

    exp_config = CONFIG_MAP[config_name]
    objective_weights = exp_config['objective_weights']
    epochs = exp_config['epochs']

    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Output directory
    seed_dir = EXP_DIR / config_name / f'seed_{seed}'
    seed_dir.mkdir(parents=True, exist_ok=True)
    model_path = seed_dir / 'model.pth'

    print(f'\n{"=" * 60}')
    print(f'Config: {config_name} | seed: {seed} | epochs: {epochs}')
    print(f'Loss weights: {objective_weights}')
    print(f'Output directory: {seed_dir}')
    print(f'{"=" * 60}\n')

    # Scheduler parameters
    warmup_epochs = 20
    decay_epochs = 20
    cosine_epochs = epochs - warmup_epochs - decay_epochs

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
        cosine_epochs=cosine_epochs,
        cosine_eta_min=1e-5,
        learnable=False,
        save_path=str(model_path),
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )

    solver = EdgeWeightSolver(config)

    # Single-graph DataLoader (training = evaluation)
    train_dl = DataLoader([graph], batch_size=1, shuffle=False)

    # Train or load an existing model
    metrics_csv = seed_dir / 'metrics.csv'
    if model_path.exists() and metrics_csv.exists():
        assert_training_complete(model_path, epochs)
        print(f'Existing model + metrics found, skipping: {seed_dir}')
        return pd.read_csv(metrics_csv, index_col=0).T.to_dict()

    if model_path.exists():
        assert_training_complete(model_path, epochs)
        print(f'Existing model {model_path} found; skipping training, loading and evaluating')
        solver.init_model(train_dl, objective_weights)
        solver._load_checkpoint()
    else:
        print(f'Training: {config.conv_type}, hidden={config.hidden_dim}, '
              f'epochs={config.epochs}, device={config.device}')
        solver.train_multi_graph(
            train_dl, test_dataloader=None,
            objective_weights=objective_weights)
        print('Training complete')

    # Evaluate
    print('\nEvaluating...')
    grid_demands_dir = seed_dir / 'grid_demands'
    metrics, _ = predict_and_evaluate(
        solver, graph, grid_gdf, step_size_m,
        region_gdf, subs_gdf, ntl_values,
        save_dir=grid_demands_dir,
    )

    # Save metrics
    metrics_rows = []
    for method_name, m in metrics.items():
        metrics_rows.append({
            'method': method_name,
            'corr': round(m['corr'], 4),
            'rmse': round(m['rmse'], 4),
            'mae': round(m['mae'], 4),
            'conservation_error': round(m['conservation_error'], 6),
        })
        print(f'  {method_name}: corr={m["corr"]:.4f}, '
              f'RMSE={m["rmse"]:.4f}, MAE={m["mae"]:.4f}')

    metrics_df = pd.DataFrame(metrics_rows).set_index('method')
    metrics_df.to_csv(metrics_csv)

    # Save training log
    training_log = {
        'config_name': config_name,
        'seed': seed,
        'epochs': epochs,
        'objective_weights': objective_weights,
        'model_config': {
            'hidden_dim': config.hidden_dim,
            'embedding_dim': config.embedding_dim,
            'num_layers': config.num_layers,
            'conv_type': config.conv_type,
        },
        'note': 'Single-graph training (train=eval), not a generalization test',
    }
    with open(seed_dir / 'training_log.json', 'w', encoding='utf-8') as f:
        json.dump(training_log, f, ensure_ascii=False, indent=2)

    return metrics


def aggregate_seeds(config_name, seeds):
    """Aggregate results across multiple seeds and produce a summary CSV (mean±std)."""
    summary_dir = EXP_DIR / 'summary'
    summary_dir.mkdir(parents=True, exist_ok=True)

    dfs = []
    for seed in seeds:
        csv_path = EXP_DIR / config_name / f'seed_{seed}' / 'metrics.csv'
        if csv_path.exists():
            dfs.append(pd.read_csv(csv_path, index_col=0))

    if not dfs:
        print(f'Warning: no available results for {config_name}')
        return None

    # Intersect columns (corr, rmse, mae)
    metric_cols = ['corr', 'rmse', 'mae']
    stacked = np.stack([df[metric_cols].values for df in dfs])
    mean_vals = stacked.mean(axis=0)
    std_vals = stacked.std(axis=0)

    methods = dfs[0].index
    summary_data = {}
    for i, col in enumerate(metric_cols):
        summary_data[f'{col}_mean'] = mean_vals[:, i].round(4)
        summary_data[f'{col}_std'] = std_vals[:, i].round(4)
        summary_data[col] = [
            f'{mean_vals[j, i]:.4f}±{std_vals[j, i]:.4f}'
            for j in range(len(methods))
        ]

    summary_df = pd.DataFrame(summary_data, index=methods)
    summary_df.to_csv(summary_dir / f'{config_name}_metrics.csv')

    print(f'\n=== {config_name} summary ({len(dfs)} seeds) ===')
    display_cols = [c for c in metric_cols]
    print(summary_df[display_cols].to_string())

    return summary_df


# ════════════════════════════════════════════════════════════
# Main entry point
# ════════════════════════════════════════════════════════════

ALL_CONFIGS = list(CONFIG_MAP.keys())
DEFAULT_SEEDS = [42, 123, 456]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Börde single-graph training')
    parser.add_argument('--config', type=str, choices=ALL_CONFIGS,
                        help='Training config (mutually exclusive with --all)')
    parser.add_argument('--seed', type=int,
                        help='Random seed (mutually exclusive with --all)')
    parser.add_argument('--all', action='store_true',
                        help='Run all configs x all seeds')
    parser.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS,
                        help=f'List of seeds to run (default {DEFAULT_SEEDS})')
    parser.add_argument('--configs', type=str, nargs='+',
                        choices=ALL_CONFIGS, default=ALL_CONFIGS,
                        help='List of configs to run (default: all)')
    args = parser.parse_args()

    # Argument validation
    if args.all:
        run_configs = args.configs
        run_seeds = args.seeds
    elif args.config and args.seed is not None:
        run_configs = [args.config]
        run_seeds = [args.seed]
    else:
        parser.error('Use --all to run everything, or specify both --config and --seed')

    print(f'PyTorch: {torch.__version__}')
    print(f'CUDA: {torch.cuda.is_available()}')

    # Load data + build graph (done once)
    (grid_gdf, step_size_m, region_gdf, subs_gdf,
     ntl_values, prox_scores, rci_mask) = load_data()
    graph = build_graph(
        grid_gdf, region_gdf, subs_gdf,
        ntl_values, prox_scores, rci_mask)

    total = len(run_configs) * len(run_seeds)
    current = 0
    for cfg in run_configs:
        for s in run_seeds:
            current += 1
            print(f'\n{"#" * 60}')
            print(f'# Task {current}/{total}: config={cfg}, seed={s}')
            print(f'{"#" * 60}')
            run_training(cfg, s, graph, grid_gdf, step_size_m,
                         region_gdf, subs_gdf, ntl_values)

        # Aggregate all seeds for the current config
        aggregate_seeds(cfg, run_seeds)

    print('\n' + '=' * 60)
    print(f'All done! Ran {total} tasks total')
    print('=' * 60)
