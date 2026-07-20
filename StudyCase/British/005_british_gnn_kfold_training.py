"""
004 GNN K-Fold Cross-Validation Training Script

Adapted from 004_british_gnn_training.ipynb, using 4-fold cross-validation.
After each fold finishes training, predicts the edge weights for all regions,
computes RMSE / MAE / Correlation, and finally saves the per-fold metric
tables and model files.

Usage:
    python 004_british_gnn_kfold_training.py
"""

import sys
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
from sklearn.model_selection import KFold
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from shapely.geometry import Point

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from SpatialAllocation.GNN.utils.GraphBuilder import (
    preprocess_features, prepare_hetero_graph_from_processed
)
from SpatialAllocation.GNN.core.EdgeWeightSolver import EdgeWeightSolver
from SpatialAllocation.GNN.core.ModelConfig import ModelConfig
from SpatialAllocation.Allocator import allocator_registry
from SpatialAllocation.Allocator.clustering.do_clustering import do_clustering
from SpatialAllocation.Weighter import weighter_registry

warnings.filterwarnings('ignore', category=FutureWarning)

# ════════════════════════════════════════════════════════════
# Configuration
# ════════════════════════════════════════════════════════════

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

N_FOLDS = 4

# ─── Path constants ───
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / 'results' / 'intermediate'
ASSEMBLED_DIR = DATA_DIR / 'features' / 'assembled'
EXTRACTED_DIR = DATA_DIR / 'features' / 'extracted'
OUTPUT_DIR = SCRIPT_DIR / 'results' / 'static_allocation'
GNN_DIR = DATA_DIR / 'GNN'
KFOLD_DIR = GNN_DIR / 'kfold'
KFOLD_DIR.mkdir(parents=True, exist_ok=True)

# ─── Region configuration ───
ALL_LOCATIONS = [
    'London',
    'TLH2', 'TLH3', 'TLJ1', 'TLF1', 'TLF2',
    'TLC1', 'TLC2', 'TLD6', 'TLG1', 'TLG2', 'TLE4',
    'TLH1', 'TLE3', 'TLD3', 'TLD4',
]

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
TARGET_CRS = 'EPSG:27700'

# ─── Land-use mapping ───
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


# ════════════════════════════════════════════════════════════
# Utility functions (consistent with the notebook)
# ════════════════════════════════════════════════════════════

def compute_demand(grid_gdf, region_sub, weighter_result, demand_col='demand'):
    """Combine the weighter output weights with region percentages to compute grid-point demand."""
    W = weighter_result.weights
    gdf = grid_gdf.copy()
    gdf[demand_col] = 0.0
    region_info = region_sub.set_index('ITL3')

    for itl3, group in gdf.groupby('ITL3'):
        if itl3 not in region_info.index:
            continue
        total_demand = region_info.loc[itl3, 'Demand (MVA)']
        idx = group.index

        if W.ndim == 2:
            pcts = np.array([region_info.loc[itl3, c] for c in PCT_COLS])
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
    """NTL post-correction: base_demand x ntl_factor, renormalized per ITL3 to preserve demand conservation."""
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
    """Proximity post-correction: base_demand x prox_factor, renormalized per ITL3 to preserve demand conservation."""
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
# Data loading
# ════════════════════════════════════════════════════════════

def load_data():
    """Load data for all regions, returning grids, ntl_dict, region_dict, subs_dict."""
    print('=' * 60)
    print('Loading data...')
    print('=' * 60)

    region_gdf = gpd.read_file(str(DATA_DIR / 'ITL3_region.gpkg'))
    substations_gdf = gpd.read_file(str(DATA_DIR / 'substations.gpkg'))
    print(f'ITL3 regions: {region_gdf.shape[0]} rows')
    print(f'Substations: {substations_gdf.shape[0]} rows')

    grids = {}
    for loc in ALL_LOCATIONS:
        path = ASSEMBLED_DIR / f'{loc}_grid_points.pickle'
        with open(path, 'rb') as f:
            grid_gdf, step_size_m = pickle.load(f)
        grids[loc] = (grid_gdf, step_size_m)

    ntl_dict = {}
    for loc in ALL_LOCATIONS:
        ntl_path = EXTRACTED_DIR / f'{loc}_ntl.npz'
        ntl_npz = np.load(ntl_path, allow_pickle=True)
        ntl_values = ntl_npz['data'][:, 0]
        ntl_dict[loc] = ntl_values

    region_dict = {}
    subs_dict = {}
    for loc in ALL_LOCATIONS:
        grid_gdf, step_size_m = grids[loc]
        study_itl3 = grid_gdf['ITL3'].unique()
        region_dict[loc] = region_gdf[region_gdf['ITL3'].isin(study_itl3)].copy()
        subs_dict[loc] = substations_gdf[substations_gdf['ITL3'].isin(study_itl3)].copy().reset_index(drop=True)

    # Compute landuse_demand
    for loc in ALL_LOCATIONS:
        grid_gdf, step_size_m = grids[loc]
        region_sub = region_dict[loc]
        subs_sub = subs_dict[loc]

        gpm = weighter_registry.create('gpm', config={'mode': 'categorical', 'proportion_columns': LU_COLS})
        gpm_res = gpm.compute(grid_gdf, target_gdf=subs_sub)
        grid_gdf = compute_demand(grid_gdf, region_sub, gpm_res, demand_col='landuse_demand')
        grids[loc] = (grid_gdf, step_size_m)

    print(f'Loaded data for {len(ALL_LOCATIONS)} regions')
    return grids, ntl_dict, region_dict, subs_dict


# ════════════════════════════════════════════════════════════
# Graph construction
# ════════════════════════════════════════════════════════════

def build_graphs(grids, region_dict, subs_dict):
    """Build the HeteroData graph for all regions."""
    print('\n' + '=' * 60)
    print('Building HeteroData graphs...')
    print('=' * 60)

    graphs = {}

    for loc in ALL_LOCATIONS:
        grid_gdf, step_size_m = grids[loc]
        region_sub = region_dict[loc]
        subs_sub = subs_dict[loc]

        # Derive the landuse category column
        lu_prop_cols = [c for c in LU_PROP_TO_CATEGORY if c in grid_gdf.columns]
        if 'landuse' not in grid_gdf.columns and lu_prop_cols:
            categories = [LU_PROP_TO_CATEGORY[c] for c in lu_prop_cols]
            max_idx = grid_gdf[lu_prop_cols].values.argmax(axis=1)
            grid_gdf = grid_gdf.copy()
            grid_gdf['landuse'] = [categories[i] for i in max_idx]
            grids[loc] = (grid_gdf, step_size_m)

        # Coordinate projection + normalization
        gdf_a = grid_gdf.copy().to_crs('EPSG:3857')
        gdf_t = subs_sub.copy().to_crs('EPSG:3857')
        gdf_s = region_sub.copy()
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

        agent_cols_for_graph = [c for c in AGENT_FEATURE_COLS if c in gdf_a_scaled.columns]
        source_cols_for_graph = [c for c in SOURCE_FEATURE_COLS if c in gdf_s_scaled.columns]

        features_a = preprocess_features(gdf_a_scaled[agent_cols_for_graph + ['geometry']])
        features_s = preprocess_features(gdf_s_scaled[source_cols_for_graph + ['geometry']])

        hetero_data = prepare_hetero_graph_from_processed(
            gdf_s_scaled, gdf_a_scaled,
            processed_features_s=features_s,
            processed_features_a=features_a,
            relation_column='ITL3',
        )

        graphs[loc] = hetero_data
        print(f'  {loc}: agent={hetero_data["agent"].num_nodes}, '
              f'source={hetero_data["source"].num_nodes}, '
              f'edges={hetero_data["source", "connects_to", "agent"].edge_index.shape[1]}')

    return graphs


# ════════════════════════════════════════════════════════════
# Prediction + post-correction + aggregation (single region)
# ════════════════════════════════════════════════════════════

def predict_and_evaluate_location(loc, solver, graphs, grids, ntl_dict,
                                   region_dict, subs_dict):
    """Run prediction -> post-correction -> aggregation -> evaluation for a single region, returning {method: metrics_dict}."""
    grid_gdf, step_size_m = grids[loc]
    region_sub = region_dict[loc]
    subs_sub = subs_dict[loc]
    graph = graphs[loc]
    ntl_values = ntl_dict[loc]

    # Predict edge weights
    edge_weights_df = solver.predict_edge_weights(graph)

    # Map back to grid_gdf -> gnn_demand
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

    # ── Voronoi allocation ──
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

    # ── CIVD allocation ──
    coords_4326 = np.column_stack([subs_sub.geometry.x.values, subs_sub.geometry.y.values])
    cluster_gdf, centroid_gdf = do_clustering(coords_4326, method='hdbscan', min_cluster_size=2)
    centroid_gdf = centroid_gdf.reset_index(drop=True)

    target_civd = subs_sub.copy()
    target_civd['cluster_label'] = cluster_gdf['cluster_label']

    # Cache the CIVD allocation
    civd_cache_path = OUTPUT_DIR / f'{loc}_civd_cache.pickle'
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
# K-Fold training main loop
# ════════════════════════════════════════════════════════════

def run_kfold_training():
    """Run K-fold cross-validation training."""

    # 1. Load data & build graphs
    grids, ntl_dict, region_dict, subs_dict = load_data()
    graphs = build_graphs(grids, region_dict, subs_dict)

    # 2. Prepare the K-fold split
    location_array = np.array(ALL_LOCATIONS)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    # Collect metrics across all folds: [{method: {loc: {metric: value}}}]
    fold_rmse_tables = []
    fold_mae_tables = []
    fold_corr_tables = []

    for fold_idx, (train_indices, test_indices) in enumerate(kf.split(location_array)):
        train_locs = location_array[train_indices].tolist()
        test_locs = location_array[test_indices].tolist()

        print('\n' + '=' * 60)
        print(f'Fold {fold_idx + 1}/{N_FOLDS}')
        print(f'  Training regions ({len(train_locs)}): {train_locs}')
        print(f'  Test regions ({len(test_locs)}): {test_locs}')
        print('=' * 60)

        # Per-fold subdirectory
        fold_dir = KFOLD_DIR / f'fold{fold_idx + 1}'
        fold_dir.mkdir(parents=True, exist_ok=True)

        # Model save path
        model_path = fold_dir / 'model.pth'

        config = ModelConfig(
            epochs=200,
            hidden_dim=256,
            embedding_dim=128,
            num_layers=3,
            conv_type='hgt',
            allocation_temperature_start=0.01,
            learning_rate=1e-3,
            weight_decay=1e-4,
            use_scheduler=True,
            warmup_epochs=20,
            decay_epochs=20,
            cosine_epochs=160,
            cosine_eta_min=1e-5,
            learnable=False,
            save_path=str(model_path),
            device='cuda' if torch.cuda.is_available() else 'cpu',
        )

        train_graphs = [graphs[loc] for loc in train_locs]
        test_graphs = [graphs[loc] for loc in test_locs]
        train_dl = DataLoader(train_graphs, batch_size=1, shuffle=False)
        test_dl = DataLoader(test_graphs, batch_size=1, shuffle=False)

        # Train
        solver = EdgeWeightSolver(config)
        objective_weights = {'landuse_prediction_loss': 1.0}

        print(f'Training config: {config.conv_type}, hidden={config.hidden_dim}, epochs={config.epochs}')
        print(f'Device: {config.device}')
        solver.train_multi_graph(train_dl, test_dataloader=test_dl, objective_weights=objective_weights)
        print(f'Fold {fold_idx + 1} training complete, model saved to: {fold_dir}')

        # Predict & evaluate for all regions
        fold_rmse = {}   # {method: {loc: rmse}}
        fold_mae = {}
        fold_corr = {}

        for loc in ALL_LOCATIONS:
            role = 'TRAIN' if loc in train_locs else 'TEST'
            print(f'\n  Evaluating {loc} ({role})...')

            loc_metrics = predict_and_evaluate_location(
                loc, solver, graphs, grids, ntl_dict, region_dict, subs_dict
            )

            for method_name, m in loc_metrics.items():
                if method_name not in fold_rmse:
                    fold_rmse[method_name] = {}
                    fold_mae[method_name] = {}
                    fold_corr[method_name] = {}
                fold_rmse[method_name][loc] = round(m['rmse'], 4)
                fold_mae[method_name][loc] = round(m['mae'], 4)
                fold_corr[method_name][loc] = round(m['corr'], 4)

                print(f'    {method_name}: corr={m["corr"]:.4f}, RMSE={m["rmse"]:.4f}, MAE={m["mae"]:.4f}')

        fold_rmse_tables.append(fold_rmse)
        fold_mae_tables.append(fold_mae)
        fold_corr_tables.append(fold_corr)

        # Save this fold's tables
        rmse_df = pd.DataFrame(fold_rmse).T
        mae_df = pd.DataFrame(fold_mae).T
        corr_df = pd.DataFrame(fold_corr).T

        rmse_df.to_csv(fold_dir / 'rmse.csv')
        mae_df.to_csv(fold_dir / 'mae.csv')
        corr_df.to_csv(fold_dir / 'corr.csv')

    # ════════════════════════════════════════════════════════════
    # Aggregate metrics across all folds
    # ════════════════════════════════════════════════════════════

    print('\n' + '=' * 60)
    print('Aggregating K-fold results...')
    print('=' * 60)

    # Collect all method names
    all_methods = sorted(fold_rmse_tables[0].keys())

    # Compute the metric for each (method, loc) from the fold in which it served as the test set
    # Each region serves as the test set exactly once across the K folds
    kf_splits = list(KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(location_array))

    # Build the test-only metric table: only take the result from the fold where the region was the test set
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

    # Add a mean column
    test_rmse_df['mean'] = test_rmse_df.mean(axis=1).round(4)
    test_mae_df['mean'] = test_mae_df.mean(axis=1).round(4)
    test_corr_df['mean'] = test_corr_df.mean(axis=1).round(4)

    # Save the aggregated tables
    test_rmse_df.to_csv(KFOLD_DIR / 'kfold_test_rmse.csv')
    test_mae_df.to_csv(KFOLD_DIR / 'kfold_test_mae.csv')
    test_corr_df.to_csv(KFOLD_DIR / 'kfold_test_corr.csv')

    print(f'\n=== K-fold test-set RMSE (each region taken from the fold where it was the test set) ===')
    print(test_rmse_df.to_string())

    print(f'\n=== K-fold test-set MAE ===')
    print(test_mae_df.to_string())

    print(f'\n=== K-fold test-set Correlation ===')
    print(test_corr_df.to_string())

    # Also save the train/test split info for each fold
    split_info = {}
    for fold_idx, (train_indices, test_indices) in enumerate(kf_splits):
        split_info[f'fold_{fold_idx + 1}'] = {
            'train': location_array[train_indices].tolist(),
            'test': location_array[test_indices].tolist(),
        }
    with open(KFOLD_DIR / 'kfold_splits.json', 'w', encoding='utf-8') as f:
        json.dump(split_info, f, ensure_ascii=False, indent=2)

    print(f'\nAll results saved to: {KFOLD_DIR}')
    print('File list:')
    print('  - kfold_test_rmse.csv    (RMSE for each region when used as the test set)')
    print('  - kfold_test_mae.csv     (MAE for each region when used as the test set)')
    print('  - kfold_test_corr.csv    (Correlation for each region when used as the test set)')
    print('  - kfold_splits.json      (train/test region split for each fold)')
    for i in range(N_FOLDS):
        print(f'  - fold{i + 1}/model.pth        (best model for fold {i + 1})')
        print(f'  - fold{i + 1}/rmse|mae|corr.csv (detailed metrics for all regions, fold {i + 1})')


if __name__ == '__main__':
    run_kfold_training()
