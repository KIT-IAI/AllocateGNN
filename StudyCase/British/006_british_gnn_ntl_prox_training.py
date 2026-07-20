"""
009 GNN + NTL/Proximity Prior-Loss Training Script

Adapted from 009_british_gnn_ntl_prox_training.ipynb.
Incorporates NTL and Proximity priors into the training loop as a forward
KL-divergence regularization loss that directly constrains the spatial
distribution of w_{s->a}.

Pipeline: load data -> precompute NTL/Proximity/RCI
     -> build HeteroData -> train EdgeWeightSolver
     -> predict w_sa -> aggregate to substations -> evaluate metrics -> update leaderboard

Usage:
    python 009_british_gnn_ntl_prox_training.py
"""

import sys
import pickle
import time
import json
from datetime import datetime
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import torch
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import StandardScaler
from shapely.geometry import Point
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error

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
from SpatialAllocation.FeatureExtractor.correctors.proximity_corrector import ProximityCorrector

warnings.filterwarnings('ignore', category=FutureWarning)

# ════════════════════════════════════════════════════════════
# Configuration
# ════════════════════════════════════════════════════════════

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ─── Path constants ───
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / 'results' / 'intermediate'
ASSEMBLED_DIR = DATA_DIR / 'features' / 'assembled'
EXTRACTED_DIR = DATA_DIR / 'features' / 'extracted'
OUTPUT_DIR = SCRIPT_DIR / 'results' / 'static_allocation'

# ─── Experiment name ───
EXPERIMENT_NAME = 'GNN_proximity'
GNN_DIR = DATA_DIR / 'GNN_TEST' / EXPERIMENT_NAME
GNN_DIR.mkdir(parents=True, exist_ok=True)

# Leaderboard lives at the GNN_TEST root directory
LEADERBOARD_PATH = DATA_DIR / 'GNN_TEST' / 'gnn_leaderboard.json'

# ─── Training configuration ───
OVERWRITE = True
MODEL_NAME = f'model_{SEED}_proximity_prior.pth'
MODEL_PATH = GNN_DIR / MODEL_NAME

# ─── Region configuration ───
TRAIN_LOCATIONS = [
    'London',
    'TLH2', 'TLH3', 'TLJ1', 'TLF1', 'TLF2',
    'TLC1', 'TLC2', 'TLD6', 'TLG1', 'TLG2', 'TLE4',
]
TEST_LOCATIONS = ['TLH1', 'TLE3', 'TLD3', 'TLD4']
STUDY_REGIONS = TRAIN_LOCATIONS + TEST_LOCATIONS

# ─── Agent adjacency type ───
AGENT_CONNECTIVITY = 'moore'

# ─── Feature list ───
AGENT_FEATURE_COLS = [
    'lu_residential_prop', 'lu_commercial_prop', 'lu_industrial_prop',
    'lu_agricultural_prop', 'lu_others_prop',
]

SOURCE_FEATURE_COLS = [
    'residential_percent', 'commercial_percent', 'industrial_percent',
    'agricultural_percent', 'others_percent',
]

# ─── RCI threshold ───
RCI_THRESHOLD = 0.5

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

# ─── Leaderboard configuration ───
BASELINE_METHOD = 'ITL3_average'
GNN_METHOD = 'civd_GNN'


# ════════════════════════════════════════════════════════════
# Utility functions
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
# Data loading + precompute NTL / Proximity / RCI
# ════════════════════════════════════════════════════════════

def load_data():
    """Load data for all regions and precompute NTL/Proximity/RCI."""
    print('=' * 60)
    print('Loading data + precomputing prior features...')
    print('=' * 60)

    region_gdf = gpd.read_file(str(DATA_DIR / 'ITL3_region.gpkg'))
    substations_gdf = gpd.read_file(str(DATA_DIR / 'substations.gpkg'))
    print(f'ITL3 regions: {region_gdf.shape[0]} rows')
    print(f'Substations: {substations_gdf.shape[0]} rows')

    # Load the assembled grid_gdf
    grids = {}
    for loc in STUDY_REGIONS:
        path = ASSEMBLED_DIR / f'{loc}_grid_points.pickle'
        with open(path, 'rb') as f:
            grid_gdf, step_size_m = pickle.load(f)
        grids[loc] = (grid_gdf, step_size_m)
        print(f'  {loc}: {len(grid_gdf)} grid points, step={step_size_m}m')

    # Load NTL data
    ntl_dict = {}
    for loc in STUDY_REGIONS:
        ntl_path = EXTRACTED_DIR / f'{loc}_ntl.npz'
        ntl_npz = np.load(ntl_path, allow_pickle=True)
        ntl_values = ntl_npz['data'][:, 0]
        ntl_dict[loc] = ntl_values
        grid_gdf, _ = grids[loc]
        assert len(ntl_values) == len(grid_gdf), \
            f'{loc}: NTL length {len(ntl_values)} != grid length {len(grid_gdf)}'
        print(f'  {loc} NTL: N={len(ntl_values)}, range=[{ntl_values.min():.2f}, {ntl_values.max():.2f}]')

    # Organize the data dicts per region
    region_dict = {}
    subs_dict = {}
    for loc in STUDY_REGIONS:
        grid_gdf, step_size_m = grids[loc]
        study_itl3 = grid_gdf['ITL3'].unique()
        region_dict[loc] = region_gdf[region_gdf['ITL3'].isin(study_itl3)].copy()
        subs_dict[loc] = substations_gdf[substations_gdf['ITL3'].isin(study_itl3)].copy().reset_index(drop=True)
        print(f'  {loc}: {len(region_dict[loc])} ITL3 regions, {len(subs_dict[loc])} substations')

    # Precompute Proximity scores
    proximity_dict = {}
    for loc in STUDY_REGIONS:
        grid_gdf, _ = grids[loc]
        subs_sub = subs_dict[loc]
        prox_scores = ProximityCorrector.compute_scores(grid_gdf, subs_sub, gamma=1.0)
        proximity_dict[loc] = prox_scores
        print(f'  {loc} Proximity: range=[{prox_scores.min():.4f}, {prox_scores.max():.4f}]')

    # Precompute the RCI mask
    rci_dict = {}
    for loc in STUDY_REGIONS:
        grid_gdf, _ = grids[loc]
        rci = (grid_gdf['lu_residential_prop'] + grid_gdf['lu_commercial_prop']
               + grid_gdf['lu_industrial_prop']).values
        rci_dict[loc] = rci > RCI_THRESHOLD
        print(f'  {loc} RCI: {rci_dict[loc].sum()}/{len(rci_dict[loc])} ({rci_dict[loc].mean():.1%})')

    # Compute landuse_demand for each region
    for loc in STUDY_REGIONS:
        grid_gdf, step_size_m = grids[loc]
        region_sub = region_dict[loc]
        subs_sub = subs_dict[loc]

        gpm = weighter_registry.create('gpm', config={'mode': 'categorical', 'proportion_columns': LU_COLS})
        gpm_res = gpm.compute(grid_gdf, target_gdf=subs_sub)
        grid_gdf = compute_demand(grid_gdf, region_sub, gpm_res, demand_col='landuse_demand')
        grids[loc] = (grid_gdf, step_size_m)
        print(f'  {loc} landuse_demand total: {grid_gdf["landuse_demand"].sum():.1f} MVA')

    print(f'Loaded data for {len(STUDY_REGIONS)} regions')
    return grids, ntl_dict, proximity_dict, rci_dict, region_dict, subs_dict


# ════════════════════════════════════════════════════════════
# Graph construction + inject NTL/Proximity/RCI
# ════════════════════════════════════════════════════════════

def build_graphs(grids, ntl_dict, proximity_dict, rci_dict, region_dict, subs_dict):
    """Build the HeteroData graph for all regions, and inject NTL/Proximity/RCI."""
    print('\n' + '=' * 60)
    print('Building HeteroData graphs + injecting NTL/Proximity/RCI...')
    print('=' * 60)

    graphs = {}

    for loc in STUDY_REGIONS:
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
            agent_connectivity=AGENT_CONNECTIVITY,
        )

        # Inject NTL / Proximity / RCI into the HeteroData
        hetero_data['agent'].ntl_values = torch.tensor(ntl_dict[loc], dtype=torch.float32)
        hetero_data['agent'].proximity_scores = torch.tensor(proximity_dict[loc], dtype=torch.float32)
        hetero_data['agent'].rci_mask = torch.tensor(rci_dict[loc], dtype=torch.bool)

        graphs[loc] = hetero_data

        split = 'TRAIN' if loc in TRAIN_LOCATIONS else 'TEST'
        n_edges_sa = hetero_data['source', 'connects_to', 'agent'].edge_index.shape[1]
        has_agent_adj = ('agent', 'near', 'agent') in hetero_data.edge_types
        print(f'  {loc} [{split}]: agent={hetero_data["agent"].num_nodes}, '
              f'source={hetero_data["source"].num_nodes}, '
              f'edges_sa={n_edges_sa}'
              f'{", agent_adj=" + str(hetero_data["agent", "near", "agent"].edge_index.shape[1]) if has_agent_adj else ""}')

    return graphs


# ════════════════════════════════════════════════════════════
# Training
# ════════════════════════════════════════════════════════════

def train_model(graphs):
    """Train the GNN model (L_landuse + L_ntl_prior + L_proximity_prior)."""
    print('\n' + '=' * 60)
    print('Training GNN...')
    print('=' * 60)

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
        save_path=str(MODEL_PATH),
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )

    train_graphs = [graphs[loc] for loc in TRAIN_LOCATIONS]
    test_graphs = [graphs[loc] for loc in TEST_LOCATIONS]
    train_dl = DataLoader(train_graphs, batch_size=1, shuffle=False)
    test_dl = DataLoader(test_graphs, batch_size=1, shuffle=False)

    solver = EdgeWeightSolver(config)
    objective_weights = {
        'landuse_prediction_loss': 1.0,
        # 'ntl_prior': 0.1,
        'proximity_prior': 0.1,
    }

    if not OVERWRITE and MODEL_PATH.exists():
        solver.init_model(train_dl, objective_weights)
        print(f'Found existing model: {MODEL_PATH.name}, skipping training. Set OVERWRITE=True to force retraining.')
    else:
        print(f'Training config: {config.conv_type}, hidden={config.hidden_dim}, epochs={config.epochs}')
        print(f'Device: {config.device}')
        print(f'Loss function: {objective_weights}')
        print(f'Number of training graphs: {len(train_graphs)}')
        print(f'Number of test graphs: {len(test_graphs)}')
        print()
        solver.train_multi_graph(train_dl, test_dataloader=test_dl, objective_weights=objective_weights)

    return solver, config, objective_weights


# ════════════════════════════════════════════════════════════
# Prediction + aggregation to substations
# ════════════════════════════════════════════════════════════

def predict_and_aggregate(solver, graphs, grids, region_dict, subs_dict):
    """Run prediction for all regions -> aggregate demand to substations (no post-processing)."""
    print('\n' + '=' * 60)
    print('Predicting edge weights -> aggregating demand to substations...')
    print('=' * 60)

    results_gnn = {}

    for loc in STUDY_REGIONS:
        grid_gdf, step_size_m = grids[loc]
        region_sub = region_dict[loc]
        subs_sub = subs_dict[loc]
        graph = graphs[loc]

        # Predict source->agent edge weights
        edge_weights_df = solver.predict_edge_weights(graph)
        print(f'{loc}: predicted {len(edge_weights_df)} edge weights')

        # Verify the per-source normalization
        weight_sums = edge_weights_df.groupby('source_node_idx')['predicted_weight'].sum()
        print(f'  per-source weight sum: mean={weight_sums.mean():.4f}, std={weight_sums.std():.6f}')

        # Map the edge weights back onto grid_gdf -> gnn_demand
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

        print(f'  Total GNN demand: {grid_gdf["gnn_demand"].sum():.1f} MVA')
        print(f'  Total region demand: {region_sub["Demand (MVA)"].sum():.1f} MVA')

        grids[loc] = (grid_gdf, step_size_m)

        # Voronoi allocation
        alloc_voronoi = allocator_registry.create('voronoi')
        voronoi_res = alloc_voronoi.allocate(grid_gdf, subs_sub)

        demands = grid_gdf['gnn_demand'].values

        subs_result_voronoi = subs_sub.copy()
        subs_result_voronoi['allocated_demand'] = 0.0
        for target_idx in range(len(subs_sub)):
            mask = voronoi_res.assignment == target_idx
            subs_result_voronoi.loc[target_idx, 'allocated_demand'] = demands[mask].sum()
        print(f'  Total voronoi_GNN allocated demand: {subs_result_voronoi["allocated_demand"].sum():.1f} MVA')

        # CIVD allocation
        coords_4326 = np.column_stack([subs_sub.geometry.x.values, subs_sub.geometry.y.values])
        cluster_gdf, centroid_gdf = do_clustering(coords_4326, method='hdbscan', min_cluster_size=2)
        centroid_gdf = centroid_gdf.reset_index(drop=True)
        print(f'  HDBSCAN cluster count: {cluster_gdf["cluster_label"].nunique()}')

        target_civd = subs_sub.copy()
        target_civd['cluster_label'] = cluster_gdf['cluster_label']

        civd_cache_path = OUTPUT_DIR / f'{loc}_civd_cache.pickle'
        if civd_cache_path.exists():
            with open(civd_cache_path, 'rb') as f:
                cache = pickle.load(f)
            civd_assignment = cache['assignment']
            print(f'  Reusing CIVD cache: {civd_cache_path.name}')
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
            print(f'  CIVD solve complete')

        subs_result_civd = subs_sub.copy()
        subs_result_civd['allocated_demand'] = 0.0
        cluster_demands = {}
        for label in np.unique(civd_assignment):
            mask = civd_assignment == label
            cluster_demands[label] = demands[mask].sum()
        for label, total_d in cluster_demands.items():
            members = cluster_gdf[cluster_gdf['cluster_label'] == label].index
            n_members = len(members)
            if n_members > 0:
                for idx in members:
                    if idx < len(subs_result_civd):
                        subs_result_civd.loc[idx, 'allocated_demand'] += total_d / n_members
        print(f'  Total civd_GNN allocated demand: {subs_result_civd["allocated_demand"].sum():.1f} MVA')

        results_gnn[loc] = {
            'voronoi_GNN': subs_result_voronoi,
            'civd_GNN': subs_result_civd,
        }

    return results_gnn, grids


# ════════════════════════════════════════════════════════════
# Metric evaluation + saving
# ════════════════════════════════════════════════════════════

def evaluate_and_save(results_gnn, subs_dict):
    """Compute RMSE / Corr / MAE and save the tables."""
    print('\n' + '=' * 60)
    print('Computing metrics...')
    print('=' * 60)

    # Read the baseline RMSE table
    rmse_csv = OUTPUT_DIR / 'all_regions_rmse.csv'
    rmse_table = pd.read_csv(rmse_csv, index_col=0)

    corr_csv = OUTPUT_DIR / 'all_regions_corr.csv'
    corr_table = pd.read_csv(corr_csv, index_col=0) if corr_csv.exists() else pd.DataFrame()

    mae_csv_path = OUTPUT_DIR / 'all_regions_mae.csv'
    mae_table = pd.read_csv(mae_csv_path, index_col=0) if mae_csv_path.exists() else pd.DataFrame()

    all_metrics = []
    for loc in STUDY_REGIONS:
        split = 'TRAIN' if loc in TRAIN_LOCATIONS else 'TEST'
        for method_name, subs_result in results_gnn[loc].items():
            m = evaluate_allocation(subs_result)
            rmse_table.loc[method_name, loc] = round(m['rmse'], 4)
            corr_table.loc[method_name, loc] = round(m['corr'], 4)
            mae_table.loc[method_name, loc] = round(m['mae'], 4)
            all_metrics.append({'region': loc, 'split': split, 'method': method_name, **m})
            print(f'[{split}] {loc} {method_name}: corr={m["corr"]:.4f}, RMSE={m["rmse"]:.4f}, MAE={m["mae"]:.4f}')

    # Save the tables
    rmse_table.to_csv(GNN_DIR / 'all_regions_rmse.csv')
    corr_table.to_csv(GNN_DIR / 'all_regions_corr.csv')
    mae_table.to_csv(GNN_DIR / 'all_regions_mae.csv')
    print(f'\nTables saved to: {GNN_DIR}')

    # Print the summary
    print(f'\n=== Updated RMSE table ===')
    print(rmse_table.to_string())

    gnn_metrics_df = pd.DataFrame(all_metrics).round(4)
    print('\n=== Detailed GNN method metrics ===')
    print(gnn_metrics_df[['split', 'region', 'method', 'corr', 'rmse', 'mae', 'conservation_error']].to_string())

    print('\n=== Train vs Test average metrics ===')
    summary = gnn_metrics_df.groupby(['split', 'method'])[['corr', 'rmse', 'mae']].mean().round(4)
    print(summary.to_string())

    return rmse_table, corr_table


# ════════════════════════════════════════════════════════════
# Leaderboard update
# ════════════════════════════════════════════════════════════

def update_leaderboard(config, objective_weights, rmse_table, corr_table):
    """Compute RRMSE and update the leaderboard."""
    print('\n' + '=' * 60)
    print('Updating leaderboard...')
    print('=' * 60)

    # 1. Compute the RRMSE for the current experiment
    per_region_rrmse = {}
    per_region_corr = {}
    for loc in STUDY_REGIONS:
        baseline_rmse = float(rmse_table.loc[BASELINE_METHOD, loc])
        gnn_rmse = float(rmse_table.loc[GNN_METHOD, loc])
        per_region_rrmse[loc] = round(gnn_rmse / baseline_rmse, 4) if baseline_rmse > 0 else None

        gnn_corr_val = corr_table.loc[GNN_METHOD, loc]
        per_region_corr[loc] = round(float(gnn_corr_val), 4) if pd.notna(gnn_corr_val) else None

    train_rrmse_vals = [per_region_rrmse[loc] for loc in TRAIN_LOCATIONS if per_region_rrmse[loc] is not None]
    test_rrmse_vals = [per_region_rrmse[loc] for loc in TEST_LOCATIONS if per_region_rrmse[loc] is not None]
    train_corr_vals = [per_region_corr[loc] for loc in TRAIN_LOCATIONS if per_region_corr[loc] is not None]
    test_corr_vals = [per_region_corr[loc] for loc in TEST_LOCATIONS if per_region_corr[loc] is not None]

    current_entry = {
        'experiment_name': EXPERIMENT_NAME,
        'timestamp': datetime.now().isoformat(),
        'test_median_rrmse': round(float(np.median(test_rrmse_vals)), 4),
        'train_median_rrmse': round(float(np.median(train_rrmse_vals)), 4),
        'test_median_corr': round(float(np.median(test_corr_vals)), 4),
        'train_median_corr': round(float(np.median(train_corr_vals)), 4),
        'per_region_rrmse': per_region_rrmse,
        'per_region_corr': per_region_corr,
        'config_snapshot': {
            'conv_type': config.conv_type,
            'hidden_dim': config.hidden_dim,
            'embedding_dim': config.embedding_dim,
            'epochs': config.epochs,
            'learning_rate': config.learning_rate,
            'objective_weights': objective_weights,
        },
    }

    # 2. Load the leaderboard
    if LEADERBOARD_PATH.exists():
        with open(LEADERBOARD_PATH, 'r', encoding='utf-8') as f:
            leaderboard = json.load(f)
    else:
        leaderboard = {
            'version': 1,
            'baseline_method': BASELINE_METHOD,
            'gnn_method': GNN_METHOD,
            'global_best': None,
            'history': [],
        }

    # 3. Compare and update
    prev_best = leaderboard['global_best']
    is_best = (prev_best is None) or (current_entry['test_median_rrmse'] < prev_best['test_median_rrmse'])

    if is_best:
        leaderboard['global_best'] = {k: v for k, v in current_entry.items()}

    history_entry = {
        'experiment_name': EXPERIMENT_NAME,
        'timestamp': current_entry['timestamp'],
        'test_median_rrmse': current_entry['test_median_rrmse'],
        'train_median_rrmse': current_entry['train_median_rrmse'],
        'test_median_corr': current_entry['test_median_corr'],
        'train_median_corr': current_entry['train_median_corr'],
        'is_best': is_best,
    }
    leaderboard['history'].append(history_entry)

    # 4. Save
    with open(LEADERBOARD_PATH, 'w', encoding='utf-8') as f:
        json.dump(leaderboard, f, indent=2, ensure_ascii=False)

    # 5. Print the results
    print(f'Experiment: {EXPERIMENT_NAME}')
    print(f'Leaderboard: {LEADERBOARD_PATH}')
    print()

    rrmse_rows = []
    for loc in STUDY_REGIONS:
        split = 'TRAIN' if loc in TRAIN_LOCATIONS else 'TEST'
        rrmse_val = per_region_rrmse[loc]
        corr_val = per_region_corr[loc]
        baseline_rmse = float(rmse_table.loc[BASELINE_METHOD, loc])
        gnn_rmse = float(rmse_table.loc[GNN_METHOD, loc])
        status = '< baseline' if rrmse_val < 1.0 else '> baseline'
        rrmse_rows.append({
            'Region': loc, 'Split': split,
            f'RMSE({BASELINE_METHOD})': round(baseline_rmse, 2),
            f'RMSE({GNN_METHOD})': round(gnn_rmse, 2),
            'RRMSE': rrmse_val, 'Corr': corr_val, 'Status': status,
        })

    rrmse_df = pd.DataFrame(rrmse_rows)
    print('=== Per-region RRMSE ===')
    print(rrmse_df.to_string(index=False))

    print(f'\n=== Current experiment summary ===')
    print(f'  Test  median RRMSE: {current_entry["test_median_rrmse"]:.4f}')
    print(f'  Train median RRMSE: {current_entry["train_median_rrmse"]:.4f}')
    print(f'  Test  median Corr:  {current_entry["test_median_corr"]:.4f}')
    print(f'  Train median Corr:  {current_entry["train_median_corr"]:.4f}')

    best = leaderboard['global_best']
    if is_best:
        if prev_best is None:
            print(f'\n*** First record, set as the global best ***')
        else:
            delta = prev_best['test_median_rrmse'] - current_entry['test_median_rrmse']
            print(f'\n*** New global best! (RRMSE reduced by {delta:.4f}, previous best: {prev_best["experiment_name"]}) ***')
    else:
        delta = current_entry['test_median_rrmse'] - best['test_median_rrmse']
        print(f'\nDid not beat the global best "{best["experiment_name"]}" (RRMSE gap +{delta:.4f})')
        print(f'  Best test_median_rrmse: {best["test_median_rrmse"]:.4f}')

        wins, losses = [], []
        for loc in TEST_LOCATIONS:
            cur = per_region_rrmse[loc]
            bst = best['per_region_rrmse'].get(loc)
            if cur is not None and bst is not None:
                if cur < bst:
                    wins.append(f'{loc}({cur:.3f} vs {bst:.3f})')
                else:
                    losses.append(f'{loc}({cur:.3f} vs {bst:.3f})')
        if wins:
            print(f'  Winning regions: {", ".join(wins)}')
        if losses:
            print(f'  Losing regions: {", ".join(losses)}')

    print(f'\n=== Leaderboard history ({len(leaderboard["history"])} entries) ===')
    hist_df = pd.DataFrame(leaderboard['history'])
    hist_df = hist_df.sort_values('test_median_rrmse')
    print(hist_df[['experiment_name', 'test_median_rrmse', 'train_median_rrmse', 'test_median_corr', 'is_best']].to_string())


# ════════════════════════════════════════════════════════════
# Main entry point
# ════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print(f'PyTorch: {torch.__version__}')
    print(f'CUDA: {torch.cuda.is_available()}')
    print(f'Experiment name: {EXPERIMENT_NAME}')
    print(f'Agent features: {len(AGENT_FEATURE_COLS)} dims')
    print(f'Source features: {len(SOURCE_FEATURE_COLS)} dims')
    print(f'Model path: {MODEL_PATH}')
    print(f'OVERWRITE: {OVERWRITE}')
    print(f'Training regions: {len(TRAIN_LOCATIONS)}')
    print(f'Test regions: {len(TEST_LOCATIONS)}')
    print(f'Agent adjacency: {AGENT_CONNECTIVITY}')
    print()

    # 1. Load data
    grids, ntl_dict, proximity_dict, rci_dict, region_dict, subs_dict = load_data()

    # 2. Build graphs
    graphs = build_graphs(grids, ntl_dict, proximity_dict, rci_dict, region_dict, subs_dict)

    # 3. Train
    solver, config, objective_weights = train_model(graphs)

    # 4. Predict + aggregate
    results_gnn, grids = predict_and_aggregate(solver, graphs, grids, region_dict, subs_dict)

    # 5. Evaluate metrics
    rmse_table, corr_table = evaluate_and_save(results_gnn, subs_dict)

    # 6. Leaderboard
    update_leaderboard(config, objective_weights, rmse_table, corr_table)

    print('\n' + '=' * 60)
    print('All done!')
    print('=' * 60)
