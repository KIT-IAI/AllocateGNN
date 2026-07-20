"""
027 - Feature-level fusion arm training script

Adapted from 005_kfold_prior_training.py (005 and all of its output
artifacts are treated as a frozen, read-only reference). The only
substantive scientific change is moving NTL and Proximity from the
"loss prior / post-hoc correction" channel into an **input feature**
channel (feature-level fusion): AGENT_FEATURE_COLS gains two new columns,
'ntl_feat' / 'prox_feat', taking the agent feature dimensionality from
5 to 7.
- Transform = log1p, using the same functional form as the NTL/proximity
  prior losses used elsewhere in this suite (torch.log(1 + clamp(x, min=0)));
- Standardization = per-region z-score (computed over the agent set within
  each location, numpy convention with ddof=0); because each region is
  standardized independently, no statistics are shared between train/test
  folds, so this is inherently free of cross-fold leakage.

Engineering changes relative to 005:
- EXP0_DIR now points to results/exp_r212_fusion (a single change that
  redirects both training output and the graph cache; load_or_cache_all
  is not imported from the original 005 module because its default
  argument would be evaluated at import time and would bind to the old
  path);
- PROJECT_ROOT uses the same walk-up directory resolution as 020 (walks
  upward to find the directory containing SpatialAllocation);
- CONFIG_MAP only keeps the baseline configuration (the full sweep would
  be 3 seeds x 4 folds = 12 combinations);
- Added --cache-only: builds the graph cache, validates the feature
  dimensionality, and exits without entering the training loop;
- Everything else matches 005 line-for-line: tau=0.01 (both in the
  resume branch and the normal branch), AGENT_CONNECTIVITY=None,
  STATIC_DIR is only read for the CIVD cache, and the post-training
  inference evaluation and all post-hoc correction arms are unchanged.

Usage:
    python 027_train_feature_fusion.py --cache-only                       # build graph cache only
    python 027_train_feature_fusion.py --config baseline --seed 42
    python 027_train_feature_fusion.py --all --seeds 42 123 456           # full sweep, 12 combinations

Configuration map:
    baseline:   {'landuse_prediction_loss': 1.0}
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
from sklearn.model_selection import KFold
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from shapely.geometry import Point

# Project root: walk upward to find the directory containing
# SpatialAllocation (same approach as 020; 005's fixed parent-count
# approach would point outside the repository)
SCRIPT_DIR = Path(__file__).resolve().parent
_p = SCRIPT_DIR
while not (_p / 'SpatialAllocation').exists():
    if _p.parent == _p:
        raise RuntimeError('Could not find repository root (SpatialAllocation package)')
    _p = _p.parent
PROJECT_ROOT = _p
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
STATIC_DIR = SCRIPT_DIR / 'results' / 'static_allocation'
# Fusion-arm output directory (a single change redirects both training
# artifacts and the graph cache; the variable name is kept as EXP0_DIR to
# minimize the diff against 005 -- the default argument and the main
# entry point both derive from it)
EXP0_DIR = SCRIPT_DIR / 'results' / 'exp_r212_fusion'

N_FOLDS = 4

# --- Region configuration ---
ALL_LOCATIONS = [
    'London',
    'TLH2', 'TLH3', 'TLJ1', 'TLF1', 'TLF2',
    'TLC1', 'TLC2', 'TLD6', 'TLG1', 'TLG2', 'TLE4',
    'TLH1', 'TLE3', 'TLD3', 'TLD4',
]

# --- Feature control ---
# Adds two fusion input columns, NTL and Proximity (agent feature
# dimensionality 5 -> 7); the column values are written inside
# build_graphs() before the feature-column selection step (log1p +
# per-region z-score)
AGENT_FEATURE_COLS = [
    'lu_residential_prop', 'lu_commercial_prop', 'lu_industrial_prop',
    'lu_agricultural_prop', 'lu_others_prop',
    'ntl_feat', 'prox_feat',
]

SOURCE_FEATURE_COLS = [
    'residential_percent', 'commercial_percent', 'industrial_percent',
    'agricultural_percent', 'others_percent',
]

# --- Post-correction constants ---
RCI_THRESHOLD = 0.5
PROXIMITY_GAMMA = 2.0
DIST_CLAMP_KM = 0.01
TARGET_CRS = 'EPSG:27700'

# --- Landuse mapping ---
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

# --- Configuration map ---
# Only baseline is retained: the fusion arm's signal entry point is the
# input feature channel rather than the loss prior, so the ntl/proximity/
# ntl_prox prior configurations are semantically inconsistent with this
# experiment and have all been removed (the full sweep is 12 combinations)
CONFIG_MAP = {
    'baseline': {
        'objective_weights': {'landuse_prediction_loss': 1.0},
        'epochs': 200,
    },
}

AGENT_CONNECTIVITY = None


# ════════════════════════════════════════════════════════════
# Utility functions
# ════════════════════════════════════════════════════════════

def compute_demand(grid_gdf, region_sub, weighter_result, demand_col='demand'):
    """Combine weighter output weights with regional percentages to produce grid-level demand."""
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
    """Load data for all regions and precompute NTL/Proximity/RCI."""
    print('=' * 60)
    print('Loading data and precomputing prior features...')
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
        subs_dict[loc] = substations_gdf[
            substations_gdf['ITL3'].isin(study_itl3)
        ].copy().reset_index(drop=True)

    # Precompute proximity scores (uses the same PROXIMITY_GAMMA as the post-correction step)
    proximity_dict = {}
    for loc in ALL_LOCATIONS:
        grid_gdf, _ = grids[loc]
        subs_sub = subs_dict[loc]
        prox_scores = ProximityCorrector.compute_scores(grid_gdf, subs_sub, gamma=PROXIMITY_GAMMA)
        proximity_dict[loc] = prox_scores

    # Precompute RCI mask
    rci_dict = {}
    for loc in ALL_LOCATIONS:
        grid_gdf, _ = grids[loc]
        rci = (grid_gdf['lu_residential_prop'] + grid_gdf['lu_commercial_prop']
               + grid_gdf['lu_industrial_prop']).values
        rci_dict[loc] = rci > RCI_THRESHOLD

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
    return grids, ntl_dict, proximity_dict, rci_dict, region_dict, subs_dict


# ════════════════════════════════════════════════════════════
# Graph construction (including NTL/Proximity/RCI injection)
# ════════════════════════════════════════════════════════════

def _region_zscore_log1p(values, label):
    """Fusion input column transform: log1p followed by per-region z-score.

    - Values are clipped to [0, +inf) and then passed through log1p,
      matching the same functional form used for the NTL/proximity prior
      losses elsewhere in this suite (torch.log(1 + clamp(x, min=0)));
    - The z-score is computed over **all agents within a single location**
      (numpy convention, ddof=0); because each region's statistics are
      computed independently, there is zero information sharing between
      train/test folds, so this is inherently free of cross-fold leakage;
    - Raises an error when std is zero or non-finite rather than silently
      continuing, because silently zeroing out the NTL feature would be
      one of the most dangerous failure modes here -- missing or
      degenerate inputs must fail loudly rather than be masked.
    """
    arr = np.log1p(np.clip(np.asarray(values, dtype=np.float64), 0.0, None))
    std = arr.std()  # numpy default is ddof=0
    if not np.isfinite(std) or std <= 0:
        raise ValueError(f'Fusion input column has zero or invalid within-region variance ({label}); refusing to continue silently')
    return ((arr - arr.mean()) / std).astype(np.float32)


def build_graphs(grids, ntl_dict, proximity_dict, rci_dict, region_dict, subs_dict,
                 inject_priors=True):
    """Build HeteroData graphs for all regions. Injects NTL/Proximity/RCI when inject_priors=True."""
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

        # -- The only substantive scientific change: write the two fusion
        # input columns before the feature-column selection step --
        # Raw values were already prepared in load_data() (ntl_dict /
        # proximity_dict, aligned by grid row position)
        assert len(ntl_dict[loc]) == len(gdf_a_scaled), f'{loc}: ntl length does not match agent count'
        assert len(proximity_dict[loc]) == len(gdf_a_scaled), f'{loc}: prox length does not match agent count'
        gdf_a_scaled['ntl_feat'] = _region_zscore_log1p(ntl_dict[loc], f'{loc}/ntl_feat')
        gdf_a_scaled['prox_feat'] = _region_zscore_log1p(proximity_dict[loc], f'{loc}/prox_feat')

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

        # Inject NTL / Proximity / RCI (required by the prior losses)
        if inject_priors:
            hetero_data['agent'].ntl_values = torch.tensor(
                ntl_dict[loc], dtype=torch.float32)
            hetero_data['agent'].proximity_scores = torch.tensor(
                proximity_dict[loc], dtype=torch.float32)
            hetero_data['agent'].rci_mask = torch.tensor(
                rci_dict[loc], dtype=torch.bool)

        graphs[loc] = hetero_data
        print(f'  {loc}: agent={hetero_data["agent"].num_nodes}, '
              f'source={hetero_data["source"].num_nodes}, '
              f'edges={hetero_data["source", "connects_to", "agent"].edge_index.shape[1]}')

    return graphs


# ════════════════════════════════════════════════════════════
# Prediction + post-correction + aggregation (single region)
# ════════════════════════════════════════════════════════════

def predict_and_evaluate_location(loc, solver, graphs, grids, ntl_dict,
                                   region_dict, subs_dict,
                                   save_grid_demands_dir=None):
    """Run prediction -> post-correction -> aggregation -> evaluation for a
    single region, returning {method: metrics_dict}.

    save_grid_demands_dir: if not None, saves the grid_demands pickle to
    this directory.
    """
    grid_gdf, step_size_m = grids[loc]
    region_sub = region_dict[loc]
    subs_sub = subs_dict[loc]
    graph = graphs[loc]
    ntl_values = ntl_dict[loc]

    # Predict edge weights
    edge_weights_df = solver.predict_edge_weights(graph)

    # Map onto grid_gdf -> gnn_demand
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

    # Save grid_demands (used by downstream experiments 3/5/7)
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

    # Cache CIVD allocation
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
# K-Fold training main loop
# ════════════════════════════════════════════════════════════

def load_or_cache_all(cache_dir: Path = EXP0_DIR / 'graph_cache'):
    """Load data and build graphs, preferring to read from the on-disk cache when available. Returns (graphs, grids, ntl_dict, region_dict, subs_dict)."""
    cache_file = cache_dir / 'cached_graphs.pickle'

    if cache_file.exists():
        print(f'Loading graphs from cache: {cache_file}')
        with open(cache_file, 'rb') as f:
            cached = pickle.load(f)
        return (cached['graphs'], cached['grids'], cached['ntl_dict'],
                cached['region_dict'], cached['subs_dict'])

    # No cache -> full build
    grids, ntl_dict, proximity_dict, rci_dict, region_dict, subs_dict = load_data()
    graphs = build_graphs(
        grids, ntl_dict, proximity_dict, rci_dict, region_dict, subs_dict,
        inject_priors=True,  # Always inject priors (attributes unused by baseline do not affect training)
    )

    # Save cache
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump({
            'graphs': graphs,
            'grids': grids,
            'ntl_dict': ntl_dict,
            'region_dict': region_dict,
            'subs_dict': subs_dict,
        }, f)
    print(f'Graph cache saved: {cache_file}')

    return graphs, grids, ntl_dict, region_dict, subs_dict


def assert_training_complete(model_path: Path, expected_epochs: int) -> None:
    """Hardened check that a resumed training run actually completed.

    model.pth is a best-checkpoint snapshot that can be written to disk at
    any point during the training loop, so a hard interruption (crash,
    kill, power loss) can leave a partial checkpoint behind;
    *_training_log.json is only written once the full epoch loop has
    finished. Therefore the mere presence of model.pth is not sufficient
    evidence that training completed -- the log must also exist and
    report the full expected number of epochs, otherwise this refuses to
    proceed and instructs the caller how to recover.
    """
    log_path = model_path.with_name(model_path.name[:-4] + '_training_log.json')
    if not log_path.exists():
        raise SystemExit(f'Truncated training run: {model_path} exists but '
                         f'{log_path.name} is missing -- move this directory '
                         'aside for inspection and rerun from scratch '
                         '(do not resume directly).')
    with open(log_path, encoding='utf-8') as f:
        _log = json.load(f)
    n_logged = len(_log.get('train_losses', {}).get('total', []))
    if n_logged != expected_epochs:
        raise SystemExit(f'Incomplete training log: {log_path} '
                         f'({n_logged}/{expected_epochs} epochs) -- move '
                         'this directory aside for inspection and rerun '
                         'from scratch (do not resume directly).')


def run_kfold_training(config_name: str, seed: int,
                       graphs, grids, ntl_dict, region_dict, subs_dict):
    """Run K-Fold cross-validation training for a single configuration x seed combination."""

    exp_config = CONFIG_MAP[config_name]
    objective_weights = exp_config['objective_weights']
    epochs = exp_config['epochs']

    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Output directory
    seed_dir = EXP0_DIR / f'seed_{seed}' / config_name
    seed_dir.mkdir(parents=True, exist_ok=True)

    print(f'\n{"=" * 60}')
    print(f'Configuration: {config_name} | seed: {seed} | epochs: {epochs}')
    print(f'Loss function: {objective_weights}')
    print(f'Output directory: {seed_dir}')
    print(f'{"=" * 60}\n')

    # 2. K-Fold split (random_state follows seed)
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
        print(f'  Training regions ({len(train_locs)}): {train_locs}')
        print(f'  Test regions ({len(test_locs)}): {test_locs}')
        print(f'{"=" * 60}')

        fold_dir = seed_dir / f'fold{fold_idx + 1}'
        fold_dir.mkdir(parents=True, exist_ok=True)
        model_path = fold_dir / 'model.pth'

        # -- Check whether this fold is already fully complete (both model and metric CSVs exist) --
        fold_csvs = [fold_dir / f'{name}.csv' for name in ('rmse', 'mae', 'corr')]
        if model_path.exists() and all(f.exists() for f in fold_csvs):
            assert_training_complete(model_path, epochs)
            # Check whether grid_demands are missing -- if so, load the
            # model and rerun inference to fill them in
            grid_demands_dir = fold_dir / 'grid_demands'
            expected_gd = [grid_demands_dir / f'{loc}_grid_demands.pickle'
                           for loc in ALL_LOCATIONS]
            missing_gd = [p for p in expected_gd if not p.exists()]

            if missing_gd:
                print(f'Fold {fold_idx + 1} metrics already complete, but {len(missing_gd)} grid_demands files are missing, running inference to fill them in...')

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
                train_graphs_tmp = [graphs[loc] for loc in train_locs]
                train_dl_tmp = DataLoader(train_graphs_tmp, batch_size=1, shuffle=False)
                solver.init_model(train_dl_tmp, objective_weights)
                solver._load_checkpoint()

                for loc in ALL_LOCATIONS:
                    gd_file = grid_demands_dir / f'{loc}_grid_demands.pickle'
                    if gd_file.exists():
                        continue
                    print(f'  Generating missing grid_demands: {loc}')
                    predict_and_evaluate_location(
                        loc, solver, graphs, grids, ntl_dict, region_dict, subs_dict,
                        save_grid_demands_dir=grid_demands_dir,
                    )
                print(f'Fold {fold_idx + 1} grid_demands backfill complete')
            else:
                print(f'Fold {fold_idx + 1} already complete (including grid_demands), skipping')

            saved_rmse = pd.read_csv(fold_dir / 'rmse.csv', index_col=0)
            saved_mae = pd.read_csv(fold_dir / 'mae.csv', index_col=0)
            saved_corr = pd.read_csv(fold_dir / 'corr.csv', index_col=0)
            fold_rmse_tables.append(saved_rmse.T.to_dict())
            fold_mae_tables.append(saved_mae.T.to_dict())
            fold_corr_tables.append(saved_corr.T.to_dict())
            continue

        # -- Normal training / load model + evaluate --
        # Dynamically adjust the scheduler based on epochs
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

        train_graphs = [graphs[loc] for loc in train_locs]
        test_graphs = [graphs[loc] for loc in test_locs]
        train_dl = DataLoader(train_graphs, batch_size=1, shuffle=False)
        test_dl = DataLoader(test_graphs, batch_size=1, shuffle=False)

        # Check whether a trained model already exists -> skip training
        if model_path.exists():
            assert_training_complete(model_path, epochs)
            print(f'Model already exists at {model_path}, skipping training; initializing model structure and loading checkpoint')
            solver.init_model(train_dl, objective_weights)
            solver._load_checkpoint()
        else:
            print(f'Training configuration: {config.conv_type}, hidden={config.hidden_dim}, epochs={config.epochs}')
            print(f'Device: {config.device}')
            solver.train_multi_graph(train_dl, test_dataloader=test_dl,
                                     objective_weights=objective_weights)
            print(f'Fold {fold_idx + 1} training complete')

        # Predict & evaluate for all regions
        fold_rmse = {}
        fold_mae = {}
        fold_corr = {}

        grid_demands_dir = fold_dir / 'grid_demands'

        for loc in ALL_LOCATIONS:
            role = 'TRAIN' if loc in train_locs else 'TEST'
            print(f'\n  Evaluating {loc} ({role})...')

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

        # Save this fold's tables
        pd.DataFrame(fold_rmse).T.to_csv(fold_dir / 'rmse.csv')
        pd.DataFrame(fold_mae).T.to_csv(fold_dir / 'mae.csv')
        pd.DataFrame(fold_corr).T.to_csv(fold_dir / 'corr.csv')

    # ════════════════════════════════════════════════════════════
    # Summarize K-Fold results
    # ════════════════════════════════════════════════════════════

    print(f'\n{"=" * 60}')
    print('Summarizing K-Fold results...')
    print(f'{"=" * 60}')

    all_methods = sorted(fold_rmse_tables[0].keys())
    kf_splits = list(KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed).split(location_array))

    # Only take the fold's results for the region when it was in the test set
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

    print(f'\n=== K-Fold Test Set RMSE ===')
    print(test_rmse_df.to_string())

    print(f'\n=== K-Fold Test Set MAE ===')
    print(test_mae_df.to_string())

    print(f'\n=== K-Fold Test Set Correlation ===')
    print(test_corr_df.to_string())

    # Save split information
    split_info = {}
    for fold_idx, (train_indices, test_indices) in enumerate(kf_splits):
        split_info[f'fold_{fold_idx + 1}'] = {
            'train': location_array[train_indices].tolist(),
            'test': location_array[test_indices].tolist(),
        }
    with open(seed_dir / 'kfold_splits.json', 'w', encoding='utf-8') as f:
        json.dump(split_info, f, ensure_ascii=False, indent=2)

    print(f'\nAll results saved to: {seed_dir}')


# ════════════════════════════════════════════════════════════
# Main entry point
# ════════════════════════════════════════════════════════════

ALL_CONFIGS = list(CONFIG_MAP.keys())
DEFAULT_SEEDS = [42, 123, 456]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature-level fusion arm K-Fold training')
    parser.add_argument('--config', type=str,
                        choices=['baseline'],
                        help='Training configuration (mutually exclusive with --all; the fusion arm only has baseline)')
    parser.add_argument('--seed', type=int,
                        help='Random seed (mutually exclusive with --all)')
    parser.add_argument('--all', action='store_true',
                        help='Run all configurations x all seeds')
    parser.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS,
                        help=f'Used together with --all to specify the seed list (default {DEFAULT_SEEDS})')
    parser.add_argument('--configs', type=str, nargs='+',
                        choices=['baseline'],
                        default=ALL_CONFIGS,
                        help=f'Used together with --all to specify the configuration list (the fusion arm only has baseline)')
    parser.add_argument('--rebuild-cache', action='store_true',
                        help='Force rebuild of the graph cache (use after data changes)')
    parser.add_argument('--cache-only', action='store_true',
                        help='Build/validate the graph cache only, then exit '
                             'without entering the training loop (useful for '
                             'preparing the cache ahead of a separate training run)')
    args = parser.parse_args()

    # Argument validation (--cache-only does not need config/seed)
    if args.cache_only:
        run_configs = []
        run_seeds = []
    elif args.all:
        run_configs = args.configs
        run_seeds = args.seeds
    elif args.config and args.seed is not None:
        run_configs = [args.config]
        run_seeds = [args.seed]
    else:
        parser.error('Use --all to run everything, or specify both --config '
                     'and --seed, or use --cache-only to build the graph '
                     'cache only')

    print(f'PyTorch: {torch.__version__}')
    print(f'CUDA: {torch.cuda.is_available()}')

    # Data & graphs are loaded/built only once (or read from cache)
    cache_dir = EXP0_DIR / 'graph_cache'
    if args.rebuild_cache:
        cache_file = cache_dir / 'cached_graphs.pickle'
        if cache_file.exists():
            cache_file.unlink()
            print('Deleted old cache, will rebuild')
    graphs, grids, ntl_dict, region_dict, subs_dict = load_or_cache_all(cache_dir)

    # -- --cache-only: validate feature dimensionality and exit directly
    # (without entering the training loop) --
    if args.cache_only:
        print('\n' + '=' * 60)
        print('--cache-only: validating graph cache agent feature dimensionality...')
        print('=' * 60)
        expected_dim = len(AGENT_FEATURE_COLS)
        for loc in ALL_LOCATIONS:
            dim = graphs[loc]['agent'].x.shape[1]
            status = 'OK' if dim == expected_dim else f'expected {expected_dim}'
            print(f'  {loc}: agent feature dim = {dim} [{status}]')
            assert dim == expected_dim, (
                f'{loc}: agent feature dim {dim} != {expected_dim} -- '
                f'this may be a stale cache, rebuild with --rebuild-cache')
        print(f'\nGraph cache ready: {cache_dir / "cached_graphs.pickle"}')
        print('Cache build complete, did not enter the training loop (--cache-only)')
        sys.exit(0)

    total = len(run_configs) * len(run_seeds)
    current = 0
    for cfg in run_configs:
        for s in run_seeds:
            current += 1
            print(f'\n{"#" * 60}')
            print(f'# Task {current}/{total}: config={cfg}, seed={s}')
            print(f'{"#" * 60}')
            run_kfold_training(cfg, s, graphs, grids, ntl_dict, region_dict, subs_dict)

    print('\n' + '=' * 60)
    print(f'All done! Ran {total} tasks in total')
    print('=' * 60)
