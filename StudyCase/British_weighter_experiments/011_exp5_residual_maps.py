"""
011 Spatial Residual Maps (Exp 5) - Voronoi Baseline

Selects representative regions; 4 methods x N region subplots.
Methods: GPM(voronoi_gpm), GPMpostNP(best static), GNNpostP(best GNN), GNNpostNP(antagonism case)

Usage:
    python 011_exp5_residual_maps.py
"""

import sys
import pickle
import json
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scienceplots
plt.style.use(['science', 'no-latex'])

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from SpatialAllocation.Allocator import allocator_registry
from SpatialAllocation.Weighter import weighter_registry
from SpatialAllocation.FeatureExtractor.correctors import corrector_registry
from SpatialAllocation.FeatureExtractor.correctors.proximity_corrector import ProximityCorrector

DATA_DIR = SCRIPT_DIR / 'results' / 'intermediate'
ASSEMBLED_DIR = DATA_DIR / 'features' / 'assembled'
EXTRACTED_DIR = DATA_DIR / 'features' / 'extracted'
EXP0_DIR = SCRIPT_DIR / 'results' / 'exp0_kfold_prior'

LU_COLS = ['lu_residential_prop', 'lu_commercial_prop', 'lu_industrial_prop',
           'lu_agricultural_prop', 'lu_others_prop']
PCT_COLS = ['residential_percent', 'commercial_percent', 'industrial_percent',
            'agricultural_percent', 'others_percent']
OUTPUT_DIR = SCRIPT_DIR / 'results' / 'exp5_residual_maps'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_REGIONS = ['London', 'TLD4', 'TLG2']
SEED = 42

ALL_LOCATIONS = [
    'London',
    'TLH2', 'TLH3', 'TLJ1', 'TLF1', 'TLF2',
    'TLC1', 'TLC2', 'TLD6', 'TLG1', 'TLG2', 'TLE4',
    'TLH1', 'TLE3', 'TLD3', 'TLD4',
]


def find_fold_for_region(loc, config_name, seed):
    """Find the fold number in which this region was used as the test set."""
    splits_path = EXP0_DIR / f'seed_{seed}' / config_name / 'kfold_splits.json'
    if not splits_path.exists():
        return None
    with open(splits_path, 'r') as f:
        splits = json.load(f)
    for fold_name, info in splits.items():
        if loc in info['test']:
            return fold_name.split('_')[1]
    return None


def load_substation_residuals(loc, config_name, demand_key):
    """Compute substation-level residuals = predicted - actual (Voronoi partition)."""
    fold_num = find_fold_for_region(loc, config_name, SEED)
    if fold_num is None:
        return None, None

    gd_path = (EXP0_DIR / f'seed_{SEED}' / config_name
               / f'fold{fold_num}' / 'grid_demands' / f'{loc}_grid_demands.pickle')
    if not gd_path.exists():
        return None, None

    with open(gd_path, 'rb') as f:
        grid_demands = pickle.load(f)

    if demand_key not in grid_demands:
        return None, None

    with open(ASSEMBLED_DIR / f'{loc}_grid_points.pickle', 'rb') as f:
        grid_gdf, _ = pickle.load(f)

    substations_gdf = gpd.read_file(str(DATA_DIR / 'substations.gpkg'))
    study_itl3 = grid_gdf['ITL3'].unique()
    subs_sub = substations_gdf[substations_gdf['ITL3'].isin(study_itl3)].copy().reset_index(drop=True)

    # Voronoi partition
    alloc_voronoi = allocator_registry.create('voronoi')
    voronoi_res = alloc_voronoi.allocate(grid_gdf, subs_sub)

    demand_arr = grid_demands[demand_key]
    predicted = np.zeros(len(subs_sub))
    for target_idx in range(len(subs_sub)):
        mask = voronoi_res.assignment == target_idx
        predicted[target_idx] = demand_arr[mask].sum()

    actual = subs_sub['Demand (MVA)'].values
    residual = predicted - actual

    subs_result = subs_sub.copy()
    subs_result['residual'] = residual
    subs_result['predicted'] = predicted

    return subs_result, actual


def _compute_demand(grid_gdf, region_sub, weighter_result, demand_col='demand'):
    """Combine weighter output weights with regional percentages to derive grid-cell demand."""
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


def load_static_residuals(loc, static_method='voronoi_prox2_ntl_gpm'):
    """Recompute static-method residuals from source data (independent of the 003 pickle)."""
    with open(ASSEMBLED_DIR / f'{loc}_grid_points.pickle', 'rb') as f:
        grid_gdf, _ = pickle.load(f)

    substations_gdf = gpd.read_file(str(DATA_DIR / 'substations.gpkg'))
    region_gdf_local = gpd.read_file(str(DATA_DIR / 'ITL3_region.gpkg'))
    study_itl3 = grid_gdf['ITL3'].unique()
    region_sub = region_gdf_local[region_gdf_local['ITL3'].isin(study_itl3)].copy()
    subs_sub = substations_gdf[substations_gdf['ITL3'].isin(study_itl3)].copy().reset_index(drop=True)

    # Base demand weights
    if 'gpm' in static_method:
        weighter = weighter_registry.create('gpm', config={
            'mode': 'categorical', 'proportion_columns': LU_COLS})
    else:
        weighter = weighter_registry.create('uniform', config={})
    w_res = weighter.compute(grid_gdf, target_gdf=subs_sub)
    grid_gdf = _compute_demand(grid_gdf, region_sub, w_res, demand_col='demand')
    demand_col = 'demand'

    # NTL correction
    if 'ntl' in static_method:
        ntl_path = EXTRACTED_DIR / f'{loc}_ntl.npz'
        ntl_values = np.load(ntl_path, allow_pickle=True)['data'][:, 0]
        ntl_corrector = corrector_registry.create('ntl')
        ntl_corrector.correct(grid_gdf, region_sub, demand_col,
                              ntl_values, 'ntl_demand')
        demand_col = 'ntl_demand'

    # Proximity correction
    if 'prox' in static_method:
        gamma = 2.0 if 'prox2' in static_method else 1.0
        prox_scores = ProximityCorrector.compute_scores(
            grid_gdf, subs_sub, gamma=gamma)
        prox_corrector = corrector_registry.create('proximity')
        prox_corrector.correct(grid_gdf, region_sub, demand_col,
                               prox_scores, 'prox_demand')
        demand_col = 'prox_demand'

    # Voronoi allocation to substations
    alloc = allocator_registry.create('voronoi')
    voronoi_res = alloc.allocate(grid_gdf, subs_sub)

    demands = grid_gdf[demand_col].values
    subs_result = subs_sub.copy()
    subs_result['allocated_demand'] = 0.0
    for target_idx in range(len(subs_sub)):
        mask = voronoi_res.assignment == target_idx
        subs_result.loc[target_idx, 'allocated_demand'] = demands[mask].sum()

    subs_result['residual'] = subs_result['allocated_demand'] - subs_result['Demand (MVA)']
    return subs_result


def plot_residual_maps():
    """Plot spatial residual maps (Voronoi baseline)."""
    print('=' * 60)
    print('Exp 5: Spatial Residual Maps (Voronoi Baseline)')
    print('=' * 60)

    # 4 methods (Figure 2)
    methods = [
        ('GPM', 'static', 'voronoi_gpm', None),
        ('GPMpostNP', 'static', 'voronoi_prox2_ntl_gpm', None),
        ('GNNpostP', 'baseline', None, 'prox_gnn_demand'),
        ('GNNpostNP', 'baseline', None, 'ntl_prox_gnn_demand'),
    ]

    region_gdf = gpd.read_file(str(DATA_DIR / 'ITL3_region.gpkg'))

    fig, axes = plt.subplots(len(TARGET_REGIONS), len(methods),
                              figsize=(3.5, 2.5))

    if len(TARGET_REGIONS) == 1:
        axes = axes.reshape(1, -1)

    # Collect all residual values to determine a unified color scale
    all_residuals = []
    residual_data = {}

    for row_idx, loc in enumerate(TARGET_REGIONS):
        for col_idx, (title, config_name, static_method, demand_key) in enumerate(methods):
            if static_method:
                subs_result = load_static_residuals(loc, static_method)
            else:
                subs_result, _ = load_substation_residuals(loc, config_name, demand_key)

            residual_data[(row_idx, col_idx)] = subs_result
            if subs_result is not None:
                all_residuals.extend(subs_result['residual'].values)

    if not all_residuals:
        print('[Warning] No available residual data')
        plt.close(fig)
        return

    # Unified color scale
    vmax = max(abs(np.percentile(all_residuals, 5)),
               abs(np.percentile(all_residuals, 95)))
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    for row_idx, loc in enumerate(TARGET_REGIONS):
        # Load region boundary
        with open(ASSEMBLED_DIR / f'{loc}_grid_points.pickle', 'rb') as f:
            grid_gdf, _ = pickle.load(f)
        study_itl3 = grid_gdf['ITL3'].unique()
        region_boundary = region_gdf[region_gdf['ITL3'].isin(study_itl3)]

        for col_idx, (title, _, _, _) in enumerate(methods):
            ax = axes[row_idx, col_idx]
            subs_result = residual_data[(row_idx, col_idx)]

            # Base map: region boundary
            if not region_boundary.empty:
                region_boundary.plot(ax=ax, facecolor='lightgray', edgecolor='black',
                                    linewidth=0.5, alpha=0.3)

            if subs_result is not None:
                subs_plot = subs_result.to_crs(region_boundary.crs) if region_boundary.crs else subs_result
                scatter = ax.scatter(
                    subs_plot.geometry.x, subs_plot.geometry.y,
                    c=subs_plot['residual'].values,
                    cmap='RdBu_r', norm=norm,
                    s=10, edgecolor='black', linewidth=0.3, zorder=5,
                )
            else:
                ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes,
                        ha='center', va='center', fontsize=8)

            if row_idx == 0:
                ax.set_title(title, fontsize=8)
            if col_idx == 0:
                ax.set_ylabel(loc, fontsize=8, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])

    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, label='Residual (Predicted - Actual) [MVA]')
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label('Residual (Predicted - Actual) [MVA]', fontsize=8)

    plt.subplots_adjust(right=0.9, wspace=0.05, hspace=0.05)
    fig.savefig(OUTPUT_DIR / 'figure2_residual_maps.png', dpi=200, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / 'figure2_residual_maps.pdf', bbox_inches='tight')
    plt.close(fig)

    print(f'Exp 5 complete. Output: {OUTPUT_DIR}')


if __name__ == '__main__':
    plot_residual_maps()
