"""
011 Börde residual maps

Spatial visualization: Gemeinde boundaries + substation locations + residual coloring.
Shows 4 methods: uniform_average / best static / GNN baseline / GNN best.

Usage:
    python 011_boerde_residual_maps.py
"""

import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = SCRIPT_DIR / 'results' / 'intermediate'
STATIC_DIR = SCRIPT_DIR / 'results' / 'static_allocation'
MODEL_DIR = SCRIPT_DIR / 'results' / 'models'
OUTPUT_DIR = SCRIPT_DIR / 'results' / 'figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEMAND_COL = 'p_mw'
RELATION_COL = 'Name'
DERIVED_DEMAND_COL = 'Demand (MVA)'
TARGET_CRS = 'EPSG:25832'


def compute_residuals_for_static(subs_gdf, region_gdf, grid_gdf, method_name):
    """Reconstruct per-substation allocation results for a static method and return the residuals."""
    from SpatialAllocation.Allocator import allocator_registry
    from SpatialAllocation.Weighter import weighter_registry
    from SpatialAllocation.FeatureExtractor.correctors.proximity_corrector import ProximityCorrector

    total_demand = region_gdf[DERIVED_DEMAND_COL].sum()
    actual = subs_gdf[DEMAND_COL].values

    if method_name == 'uniform_average':
        predicted = np.full(len(subs_gdf), total_demand / len(subs_gdf))
    elif method_name == 'gemeinde_average':
        predicted = np.zeros(len(subs_gdf))
        for gem, group in subs_gdf.groupby('Gemeinde'):
            reg = region_gdf[region_gdf[RELATION_COL] == gem]
            if not reg.empty:
                predicted[group.index] = reg[DERIVED_DEMAND_COL].iloc[0] / len(group)
    else:
        # Other static methods would need to be recomputed (simplified handling here)
        predicted = np.full(len(subs_gdf), total_demand / len(subs_gdf))

    return predicted - actual


def compute_residuals_for_gnn(subs_gdf, grid_gdf, gd_path, demand_key):
    """Reconstruct GNN allocation residuals from grid_demands."""
    from SpatialAllocation.Allocator import allocator_registry

    with open(gd_path, 'rb') as f:
        grid_demands = pickle.load(f)

    demand_arr = grid_demands[demand_key]
    alloc = allocator_registry.create('voronoi')
    voronoi_res = alloc.allocate(grid_gdf, subs_gdf)

    predicted = np.zeros(len(subs_gdf))
    for t_idx in range(len(subs_gdf)):
        mask = voronoi_res.assignment == t_idx
        predicted[t_idx] = demand_arr[mask].sum()

    return predicted - subs_gdf[DEMAND_COL].values


def main():
    # Load data
    region_gdf = gpd.read_file(str(DATA_DIR / 'source_regions.gpkg'))
    subs_gdf = gpd.read_file(str(DATA_DIR / 'substations.gpkg'))

    gemeinde_demand = subs_gdf.groupby('Gemeinde')[DEMAND_COL].sum()
    region_gdf[DERIVED_DEMAND_COL] = (
        region_gdf[RELATION_COL].map(gemeinde_demand).fillna(0.0))

    with open(DATA_DIR / 'features' / 'assembled' / 'boerde_grid_points.pickle', 'rb') as f:
        grid_gdf, _ = pickle.load(f)

    # Reproject
    region_proj = region_gdf.to_crs(TARGET_CRS)
    subs_proj = subs_gdf.to_crs(TARGET_CRS)

    # Prepare panels
    methods = {
        'Uniform Average': ('static', 'uniform_average', None),
        'Gemeinde Average\n(Oracle)': ('static', 'gemeinde_average', None),
    }

    # Find available GNN results
    for config in ['baseline', 'ntl_prox']:
        gd_path = (MODEL_DIR / config / 'seed_42'
                   / 'grid_demands' / 'boerde_grid_demands.pickle')
        if gd_path.exists():
            label = f'GNN [{config}]'
            methods[label] = ('gnn', gd_path, 'gnn_demand')

    n_methods = len(methods)
    if n_methods == 0:
        print('No methods available')
        return

    fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 6))
    if n_methods == 1:
        axes = [axes]

    # Compute the global residual range
    all_residuals = []
    residual_data = {}
    for title, (mtype, *args) in methods.items():
        if mtype == 'static':
            res = compute_residuals_for_static(
                subs_gdf, region_gdf, grid_gdf, args[0])
        else:
            res = compute_residuals_for_gnn(
                subs_gdf, grid_gdf, args[0], args[1])
        residual_data[title] = res
        all_residuals.extend(res)

    vmax = max(abs(min(all_residuals)), abs(max(all_residuals)))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    for ax, (title, residuals) in zip(axes, residual_data.items()):
        # Plot Gemeinde boundaries
        region_proj.plot(ax=ax, facecolor='#f0f0f0', edgecolor='gray',
                         linewidth=0.5)

        # Plot substation residual points
        subs_plot = subs_proj.copy()
        subs_plot['residual'] = residuals
        scatter = ax.scatter(
            subs_plot.geometry.x, subs_plot.geometry.y,
            c=residuals, cmap='RdBu_r', norm=norm,
            s=80, edgecolors='black', linewidths=0.5, zorder=5)

        ax.set_title(title, fontsize=11)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

        # RMSE annotation
        rmse = np.sqrt(np.mean(residuals ** 2))
        ax.text(0.02, 0.98, f'RMSE={rmse:.2f}',
                transform=ax.transAxes, va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Shared colorbar
    cbar = fig.colorbar(scatter, ax=axes, shrink=0.6, pad=0.02)
    cbar.set_label('Residual (MW): predicted - actual')

    plt.suptitle('Börde substation load allocation residuals', fontsize=14)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'residual_maps.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {OUTPUT_DIR / "residual_maps.png"}')


if __name__ == '__main__':
    main()
