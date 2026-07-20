"""
013 Börde scatter plots

Predicted vs actual scatter plot for the 13 substations.
Multi-panel: uniform / best static / GNN baseline / GNN best.
Each point is annotated with the substation name.

Usage:
    python 013_boerde_scatter_plots.py
"""

import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = SCRIPT_DIR / 'results' / 'intermediate'
MODEL_DIR = SCRIPT_DIR / 'results' / 'models'
OUTPUT_DIR = SCRIPT_DIR / 'results' / 'figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEMAND_COL = 'p_mw'
RELATION_COL = 'Name'
DERIVED_DEMAND_COL = 'Demand (MVA)'


def get_predicted_uniform(subs_gdf, region_gdf):
    total = region_gdf[DERIVED_DEMAND_COL].sum()
    return np.full(len(subs_gdf), total / len(subs_gdf))


def get_predicted_gemeinde_avg(subs_gdf, region_gdf):
    predicted = np.zeros(len(subs_gdf))
    for gem, group in subs_gdf.groupby('Gemeinde'):
        reg = region_gdf[region_gdf[RELATION_COL] == gem]
        if not reg.empty:
            predicted[group.index] = reg[DERIVED_DEMAND_COL].iloc[0] / len(group)
    return predicted


def get_predicted_gnn(subs_gdf, grid_gdf, config, seed=42, demand_key='gnn_demand'):
    gd_path = (MODEL_DIR / config / f'seed_{seed}'
               / 'grid_demands' / 'boerde_grid_demands.pickle')
    if not gd_path.exists():
        return None

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
    return predicted


def main():
    # Load data
    region_gdf = gpd.read_file(str(DATA_DIR / 'source_regions.gpkg'))
    subs_gdf = gpd.read_file(str(DATA_DIR / 'substations.gpkg'))

    gemeinde_demand = subs_gdf.groupby('Gemeinde')[DEMAND_COL].sum()
    region_gdf[DERIVED_DEMAND_COL] = (
        region_gdf[RELATION_COL].map(gemeinde_demand).fillna(0.0))

    with open(DATA_DIR / 'features' / 'assembled' / 'boerde_grid_points.pickle', 'rb') as f:
        grid_gdf, _ = pickle.load(f)

    actual = subs_gdf[DEMAND_COL].values
    labels = subs_gdf['Kennzeichen'].values

    # Prepare methods
    panels = [
        ('Uniform Average', get_predicted_uniform(subs_gdf, region_gdf)),
        ('Gemeinde Average (Oracle)', get_predicted_gemeinde_avg(subs_gdf, region_gdf)),
    ]

    for config in ['baseline', 'ntl_prox']:
        pred = get_predicted_gnn(subs_gdf, grid_gdf, config)
        if pred is not None:
            panels.append((f'GNN [{config}]', pred))

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (title, predicted) in zip(axes, panels):
        ax.scatter(actual, predicted, s=50, c='steelblue',
                   edgecolors='black', linewidths=0.5, zorder=5)

        # Annotate substation names
        for i, label in enumerate(labels):
            ax.annotate(str(label), (actual[i], predicted[i]),
                        fontsize=6, ha='left', va='bottom',
                        xytext=(3, 3), textcoords='offset points')

        # Diagonal line
        lims = [0, max(actual.max(), predicted.max()) * 1.1]
        ax.plot(lims, lims, 'k--', alpha=0.3, linewidth=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        # Metrics
        corr, _ = pearsonr(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        ax.text(0.05, 0.95, f'r={corr:.3f}\nRMSE={rmse:.2f}',
                transform=ax.transAxes, va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        ax.set_xlabel('Actual load (MW)')
        ax.set_ylabel('Predicted load (MW)')
        ax.set_title(title, fontsize=11)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Börde substations: predicted vs actual load', fontsize=14)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'scatter_plots.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {OUTPUT_DIR / "scatter_plots.png"}')


if __name__ == '__main__':
    main()
