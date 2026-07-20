"""
015 Börde NTL-Proximity correlation analysis

Analyzes the correlation between NTL intensity vs proximity score vs actual load.
Computes Pearson + Spearman correlation coefficients.

Usage:
    python 015_boerde_ntl_prox_correlation.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import cdist

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / 'results' / 'intermediate'
OUTPUT_DIR = SCRIPT_DIR / 'results' / 'figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEMAND_COL = 'p_mw'
TARGET_CRS = 'EPSG:25832'
DIST_CLAMP_KM = 0.01


def main():
    # Load data
    subs_gdf = gpd.read_file(str(DATA_DIR / 'substations.gpkg'))

    # Load NTL
    ntl_npz = np.load(DATA_DIR / 'features' / 'extracted' / 'boerde_ntl.npz',
                       allow_pickle=True)
    ntl_all = ntl_npz['data'][:, 0]

    # Load grid
    import pickle
    with open(DATA_DIR / 'features' / 'assembled' / 'boerde_grid_points.pickle', 'rb') as f:
        grid_gdf, _ = pickle.load(f)

    # Compute the average NTL for each substation (within its Voronoi region)
    sys.path.insert(0, str(SCRIPT_DIR.parent.parent))
    from SpatialAllocation.Allocator import allocator_registry

    alloc = allocator_registry.create('voronoi')
    voronoi_res = alloc.allocate(grid_gdf, subs_gdf)

    n_subs = len(subs_gdf)
    subs_ntl_mean = np.zeros(n_subs)
    subs_ntl_sum = np.zeros(n_subs)
    subs_n_points = np.zeros(n_subs)

    for t_idx in range(n_subs):
        mask = voronoi_res.assignment == t_idx
        subs_ntl_mean[t_idx] = ntl_all[mask].mean() if mask.sum() > 0 else 0
        subs_ntl_sum[t_idx] = ntl_all[mask].sum()
        subs_n_points[t_idx] = mask.sum()

    # Proximity score (proximity of each substation to the other substations)
    subs_proj = subs_gdf.to_crs(TARGET_CRS)
    subs_coords = np.column_stack([subs_proj.geometry.x.values,
                                    subs_proj.geometry.y.values])
    dist_km = cdist(subs_coords, subs_coords, 'euclidean') / 1000.0
    np.fill_diagonal(dist_km, np.inf)
    subs_proximity = np.sum(np.maximum(dist_km, DIST_CLAMP_KM) ** (-2), axis=1)

    actual = subs_gdf[DEMAND_COL].values
    labels = subs_gdf['Kennzeichen'].values

    # Correlation coefficients
    corr_results = {}
    pairs = [
        ('NTL_mean', subs_ntl_mean),
        ('NTL_sum', subs_ntl_sum),
        ('Proximity', subs_proximity),
        ('N_points', subs_n_points),
    ]

    print('=' * 60)
    print('Börde NTL / Proximity vs actual load correlation')
    print('=' * 60)

    for name, values in pairs:
        pr, p_pr = pearsonr(actual, values)
        sr, p_sr = spearmanr(actual, values)
        corr_results[name] = {
            'pearson_r': pr, 'pearson_p': p_pr,
            'spearman_rho': sr, 'spearman_p': p_sr,
        }
        print(f'{name:12s}: Pearson r={pr:.4f} (p={p_pr:.4f}), '
              f'Spearman ρ={sr:.4f} (p={p_sr:.4f})')

    corr_df = pd.DataFrame(corr_results).T
    corr_df.to_csv(OUTPUT_DIR.parent / 'ntl_prox_correlation.csv')

    # Plot: 3 panels (NTL_mean, NTL_sum, Proximity) vs actual
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    plot_pairs = [
        ('NTL Mean', subs_ntl_mean, 'NTL Mean Radiance'),
        ('NTL Sum', subs_ntl_sum, 'NTL Sum Radiance'),
        ('Proximity Score', subs_proximity, 'Proximity (γ=2)'),
    ]

    for ax, (title, x_vals, xlabel) in zip(axes, plot_pairs):
        ax.scatter(x_vals, actual, s=60, c='steelblue',
                   edgecolors='black', linewidths=0.5, zorder=5)

        for i, label in enumerate(labels):
            ax.annotate(str(label), (x_vals[i], actual[i]),
                        fontsize=6, ha='left', va='bottom',
                        xytext=(3, 3), textcoords='offset points')

        # Trend line
        z = np.polyfit(x_vals, actual, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
        ax.plot(x_line, p(x_line), 'r--', alpha=0.5, linewidth=1)

        pr, _ = pearsonr(x_vals, actual)
        sr, _ = spearmanr(x_vals, actual)
        ax.text(0.05, 0.95, f'Pearson r={pr:.3f}\nSpearman ρ={sr:.3f}',
                transform=ax.transAxes, va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        ax.set_xlabel(xlabel)
        ax.set_ylabel('Actual load (MW)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Börde: NTL / Proximity vs actual substation load', fontsize=14)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'ntl_prox_correlation.png', dpi=150,
                bbox_inches='tight')
    plt.close()
    print(f'\nSaved: {OUTPUT_DIR / "ntl_prox_correlation.png"}')
    print(f'CSV: {OUTPUT_DIR.parent / "ntl_prox_correlation.csv"}')


if __name__ == '__main__':
    main()
