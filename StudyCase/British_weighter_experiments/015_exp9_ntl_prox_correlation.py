"""
015 NTL vs Proximity correlation test (Exp 9)

Does not depend on Exp 0; reads raw NTL directly and computes Proximity.
Pearson + Spearman + scatter plot.

Usage:
    python 015_exp9_ntl_prox_correlation.py
"""

import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'no-latex'])
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import cdist

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = SCRIPT_DIR / 'results' / 'intermediate'
ASSEMBLED_DIR = DATA_DIR / 'features' / 'assembled'
EXTRACTED_DIR = DATA_DIR / 'features' / 'extracted'
OUTPUT_DIR = SCRIPT_DIR / 'results' / 'exp9_ntl_prox_corr'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_LOCATIONS = [
    'London',
    'TLH2', 'TLH3', 'TLJ1', 'TLF1', 'TLF2',
    'TLC1', 'TLC2', 'TLD6', 'TLG1', 'TLG2', 'TLE4',
    'TLH1', 'TLE3', 'TLD3', 'TLD4',
]

RCI_THRESHOLD = 0.5
PROXIMITY_GAMMA = 2.0
DIST_CLAMP_KM = 0.01
TARGET_CRS = 'EPSG:27700'


def compute_proximity_scores(grid_gdf, subs_sub, gamma=PROXIMITY_GAMMA):
    """Compute proximity scores."""
    grid_proj = grid_gdf.to_crs(TARGET_CRS)
    subs_proj = subs_sub.to_crs(TARGET_CRS)

    grid_coords = np.column_stack([grid_proj.geometry.x.values, grid_proj.geometry.y.values])
    subs_coords = np.column_stack([subs_proj.geometry.x.values, subs_proj.geometry.y.values])

    dist_km = cdist(grid_coords, subs_coords, metric='euclidean') / 1000.0
    dist_km = np.maximum(dist_km, DIST_CLAMP_KM)
    return np.sum(dist_km ** (-gamma), axis=1)


def compute_ntl_factor(ntl_values, rci_mask):
    """Compute the NTL correction factor (raw factor, before per-region normalization)."""
    rci_nonzero = ntl_values[rci_mask & (ntl_values > 0)]
    if len(rci_nonzero) > 0:
        epsilon = np.percentile(rci_nonzero, 5)
    else:
        nonzero = ntl_values[ntl_values > 0]
        epsilon = np.percentile(nonzero, 5) if len(nonzero) > 0 else 0.1

    rci_ntl = ntl_values[rci_mask]
    ntl_median = np.median(rci_ntl) if len(rci_ntl) > 0 else np.median(ntl_values)
    if ntl_median <= 0:
        ntl_median = epsilon

    return np.log(1 + ntl_values + epsilon) / np.log(1 + ntl_median)


def compute_prox_factor(prox_scores, rci_mask):
    """Compute the Proximity correction factor."""
    rci_prox = prox_scores[rci_mask]
    prox_median = np.median(rci_prox) if len(rci_prox) > 0 else np.median(prox_scores)
    if prox_median <= 0:
        prox_median = 1e-6
    return np.log(1 + prox_scores) / np.log(1 + prox_median)


def run_correlation_analysis():
    """Run the NTL vs Proximity correlation analysis."""
    print('=' * 60)
    print('Exp 9: NTL vs Proximity correlation')
    print('=' * 60)

    region_gdf = gpd.read_file(str(DATA_DIR / 'ITL3_region.gpkg'))
    substations_gdf = gpd.read_file(str(DATA_DIR / 'substations.gpkg'))

    all_ntl_factors = []
    all_prox_factors = []
    all_regions = []
    per_region_corr = []

    for loc in ALL_LOCATIONS:
        print(f'  Processing {loc}...')

        with open(ASSEMBLED_DIR / f'{loc}_grid_points.pickle', 'rb') as f:
            grid_gdf, _ = pickle.load(f)

        ntl_npz = np.load(EXTRACTED_DIR / f'{loc}_ntl.npz', allow_pickle=True)
        ntl_values = ntl_npz['data'][:, 0]

        study_itl3 = grid_gdf['ITL3'].unique()
        subs_sub = substations_gdf[substations_gdf['ITL3'].isin(study_itl3)].copy().reset_index(drop=True)

        prox_scores = compute_proximity_scores(grid_gdf, subs_sub)

        rci_sum = (grid_gdf['lu_residential_prop'].values
                  + grid_gdf['lu_commercial_prop'].values
                  + grid_gdf['lu_industrial_prop'].values)
        rci_mask = rci_sum > RCI_THRESHOLD

        ntl_factor = compute_ntl_factor(ntl_values, rci_mask)
        prox_factor = compute_prox_factor(prox_scores, rci_mask)

        # Keep only RCI points
        ntl_f_rci = ntl_factor[rci_mask]
        prox_f_rci = prox_factor[rci_mask]

        all_ntl_factors.extend(ntl_f_rci.tolist())
        all_prox_factors.extend(prox_f_rci.tolist())
        all_regions.extend([loc] * len(ntl_f_rci))

        # Per-region correlation
        if len(ntl_f_rci) > 10:
            r_pearson, p_pearson = pearsonr(ntl_f_rci, prox_f_rci)
            r_spearman, p_spearman = spearmanr(ntl_f_rci, prox_f_rci)
        else:
            r_pearson, p_pearson = float('nan'), 1.0
            r_spearman, p_spearman = float('nan'), 1.0

        per_region_corr.append({
            'region': loc,
            'n_rci': int(rci_mask.sum()),
            'pearson_r': round(r_pearson, 4),
            'pearson_p': round(p_pearson, 6),
            'spearman_rho': round(r_spearman, 4),
            'spearman_p': round(p_spearman, 6),
        })

        print(f'    N_RCI={rci_mask.sum()}, Pearson={r_pearson:.4f}, Spearman={r_spearman:.4f}')

    # Global correlation
    all_ntl = np.array(all_ntl_factors)
    all_prox = np.array(all_prox_factors)

    r_global_pearson, p_global_pearson = pearsonr(all_ntl, all_prox)
    r_global_spearman, p_global_spearman = spearmanr(all_ntl, all_prox)

    print(f'\n=== Global correlation ===')
    print(f'  N = {len(all_ntl)}')
    print(f'  Pearson r = {r_global_pearson:.4f} (p = {p_global_pearson:.2e})')
    print(f'  Spearman ρ = {r_global_spearman:.4f} (p = {p_global_spearman:.2e})')

    # Save the per-region table
    region_df = pd.DataFrame(per_region_corr)
    region_df.to_csv(OUTPUT_DIR / 'per_region_correlation.csv', index=False)

    # Global summary
    global_summary = pd.DataFrame([{
        'scope': 'global',
        'n': len(all_ntl),
        'pearson_r': round(r_global_pearson, 4),
        'pearson_p': p_global_pearson,
        'spearman_rho': round(r_global_spearman, 4),
        'spearman_p': p_global_spearman,
    }])
    global_summary.to_csv(OUTPUT_DIR / 'global_correlation.csv', index=False)

    # Scatter plot
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    # Color by region
    unique_regions = sorted(set(all_regions))
    cmap = plt.cm.get_cmap('tab20', len(unique_regions))
    region_colors = {r: cmap(i) for i, r in enumerate(unique_regions)}

    # Random sampling to avoid overplotting
    n_total = len(all_ntl)
    max_points = 50000
    if n_total > max_points:
        idx = np.random.choice(n_total, max_points, replace=False)
    else:
        idx = np.arange(n_total)

    for i, loc in enumerate(unique_regions):
        loc_mask = np.array([r == loc for r in np.array(all_regions)[idx]])
        if loc_mask.sum() > 0:
            ax.scatter(all_ntl[idx][loc_mask], all_prox[idx][loc_mask],
                       c=[region_colors[loc]], s=1, alpha=0.3, label=loc)

    ax.set_xlabel('NTL Factor', fontsize=8)
    ax.set_ylabel('Proximity Factor', fontsize=8)
    ax.set_title(f'NTL vs Proximity Factor\n'
                 f'Pearson r={r_global_pearson:.3f}, Spearman $\\rho$={r_global_spearman:.3f}',
                 fontsize=8)
    ax.tick_params(axis='both', labelsize=8)

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=5,
              markerscale=3, ncol=1)
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'figure_ntl_prox_scatter.png', dpi=200, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / 'figure_ntl_prox_scatter.pdf', bbox_inches='tight')
    plt.close(fig)

    print(f'\nExp 9 complete, output: {OUTPUT_DIR}')


if __name__ == '__main__':
    run_correlation_analysis()
