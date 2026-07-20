"""
012 Regional Heterogeneity Analysis (Exp 6) - Voronoi Baseline

Groups regions by load density + Shannon entropy to analyze the applicability
conditions of each method. All methods use the Voronoi baseline.

Usage:
    python 012_exp6_heterogeneity.py
"""

import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = SCRIPT_DIR / 'results' / 'intermediate'
ASSEMBLED_DIR = DATA_DIR / 'features' / 'assembled'
STATIC_DIR = SCRIPT_DIR / 'results' / 'static_allocation'
EXP0_DIR = SCRIPT_DIR / 'results' / 'exp0_kfold_prior'
OUTPUT_DIR = SCRIPT_DIR / 'results' / 'exp6_heterogeneity'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_LOCATIONS = [
    'London',
    'TLH2', 'TLH3', 'TLJ1', 'TLF1', 'TLF2',
    'TLC1', 'TLC2', 'TLD6', 'TLG1', 'TLG2', 'TLE4',
    'TLH1', 'TLE3', 'TLD3', 'TLD4',
]

SEEDS = [42, 123, 456]

LU_COLS = [
    'lu_residential_prop', 'lu_commercial_prop', 'lu_industrial_prop',
    'lu_agricultural_prop', 'lu_others_prop',
]


def compute_shannon_entropy(proportions):
    """Compute Shannon entropy (5-dimensional land-use proportions)."""
    p = np.clip(proportions, 1e-10, 1.0)
    return -np.sum(p * np.log(p))


def compute_region_metadata():
    """Compute per-region metadata (load density, mixing degree)."""
    region_gdf = gpd.read_file(str(DATA_DIR / 'ITL3_region.gpkg'))
    substations_gdf = gpd.read_file(str(DATA_DIR / 'substations.gpkg'))

    metadata = []

    for loc in ALL_LOCATIONS:
        with open(ASSEMBLED_DIR / f'{loc}_grid_points.pickle', 'rb') as f:
            grid_gdf, step_size_m = pickle.load(f)

        study_itl3 = grid_gdf['ITL3'].unique()
        region_sub = region_gdf[region_gdf['ITL3'].isin(study_itl3)].copy()
        subs_sub = substations_gdf[substations_gdf['ITL3'].isin(study_itl3)].copy()

        # Load density = total demand / region area
        total_demand = region_sub['Demand (MVA)'].sum()
        region_proj = region_sub.to_crs('EPSG:27700')
        total_area_km2 = region_proj.geometry.area.sum() / 1e6
        load_density = total_demand / total_area_km2 if total_area_km2 > 0 else 0

        # Shannon entropy (based on the mean land-use proportions across grid points)
        lu_values = grid_gdf[LU_COLS].values
        mean_lu = lu_values.mean(axis=0)
        mean_lu = mean_lu / mean_lu.sum()  # normalize
        entropy = compute_shannon_entropy(mean_lu)

        n_substations = len(subs_sub)
        n_agents = len(grid_gdf)

        metadata.append({
            'region': loc,
            'total_demand_mva': round(total_demand, 2),
            'area_km2': round(total_area_km2, 2),
            'load_density': round(load_density, 4),
            'shannon_entropy': round(entropy, 4),
            'n_substations': n_substations,
            'n_agents': n_agents,
            'n_itl3': len(study_itl3),
        })

    return pd.DataFrame(metadata)


def load_method_rmse(config_name, method='voronoi_GNN'):
    """Load the 16-region RMSE for a given config (averaged across seeds)."""
    all_vals = {}
    for seed in SEEDS:
        path = EXP0_DIR / f'seed_{seed}' / config_name / 'kfold_test_rmse.csv'
        if not path.exists():
            continue
        df = pd.read_csv(path, index_col=0)
        if method not in df.index:
            continue
        for loc in ALL_LOCATIONS:
            if loc in df.columns:
                if loc not in all_vals:
                    all_vals[loc] = []
                all_vals[loc].append(float(df.loc[method, loc]))

    return {loc: np.mean(vals) for loc, vals in all_vals.items()}


def run_heterogeneity_analysis():
    """Run the regional heterogeneity analysis (Voronoi baseline)."""
    print('=' * 60)
    print('Exp 6: Regional Heterogeneity Analysis (Voronoi Baseline)')
    print('=' * 60)

    # 1. Compute region metadata
    meta_df = compute_region_metadata()
    meta_df.to_csv(OUTPUT_DIR / 'region_metadata.csv', index=False)
    print('\n=== Region Metadata ===')
    print(meta_df.to_string(index=False))

    # 2. Grouping
    # Load density terciles
    density_terciles = meta_df['load_density'].quantile([1/3, 2/3]).values
    meta_df['density_group'] = pd.cut(
        meta_df['load_density'],
        bins=[-np.inf, density_terciles[0], density_terciles[1], np.inf],
        labels=['Low density', 'Medium density', 'High density'],
    )

    # Shannon entropy median split
    entropy_median = meta_df['shannon_entropy'].median()
    meta_df['entropy_group'] = np.where(
        meta_df['shannon_entropy'] >= entropy_median, 'High mixing', 'Low mixing')

    print(f'\nDensity thresholds: {density_terciles}')
    print(f'Entropy median: {entropy_median:.4f}')

    # 3. Load RMSE for each method (all Voronoi baseline)
    methods_to_compare = {
        'Uni (voronoi uniform)': None,  # static, handled separately
        'GNN (baseline)': ('baseline', 'voronoi_GNN'),
        'GNNpostP (Prox post-correction)': ('baseline', 'voronoi_prox_GNN'),
        'GNNpostNP (NTL+Prox post-correction)': ('baseline', 'voronoi_ntl_prox_GNN'),
        'GNNpriorN (NTL prior)': ('ntl', 'voronoi_GNN'),
        'GNNpriorP (Prox prior)': ('proximity', 'voronoi_GNN'),
        'GNNpriorNP (NTL+Prox prior)': ('ntl_prox', 'voronoi_GNN'),
    }

    # Static method
    static_rmse_path = STATIC_DIR / 'all_regions_rmse.csv'
    static_rmse = None
    if static_rmse_path.exists():
        static_rmse = pd.read_csv(static_rmse_path, index_col=0)

    # Load Uni (voronoi uniform)
    if static_rmse is not None and 'voronoi' in static_rmse.index:
        meta_df['Uni (voronoi uniform)'] = [
            float(static_rmse.loc['voronoi', loc])
            if loc in static_rmse.columns else np.nan
            for loc in ALL_LOCATIONS
        ]

    # GNN methods
    for method_label, config_info in methods_to_compare.items():
        if config_info is None:
            continue  # Uni already handled
        config_name, gnn_method = config_info
        rmse_dict = load_method_rmse(config_name, gnn_method)
        meta_df[method_label] = meta_df['region'].map(rmse_dict)

    # Static methods
    if static_rmse is not None:
        for static_label, static_method in [
            ('GPM (voronoi_gpm)', 'voronoi_gpm'),
            ('UniP (voronoi_prox2)', 'voronoi_prox2'),
            ('GPMpostP (voronoi_prox2_gpm)', 'voronoi_prox2_gpm'),
            ('GPMpostN (voronoi_ntl_gpm)', 'voronoi_ntl_gpm'),
            ('GPMpostNP (best static)', 'voronoi_prox2_ntl_gpm'),
        ]:
            if static_method in static_rmse.index:
                meta_df[static_label] = [
                    float(static_rmse.loc[static_method, loc])
                    if loc in static_rmse.columns else np.nan
                    for loc in ALL_LOCATIONS
                ]

    # 4. Grouped summary
    method_cols = [c for c in meta_df.columns
                   if c.startswith(('Uni', 'GNN', 'GPM'))]

    print('\n=== Grouped by Load Density ===')
    density_summary = meta_df.groupby('density_group')[method_cols].agg(['mean', 'std']).round(4)
    density_summary.to_csv(OUTPUT_DIR / 'table3_density_group.csv')
    print(density_summary.to_string())

    print('\n=== Grouped by Land-Use Mixing Degree ===')
    entropy_summary = meta_df.groupby('entropy_group')[method_cols].agg(['mean', 'std']).round(4)
    entropy_summary.to_csv(OUTPUT_DIR / 'table3_entropy_group.csv')
    print(entropy_summary.to_string())

    # 5. Cross grouping
    print('\n=== Cross Grouping (Density x Mixing Degree) ===')
    cross = meta_df.groupby(['density_group', 'entropy_group'])[method_cols].mean().round(4)
    cross.to_csv(OUTPUT_DIR / 'table3_cross_group.csv')
    print(cross.to_string())

    # Save full metadata (including groups and metrics)
    meta_df.to_csv(OUTPUT_DIR / 'region_metadata_with_groups.csv', index=False)

    print(f'\nExp 6 complete. Output: {OUTPUT_DIR}')


if __name__ == '__main__':
    run_heterogeneity_analysis()
