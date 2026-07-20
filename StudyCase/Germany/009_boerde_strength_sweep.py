"""
009 Börde post-processing parameter sweep (Part A)

Loads grid_demands from the 005 baseline model and sweeps post-processing
parameter combinations. Does not retrain the model, only adjusts the
post-processing (NTL/WC/Proximity) parameters.

Usage:
    python 009_boerde_strength_sweep.py
"""

import sys
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from SpatialAllocation.Allocator import allocator_registry

warnings.filterwarnings('ignore', category=FutureWarning)

# ─── Constants ───
DATA_DIR = SCRIPT_DIR / 'results' / 'intermediate'
MODEL_DIR = SCRIPT_DIR / 'results' / 'models'
OUTPUT_DIR = SCRIPT_DIR / 'results' / 'figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RELATION_COL = 'Name'
DEMAND_COL = 'p_mw'
DERIVED_DEMAND_COL = 'Demand (MVA)'
TARGET_CRS = 'EPSG:25832'
DIST_CLAMP_KM = 0.01
RCI_THRESHOLD = 0.5

GAMMA_RANGE = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
SEEDS = [42, 123, 456]


def evaluate_allocation(subs_result, actual_col=DEMAND_COL, alloc_col='allocated_demand'):
    actual = subs_result[actual_col].values
    allocated = subs_result[alloc_col].values
    corr, _ = pearsonr(actual, allocated)
    rmse = np.sqrt(mean_squared_error(actual, allocated))
    mae = mean_absolute_error(actual, allocated)
    return {'corr': corr, 'rmse': rmse, 'mae': mae}


def apply_proximity_correction(grid_gdf, region_sub, base_demand,
                                subs_gdf, gamma):
    """Apply proximity correction to base_demand."""
    grid_proj = grid_gdf.to_crs(TARGET_CRS)
    subs_proj = subs_gdf.to_crs(TARGET_CRS)
    gc = np.column_stack([grid_proj.geometry.x.values, grid_proj.geometry.y.values])
    sc = np.column_stack([subs_proj.geometry.x.values, subs_proj.geometry.y.values])
    dist_km = np.maximum(cdist(gc, sc, 'euclidean') / 1000.0, DIST_CLAMP_KM)
    prox = np.sum(dist_km ** (-gamma), axis=1)

    rci_sum = (grid_gdf['lu_residential_prop'].values
               + grid_gdf['lu_commercial_prop'].values
               + grid_gdf['lu_industrial_prop'].values)
    rci_mask = rci_sum > RCI_THRESHOLD

    corrected = np.zeros(len(grid_gdf))
    region_info = region_sub.set_index(RELATION_COL)

    for name, group in grid_gdf.groupby(RELATION_COL):
        if name not in region_info.index:
            continue
        total = region_info.loc[name, DERIVED_DEMAND_COL]
        idx = group.index
        pg = prox[idx]
        rci_g = rci_mask[idx]
        rci_p = pg[rci_g]
        med = np.median(rci_p) if len(rci_p) > 0 else np.median(pg)
        if med <= 0:
            med = 1e-6
        factor = np.log(1 + pg) / np.log(1 + med)
        raw = base_demand[idx] * factor
        s = raw.sum()
        if s > 0:
            corrected[idx] = total * raw / s
        else:
            corrected[idx] = total / len(group)

    return corrected


def main():
    # Load data
    region_gdf = gpd.read_file(str(DATA_DIR / 'source_regions.gpkg'))
    subs_gdf = gpd.read_file(str(DATA_DIR / 'substations.gpkg'))

    gemeinde_demand = subs_gdf.groupby('Gemeinde')[DEMAND_COL].sum()
    region_gdf[DERIVED_DEMAND_COL] = (
        region_gdf[RELATION_COL].map(gemeinde_demand).fillna(0.0))

    with open(DATA_DIR / 'features' / 'assembled' / 'boerde_grid_points.pickle', 'rb') as f:
        grid_gdf, step_size_m = pickle.load(f)

    # NTL
    ntl_npz = np.load(DATA_DIR / 'features' / 'extracted' / 'boerde_ntl.npz',
                       allow_pickle=True)
    ntl_values = ntl_npz['data'][:, 0]

    # Voronoi allocator
    alloc = allocator_registry.create('voronoi')
    voronoi_res = alloc.allocate(grid_gdf, subs_gdf)

    # Sweep results
    sweep_results = []

    for seed in SEEDS:
        gd_path = (MODEL_DIR / 'baseline' / f'seed_{seed}'
                   / 'grid_demands' / 'boerde_grid_demands.pickle')
        if not gd_path.exists():
            print(f'Skipping seed {seed} (grid_demands not found)')
            continue

        with open(gd_path, 'rb') as f:
            grid_demands = pickle.load(f)

        base_demand = grid_demands['gnn_demand']

        for gamma in GAMMA_RANGE:
            for use_ntl in [False, True]:
                for use_wc in [False, True]:
                    # Apply the correction chain
                    demand = base_demand.copy()

                    if use_wc:
                        wc_factor = 1.0 - grid_gdf['wc_others_ratio'].values
                        demand = demand * wc_factor
                        # Renormalize
                        region_info = region_gdf.set_index(RELATION_COL)
                        for name, group in grid_gdf.groupby(RELATION_COL):
                            if name not in region_info.index:
                                continue
                            total = region_info.loc[name, DERIVED_DEMAND_COL]
                            idx = group.index
                            s = demand[idx].sum()
                            if s > 0:
                                demand[idx] = total * demand[idx] / s

                    if use_ntl:
                        # Simplified NTL correction
                        rci_sum = (grid_gdf['lu_residential_prop'].values
                                   + grid_gdf['lu_commercial_prop'].values
                                   + grid_gdf['lu_industrial_prop'].values)
                        rci_mask = rci_sum > RCI_THRESHOLD
                        region_info = region_gdf.set_index(RELATION_COL)
                        corrected = np.zeros(len(grid_gdf))
                        for name, group in grid_gdf.groupby(RELATION_COL):
                            if name not in region_info.index:
                                continue
                            total = region_info.loc[name, DERIVED_DEMAND_COL]
                            idx = group.index
                            ng = ntl_values[idx]
                            rci_g = rci_mask[idx]
                            rci_nz = ng[rci_g & (ng > 0)]
                            eps = (np.percentile(rci_nz, 5) if len(rci_nz) > 0
                                   else (np.percentile(ng[ng > 0], 5)
                                         if (ng > 0).any() else 0.1))
                            rci_ntl = ng[rci_g]
                            med = (np.median(rci_ntl) if len(rci_ntl) > 0
                                   else np.median(ng))
                            if med <= 0:
                                med = eps
                            factor = np.log(1 + ng + eps) / np.log(1 + med)
                            raw = demand[idx] * factor
                            s = raw.sum()
                            if s > 0:
                                corrected[idx] = total * raw / s
                            else:
                                corrected[idx] = total / len(group)
                        demand = corrected

                    # Proximity correction
                    demand = apply_proximity_correction(
                        grid_gdf, region_gdf, demand, subs_gdf, gamma)

                    # Voronoi aggregation + evaluation
                    subs_result = subs_gdf.copy()
                    subs_result['allocated_demand'] = 0.0
                    for t_idx in range(len(subs_gdf)):
                        mask = voronoi_res.assignment == t_idx
                        subs_result.loc[t_idx, 'allocated_demand'] = demand[mask].sum()

                    m = evaluate_allocation(subs_result)
                    sweep_results.append({
                        'seed': seed,
                        'gamma': gamma,
                        'ntl': use_ntl,
                        'wc': use_wc,
                        **m,
                    })

    if not sweep_results:
        print('No results available, please run 005 first')
        return

    sweep_df = pd.DataFrame(sweep_results)
    sweep_df.to_csv(OUTPUT_DIR.parent / 'strength_sweep.csv', index=False)

    # Aggregate by gamma (mean across seeds)
    agg = sweep_df.groupby(['gamma', 'ntl', 'wc']).agg(
        rmse_mean=('rmse', 'mean'), rmse_std=('rmse', 'std'),
        corr_mean=('corr', 'mean'), corr_std=('corr', 'std'),
    ).reset_index()

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    combos = [
        (False, False, 'GNN only', 'o-'),
        (True, False, 'GNN+NTL', 's-'),
        (False, True, 'GNN+WC', '^-'),
        (True, True, 'GNN+NTL+WC', 'D-'),
    ]

    for ntl_flag, wc_flag, label, style in combos:
        sub = agg[(agg['ntl'] == ntl_flag) & (agg['wc'] == wc_flag)]
        if sub.empty:
            continue
        axes[0].errorbar(sub['gamma'], sub['rmse_mean'], yerr=sub['rmse_std'],
                         fmt=style, label=label, capsize=3)
        axes[1].errorbar(sub['gamma'], sub['corr_mean'], yerr=sub['corr_std'],
                         fmt=style, label=label, capsize=3)

    axes[0].set_xlabel('Proximity γ')
    axes[0].set_ylabel('RMSE (MW)')
    axes[0].set_title('RMSE vs Proximity γ')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Proximity γ')
    axes[1].set_ylabel('Pearson Correlation')
    axes[1].set_title('Correlation vs Proximity γ')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'strength_sweep.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {OUTPUT_DIR / "strength_sweep.png"}')
    print(f'CSV: {OUTPUT_DIR.parent / "strength_sweep.csv"}')


if __name__ == '__main__':
    main()
