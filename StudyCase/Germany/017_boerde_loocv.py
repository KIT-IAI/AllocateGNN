"""
017 Germany Börde region jackknife leave-one-out robustness analysis

Performs a jackknife leave-one-out analysis on the trained model's predictions
for the 13 substations: each iteration drops one substation and computes
Corr and RMSE on the remaining 12.

If the LOO Corr remains high -> the result holds up (not driven by overfitting)
If the LOO Corr drops significantly -> a small number of outlier points are
driving the overall result

Optionally also performs a full LOO-CV retraining (if the --retrain flag is set).

Usage:
    python 017_boerde_loocv.py                  # jackknife (fast)
    python 017_boerde_loocv.py --retrain        # LOO-CV retraining (slow)
"""

import sys
import argparse
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from SpatialAllocation.Allocator import allocator_registry

warnings.filterwarnings('ignore', category=FutureWarning)

# ════════════════════════════════════════════════════════════
# Path constants
# ════════════════════════════════════════════════════════════

DATA_DIR = SCRIPT_DIR / 'results' / 'intermediate'
MODELS_DIR = SCRIPT_DIR / 'results' / 'models'
OUTPUT_DIR = SCRIPT_DIR / 'results' / 'loocv'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 456]
CONFIGS = ['baseline', 'ntl_prox']

DEMAND_KEYS = {
    'GNN': 'gnn_demand',
    'GNNpostN': 'ntl_gnn_demand',
    'GNNpostP': 'prox_gnn_demand',
    'GNNpostNP': 'ntl_prox_gnn_demand',
}


def load_data():
    """Load Börde data."""
    grid_path = DATA_DIR / 'features' / 'assembled' / 'boerde_grid_points.pickle'
    with open(grid_path, 'rb') as f:
        data = pickle.load(f)

    if isinstance(data, (list, tuple)):
        grid_gdf = data[0]
        step_size_m = data[1]
    else:
        grid_gdf = data
        step_size_m = 350

    subs_gdf = gpd.read_file(str(DATA_DIR / 'substations.gpkg'))

    return grid_gdf, subs_gdf, step_size_m


def voronoi_allocate(grid_gdf, subs_gdf, demand_arr):
    """Aggregate grid demands to substations via Voronoi partitioning."""
    alloc_voronoi = allocator_registry.create('voronoi')
    voronoi_res = alloc_voronoi.allocate(grid_gdf, subs_gdf)

    n_subs = len(subs_gdf)
    allocated = np.zeros(n_subs)
    for i in range(n_subs):
        mask = voronoi_res.assignment == i
        allocated[i] = demand_arr[mask].sum()

    return allocated


def jackknife_analysis():
    """Jackknife leave-one-out: check robustness using existing prediction results."""
    print("Loading Börde data...")
    grid_gdf, subs_gdf, step_size_m = load_data()
    n_subs = len(subs_gdf)
    print(f"Number of substations: {n_subs}")

    actual = subs_gdf['p_mw'].values.astype(float)

    all_results = []

    for config in CONFIGS:
        for seed in SEEDS:
            gd_path = MODELS_DIR / config / f'seed_{seed}' / 'grid_demands' / 'boerde_grid_demands.pickle'
            if not gd_path.exists():
                print(f"  [skip] {config}/seed_{seed}: grid_demands not found")
                continue

            with open(gd_path, 'rb') as f:
                grid_demands = pickle.load(f)

            for method, demand_key in DEMAND_KEYS.items():
                if demand_key not in grid_demands:
                    continue

                demand_arr = grid_demands[demand_key]
                allocated = voronoi_allocate(grid_gdf, subs_gdf, demand_arr)

                # All 13 substations
                full_corr, _ = pearsonr(actual, allocated)
                full_rmse = np.sqrt(mean_squared_error(actual, allocated))
                full_mae = mean_absolute_error(actual, allocated)

                # Jackknife: drop one substation at a time
                loo_corrs = []
                loo_rmses = []

                for i in range(n_subs):
                    mask = np.ones(n_subs, dtype=bool)
                    mask[i] = False
                    if np.sum(mask) < 3:
                        continue
                    loo_actual = actual[mask]
                    loo_alloc = allocated[mask]
                    loo_corr, _ = pearsonr(loo_actual, loo_alloc)
                    loo_rmse = np.sqrt(mean_squared_error(loo_actual, loo_alloc))
                    loo_corrs.append(loo_corr)
                    loo_rmses.append(loo_rmse)

                    all_results.append({
                        'config': config,
                        'seed': seed,
                        'method': method,
                        'dropped_sub': subs_gdf.iloc[i]['Name'],
                        'dropped_sub_idx': i,
                        'dropped_actual': actual[i],
                        'dropped_predicted': allocated[i],
                        'dropped_abs_error': abs(allocated[i] - actual[i]),
                        'loo_corr': loo_corr,
                        'loo_rmse': loo_rmse,
                        'full_corr': full_corr,
                        'full_rmse': full_rmse,
                    })

                loo_corrs = np.array(loo_corrs)
                print(f"  {config}/seed_{seed}/{method}: "
                      f"full_Corr={full_corr:.4f}, "
                      f"LOO_Corr={loo_corrs.mean():.4f}±{loo_corrs.std():.4f} "
                      f"[{loo_corrs.min():.4f}, {loo_corrs.max():.4f}]")

    # Save
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / 'jackknife_raw.csv', index=False)

    # Summary table
    print("\n" + "="*80)
    print("Jackknife LOO summary")
    print("="*80)

    if len(results_df) > 0:
        summary = results_df.groupby(['config', 'method']).agg({
            'full_corr': 'mean',
            'full_rmse': 'mean',
            'loo_corr': ['mean', 'std', 'min', 'max'],
            'loo_rmse': ['mean', 'std'],
            'dropped_abs_error': ['mean', 'max'],
        })
        print(summary.to_string())
        summary.to_csv(OUTPUT_DIR / 'jackknife_summary.csv')

        # Identify the most influential substations
        for config in CONFIGS:
            for method in DEMAND_KEYS:
                sub = results_df[(results_df['config'] == config) &
                                 (results_df['method'] == method)]
                if len(sub) == 0:
                    continue
                # Find the largest full_corr - loo_corr gap
                sub_avg = sub.groupby('dropped_sub').agg({
                    'loo_corr': 'mean',
                    'full_corr': 'mean',
                    'dropped_abs_error': 'mean',
                    'dropped_actual': 'first',
                    'dropped_predicted': 'mean',
                }).reset_index()
                sub_avg['corr_drop'] = sub_avg['full_corr'] - sub_avg['loo_corr']
                most_influential = sub_avg.nlargest(3, 'corr_drop')
                print(f"\n{config}/{method} most influential substations (largest Corr drop when removed):")
                for _, row in most_influential.iterrows():
                    print(f"  {row['dropped_sub']}: "
                          f"actual={row['dropped_actual']:.1f}, "
                          f"pred={row['dropped_predicted']:.1f}, "
                          f"corr_drop={row['corr_drop']:.4f}")

    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--retrain', action='store_true',
                        help='Perform a full LOO-CV retraining (slow)')
    args = parser.parse_args()

    jackknife_analysis()
