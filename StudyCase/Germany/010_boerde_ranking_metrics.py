"""
010 Börde ranking metrics

Computes Spearman ρ + Top-k Precision/Recall for all methods.
N=13 substations; the statistical power limitation is noted explicitly.

Usage:
    python 010_boerde_ranking_metrics.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.stats import spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
STATIC_DIR = SCRIPT_DIR / 'results' / 'static_allocation'
MODEL_DIR = SCRIPT_DIR / 'results' / 'models'
OUTPUT_DIR = SCRIPT_DIR / 'results'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEMAND_COL = 'p_mw'
SEEDS = [42, 123, 456]
GNN_CONFIGS = ['baseline', 'ntl', 'proximity', 'ntl_prox']
TOP_K_VALUES = [3, 5]


def ranking_metrics(actual, predicted, top_k_list=None):
    """Compute ranking metrics: Spearman ρ, Top-k Precision/Recall."""
    if top_k_list is None:
        top_k_list = [3, 5]

    rho, p_value = spearmanr(actual, predicted)
    result = {'spearman_rho': rho, 'spearman_p': p_value}

    actual_rank = np.argsort(-actual)  # descending order
    pred_rank = np.argsort(-predicted)

    for k in top_k_list:
        if k > len(actual):
            continue
        actual_top = set(actual_rank[:k])
        pred_top = set(pred_rank[:k])
        tp = len(actual_top & pred_top)
        result[f'top{k}_precision'] = tp / k
        result[f'top{k}_recall'] = tp / k  # k == k, so precision = recall

    return result


def main():
    # Load actual load
    data_dir = SCRIPT_DIR / 'results' / 'intermediate'
    subs_gdf = gpd.read_file(str(data_dir / 'substations.gpkg'))
    actual = subs_gdf[DEMAND_COL].values
    n_subs = len(actual)

    print(f'Number of substations: {n_subs}')
    print(f'Note: N={n_subs} is small, so the Spearman test has limited statistical power\n')

    rows = []

    # Static methods
    static_path = STATIC_DIR / 'boerde_static_metrics.csv'
    if static_path.exists():
        static_df = pd.read_csv(static_path, index_col=0)

        # Reconstructing the allocation results for ranking requires the original results
        # This would need to be rebuilt from the 003 CIVD cache
        # Simplification: use corr from the metrics as a ranking reference, or rebuild
        # directly from the 003 results dict
        # Since 003 is a notebook, only GNN results are handled here
        print('Ranking metrics for static methods require the original allocation results (allocated_demand per substation)')
        print('Recommend additionally saving per-substation allocation results in the 003 notebook')
        print('Currently only computing ranking metrics for GNN methods\n')

    # GNN methods
    for config in GNN_CONFIGS:
        seed_metrics = []
        for seed in SEEDS:
            metrics_path = MODEL_DIR / config / f'seed_{seed}' / 'metrics.csv'
            if not metrics_path.exists():
                continue

            # Requires per-substation predicted values
            # Reconstructed from grid_demands
            gd_path = (MODEL_DIR / config / f'seed_{seed}'
                       / 'grid_demands' / 'boerde_grid_demands.pickle')
            if not gd_path.exists():
                continue

            import pickle
            with open(gd_path, 'rb') as f:
                grid_demands = pickle.load(f)

            # Requires reconstructing the grid + voronoi allocation
            assembled_dir = data_dir / 'features' / 'assembled'
            with open(assembled_dir / 'boerde_grid_points.pickle', 'rb') as f:
                grid_gdf, _ = pickle.load(f)

            from SpatialAllocation.Allocator import allocator_registry
            alloc = allocator_registry.create('voronoi')
            voronoi_res = alloc.allocate(grid_gdf, subs_gdf)

            for demand_key in grid_demands:
                demand_arr = grid_demands[demand_key]
                predicted = np.zeros(n_subs)
                for t_idx in range(n_subs):
                    mask = voronoi_res.assignment == t_idx
                    predicted[t_idx] = demand_arr[mask].sum()

                rm = ranking_metrics(actual, predicted, TOP_K_VALUES)
                rm['method'] = demand_key
                rm['config'] = config
                rm['seed'] = seed
                seed_metrics.append(rm)

        if seed_metrics:
            seed_df = pd.DataFrame(seed_metrics)
            # Aggregate by method across seeds
            for method in seed_df['method'].unique():
                sub = seed_df[seed_df['method'] == method]
                row = {
                    'method': f'{method} [{config}]',
                    'spearman_rho': f'{sub["spearman_rho"].mean():.4f}±{sub["spearman_rho"].std():.4f}',
                    'spearman_rho_val': sub['spearman_rho'].mean(),
                }
                for k in TOP_K_VALUES:
                    col = f'top{k}_precision'
                    if col in sub.columns:
                        row[col] = f'{sub[col].mean():.4f}±{sub[col].std():.4f}'
                        row[f'{col}_val'] = sub[col].mean()
                rows.append(row)

    if not rows:
        print('No results available, please run 005 first')
        return

    result_df = pd.DataFrame(rows).set_index('method')
    if 'spearman_rho_val' in result_df.columns:
        result_df = result_df.sort_values('spearman_rho_val', ascending=False)

    display_cols = ['spearman_rho'] + [f'top{k}_precision' for k in TOP_K_VALUES]
    display_cols = [c for c in display_cols if c in result_df.columns]

    print('=' * 70)
    print('Börde ranking metrics (Voronoi allocation)')
    print(f'Note: N={n_subs}, Spearman test has limited statistical power')
    print('=' * 70)
    print(result_df[display_cols].to_string())

    result_df.to_csv(OUTPUT_DIR / 'ranking_metrics.csv')
    print(f'\nSaved to {OUTPUT_DIR / "ranking_metrics.csv"}')


if __name__ == '__main__':
    main()
