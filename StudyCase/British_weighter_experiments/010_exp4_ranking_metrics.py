"""
010 Supplementary ranking metrics (Exp 4) — Voronoi baseline

Spearman rho + Top-k Precision/Recall.
Pure post-processing, no retraining required.
Aggregates to substations using Voronoi partitioning.

Usage:
    python 010_exp4_ranking_metrics.py
"""

import sys
import pickle
import math
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from SpatialAllocation.Allocator import allocator_registry
from SpatialAllocation.Allocator.clustering.do_clustering import do_clustering

DATA_DIR = SCRIPT_DIR / 'results' / 'intermediate'
ASSEMBLED_DIR = DATA_DIR / 'features' / 'assembled'
EXTRACTED_DIR = DATA_DIR / 'features' / 'extracted'
STATIC_DIR = SCRIPT_DIR / 'results' / 'static_allocation'
EXP0_DIR = SCRIPT_DIR / 'results' / 'exp0_kfold_prior'
OUTPUT_DIR = SCRIPT_DIR / 'results' / 'exp4_ranking_metrics'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_LOCATIONS = [
    'London',
    'TLH2', 'TLH3', 'TLJ1', 'TLF1', 'TLF2',
    'TLC1', 'TLC2', 'TLD6', 'TLG1', 'TLG2', 'TLE4',
    'TLH1', 'TLE3', 'TLD3', 'TLD4',
]

SEEDS = [42, 123, 456]


def compute_topk_metrics(actual, predicted, k):
    """Compute Top-k Precision and Recall."""
    if k <= 0 or len(actual) < k:
        return {'precision': float('nan'), 'recall': float('nan')}

    actual_topk = set(np.argsort(actual)[-k:])
    predicted_topk = set(np.argsort(predicted)[-k:])

    overlap = len(actual_topk & predicted_topk)
    precision = overlap / k
    recall = overlap / k  # |actual_topk| == k
    return {'precision': precision, 'recall': recall}


def load_substation_predictions(loc, config_name, seed, fold_num):
    """Reconstruct substation-level predictions.

    Reconstructed from grid_demands and the Voronoi allocation.
    """
    gd_path = (EXP0_DIR / f'seed_{seed}' / config_name
               / f'fold{fold_num}' / 'grid_demands' / f'{loc}_grid_demands.pickle')

    if not gd_path.exists():
        return None

    with open(gd_path, 'rb') as f:
        grid_demands = pickle.load(f)

    # Load grid points and substations
    with open(ASSEMBLED_DIR / f'{loc}_grid_points.pickle', 'rb') as f:
        grid_gdf, _ = pickle.load(f)

    substations_gdf = gpd.read_file(str(DATA_DIR / 'substations.gpkg'))
    study_itl3 = grid_gdf['ITL3'].unique()
    subs_sub = substations_gdf[substations_gdf['ITL3'].isin(study_itl3)].copy().reset_index(drop=True)

    # Voronoi allocation
    alloc_voronoi = allocator_registry.create('voronoi')
    voronoi_res = alloc_voronoi.allocate(grid_gdf, subs_sub)

    actual = subs_sub['Demand (MVA)'].values

    predictions = {}
    for demand_key in grid_demands:
        demand_arr = grid_demands[demand_key]
        predicted = np.zeros(len(subs_sub))
        for target_idx in range(len(subs_sub)):
            mask = voronoi_res.assignment == target_idx
            predicted[target_idx] = demand_arr[mask].sum()
        predictions[demand_key] = predicted

    return actual, predictions


def run_ranking_metrics():
    """Compute Spearman rho + Top-k Precision/Recall."""
    print('=' * 60)
    print('Exp 4: ranking metrics')
    print('=' * 60)

    import json

    all_results = []

    for seed in SEEDS:
        # Get the fold split
        for config_name in ['baseline', 'ntl', 'proximity', 'ntl_prox']:
            splits_path = EXP0_DIR / f'seed_{seed}' / config_name / 'kfold_splits.json'
            if not splits_path.exists():
                continue

            with open(splits_path, 'r') as f:
                splits = json.load(f)

            for fold_name, split_info in splits.items():
                fold_num = fold_name.split('_')[1]
                test_locs = split_info['test']

                for loc in test_locs:
                    result = load_substation_predictions(loc, config_name, seed, fold_num)
                    if result is None:
                        continue

                    actual, predictions = result
                    n_subs = len(actual)
                    k = max(3, math.ceil(n_subs * 0.1))

                    for demand_key, predicted in predictions.items():
                        # Spearman
                        try:
                            rho, p_spearman = spearmanr(actual, predicted)
                        except ValueError:
                            rho, p_spearman = float('nan'), 1.0

                        # Top-k
                        topk = compute_topk_metrics(actual, predicted, k)

                        all_results.append({
                            'seed': seed,
                            'config': config_name,
                            'fold': int(fold_num),
                            'region': loc,
                            'method': demand_key,
                            'n_substations': n_subs,
                            'k': k,
                            'spearman_rho': round(rho, 4),
                            'p_spearman': round(p_spearman, 6),
                            'topk_precision': round(topk['precision'], 4),
                            'topk_recall': round(topk['recall'], 4),
                        })

    if not all_results:
        print('[Warning] no results (grid demands files may not exist)')
        return

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / 'ranking_metrics_raw.csv', index=False)

    # Summary: aggregate by config x method (across seed x region)
    summary = results_df.groupby(['config', 'method']).agg({
        'spearman_rho': ['mean', 'std'],
        'topk_precision': ['mean', 'std'],
        'topk_recall': ['mean', 'std'],
    }).round(4)
    summary.columns = ['_'.join(col) for col in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(OUTPUT_DIR / 'ranking_metrics_summary.csv', index=False)

    print('\n=== Ranking metrics summary (voronoi_GNN method) ===')
    # Only display the gnn_demand method
    gnn_only = summary[summary['method'] == 'gnn_demand']
    if not gnn_only.empty:
        print(gnn_only.to_string(index=False))

    print(f'\nExp 4 complete, output: {OUTPUT_DIR}')


if __name__ == '__main__':
    run_ranking_metrics()
