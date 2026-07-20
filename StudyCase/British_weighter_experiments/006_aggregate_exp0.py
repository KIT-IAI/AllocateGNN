"""
006 Cross-seed aggregation script (Exp 0 summary)

Iterates over kfold_test_*.csv for 3 seeds x 4 configs,
computes mean±std summary tables, and saves them to the summary/ directory.

Usage:
    python 006_aggregate_exp0.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
EXP0_DIR = SCRIPT_DIR / 'results' / 'exp0_kfold_prior'
SUMMARY_DIR = EXP0_DIR / 'summary'
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 456]
CONFIGS = ['baseline', 'ntl', 'proximity', 'ntl_prox']
METRICS = ['rmse', 'mae', 'corr']


def load_kfold_csv(seed, config_name, metric):
    """Load kfold_test_{metric}.csv for a single seed/config."""
    path = EXP0_DIR / f'seed_{seed}' / config_name / f'kfold_test_{metric}.csv'
    if not path.exists():
        print(f'  [Warning] file not found: {path}')
        return None
    return pd.read_csv(path, index_col=0)


def aggregate():
    """Aggregate results across all seeds x configs."""
    print('=' * 60)
    print('Aggregating Exp 0 results...')
    print('=' * 60)

    for metric in METRICS:
        print(f'\n--- {metric.upper()} ---')

        # Collect the mean column for each config across seeds
        summary_rows = []

        for config_name in CONFIGS:
            seed_means = []
            seed_dfs = []

            for seed in SEEDS:
                df = load_kfold_csv(seed, config_name, metric)
                if df is None:
                    continue
                seed_dfs.append(df)
                if 'mean' in df.columns:
                    seed_means.append(df['mean'])

            if not seed_dfs:
                print(f'  {config_name}: no data available')
                continue

            # Aggregate across seeds, per method x per region
            all_methods = seed_dfs[0].index.tolist()
            # Region columns after dropping the mean column
            region_cols = [c for c in seed_dfs[0].columns if c != 'mean']

            for method in all_methods:
                values = []
                for df in seed_dfs:
                    if method in df.index:
                        values.append(df.loc[method, region_cols].values.astype(float))

                if not values:
                    continue

                arr = np.array(values)  # (n_seeds, n_regions)
                mean_per_region = arr.mean(axis=0)
                std_per_region = arr.std(axis=0)
                grand_mean = mean_per_region.mean()
                grand_std = mean_per_region.std()

                summary_rows.append({
                    'config': config_name,
                    'method': method,
                    'mean': round(grand_mean, 4),
                    'std': round(grand_std, 4),
                    'seed_mean': round(np.mean([v.mean() for v in values]), 4),
                    'seed_std': round(np.std([v.mean() for v in values]), 4),
                    'n_seeds': len(values),
                })

        summary_df = pd.DataFrame(summary_rows)
        out_path = SUMMARY_DIR / f'summary_{metric}.csv'
        summary_df.to_csv(out_path, index=False)
        print(f'  Saved: {out_path}')
        print(summary_df.to_string(index=False))

    # Save per-region detail tables (one file per config, values from 3 seeds side by side)
    for metric in METRICS:
        for config_name in CONFIGS:
            dfs_by_seed = {}
            for seed in SEEDS:
                df = load_kfold_csv(seed, config_name, metric)
                if df is not None:
                    dfs_by_seed[seed] = df

            if not dfs_by_seed:
                continue

            # Output a detailed per-region table for the voronoi_GNN method
            target_method = 'voronoi_GNN'
            rows = []
            for seed, df in dfs_by_seed.items():
                if target_method in df.index:
                    row = df.loc[target_method].to_dict()
                    row['seed'] = seed
                    rows.append(row)

            if rows:
                detail_df = pd.DataFrame(rows).set_index('seed')
                detail_path = SUMMARY_DIR / f'detail_{config_name}_{metric}_voronoi_GNN.csv'
                detail_df.to_csv(detail_path)

    # Best epoch summary
    print(f'\n--- Best Epoch Summary ---')
    import json
    best_epoch_rows = []
    for seed in SEEDS:
        for config_name in CONFIGS:
            for fold in range(1, 5):
                # Locate the training log
                fold_dir = EXP0_DIR / f'seed_{seed}' / config_name / f'fold{fold}'
                log_files = list(fold_dir.glob('*_training_log.json')) if fold_dir.exists() else []
                if not log_files:
                    continue
                with open(log_files[0], 'r', encoding='utf-8') as f:
                    log = json.load(f)
                best_epoch_rows.append({
                    'seed': seed,
                    'config': config_name,
                    'fold': fold,
                    'best_epoch': log.get('best_epoch', None),
                    'best_loss': round(log.get('best_loss', float('nan')), 6),
                    'total_epochs': log.get('config', {}).get('epochs', None),
                })

    if best_epoch_rows:
        be_df = pd.DataFrame(best_epoch_rows)
        be_path = SUMMARY_DIR / 'best_epoch_summary.csv'
        be_df.to_csv(be_path, index=False)
        print(f'  Saved: {be_path}')
        print(be_df.to_string(index=False))
    else:
        print('  No training log files found')

    print(f'\nAggregation complete, results directory: {SUMMARY_DIR}')


if __name__ == '__main__':
    aggregate()
