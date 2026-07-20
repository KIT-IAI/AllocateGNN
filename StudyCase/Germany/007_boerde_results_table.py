"""
007 Börde results comparison table

Merges 003 static allocation + 005 GNN training results into a unified comparison table.
GNN rows show mean±std (across seeds); static rows show fixed values.

Usage:
    python 007_boerde_results_table.py
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
STATIC_DIR = SCRIPT_DIR / 'results' / 'static_allocation'
MODEL_DIR = SCRIPT_DIR / 'results' / 'models'
SUMMARY_DIR = MODEL_DIR / 'summary'
OUTPUT_DIR = SCRIPT_DIR / 'results'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METRIC_COLS = ['rmse', 'mae', 'corr']
GNN_CONFIGS = ['baseline', 'ntl', 'proximity', 'ntl_prox']


def load_static_metrics():
    """Load 003 static allocation metrics."""
    path = STATIC_DIR / 'boerde_static_metrics.csv'
    if not path.exists():
        print(f'Warning: static metrics file not found: {path}')
        return pd.DataFrame()
    return pd.read_csv(path, index_col=0)


def load_gnn_summary(config_name):
    """Load 005 GNN training summary (mean±std)."""
    path = SUMMARY_DIR / f'{config_name}_metrics.csv'
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0)


def main():
    # Load static metrics
    static_df = load_static_metrics()
    if static_df.empty:
        print('No static metrics found, exiting')
        return

    # Prepare output table
    rows = []

    # Static methods (fixed values)
    for method in static_df.index:
        row = {'method': method, 'type': 'static'}
        for col in METRIC_COLS:
            if col in static_df.columns:
                val = static_df.loc[method, col]
                row[f'{col}'] = f'{val:.4f}'
                row[f'{col}_val'] = val
        rows.append(row)

    # GNN methods (mean±std)
    for config in GNN_CONFIGS:
        summary = load_gnn_summary(config)
        if summary is None:
            print(f'Skipping GNN config {config} (no summary file)')
            continue

        for method in summary.index:
            row = {
                'method': f'{method} [{config}]',
                'type': f'gnn_{config}',
            }
            for col in METRIC_COLS:
                mean_col = f'{col}_mean'
                std_col = f'{col}_std'
                if mean_col in summary.columns and std_col in summary.columns:
                    mean_v = summary.loc[method, mean_col]
                    std_v = summary.loc[method, std_col]
                    row[col] = f'{mean_v:.4f}±{std_v:.4f}'
                    row[f'{col}_val'] = mean_v
                elif col in summary.columns:
                    row[col] = summary.loc[method, col]
                    row[f'{col}_val'] = float(
                        str(summary.loc[method, col]).split('±')[0])
            rows.append(row)

    result_df = pd.DataFrame(rows)
    result_df = result_df.set_index('method')

    # Sort by RMSE
    if 'rmse_val' in result_df.columns:
        result_df = result_df.sort_values('rmse_val')

    # Display
    display_cols = [c for c in METRIC_COLS if c in result_df.columns]
    display_cols.append('type')
    print('\n' + '=' * 80)
    print('Börde method comparison table')
    print('=' * 80)
    print(result_df[display_cols].to_string())

    # Save
    result_df.to_csv(OUTPUT_DIR / 'comparison_table.csv')
    print(f'\nSaved to {OUTPUT_DIR / "comparison_table.csv"}')

    # LaTeX format (optional)
    latex_cols = [c for c in METRIC_COLS if c in result_df.columns]
    latex_df = result_df[latex_cols]
    latex_path = OUTPUT_DIR / 'comparison_table.tex'
    latex_df.to_latex(latex_path, escape=False, caption='Börde allocation method comparison',
                      label='tab:boerde_comparison')
    print(f'LaTeX table saved to {latex_path}')


if __name__ == '__main__':
    main()
