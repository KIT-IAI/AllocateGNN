"""
014 Improvement ratio visualization (Exp 8) - Voronoi allocation base

Uses the industry baseline Uni (voronoi uniform) as the reference floor,
and computes the Improvement Ratio for each method = (Uni_RMSE - Method_RMSE) / Uni_RMSE x 100%.

Horizontal bar chart, sorted from best to worst RMSE.
Colors: static (green tones), GNN+post-correction (blue tones), GNN+prior (orange tones).
GNNpostNP is marked with diagonal hatching to flag antagonism.

Usage:
    python 014_exp8_oracle_gap.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'no-latex'])
from matplotlib.patches import Patch

SCRIPT_DIR = Path(__file__).resolve().parent
STATIC_DIR = SCRIPT_DIR / 'results' / 'static_allocation'
EXP0_DIR = SCRIPT_DIR / 'results' / 'exp0_kfold_prior'
OUTPUT_DIR = SCRIPT_DIR / 'results' / 'exp8_improvement_ratio'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_LOCATIONS = [
    'London',
    'TLH2', 'TLH3', 'TLJ1', 'TLF1', 'TLF2',
    'TLC1', 'TLC2', 'TLD6', 'TLG1', 'TLG2', 'TLE4',
    'TLH1', 'TLE3', 'TLD3', 'TLD4',
]

SEEDS = [42, 123, 456]

# Uni = voronoi uniform (industry baseline)
B0_METHOD = 'voronoi'


def load_method_rmse(config_name, method):
    """Load the seed-averaged RMSE across the 16 regions for a GNN method."""
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

    if not all_vals:
        return None
    return np.array([np.mean(all_vals.get(loc, [np.nan])) for loc in ALL_LOCATIONS])


def load_static_rmse(method):
    """Load the RMSE for a static method."""
    path = STATIC_DIR / 'all_regions_rmse.csv'
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0)
    if method not in df.index:
        return None
    return df.loc[method, ALL_LOCATIONS].values.astype(float)


def run_improvement_ratio():
    """Compute the Improvement Ratio and plot the horizontal bar chart."""
    print('=' * 60)
    print('Exp 8: Improvement ratio visualization (Voronoi allocation base)')
    print('=' * 60)

    # Uni baseline RMSE
    b0_rmse = load_static_rmse(B0_METHOD)
    if b0_rmse is None:
        print(f'[Error] Uni baseline method {B0_METHOD} not found')
        return
    b0_mean = np.nanmean(b0_rmse)
    print(f'Uni (voronoi uniform) mean RMSE: {b0_mean:.4f}')

    # 14 methods (excluding Uni itself, which is used as the reference)
    # (label, source type, method name / GNN config, type label)
    methods = [
        ('GPM: voronoi_gpm', 'static', 'voronoi_gpm', None, 'Static'),
        ('UniP: voronoi_prox2', 'static', 'voronoi_prox2', None, 'Static'),
        ('UniN: voronoi_ntl', 'static', 'voronoi_ntl', None, 'Static'),
        ('UniNP: voronoi_prox2_ntl', 'static', 'voronoi_prox2_ntl', None, 'Static'),
        ('GPMpostP: voronoi_prox2_gpm', 'static', 'voronoi_prox2_gpm', None, 'Static'),
        ('GPMpostN: voronoi_ntl_gpm', 'static', 'voronoi_ntl_gpm', None, 'Static'),
        ('GPMpostNP: voronoi_prox2_ntl_gpm', 'static', 'voronoi_prox2_ntl_gpm', None, 'Static'),
        ('GNN: voronoi_GNN', 'gnn', 'baseline', 'voronoi_GNN', 'GNN+Post-Corr'),
        ('GNNpostN: voronoi_ntl_GNN', 'gnn', 'baseline', 'voronoi_ntl_GNN', 'GNN+Post-Corr'),
        ('GNNpostP: voronoi_prox_GNN', 'gnn', 'baseline', 'voronoi_prox_GNN', 'GNN+Post-Corr'),
        ('GNNpostNP: voronoi_ntl_prox_GNN', 'gnn', 'baseline', 'voronoi_ntl_prox_GNN', 'GNN+Post-Corr'),
        ('GNNpriorN: voronoi_GNN [NTL prior]', 'gnn', 'ntl', 'voronoi_GNN', 'GNN+Prior'),
        ('GNNpriorP: voronoi_GNN [Prox prior]', 'gnn', 'proximity', 'voronoi_GNN', 'GNN+Prior'),
        ('GNNpriorNP: voronoi_GNN [NTL+Prox prior]', 'gnn', 'ntl_prox', 'voronoi_GNN', 'GNN+Prior'),
    ]

    results = []

    for entry in methods:
        label = entry[0]
        source = entry[1]
        method_type = entry[-1]

        if source == 'static':
            method_name = entry[2]
            method_rmse = load_static_rmse(method_name)
        else:
            config_name = entry[2]
            gnn_method = entry[3]
            method_rmse = load_method_rmse(config_name, gnn_method)

        if method_rmse is None:
            results.append({
                'method': label, 'type': method_type,
                'mean_rmse': np.nan, 'improvement_ratio': np.nan,
            })
            continue

        mean_rmse = np.nanmean(method_rmse)
        improvement = (b0_mean - mean_rmse) / b0_mean * 100

        results.append({
            'method': label,
            'type': method_type,
            'mean_rmse': round(mean_rmse, 4),
            'improvement_ratio': round(improvement, 2),
        })

        print(f'  {label}: RMSE={mean_rmse:.4f}, Improvement={improvement:.2f}%')

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / 'improvement_ratio.csv', index=False)

    # Plot horizontal bar chart (Figure 4)
    valid = results_df.dropna(subset=['mean_rmse']).copy()
    # Sort by RMSE descending (improvement ratio ascending), so top of the chart = best
    valid = valid.sort_values('mean_rmse', ascending=True)

    # Color mapping
    color_map = {
        'Static': 'tab:green',
        'GNN+Post-Corr': 'tab:blue',
        'GNN+Prior': 'tab:orange',
    }

    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    y_pos = range(len(valid))
    colors = [color_map.get(t, 'gray') for t in valid['type']]

    bars = ax.barh(y_pos, valid['improvement_ratio'], color=colors, alpha=0.85,
                   edgecolor='black', linewidth=0.3)

    # GNNpostNP hatching for antagonism
    for bar, method_name in zip(bars, valid['method']):
        if 'GNNpostNP:' in method_name:
            bar.set_hatch('///')
            bar.set_edgecolor('darkred')
            bar.set_linewidth(0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(valid['method'], fontsize=6)
    ax.set_xlabel('Improvement Ratio (%)', fontsize=8)
    ax.set_title('Improvement over Uni Baseline', fontsize=8)
    ax.tick_params(axis='both', labelsize=8)

    # Uni reference line
    ax.axvline(x=0, color='black', linewidth=0.8, linestyle='-')

    # Value annotations
    for bar, val in zip(bars, valid['improvement_ratio']):
        if np.isnan(val):
            continue
        x_pos = bar.get_width()
        ax.text(x_pos + 0.3, bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}%', va='center', fontsize=6)

    # Legend
    legend_elements = [
        Patch(facecolor='tab:green', alpha=0.85, edgecolor='black',
              linewidth=0.5, label='Static'),
        Patch(facecolor='tab:blue', alpha=0.85, edgecolor='black',
              linewidth=0.5, label='GNN + Post-Corr'),
        Patch(facecolor='tab:orange', alpha=0.85, edgecolor='black',
              linewidth=0.5, label='GNN + Prior'),
        Patch(facecolor='white', edgecolor='darkred', linewidth=1.0,
              hatch='///', label='Antagonism'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=6)

    ax.grid(True, alpha=0.2, axis='x')
    ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'figure4_improvement_ratio.png', dpi=200, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / 'figure4_improvement_ratio.pdf', bbox_inches='tight')
    plt.close(fig)

    print(f'\nExp 8 complete, output: {OUTPUT_DIR}')


if __name__ == '__main__':
    run_improvement_ratio()
