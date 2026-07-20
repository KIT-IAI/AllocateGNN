"""
008 Statistical Significance Testing (Exp 2) - Voronoi Baseline

9 comparison pairs x Wilcoxon signed-rank test x Holm-Bonferroni correction.
Tests are performed separately for the RMSE, MAE, and Corr metrics.

Usage:
    python 008_exp2_significance.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, ttest_rel

SCRIPT_DIR = Path(__file__).resolve().parent
STATIC_DIR = SCRIPT_DIR / 'results' / 'static_allocation'
EXP0_DIR = SCRIPT_DIR / 'results' / 'exp0_kfold_prior'
OUTPUT_DIR = SCRIPT_DIR / 'results' / 'exp2_significance'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 456]
ALL_LOCATIONS = [
    'London',
    'TLH2', 'TLH3', 'TLJ1', 'TLF1', 'TLF2',
    'TLC1', 'TLC2', 'TLD6', 'TLG1', 'TLG2', 'TLE4',
    'TLH1', 'TLE3', 'TLD3', 'TLD4',
]

# Comparison pair definitions (Voronoi baseline)
# (label, method_a, config_a, method_b, config_b, hypothesis)
# For RMSE/MAE: method_a is expected to be better (lower value)
# For Corr: method_a is expected to be better (higher value)
COMPARISONS = [
    ('GNN vs GPM', 'voronoi_GNN', 'baseline', 'voronoi_gpm', 'static',
     'Whether GNN significantly outperforms GPM'),
    ('GNNpostP vs GNN', 'voronoi_prox_GNN', 'baseline', 'voronoi_GNN', 'baseline',
     'Whether Prox-only post-correction significantly improves GNN'),
    ('GNNpostN vs GNN', 'voronoi_ntl_GNN', 'baseline', 'voronoi_GNN', 'baseline',
     'Whether NTL-only post-correction significantly improves GNN'),
    ('GNNpostNP vs GNNpostP', 'voronoi_ntl_prox_GNN', 'baseline', 'voronoi_prox_GNN', 'baseline',
     'Whether the combined post-correction shows significant antagonism (core finding)'),
    ('GPMpostNP vs GPMpostN', 'voronoi_prox2_ntl_gpm', 'static', 'voronoi_ntl_gpm', 'static',
     'Whether stacking Proximity on the static method shows significant synergy'),
    ('GNNpriorN vs GNN', 'voronoi_GNN', 'ntl', 'voronoi_GNN', 'baseline',
     'Whether the NTL prior significantly outperforms uncorrected GNN'),
    ('GPMpostP vs GPM', 'voronoi_prox2_gpm', 'static', 'voronoi_gpm', 'static',
     'Static Prox2+GPM synergy'),
    ('GPMpostNP vs GPMpostP', 'voronoi_prox2_ntl_gpm', 'static', 'voronoi_prox2_gpm', 'static',
     'Incremental effect of static NTL on top of Prox2+GPM'),
    ('GNNpostP vs GPMpostNP', 'voronoi_prox_GNN', 'baseline', 'voronoi_prox2_ntl_gpm', 'static',
     'GNN+Prox post-correction vs. best static method'),
]

N_COMPARISONS = len(COMPARISONS)


def load_region_values(method, config_name, metric, seed=None):
    """Load metric values for the 16 regions.

    For static configs, reads directly from the static_allocation CSV.
    For GNN configs, reads the kfold results from exp0.
    """
    if config_name == 'static':
        path = STATIC_DIR / f'all_regions_{metric}.csv'
        if not path.exists():
            return None
        df = pd.read_csv(path, index_col=0)
        if method not in df.index:
            return None
        return df.loc[method, ALL_LOCATIONS].values.astype(float)

    if seed is not None:
        path = EXP0_DIR / f'seed_{seed}' / config_name / f'kfold_test_{metric}.csv'
        if not path.exists():
            return None
        df = pd.read_csv(path, index_col=0)
        if method not in df.index:
            return None
        cols = [c for c in ALL_LOCATIONS if c in df.columns]
        return df.loc[method, cols].values.astype(float)

    # Average across seeds
    all_vals = []
    for s in SEEDS:
        v = load_region_values(method, config_name, metric, seed=s)
        if v is not None:
            all_vals.append(v)
    if not all_vals:
        return None
    return np.mean(all_vals, axis=0)


def holm_bonferroni(p_values):
    """Holm-Bonferroni correction. Returns the list of corrected p-values."""
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    corrected = [None] * n
    cummax = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adjusted = p * (n - rank)
        cummax = max(cummax, adjusted)
        corrected[orig_idx] = min(cummax, 1.0)
    return corrected


def run_significance_tests():
    """Run all significance tests (Voronoi baseline)."""
    print('=' * 60)
    print('Exp 2: Statistical Significance Testing (Voronoi Baseline)')
    print('=' * 60)

    results = []

    for metric in ['rmse', 'mae', 'corr']:
        print(f'\n--- {metric.upper()} ---')

        raw_p_values = []
        metric_results = []

        for comp_label, method_a, config_a, method_b, config_b, hypothesis in COMPARISONS:
            # 16-region values averaged across seeds
            vals_a = load_region_values(method_a, config_a, metric)
            vals_b = load_region_values(method_b, config_b, metric)

            if vals_a is None or vals_b is None:
                print(f'  {comp_label}: data unavailable')
                metric_results.append({
                    'metric': metric, 'comparison': comp_label,
                    'hypothesis': hypothesis,
                    'mean_a': None, 'mean_b': None, 'mean_diff': None,
                    'wilcoxon_p': None, 'ttest_p': None,
                    'holm_p': None, 'significant_005': None,
                })
                raw_p_values.append(1.0)
                continue

            diff = vals_a - vals_b  # For RMSE/MAE: negative = a is better; for Corr: positive = a is better
            mean_diff = diff.mean()

            # Wilcoxon signed-rank
            try:
                stat_w, p_wilcoxon = wilcoxon(vals_a, vals_b, alternative='two-sided')
            except ValueError:
                p_wilcoxon = 1.0

            # Paired t-test
            try:
                stat_t, p_ttest = ttest_rel(vals_a, vals_b)
            except ValueError:
                p_ttest = 1.0

            raw_p_values.append(p_wilcoxon)
            metric_results.append({
                'metric': metric,
                'comparison': comp_label,
                'hypothesis': hypothesis,
                'mean_a': round(vals_a.mean(), 4),
                'mean_b': round(vals_b.mean(), 4),
                'mean_diff': round(mean_diff, 4),
                'wilcoxon_p': round(p_wilcoxon, 6),
                'ttest_p': round(p_ttest, 6),
                'holm_p': None,  # filled in below
                'significant_005': None,
            })

        # Holm-Bonferroni correction
        corrected_ps = holm_bonferroni(raw_p_values)
        for i, res in enumerate(metric_results):
            res['holm_p'] = round(corrected_ps[i], 6)
            res['significant_005'] = corrected_ps[i] < 0.05

        for res in metric_results:
            sig_mark = '**' if (res['holm_p'] or 1.0) < 0.01 else (
                '*' if res.get('significant_005') else '')
            diff_str = f"{res['mean_diff']:+.4f}" if res['mean_diff'] is not None else 'N/A'
            p_str = f"{res['wilcoxon_p']:.4f}" if res['wilcoxon_p'] is not None else 'N/A'
            holm_str = f"{res['holm_p']:.4f}" if res['holm_p'] is not None else 'N/A'
            print(f"  {res['comparison']}: diff={diff_str}, "
                  f"p_wilcoxon={p_str}, p_holm={holm_str} {sig_mark}")

        results.extend(metric_results)

        # Per-seed independent tests (RMSE metric)
        if metric == 'rmse':
            print(f'\n  --- Per-seed stability ({metric}) ---')
            for comp_label, method_a, config_a, method_b, config_b, _ in COMPARISONS:
                seed_ps = []
                for seed in SEEDS:
                    va = load_region_values(method_a, config_a, metric, seed=seed)
                    vb = load_region_values(method_b, config_b, metric, seed=seed)
                    if va is None or vb is None:
                        continue
                    try:
                        _, p = wilcoxon(va, vb, alternative='two-sided')
                    except ValueError:
                        p = 1.0
                    seed_ps.append(p)

                if seed_ps:
                    print(f'    {comp_label}: p_values={[round(p, 4) for p in seed_ps]}, '
                          f'median={np.median(seed_ps):.4f}')

    # Save results
    results_df = pd.DataFrame(results)
    out_path = OUTPUT_DIR / 'significance_tests.csv'
    results_df.to_csv(out_path, index=False)
    print(f'\nResults saved to: {out_path}')
    print(results_df.to_string(index=False))

    # Multi-seed stability summary (Voronoi baseline)
    print('\n--- Multi-seed method stability (voronoi_GNN) ---')
    stability_rows = []
    for config_name in ['baseline', 'ntl', 'proximity', 'ntl_prox']:
        for metric in ['rmse', 'mae', 'corr']:
            seed_means = []
            for seed in SEEDS:
                v = load_region_values('voronoi_GNN', config_name, metric, seed=seed)
                if v is not None:
                    seed_means.append(v.mean())
            if seed_means:
                stability_rows.append({
                    'config': config_name,
                    'metric': metric,
                    'mean': round(np.mean(seed_means), 4),
                    'std': round(np.std(seed_means), 4),
                    'values': [round(v, 4) for v in seed_means],
                })

    if stability_rows:
        stab_df = pd.DataFrame(stability_rows)
        stab_path = OUTPUT_DIR / 'seed_stability.csv'
        stab_df.to_csv(stab_path, index=False)
        print(stab_df.to_string(index=False))

    print(f'\nExp 2 complete. Output: {OUTPUT_DIR}')


if __name__ == '__main__':
    run_significance_tests()
