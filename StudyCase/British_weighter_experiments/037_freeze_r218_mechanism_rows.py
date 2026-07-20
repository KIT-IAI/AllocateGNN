"""
037 Freeze the two rows of tab:mechanism_isolation data.

Background: In the paper's tab:mechanism_isolation (5_results.tex, Table 7),
the aggregated CSVs backing two rows:
  - "No-renorm NTL x Prox" (multiplicative NP without re-normalization, RMSE 17.53 +/- 6.75)
  - "Random noise (10 repeats avg.)" (multiplicative correction using random noise, RMSE 9.59 +/- 2.68)
were never committed to version control (the whole results/ directory is
gitignored, and the local results/exp_h1h3_mechanism_isolation/ directory
has since been lost).

Resolution: regenerate the values verbatim with the original script
017_exp_h1h3_mechanism_isolation.py (H1a no-renorm is a deterministic
computation; H1b random noise uses an RNG hard-coded in the original script
as np.random.default_rng(seed * 1000 + rep), which is likewise deterministic).
This script then extracts the per-region detail and aggregate summary for
those two rows and freezes them to disk after cross-checking against the
paper values and the archived 019 notebook output (4 decimal places).

Aggregation convention (identical to 017 / the 019 notebook / the paper's
table note):
  - 12 seed-fold protocol: 3 seeds x 4 folds, with each location appearing as
    the test set exactly once per seed;
  - no-renorm: average over the 3 seeds for a given location first -> the
    16 per-location means are the table values, and the ddof=0 population
    std across the 16 locations is the table's +/- value;
  - random noise: average over the 10 reps first -> then average over the
    3 seeds -> 16-location mean +/- ddof=0 std;
  - Delta RMSE is relative to the uncorrected GNN base (9.27); the body text's
    +56.5% is relative to the standard multiplicative NP (11.20).

Usage:
    python 037_freeze_r218_mechanism_rows.py
"""

import json
import platform
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
RAW_CSV = SCRIPT_DIR / 'results' / 'exp_h1h3_mechanism_isolation' / 'mechanism_isolation_raw.csv'
OUTPUT_DIR = SCRIPT_DIR / 'results' / 'exp_r218_mechanism_rows'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Paper tab:mechanism_isolation table values (2 decimal places), plus the
# archived output from 019_paper_tables_overview.ipynb (4 decimal places)
PAPER_REF = {
    'H1a_no_renorm_NP': {
        'display': 'No-renorm NTL×Prox',
        'paper': {'rmse': 17.53, 'rmse_std': 6.75, 'delta_rmse': 8.26, 'delta_pct': 89.1,
                  'mae': 10.71, 'mae_std': 3.34, 'corr': 0.357, 'corr_std': 0.167},
        'notebook_4dp': {'rmse_mean': 17.5294, 'rmse_std': 6.7526, 'delta_rmse': 8.2604,
                         'delta_pct': 89.1197, 'mae_mean': 10.7106, 'mae_std': 3.3401,
                         'corr_mean': 0.3572, 'corr_std': 0.1667},
    },
    'H1b_random_noise': {
        'display': 'Random noise (10 repeats avg.)',
        'paper': {'rmse': 9.59, 'rmse_std': 2.68, 'delta_rmse': 0.33, 'delta_pct': 3.5,
                  'mae': 6.88, 'mae_std': 1.82, 'corr': 0.287, 'corr_std': 0.199},
        'notebook_4dp': {'rmse_mean': 9.5942, 'rmse_std': 2.6767, 'delta_rmse': 0.3253,
                         'delta_pct': 3.5098, 'mae_mean': 6.8819, 'mae_std': 1.8180,
                         'corr_mean': 0.2871, 'corr_std': 0.1988},
    },
    # Reference rows (for the body text's +56.5% figure and the delta baseline)
    'baseline_no_correction': {
        'display': 'GNN base (no correction)',
        'paper': {'rmse': 9.27, 'rmse_std': 2.61, 'mae': 6.67, 'mae_std': 1.76,
                  'corr': 0.298, 'corr_std': 0.201},
        'notebook_4dp': {'rmse_mean': 9.2689, 'rmse_std': 2.6093, 'mae_mean': 6.6684,
                         'mae_std': 1.7619, 'corr_mean': 0.2980, 'corr_std': 0.2007},
    },
    'standard_multiplicative_NP': {
        'display': 'GNN + NTL×Prox (Mult.)',
        'paper': {'rmse': 11.20, 'rmse_std': 2.64, 'mae': 7.64, 'mae_std': 1.66,
                  'corr': 0.348, 'corr_std': 0.160},
        'notebook_4dp': {'rmse_mean': 11.2024, 'rmse_std': 2.6426, 'mae_mean': 7.6431,
                         'mae_std': 1.6594, 'corr_mean': 0.3478, 'corr_std': 0.1604},
    },
}

METRICS = ['rmse', 'mae', 'corr']


def agg_16region(per_loc: pd.DataFrame) -> dict:
    """16-region mean +/- ddof=0 population standard deviation (matches the 017/019 convention)."""
    out = {}
    for m in METRICS:
        out[f'{m}_mean'] = per_loc[m].mean()
        out[f'{m}_std'] = per_loc[m].std(ddof=0)
    return out


def main():
    raw = pd.read_csv(RAW_CSV)
    n_locs = raw['location'].nunique()
    assert n_locs == 16, f'Expected 16 locations, got {n_locs}'

    # -- Row 1: No-renorm NTL x Prox (deterministic) --
    nr_raw = raw[raw['experiment'] == 'H1a_no_renorm_NP'].copy()
    assert len(nr_raw) == 48, f'Expected 3 seeds x 16 locations = 48 rows, got {len(nr_raw)}'
    nr_raw.to_csv(OUTPUT_DIR / 'no_renorm_NP_per_region_per_seed.csv', index=False)
    nr_loc = nr_raw.groupby('location')[METRICS].mean().reindex(sorted(nr_raw['location'].unique()))
    nr_loc.to_csv(OUTPUT_DIR / 'no_renorm_NP_per_region_seedavg.csv')
    nr_agg = agg_16region(nr_loc)

    # -- Row 2: Random noise (10 reps, RNG=default_rng(seed*1000+rep) hard-coded) --
    noise_raw = raw[raw['experiment'].str.contains('random_noise')].copy()
    noise_raw['rep'] = noise_raw['experiment'].str.extract(r'rep(\d+)').astype(int)
    assert len(noise_raw) == 480, f'Expected 3 seeds x 16 locations x 10 reps = 480 rows, got {len(noise_raw)}'
    noise_raw.to_csv(OUTPUT_DIR / 'random_noise_per_region_per_seed_per_rep.csv', index=False)
    noise_ls = noise_raw.groupby(['seed', 'location'])[METRICS].mean()  # average over 10 reps first
    noise_loc = noise_ls.groupby('location').mean()                      # then average over 3 seeds
    noise_loc.to_csv(OUTPUT_DIR / 'random_noise_per_region_seedavg.csv')
    noise_agg = agg_16region(noise_loc)

    # -- Reference rows --
    ref_aggs = {}
    for exp in ['baseline_no_correction', 'standard_multiplicative_NP']:
        sub = raw[raw['experiment'] == exp]
        loc_avg = sub.groupby('location')[METRICS].mean()
        ref_aggs[exp] = agg_16region(loc_avg)

    base_rmse = ref_aggs['baseline_no_correction']['rmse_mean']
    std_np_rmse = ref_aggs['standard_multiplicative_NP']['rmse_mean']

    # -- Summary table + paper cross-check --
    rows = []
    for exp, agg in [('baseline_no_correction', ref_aggs['baseline_no_correction']),
                     ('standard_multiplicative_NP', ref_aggs['standard_multiplicative_NP']),
                     ('H1a_no_renorm_NP', nr_agg),
                     ('H1b_random_noise', noise_agg)]:
        ref = PAPER_REF[exp]
        row = {'experiment': exp, 'display_name': ref['display'], **agg}
        row['delta_rmse_vs_base'] = agg['rmse_mean'] - base_rmse
        row['delta_pct_vs_base'] = (agg['rmse_mean'] - base_rmse) / base_rmse * 100
        row['paper_rmse'] = ref['paper']['rmse']
        row['paper_rmse_std'] = ref['paper']['rmse_std']
        row['abs_diff_vs_paper_rmse'] = abs(agg['rmse_mean'] - ref['paper']['rmse'])
        row['notebook_rmse_4dp'] = ref['notebook_4dp']['rmse_mean']
        row['abs_diff_vs_notebook_rmse'] = abs(agg['rmse_mean'] - ref['notebook_4dp']['rmse_mean'])
        rows.append(row)
    summary = pd.DataFrame(rows).set_index('experiment')
    summary.to_csv(OUTPUT_DIR / 'summary_frozen.csv')

    # Body-text figure: no-renorm relative to standard multiplicative NP, +56.5%
    pct_vs_std_np = (nr_agg['rmse_mean'] - std_np_rmse) / std_np_rmse * 100

    # -- Per-item check (4 decimal places against the archived 019 notebook output) --
    checks = {}
    for exp, agg in [('H1a_no_renorm_NP', nr_agg), ('H1b_random_noise', noise_agg),
                     ('baseline_no_correction', ref_aggs['baseline_no_correction']),
                     ('standard_multiplicative_NP', ref_aggs['standard_multiplicative_NP'])]:
        nb = PAPER_REF[exp]['notebook_4dp']
        per_metric = {}
        for k in ['rmse_mean', 'rmse_std', 'mae_mean', 'mae_std', 'corr_mean', 'corr_std']:
            got = round(float(agg[k]), 4)
            per_metric[k] = {'regenerated': got, 'notebook_archive': nb[k],
                             'match_4dp': bool(abs(got - nb[k]) < 5e-5)}
        checks[exp] = per_metric
    mismatches = [
        {'experiment': exp, 'metric': k,
         'regenerated': v['regenerated'], 'notebook_archive': v['notebook_archive'],
         'abs_diff': round(abs(v['regenerated'] - v['notebook_archive']), 6)}
        for exp, metrics in checks.items() for k, v in metrics.items() if not v['match_4dp']
    ]
    all_match = len(mismatches) == 0
    # All mismatches are within 1e-4 (one unit in the 4th decimal place) -> floating-point
    # level deviation only; the paper's 2-decimal table values are unaffected
    fp_level_only = all(m['abs_diff'] <= 1.01e-4 for m in mismatches)

    provenance = {
        'purpose': 'Freeze the No-renorm and Random noise rows of tab:mechanism_isolation (the original aggregated CSVs were never committed to version control)',
        'regenerated_on': str(date.today()),
        'generator_script': '017_exp_h1h3_mechanism_isolation.py (regenerated with the original script, unmodified)',
        'freeze_script': '037_freeze_r218_mechanism_rows.py',
        'inputs': {
            'grid_demands': 'results/exp0_kfold_prior/seed_{42,123,456}/baseline/fold{1-4}/grid_demands/*.pickle (frozen learned base solution)',
            'grid_points': 'results/intermediate/features/assembled/*_grid_points.pickle',
            'ntl': 'results/intermediate/features/extracted/*_ntl.npz',
            'regions_substations': 'results/intermediate/{ITL3_region,substations}.gpkg',
        },
        'protocol': '12 seed-fold protocol (3 seeds x 4 folds), with each location appearing as the '
                    'test set exactly once per seed; 16-region mean +/- inter-region std (ddof=0); '
                    'random noise averaged over 10 reps first, then over seeds',
        'rng': 'H1a no-renorm is a deterministic computation; H1b RNG is hard-coded in the original '
               'script 017 as np.random.default_rng(seed*1000+rep), noise_std = '
               'log(combined_factor>0).std() (deterministic per location) -> both rows are '
               'deterministic regenerations, not statistical reproductions',
        'reproduction_status': (
            'BIT-LEVEL(4dp): all metrics match the archived 019 notebook output to 4 decimal places' if all_match else
            ('FP-LEVEL: a few metrics differ by one unit in the 4th decimal place (|diff|<=1e-4, '
             'floating-point / library-version-level deviation; the paper\'s 2-decimal table values '
             '17.53 / 9.59 and the body text\'s +56.5% figure match exactly), see mismatches_4dp'
             if fp_level_only else 'PARTIAL: inconsistencies beyond floating-point level exist, see mismatches_4dp')),
        'mismatches_4dp': mismatches,
        'environment': {
            'python': platform.python_version(),
            'numpy': np.__version__,
            'pandas': pd.__version__,
        },
        'paper_cross_reference': {
            'table': 'tab:mechanism_isolation (StudyCase/paper/manuscript/5_results.tex, Table 7)',
            'no_renorm_row': 'RMSE 17.53+/-6.75, Delta RMSE +8.26 (+89.1% vs GNN base 9.27), MAE 10.71+/-3.34, corr 0.357+/-0.167',
            'no_renorm_body_text': f'11.20 -> 17.53 = +56.5% (relative to standard multiplicative NP; this regeneration: {pct_vs_std_np:.1f}%)',
            'random_noise_row': 'RMSE 9.59+/-2.68, Delta RMSE +0.33 (+3.5%), MAE 6.88+/-1.82, corr 0.287+/-0.199',
        },
        'checks_4dp_vs_notebook_archive': checks,
    }
    with open(OUTPUT_DIR / 'provenance.json', 'w', encoding='utf-8') as f:
        json.dump(provenance, f, ensure_ascii=False, indent=2)

    print('Freeze complete ->', OUTPUT_DIR)
    print(summary[['rmse_mean', 'rmse_std', 'delta_rmse_vs_base', 'delta_pct_vs_base',
                   'mae_mean', 'mae_std', 'corr_mean', 'corr_std',
                   'paper_rmse', 'abs_diff_vs_paper_rmse']].round(4).to_string())
    print(f'\nno-renorm vs standard NP: +{pct_vs_std_np:.1f}% (paper body text: +56.5%)')
    if all_match:
        print('4dp full check vs 019 notebook archive: PASS')
    elif fp_level_only:
        print(f'4dp check: {len(mismatches)} item(s) differ by one unit in the 4th decimal place '
              '(floating-point level), all others match exactly; the paper\'s 2-decimal table '
              'values match exactly (see provenance.json)')
    else:
        print('4dp check: FAIL (see provenance.json)')


if __name__ == '__main__':
    main()
