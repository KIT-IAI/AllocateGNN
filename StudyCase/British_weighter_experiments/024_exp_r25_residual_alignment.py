# -*- coding: utf-8 -*-
"""
024 - Residual alignment diagnostic

For a post-hoc correction to be effective, the effective correction
factor must align with the base allocation's residual (errors **or**
ratios). This experiment quantifies that alignment, per (base, region,
signal) combination, at the **substation level**, and pairs it with the
combination's delta-RMSE (corrected - base) to provide empirical
grounding for a conditional-alignment phase diagram (sigma_r and sigma_c
are written to disk per combination).

=== Definitions ===
- Base allocation in {Uniform, GPM, GNN per seed x fold}:
    * Uniform = per-ITL3 uniform allocation (total_demand/len(group));
      this is the same as the 'average_demand' produced by the uniform
      weighter + compute_demand in 003/005;
    * GPM = reconstructed from the 005 definition: categorical GPM
      weighter (dominant-landuse one-hot) + compute_demand (regional
      landuse-percentage weighting followed by per-ITL3 renormalization);
    * GNN = grid_demands['gnn_demand'] from the exp0 baseline
      configuration (a frozen, read-only artifact); for each (seed, fold)
      only that fold's **test region** is used, matching the same
      out-of-sample convention used in 017.
- Signal in {N, P, NP}: ntl_factor / prox_factor / their product, as
  defined in the shared compute_factors module (identical definition to
  017).
- Correction = shared_correction_utils.apply_standard_multiplicative
  (multiplicative correction + per-ITL3 renormalization); corrected
  demand is aggregated to substations via Voronoi assignment.
- At the substation level:
    rho_j = (d_true_j + eps) / (d_base_j + eps)   -- residual ratio
    F_j = (d_corr_j + eps) / (d_base_j + eps)     -- effective correction factor
  eps = regional total demand x 1e-6 (protects against division by zero
  in the log; the count of protected samples is written to disk).
- Ratio-based measures: Pearson/Spearman corr(log F, log rho) plus the
  log-log regression slope (computed in both directions).
- Difference-based measures: corr(F - F_bar, d_true - d_base).
- sigma_r = std(log rho), sigma_c = std(log F) (numpy population standard
  deviation, ddof=0), used as inputs to the alignment phase diagram.
- delta-RMSE = RMSE(corrected) - RMSE(base), where RMSE is computed on
  the **raw** (non-eps-protected) allocation values via the shared
  evaluate_allocation module.

=== Why only the substation level (agent-level analysis was dropped) ===
There is no observed ground truth at the agent (grid-point) level: actual
demand is only recorded at the substation level ('Demand (MVA)'), so a
"residual" is undefined at the grid-point level -- any agent-level rho
would first require assuming a grid-point-level ground-truth
distribution, i.e. using one unvalidated allocation model to evaluate
another. The alignment diagnostic is therefore restricted to the
substation level; this explanation is also written into
alignment_summary.json's meta.notes.

=== Built-in anchoring (guards against drift in the base-allocation
reconstruction) ===
- Uniform/GPM base RMSE is anchored against the voronoi / voronoi_gpm
  rows of each region's frozen {loc}_metrics_summary.csv (4 decimal
  places) -> tolerance 6e-5;
- Correction-arm (N/NP) RMSE is anchored against the voronoi_ntl /
  voronoi_prox2_ntl / voronoi_ntl_gpm / voronoi_prox2_ntl_gpm rows of the
  frozen all_regions_rmse.csv (2 decimal places) -> tolerance 0.005+1e-9;
  the P-only arm has no corresponding row in the frozen artifacts and is
  recorded as not_anchored;
- GNN base RMSE is anchored against the voronoi_GNN row of that
  (seed, fold)'s frozen fold/rmse.csv (4 decimal places) -> tolerance
  6e-5.
Any anchor failure raises an error immediately (an incorrect base
invalidates everything downstream).

Outputs:
    results/exp_r25/alignment_per_region.csv   -- per (base, region, signal) detail
    results/exp_r25/alignment_summary.json     -- summary by base type + anchor report

CPU-only, no randomness (deterministic post-processing). All pre-existing
frozen artifacts are read-only inputs.

Usage:
    python 024_exp_r25_residual_alignment.py
"""

import sys
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import shared_correction_utils as scu  # noqa: E402
from SpatialAllocation.Weighter import weighter_registry  # noqa: E402

# ════════════════════════════════════════════════════════════
# Paths and constants
# ════════════════════════════════════════════════════════════

EXP0_DIR = SCRIPT_DIR / 'results' / 'exp0_kfold_prior'
STATIC_DIR = SCRIPT_DIR / 'results' / 'static_allocation'
OUTPUT_DIR = SCRIPT_DIR / 'results' / 'exp_r25'

SEEDS = [42, 123, 456]
SIGNALS = ['N', 'P', 'NP']
EPS_RATIO = 1e-6            # eps = regional total demand x EPS_RATIO

# Anchor tolerance: for 4-decimal frozen values, half of the last digit
# (5e-5) plus floating-point slack; for 2-decimal values, 0.005
TOL_ANCHOR_4DP = 6e-5
TOL_ANCHOR_2DP = 0.005 + 1e-9
# GNN base anchor tolerance: beyond the 4-decimal rounding (5e-5), this
# must also tolerate GPU re-inference drift from the "backfill inference"
# branch in 005 (GPU forward passes are not bit-for-bit reproducible;
# measured on 2026-07-14 across 48 (seed, fold, loc) combinations, the
# maximum deviation was 8.7e-5 and the median was 2.5e-5). 5e-4 is chosen
# as roughly an order of magnitude above the measured maximum, while
# still two orders of magnitude below the deviation produced by any
# seed/fold mismatch (>= 1e-2 magnitude), so the anchor's ability to
# catch mismatches is unaffected.
TOL_ANCHOR_GNN = 5e-4

# Landuse columns and regional percentage columns from 005/003 (used to
# reconstruct the GPM base; order corresponds one-to-one)
LU_COLS = [
    'lu_residential_prop', 'lu_commercial_prop', 'lu_industrial_prop',
    'lu_agricultural_prop', 'lu_others_prop',
]
PCT_COLS = [
    'residential_percent', 'commercial_percent', 'industrial_percent',
    'agricultural_percent', 'others_percent',
]


# ════════════════════════════════════════════════════════════
# Static base reconstruction (matches the 005/003 definitions line-for-line)
# ════════════════════════════════════════════════════════════

def compute_uniform_base(grid_gdf, region_sub) -> np.ndarray:
    """Uniform base: per-ITL3 uniform allocation = total_demand / len(group).

    Equivalent to the 'average_demand' produced by the uniform weighter
    (W all ones) + compute_demand in 003.
    """
    region_info = region_sub.set_index('ITL3')
    base = np.zeros(len(grid_gdf))
    for itl3, group in grid_gdf.groupby('ITL3'):
        if itl3 not in region_info.index:
            continue
        total_demand = region_info.loc[itl3, 'Demand (MVA)']
        base[group.index] = total_demand / len(group)
    return base


def compute_gpm_base(grid_gdf, region_sub, subs_sub):
    """GPM base: line-for-line reconstruction of the 'landuse_demand'
    column from 005's load_data.

    categorical GPM weighter (dominant-landuse one-hot, W shape (N,5)) ->
    score = W @ regional landuse percentages -> per-ITL3 normalization x
    total_demand. The fallback branch (score_sum <= 0 -> uniform) matches
    005's compute_demand exactly.

    Returns (base array, number of ITL3 groups that triggered the
    fallback) -- the latter is written to disk for transparency.
    """
    gpm = weighter_registry.create(
        'gpm', config={'mode': 'categorical', 'proportion_columns': LU_COLS})
    gpm_res = gpm.compute(grid_gdf, target_gdf=subs_sub)
    W = gpm_res.weights

    region_info = region_sub.set_index('ITL3')
    base = np.zeros(len(grid_gdf))
    n_fallback = 0
    for itl3, group in grid_gdf.groupby('ITL3'):
        if itl3 not in region_info.index:
            continue
        total_demand = region_info.loc[itl3, 'Demand (MVA)']
        idx = group.index
        pcts = np.array([region_info.loc[itl3, c] for c in PCT_COLS])
        score = W[idx] @ pcts
        score_sum = score.sum()
        if score_sum > 0:
            base[idx] = total_demand * score / score_sum
        else:
            base[idx] = total_demand / len(group)
            n_fallback += 1
    return base, n_fallback


# ════════════════════════════════════════════════════════════
# Alignment statistics
# ════════════════════════════════════════════════════════════

def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Slope b of the OLS regression y = a + b*x (hand-implemented, deterministic)."""
    xc = x - x.mean()
    yc = y - y.mean()
    return float((xc @ yc) / (xc @ xc))


def alignment_stats(d_true: np.ndarray, d_base: np.ndarray,
                    d_corr: np.ndarray, eps: float) -> dict:
    """Compute substation-level alignment statistics for one
    (base, region, signal) combination.

    The ratio-based measures operate in log space; the difference-based
    measures operate in the raw value space. eps is only used to protect
    the ratio/log computation -- the difference-based d_true - d_base
    uses the raw values.
    """
    d_true = np.asarray(d_true, dtype=float)
    d_base = np.asarray(d_base, dtype=float)
    d_corr = np.asarray(d_corr, dtype=float)
    n = len(d_true)
    assert n >= 3, f'substation count {n} < 3, cannot compute correlation'

    # eps protection counts (number of samples that were <= 0 before protection)
    n_prot_true = int((d_true <= 0).sum())
    n_prot_base = int((d_base <= 0).sum())
    n_prot_corr = int((d_corr <= 0).sum())
    n_prot_any = int(((d_true <= 0) | (d_base <= 0) | (d_corr <= 0)).sum())

    # Ratio-based measures (log space)
    log_rho = np.log((d_true + eps) / (d_base + eps))
    log_f = np.log((d_corr + eps) / (d_base + eps))

    if np.std(log_f) == 0 or np.std(log_rho) == 0:
        raise RuntimeError('log F or log rho has zero variance -- correlation is undefined, check this combination')

    pearson_log = float(pearsonr(log_f, log_rho)[0])
    spearman_log = float(spearmanr(log_f, log_rho)[0])
    slope_logf_on_logrho = _ols_slope(log_rho, log_f)   # log F ~ log rho
    slope_logrho_on_logf = _ols_slope(log_f, log_rho)   # log rho ~ log F

    # Difference-based measure: corr(F - F_bar, d_true - d_base)
    f_eff = (d_corr + eps) / (d_base + eps)
    f_centered = f_eff - f_eff.mean()
    resid_diff = d_true - d_base
    if np.std(f_centered) == 0 or np.std(resid_diff) == 0:
        raise RuntimeError('Difference-based measure has zero variance -- correlation is undefined, check this combination')
    pearson_diff = float(pearsonr(f_centered, resid_diff)[0])
    spearman_diff = float(spearmanr(f_centered, resid_diff)[0])

    # sigma_r / sigma_c (inputs to the alignment phase diagram; ddof=0)
    sigma_r = float(np.std(log_rho))
    sigma_c = float(np.std(log_f))

    return {
        'n_substations': n,
        'eps': float(eps),
        'n_eps_protected_true': n_prot_true,
        'n_eps_protected_base': n_prot_base,
        'n_eps_protected_corrected': n_prot_corr,
        'n_eps_protected_any': n_prot_any,
        'pearson_log': pearson_log,
        'spearman_log': spearman_log,
        'slope_logF_on_logrho': slope_logf_on_logrho,
        'slope_logrho_on_logF': slope_logrho_on_logf,
        'pearson_diff': pearson_diff,
        'spearman_diff': spearman_diff,
        'sigma_r': sigma_r,
        'sigma_c': sigma_c,
        'sigma_ratio': sigma_c / sigma_r,
    }


# ════════════════════════════════════════════════════════════
# Region context + per-base analysis
# ════════════════════════════════════════════════════════════

def load_location_context(loc: str, assignment_cache: dict) -> dict:
    """Load all shared quantities for one region (computed once per region)."""
    grid_gdf, region_sub, subs_sub, ntl_values = scu.load_grid_and_subs(loc)

    # RangeIndex requirement (group.index is used as a positional index)
    assert (grid_gdf.index == np.arange(len(grid_gdf))).all(), \
        f'{loc}: grid_gdf is not a RangeIndex -- positional index semantics are broken'
    # ITL3 coverage: ensure no grid point falls outside the correction/renorm scope
    assert set(grid_gdf['ITL3'].unique()) <= set(region_sub['ITL3']), \
        f'{loc}: there are ITL3 values not covered by region_sub'

    prox_scores = scu.compute_prox_scores(grid_gdf, subs_sub)
    ntl_factor, prox_factor = scu.compute_factors(grid_gdf, ntl_values, prox_scores)
    factors = {'N': ntl_factor, 'P': prox_factor, 'NP': ntl_factor * prox_factor}

    assignment = scu.compute_voronoi_assignment(
        grid_gdf, subs_sub, cache=assignment_cache, cache_key=loc)

    d_true = subs_sub['Demand (MVA)'].values.astype(float)
    region_total = float(np.asarray(region_sub['Demand (MVA)'], dtype=float).sum())
    eps = region_total * EPS_RATIO

    return {
        'loc': loc, 'grid_gdf': grid_gdf, 'region_sub': region_sub,
        'subs_sub': subs_sub, 'factors': factors, 'assignment': assignment,
        'd_true': d_true, 'eps': eps,
    }


def analyse_base(ctx: dict, base_demand: np.ndarray,
                 base_type: str, base_id: str, seed, fold) -> list:
    """Compute alignment statistics and delta-RMSE for one base x three signals, returning a list of records."""
    grid_gdf = ctx['grid_gdf']
    region_sub = ctx['region_sub']
    subs_sub = ctx['subs_sub']
    assignment = ctx['assignment']

    subs_base = scu.aggregate_by_assignment(subs_sub, assignment, base_demand)
    d_base = subs_base['allocated_demand'].values.astype(float)
    m_base = scu.evaluate_allocation(subs_base)

    rows = []
    for signal in SIGNALS:
        corrected = scu.apply_standard_multiplicative(
            base_demand, ctx['factors'][signal], grid_gdf, region_sub)
        subs_corr = scu.aggregate_by_assignment(subs_sub, assignment, corrected)
        d_corr = subs_corr['allocated_demand'].values.astype(float)
        m_corr = scu.evaluate_allocation(subs_corr)

        stats = alignment_stats(ctx['d_true'], d_base, d_corr, ctx['eps'])
        rows.append({
            'base_type': base_type,
            'base_id': base_id,
            'seed': seed,
            'fold': fold,
            'location': ctx['loc'],
            'signal': signal,
            **stats,
            'rmse_base': float(m_base['rmse']),
            'rmse_corrected': float(m_corr['rmse']),
            'delta_rmse': float(m_corr['rmse'] - m_base['rmse']),
        })
    return rows


# ════════════════════════════════════════════════════════════
# Frozen-artifact anchoring
# ════════════════════════════════════════════════════════════

def _check_anchor(dev: float, tol: float, what: str, anchor_log: list):
    """Record one anchor check and raise if it exceeds tolerance."""
    anchor_log.append({'what': what, 'abs_dev': dev, 'tol': tol,
                       'status': 'ok' if dev <= tol else 'FAIL'})
    if dev > tol:
        raise RuntimeError(f'Anchor check failed: {what} deviation {dev:.3e} > tolerance {tol:.3e}')


def anchor_static(loc: str, rows: list, frozen_all: pd.DataFrame,
                  anchor_log: list):
    """Anchor the static bases and N/NP correction arms against frozen
    artifacts (the P arm has no frozen row and is recorded as skipped)."""
    # Base RMSE, anchored against {loc}_metrics_summary.csv (4 decimal places)
    ms = pd.read_csv(STATIC_DIR / f'{loc}_metrics_summary.csv', index_col=0)
    by = {(r['base_type'], r['signal']): r for r in rows}
    _check_anchor(abs(by[('uniform', 'N')]['rmse_base'] - float(ms.loc['voronoi', 'rmse'])),
                  TOL_ANCHOR_4DP, f'{loc}/uniform base vs metrics_summary voronoi', anchor_log)
    _check_anchor(abs(by[('gpm', 'N')]['rmse_base'] - float(ms.loc['voronoi_gpm', 'rmse'])),
                  TOL_ANCHOR_4DP, f'{loc}/gpm base vs metrics_summary voronoi_gpm', anchor_log)

    # Correction-arm RMSE, anchored against all_regions_rmse.csv (2 decimal places)
    arm_map = {
        ('uniform', 'N'): 'voronoi_ntl',
        ('uniform', 'NP'): 'voronoi_prox2_ntl',
        ('gpm', 'N'): 'voronoi_ntl_gpm',
        ('gpm', 'NP'): 'voronoi_prox2_ntl_gpm',
    }
    for key, frozen_row in arm_map.items():
        _check_anchor(abs(by[key]['rmse_corrected'] - float(frozen_all.loc[frozen_row, loc])),
                      TOL_ANCHOR_2DP, f'{loc}/{key[0]}+{key[1]} vs all_regions {frozen_row}',
                      anchor_log)


def anchor_gnn(loc: str, seed: int, fold_dir_name: str, rmse_base: float,
               anchor_log: list):
    """Anchor the GNN base RMSE against the voronoi_GNN row of that
    fold's frozen rmse.csv.

    Tolerance = TOL_ANCHOR_GNN (4-decimal rounding + GPU backfill
    re-inference drift, see the constant's comment above); the purpose of
    this anchor is to catch seed/fold/region pickle mismatches (mismatches
    produce deviations of >= 1e-2 magnitude).
    """
    fold_csv = EXP0_DIR / f'seed_{seed}' / 'baseline' / fold_dir_name / 'rmse.csv'
    frozen = pd.read_csv(fold_csv, index_col=0)
    _check_anchor(abs(rmse_base - float(frozen.loc['voronoi_GNN', loc])),
                  TOL_ANCHOR_GNN,
                  f'{loc}/gnn seed{seed} {fold_dir_name} vs fold rmse.csv voronoi_GNN',
                  anchor_log)


# ════════════════════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════════════════════

def _agg_cell(df: pd.DataFrame) -> dict:
    """One summary cell: median alignment/sigma ratio + delta-RMSE overview."""
    return {
        'n': int(len(df)),
        'median_pearson_log': float(df['pearson_log'].median()),
        'median_spearman_log': float(df['spearman_log'].median()),
        'median_pearson_diff': float(df['pearson_diff'].median()),
        'median_slope_logF_on_logrho': float(df['slope_logF_on_logrho'].median()),
        'median_sigma_r': float(df['sigma_r'].median()),
        'median_sigma_c': float(df['sigma_c'].median()),
        'median_sigma_ratio': float(df['sigma_ratio'].median()),
        'mean_delta_rmse': float(df['delta_rmse'].mean()),
        'median_delta_rmse': float(df['delta_rmse'].median()),
        'frac_delta_rmse_negative': float((df['delta_rmse'] < 0).mean()),
    }


def build_summary(df: pd.DataFrame, anchor_log: list,
                  gpm_fallback_regions: int) -> dict:
    """Summarize by base type (x signal), plus static-vs-GNN headline numbers and the anchor report."""
    by_base = {}
    for bt in ('uniform', 'gpm', 'gnn'):
        sub = df[df['base_type'] == bt]
        cell = {sig: _agg_cell(sub[sub['signal'] == sig]) for sig in SIGNALS}
        cell['all_signals'] = _agg_cell(sub)
        by_base[bt] = cell

    static_df = df[df['base_type'].isin(['uniform', 'gpm'])]
    gnn_df = df[df['base_type'] == 'gnn']
    headline = {
        'note': 'static = uniform + gpm combined; alignment = corr(log F, log rho)',
        'static': _agg_cell(static_df),
        'gnn': _agg_cell(gnn_df),
    }

    eps_prot = {
        'rule': f'eps = regional total demand x {EPS_RATIO}',
        'total_protected_any': int(df['n_eps_protected_any'].sum()),
        'rows_with_any_protection': int((df['n_eps_protected_any'] > 0).sum()),
        'by_base_type': {
            bt: {
                'protected_true': int(df.loc[df['base_type'] == bt, 'n_eps_protected_true'].sum()),
                'protected_base': int(df.loc[df['base_type'] == bt, 'n_eps_protected_base'].sum()),
                'protected_corrected': int(df.loc[df['base_type'] == bt, 'n_eps_protected_corrected'].sum()),
                'protected_any': int(df.loc[df['base_type'] == bt, 'n_eps_protected_any'].sum()),
            } for bt in ('uniform', 'gpm', 'gnn')
        },
    }

    n_regions = df['location'].nunique()
    expected = {
        'uniform': n_regions * len(SIGNALS),
        'gpm': n_regions * len(SIGNALS),
        'gnn': len(SEEDS) * n_regions * len(SIGNALS),
    }
    expected['total'] = sum(expected.values())

    anchors_failed = [a for a in anchor_log if a['status'] != 'ok']
    return {
        'meta': {
            'script': '024_exp_r25_residual_alignment.py',
            'purpose': 'Residual alignment diagnostic (substation level)',
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'notes': [
                'Substation level only: there is no observed ground truth '
                'at the agent (grid-point) level, so grid-point residuals '
                'are undefined; any agent-level rho would require first '
                'assuming an unvalidated grid-point-level ground-truth '
                'allocation. Agent-level analysis was therefore dropped, '
                'and only the substation level is retained.',
                'GNN base = exp0 baseline grid_demands (gnn_demand); for '
                'each (seed, fold) only that fold\'s test region is used '
                '(the same out-of-sample convention as 017).',
                'Uniform base combined with multiplicative correction '
                'does not degenerate (unlike additive correction, where '
                'the correction coefficient alpha is supplied by '
                'variation in the base, so it fails to have any effect '
                'on a uniform base).',
                'delta-RMSE is computed using the raw (non-eps-protected) '
                'allocation values.',
            ],
            'gpm_compute_demand_fallback_groups': gpm_fallback_regions,
        },
        'row_counts': {
            'expected': expected,
            'actual': {
                'uniform': int((df['base_type'] == 'uniform').sum()),
                'gpm': int((df['base_type'] == 'gpm').sum()),
                'gnn': int((df['base_type'] == 'gnn').sum()),
                'total': int(len(df)),
            },
        },
        'eps_protection': eps_prot,
        'anchor_check': {
            'n_anchors': len(anchor_log),
            'n_failed': len(anchors_failed),
            'max_abs_dev_static_base_anchors': float(max(
                (a['abs_dev'] for a in anchor_log if a['tol'] == TOL_ANCHOR_4DP),
                default=float('nan'))),
            'max_abs_dev_corrected_arm_anchors': float(max(
                (a['abs_dev'] for a in anchor_log if a['tol'] == TOL_ANCHOR_2DP),
                default=float('nan'))),
            'max_abs_dev_gnn_base_anchors': float(max(
                (a['abs_dev'] for a in anchor_log if a['tol'] == TOL_ANCHOR_GNN),
                default=float('nan'))),
            'gnn_tolerance_note': 'GNN anchor tolerance 5e-4 = 4dp rounding '
                                  '+ GPU re-inference drift from the 005 '
                                  'backfill branch (measured maximum '
                                  '8.7e-5); mismatch-detection is unaffected',
            'not_anchored': ['uniform_P', 'gpm_P'],
            'not_anchored_reason': 'the voronoi_prox2 / voronoi_prox2_gpm '
                                   'rows are not present in the frozen '
                                   'all_regions_rmse.csv (the P-only arm '
                                   'has no frozen anchor point)',
        },
        'alignment_by_base_type': by_base,
        'static_vs_gnn_headline': headline,
    }


# ════════════════════════════════════════════════════════════
# Main flow
# ════════════════════════════════════════════════════════════

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    frozen_all = pd.read_csv(STATIC_DIR / 'all_regions_rmse.csv', index_col=0)

    # Fold assignment per seed: loc -> fold directory name (each region
    # appears as the test set exactly once per seed)
    fold_of = {}
    for seed in SEEDS:
        splits_path = EXP0_DIR / f'seed_{seed}' / 'baseline' / 'kfold_splits.json'
        with open(splits_path, encoding='utf-8') as f:
            splits = json.load(f)
        mapping = {}
        for fold_key, info in splits.items():
            for loc in info['test']:
                assert loc not in mapping, f'seed {seed}: {loc} appears in more than one test fold'
                mapping[loc] = fold_key.replace('_', '')   # fold_1 -> fold1
        assert set(mapping) == set(scu.ALL_LOCATIONS), \
            f'seed {seed}: test folds do not cover all 16 regions'
        fold_of[seed] = mapping

    assignment_cache = {}
    records = []
    anchor_log = []
    gpm_fallback_regions = 0

    for loc in scu.ALL_LOCATIONS:
        print(f'\n=== {loc} ===')
        ctx = load_location_context(loc, assignment_cache)

        # -- Static bases --
        uni_base = compute_uniform_base(ctx['grid_gdf'], ctx['region_sub'])
        gpm_base, n_fallback = compute_gpm_base(
            ctx['grid_gdf'], ctx['region_sub'], ctx['subs_sub'])
        gpm_fallback_regions += n_fallback

        loc_rows = []
        loc_rows += analyse_base(ctx, uni_base, 'uniform', 'uniform', '-', '-')
        loc_rows += analyse_base(ctx, gpm_base, 'gpm', 'gpm', '-', '-')
        anchor_static(loc, loc_rows, frozen_all, anchor_log)

        # -- GNN base (per seed, using the fold where this region is the test set) --
        for seed in SEEDS:
            fold_dir_name = fold_of[seed][loc]
            gd_path = (EXP0_DIR / f'seed_{seed}' / 'baseline' / fold_dir_name
                       / 'grid_demands' / f'{loc}_grid_demands.pickle')
            with open(gd_path, 'rb') as f:
                grid_demands = pickle.load(f)
            gnn_base = np.asarray(grid_demands['gnn_demand'], dtype=float)
            assert len(gnn_base) == len(ctx['grid_gdf']), \
                f'{loc} seed{seed}: grid_demands length does not match grid point count'

            gnn_rows = analyse_base(
                ctx, gnn_base, 'gnn', f'gnn_seed{seed}_{fold_dir_name}',
                str(seed), fold_dir_name)
            anchor_gnn(loc, seed, fold_dir_name, gnn_rows[0]['rmse_base'], anchor_log)
            loc_rows += gnn_rows

        records += loc_rows
        by_sig = {r['signal']: r for r in loc_rows if r['base_type'] == 'uniform'}
        print(f"  uniform: pearson_log(NP)={by_sig['NP']['pearson_log']:.3f}, "
              f"dRMSE(NP)={by_sig['NP']['delta_rmse']:.3f} | "
              f"anchors passed so far: {len(anchor_log)}")

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_DIR / 'alignment_per_region.csv', index=False)

    summary = build_summary(df, anchor_log, gpm_fallback_regions)
    with open(OUTPUT_DIR / 'alignment_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print('\n' + '=' * 70)
    print(f'Rows: {len(df)} (expected {summary["row_counts"]["expected"]["total"]}) | '
          f'{len(anchor_log)} anchors all passed | '
          f'eps-protected samples: {summary["eps_protection"]["total_protected_any"]}')
    hl = summary['static_vs_gnn_headline']
    print(f"Static base median alignment (pearson_log)={hl['static']['median_pearson_log']:.4f}, "
          f"median sigma_c/sigma_r={hl['static']['median_sigma_ratio']:.4f}")
    print(f"GNN base median alignment (pearson_log)={hl['gnn']['median_pearson_log']:.4f}, "
          f"median sigma_c/sigma_r={hl['gnn']['median_sigma_ratio']:.4f}")
    print(f'Outputs: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
