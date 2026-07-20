# -*- coding: utf-8 -*-
"""
034 Pre-deployment subsampling validation of the antagonism discriminator.

Conditional statement (same convention as exp_r26): multiplicative correction
plus renormalization reduces error if and only if
    rho > sigma_c / (2 * sigma_r)
where rho = corr(log F, log rho_resid), sigma_r = std(log rho_resid),
sigma_c = std(log F) (substation level, ddof=0, eps = regional total demand
x 1e-6 -- same protocol as 024/014).

This experiment turns that conditional statement into a **practical
deployment diagnostic**: given only k measured substations, how reliably can
a deployer decide -- *before* running the correction -- whether multiplicative
correction will help or hurt? This replaces the blanket recommendation of
"always apply post-hoc correction" with a testable decision rule, and
quantifies how many measured stations that rule needs to be reliable.

=== Discriminator definition (observability tiers from a deployment
    standpoint) ===
- **Observable (no measurement needed)**: d_base and d_corr can be computed
  for every substation (the allocation outputs are the deployer's own
  results), so sigma_c = std(log F) **can always be computed on the full
  set, no subsampling needed**; the same holds for eps (regional total
  demand is an input to the allocation task).
- **Requires measurement**: rho_hat and sigma_r_hat depend on d_true --
  available only for the k sampled stations:
      rho_hat = corr(log F_k, log rho_k), sigma_r_hat = std(log rho_k)  (ddof=0)
- **Discriminator output**: help iff rho_hat > sigma_c / (2 * sigma_r_hat).
- **Ground truth**: the sign of the actual full-set delta-RMSE for that
  combination (help iff delta-RMSE < 0; this value is read from existing
  results, not recomputed here).
- **Degeneracy guard**: if log F or log rho has zero variance on the k
  sampled stations, the correlation is undefined and the discriminator
  conservatively outputs "hurt" ("do not correct without evidence"); these
  cases are counted and recorded (n_degenerate).

=== Combination universe (420 combinations across two cases, all
    reconstructed from existing results) ===
- UK: the 240 combinations from exp_r25 ({Uniform, GPM, GNN x 3 seeds} x 16
  regions x {N, P, NP}). The substation-level triple (d_true, d_base, d_corr)
  is reconstructed from the same inputs using the function pattern in 024,
  and anchored combination-by-combination against the existing
  alignment_per_region.csv (pearson_log, sigma_r, sigma_c, rmse_base,
  rmse_corrected, delta_rmse) with rtol=atol=1e-9 (fatal on mismatch).
- AU: the 180 combinations from 014 (same structure x 12 SA4 regions),
  reconstructed using the function pattern in 011, anchored against the
  existing au_phase_points.csv (same tolerance, fatal on mismatch).
- Median substation count per UK region is about 101 (minimum TLC1=50);
  median for AU is about 12 (minimum 5). For k in {3,5,8,10,15,20}, any k
  greater than or equal to the region's station count is truncated (that k
  tier produces no row for that region); truncations are recorded in the
  k_truncation_registry field of diagnostic_summary.json. A full-set tier
  (k_label='full') is also evaluated once, deterministically.

=== Subsampling protocol ===
For each (combination, k) tier: draw k stations without replacement,
repeated N_REPS=500 times. The random source is an independent
SeedSequence([ROOT_SEED, sha256(combination_key), k]) derived per
(combination, k) pair -- order-independent, and reproducible per
combination (tests re-run and check individual combinations against this).
Accuracy = the fraction of the 500 discriminator outputs that match the
ground truth. The full-set tier has only one possible subset, so it is
evaluated once (this reproduces, at the row level, the same
prediction_correct value computed by the phase-diagram script; an internal
consistency assertion checks this).

=== Measurement-free heuristic (secondary discriminator) ===
A logistic regression discriminator built only from observables that
require **no measured demand**:
  sigma_c (computable on the full set) + base-type one-hot +
  base allocation entropy entropy_norm (normalized entropy of
  p_i = d_i/sum(d) within a region, same formula as concentration_of in
  028; for the UK GNN tier this is read directly from the existing
  concentration_metrics.csv produced by exp_r213, filtered to the baseline
  config/arm and kappa=1, and anchored against the reconstructed value to
  <=1e-9; for the UK static tiers and all AU tiers, no matching row exists
  in that file, so the value is computed directly here and its provenance
  is recorded).
Reports in-sample accuracy (an **upper-bound** figure) and
leave-one-region-out cross-validation accuracy, compared against two
references: (1) a "blanket-rule baseline" (always-apply, i.e. the default
policy of unconditionally applying correction); (2) the ground-truth-based
discriminator (k-tier curves and the full-set tier). This quantifies how far
one can get without any measurement.

=== Outputs (results/exp_deploy_diag/, 6 files + notebook) ===
  subsample_accuracy.csv    per-(combination x k tier) accuracy detail
                            (2460 rows)
  k_accuracy_curves.csv     per-case x base-group k-accuracy aggregate
                            curves (including coverage)
  min_k_table.csv           minimum k needed to reach 80%/90% accuracy
                            (first-crossing rule)
  truthfree_baseline.json   measurement-free logistic discriminator +
                            blanket baseline + two-tier comparison
  combo_vectors.npz         substation-level log rho / log F vectors for
                            all 420 combinations (used by tests for
                            spot-checked re-runs and by the notebook;
                            derived from existing inputs)
  diagnostic_summary.json   headline results + anchoring report +
                            truncation registry + verdict fields generated
                            by the rules below (never hand-written)
  notebooks/r_deploy_diagnostic.ipynb   read-only presentation layer

Runs entirely on CPU (limited to 4 threads, since a GPU tau-sweep may be
running concurrently); the only randomness is the subsampling RNG (protocol
above). All existing input results are treated as read-only.

Usage:
    python 034_exp_deploy_diagnostic.py    (allocategnn env)
"""

import os

# CPU-only discipline: a GPU tau-sweep may be running concurrently, so this
# script stays on CPU and limits itself to 4 threads -- must be set before
# importing numpy.
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
           'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '4')

import hashlib
import importlib.util
import json
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Force UTF-8 console output on Windows (same pattern as 028/014)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
for _extra in (str(SCRIPT_DIR), str(PROJECT_ROOT)):
    if _extra not in sys.path:
        sys.path.insert(0, _extra)

import shared_correction_utils as scu  # noqa: E402

# ════════════════════════════════════════════════════════════
# Paths and constants
# ════════════════════════════════════════════════════════════

AU_DIR = PROJECT_ROOT / 'StudyCase' / 'Australia'
AU_TRAIN_ROOT = AU_DIR / 'data' / 'processed' / 'training'

UK_ALIGN_CSV = SCRIPT_DIR / 'results' / 'exp_r25' / 'alignment_per_region.csv'
AU_POINTS_CSV = (AU_DIR / 'data' / 'processed' / 'evaluation'
                 / 'au_phase_points.csv')
S7_CONC_CSV = SCRIPT_DIR / 'results' / 'exp_r213' / 'concentration_metrics.csv'

EXP0_DIR = SCRIPT_DIR / 'results' / 'exp0_kfold_prior'
OUTPUT_DIR = SCRIPT_DIR / 'results' / 'exp_deploy_diag'

SEEDS = [42, 123, 456]
SIGNALS = ['N', 'P', 'NP']
EPS_RATIO = 1e-6                    # same protocol as 024/014 (used for reconstruction self-check)

K_LIST = [3, 5, 8, 10, 15, 20]      # subsampling tiers (k >= region station count -> truncated)
N_REPS = 500                        # repetitions per (combination, k) tier
ROOT_SEED = 20260715                # root seed for subsampling
TARGETS = [0.8, 0.9]                # target accuracy levels for minimum-k determination

# Anchoring tolerance: the reference CSV is stored at full precision and the
# reconstruction follows the same deterministic pipeline, so the expected
# deviation is ~1e-12
ANCHOR_RTOL = 1e-9
ANCHOR_ATOL = 1e-9
ENTROPY_ANCHOR_TOL = 1e-9           # UK GNN entropy vs. existing reference value

# Verdict thresholds (same rule used for the UK/AU comparisons in 025/014;
# generated programmatically, never hand-written)
VERDICT_TIERS = [(0.9, 'supported'), (0.5, 'partially_supported')]
VERDICT_RULE_TEXT = ('value >= 0.9 -> supported; value >= 0.5 -> '
                     'partially_supported; else not_supported')

# Aggregated base-type groups (static = uniform + gpm combined)
GROUPS = {
    'all': ('uniform', 'gpm', 'gnn'),
    'static': ('uniform', 'gpm'),
    'uniform': ('uniform',),
    'gpm': ('gpm',),
    'gnn': ('gnn',),
}

# Columns compared during anchoring (reconstructed value vs. existing
# reference CSV, per combination)
ANCHOR_COLS = ['pearson_log', 'sigma_r', 'sigma_c',
               'rmse_base', 'rmse_corrected', 'delta_rmse']


def verdict_of(value: float) -> str:
    """Numeric rule -> verdict label (generated, not hand-written; same function as used in 025/014)."""
    for threshold, label in VERDICT_TIERS:
        if value >= threshold:
            return label
    return 'not_supported'


def _load_script_module(name: str, path: Path):
    """Load a script module whose filename starts with a digit via importlib
    (same approach used elsewhere in this repo, e.g. in 014 and test_r25, to
    reuse functions instead of duplicating them)."""
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ════════════════════════════════════════════════════════════
# Discriminator core (pure functions; tests import these directly for
# spot checks)
# ════════════════════════════════════════════════════════════

def stable_combo_seed(key: str) -> int:
    """Combination key -> stable 64-bit seed component (sha256, consistent across runs/platforms)."""
    return int.from_bytes(hashlib.sha256(key.encode('utf-8')).digest()[:8],
                          'little')


def entropy_norm_of(demand: np.ndarray) -> float:
    """Normalized entropy of the demand distribution within a region (same formula as concentration_of in 028, entropy term only)."""
    d = np.asarray(demand, dtype=float)
    n = len(d)
    total = d.sum()
    assert total > 0, 'region total demand is 0 -- entropy is undefined'
    p = d / total
    pos = p > 0
    return float(-(p[pos] * np.log(p[pos])).sum() / np.log(n))


def subsample_curve_for_combo(log_rho: np.ndarray, log_f: np.ndarray,
                              sigma_c_full: float, truth_help: bool,
                              combo_key: str,
                              k_list=tuple(K_LIST), n_reps: int = N_REPS,
                              root_seed: int = ROOT_SEED):
    """Subsampling discriminator curve across all k tiers for one combination.

    Protocol (see module docstring): each (combination, k) tier uses an
    independent RNG = SeedSequence([root_seed, sha256(combo_key), k]),
    order-independent. The full-set tier (k = len(log_rho), k_label='full')
    has only one possible subset, so it is evaluated once. Degenerate
    subsamples (zero variance in log F or log rho on the k sampled stations)
    conservatively output "hurt" and are counted.

    Returns (rows, truncated_ks): rows = one dict per k tier;
    truncated_ks = tiers skipped because k >= the station count.
    """
    log_rho = np.asarray(log_rho, dtype=float)
    log_f = np.asarray(log_f, dtype=float)
    n = len(log_rho)
    assert len(log_f) == n and n >= 3, f'unexpected vector length n={n}'
    truth_help = bool(truth_help)

    ks = [(str(k), int(k)) for k in k_list if k < n] + [('full', n)]
    truncated_ks = [int(k) for k in k_list if k >= n]

    rows = []
    for k_label, k in ks:
        reps = 1 if k == n else n_reps
        rng = np.random.default_rng(np.random.SeedSequence(
            [root_seed, stable_combo_seed(combo_key), k]))
        # Sampling without replacement: for each row, the first k indices of
        # an argsort of random numbers form a uniformly random k-subset
        idx = np.argsort(rng.random((reps, n)), axis=1)[:, :k]

        x = log_f[idx]                       # (reps, k)
        y = log_rho[idx]
        xm = x - x.mean(axis=1, keepdims=True)
        ym = y - y.mean(axis=1, keepdims=True)
        sxx = (xm * xm).sum(axis=1)
        syy = (ym * ym).sum(axis=1)
        sxy = (xm * ym).sum(axis=1)

        degenerate = (sxx <= 0.0) | (syy <= 0.0)
        with np.errstate(invalid='ignore', divide='ignore'):
            rho_hat = sxy / np.sqrt(sxx * syy)
            sigma_r_hat = np.sqrt(syy / k)          # ddof=0 (same convention as 024)
            pred = rho_hat > sigma_c_full / (2.0 * sigma_r_hat)
        pred = np.where(degenerate, False, pred)     # degenerate -> conservative hurt

        ok = ~degenerate
        rows.append({
            'k_label': k_label,
            'k': int(k),
            'n_reps': int(reps),
            'accuracy': float((pred == truth_help).mean()),
            'frac_pred_help': float(pred.mean()),
            'n_degenerate': int(degenerate.sum()),
            'mean_rho_hat': float(rho_hat[ok].mean()) if ok.any() else float('nan'),
            'mean_sigma_r_hat': float(sigma_r_hat[ok].mean()) if ok.any() else float('nan'),
        })
    return rows, truncated_ks


def aggregate_curves(sub_df: pd.DataFrame) -> pd.DataFrame:
    """Per-combination detail -> per-case x base-group k-accuracy aggregate
    curves (including coverage).

    Aggregation rule (the single convention used consistently by min_k
    determination and by the tests): for each (case, group, k tier), take
    the simple mean of accuracy over the combinations that have a row for
    that tier; coverage = number of combinations with a row for that tier /
    total combinations in the group (truncation makes AU coverage < 1 at
    high k tiers, so it must be read alongside the accuracy figure).
    """
    rows = []
    for case in sorted(sub_df['case'].unique()):
        cdf = sub_df[sub_df['case'] == case]
        for gname, bts in GROUPS.items():
            gdf = cdf[cdf['base_type'].isin(bts)]
            n_total = gdf[['base_id', 'location', 'signal']].drop_duplicates().shape[0]
            for k_label, kdf in gdf.groupby('k_label'):
                rows.append({
                    'case': case,
                    'group': gname,
                    'k_label': str(k_label),
                    'k_numeric': (float(k_label) if k_label != 'full'
                                  else float('nan')),
                    'accuracy': float(kdf['accuracy'].mean()),
                    'frac_pred_help': float(kdf['frac_pred_help'].mean()),
                    'n_combos': int(len(kdf)),
                    'n_combos_total': int(n_total),
                    'coverage': float(len(kdf) / n_total),
                })
    out = pd.DataFrame(rows)
    out['_sort_k'] = out['k_numeric'].fillna(np.inf)
    out = (out.sort_values(['case', 'group', '_sort_k'])
              .drop(columns='_sort_k').reset_index(drop=True))
    return out


def min_k_from_curves(curves: pd.DataFrame,
                      targets=tuple(TARGETS)) -> pd.DataFrame:
    """Aggregate curves -> minimum-k table (first-crossing rule).

    Rule: scan tiers in ascending k order ('full' sorted last); the first
    tier with accuracy >= target is min_k. If no tier reaches the target,
    min_k='none' and the curve's maximum accuracy is recorded instead.
    """
    rows = []
    for (case, group), g in curves.groupby(['case', 'group']):
        g = g.copy()
        g['_sort_k'] = g['k_numeric'].fillna(np.inf)
        g = g.sort_values('_sort_k')
        for target in targets:
            hit = g[g['accuracy'] >= target]
            if len(hit):
                first = hit.iloc[0]
                rows.append({
                    'case': case, 'group': group, 'target': float(target),
                    'min_k': first['k_label'],
                    'accuracy_at_min_k': float(first['accuracy']),
                    'coverage_at_min_k': float(first['coverage']),
                    'max_accuracy_on_curve': float(g['accuracy'].max()),
                })
            else:
                rows.append({
                    'case': case, 'group': group, 'target': float(target),
                    'min_k': 'none',
                    'accuracy_at_min_k': float('nan'),
                    'coverage_at_min_k': float('nan'),
                    'max_accuracy_on_curve': float(g['accuracy'].max()),
                })
    return (pd.DataFrame(rows)
            .sort_values(['case', 'group', 'target']).reset_index(drop=True))


# ════════════════════════════════════════════════════════════
# Combination reconstruction (UK uses the function pattern from 024; AU
# uses the pattern from 011/014)
# ════════════════════════════════════════════════════════════

def _anchor_combo(rec: dict, frozen_row: pd.Series, anchor_devs: list,
                  what: str) -> None:
    """Anchor reconstructed statistics against the reference CSV row, column by column (fatal on mismatch)."""
    for col in ANCHOR_COLS:
        got, ref = float(rec[col]), float(frozen_row[col])
        dev = abs(got - ref)
        tol = ANCHOR_ATOL + ANCHOR_RTOL * abs(ref)
        anchor_devs.append(dev)
        if dev > tol:
            raise RuntimeError(
                f'Anchoring failed: {what}/{col} reconstructed {got!r} vs. reference {ref!r} '
                f'(deviation {dev:.3e} > tolerance {tol:.3e})')


def _combo_stats(d_true: np.ndarray, d_base: np.ndarray, d_corr: np.ndarray,
                 eps: float, rmse_base: float, rmse_corr: float) -> dict:
    """Substation-level triple -> vectors and anchoring statistics needed for the discriminator (subset of alignment_stats in 024)."""
    log_rho = np.log((d_true + eps) / (d_base + eps))
    log_f = np.log((d_corr + eps) / (d_base + eps))
    assert np.std(log_f) > 0 and np.std(log_rho) > 0, 'log vector has zero variance'
    pearson = float(np.corrcoef(log_f, log_rho)[0, 1])
    return {
        'log_rho': log_rho,
        'log_f': log_f,
        'pearson_log': pearson,
        'sigma_r': float(np.std(log_rho)),
        'sigma_c': float(np.std(log_f)),
        'rmse_base': float(rmse_base),
        'rmse_corrected': float(rmse_corr),
        'delta_rmse': float(rmse_corr - rmse_base),
    }


def _fold_mapping(train_root: Path, seeds, all_locations, config='baseline') -> dict:
    """Fold assignment per seed: loc -> fold directory name (same structure as 024/014)."""
    fold_of = {}
    for seed in seeds:
        splits_path = train_root / f'seed_{seed}' / config / 'kfold_splits.json'
        splits = json.loads(splits_path.read_text(encoding='utf-8'))
        mapping = {}
        for fold_key, info in splits.items():
            for loc in info['test']:
                assert loc not in mapping, f'seed {seed}: {loc} appears in more than one test fold'
                mapping[loc] = fold_key.replace('_', '')     # fold_1 -> fold1
        assert set(mapping) == set(all_locations), \
            f'seed {seed}: test folds do not cover all regions'
        fold_of[seed] = mapping
    return fold_of


def _analyse_combo_base(case: str, ctx: dict, d_true: np.ndarray, eps: float,
                        base_demand: np.ndarray, base_type: str, base_id: str,
                        seed, fold, frozen_lookup: dict, anchor_devs: list,
                        entropy: dict, actual_col: str) -> list:
    """One base x three signals: reconstruct substation-level vectors, anchor them, and assemble combination records."""
    grid_gdf = ctx['grid_gdf']
    region_sub = ctx['region_sub']
    subs_sub = ctx['subs_sub']
    assignment = ctx['assignment']
    loc = ctx['loc']

    subs_base = scu.aggregate_by_assignment(subs_sub, assignment, base_demand)
    d_base = subs_base['allocated_demand'].values.astype(float)
    rmse_base = scu.evaluate_allocation(subs_base, actual_col=actual_col)['rmse']

    combos = []
    for signal in SIGNALS:
        corrected = scu.apply_standard_multiplicative(
            base_demand, ctx['factors'][signal], grid_gdf, region_sub)
        subs_corr = scu.aggregate_by_assignment(subs_sub, assignment, corrected)
        d_corr = subs_corr['allocated_demand'].values.astype(float)
        rmse_corr = scu.evaluate_allocation(subs_corr, actual_col=actual_col)['rmse']

        rec = _combo_stats(d_true, d_base, d_corr, eps, rmse_base, rmse_corr)
        frozen = frozen_lookup[(base_id, loc, signal)]
        _anchor_combo(rec, frozen, anchor_devs, f'{case}/{base_id}/{loc}/{signal}')

        combos.append({
            'case': case,
            'base_type': base_type,
            'base_id': base_id,
            'seed': str(seed),
            'fold': str(fold),
            'location': loc,
            'signal': signal,
            'n_substations': int(len(d_true)),
            'truth_help': bool(float(frozen['delta_rmse']) < 0),   # read from existing results
            'pearson_full': rec['pearson_log'],
            'sigma_r_full': rec['sigma_r'],
            'sigma_c_full': rec['sigma_c'],
            'delta_rmse_frozen': float(frozen['delta_rmse']),
            'entropy_norm': float(entropy['value']),
            'entropy_provenance': entropy['provenance'],
            'log_rho': rec['log_rho'],
            'log_f': rec['log_f'],
        })
    return combos


def rebuild_uk_combos() -> tuple:
    """Reconstruct the 240 UK combinations (using the function pattern from 024), anchor them against existing results, and compute entropy features."""
    mod024 = _load_script_module(
        'exp_r25_024', SCRIPT_DIR / '024_exp_r25_residual_alignment.py')

    frozen = pd.read_csv(UK_ALIGN_CSV, dtype={'seed': str, 'fold': str},
                         float_precision='round_trip')
    assert len(frozen) == 240, f'UK reference row count {len(frozen)} != 240'
    frozen_lookup = {(r['base_id'], r['location'], r['signal']): r
                     for _, r in frozen.iterrows()}

    conc = pd.read_csv(S7_CONC_CSV)
    conc_base = conc[(conc['config'] == 'baseline') & (conc['arm'] == 'baseline')
                     & (conc['kappa'] == 1.0)]

    fold_of = _fold_mapping(EXP0_DIR, SEEDS, scu.ALL_LOCATIONS)

    assignment_cache = {}
    anchor_devs = []
    entropy_anchor_devs = []
    combos = []
    for loc in scu.ALL_LOCATIONS:
        t0 = time.time()
        ctx = mod024.load_location_context(loc, assignment_cache)
        d_true, eps = ctx['d_true'], ctx['eps']

        # -- Static bases (reusing the same definitions as 024; entropy is
        # computed directly here since there is no matching row in the
        # existing concentration metrics) --
        uni_base = mod024.compute_uniform_base(ctx['grid_gdf'], ctx['region_sub'])
        gpm_base, _ = mod024.compute_gpm_base(
            ctx['grid_gdf'], ctx['region_sub'], ctx['subs_sub'])
        combos += _analyse_combo_base(
            'UK', ctx, d_true, eps, uni_base, 'uniform', 'uniform', '-', '-',
            frozen_lookup, anchor_devs,
            {'value': entropy_norm_of(uni_base), 'provenance': 'computed'},
            actual_col='Demand (MVA)')
        combos += _analyse_combo_base(
            'UK', ctx, d_true, eps, gpm_base, 'gpm', 'gpm', '-', '-',
            frozen_lookup, anchor_devs,
            {'value': entropy_norm_of(gpm_base), 'provenance': 'computed'},
            actual_col='Demand (MVA)')

        # -- GNN base (using existing grid_demands; entropy is read from
        # the existing concentration metrics and anchored against the
        # reconstructed value) --
        for seed in SEEDS:
            fold_dir = fold_of[seed][loc]
            gd_path = (EXP0_DIR / f'seed_{seed}' / 'baseline' / fold_dir
                       / 'grid_demands' / f'{loc}_grid_demands.pickle')
            with open(gd_path, 'rb') as f:
                gnn_base = np.asarray(pickle.load(f)['gnn_demand'], dtype=float)
            assert len(gnn_base) == len(ctx['grid_gdf'])

            row = conc_base[(conc_base['seed'] == seed)
                            & (conc_base['location'] == loc)]
            assert len(row) == 1, f'reference entropy row (seed{seed},{loc}) not unique/{len(row)}'
            assert row.iloc[0]['fold'] == fold_dir, \
                f'reference fold {row.iloc[0]["fold"]} != fold mapping {fold_dir}'
            ent_frozen = float(row.iloc[0]['entropy_norm'])
            dev = abs(ent_frozen - entropy_norm_of(gnn_base))
            entropy_anchor_devs.append(dev)
            if dev > ENTROPY_ANCHOR_TOL:
                raise RuntimeError(
                    f'Reference entropy anchoring failed seed{seed}/{loc}: deviation {dev:.3e}')

            combos += _analyse_combo_base(
                'UK', ctx, d_true, eps, gnn_base, 'gnn',
                f'gnn_seed{seed}_{fold_dir}', seed, fold_dir,
                frozen_lookup, anchor_devs,
                {'value': ent_frozen, 'provenance': 's7_frozen'},
                actual_col='Demand (MVA)')
        print(f'  UK {loc}: {len(d_true)} stations, reconstruction+anchoring '
              f'complete ({time.time() - t0:.1f}s, {len(combos)} combinations so far)')

    assert len(combos) == 240, f'UK reconstructed combination count {len(combos)} != 240'
    return combos, anchor_devs, entropy_anchor_devs


def rebuild_au_combos() -> tuple:
    """Reconstruct the 180 AU combinations (using the function pattern from 011/014), anchor them against existing results, and compute entropy features directly."""
    au011 = _load_script_module('au011', AU_DIR / '011_static_baselines.py')

    frozen = pd.read_csv(AU_POINTS_CSV, dtype={'seed': str, 'fold': str},
                         float_precision='round_trip', encoding='utf-8-sig')
    assert len(frozen) == 180, f'AU reference row count {len(frozen)} != 180'
    frozen_lookup = {(r['base_id'], r['location'], r['signal']): r
                     for _, r in frozen.iterrows()}

    step_table = pd.read_csv(AU_DIR / 'data' / 'processed' / 'grid'
                             / 'grid_step_size_table.csv', encoding='utf-8-sig')
    all_locations = step_table['loc_key'].tolist()
    assert len(all_locations) == 12

    b2 = json.loads((AU_DIR / 'docs' / 'b1b2' / 'b2_features.json')
                    .read_text(encoding='utf-8'))
    regions = au011.load_regions()
    stations = au011.load_stations(regions)
    fold_of = _fold_mapping(AU_TRAIN_ROOT, SEEDS, all_locations)

    assignment_cache = {}
    anchor_devs = []
    combos = []
    for loc in all_locations:
        t0 = time.time()
        ctx = au011.load_location_context(loc, regions, stations, b2,
                                          assignment_cache)
        d_true = ctx['subs_sub']['peak_mw'].values.astype(float)
        eps = float(np.asarray(ctx['region_sub']['Demand (MVA)'],
                               dtype=float).sum()) * EPS_RATIO

        uni_base = au011.compute_uniform_base(ctx['grid_gdf'], ctx['region_sub'])
        gpm_base, _ = au011.compute_gpm_base(
            ctx['grid_gdf'], ctx['region_sub'], ctx['subs_sub'])
        combos += _analyse_combo_base(
            'AU', ctx, d_true, eps, uni_base, 'uniform', 'uniform', '-', '-',
            frozen_lookup, anchor_devs,
            {'value': entropy_norm_of(uni_base), 'provenance': 'computed'},
            actual_col='peak_mw')
        combos += _analyse_combo_base(
            'AU', ctx, d_true, eps, gpm_base, 'gpm', 'gpm', '-', '-',
            frozen_lookup, anchor_devs,
            {'value': entropy_norm_of(gpm_base), 'provenance': 'computed'},
            actual_col='peak_mw')

        for seed in SEEDS:
            fold_dir = fold_of[seed][loc]
            gd_path = (AU_TRAIN_ROOT / f'seed_{seed}' / 'baseline' / fold_dir
                       / 'grid_demands' / f'{loc}_grid_demands.pickle')
            with open(gd_path, 'rb') as f:
                gnn_base = np.asarray(pickle.load(f)['GNN'], dtype=float)
            assert len(gnn_base) == len(ctx['grid_gdf'])

            combos += _analyse_combo_base(
                'AU', ctx, d_true, eps, gnn_base, 'gnn',
                f'gnn_seed{seed}_{fold_dir}', seed, fold_dir,
                frozen_lookup, anchor_devs,
                {'value': entropy_norm_of(gnn_base), 'provenance': 'computed'},
                actual_col='peak_mw')
        print(f'  AU {loc}: {len(d_true)} stations, reconstruction+anchoring '
              f'complete ({time.time() - t0:.1f}s, {len(combos)} combinations so far)')

    assert len(combos) == 180, f'AU reconstructed combination count {len(combos)} != 180'
    return combos, anchor_devs


# ════════════════════════════════════════════════════════════
# Measurement-free heuristic (secondary discriminator)
# ════════════════════════════════════════════════════════════

FEATURE_SETS = {
    'sigma_c_only': ['sigma_c_full'],
    'sigma_c_basetype': ['sigma_c_full', 'bt_uniform', 'bt_gpm', 'bt_gnn'],
    'full_observable': ['sigma_c_full', 'bt_uniform', 'bt_gpm', 'bt_gnn',
                        'entropy_norm'],
}


def _fit_logistic(X: np.ndarray, y: np.ndarray):
    """Standardize + L2 logistic regression (sklearn, deterministic solver)."""
    import warnings

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    pipe = make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=2000, solver='lbfgs'))
    with warnings.catch_warnings():
        # newer scipy removed the lbfgs iprint option -- this is sklearn
        # compatibility-layer noise, unrelated to this experiment
        warnings.filterwarnings('ignore', message='Unknown solver options')
        pipe.fit(X, y)
    return pipe


def eval_truthfree_model(df: pd.DataFrame, features: list) -> dict:
    """Evaluate a measurement-free discriminator for one feature set: in-sample (upper bound) + leave-one-region-out."""
    X = df[features].values.astype(float)
    y = df['truth_help'].astype(int).values
    groups = (df['case'] + '|' + df['location']).values

    pipe = _fit_logistic(X, y)
    in_sample_pred = pipe.predict(X)

    logo_pred = np.empty(len(y), dtype=int)
    for gname in np.unique(groups):
        te = groups == gname
        tr = ~te
        if len(np.unique(y[tr])) < 2:            # single-class training fold -> majority class
            logo_pred[te] = int(y[tr].mean() >= 0.5)
        else:
            logo_pred[te] = _fit_logistic(X[tr], y[tr]).predict(X[te])

    lr = pipe.named_steps['logisticregression']
    per_case = {}
    for case in ('UK', 'AU'):
        m = (df['case'] == case).values
        if not m.any():          # no rows for the other case when fitting within a single case -- skip
            continue
        per_case[case] = {
            'in_sample_accuracy': float((in_sample_pred[m] == y[m]).mean()),
            'logo_cv_accuracy': float((logo_pred[m] == y[m]).mean()),
        }
    return {
        'features': features,
        'n_rows': int(len(y)),
        'in_sample_accuracy': float((in_sample_pred == y).mean()),
        'logo_cv_accuracy': float((logo_pred == y).mean()),
        'per_case': per_case,
        'coefficients': {f: float(c) for f, c in zip(features, lr.coef_[0])},
        'intercept': float(lr.intercept_[0]),
        'note': 'in_sample = upper-bound figure (fit and evaluated on the '
                'same data); logo_cv = leave-one-region-out (groups = '
                'case|location)',
    }


def build_truthfree_report(meta_df: pd.DataFrame, curves: pd.DataFrame) -> dict:
    """Comparison report: measurement-free discriminator vs. blanket-rule baseline vs. ground-truth-based discriminator (two tiers)."""
    df = meta_df.copy()
    for bt in ('uniform', 'gpm', 'gnn'):
        df[f'bt_{bt}'] = (df['base_type'] == bt).astype(float)

    models = {name: eval_truthfree_model(df, feats)
              for name, feats in FEATURE_SETS.items()}
    # Fit separately within each case (a deployer knows which case they are
    # in, but typically has no cross-case training set)
    per_case_models = {}
    for case in ('UK', 'AU'):
        sub = df[df['case'] == case].reset_index(drop=True)
        per_case_models[case] = eval_truthfree_model(
            sub, FEATURE_SETS['full_observable'])

    # Blanket-rule baseline: always-apply (the default policy of
    # unconditionally applying correction) + majority class
    baselines = {}
    for case in ('UK', 'AU', 'pooled'):
        sub = df if case == 'pooled' else df[df['case'] == case]
        help_rate = float(sub['truth_help'].mean())
        baselines[case] = {
            'always_apply_accuracy': help_rate,
            'never_apply_accuracy': float(1 - help_rate),
            'majority_class_accuracy': float(max(help_rate, 1 - help_rate)),
        }

    # Ground-truth-based discriminator reference: k in {3,5,10} and the
    # full-set tier (read from the case x 'all' aggregate curve)
    truth_based = {}
    for case in ('UK', 'AU'):
        cc = curves[(curves['case'] == case) & (curves['group'] == 'all')]
        pts = {}
        for lbl in ('3', '5', '10', 'full'):
            row = cc[cc['k_label'] == lbl]
            if len(row) == 1:
                pts[f'k_{lbl}'] = {
                    'accuracy': float(row.iloc[0]['accuracy']),
                    'coverage': float(row.iloc[0]['coverage']),
                }
        truth_based[case] = pts

    return {
        'design': {
            'label': 'truth_help (sign of the full-set delta-RMSE, read from '
                     'existing results)',
            'observables': 'Quantities available without any measured '
                           'demand: sigma_c (full-set std(log F)), base-type '
                           'one-hot, base allocation entropy entropy_norm',
            'entropy_provenance': 'UK gnn = read from the existing '
                                  'concentration_metrics.csv (kappa=1, '
                                  'baseline arm, anchored against the '
                                  'reconstruction to <=1e-9); UK static + all '
                                  'AU = computed directly from the '
                                  'reconstructed base (same formula as 028)',
            'model': 'StandardScaler + L2 LogisticRegression (sklearn lbfgs)',
        },
        'models_pooled': models,
        'models_per_case': per_case_models,
        'slogan_baselines': baselines,
        'truth_based_reference': truth_based,
    }


# ════════════════════════════════════════════════════════════
# Main pipeline
# ════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    print('=' * 70)
    print('Pre-deployment subsampling validation of the antagonism '
          'discriminator (034) -- UK 240 + AU 180 combinations')
    print('=' * 70)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # -- 1. Reconstruction + anchoring --
    print('\n═══ UK combination reconstruction (function pattern from 024, '
          'anchored combination-by-combination against existing exp_r25 '
          'results) ═══')
    uk_combos, uk_devs, uk_ent_devs = rebuild_uk_combos()
    print('\n═══ AU combination reconstruction (function pattern from 011, '
          'anchored combination-by-combination against existing '
          'au_phase_points results) ═══')
    au_combos, au_devs = rebuild_au_combos()
    combos = uk_combos + au_combos
    print(f'\nAll anchors passed: UK max deviation {max(uk_devs):.3e} | '
          f'AU max deviation {max(au_devs):.3e} | '
          f'reference entropy max deviation {max(uk_ent_devs):.3e}')

    # -- 2. Subsampling discrimination --
    print('\n═══ Subsampling discrimination (each combination x k tier x '
          '500 repetitions) ═══')
    sub_rows = []
    trunc_registry = {}
    n_knife_edge = 0
    for i, c in enumerate(combos):
        combo_key = f"{c['case']}|{c['base_id']}|{c['location']}|{c['signal']}"
        rows, truncated = subsample_curve_for_combo(
            c['log_rho'], c['log_f'], c['sigma_c_full'], c['truth_help'],
            combo_key)
        reg_key = f"{c['case']}|{c['location']}"
        trunc_registry.setdefault(reg_key, {
            'n_substations': c['n_substations'],
            'truncated_ks': truncated,
            'available_ks': [k for k in K_LIST if k < c['n_substations']],
        })

        # Internal consistency check: the single full-set-tier decision
        # must match the phase-diagram rule at the row level
        full_row = rows[-1]
        assert full_row['k_label'] == 'full'
        rule_pred = c['pearson_full'] > c['sigma_c_full'] / (2 * c['sigma_r_full'])
        sub_pred = full_row['frac_pred_help'] >= 0.5      # single evaluation -> 0 or 1
        if bool(sub_pred) != bool(rule_pred):
            gap = abs(c['pearson_full']
                      - c['sigma_c_full'] / (2 * c['sigma_r_full']))
            assert gap < 1e-12, \
                f'{combo_key}: full-set-tier decision disagrees with the phase-diagram rule and is not a near-boundary case (gap={gap:.3e})'
            n_knife_edge += 1

        base_cols = {k: c[k] for k in
                     ('case', 'base_type', 'base_id', 'seed', 'fold',
                      'location', 'signal', 'n_substations', 'truth_help')}
        for r in rows:
            sub_rows.append({**base_cols, **r})
        if (i + 1) % 60 == 0:
            print(f'  {i + 1}/{len(combos)} combinations complete '
                  f'({time.time() - t_start:.0f}s)')

    sub_df = pd.DataFrame(sub_rows)
    sub_df.to_csv(OUTPUT_DIR / 'subsample_accuracy.csv', index=False)
    print(f'subsample_accuracy.csv: {len(sub_df)} rows')

    # -- 3. Aggregate curves + minimum-k table --
    curves = aggregate_curves(sub_df)
    curves.to_csv(OUTPUT_DIR / 'k_accuracy_curves.csv', index=False)
    min_k = min_k_from_curves(curves)
    min_k.to_csv(OUTPUT_DIR / 'min_k_table.csv', index=False)
    print(f'k_accuracy_curves.csv: {len(curves)} rows | '
          f'min_k_table.csv: {len(min_k)} rows')

    # -- 4. Save combination vectors (used by tests for spot-checked
    # re-runs and by the notebook) --
    meta_cols = ['case', 'base_type', 'base_id', 'seed', 'fold', 'location',
                 'signal', 'n_substations', 'truth_help', 'pearson_full',
                 'sigma_r_full', 'sigma_c_full', 'delta_rmse_frozen',
                 'entropy_norm', 'entropy_provenance']
    meta_df = pd.DataFrame([{k: c[k] for k in meta_cols} for c in combos])
    ptr = np.cumsum([0] + [c['n_substations'] for c in combos])
    np.savez_compressed(
        OUTPUT_DIR / 'combo_vectors.npz',
        ptr=ptr,
        log_rho_concat=np.concatenate([c['log_rho'] for c in combos]),
        log_f_concat=np.concatenate([c['log_f'] for c in combos]),
        **{f'meta_{k}': meta_df[k].values.astype(str) if meta_df[k].dtype == object
           else meta_df[k].values for k in meta_cols},
    )

    # -- 5. Measurement-free heuristic --
    print('\n═══ Measurement-free heuristic (secondary discriminator) ═══')
    truthfree = build_truthfree_report(meta_df, curves)
    with open(OUTPUT_DIR / 'truthfree_baseline.json', 'w', encoding='utf-8') as f:
        json.dump(truthfree, f, ensure_ascii=False, indent=2)

    # -- 6. Summary + rule-generated verdicts --
    def _curve_point(case, group, k_label):
        row = curves[(curves['case'] == case) & (curves['group'] == group)
                     & (curves['k_label'] == k_label)]
        return None if len(row) != 1 else {
            'accuracy': float(row.iloc[0]['accuracy']),
            'coverage': float(row.iloc[0]['coverage']),
            'n_combos': int(row.iloc[0]['n_combos']),
        }

    def _min_k_cell(case, group, target):
        row = min_k[(min_k['case'] == case) & (min_k['group'] == group)
                    & (min_k['target'] == target)]
        assert len(row) == 1
        r = row.iloc[0]
        return {'min_k': r['min_k'],
                'accuracy_at_min_k': (None if pd.isna(r['accuracy_at_min_k'])
                                      else float(r['accuracy_at_min_k'])),
                'coverage_at_min_k': (None if pd.isna(r['coverage_at_min_k'])
                                      else float(r['coverage_at_min_k'])),
                'max_accuracy_on_curve': float(r['max_accuracy_on_curve'])}

    headline_min_k = {
        case: {group: {str(t): _min_k_cell(case, group, t) for t in TARGETS}
               for group in GROUPS}
        for case in ('UK', 'AU')
    }

    # Verdict fields (generated from the numeric rule, never hand-written)
    verdicts = {}
    for case in ('UK', 'AU'):
        pt = _curve_point(case, 'all', '10')
        value = pt['accuracy'] if pt else 0.0
        verdicts[f'{case.lower()}_deployable_at_k10'] = {
            'description': f'{case}: discriminator accuracy using 10 '
                           f'measured stations (case x all aggregate curve, '
                           f'k=10 tier; see the coverage field)',
            'value': value,
            'coverage': pt['coverage'] if pt else None,
            'value_rule': 'value = k_accuracy_curves[case, all, k=10].accuracy',
            'verdict': verdict_of(value),
        }
    tf_value = truthfree['models_pooled']['full_observable']['logo_cv_accuracy']
    verdicts['truthfree_sufficient'] = {
        'description': 'Whether the measurement-free heuristic (sigma_c + '
                       'base type + entropy, pooled LOGO-CV) can substitute '
                       'for the measurement-based discriminator',
        'value': tf_value,
        'value_rule': 'value = models_pooled.full_observable.logo_cv_accuracy',
        'verdict': verdict_of(tf_value),
    }

    summary = {
        'meta': {
            'script': '034_exp_deploy_diagnostic.py',
            'purpose': 'Pre-deployment subsampling validation of the '
                       'antagonism discriminator (turning the conditional '
                       'statement into a practical diagnostic)',
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'theory': 'help iff rho > sigma_c/(2*sigma_r) (same convention as exp_r26)',
            'discriminator': 'help iff rho_hat(k stations) > sigma_c(full set)/'
                             '(2*sigma_r_hat(k stations)); degenerate '
                             'subsamples -> conservative hurt',
            'truth_rule': 'truth_help iff existing delta_RMSE < 0 (full station set)',
            'protocol': {
                'k_list': K_LIST,
                'n_reps': N_REPS,
                'root_seed': ROOT_SEED,
                'rng': 'SeedSequence([root_seed, sha256(case|base_id|loc|signal)'
                       '[:8], k]) -- independent per (combination, k), '
                       'order-independent',
                'sigma_c_note': "sigma_c is always computable on the full "
                                "set, no subsampling needed: d_base/d_corr "
                                "are the deployer's own allocation outputs "
                                "and do not depend on measured demand",
            },
            'inputs': [
                'results/exp_r25/alignment_per_region.csv (UK anchoring + '
                'ground truth, read-only)',
                'StudyCase/Australia/data/processed/evaluation/'
                'au_phase_points.csv (AU anchoring + ground truth, read-only)',
                'results/exp0_kfold_prior/**/grid_demands (UK GNN base, read-only)',
                'StudyCase/Australia/data/processed/training/**/grid_demands'
                ' (AU GNN base, read-only)',
                'results/exp_r213/concentration_metrics.csv (UK GNN entropy, '
                'read-only)',
            ],
            'notes': [
                'Verdict fields (value/verdict) are always generated from '
                'the numeric rule, never hand-written.',
                'The full-set tier has only one possible subset, so it is '
                'evaluated once; an internal assertion checks its '
                'agreement with the phase-diagram rule (near-boundary '
                'tolerance 1e-12).',
                'coverage < 1 on the aggregate curve means that k tier was '
                'truncated for some regions (mainly at high k for AU); '
                'read the accuracy figure together with coverage.',
                'The aggregate accuracy at k=full equals the share of '
                'prediction_correct in the phase diagram (the '
                "discriminator's information ceiling = the accuracy of the "
                'theoretical line itself).',
            ],
        },
        'row_counts': {
            'n_combos_uk': len(uk_combos),
            'n_combos_au': len(au_combos),
            'n_subsample_rows': int(len(sub_df)),
            'n_knife_edge_full_rows': n_knife_edge,
        },
        'anchor_check': {
            'uk': {'n_values': len(uk_devs), 'max_abs_dev': float(max(uk_devs)),
                   'rule': f'reconstruction vs. existing exp_r25 results '
                           f'({ANCHOR_COLS}), rtol={ANCHOR_RTOL}, atol={ANCHOR_ATOL}'},
            'au': {'n_values': len(au_devs), 'max_abs_dev': float(max(au_devs)),
                   'rule': f'reconstruction vs. existing au_phase_points '
                           f'results ({ANCHOR_COLS}), rtol={ANCHOR_RTOL}, '
                           f'atol={ANCHOR_ATOL}'},
            'uk_gnn_entropy': {'n_values': len(uk_ent_devs),
                               'max_abs_dev': float(max(uk_ent_devs)),
                               'tol': ENTROPY_ANCHOR_TOL},
            'all_passed': True,        # any failure raises immediately, so
                                        # reaching this point means all checks passed
        },
        'k_truncation_registry': trunc_registry,
        'headline_min_k': headline_min_k,
        'curve_points_k10': {case: _curve_point(case, 'all', '10')
                             for case in ('UK', 'AU')},
        'curve_points_full': {case: _curve_point(case, 'all', 'full')
                              for case in ('UK', 'AU')},
        'truthfree_headline': {
            'pooled_full_observable': truthfree['models_pooled']['full_observable'],
            'slogan_always_apply': {
                case: truthfree['slogan_baselines'][case]['always_apply_accuracy']
                for case in ('UK', 'AU', 'pooled')},
        },
        'verdicts': verdicts,
        'verdict_rule': VERDICT_RULE_TEXT,
    }
    with open(OUTPUT_DIR / 'diagnostic_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # -- 7. Console headline summary --
    print('\n' + '═' * 70)
    print('Headline results (full detail in diagnostic_summary.json)')
    print('═' * 70)
    for case in ('UK', 'AU'):
        cells = headline_min_k[case]['all']
        print(f"{case} - all: min_k(80%)={cells['0.8']['min_k']}, "
              f"min_k(90%)={cells['0.9']['min_k']} "
              f"(curve max {cells['0.9']['max_accuracy_on_curve']:.3f})")
        for group in ('static', 'gnn'):
            g = headline_min_k[case][group]
            print(f"  {group}: min_k(80%)={g['0.8']['min_k']}, "
                  f"min_k(90%)={g['0.9']['min_k']} "
                  f"(max {g['0.9']['max_accuracy_on_curve']:.3f})")
    for name, v in verdicts.items():
        print(f"{name}: value={v['value']:.4f} -> {v['verdict']}")
    tf = truthfree['models_pooled']
    print(f"Measurement-free: sigma_c only LOGO={tf['sigma_c_only']['logo_cv_accuracy']:.3f} | "
          f"+basetype={tf['sigma_c_basetype']['logo_cv_accuracy']:.3f} | "
          f"+entropy={tf['full_observable']['logo_cv_accuracy']:.3f} "
          f"(in-sample upper bound {tf['full_observable']['in_sample_accuracy']:.3f})")
    print(f"Blanket-rule baseline always-apply: UK "
          f"{truthfree['slogan_baselines']['UK']['always_apply_accuracy']:.3f} / "
          f"AU {truthfree['slogan_baselines']['AU']['always_apply_accuracy']:.3f}")
    print(f'\nOutput directory: {OUTPUT_DIR}')
    print(f'Completed in {time.time() - t_start:.1f}s')


if __name__ == '__main__':
    main()
