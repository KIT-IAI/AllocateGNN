# -*- coding: utf-8 -*-
"""
028 - Tau and concentration analysis

At convergence, the trained allocation temperature is tau ~= 0.01, which
drives the per-source softmax weights close to one-hot (highly
concentrated). This raises the question of whether that concentration
acts as a moderating variable for the post-correction antagonism (postNP
performing worse than the GNN baseline), and whether the antagonism
changes as tau is varied.

=== Constructing inference-time evidence (power rescaling is equivalent
to changing tau at inference time) ===
GNN weights come from softmax(z/tau0) (tau0 = allocation_temperature_start
= 0.01, as defined in 005). Applying power rescaling to the already
normalized weights w
    w_kappa = w^kappa / sum(w^kappa)    (per-source, kappa > 0)
is exactly equivalent to changing the inference-time temperature to
tau_eff = tau0/kappa: normalizing softmax(z/tau0)^kappa gives
softmax(z/(tau0/kappa)) (the power-temperature duality of softmax, which
is invariant to constant shifts of the logits). kappa in {0.25, 0.5, 1, 2,
4} corresponds to tau_eff in {0.04, 0.02, 0.01, 0.005, 0.0025} -- kappa<1
flattens the distribution (raises the effective temperature), kappa>1
sharpens it (lowers the temperature), and kappa=1 is the identity (the
anchor setting).

=== Underflow diagnostic (run first, decides the execution branch) ===
Near one-hot at tau~=0.01, tail weights may underflow to exactly 0 under
float32 arithmetic; power rescaling with kappa<1 cannot raise weights
that are already stored as 0, which would systematically bias the
results toward "staying concentrated" and contaminate the core direction
of this analysis. A diagnostic therefore runs first:
  1. For **all 48 combinations** (3 seeds x 4 configs x 4 folds) x 16
     study regions, back out w_sa = gnn_demand / D_r from grid_demands
     (a star graph in which each agent belongs to exactly one ITL3
     source, using the same approach as 026), and compute, for each
     source, the fraction of weights that are exactly zero;
  2. Upper bound on the mass truncated at kappa=0.25: for a weight stored
     as 0, its true value is < the smallest positive float32 subnormal
     B = 1.4e-45 (any true value >= B/2 would round to a nonzero value,
     so B is a conservative upper bound). Assuming every such true value
     is exactly B, the mass those weights would carry at kappa=0.25 is
         U_r = n_zero * B^kappa / (sum_{w>0} w^kappa + n_zero * B^kappa)
         (per source)
     which is the upper bound on the mass truncated by power rescaling.
  3. Branching rule: if U_r < 1% for every (combination, source) ->
     branch = power_rescale (the main path implemented in this file);
     otherwise branch = model_forward (load the model as in 018, run a
     CPU float64 forward pass, and re-run inference directly at the new
     tau -- see the docstring of run_model_forward_branch). The chosen
     branch is written to underflow_diagnostic.json.
  Note on expected magnitude: B^0.25 ~= 6.1e-12, while
  sum_{w>0} w^0.25 >= (max w)^0.25 >= (1/n)^0.25 ~= 0.06 (n <= 5.3e4), so
  breaching U_r > 1% would require n_zero >= 1e8 zero weights, three
  orders of magnitude beyond the total size of a source -- barring
  pathological data such as negative weights, power_rescale is almost
  certain to be selected. The diagnostic is still run in full and its
  output is written to disk as supporting evidence.

=== Recomputation performed at each kappa setting ===
For each (seed, config), following the **test-region** convention
defined by its kfold_splits.json (consistent with 017/024: each region
serves as the test region exactly once):
  - Base demand: base_kappa = M_r * w^kappa / sum(w^kappa) (per ITL3;
    M_r = sum(gnn_demand), i.e. the measured region total rather than
    the nominal D_r -- the two differ by a relative ~1e-7, which is the
    float32 softmax storage error and has no scientific significance,
    but using M_r makes the kappa=1 setting match the stored demand
    bit-for-bit to ~1e-15, satisfying the strict anchor requirement
    that "recomputed values from the same grid_demands must agree to a
    relative tolerance < 1e-9". Note: this is not anchored against the
    frozen fold CSVs -- prior measurements found a maximum GPU-rerun
    drift of 8.7e-5 (see the TOL_ANCHOR_GNN comment in 024), so the
    frozen CSVs are used only for an informational cross-check);
  - The four post-correction arms (shared module, using the frozen
    numerical path from 017):
      postN  = apply_standard_multiplicative(base_kappa, ntl_factor)
      postP  = apply_standard_multiplicative(base_kappa, prox_factor)
      postNP = apply_standard_multiplicative(base_kappa, ntl_factor*prox_factor)
      addNP  = apply_additive_correction(base_kappa, ntl_factor*prox_factor)
  - Per-region RMSE/MAE/corr: the Voronoi assignment is computed once
    per region and reused across all combinations and kappa settings
    (the shared module's two-stage design avoids ~20,000 redundant
    sjoin_nearest calls).

=== Concentration metrics (per combination x region x kappa x arm; the
uncorrected case is the baseline arm, the corrected cases are the
correction arms) ===
For the within-region demand distribution p_i = d_i / sum(d):
  - Normalized entropy H = -sum(p * log p) / log n in [0,1] (smaller
    means more concentrated; H for the baseline arm decreases
    non-strictly monotonically with kappa -- a classical
    escort-distribution result, dH/dkappa = -kappa * Var_{p_kappa}(log w)
    <= 0; at the region level this is a mixture over sources with fixed
    mixture weights M_r, so monotonicity is preserved. Note: the
    correction arms are not guaranteed to be monotonic -- once log f is
    added to the logits, the entropy peak may no longer sit at kappa=1,
    so the monotonicity claim is restricted to the baseline arm);
  - Gini coefficient (0 = uniform, -> 1 = extremely concentrated);
  - top-1% / 5% / 10% mass (the share of total regional demand held by
    the top q% of agents by demand).

=== Antagonism-vs-concentration regression ===
For each kappa setting: y = per-region antagonism magnitude (postNP RMSE
minus baseline RMSE, averaged over seeds), x = baseline concentration
(gini / 1-H / top-10% share of the baseline arm, averaged over seeds).
OLS (scipy linregress) plus Spearman correlation, computed separately
per config (n=16) and pooled across the four configs (n=64). All results
are written to regression.json.

=== Anchoring ===
  1. kappa=1 strict anchor (fatal): for all 192 (combination, test
     region) pairs x 5 arms, the kappa=1 RMSE must agree with a direct
     recomputation from the same grid_demands (without power rescaling)
     to a relative deviation < 1e-9;
  2. Table 3 weak anchor (fatal): the kappa=1 setting, aggregated using
     the paper's convention (average over seeds per (config, region)
     first, then take the mean across the 16 regions), must agree with
     the paper's Table 3 (tab:main) numbers to within +/-0.005: GNN 9.27
     / GNNpostN 9.45 / GNNpostP 9.42 / GNNpostNP 11.20 / GNNpriorN 9.19
     / GNNpriorP 9.14 / GNNpriorNP 9.03;
  3. Frozen fold CSV cross-check (informational, not fatal): the
     kappa=1 direct computation is compared against the
     voronoi_{,ntl_,prox_,ntl_prox_}GNN rows of fold/rmse.csv, and the
     maximum deviation is recorded (expected to be <= ~1e-4, the scale
     of GPU-rerun drift; deviations above 5e-4 are flagged with
     status=warn and written to disk without aborting the run).

=== Output artifacts (results/exp_r213/, 4 files) ===
  underflow_diagnostic.json   diagnostic results + branch decision
  tau_sweep.csv               (seed, config, fold, location, kappa,
                               tau_effective, arm) -> rmse/mae/corr,
                               3x4x16x5x5 = 4800 rows
  concentration_metrics.csv   same keys -> n_agents/entropy_norm/gini/
                               top{1,5,10}_share
  regression.json             meta + anchor report + kappa-curve
                               aggregation + antagonism-vs-concentration
                               regression + rule-generated headline
                               conclusions (never hand-written)

CPU-only (numpy/scipy, no torch; the GPU is concurrently busy training a
separate fusion-arm model, so thread counts are capped at 4),
deterministic post-processing, no randomness (the numpy seed is still
fixed and logged for consistency). All pre-existing output artifacts are
treated as read-only inputs.

Usage:
    python 028_exp_r213_tau_concentration.py
"""

import os
# CPU policy: the GPU is busy overnight training a separate fusion-arm
# model, so this script stays entirely on CPU with a capped thread
# count; these must be set before numpy is imported (same technique
# used elsewhere in this suite for controlling the runtime environment).
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')

import json
import pickle
import platform
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress, spearmanr

# The default Windows console codepage (cp1252) cannot reliably encode
# non-ASCII output -- force UTF-8 uniformly (does not affect file artifacts)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import shared_correction_utils as scu  # noqa: E402

# ════════════════════════════════════════════════════════════
# Constants
# ════════════════════════════════════════════════════════════

EXP0_DIR = SCRIPT_DIR / 'results' / 'exp0_kfold_prior'
OUTPUT_DIR = SCRIPT_DIR / 'results' / 'exp_r213'

SEEDS = [42, 123, 456]
CONFIGS = ['baseline', 'ntl', 'proximity', 'ntl_prox']
N_FOLDS = 4
ALL_LOCATIONS = scu.ALL_LOCATIONS               # 16 study regions (consistent with 017)
BASE_COL = 'gnn_demand'                         # base = uncorrected GNN demand

KAPPAS = [0.25, 0.5, 1.0, 2.0, 4.0]             # power-rescaling settings
TAU_TRAIN = 0.01                                # allocation_temperature_start, as defined in 005
ARMS = ['baseline', 'postN', 'postP', 'postNP', 'addNP']
CORRECTION_ARMS = ['postN', 'postP', 'postNP', 'addNP']

# ── Underflow diagnostic ──
KAPPA_DIAG = 0.25                               # setting used to evaluate the truncated-mass upper bound (the most flattened setting is the worst case)
# Conservative upper bound on the true value of a float32 weight stored
# as 0 = the smallest positive subnormal (~= 1.401e-45)
ZERO_WEIGHT_TRUE_VALUE_BOUND = float(np.finfo(np.float32).smallest_subnormal)
TRUNCATION_MASS_THRESHOLD = 0.01                # > 1% of regional mass -> switch to the model_forward branch

# ── Anchor tolerances ──
TOL_KAPPA1_SELF_REL = 1e-9                      # strict anchor: kappa=1 vs. same-source direct RMSE, relative
TOL_TABLE3 = 0.005 + 1e-9                       # weak anchor against paper Table 3 (2 decimal places)
TOL_FROZEN_CSV_INFO = 5e-4                      # informational cross-check against the frozen fold CSVs
                                                # (same convention as TOL_ANCHOR_GNN in 024:
                                                #  4 d.p. rounding + measured GPU-rerun drift,
                                                #  max 8.7e-5, plus one extra order of magnitude of margin)

# RMSE figures from the paper's Table 3 (tab:main) -- weak anchor +/-0.005.
# Keys = (config, arm). addNP has no corresponding row in Table 3 (it
# appears instead in the mechanism-isolation table produced by 017); the
# per-config correction arms are likewise only reported in Table 3 for
# the baseline config.
TABLE3_RMSE = {
    ('baseline', 'baseline'): 9.27,    # GNN
    ('baseline', 'postN'):    9.45,    # GNNpostN
    ('baseline', 'postP'):    9.42,    # GNNpostP
    ('baseline', 'postNP'):   11.20,   # GNNpostNP
    ('ntl', 'baseline'):      9.19,    # GNNpriorN
    ('proximity', 'baseline'): 9.14,   # GNNpriorP
    ('ntl_prox', 'baseline'): 9.03,    # GNNpriorNP
}

# Frozen fold CSV row names mapped to this experiment's arm names
# (addNP has no frozen counterpart row)
FROZEN_CSV_ROW = {
    'baseline': 'voronoi_GNN',
    'postN': 'voronoi_ntl_GNN',
    'postP': 'voronoi_prox_GNN',
    'postNP': 'voronoi_ntl_prox_GNN',
}

TOP_SHARES = [0.01, 0.05, 0.10]                 # top-1/5/10% mass
REG_PREDICTORS = ['gini', 'one_minus_entropy', 'top10_share']

RNG_SEED = 20260714       # this experiment has no randomness; the seed is still fixed and logged for consistency


def _status(ok: bool) -> str:
    """Status fields are generated from numeric rules, never hand-written."""
    return 'ok' if ok else 'fail'


# ════════════════════════════════════════════════════════════
# Region context (computed once per region, reused across all
# combinations and kappa settings)
# ════════════════════════════════════════════════════════════

_LOCATION_CTX: dict = {}
_VORONOI_CACHE: dict = {}


def get_location_context(loc: str) -> dict:
    """Load and cache the static context for one region (grid, factors,
    ITL3 groups, Voronoi assignment).

    The factors and assignment depend only on the region itself
    (independent of seed/config/fold), so each is computed only once.
    """
    if loc in _LOCATION_CTX:
        return _LOCATION_CTX[loc]

    grid_gdf, region_sub, subs_sub, ntl_values = scu.load_grid_and_subs(loc)
    prox_scores = scu.compute_prox_scores(grid_gdf, subs_sub)
    ntl_factor, prox_factor = scu.compute_factors(grid_gdf, ntl_values, prox_scores)

    region_info = region_sub.set_index('ITL3')
    # RangeIndex semantics: group.index is used directly as a numpy positional index
    itl3_groups = [(itl3, np.asarray(group.index))
                   for itl3, group in grid_gdf.groupby('ITL3')
                   if itl3 in region_info.index]
    covered = sum(len(idx) for _, idx in itl3_groups)
    assert covered == len(grid_gdf), (
        f'{loc}: incomplete ITL3 coverage ({covered}/{len(grid_gdf)}) -- '
        f'an earlier validation step should already have excluded this case')
    region_demand = {itl3: float(region_info.loc[itl3, 'Demand (MVA)'])
                     for itl3, _ in itl3_groups}

    assignment = scu.compute_voronoi_assignment(
        grid_gdf, subs_sub, cache=_VORONOI_CACHE, cache_key=loc)

    ctx = {
        'grid_gdf': grid_gdf,
        'region_sub': region_sub,
        'subs_sub': subs_sub,
        'itl3_groups': itl3_groups,
        'region_demand': region_demand,
        'assignment': assignment,
        'factors': {
            'N': ntl_factor,
            'P': prox_factor,
            'NP': ntl_factor * prox_factor,
        },
    }
    _LOCATION_CTX[loc] = ctx
    return ctx


def load_gnn_demand(seed: int, config: str, fold_dir_name: str, loc: str) -> np.ndarray:
    """Read the gnn_demand column from the frozen grid_demands file (read-only)."""
    p = (EXP0_DIR / f'seed_{seed}' / config / fold_dir_name
         / 'grid_demands' / f'{loc}_grid_demands.pickle')
    with open(p, 'rb') as f:
        grid_demands = pickle.load(f)
    return np.asarray(grid_demands[BASE_COL], dtype=float)


def get_fold_splits(seed: int, config: str) -> dict:
    """Read the (seed, config)-specific kfold_splits.json and return
    {fold_dir_name: [test_locs]}.

    JSON keys are fold_1, ... while directory names are fold1, ...
    (same renaming convention as 017).
    """
    splits_path = EXP0_DIR / f'seed_{seed}' / config / 'kfold_splits.json'
    with open(splits_path, encoding='utf-8') as f:
        splits = json.load(f)
    out = {}
    for fold_key, fold_info in splits.items():
        test_locs = fold_info['test'] if isinstance(fold_info, dict) else fold_info
        out[fold_key.replace('_', '')] = list(test_locs)
    return out


# ════════════════════════════════════════════════════════════
# Stage 1 - Underflow diagnostic
# ════════════════════════════════════════════════════════════

def run_underflow_diagnostic() -> dict:
    """Scan grid_demands for all 48 combinations x 16 regions, back out
    w_sa, and compute underflow statistics.

    Returns the diagnostic dict (including the branch decision); main()
    writes it to underflow_diagnostic.json.
    """
    print('═' * 68)
    print('Stage 1 - Underflow diagnostic (48 combinations x 16 regions, back out w_sa)')
    print('═' * 68)

    B = ZERO_WEIGHT_TRUE_VALUE_BOUND
    B_pow = B ** KAPPA_DIAG

    per_combo = {}
    overall_zero = 0
    overall_agents = 0
    overall_max_source_zero_frac = 0.0
    overall_max_source_trunc = 0.0
    overall_max_loc_trunc = 0.0
    wsum_ratio_max = 0.0          # |sum(w)-1|/(n*eps32), same convention used as a sanity check in 026
    negative_weight_count = 0     # negative weights indicate pathological data -- fail fast if found
    eps32 = float(np.finfo(np.float32).eps)

    n_files = 0
    for seed in SEEDS:
        for config in CONFIGS:
            for fold_i in range(1, N_FOLDS + 1):
                fold_dir_name = f'fold{fold_i}'
                combo_key = f'seed_{seed}/{config}/{fold_dir_name}'
                combo_zero = 0
                combo_agents = 0
                combo_max_source_zero_frac = 0.0
                combo_max_source_trunc = 0.0
                combo_max_loc_trunc = 0.0

                for loc in ALL_LOCATIONS:
                    ctx = get_location_context(loc)
                    g = load_gnn_demand(seed, config, fold_dir_name, loc)
                    n_files += 1
                    if (g < 0).any():
                        negative_weight_count += int((g < 0).sum())

                    loc_trunc_num = 0.0     # sum_r D_r * U_r
                    loc_trunc_den = 0.0     # sum_r D_r
                    for itl3, idx in ctx['itl3_groups']:
                        D_r = ctx['region_demand'][itl3]
                        w = g[idx] / D_r
                        n = len(idx)
                        n_zero = int((w == 0.0).sum())
                        combo_zero += n_zero
                        combo_agents += n
                        zero_frac = n_zero / n
                        combo_max_source_zero_frac = max(combo_max_source_zero_frac,
                                                         zero_frac)
                        # Sanity check that sum(w) ~= 1 within each source (same convention as 026)
                        dev = float(abs(w.sum() - 1.0))
                        wsum_ratio_max = max(wsum_ratio_max, dev / (n * eps32))
                        # Upper bound on truncated mass at kappa=0.25
                        pos = w > 0
                        s_pos = float((w[pos] ** KAPPA_DIAG).sum())
                        t = n_zero * B_pow
                        u_r = t / (s_pos + t) if (s_pos + t) > 0 else 1.0
                        combo_max_source_trunc = max(combo_max_source_trunc, u_r)
                        loc_trunc_num += D_r * u_r
                        loc_trunc_den += D_r
                    combo_max_loc_trunc = max(combo_max_loc_trunc,
                                              loc_trunc_num / loc_trunc_den)

                per_combo[combo_key] = {
                    'n_agents_total': combo_agents,
                    'n_zero_weights': combo_zero,
                    'zero_weight_frac': combo_zero / combo_agents,
                    'max_source_zero_frac': combo_max_source_zero_frac,
                    'max_source_truncated_mass_ub': combo_max_source_trunc,
                    'max_location_truncated_mass_ub': combo_max_loc_trunc,
                }
                overall_zero += combo_zero
                overall_agents += combo_agents
                overall_max_source_zero_frac = max(overall_max_source_zero_frac,
                                                   combo_max_source_zero_frac)
                overall_max_source_trunc = max(overall_max_source_trunc,
                                               combo_max_source_trunc)
                overall_max_loc_trunc = max(overall_max_loc_trunc, combo_max_loc_trunc)
            print(f'  seed_{seed}/{config}: cumulative zero weights = {overall_zero}')

    if negative_weight_count > 0:
        raise RuntimeError(
            f'Found {negative_weight_count} negative gnn_demand values -- the '
            f'weight back-out precondition is violated, the frozen artifacts '
            f'appear corrupted; stopping and reporting the issue.')

    all_below = overall_max_source_trunc < TRUNCATION_MASS_THRESHOLD
    branch = 'power_rescale' if all_below else 'model_forward'

    diagnostic = {
        'meta': {
            'script': '028_exp_r213_tau_concentration.py',
            'plan_step': 'Stage 1: underflow diagnostic for the tau/concentration analysis',
            'n_combos': len(per_combo),
            'n_files_scanned': n_files,
            'kappa_diag': KAPPA_DIAG,
            'zero_weight_true_value_bound': B,
            'bound_note': ('The true value of a float32 weight stored as 0 is < the '
                           'smallest positive subnormal, 1.401e-45 (any true value '
                           '>= B/2 would round to nonzero); the upper-bound formula '
                           'U_r = n_zero * B^kappa / (sum_{w>0} w^kappa + n_zero * B^kappa) '
                           'assumes every zero weight has the maximal true value = B, '
                           'making it a conservative upper bound.'),
            'threshold': TRUNCATION_MASS_THRESHOLD,
            'timestamp': datetime.now().isoformat(timespec='seconds'),
        },
        'per_combo': per_combo,
        'overall': {
            'total_agents_scanned': overall_agents,
            'total_zero_weights': overall_zero,
            'zero_weight_frac': overall_zero / overall_agents,
            'max_source_zero_frac': overall_max_source_zero_frac,
            'max_source_truncated_mass_ub': overall_max_source_trunc,
            'max_location_truncated_mass_ub': overall_max_loc_trunc,
            'negative_weight_count': negative_weight_count,
            'wsum_dev_over_n_eps32_max': wsum_ratio_max,
            'threshold': TRUNCATION_MASS_THRESHOLD,
            'all_below_threshold': bool(all_below),
        },
        'branch': branch,
        'branch_rule': ('power_rescale if the kappa=0.25 truncated-mass upper bound '
                        'is < 1% for every (combination, source); otherwise '
                        'model_forward (same approach as 018: CPU float64 forward '
                        'pass, re-run inference at the new tau).'),
        'cross_reference': ("026's diagnostics.zero_weight_count measured "
                            'seed_42/baseline/fold1 = 0; this diagnostic extends '
                            'that check to all 48 combinations.'),
    }

    ov = diagnostic['overall']
    print(f"\n  Total zero weights = {ov['total_zero_weights']} / {ov['total_agents_scanned']} "
          f"(fraction {ov['zero_weight_frac']:.3e})")
    print(f"  Truncated-mass upper bound: max_source = {ov['max_source_truncated_mass_ub']:.3e}, "
          f"max_location = {ov['max_location_truncated_mass_ub']:.3e} "
          f"(threshold {TRUNCATION_MASS_THRESHOLD})")
    print(f"  sum(w) sanity check: max |sum(w)-1|/(n*eps32) = {ov['wsum_dev_over_n_eps32_max']:.3f} (< 1 is normal floating-point accumulation)")
    print(f"  -> branch = {branch}")
    return diagnostic


# ════════════════════════════════════════════════════════════
# Stage 2 - Power rescaling + correction arms + evaluation
# ════════════════════════════════════════════════════════════

def power_rescale_base(g: np.ndarray, ctx: dict, kappa: float) -> np.ndarray:
    """Power-rescale the base demand: per-ITL3 base_kappa = M_r * w^kappa /
    sum(w^kappa), with M_r = sum(g) (the measured region total).

    w = g / D_r is the weight recovered by back-out. Using M_r rather
    than the nominal D_r makes kappa=1 agree with g bit-for-bit to
    ~1e-15 (the precondition for the same-source strict anchor); the two
    differ by a relative ~1e-7 (float32 softmax storage error), which
    has no scientific significance -- see the module docstring.
    """
    out = np.zeros_like(g, dtype=float)
    for itl3, idx in ctx['itl3_groups']:
        w = g[idx] / ctx['region_demand'][itl3]
        t = w ** kappa
        s = t.sum()
        if s <= 0:
            raise RuntimeError(
                f'{itl3}: sum(w^kappa) = {s} (kappa={kappa}) -- an all-zero '
                f'source, power rescaling is undefined here; the underflow '
                f'diagnostic should already have caught this case.')
        out[idx] = g[idx].sum() * t / s
    return out


def compute_demand_arms(base: np.ndarray, ctx: dict) -> dict:
    """Base demand plus the four post-correction arms (shared module,
    using the frozen numerical path from 017)."""
    grid_gdf = ctx['grid_gdf']
    region_sub = ctx['region_sub']
    fac = ctx['factors']
    return {
        'baseline': base,
        'postN': scu.apply_standard_multiplicative(base, fac['N'], grid_gdf, region_sub),
        'postP': scu.apply_standard_multiplicative(base, fac['P'], grid_gdf, region_sub),
        'postNP': scu.apply_standard_multiplicative(base, fac['NP'], grid_gdf, region_sub),
        'addNP': scu.apply_additive_correction(base, fac['NP'], grid_gdf, region_sub),
    }


def evaluate_demand(demand: np.ndarray, ctx: dict) -> dict:
    """Voronoi aggregation (cached assignment) -> per-region metrics
    (shared module)."""
    subs_result = scu.aggregate_by_assignment(ctx['subs_sub'], ctx['assignment'], demand)
    return scu.evaluate_allocation(subs_result)


def concentration_of(demand: np.ndarray) -> dict:
    """Concentration metrics for the within-region demand distribution
    (defined in the module docstring)."""
    d = np.asarray(demand, dtype=float)
    n = len(d)
    total = d.sum()
    assert total > 0, 'Total regional demand is 0 -- concentration is undefined'
    p = d / total

    pos = p > 0
    entropy_norm = float(-(p[pos] * np.log(p[pos])).sum() / np.log(n))

    srt = np.sort(d)                                    # ascending order
    gini = float(((2 * np.arange(1, n + 1) - n - 1) * srt).sum() / (n * total))

    desc = srt[::-1]
    cum = np.cumsum(desc)
    shares = {}
    for q in TOP_SHARES:
        k = max(1, int(np.ceil(q * n)))
        shares[f'top{int(q * 100)}_share'] = float(cum[k - 1] / total)

    return {'n_agents': n, 'entropy_norm': entropy_norm, 'gini': gini, **shares}


def run_kappa_sweep() -> tuple:
    """Sweep all kappa settings: produce tau_sweep rows, concentration
    rows, and anchor records.

    Returns (sweep_rows, conc_rows, anchor_report).
    """
    print('\n' + '═' * 68)
    print('Stage 2 - Kappa power-rescaling sweep (test-region convention, 192 combination-regions x 5 kappa x 5 arms)')
    print('═' * 68)

    sweep_rows = []
    conc_rows = []

    # Accumulators for the kappa=1 same-source strict anchor
    k1_max_rel_dev = 0.0
    k1_worst = None
    k1_n_cells = 0

    # Accumulators for the informational cross-check against the frozen fold CSVs
    frozen_devs = []

    for seed in SEEDS:
        for config in CONFIGS:
            splits = get_fold_splits(seed, config)
            covered_locs = []
            for fold_dir_name, test_locs in splits.items():
                fold_csv_path = (EXP0_DIR / f'seed_{seed}' / config
                                 / fold_dir_name / 'rmse.csv')
                fold_csv = pd.read_csv(fold_csv_path, index_col=0)
                for loc in test_locs:
                    covered_locs.append(loc)
                    ctx = get_location_context(loc)
                    g = load_gnn_demand(seed, config, fold_dir_name, loc)

                    # -- Direct same-source computation (kappa=1 strict-anchor
                    # reference, from the same grid_demands, without rescaling) --
                    direct_metrics = {arm: evaluate_demand(d, ctx)
                                      for arm, d in compute_demand_arms(g, ctx).items()}

                    # -- Informational cross-check against the frozen fold CSVs
                    # (not fatal, records GPU-rerun drift) --
                    for arm, row_name in FROZEN_CSV_ROW.items():
                        frozen_devs.append(abs(direct_metrics[arm]['rmse']
                                               - float(fold_csv.loc[row_name, loc])))

                    for kappa in KAPPAS:
                        base_k = power_rescale_base(g, ctx, kappa)
                        arms_k = compute_demand_arms(base_k, ctx)
                        for arm, demand in arms_k.items():
                            m = evaluate_demand(demand, ctx)
                            sweep_rows.append({
                                'seed': seed, 'config': config,
                                'fold': fold_dir_name, 'location': loc,
                                'kappa': kappa, 'tau_effective': TAU_TRAIN / kappa,
                                'arm': arm, **m,
                            })
                            conc_rows.append({
                                'seed': seed, 'config': config,
                                'fold': fold_dir_name, 'location': loc,
                                'kappa': kappa, 'tau_effective': TAU_TRAIN / kappa,
                                'arm': arm, **concentration_of(demand),
                            })
                            # kappa=1 same-source strict anchor
                            if kappa == 1.0:
                                k1_n_cells += 1
                                rel = (abs(m['rmse'] - direct_metrics[arm]['rmse'])
                                       / direct_metrics[arm]['rmse'])
                                if rel > k1_max_rel_dev:
                                    k1_max_rel_dev = rel
                                    k1_worst = f'seed_{seed}/{config}/{fold_dir_name}/{loc}/{arm}'
            # Test-region coverage completeness: each (seed, config) must
            # cover each of the 16 regions exactly once
            assert sorted(covered_locs) == sorted(ALL_LOCATIONS), (
                f'seed_{seed}/{config}: unexpected test-region coverage {sorted(covered_locs)}')
            print(f'  seed_{seed}/{config}: 16 test regions x 5 kappa x 5 arms complete')

    if k1_max_rel_dev >= TOL_KAPPA1_SELF_REL:
        raise RuntimeError(
            f'kappa=1 same-source strict anchor failed: max rel dev = '
            f'{k1_max_rel_dev:.3e} >= {TOL_KAPPA1_SELF_REL:g} '
            f'(worst = {k1_worst}) -- the power-rescaling implementation has a bug.')

    frozen_devs = np.asarray(frozen_devs)
    anchor_report = {
        'kappa1_self_anchor': {
            'description': ('Relative deviation of the kappa=1 RMSE from a direct '
                            'recomputation using the same grid_demands (without '
                            'power rescaling) -- these must agree by construction'),
            'n_cells': k1_n_cells,
            'max_rel_dev_rmse': float(k1_max_rel_dev),
            'worst_cell': k1_worst,
            'tol': TOL_KAPPA1_SELF_REL,
            'status': _status(k1_max_rel_dev < TOL_KAPPA1_SELF_REL),
        },
        'frozen_csv_crosscheck': {
            'description': ('Informational cross-check (not fatal): absolute '
                            'deviation of the direct same-source RMSE from the '
                            'frozen fold rmse.csv (4 d.p.). Not used as a strict '
                            'anchor -- exp0 has measured GPU-rerun drift (up to '
                            '8.7e-5, median 2.5e-5), see the TOL_ANCHOR_GNN '
                            'comment in 024.'),
            'n_cells': int(len(frozen_devs)),
            'max_abs_dev': float(frozen_devs.max()),
            'median_abs_dev': float(np.median(frozen_devs)),
            'tol_informational': TOL_FROZEN_CSV_INFO,
            'status': ('ok' if frozen_devs.max() < TOL_FROZEN_CSV_INFO else 'warn'),
        },
    }
    return sweep_rows, conc_rows, anchor_report


def run_model_forward_branch() -> None:
    """The model_forward branch (a reserved switch point, not
    implemented in this run).

    If the diagnostic finds a kappa=0.25 truncated-mass upper bound >=
    1% of regional mass, power rescaling is no longer valid, and
    inference would instead need to follow the model-loading approach
    used in 018_exp_h2_embedding_probing.py: for each (seed, config,
    fold), load the frozen exp0 model.pth (map_location='cpu', with a
    moderate torch.set_num_threads), run a float64 forward pass to the
    EdgeWeightLayer logits, and directly redo the per-source softmax at
    tau_eff = tau0/kappa (a small star graph, so this runs in minutes).

    Expected magnitude (see the module docstring): reaching U_r >= 1%
    would require >= 1e8 zero weights within a single source, three
    orders of magnitude beyond the size of a source -- mathematically
    unreachable. This branch could only be selected if the diagnostic
    finds pathological data (e.g. negative weights or an all-zero
    source), and that kind of pathology already triggers a fail-fast
    error elsewhere. The forward-pass path is therefore not implemented
    in this run; if this branch is ever actually selected, it fails
    explicitly here with implementation guidance rather than silently
    producing results contaminated by truncation.
    """
    raise RuntimeError(
        'branch = model_forward: the underflow diagnostic determined that '
        'power rescaling is not valid (truncated mass >= 1%). This branch '
        'is a reserved switch point and is not implemented -- please '
        'implement a CPU float64 forward pass (map_location="cpu") '
        'following the model-loading approach used in '
        '018_exp_h2_embedding_probing.py, redo the inference-time softmax '
        'at tau_eff = tau0/kappa, and re-run this experiment. The '
        'diagnostic has been written to '
        'results/exp_r213/underflow_diagnostic.json -- please inspect its '
        'overall section first.')


# ════════════════════════════════════════════════════════════
# Stage 3 - Aggregation, weak anchor, regression, headline
# ════════════════════════════════════════════════════════════

def seed_averaged_table(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """First step of the paper's aggregation convention: average over
    seeds for each (config, arm, kappa, location)."""
    return (df.groupby(['config', 'arm', 'kappa', 'location'])[value_col]
              .mean().reset_index())


def check_table3_weak_anchor(sweep_df: pd.DataFrame) -> dict:
    """Weak anchor (fatal): the kappa=1 setting, aggregated using the
    paper's convention, must match the paper's Table 3 numbers to
    +/-0.005."""
    loc_avg = seed_averaged_table(sweep_df, 'rmse')
    k1 = loc_avg[loc_avg['kappa'] == 1.0]
    cells = {}
    all_ok = True
    for (config, arm), expected in TABLE3_RMSE.items():
        vals = k1[(k1['config'] == config) & (k1['arm'] == arm)]['rmse']
        assert len(vals) == len(ALL_LOCATIONS), (config, arm, len(vals))
        actual = float(vals.mean())
        dev = abs(actual - expected)
        ok = dev < TOL_TABLE3
        all_ok &= ok
        cells[f'{config}/{arm}'] = {
            'table3_value': expected,
            'recomputed_kappa1': actual,
            'abs_dev': dev,
            'status': _status(ok),
        }
    report = {
        'description': ('Weak anchor of the kappa=1 setting, aggregated using '
                        'the paper convention (average over seeds per '
                        '(config, region) first, then mean across the 16 '
                        'regions), against the paper Table 3 (tab:main) RMSE '
                        'numbers'),
        'tol': TOL_TABLE3,
        'cells': cells,
        'all_ok': bool(all_ok),
        'status': _status(all_ok),
    }
    if not all_ok:
        bad = {k: v for k, v in cells.items() if v['status'] != 'ok'}
        raise RuntimeError(f'Table 3 weak anchor failed (+/-{TOL_TABLE3:g}): {bad}')
    return report


def build_kappa_curves(sweep_df: pd.DataFrame) -> dict:
    """Kappa-curve aggregation: mean / std(ddof=0) across the 16 regions
    for each (config, arm, kappa)."""
    loc_avg = seed_averaged_table(sweep_df, 'rmse')
    curves: dict = {}
    for (config, arm, kappa), grp in loc_avg.groupby(['config', 'arm', 'kappa']):
        vals = grp['rmse'].to_numpy()          # convert to numpy before computing ddof=0
        curves.setdefault(config, {}).setdefault(arm, {})[str(kappa)] = {
            'rmse_mean': float(vals.mean()),
            'rmse_std_ddof0': float(vals.std()),
        }
    return curves


def build_antagonism_table(sweep_df: pd.DataFrame) -> pd.DataFrame:
    """Per-(config, kappa, location) antagonism magnitude
    delta = postNP - baseline (after averaging over seeds)."""
    loc_avg = seed_averaged_table(sweep_df, 'rmse')
    wide = loc_avg.pivot_table(index=['config', 'kappa', 'location'],
                               columns='arm', values='rmse').reset_index()
    wide['delta_np'] = wide['postNP'] - wide['baseline']
    return wide


def build_regressions(sweep_df: pd.DataFrame, conc_df: pd.DataFrame) -> dict:
    """Antagonism-vs-concentration regression (defined in the module
    docstring): per kappa, both per-config (n=16) and pooled (n=64)."""
    delta = build_antagonism_table(sweep_df)

    # Baseline concentration (baseline arm), averaged over seeds down to
    # (config, kappa, location)
    base_conc = conc_df[conc_df['arm'] == 'baseline'].copy()
    base_conc['one_minus_entropy'] = 1.0 - base_conc['entropy_norm']
    conc_avg = (base_conc.groupby(['config', 'kappa', 'location'])
                [['gini', 'one_minus_entropy', 'top10_share']].mean().reset_index())

    merged = delta.merge(conc_avg, on=['config', 'kappa', 'location'], validate='1:1')

    def _one_regression(sub: pd.DataFrame, predictor: str) -> dict:
        x = sub[predictor].to_numpy(dtype=float)
        y = sub['delta_np'].to_numpy(dtype=float)
        lr = linregress(x, y)
        rho, rho_p = spearmanr(x, y)
        return {
            'n': int(len(sub)),
            'slope': float(lr.slope),
            'intercept': float(lr.intercept),
            'pearson_r': float(lr.rvalue),
            'p_value': float(lr.pvalue),
            'stderr': float(lr.stderr),
            'spearman_rho': float(rho),
            'spearman_p': float(rho_p),
        }

    regressions: dict = {}
    for kappa in KAPPAS:
        sub_k = merged[merged['kappa'] == kappa]
        entry: dict = {}
        for config in CONFIGS:
            sub = sub_k[sub_k['config'] == config]
            entry[config] = {p: _one_regression(sub, p) for p in REG_PREDICTORS}
        entry['pooled_all_configs'] = {p: _one_regression(sub_k, p)
                                       for p in REG_PREDICTORS}
        regressions[str(kappa)] = entry
    return regressions


def build_headline(sweep_df: pd.DataFrame, regressions: dict) -> dict:
    """Rule-generated headline conclusions (never hand-written)."""
    delta = build_antagonism_table(sweep_df)
    antagonism_by_kappa: dict = {}
    for config in CONFIGS:
        antagonism_by_kappa[config] = {}
        for kappa in KAPPAS:
            vals = delta[(delta['config'] == config)
                         & (delta['kappa'] == kappa)]['delta_np'].to_numpy()
            antagonism_by_kappa[config][str(kappa)] = {
                'delta_np_mean': float(vals.mean()),
                'delta_np_std_ddof0': float(vals.std()),
                'n_regions_worse': int((vals > 0).sum()),
            }

    headline: dict = {'antagonism_by_kappa': antagonism_by_kappa}

    # Directional reading 1: direction of the antagonism magnitude with
    # kappa (per config: Spearman rank correlation of the means across
    # the 5 kappa settings)
    trend = {}
    for config in CONFIGS:
        means = [antagonism_by_kappa[config][str(k)]['delta_np_mean'] for k in KAPPAS]
        rho, _ = spearmanr(KAPPAS, means)
        trend[config] = {
            'delta_np_means_by_kappa': dict(zip([str(k) for k in KAPPAS],
                                                [float(m) for m in means])),
            'spearman_rho_vs_kappa': float(rho),
            'direction': ('increasing_with_kappa' if rho > 0
                          else 'decreasing_with_kappa' if rho < 0
                          else 'flat'),
            'monotone_nondecreasing': bool(all(means[i] <= means[i + 1] + 1e-12
                                               for i in range(len(means) - 1))),
        }
    headline['antagonism_vs_kappa_trend'] = trend

    # Directional reading 2: slope direction of the pooled
    # antagonism-vs-concentration regression at kappa=1 (the paper's
    # condition)
    k1_pooled = regressions['1.0']['pooled_all_configs']
    headline['concentration_slope_at_kappa1'] = {
        p: {
            'slope': k1_pooled[p]['slope'],
            'pearson_r': k1_pooled[p]['pearson_r'],
            'p_value': k1_pooled[p]['p_value'],
            'direction': ('antagonism_increases_with_concentration'
                          if k1_pooled[p]['slope'] > 0
                          else 'antagonism_decreases_with_concentration'),
        }
        for p in REG_PREDICTORS
    }
    return headline


# ════════════════════════════════════════════════════════════
# Main pipeline
# ════════════════════════════════════════════════════════════

def main() -> None:
    np.random.seed(RNG_SEED)   # this experiment makes no random calls; seed fixed and logged for consistency
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # -- Stage 1: underflow diagnostic (decides the branch, written to disk first) --
    diagnostic = run_underflow_diagnostic()
    with open(OUTPUT_DIR / 'underflow_diagnostic.json', 'w', encoding='utf-8') as f:
        json.dump(diagnostic, f, ensure_ascii=False, indent=2)
    print(f"\nDiagnostic written to {OUTPUT_DIR / 'underflow_diagnostic.json'}")

    if diagnostic['branch'] == 'model_forward':
        run_model_forward_branch()   # reserved switch point: fails explicitly with implementation guidance

    # -- Stage 2: kappa sweep --
    sweep_rows, conc_rows, anchor_report = run_kappa_sweep()
    sweep_df = pd.DataFrame(sweep_rows)
    conc_df = pd.DataFrame(conc_rows)
    sweep_df.to_csv(OUTPUT_DIR / 'tau_sweep.csv', index=False)
    conc_df.to_csv(OUTPUT_DIR / 'concentration_metrics.csv', index=False)
    print(f'\ntau_sweep.csv: {len(sweep_df)} rows; concentration_metrics.csv: {len(conc_df)} rows')

    # -- Stage 3: weak anchor + aggregation + regression + headline --
    anchor_report['table3_weak_anchor'] = check_table3_weak_anchor(sweep_df)
    kappa_curves = build_kappa_curves(sweep_df)
    regressions = build_regressions(sweep_df, conc_df)
    headline = build_headline(sweep_df, regressions)

    results = {
        'meta': {
            'script': '028_exp_r213_tau_concentration.py',
            'plan_step': 'Tau and concentration analysis',
            'branch': diagnostic['branch'],
            'tau_train': TAU_TRAIN,
            'kappas': KAPPAS,
            'tau_effective': {str(k): TAU_TRAIN / k for k in KAPPAS},
            'kappa_tau_duality': ('normalizing softmax(z/tau0)^kappa is equivalent '
                                  'to softmax(z/(tau0/kappa)): power rescaling is '
                                  'the same as changing tau to tau0/kappa at '
                                  'inference time'),
            'seeds': SEEDS,
            'configs': CONFIGS,
            'arms': ARMS,
            'base_col': BASE_COL,
            'protocol': ('test-fold convention (each of the 16 regions serves as '
                         'the test region exactly once per (seed, config)); '
                         'aggregation = average over seeds per (config, region) '
                         'first, then mean/std(ddof=0) across the 16 regions'),
            'mass_convention': ('power-rescaling mass M_r = sum(gnn_demand) (the '
                                'measured region total), not the nominal D_r -- '
                                'the two differ by a relative ~1e-7 (float32 '
                                'storage error); using M_r makes kappa=1 agree '
                                'with the stored demand bit-for-bit (the '
                                'precondition for the same-source strict anchor)'),
            'rng_seed': RNG_SEED,
            'deterministic': True,
            'cpu_only': True,
            'python': platform.python_version(),
            'numpy': np.__version__,
            'timestamp': datetime.now().isoformat(timespec='seconds'),
        },
        'anchors': anchor_report,
        'kappa_curves_rmse': kappa_curves,
        'regressions': regressions,
        'headline': headline,
    }
    with open(OUTPUT_DIR / 'regression.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # -- Console summary --
    print('\n' + '=' * 68)
    print('Tau/Concentration Analysis Overview')
    print('=' * 68)
    print(f"branch = {diagnostic['branch']}")
    ka = anchor_report['kappa1_self_anchor']
    print(f"kappa=1 same-source strict anchor: max rel dev = {ka['max_rel_dev_rmse']:.3e} "
          f"(< {ka['tol']:g}) -> {ka['status']}")
    fz = anchor_report['frozen_csv_crosscheck']
    print(f"Frozen CSV cross-check (informational): max = {fz['max_abs_dev']:.3e}, "
          f"median = {fz['median_abs_dev']:.3e} -> {fz['status']}")
    t3 = anchor_report['table3_weak_anchor']
    print(f"Table 3 weak anchor (+/-{TOL_TABLE3:g}): {t3['status']}")
    for cell, v in t3['cells'].items():
        print(f"    {cell:22s} paper {v['table3_value']:6.2f}  recomputed "
              f"{v['recomputed_kappa1']:8.4f}  dev {v['abs_dev']:.4f}")
    print('\nAntagonism magnitude delta(postNP-baseline) mean vs. kappa (baseline config):')
    for k in KAPPAS:
        e = headline['antagonism_by_kappa']['baseline'][str(k)]
        print(f"    kappa={k:<5g} tau_eff={TAU_TRAIN / k:<7g} delta = {e['delta_np_mean']:+.4f} "
              f"+/- {e['delta_np_std_ddof0']:.4f}  (regions worse {e['n_regions_worse']}/16)")
    sl = headline['concentration_slope_at_kappa1']['gini']
    print(f"\nkappa=1 pooled antagonism-vs-concentration (gini) regression: slope = {sl['slope']:+.4f}, "
          f"r = {sl['pearson_r']:+.3f}, p = {sl['p_value']:.3g} -> {sl['direction']}")
    print(f'\nArtifacts written to {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
