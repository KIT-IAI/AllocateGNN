# -*- coding: utf-8 -*-
"""
023 - Additive correction full matrix (static and GNN bases)

The mechanism-isolation results originally reported additive correction
only on the GNN base flow (see the `mechanism_isolation` table), leaving a
gap in the "2 correction forms x 3 bases x 3 signals" matrix: the two
static bases (Uniform, GPM) had never been evaluated under additive
correction with {N, P, NP} signals. This experiment fills in the full
matrix and applies the standard significance-testing protocol to the
newly added head-to-head comparisons.

=== Full matrix definition (21 arms = 3 bases x 7 variants) ===
- Bases: Uniform (uniform allocation per ITL3 region) / GPM (categorical
  land-use weighting) / GNN (exp0 baseline `grid_demands['gnn_demand']`;
  for each (seed, region) pair, the fold in which that region served as
  the test set is used -- same out-of-sample convention as in 017).
- Variants: uncorrected / multiplicative N/P/NP / additive N/P/NP.
- All corrections go through the shared module (identical definitions to
  017): factors are `ntl_factor` / `prox_factor` / their product from
  `compute_factors`; multiplicative correction uses
  `apply_standard_multiplicative`; additive correction uses
  `apply_additive_correction` (scale alpha = base_std/offset_std, ddof=0).
- Aggregation: Voronoi assignment (EPSG:3857, a fixed convention for all
  arms) -> per-substation -> rmse/mae/corr.
- For GNN arms, metrics are computed per seed first and then averaged
  across seeds; the per-seed values are also saved separately.

=== Alignment discipline: verify against existing anchors before extending ===
The 15 pre-existing arms (8 static + 4 GNN multiplicative + 3 GNN
additive) must pass every available anchor check below before any new
arm is added:
(1) Anchor against the existing frozen output (agreement within stored
    precision -- the frozen CSVs store values at 2 or 4 decimal places):
    the 6 static arms are checked against `all_regions_{metric}.csv`
    (rmse/mae at 2dp -> tolerance 0.005; corr at 4dp -> tolerance 5e-5).
    UniP and GPMpostP have no corresponding row in the frozen CSV and are
    honestly recorded as `not_in_frozen_csv`. The 4 GNN arms are checked
    against `kfold_test_{metric}.csv` (4dp) with a tolerance of 5e-4
    (empirically, `grid_demands` differs from the frozen fold CSVs by up
    to 8.7e-5 due to GPU re-inference drift on rerun; 5e-4 is an order of
    magnitude above that measured drift while still two orders of
    magnitude tighter than a seed/fold mismatch would produce).
(2) Anchor against the full-precision recomputation (a strong,
    full-precision anchor): GPM/GPMpostN/GPMpostP/GPMpostNP are checked
    region-by-region against `exp_r216/recomputed_static_arms.csv` with
    |dev| <= max(1e-9, 1e-6*|ref|) (that recomputation runs the chained
    correction logic in 003, while this script applies the combined-
    factor single-pass renormalization from 017; the two are
    algebraically identical in real arithmetic, so floating-point noise
    is ~1e-13). This gives GPMpostP its only full-precision anchor.
(3) Anchor against the published paper tables (a weak anchor, +-0.005):
    the 8 static arms against Table 3 (tab:main), the 4 GNN
    multiplicative arms against Table 3, and the 3 GNN additive arms
    against tab:mechanism_isolation. Two evaluation conventions are
    checked -- aggregating at full precision, or aggregating after
    rounding to the stored precision (the published numbers were derived
    from the mean of the rounded CSVs, the same convention used in the
    full-precision recomputation) -- and passing either one within
    +-0.005 counts as a pass.
(4) Same-source strong anchor (GNN multiplicative arms only): the
    `grid_demands` pickle also stores the corrected demand values
    written by 005 (`ntl`/`prox`/`ntl_prox_gnn_demand`); evaluating these
    directly and comparing against the multiplicative arms this script
    recomputes from `gnn_demand` gives |dev| <= max(1e-9, 1e-6*|ref|),
    demonstrating that the correction pipeline reproduces 005's output
    bit-for-bit and is unaffected by GPU re-inference drift (an anchor
    derived from the same underlying source recomputation).
UniP's only numerical anchor is the paper's Table 3 (it has no row in
the frozen CSV and was not recomputed at full precision); this is
recorded honestly.

=== Structural degeneracy of Uniform x additive (reported honestly, not a bug) ===
The additive correction's scale factor alpha = base_std/offset_std is
supplied entirely by **the base allocation's own within-region
variance**. For the Uniform base, within-ITL3-region base_std = 0, so
alpha = 0 and the correction reduces exactly to the baseline (leaving
only the floating-point noise of one idempotent renormalization pass).
The matrix retains this arm under its matched definition and records it
as-is; both `full_matrix.csv` and `anchor_report.json` flag it with
`structural_degeneration=true`. This is a direct, expected consequence
of the fact that the additive correction's scale is derived from the
base allocation's own variance -- useful as material for the methods
section.

=== New head-to-head comparisons (same statistical protocol as 031) ===
Procedure: seed-average -> paired difference across the 16 regions ->
exact sign-flip permutation test (full enumeration of 2^16 sign
patterns) + Holm correction (one family per metric) + paired-region
bootstrap CI (outer resampling only, B=10^4; no bootstrap p-value is
reported). Static-vs-static comparisons have no seed dimension and use
the `no_seed` branch. Comparisons involving a GNN arm additionally
report per-seed permutation p-values (fully transparent, not pooled;
the Cauchy-combination column is for reference only). The degeneracy-
verification comparisons (UniAdd* vs Uni) form their own separate
family -- they are structural-identity checks, not scientific
hypotheses, and are kept out of the Holm family used for the real
comparisons to avoid diluting it.

Outputs:
    results/exp_r12/full_matrix.csv           -- 21 arms x 3 metrics x 16 regions (GNN arms seed-averaged)
    results/exp_r12/full_matrix_per_seed.csv  -- GNN arms (7), per-seed values across the 3 seeds
    results/exp_r12/new_comparisons.csv       -- statistical tests for the new comparisons (protocol fields)
    results/exp_r12/anchor_report.json        -- anchor-check report + degeneracy flags + protocol metadata

All conclusion/pass-fail fields are generated by explicit numeric rules
rather than written by hand; the entire computation runs on CPU with a
fixed random seed; existing frozen outputs are only read, never
modified. Run with: `python 023_exp_r12_additive_full_matrix.py`.
"""

import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import shared_correction_utils as scu                       # noqa: E402
from revision_statistics import (                            # noqa: E402
    exact_sign_flip_permutation, paired_region_bootstrap,
    cauchy_combination, holm,
)
from SpatialAllocation.Weighter import weighter_registry     # noqa: E402

# ════════════════════════════════════════════════════════════
# Paths and constants
# ════════════════════════════════════════════════════════════

STATIC_DIR = SCRIPT_DIR / 'results' / 'static_allocation'
EXP0_DIR = SCRIPT_DIR / 'results' / 'exp0_kfold_prior'
S5_RECOMPUTED_CSV = SCRIPT_DIR / 'results' / 'exp_r216' / 'recomputed_static_arms.csv'
OUTPUT_DIR = SCRIPT_DIR / 'results' / 'exp_r12'

SEEDS = [42, 123, 456]
METRICS = ['rmse', 'mae', 'corr']
SIGNALS = ['N', 'P', 'NP']
ALL_LOCATIONS = scu.ALL_LOCATIONS          # 16 study regions (matrix column order)
N_REGIONS = len(ALL_LOCATIONS)
ALPHA = 0.05

# All random sources are fixed for determinism: the bootstrap seed is
# derived from the comparison row index
B_BOOT = 10000                              # same convention as used in 031
BOOT_SEED_BASE = 12001

# Anchor tolerances (see checks (1)-(4) above)
TOL_FROZEN_2DP = 0.005 + 1e-9               # rmse/mae: stored at 2 decimal places -> half the last digit
TOL_FROZEN_4DP = 5e-5 + 1e-9                # corr: stored at 4 decimal places -> half the last digit
TOL_GNN_FROZEN = 5e-4                       # GNN arms: 4dp storage + GPU re-inference drift on rerun (empirically measured)
TOL_PAPER = 0.005                           # weak anchor against the paper tables
STRONG_REL = 1e-6                           # full-precision strong anchor: relative tolerance
STRONG_ABS = 1e-9                           # full-precision strong anchor: absolute floor (guards corr values near zero)
TOL_DEGENERATION = 1e-9                     # upper bound on floating-point noise from the idempotent renormalization in UniAdd == Uni

# Stored decimal precision of the frozen CSV / kfold CSV files (used for
# the dual-convention rounding in check (3))
STORE_DP_STATIC = {'rmse': 2, 'mae': 2, 'corr': 4}
STORE_DP_GNN = {'rmse': 4, 'mae': 4, 'corr': 4}

# Land-use columns and regional percentage columns from 005/003 (used to
# reconstruct the GPM base; order corresponds 1:1; same as in 024)
LU_COLS = ['lu_residential_prop', 'lu_commercial_prop', 'lu_industrial_prop',
           'lu_agricultural_prop', 'lu_others_prop']
PCT_COLS = ['residential_percent', 'commercial_percent', 'industrial_percent',
            'agricultural_percent', 'others_percent']

# ════════════════════════════════════════════════════════════
# Arm definitions (21 = 3 bases x {none, mult x3, add x3})
# ════════════════════════════════════════════════════════════

# (label, base, form, signal); labels follow the paper's naming convention (Uni*/GPM*/GNN*)
ARMS = [
    ('Uni',       'uniform', 'none', ''),
    ('UniN',      'uniform', 'mult', 'N'),
    ('UniP',      'uniform', 'mult', 'P'),
    ('UniNP',     'uniform', 'mult', 'NP'),
    ('UniAddN',   'uniform', 'add',  'N'),
    ('UniAddP',   'uniform', 'add',  'P'),
    ('UniAddNP',  'uniform', 'add',  'NP'),
    ('GPM',       'gpm',     'none', ''),
    ('GPMpostN',  'gpm',     'mult', 'N'),
    ('GPMpostP',  'gpm',     'mult', 'P'),
    ('GPMpostNP', 'gpm',     'mult', 'NP'),
    ('GPMaddN',   'gpm',     'add',  'N'),
    ('GPMaddP',   'gpm',     'add',  'P'),
    ('GPMaddNP',  'gpm',     'add',  'NP'),
    ('GNN',       'gnn',     'none', ''),
    ('GNNpostN',  'gnn',     'mult', 'N'),
    ('GNNpostP',  'gnn',     'mult', 'P'),
    ('GNNpostNP', 'gnn',     'mult', 'NP'),
    ('GNNaddN',   'gnn',     'add',  'N'),
    ('GNNaddP',   'gnn',     'add',  'P'),
    ('GNNaddNP',  'gnn',     'add',  'NP'),
]
ARM_META = {label: (base, form, signal) for label, base, form, signal in ARMS}
DEGENERATE_ARMS = ['UniAddN', 'UniAddP', 'UniAddNP']   # structurally degenerate arms (see note above)

# Row-name mapping into the frozen all_regions_{metric}.csv (anchor (1); missing rows are recorded honestly)
FROZEN_ROW_OF = {
    'Uni': 'voronoi',
    'UniN': 'voronoi_ntl',
    'UniP': None,                    # voronoi_prox2 is not present in the frozen CSV (previously recorded)
    'UniNP': 'voronoi_prox2_ntl',
    'GPM': 'voronoi_gpm',
    'GPMpostN': 'voronoi_ntl_gpm',
    'GPMpostP': None,                # voronoi_prox2_gpm is not present in the frozen CSV (recomputed at full precision separately)
    'GPMpostNP': 'voronoi_prox2_ntl_gpm',
}

# Row-name mapping into the full-precision recomputation table (exp_r216/recomputed_static_arms.csv) -- anchor (2)
S5_ROW_OF = {
    'GPM': 'voronoi_gpm',
    'GPMpostN': 'voronoi_ntl_gpm',
    'GPMpostP': 'voronoi_prox2_gpm',
    'GPMpostNP': 'voronoi_prox2_ntl_gpm',
}

# Row-name mapping into the frozen kfold_test_{metric}.csv -- GNN anchor (1)
KFOLD_ROW_OF = {
    'GNN': 'voronoi_GNN',
    'GNNpostN': 'voronoi_ntl_GNN',
    'GNNpostP': 'voronoi_prox_GNN',
    'GNNpostNP': 'voronoi_ntl_prox_GNN',
}

# Same-source strong anchor (4): keys of the corrected demand values that 005 writes into the grid_demands pickle
SAME_SOURCE_KEY_OF = {
    'GNNpostN': 'ntl_gnn_demand',
    'GNNpostP': 'prox_gnn_demand',
    'GNNpostNP': 'ntl_prox_gnn_demand',
}

# Paper Table 3 (tab:main), transcribed verbatim from the manuscript: {arm: {metric: (mean, std)}}
PAPER_TABLE3 = {
    'Uni':       {'rmse': (13.10, 3.70), 'mae': (9.23, 2.21), 'corr': (0.009, 0.178)},
    'UniN':      {'rmse': (11.16, 3.29), 'mae': (7.96, 2.02), 'corr': (0.094, 0.190)},
    'UniP':      {'rmse': (8.98, 2.96),  'mae': (6.71, 1.89), 'corr': (0.119, 0.194)},
    'UniNP':     {'rmse': (7.45, 2.79),  'mae': (5.50, 1.68), 'corr': (0.319, 0.194)},
    'GPM':       {'rmse': (12.39, 3.36), 'mae': (8.74, 2.11), 'corr': (0.035, 0.199)},
    'GPMpostN':  {'rmse': (10.54, 2.80), 'mae': (7.56, 1.84), 'corr': (0.128, 0.208)},
    'GPMpostP':  {'rmse': (8.55, 2.50),  'mae': (6.45, 1.72), 'corr': (0.152, 0.179)},
    'GPMpostNP': {'rmse': (7.31, 2.40),  'mae': (5.44, 1.50), 'corr': (0.332, 0.173)},
    'GNN':       {'rmse': (9.27, 2.61),  'mae': (6.67, 1.76), 'corr': (0.298, 0.201)},
    'GNNpostN':  {'rmse': (9.45, 2.55),  'mae': (6.58, 1.64), 'corr': (0.365, 0.187)},
    'GNNpostP':  {'rmse': (9.42, 2.33),  'mae': (6.59, 1.52), 'corr': (0.349, 0.166)},
    'GNNpostNP': {'rmse': (11.20, 2.64), 'mae': (7.64, 1.66), 'corr': (0.348, 0.160)},
}

# Paper tab:mechanism_isolation, transcribed verbatim from the manuscript: the 3 GNN additive arms
PAPER_MECHANISM = {
    'GNNaddNP': {'rmse': (7.34, 2.14), 'mae': (5.37, 1.41), 'corr': (0.448, 0.188)},
    'GNNaddN':  {'rmse': (8.82, 2.63), 'mae': (6.35, 1.84), 'corr': (0.452, 0.184)},
    'GNNaddP':  {'rmse': (7.03, 2.27), 'mae': (5.20, 1.46), 'corr': (0.405, 0.164)},
}

# ════════════════════════════════════════════════════════════
# New head-to-head comparisons (covers the required comparisons plus the missing symmetric cells)
# ════════════════════════════════════════════════════════════

# (label_a, label_b); family='main' = genuine hypothesis family (one Holm
# family per metric), family='degeneration' = degeneracy-verification
# family (structural-identity checks, kept in its own family)
NEW_COMPARISONS = [
    # ── Headline comparison: best additive-GNN arm vs. best-performing static arm overall ──
    dict(cid=1,  a='GNNaddP',  b='GPMpostNP', family='main'),
    dict(cid=2,  a='GNNaddNP', b='GPMpostNP', family='main'),
    # ── additive vs baseline (GNN base) ──
    dict(cid=3,  a='GNNaddN',  b='GNN', family='main'),
    dict(cid=4,  a='GNNaddP',  b='GNN', family='main'),
    dict(cid=5,  a='GNNaddNP', b='GNN', family='main'),
    # ── additive vs multiplicative, same signal (GNN base) ──
    dict(cid=6,  a='GNNaddN',  b='GNNpostN', family='main'),
    dict(cid=7,  a='GNNaddP',  b='GNNpostP', family='main'),
    dict(cid=8,  a='GNNaddNP', b='GNNpostNP', family='main'),
    # ── additive vs baseline (GPM base; static-vs-static -> no_seed branch) ──
    dict(cid=9,  a='GPMaddN',  b='GPM', family='main'),
    dict(cid=10, a='GPMaddP',  b='GPM', family='main'),
    dict(cid=11, a='GPMaddNP', b='GPM', family='main'),
    # ── additive vs multiplicative, same signal (GPM base) ──
    dict(cid=12, a='GPMaddN',  b='GPMpostN', family='main'),
    dict(cid=13, a='GPMaddP',  b='GPMpostP', family='main'),
    dict(cid=14, a='GPMaddNP', b='GPMpostNP', family='main'),
    # ── degeneracy verification (UniAdd* == Uni; a structural-identity check, not a hypothesis) ──
    dict(cid=15, a='UniAddN',  b='Uni', family='degeneration'),
    dict(cid=16, a='UniAddP',  b='Uni', family='degeneration'),
    dict(cid=17, a='UniAddNP', b='Uni', family='degeneration'),
]

# ════════════════════════════════════════════════════════════
# Static base reconstruction (same approach as 024; closely mirrors the 003/005 definitions)
# ════════════════════════════════════════════════════════════

def compute_uniform_base(grid_gdf, region_sub) -> np.ndarray:
    """Uniform base: uniform allocation per ITL3 region = total_demand / len(group)."""
    region_info = region_sub.set_index('ITL3')
    base = np.zeros(len(grid_gdf))
    for itl3, group in grid_gdf.groupby('ITL3'):
        if itl3 not in region_info.index:
            continue
        total_demand = region_info.loc[itl3, 'Demand (MVA)']
        base[group.index] = total_demand / len(group)
    return base


def compute_gpm_base(grid_gdf, region_sub, subs_sub):
    """GPM base: reconstructs the 003/005 'landuse_demand' logic (categorical
    GPM weights -> per-ITL3 normalization).

    Returns (base array, number of ITL3 groups that triggered the fallback).
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
# Region context and fold assignment
# ════════════════════════════════════════════════════════════

def load_location_context(loc: str, assignment_cache: dict) -> dict:
    """Load all shared quantities for one region (computed once per region;
    same discipline/assertions as in 024)."""
    grid_gdf, region_sub, subs_sub, ntl_values = scu.load_grid_and_subs(loc)

    # RangeIndex requirement and ITL3 coverage check
    assert (grid_gdf.index == np.arange(len(grid_gdf))).all(), \
        f'{loc}: grid_gdf is not a RangeIndex -- positional index semantics are broken'
    assert set(grid_gdf['ITL3'].unique()) <= set(region_sub['ITL3']), \
        f'{loc}: some ITL3 values are not covered by region_sub'

    prox_scores = scu.compute_prox_scores(grid_gdf, subs_sub)
    ntl_factor, prox_factor = scu.compute_factors(grid_gdf, ntl_values, prox_scores)
    factors = {'N': ntl_factor, 'P': prox_factor, 'NP': ntl_factor * prox_factor}

    assignment = scu.compute_voronoi_assignment(
        grid_gdf, subs_sub, cache=assignment_cache, cache_key=loc)

    return {'loc': loc, 'grid_gdf': grid_gdf, 'region_sub': region_sub,
            'subs_sub': subs_sub, 'factors': factors, 'assignment': assignment}


def load_fold_of() -> dict:
    """Fold assignment per seed: loc -> fold directory name (each region
    appears in the test set exactly once per seed)."""
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
        assert set(mapping) == set(ALL_LOCATIONS), \
            f'seed {seed}: the test folds do not cover all 16 regions'
        fold_of[seed] = mapping
    return fold_of


def eval_demand(ctx: dict, demand_arr: np.ndarray) -> dict:
    """Demand array -> Voronoi aggregation (assignment is cached) -> the three metrics."""
    subs_result = scu.aggregate_by_assignment(
        ctx['subs_sub'], ctx['assignment'], demand_arr)
    return scu.evaluate_allocation(subs_result)


def seven_variants(ctx: dict, base_demand: np.ndarray) -> dict:
    """Metrics for the 7 variants of one base (none + multiplicative N/P/NP
    + additive N/P/NP).

    Returns {('none',''): metrics, ('mult','N'): ..., ('add','NP'): ...}.
    """
    grid_gdf, region_sub = ctx['grid_gdf'], ctx['region_sub']
    out = {('none', ''): eval_demand(ctx, base_demand)}
    for sig in SIGNALS:
        mult = scu.apply_standard_multiplicative(
            base_demand, ctx['factors'][sig], grid_gdf, region_sub)
        out[('mult', sig)] = eval_demand(ctx, mult)
        add = scu.apply_additive_correction(
            base_demand, ctx['factors'][sig], grid_gdf, region_sub)
        out[('add', sig)] = eval_demand(ctx, add)
    return out


# ════════════════════════════════════════════════════════════
# Anchor-checking utilities
# ════════════════════════════════════════════════════════════

def _entry(max_abs_dev: float, tol: float, rule: str) -> dict:
    return {'max_abs_dev': float(max_abs_dev), 'tol': float(tol),
            'rule': rule, 'passed': bool(max_abs_dev <= tol)}


def _strong_entry(got: np.ndarray, ref: np.ndarray, rule: str) -> dict:
    """Full-precision strong anchor: elementwise |dev| <= max(STRONG_ABS, STRONG_REL*|ref|)."""
    got = np.asarray(got, dtype=float)
    ref = np.asarray(ref, dtype=float)
    dev = np.abs(got - ref)
    tol_vec = np.maximum(STRONG_ABS, STRONG_REL * np.abs(ref))
    return {'max_abs_dev': float(dev.max()),
            'max_rel_dev': float((dev / np.maximum(np.abs(ref), 1e-12)).max()),
            'rule': rule,
            'passed': bool((dev <= tol_vec).all())}


def _paper_entry(vals_full: np.ndarray, vals_rounded: np.ndarray,
                 mean_ref: float, std_ref: float) -> dict:
    """Weak paper anchor (dual convention): aggregating at full precision or
    after rounding to stored precision; passing either one within +-0.005
    counts as a pass.

    mean/std are both computed over the 16 regions with ddof=0.
    """
    m_full, s_full = float(vals_full.mean()), float(vals_full.std())
    m_rnd, s_rnd = float(vals_rounded.mean()), float(vals_rounded.std())
    mean_ok = (abs(m_full - mean_ref) <= TOL_PAPER
               or abs(m_rnd - mean_ref) <= TOL_PAPER)
    std_ok = (abs(s_full - std_ref) <= TOL_PAPER
              or abs(s_rnd - std_ref) <= TOL_PAPER)
    return {'recomputed_mean': m_full, 'recomputed_mean_of_rounded': m_rnd,
            'recomputed_std_ddof0': s_full, 'recomputed_std_of_rounded_ddof0': s_rnd,
            'paper_mean': mean_ref, 'paper_std': std_ref,
            'mean_within_0.005': bool(mean_ok), 'std_within_0.005': bool(std_ok),
            'passed': bool(mean_ok and std_ok)}


# ════════════════════════════════════════════════════════════
# Main pipeline
# ════════════════════════════════════════════════════════════

def main():
    print('=' * 70)
    print('Additive correction full matrix (023)')
    print('=' * 70)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fold_of = load_fold_of()
    assignment_cache = {}
    gpm_fallback_groups = 0

    # values[label][metric][loc] = value (GNN arms are seed-averaged); per-seed values stored separately
    values = {label: {m: {} for m in METRICS} for label, *_ in ARMS}
    per_seed_values = {label: {s: {m: {} for m in METRICS} for s in SEEDS}
                       for label, base, *_ in ARMS if base == 'gnn'}
    # Same-source strong-anchor reference: direct evaluation of the corrected demand values 005 writes to disk
    same_source_ref = {label: {s: {m: {} for m in METRICS} for s in SEEDS}
                       for label in SAME_SOURCE_KEY_OF}
    # Agent-level demand deviation for the Uniform-additive degeneracy (idempotent-renormalization floating-point noise)
    uni_add_demand_dev = {sig: 0.0 for sig in SIGNALS}
    # Two conventions for the degeneracy precondition: ptp = the within-group
    # values are exactly identical (the strict-identity precondition); std =
    # the within-group standard deviation numpy actually computes --
    # floating-point summation of equal values can leave ~1e-16 relative
    # noise, so std can be a tiny nonzero number -> alpha is a tiny nonzero
    # number, but the resulting correction shift (~1e-14*base) is well
    # within TOL_DEGENERATION
    uni_base_group_ptp_max = 0.0
    uni_base_group_std_max = 0.0

    # ── 1. Compute all 21 arms, region by region ──
    for loc in ALL_LOCATIONS:
        print(f'\n=== {loc} ===')
        ctx = load_location_context(loc, assignment_cache)
        grid_gdf, region_sub = ctx['grid_gdf'], ctx['region_sub']

        # Static bases
        uni_base = compute_uniform_base(grid_gdf, region_sub)
        gpm_base, n_fb = compute_gpm_base(grid_gdf, region_sub, ctx['subs_sub'])
        gpm_fallback_groups += n_fb

        # Degeneracy-precondition check: within each ITL3 group, the Uniform
        # base values are exactly identical (ptp = 0); numpy std is recorded
        # separately (floating-point mean noise can make it a tiny nonzero
        # number, see the variable comment above)
        for itl3, group in grid_gdf.groupby('ITL3'):
            g_vals = uni_base[group.index]
            uni_base_group_ptp_max = max(
                uni_base_group_ptp_max, float(g_vals.max() - g_vals.min()))
            uni_base_group_std_max = max(uni_base_group_std_max, float(g_vals.std()))

        # Agent-level demand deviation for the degeneracy check (additive output vs. the base itself)
        for sig in SIGNALS:
            add_demand = scu.apply_additive_correction(
                uni_base, ctx['factors'][sig], grid_gdf, region_sub)
            uni_add_demand_dev[sig] = max(
                uni_add_demand_dev[sig], float(np.abs(add_demand - uni_base).max()))

        for base_name, base_arr in (('uniform', uni_base), ('gpm', gpm_base)):
            variants = seven_variants(ctx, base_arr)
            for label, base, form, sig in ARMS:
                if base != base_name:
                    continue
                for m in METRICS:
                    values[label][m][loc] = variants[(form, sig)][m]

        # GNN base (per seed, using the fold in which this region was the
        # test set; same out-of-sample convention as 017)
        for seed in SEEDS:
            fold_dir = fold_of[seed][loc]
            gd_path = (EXP0_DIR / f'seed_{seed}' / 'baseline' / fold_dir
                       / 'grid_demands' / f'{loc}_grid_demands.pickle')
            with open(gd_path, 'rb') as f:
                grid_demands = pickle.load(f)
            for key in ['gnn_demand'] + list(SAME_SOURCE_KEY_OF.values()):
                assert key in grid_demands, f'{loc} seed{seed}: grid_demands is missing {key}'
            gnn_base = np.asarray(grid_demands['gnn_demand'], dtype=float)
            assert len(gnn_base) == len(grid_gdf), \
                f'{loc} seed{seed}: grid_demands length does not match the number of grid cells'

            variants = seven_variants(ctx, gnn_base)
            for label, base, form, sig in ARMS:
                if base != 'gnn':
                    continue
                for m in METRICS:
                    per_seed_values[label][seed][m][loc] = variants[(form, sig)][m]

            # Same-source strong-anchor reference (4): direct evaluation of the corrected demand values 005 writes to disk
            for label, key in SAME_SOURCE_KEY_OF.items():
                ref_metrics = eval_demand(
                    ctx, np.asarray(grid_demands[key], dtype=float))
                for m in METRICS:
                    same_source_ref[label][seed][m][loc] = ref_metrics[m]

        # Seed-average the GNN arms (average across seeds first, then aggregate across regions)
        for label, base, form, sig in ARMS:
            if base != 'gnn':
                continue
            for m in METRICS:
                values[label][m][loc] = float(np.mean(
                    [per_seed_values[label][s][m][loc] for s in SEEDS]))

        print(f"  Uni={values['Uni']['rmse'][loc]:.2f} "
              f"GPMaddNP={values['GPMaddNP']['rmse'][loc]:.2f} "
              f"GNNaddP={values['GNNaddP']['rmse'][loc]:.2f} (RMSE)")

    def vec(label: str, metric: str) -> np.ndarray:
        """The 16-region vector for an arm (GNN arms are seed-averaged), ordered as in ALL_LOCATIONS."""
        return np.array([values[label][metric][loc] for loc in ALL_LOCATIONS])

    def vec_seed(label: str, seed: int, metric: str) -> np.ndarray:
        return np.array([per_seed_values[label][seed][metric][loc]
                         for loc in ALL_LOCATIONS])

    # ── 2. Anchor checks (must all pass before continuing; any failure raises and no new-arm output is written) ──
    print('\n---- Anchor checks (align before extending) ----')
    anchors = {'frozen_static': {}, 's5_recomputed': {}, 'paper_table3': {},
               'frozen_gnn_kfold': {}, 'same_source_gnn': {},
               'paper_mechanism': {}, 'not_anchored': []}

    # (1) Static arms against the frozen all_regions_{metric}.csv (agreement within stored precision)
    frozen = {m: pd.read_csv(STATIC_DIR / f'all_regions_{m}.csv', index_col=0)
              for m in METRICS}
    for label, row in FROZEN_ROW_OF.items():
        if row is None:
            anchors['not_anchored'].append({
                'arm': label, 'anchor': 'frozen_static',
                'reason': 'the corresponding row is not present in the frozen all_regions_{metric}.csv (a previously identified gap in the frozen output)'})
            continue
        for m in METRICS:
            ref = frozen[m].loc[row, ALL_LOCATIONS].values.astype(float)
            tol = TOL_FROZEN_2DP if m in ('rmse', 'mae') else TOL_FROZEN_4DP
            anchors['frozen_static'][f'{label}_{m}'] = _entry(
                np.abs(vec(label, m) - ref).max(), tol,
                f'|dev| <= half the smallest stored unit ({m} stored at {STORE_DP_STATIC[m]}dp)')

    # (2) GPM-family arms against the full-precision recomputation (a strong anchor; GPMpostP's only full-precision anchor)
    s5 = pd.read_csv(S5_RECOMPUTED_CSV)
    for label, arm_row in S5_ROW_OF.items():
        for m in METRICS:
            hit = s5[(s5['arm'] == arm_row) & (s5['metric'] == m)]
            assert len(hit) == 1, f'the full-precision recomputation table is missing {arm_row}/{m}'
            ref = hit[ALL_LOCATIONS].values[0].astype(float)
            anchors['s5_recomputed'][f'{label}_{m}'] = _strong_entry(
                vec(label, m), ref,
                f'|dev| <= max({STRONG_ABS}, {STRONG_REL}*|ref|) (the recomputation runs '
                f'the chained correction from 003, this script runs the combined-factor '
                f'renormalization from 017; the two are algebraically identical in real arithmetic)')

    # (3) Paper Table 3 weak anchor (8 static arms + 4 GNN multiplicative arms, dual convention +-0.005)
    for label, refs in PAPER_TABLE3.items():
        base = ARM_META[label][0]
        dp = STORE_DP_STATIC if base != 'gnn' else STORE_DP_GNN
        for m in METRICS:
            v = vec(label, m)
            if base == 'gnn':
                # Rounding convention: round each (seed, region) value to the
                # kfold CSV's 4dp precision, then seed-average
                v_rnd = np.mean([np.round(vec_seed(label, s, m), dp[m])
                                 for s in SEEDS], axis=0)
            else:
                v_rnd = np.round(v, dp[m])
            anchors['paper_table3'][f'{label}_{m}'] = _paper_entry(
                v, v_rnd, *refs[m])

    # (1') GNN-family arms against the frozen kfold_test_{metric}.csv (per seed x region, tolerance 5e-4)
    for m in METRICS:
        kfold = {s: pd.read_csv(EXP0_DIR / f'seed_{s}' / 'baseline'
                                / f'kfold_test_{m}.csv', index_col=0)
                 for s in SEEDS}
        for label, row in KFOLD_ROW_OF.items():
            max_dev = 0.0
            for s in SEEDS:
                ref = kfold[s].loc[row, ALL_LOCATIONS].values.astype(float)
                max_dev = max(max_dev, float(np.abs(vec_seed(label, s, m) - ref).max()))
            anchors['frozen_gnn_kfold'][f'{label}_{m}'] = _entry(
                max_dev, TOL_GNN_FROZEN,
                '|dev| <= 5e-4 (4dp storage + GPU re-inference drift on rerun; empirically up to 8.7e-5)')

    # (4) Same-source strong anchor for GNN multiplicative arms (direct evaluation of the corrected demand values 005 writes to disk)
    for label in SAME_SOURCE_KEY_OF:
        for m in METRICS:
            got = np.concatenate([vec_seed(label, s, m) for s in SEEDS])
            ref = np.concatenate([
                np.array([same_source_ref[label][s][m][loc] for loc in ALL_LOCATIONS])
                for s in SEEDS])
            anchors['same_source_gnn'][f'{label}_{m}'] = _strong_entry(
                got, ref,
                'multiplicative arms recomputed by this script from gnn_demand vs. direct '
                'evaluation of the corrected demand 005 writes into the pickle (same source, '
                'unaffected by GPU re-inference drift)')

    # (3') the 3 GNN additive arms against the paper's tab:mechanism_isolation (+-0.005)
    for label, refs in PAPER_MECHANISM.items():
        for m in METRICS:
            v = vec(label, m)
            v_rnd = np.mean([np.round(vec_seed(label, s, m), STORE_DP_GNN[m])
                             for s in SEEDS], axis=0)
            anchors['paper_mechanism'][f'{label}_{m}'] = _paper_entry(
                v, v_rnd, *refs[m])

    # UniP anchor-coverage disclosure (its only numerical anchor is the paper's Table 3)
    anchors['not_anchored'].append({
        'arm': 'UniP', 'anchor': 's5_recomputed',
        'reason': "the full-precision recomputation only covers the 4 GPM-family arms; "
                  "UniP's only numerical anchor is the paper's Table 3 weak anchor"})

    # Overall verdict: must all pass before continuing
    failed = []
    for cat in ('frozen_static', 's5_recomputed', 'paper_table3',
                'frozen_gnn_kfold', 'same_source_gnn', 'paper_mechanism'):
        for k, e in anchors[cat].items():
            if not e['passed']:
                failed.append(f'{cat}/{k}')
    n_checks = sum(len(anchors[c]) for c in
                   ('frozen_static', 's5_recomputed', 'paper_table3',
                    'frozen_gnn_kfold', 'same_source_gnn', 'paper_mechanism'))
    if failed:
        raise RuntimeError(f'Anchor check failed {len(failed)}/{n_checks}: {failed} -- '
                           f'alignment must pass before extending; the new-arm output was not written')
    print(f'  All {n_checks} anchor checks passed (frozen_static '
          f'{len(anchors["frozen_static"])} | s5 {len(anchors["s5_recomputed"])} | '
          f'paper_t3 {len(anchors["paper_table3"])} | gnn_kfold '
          f'{len(anchors["frozen_gnn_kfold"])} | same_source '
          f'{len(anchors["same_source_gnn"])} | mechanism '
          f'{len(anchors["paper_mechanism"])})')

    # ── 3. Uniform x additive degeneracy verification (structural, not a bug) ──
    degeneration = {
        'structural_degeneration': True,
        'affected_arms': DEGENERATE_ARMS,
        'reason': ("the additive correction scale alpha = base_std/offset_std (ddof=0) "
                   "is supplied entirely by the base allocation's own within-region "
                   "variance; for the Uniform base, within-ITL3-group base_std = 0, "
                   "so alpha = 0 and the correction reduces exactly to the baseline "
                   "(leaving only the floating-point noise of one idempotent "
                   "renormalization pass). This is a direct, expected consequence of "
                   "the additive scale being derived from the base allocation's own "
                   "variance; the arm is kept under its matched definition and reported "
                   "as-is."),
        'uniform_base_within_group_ptp_max': float(uni_base_group_ptp_max),
        'uniform_base_within_group_std_max': float(uni_base_group_std_max),
        'std_note': ('ptp = 0 proves the within-group values are exactly identical '
                     '(the strict-identity precondition); numpy std can be a tiny '
                     'nonzero number (~1e-16 relative) due to floating-point summation '
                     'noise in the mean of equal values, so alpha can be a tiny nonzero '
                     'number and the resulting correction shift ~1e-14*base -- the '
                     'degeneracy equality holds within the TOL_DEGENERATION tolerance'),
        'max_agent_demand_abs_dev_by_signal': {
            sig: float(uni_add_demand_dev[sig]) for sig in SIGNALS},
        'metric_level_max_abs_dev': {},
    }
    assert uni_base_group_ptp_max == 0.0, \
        'Uniform base within-group values are not exactly identical (ptp != 0) -- the degeneracy precondition is violated, check the base construction'
    deg_max = 0.0
    for label in DEGENERATE_ARMS:
        for m in METRICS:
            dev = float(np.abs(vec(label, m) - vec('Uni', m)).max())
            degeneration['metric_level_max_abs_dev'][f'{label}_{m}'] = dev
            deg_max = max(deg_max, dev)
    degeneration['all_within_tol'] = bool(deg_max <= TOL_DEGENERATION)
    degeneration['tol'] = TOL_DEGENERATION
    if not degeneration['all_within_tol']:
        raise RuntimeError(f'UniAdd degeneracy deviation {deg_max:.3e} exceeds tolerance '
                           f'{TOL_DEGENERATION} -- inconsistent with the structural argument, investigate')
    print(f'  Degeneracy verification: UniAdd* == Uni, max metric-level deviation {deg_max:.3e}'
          f' (max agent-level demand deviation '
          f'{max(uni_add_demand_dev.values()):.3e})')

    # ── 4. Write the full matrix to disk ──
    matrix_rows = []
    for label, base, form, sig in ARMS:
        for m in METRICS:
            v = vec(label, m)
            matrix_rows.append({
                'arm': label, 'base': base, 'form': form, 'signal': sig,
                'metric': m,
                'structural_degeneration': label in DEGENERATE_ARMS,
                'n_seeds': len(SEEDS) if base == 'gnn' else 0,
                **{loc: values[label][m][loc] for loc in ALL_LOCATIONS},
                'mean_16regions': float(v.mean()),
                'std_ddof0_16regions': float(v.std()),
            })
    full_matrix = pd.DataFrame(matrix_rows)
    full_matrix.to_csv(OUTPUT_DIR / 'full_matrix.csv', index=False)

    per_seed_rows = []
    for label, base, form, sig in ARMS:
        if base != 'gnn':
            continue
        for s in SEEDS:
            for m in METRICS:
                v = vec_seed(label, s, m)
                per_seed_rows.append({
                    'arm': label, 'base': base, 'form': form, 'signal': sig,
                    'seed': s, 'metric': m,
                    **{loc: per_seed_values[label][s][m][loc]
                       for loc in ALL_LOCATIONS},
                    'mean_16regions': float(v.mean()),
                    'std_ddof0_16regions': float(v.std()),
                })
    pd.DataFrame(per_seed_rows).to_csv(
        OUTPUT_DIR / 'full_matrix_per_seed.csv', index=False)
    print(f'  full_matrix: {len(full_matrix)} rows (21 arms x 3 metrics)'
          f'; per_seed: {len(per_seed_rows)} rows')

    # ── 5. New head-to-head comparisons (same statistical protocol as 031) ──
    comp_rows = []
    row_idx = 0
    for comp in NEW_COMPARISONS:
        la, lb, fam = comp['a'], comp['b'], comp['family']
        base_a, base_b = ARM_META[la][0], ARM_META[lb][0]
        is_gnn = 'gnn' in (base_a, base_b)
        seed_branch = 'seed_averaged' if is_gnn else 'no_seed'

        for m in METRICS:
            diffs = vec(la, m) - vec(lb, m)
            perm = exact_sign_flip_permutation(diffs)
            boot_seed = BOOT_SEED_BASE + row_idx
            boot = paired_region_bootstrap(diffs, B=B_BOOT, seed=boot_seed)

            row = {
                'comparison_id': comp['cid'],
                'comparison': f'{la} vs {lb}',
                'arm_a': la, 'arm_b': lb, 'metric': m,
                'family': fam,
                'holm_family': f'r12_{fam}_{m}',
                'seed_branch': seed_branch,
                'n_regions': perm['n'],
                'mean_diff': float(np.mean(diffs)),
                'protocol': 'exact_sign_flip_permutation',
                'perm_p': perm['p'],
                'perm_t_obs': perm['t_obs'],
                'perm_min_attainable_p': perm['min_attainable_p'],
                'holm_p': None,                  # filled in after the loop, per family
                'new_sig': None,
                'ci_lo': boot['ci_lo'], 'ci_hi': boot['ci_hi'],
                'boot_B': boot['B'], 'boot_seed': boot_seed,
            }

            # Per-seed transparency: for comparisons involving a GNN arm,
            # report per-seed permutation p-values without pooling
            if is_gnn:
                seed_ps = {}
                for s in SEEDS:
                    a_s = vec_seed(la, s, m) if base_a == 'gnn' else vec(la, m)
                    b_s = vec_seed(lb, s, m) if base_b == 'gnn' else vec(lb, m)
                    seed_ps[s] = exact_sign_flip_permutation(a_s - b_s)['p']
                for s in SEEDS:
                    row[f'perm_p_seed_{s}'] = seed_ps[s]
                row['cauchy_combined_p'] = cauchy_combination(
                    [seed_ps[s] for s in SEEDS])
                row['cauchy_note'] = 'for reference only, not the primary statistic (the primary statistic is the permutation p-value computed after seed-averaging)'
            else:
                for s in SEEDS:
                    row[f'perm_p_seed_{s}'] = None
                row['cauchy_combined_p'] = None
                row['cauchy_note'] = None

            comp_rows.append(row)
            row_idx += 1

    comps = pd.DataFrame(comp_rows)

    # Holm correction: one family per metric (same convention as 031); the
    # degeneracy-verification family is kept separate (an identity check,
    # not a hypothesis)
    for fam_name in comps['holm_family'].unique():
        mask = comps['holm_family'] == fam_name
        pmap = {int(r['comparison_id']): float(r['perm_p'])
                for _, r in comps[mask].iterrows()}
        adj = holm(pmap)
        comps.loc[mask, 'holm_p'] = comps.loc[mask, 'comparison_id'].map(adj)
    comps['holm_p'] = comps['holm_p'].astype(float)
    comps['new_sig'] = comps['holm_p'] < ALPHA
    comps['degeneration_check'] = comps['family'] == 'degeneration'
    comps.to_csv(OUTPUT_DIR / 'new_comparisons.csv', index=False)
    print(f'  new_comparisons: {len(comps)} rows ({len(NEW_COMPARISONS)} comparisons x 3 metrics)')

    # ── 6. anchor_report.json (anchor checks + degeneracy + protocol metadata + headline numbers) ──
    headline = {}
    for cid, name in ((1, 'GNNaddP_vs_GPMpostNP'), (8, 'GNNaddNP_vs_GNNpostNP'),
                      (4, 'GNNaddP_vs_GNN'), (14, 'GPMaddNP_vs_GPMpostNP')):
        sub = comps[comps['comparison_id'] == cid]
        headline[name] = {
            m: {'mean_diff': float(sub[sub['metric'] == m]['mean_diff'].iloc[0]),
                'perm_p': float(sub[sub['metric'] == m]['perm_p'].iloc[0]),
                'holm_p': float(sub[sub['metric'] == m]['holm_p'].iloc[0])}
            for m in METRICS}

    report = {
        'meta': {
            'script': '023_exp_r12_additive_full_matrix.py',
            'purpose': 'Additive correction full matrix (3 bases x {none, multiplicative N/P/NP, additive N/P/NP})',
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'germany_arm': 'removed (the Germany/results artifacts are missing both locally and in the backup)',
            'gpm_compute_demand_fallback_groups': gpm_fallback_groups,
            'region_order': ALL_LOCATIONS,
        },
        'protocol': {
            'main_test': 'exact_sign_flip_permutation',
            'n_enumerated': 2 ** N_REGIONS,
            'min_attainable_p': 2.0 / 2 ** N_REGIONS,
            'seed_handling': 'GNN arms are tested after seed-averaging; per-seed p-values are reported in full for transparency and not pooled',
            'ci': 'paired_region_bootstrap (outer region resampling only, percentile method, no bootstrap p-value reported)',
            'boot_B': B_BOOT,
            'holm_family': 'one family per metric (the main family covers the 14 genuine comparisons; the degeneration family is kept separate since identity checks should not be mixed with genuine hypotheses)',
            'alpha': ALPHA,
            'consistent_with': '031_exp_r216_robust_stats.py',
        },
        'anchor_summary': {
            'n_checks': n_checks,
            'n_failed': 0,
            'policy': 'align before extending: any anchor failure raises an error and the new-arm output is not written',
        },
        'anchors': anchors,
        'uniform_additive_degeneration': degeneration,
        'headline_numbers': headline,
    }
    with open(OUTPUT_DIR / 'anchor_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # ── 7. Console summary ──
    print('\n---- Summary ----')
    print('Full-matrix RMSE (16-region mean +/- ddof=0 std):')
    for label, base, form, sig in ARMS:
        v = vec(label, 'rmse')
        deg = '  [structurally degenerate, == Uni]' if label in DEGENERATE_ARMS else ''
        print(f'  {label:10s} {v.mean():6.2f} ± {v.std():4.2f}{deg}')
    h = headline['GNNaddP_vs_GPMpostNP']['rmse']
    print(f"\nHeadline: GNNaddP vs GPMpostNP dRMSE = {h['mean_diff']:+.3f}, "
          f"perm p = {h['perm_p']:.4g}, holm p = {h['holm_p']:.4g}")
    print(f'\nOutput directory: {OUTPUT_DIR}')
    print('Done.')


if __name__ == '__main__':
    main()
