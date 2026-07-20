# -*- coding: utf-8 -*-
"""
026 - Renormalization identity verification

Motivation: combining "multiplicative correction + per-ITL3
renormalization" could in principle introduce an opaque double-
normalization artifact. This experiment addresses that concern directly
using **algebraic identities**:

Identity 1 (renormalization is a per-ITL3 positive scalar)
    d_mult_i = D_r * (base_i*f_i) / sum_j(base_j*f_j)
    => d_mult_i / (base_i*f_i) = D_r / sum_j(base_j*f_j) = lambda_r
       (constant within the region, and > 0)
    => the relative ratio between any pair of agents within a region is
       unchanged by the multiplicative correction:
       d_mult_i / d_mult_j = (base_i*f_i) / (base_j*f_j)
    Verification: compute ratio_i = d_mult_i/(base_i*f_i) for every agent
    within each ITL3 region, and assert that its maximum relative
    deviation from lambda_r = D_r/raw_sum is at machine precision
    (constancy of the ratio is equivalent to every agent pair's ratio
    being preserved, without needing to enumerate all O(n^2) pairs).

Identity 2 (multiplicative correction + renormalization is equivalent to
a single softmax(logits + log f) pass)
    The GNN's weights w_sa come from a per-source softmax; from
    gnn_demand = w_sa * D_r we back-solve w_sa = gnn_demand / D_r
    (star-shaped graph: each agent belongs to exactly one ITL3 source,
    following the same write-out convention used in
    005_kfold_prior_training.py). Taking logits z_a = log(w_sa) (softmax
    is invariant to a constant shift of the logits, so any representative
    value can be used for the back-solve), softmax_r(z + log f) * D_r is
    componentwise equal to "multiply by f, then renormalize per ITL3."
    => the post-hoc correction is not a bolted-on hack; it is equivalent
    to injecting log f once as a prior added to the softmax logits --
    the renormalization introduces no additional degrees of freedom.

Decomposition 3 (the no-renorm arm's RMSE increase = a scale term + a
shape term)
    d_nr_i = base_i*f_i = c_r * d_mult_i, where c_r = sum_j(base_j*f_j)/D_r.
    That is, no-renorm and mult+renorm have **exactly the same shape**
    within a region, differing only by a per-region positive scalar c_r
    (the region-level total mismatch S_r - D_r). Consequently:
      - the shape term = d_nr - c_r*d_mult is identically 0 by exact
        algebra (in floating point it is ~1e-16 relative; checked at a
        < 1e-12 tolerance rather than an exact zero);
      - the entire RMSE increase of the no-renorm arm is attributable to
        the scale term (the region-level total mismatch), verified with
        a "scale-only counterfactual" (multiply d_mult by c_r per region,
        then aggregate and evaluate): RMSE(scale-only) ~ RMSE(no-renorm),
        with delta_shape ~ 0.

Preconditions checked (labeled P6-1 through P6-4 below, matching the
precondition keys in the output; all recorded as expected/actual/status):
    P6-1  min(f) > 0 -- the correction factor is strictly positive for
          every agent (the RCI mask only affects how epsilon/median are
          computed and does not affect the factor's domain; "non-RCI =
          epsilon" is part of how the prior q is constructed and is
          unrelated to the post-hoc correction factor -- do not conflate
          the two);
    P6-2  renorm domain == softmax domain -- star-shaped-graph uniqueness
          is verified from grid_gdf's ITL3 grouping (each agent belongs
          to exactly one group, and region_sub's ITL3 values are unique),
          and the within-group sum of the back-solved weights
          Sum(w) ~ 1 confirms that the stored demand values do come from
          a per-ITL3 softmax, i.e. the renormalization grouping coincides
          with the softmax normalization grouping. Pass criterion:
          |Sum(w)-1|/(n*eps32) < 1 (a hard upper bound from the float32
          sequential-summation error model; the measured maximum ratio is
          0.045, whereas a genuine domain mismatch would produce an O(1)
          deviation -- more than 3 orders of magnitude apart, so the
          check has ample discriminating power);
    P6-3  the uniform-fallback branch (raw_sum <= 0 -> total/len, the
          same structure used in 017 and in the proximity_corrector
          module) fires 0 times;
    P6-4  ITL3 coverage = 100% (no agent is skipped due to
          `itl3 not in region_info`).

Data source (read-only): the 16 regions' grid_demands pickles from the
exp0 output at seed_42/baseline/fold1 (base column = gnn_demand). This
experiment is a purely deterministic algebraic verification with no
random source (the numpy seed is still fixed and recorded in the
metadata for consistency). Runs entirely on CPU.

Factor arms: N (ntl_factor), P (prox_factor), NP (ntl x prox) --
covering all of the paper's multiplicative correction arms (017's
standard_multiplicative_{N,P,NP} and H1a_no_renorm_{N,P,NP}).

Output: results/exp_r27/identity_check.json
      (maximum deviation under each convention, the RMSE decomposition
      table, and the precondition expected/actual/status records)

Usage:
    python 026_exp_r27_renorm_identity.py
    python 026_exp_r27_renorm_identity.py --seed 42 --config baseline --fold fold1
"""

import argparse
import json
import pickle
import platform
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Windows console default codepage (cp1252) cannot encode the Unicode
# symbols used below -- force UTF-8 (does not affect file output)
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
OUTPUT_DIR = SCRIPT_DIR / 'results' / 'exp_r27'

ALL_LOCATIONS = scu.ALL_LOCATIONS          # 16 study regions (same as in 017)
BASE_COL = 'gnn_demand'                    # base = uncorrected GNN demand

# ── Tolerance specification (matches the preconditions checked above; consistent with tests/test_r27.py) ──
TOL_CONSERVATION_REL = 1e-12     # conservation: |sum d_mult - D_r|/D_r
TOL_ID1_REL = 1e-12              # identity 1: maximum relative deviation of the ratio constant (pure float64 path)
TOL_ID2_REL_LARGE = 1e-6         # identity 2: relative tolerance for large components (back-solved through the float32 storage pipeline)
TOL_ID2_ABS_FRAC = 1e-12         # identity 2: absolute tolerance for small components (x D_r)
LARGE_COMPONENT_FRAC = 1e-12     # large/small component threshold: components > 1e-12*D_r use the relative tolerance
TOL_SHAPE_REL = 1e-12            # decomposition 3: relative tolerance for the shape term (not an exact zero)
EPS32 = float(np.finfo(np.float32).eps)   # ~ 1.19e-7
TOL_WSUM_RATIO = 1.0             # P6-2 confirmation: tolerance on |sum(w) - 1| / (n*eps32).
                                 # Error model: the scatter-softmax denominator is
                                 # accumulated element-by-element in float32, giving a
                                 # hard upper bound on the normalization error of
                                 # ~n*eps32 (the worst case for sequential summation).
                                 # Empirically, the ratio for all 85 ITL3 groups falls
                                 # between 0.007 (median) and 0.045 (max, TLC14 with
                                 # n=37811), and the absolute deviation grows linearly
                                 # with n -- a pure floating-point accumulation
                                 # signature. A genuine renorm-domain != softmax-domain
                                 # mismatch would produce an O(1) deviation, more than
                                 # 3 orders of magnitude apart, so the check has ample
                                 # discriminating power.
                                 # (Note: a flat constant threshold is not used, since
                                 #   group sizes span 3 orders of magnitude -- a flat
                                 #   threshold would false-positive on large groups and
                                 #   have no discriminating power on small groups.)

RNG_SEED = 20260714              # this experiment has no random source; the seed is still fixed and recorded for consistency


def _status(ok: bool) -> str:
    """Conclusion fields are generated by an explicit numeric rule rather than written by hand."""
    return 'ok' if ok else 'fail'


# ════════════════════════════════════════════════════════════
# Core identity computations (per region x per factor arm)
# ════════════════════════════════════════════════════════════

def softmax_path_reconstruction(w: np.ndarray, factor: np.ndarray,
                                itl3_groups: list, region_demand: dict) -> np.ndarray:
    """Reconstruction path for identity 2: softmax(log w + log f) * D_r,
    normalized independently within each ITL3 group.

    Args:
        w: back-solved weight array (all agents, w_sa = gnn_demand / D_r).
        factor: correction factor array (all agents; already asserted
            strictly positive).
        itl3_groups: [(itl3, position index array), ...] (grid_gdf
            RangeIndex semantics).
        region_demand: {itl3: D_r}.

    Note: log/exp use float64; the group's max logit is subtracted to
    prevent overflow (standard log-sum-exp stabilization; softmax's
    shift-invariance means this does not change the mathematical value).
    If w=0 (possible under float32 underflow), log gives -inf and exp
    brings it back to 0, so both paths agree at 0 and the identity still
    holds.
    """
    recon = np.zeros_like(w, dtype=float)
    for itl3, idx in itl3_groups:
        with np.errstate(divide='ignore'):
            logits = np.log(w[idx]) + np.log(factor[idx])
        m = np.max(logits)
        e = np.exp(logits - m)
        recon[idx] = region_demand[itl3] * e / e.sum()
    return recon


def check_identities_one_arm(base: np.ndarray, w: np.ndarray, factor: np.ndarray,
                             itl3_groups: list, region_demand: dict) -> dict:
    """Verify identities 1 and 2 and the grid-level part of decomposition 3
    for a single (region x factor arm) combination.

    Returns the maximum per-item deviations plus the intermediate vectors
    needed for the no-renorm decomposition.
    """
    # ── multiplicative correction + renormalization (the core two lines
    # from 017's existing logic, inlined here so we can also capture raw_sum) ──
    d_mult = np.zeros_like(base, dtype=float)
    d_nr = base * factor                       # no-renorm arm (matches 017's H1a arm exactly)
    uniform_fallback_count = 0                 # P6-3
    lambda_min = np.inf                        # 1: minimum of the renormalization scalar lambda_r (must be > 0)
    id1_spread_max = 0.0                       # 1: maximum relative deviation of the ratio constant
    conservation_dev_max = 0.0                 # conservation: |sum d_mult - D_r|/D_r
    conservation_dev_recon_max = 0.0           # conservation (reconstruction path)
    id2_rel_max_large = 0.0                    # 2: maximum relative deviation for large components
    id2_abs_frac_max_small = 0.0               # 2: maximum absolute deviation / D_r for small components
    n_large_total = 0
    n_small_total = 0
    shape_rel_max = 0.0                        # 3: maximum relative deviation of the shape term
    scale_factors = {}                         # 3: {itl3: c_r}
    scale_mismatch_mva = {}                    # 3: {itl3: S_r - D_r}

    for itl3, idx in itl3_groups:
        D_r = region_demand[itl3]
        raw = base[idx] * factor[idx]
        raw_sum = raw.sum()
        if raw_sum > 0:
            d_mult[idx] = D_r * raw / raw_sum
        else:
            # 017's uniform-fallback branch -- the precondition requires this to fire 0 times
            d_mult[idx] = D_r / len(idx)
            uniform_fallback_count += 1
            continue    # the identities do not apply in the fallback branch (the precondition already fails if the count is nonzero)

        # ── identity 1: ratio_i = d_mult_i/(base_i*f_i) is identically lambda_r = D_r/raw_sum ──
        lam = D_r / raw_sum
        lambda_min = min(lambda_min, lam)
        pos = raw > 0                          # true whenever base>0 and f>0; zero components are excluded since 0/0 is undefined
        if pos.any():
            ratio = d_mult[idx][pos] / raw[pos]
            id1_spread_max = max(id1_spread_max,
                                 float(np.abs(ratio - lam).max() / lam))

        # ── conservation (region total after renormalization equals D_r) ──
        conservation_dev_max = max(conservation_dev_max,
                                   float(abs(d_mult[idx].sum() - D_r) / D_r))

        # ── decomposition 3, grid level: shape term = d_nr - c_r*d_mult (algebraically identical to 0) ──
        c_r = raw_sum / D_r
        scale_factors[itl3] = float(c_r)
        scale_mismatch_mva[itl3] = float(raw_sum - D_r)
        shape_resid = d_nr[idx] - c_r * d_mult[idx]
        denom = float(np.abs(d_nr[idx]).max())
        if denom > 0:
            shape_rel_max = max(shape_rel_max,
                                float(np.abs(shape_resid).max() / denom))

    # ── identity 2: componentwise comparison of softmax(log w + log f)*D_r against d_mult, under both tolerance conventions ──
    d_recon = softmax_path_reconstruction(w, factor, itl3_groups, region_demand)
    for itl3, idx in itl3_groups:
        D_r = region_demand[itl3]
        diff = np.abs(d_recon[idx] - d_mult[idx])
        large = d_mult[idx] > LARGE_COMPONENT_FRAC * D_r
        n_large_total += int(large.sum())
        n_small_total += int((~large).sum())
        if large.any():
            id2_rel_max_large = max(id2_rel_max_large,
                                    float((diff[large] / d_mult[idx][large]).max()))
        if (~large).any():
            id2_abs_frac_max_small = max(id2_abs_frac_max_small,
                                         float(diff[~large].max() / D_r))
        conservation_dev_recon_max = max(conservation_dev_recon_max,
                                         float(abs(d_recon[idx].sum() - D_r) / D_r))

    return {
        'uniform_fallback_count': uniform_fallback_count,
        'lambda_min': float(lambda_min),
        'id1_ratio_spread_rel_max': id1_spread_max,
        'conservation_rel_max': conservation_dev_max,
        'conservation_recon_rel_max': conservation_dev_recon_max,
        'id2_rel_max_large': id2_rel_max_large,
        'id2_abs_frac_max_small': id2_abs_frac_max_small,
        'id2_n_large': n_large_total,
        'id2_n_small': n_small_total,
        'shape_rel_max': shape_rel_max,
        'scale_factors': scale_factors,
        'scale_mismatch_mva': scale_mismatch_mva,
        # vectors needed for the decomposition 3 RMSE counterfactual (not written to disk, only returned to the caller)
        '_d_mult': d_mult,
        '_d_nr': d_nr,
    }


# ════════════════════════════════════════════════════════════
# Main pipeline
# ════════════════════════════════════════════════════════════

def run(seed: int, config: str, fold: str) -> dict:
    """Verify the identities for all 16 regions, one at a time, and return
    the full results dict."""
    np.random.seed(RNG_SEED)   # this experiment makes no random calls; the seed is fixed and recorded for consistency

    fold_dir = EXP0_DIR / f'seed_{seed}' / config / fold
    gd_dir = fold_dir / 'grid_demands'
    if not gd_dir.exists():
        raise FileNotFoundError(f'Frozen output directory not found: {gd_dir}')

    arms = ['N', 'P', 'NP']

    # ── Precondition accumulators ──
    factor_min = {'N': np.inf, 'P': np.inf, 'NP': np.inf}       # P6-1
    star_violations = 0                                          # P6-2a: missing ITL3 / groups not covering all agents
    region_itl3_dup = 0                                          # P6-2a: duplicate ITL3 values in region_sub
    wsum_dev_max = 0.0                                           # P6-2b: max |sum(w) - 1| (absolute value, for reporting)
    wsum_ratio_max = 0.0                                         # P6-2b: max |sum(w) - 1|/(n*eps32) (used for the pass/fail decision)
    uniform_fallback_total = 0                                   # P6-3
    agents_total = 0                                             # P6-4
    agents_covered = 0                                           # P6-4
    zero_weight_count = 0                                        # diagnostic only (not a formal P6 precondition)

    # ── Identity-deviation accumulators (max value per convention, per arm before the overall max) ──
    agg = {arm: {
        'id1_ratio_spread_rel_max': 0.0,
        'lambda_min': np.inf,
        'conservation_rel_max': 0.0,
        'conservation_recon_rel_max': 0.0,
        'id2_rel_max_large': 0.0,
        'id2_abs_frac_max_small': 0.0,
        'id2_n_large': 0,
        'id2_n_small': 0,
        'shape_rel_max': 0.0,
    } for arm in arms}

    per_location = {}
    decomposition_rows = []
    scale_factor_dump = {}
    voronoi_cache = {}

    for loc in ALL_LOCATIONS:
        print(f'[{loc}] Loading frozen output and factors ...')
        with open(gd_dir / f'{loc}_grid_demands.pickle', 'rb') as f:
            grid_demands = pickle.load(f)
        base = np.asarray(grid_demands[BASE_COL], dtype=float)

        grid_gdf, region_sub, subs_sub, ntl_values = scu.load_grid_and_subs(loc)
        prox_scores = scu.compute_prox_scores(grid_gdf, subs_sub)
        ntl_factor, prox_factor = scu.compute_factors(grid_gdf, ntl_values, prox_scores)
        factors = {'N': ntl_factor, 'P': prox_factor, 'NP': ntl_factor * prox_factor}

        # ── P6-2a: star-shaped-graph uniqueness (verified from grid_gdf's ITL3 grouping) ──
        # Every agent has a non-null ITL3 => groupby partitions all agents into
        # disjoint groups, i.e. each agent belongs to exactly one renormalization
        # group; region_sub's ITL3 values are unique => the D_r lookup is unambiguous.
        n_agents = len(grid_gdf)
        star_violations += int(grid_gdf['ITL3'].isna().sum())
        region_itl3_dup += int(region_sub['ITL3'].duplicated().sum())
        itl3_groups_all = [(itl3, np.asarray(g.index))
                           for itl3, g in grid_gdf.groupby('ITL3')]
        grouped_size = sum(len(idx) for _, idx in itl3_groups_all)
        star_violations += int(n_agents - grouped_size)   # number of agents not covered by any group

        # ── P6-4: ITL3 coverage (no agent skipped due to itl3 not in region_info) ──
        region_info = region_sub.set_index('ITL3')
        covered_mask = grid_gdf['ITL3'].isin(region_info.index)
        agents_total += n_agents
        agents_covered += int(covered_mask.sum())
        itl3_groups = [(itl3, idx) for itl3, idx in itl3_groups_all
                       if itl3 in region_info.index]
        region_demand = {itl3: float(region_info.loc[itl3, 'Demand (MVA)'])
                         for itl3, _ in itl3_groups}

        # ── Back-solve w_sa = gnn_demand / D_r (star-shaped graph: each agent belongs to exactly one source) ──
        w = np.zeros_like(base)
        for itl3, idx in itl3_groups:
            w[idx] = base[idx] / region_demand[itl3]
            # P6-2b: within-group sum(w) ~ 1 => the stored demand comes from a
            # per-ITL3 softmax, and the renormalization grouping coincides with
            # the softmax normalization grouping. The pass/fail decision uses
            # the normalized ratio |sum(w)-1|/(n*eps32) from the error model
            # (see the TOL_WSUM_RATIO comment above).
            dev = float(abs(w[idx].sum() - 1.0))
            wsum_dev_max = max(wsum_dev_max, dev)
            wsum_ratio_max = max(wsum_ratio_max, dev / (len(idx) * EPS32))
        zero_weight_count += int((w == 0).sum())

        # ── P6-1 min(f) > 0 ──
        for arm in arms:
            factor_min[arm] = min(factor_min[arm], float(factors[arm].min()))

        # ── Voronoi assignment (computed once per region, reused for aggregation -- the shared module's two-stage pattern) ──
        assignment = scu.compute_voronoi_assignment(
            grid_gdf, subs_sub, cache=voronoi_cache, cache_key=loc)

        def _rmse(demand_arr: np.ndarray) -> float:
            subs_result = scu.aggregate_by_assignment(subs_sub, assignment, demand_arr)
            return float(scu.evaluate_allocation(subs_result)['rmse'])

        rmse_baseline = _rmse(base)

        loc_result = {'n_agents': n_agents, 'n_itl3': len(itl3_groups),
                      'rmse_baseline': rmse_baseline, 'arms': {}}
        scale_factor_dump[loc] = {}

        for arm in arms:
            res = check_identities_one_arm(base, w, factors[arm],
                                           itl3_groups, region_demand)
            uniform_fallback_total += res['uniform_fallback_count']

            # ── Decomposition 3 RMSE table: mult / no-renorm / scale-only counterfactual ──
            d_mult = res.pop('_d_mult')
            d_nr = res.pop('_d_nr')
            d_scale_only = d_mult.copy()
            for itl3, idx in itl3_groups:
                d_scale_only[idx] *= res['scale_factors'][itl3]

            rmse_mult = _rmse(d_mult)
            rmse_nr = _rmse(d_nr)
            rmse_scale_only = _rmse(d_scale_only)
            delta_total = rmse_nr - rmse_mult
            delta_scale = rmse_scale_only - rmse_mult
            delta_shape = rmse_nr - rmse_scale_only
            decomposition_rows.append({
                'location': loc, 'arm': arm,
                'rmse_baseline': rmse_baseline,
                'rmse_mult_renorm': rmse_mult,
                'rmse_no_renorm': rmse_nr,
                'rmse_scale_only': rmse_scale_only,
                'delta_total': delta_total,
                'delta_scale': delta_scale,
                'delta_shape': delta_shape,
                'delta_shape_rel': (abs(delta_shape) / rmse_mult
                                    if rmse_mult > 0 else 0.0),
            })

            # ── Aggregate the maximum deviation under each convention ──
            a = agg[arm]
            a['id1_ratio_spread_rel_max'] = max(a['id1_ratio_spread_rel_max'],
                                                res['id1_ratio_spread_rel_max'])
            a['lambda_min'] = min(a['lambda_min'], res['lambda_min'])
            a['conservation_rel_max'] = max(a['conservation_rel_max'],
                                            res['conservation_rel_max'])
            a['conservation_recon_rel_max'] = max(a['conservation_recon_rel_max'],
                                                  res['conservation_recon_rel_max'])
            a['id2_rel_max_large'] = max(a['id2_rel_max_large'],
                                         res['id2_rel_max_large'])
            a['id2_abs_frac_max_small'] = max(a['id2_abs_frac_max_small'],
                                              res['id2_abs_frac_max_small'])
            a['id2_n_large'] += res['id2_n_large']
            a['id2_n_small'] += res['id2_n_small']
            a['shape_rel_max'] = max(a['shape_rel_max'], res['shape_rel_max'])

            scale_factor_dump[loc][arm] = res['scale_factors']
            loc_result['arms'][arm] = {
                k: v for k, v in res.items()
                if k not in ('scale_factors', 'scale_mismatch_mva')
            }
            loc_result['arms'][arm]['scale_factor_min'] = min(res['scale_factors'].values())
            loc_result['arms'][arm]['scale_factor_max'] = max(res['scale_factors'].values())
            loc_result['arms'][arm]['scale_mismatch_mva_total'] = float(
                sum(res['scale_mismatch_mva'].values()))

        per_location[loc] = loc_result
        np_arm = loc_result['arms']['NP']
        print(f'  1 ratio deviation={np_arm["id1_ratio_spread_rel_max"]:.3e}  '
              f'2 large-component relative deviation={np_arm["id2_rel_max_large"]:.3e}  '
              f'3 shape term={np_arm["shape_rel_max"]:.3e}  (NP arm)')

    # ── Preconditions (expected/actual/status) ──
    coverage = agents_covered / agents_total if agents_total > 0 else 0.0
    preconditions = {
        'P6_1_min_factor_positive': {
            'description': 'The correction factor is strictly positive for every agent (min(f) > 0, checked separately for each of the 3 arms)',
            'expected': '> 0',
            'actual': {arm: factor_min[arm] for arm in arms},
            'status': _status(all(factor_min[arm] > 0 for arm in arms)),
        },
        'P6_2_renorm_domain_equals_softmax_domain': {
            'description': ('renorm domain = softmax domain: star-shaped-graph '
                            'uniqueness (each agent belongs to exactly one ITL3 '
                            'group; region_sub has no duplicate ITL3 values) + '
                            'the within-group sum of the back-solved weights '
                            'sum(w) ~ 1 (pass criterion = |sum(w)-1|/(n*eps32) < 1, '
                            'the hard upper bound from the float32 sequential-'
                            'summation error model; a genuine domain mismatch '
                            'would produce an O(1) deviation)'),
            'expected': {'star_violations': 0, 'region_itl3_dup': 0,
                         'wsum_dev_over_n_eps32_max': f'< {TOL_WSUM_RATIO:g}'},
            'actual': {'star_violations': star_violations,
                       'region_itl3_dup': region_itl3_dup,
                       'wsum_dev_over_n_eps32_max': wsum_ratio_max,
                       'wsum_dev_max_abs': wsum_dev_max},
            'status': _status(star_violations == 0 and region_itl3_dup == 0
                              and wsum_ratio_max < TOL_WSUM_RATIO),
        },
        'P6_3_uniform_fallback_count': {
            'description': 'Number of times the uniform-fallback branch (raw_sum <= 0) fires (across all 16 regions x 3 arms)',
            'expected': 0,
            'actual': uniform_fallback_total,
            'status': _status(uniform_fallback_total == 0),
        },
        'P6_4_itl3_coverage': {
            'description': "ITL3 coverage (every agent's ITL3 is present in the region gpkg, none skipped)",
            'expected': 1.0,
            'actual': coverage,
            'status': _status(coverage == 1.0),
        },
    }

    # ── Maximum deviation under each identity convention + status (conclusion fields generated by explicit numeric rules) ──
    for arm in arms:
        a = agg[arm]
        a['id1_status'] = _status(a['id1_ratio_spread_rel_max'] < TOL_ID1_REL
                                  and a['lambda_min'] > 0)
        a['conservation_status'] = _status(
            a['conservation_rel_max'] < TOL_CONSERVATION_REL
            and a['conservation_recon_rel_max'] < TOL_CONSERVATION_REL)
        a['id2_status'] = _status(a['id2_rel_max_large'] < TOL_ID2_REL_LARGE
                                  and a['id2_abs_frac_max_small'] < TOL_ID2_ABS_FRAC)
        a['shape_status'] = _status(a['shape_rel_max'] < TOL_SHAPE_REL)

    overall = {
        'id1_ratio_spread_rel_max': max(agg[a]['id1_ratio_spread_rel_max'] for a in arms),
        'conservation_rel_max': max(agg[a]['conservation_rel_max'] for a in arms),
        'conservation_recon_rel_max': max(agg[a]['conservation_recon_rel_max'] for a in arms),
        'id2_rel_max_large': max(agg[a]['id2_rel_max_large'] for a in arms),
        'id2_abs_frac_max_small': max(agg[a]['id2_abs_frac_max_small'] for a in arms),
        'shape_rel_max': max(agg[a]['shape_rel_max'] for a in arms),
        'delta_shape_rel_max': max(r['delta_shape_rel'] for r in decomposition_rows),
        'all_preconditions_ok': all(v['status'] == 'ok' for v in preconditions.values()),
        'all_identities_ok': all(
            agg[a][k] == 'ok' for a in arms
            for k in ('id1_status', 'conservation_status', 'id2_status', 'shape_status')),
    }

    results = {
        'meta': {
            'script': '026_exp_r27_renorm_identity.py',
            'plan_step': 'renormalization identity verification',
            'source': str(gd_dir.relative_to(SCRIPT_DIR)),
            'seed': seed, 'config': config, 'fold': fold,
            'base_col': BASE_COL,
            'arms': arms,
            'n_locations': len(ALL_LOCATIONS),
            'rng_seed': RNG_SEED,
            'deterministic': True,
            'python': platform.python_version(),
            'numpy': np.__version__,
            'timestamp': datetime.now().isoformat(timespec='seconds'),
        },
        'tolerances': {
            'conservation_rel': TOL_CONSERVATION_REL,
            'id1_rel': TOL_ID1_REL,
            'id2_rel_large': TOL_ID2_REL_LARGE,
            'id2_abs_frac_small': TOL_ID2_ABS_FRAC,
            'large_component_frac': LARGE_COMPONENT_FRAC,
            'shape_rel': TOL_SHAPE_REL,
            'wsum_dev_over_n_eps32': TOL_WSUM_RATIO,
        },
        'preconditions': preconditions,
        'diagnostics': {
            'zero_weight_count': zero_weight_count,
            'zero_weight_note': ('the count of back-solved weights that are '
                                 'exactly zero (a product of float32 underflow). '
                                 'Zero weights give 0 along both computation '
                                 'paths and do not break the identity; this is '
                                 'not a formal precondition, provided only as a '
                                 'cross-reference for underflow diagnostics.'),
        },
        'max_deviations': {arm: agg[arm] for arm in arms},
        'overall': overall,
        'decomposition_table': decomposition_rows,
        'per_location': per_location,
        'scale_factors_per_itl3': scale_factor_dump,
    }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description='Renormalization identity verification experiment')
    parser.add_argument('--seed', type=int, default=42, help='exp0 seed (default 42)')
    parser.add_argument('--config', default='baseline', help='exp0 config (default baseline)')
    parser.add_argument('--fold', default='fold1', help='exp0 fold directory name (default fold1)')
    args = parser.parse_args()

    results = run(args.seed, args.config, args.fold)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / 'identity_check.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    ov = results['overall']
    print('\n' + '=' * 68)
    print('Renormalization identity verification -- overview (max deviation across all arms and regions)')
    print('=' * 68)
    print(f"1  max relative deviation of ratio constant     : {ov['id1_ratio_spread_rel_max']:.3e}  (< {TOL_ID1_REL:g})")
    print(f"C  max relative conservation deviation           : {ov['conservation_rel_max']:.3e}  (< {TOL_CONSERVATION_REL:g})")
    print(f"C  conservation (reconstruction path)             : {ov['conservation_recon_rel_max']:.3e}  (< {TOL_CONSERVATION_REL:g})")
    print(f"2  max relative deviation, large components      : {ov['id2_rel_max_large']:.3e}  (< {TOL_ID2_REL_LARGE:g})")
    print(f"2  max absolute deviation/D_r, small components  : {ov['id2_abs_frac_max_small']:.3e}  (< {TOL_ID2_ABS_FRAC:g})")
    print(f"3  max relative deviation of shape term           : {ov['shape_rel_max']:.3e}  (< {TOL_SHAPE_REL:g})")
    print(f"3  RMSE decomposition, relative delta_shape       : {ov['delta_shape_rel_max']:.3e}")
    print(f"all preconditions ok                              : {ov['all_preconditions_ok']}")
    print(f"all identities ok                                 : {ov['all_identities_ok']}")
    print(f'\nResults written to {out_path}')


if __name__ == '__main__':
    main()
