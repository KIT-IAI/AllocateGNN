# -*- coding: utf-8 -*-
"""
Joint identifiability and training-evaluation misalignment analysis via level-set sampling.

Motivation:
    This experiment addresses two related questions about the land-use
    reconstruction loss used to train the spatial allocation model:
    (1) many different spatial allocations can reproduce the same
        aggregate land-use mixture, so it is worth checking how identifiable
        w_sa actually is given only that aggregate constraint;
    (2) a redistribution that barely changes the land-use reconstruction
        loss (L_landuse) could in principle substantially change the
        downstream substation RMSE.
This experiment turns both questions into measurable quantities: it samples
alternative allocations uniformly on the **exact level set** of the land-use
reconstruction loss, and quantifies (a) the size of the solution space under
the aggregation constraint and where the learned solution falls within it
(identifiability), and (b) the degrees of freedom in substation RMSE at
fixed reconstruction loss (evaluation sensitivity). This is a pure CPU
post-processing analysis with no retraining involved.

═══ Loss implementation verification (the constraint matrix must match the actual loss term-for-term) ═══
Verified against LandusePredictionLoss
(SpatialAllocation/GNN/Layer/LossFunction/LossFunction.py) and
_build_landuse_matrices (GraphBuilder.py):

  1. The reconstruction target is a **categorical one-hot aggregate**, not a
     continuous land-use proportion vector: each agent has a single
     category landuse = argmax(lu_*_prop) (derived during graph
     construction), and mapping_matrix has exactly one 1 per edge (column
     = s_idx * n_lu + lu_idx, verified per region by this script).
     The predicted mixture_c(s) = sum_{a in s, cat(a)=c} w_sa (i.e. the
     summed category mass), followed by row normalisation.
  2. The aggregation domain is the **ITL3 source** (a star graph: each
     agent has exactly one edge to its ITL3 source, verified per region by
     this script via bincount==1).
  3. The target vector comes from row-normalising the five {cat}_percent
     columns of ITL3_region.gpkg (landuse_ratio; verified against float32
     storage tolerance by this script).
  4. **No GVA component**: although the region gpkg also has *_gva columns,
     the loss only reads *_percent; the baseline configuration's objective
     is {'landuse_prediction_loss': 1.0} (read directly from the training
     script's CONFIG_MAP via importlib and logged by this script).
  5. The loss is invariant to an overall rescaling of w (the predicted
     mixture is row-normalised internally), so locating the level set using
     a normalised representative (sum(w)=1) does not change the loss value.

═══ Two implementation notes that follow from the above (also recorded in deviations_registry) ═══
  D1. The constraint matrix A = [5 category one-hot indicator rows; an
      all-ones row] (6 rows as originally specified), but the all-ones row
      equals the sum of the five indicator rows, so **rank(A) = number of
      categories present <= 5, not 6**; the nullspace dimension is
      n - rank(A) (verified directly by computing the rank).
  D2. The one-hot structure means the level set is a **product of
      per-category rescaled simplices** (mass is conserved within each
      category block and blocks are independent), which admits an
      **exact i.i.d. uniform sampler** (per-block Dirichlet(1,...,1)
      scaling). This experiment therefore uses dirichlet_exact as the
      primary sampler (provably uniform, no mixing-time risk); hit-and-run
      sampling is also retained in full as a cross-check, with the
      agreement between the two distributions recorded.
      (In high dimensions, an MCMC chain with fewer steps than dimensions
      would necessarily stay confined to a lower-dimensional slice and
      fail to cover the level set; the exact sampler is immune to this
      issue and is a strictly stronger implementation, not a shortcut.)

═══ Level-set definition (per ITL3) ═══
    P(m) = { w in R^n : w >= 0, sum(w) = 1, sum_{cat(a)=c} w_a = m_c for all c }
    m = the category mass vector of the learned solution (recovered by
    w_sa = gnn_demand / D_r, then normalised).
    Every point in P(m) produces the **exact same** predicted mixture as
    the learned solution, so the reconstruction loss is exactly equal
    across the whole level set. Since the baseline configuration's
    objective consists only of landuse_prediction_loss, this level set is
    a complete characterisation of the baseline training objective's
    solution space (a key advantage of this experimental design: no other
    loss term further constrains the solution).

═══ Data (all read-only) ═══
    Frozen baseline-configuration artifacts across 3 seeds; for each
    seed/region, the grid_demands from the fold in which that region was
    held out as the test set (16 regions x 3 seeds = 48 units);
    the cached graph file cached_graphs.pickle (land-use supervision
    matrices verified term-for-term, plus the landuse column);
    a control basis = per-ITL3 Uniform, sampled the same way on its own
    mixture's level set.

Outputs (results/exp_r21_r22_levelset/):
    levelset_samples_summary.csv   distribution statistics and percentiles
                                    per (seed x region x basis x sampler)
    nullspace_dims.csv             rank / nullspace dimension per (region x ITL3)
    identifiability_metrics.json   precondition checks + aggregate readouts
                                    + rule-derived conclusion fields
    rmse_samples.npz               all sampled RMSE arrays (for notebook
                                    violin plots)
    rerun_bundle.pickle            input snapshot + results for a specific
                                    unit (for exact re-run tests)

Usage:
    python 033_exp_r21_r22_levelset.py                # all 48 units
    python 033_exp_r21_r22_levelset.py --skip-hr      # debug: exact sampler only
"""

# ── Environment guards: must be set before importing numpy/torch ──
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')   # avoids a torch/MKL duplicate-OpenMP conflict
os.environ.setdefault('OMP_NUM_THREADS', '4')           # cap CPU threads (GPU training may be running concurrently)
os.environ.setdefault('MKL_NUM_THREADS', '4')

import argparse
import json
import pickle
import platform
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Force UTF-8 console output encoding (does not affect the output files, which are always UTF-8)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# ════════════════════════════════════════════════════════════
# Constants
# ════════════════════════════════════════════════════════════

EXP0_DIR = SCRIPT_DIR / 'results' / 'exp0_kfold_prior'
GRAPH_CACHE = EXP0_DIR / 'graph_cache' / 'cached_graphs.pickle'
SUMMARY_RMSE_CSV = EXP0_DIR / 'summary' / 'summary_rmse.csv'
OUTPUT_DIR = SCRIPT_DIR / 'results' / 'exp_r21_r22_levelset'
TRAINING_SCRIPT = SCRIPT_DIR / '005_kfold_prior_training.py'

CONFIG = 'baseline'
SEEDS = [42, 123, 456]
K_SAMPLES = 200                  # number of alternative allocations sampled per (region, seed, basis)
RNG_SEED = 20260715              # fixed top-level seed; see _unit_rng_key for per-unit derivation
HR_BURN_IN = 3000                # hit-and-run burn-in steps
HR_THIN = 30                     # hit-and-run sampling interval
BASES = ['gnn', 'uniform']       # basis: learned solution / per-ITL3 Uniform
SAMPLERS = ['dirichlet_exact', 'hit_and_run']

# ── Tolerance specifications (consistent with tests/test_r21_r22.py) ──
TOL_MIXTURE_ABS = 1e-12          # level-set equality: sampled category mass vs. learned mixture (per-block np.sum convention)
TOL_MSUM_ABS = 1e-12             # sum_c m_c = 1 (normalised representative)
TOL_AGG_REL = 1e-9               # bincount fast aggregation vs. shared-module per-substation aggregation (learned solution)
TOL_ANCHOR_ABS = 2e-4            # learned-solution RMSE vs. frozen kfold_test_rmse.csv.
                                 # Two contributing sources: (1) the CSV is
                                 #   rounded to 4 decimal places (<=5e-5);
                                 #   (2) for some folds, grid_demands were
                                 #   regenerated after training by reloading
                                 #   the checkpoint and re-running inference
                                 #   (mtime evidence: rmse.csv 2026-03-10 vs.
                                 #   grid_demands 2026-03-12); GPU float32
                                 #   re-inference introduces ~1e-7 relative
                                 #   drift, observed to translate into an
                                 #   RMSE deviation of up to ~8.7e-5.
                                 # 2e-4 is still a ~1e-5-relative anchor for
                                 # RMSE values in the 6-17 MVA range, whereas
                                 # any genuine mismatch (wrong fold/CRS/domain)
                                 # would be O(0.01+).
PLAN_EXPECTED_RANK = 6            # rank if the 6 constraint rows were linearly independent (actual: <=5, see note D1)

RERUN_BUNDLE_UNIT = (42, 'TLG1')  # unit snapshotted for exact-reproduction tests (3 ITL3s, a moderate size)


def _status(ok: bool) -> str:
    """Conclusion fields are derived purely from numeric rules, never written by hand."""
    return 'ok' if ok else 'fail'


# ════════════════════════════════════════════════════════════
# Level-set geometry: constraint matrix, rank, category mass
# ════════════════════════════════════════════════════════════

def build_constraint_matrix(codes: np.ndarray, n_lu: int) -> np.ndarray:
    """Builds A = [n_lu category one-hot indicator rows; an all-ones row], shape (n_lu+1, n).

    This matches the actual loss term-for-term: each category row is the
    per-category aggregation row of mapping_matrix within that ITL3 (exactly
    one 1 per agent); the all-ones row encodes the sum(w) constraint. The
    all-ones row is a linear combination of the indicator rows, so
    rank <= n_lu (see note D1 in the module docstring).
    """
    n = len(codes)
    A = np.zeros((n_lu + 1, n), dtype=float)
    A[codes, np.arange(n)] = 1.0        # row c is 1 wherever cat(a) == c
    A[n_lu, :] = 1.0
    return A


def rank_and_nullspace(codes: np.ndarray, n_lu: int) -> tuple:
    """rank(A) and nullspace dimension = n - rank(A), verified by direct rank computation."""
    A = build_constraint_matrix(codes, n_lu)
    rank = int(np.linalg.matrix_rank(A))
    return rank, len(codes) - rank


def block_masses(codes: np.ndarray, w: np.ndarray, n_lu: int) -> np.ndarray:
    """Category mass vector m_c = sum_{cat(a)=c} w_a (per-block np.sum pairwise
    summation, matching the summation path used by the equality check, to
    avoid the ~n*eps-scale spurious error from bincount's sequential
    accumulation at n ~ 50,000)."""
    m = np.zeros(n_lu, dtype=float)
    for c in range(n_lu):
        mask = codes == c
        if mask.any():
            m[c] = float(np.sum(w[mask]))
    return m


def _pin_blocks(W: np.ndarray, codes: np.ndarray, m: np.ndarray, n_lu: int) -> None:
    """Pins each sample's category-block mass back to exactly m_c (in place): w_block *= m_c / sum(w_block).

    This is a numerical-hygiene step that removes floating-point drift
    accumulated during sampling, guaranteeing the level-set equality check
    passes at the <1e-12 tolerance. It has no material effect on uniformity
    (the scaling factor is 1 +/- O(1e-15))."""
    for c in range(n_lu):
        mask = codes == c
        if not mask.any():
            continue
        s = np.sum(W[:, mask], axis=1)
        if m[c] > 0:
            W[:, mask] *= (m[c] / s)[:, None]
        else:
            W[:, mask] = 0.0


def mixture_deviation(W: np.ndarray, codes: np.ndarray, m: np.ndarray, n_lu: int) -> float:
    """Level-set equality metric: max_k max_c |sum_{cat=c} W[k] - m_c| (per-block np.sum convention)."""
    dev = 0.0
    for c in range(n_lu):
        mask = codes == c
        if not mask.any():
            continue
        s = np.sum(W[:, mask], axis=1)
        dev = max(dev, float(np.abs(s - m[c]).max()))
    return dev


# ════════════════════════════════════════════════════════════
# Samplers (independent per ITL3; both samplers target the same polytope P(m))
# ════════════════════════════════════════════════════════════

def sample_dirichlet_exact(codes: np.ndarray, m: np.ndarray, n_lu: int,
                           K: int, rng: np.random.Generator) -> np.ndarray:
    """Exact i.i.d. uniform sampler (primary sampler; see note D2 in the module docstring).

    P(m) = product over c of { w_c >= 0, sum(w_c) = m_c } (a product of
    per-category rescaled simplices), so per-block Dirichlet(1,...,1)
    sampling (equivalently, normalised i.i.d. exponentials) scaled by m_c
    gives an exact uniform distribution over the polytope.
    Returns a (K, n) sample matrix with block masses pinned to m_c.
    """
    n = len(codes)
    W = np.zeros((K, n), dtype=float)
    for c in range(n_lu):
        mask = codes == c
        k_c = int(mask.sum())
        if k_c == 0 or m[c] <= 0:
            continue                       # empty category / zero-mass block: entire block is fixed at 0 (a face of the level set)
        if k_c == 1:
            W[:, mask] = m[c]              # single-point block: no degrees of freedom, always equals m_c
            continue
        E = rng.standard_exponential((K, k_c))
        W[:, mask] = E / np.sum(E, axis=1, keepdims=True) * m[c]
    _pin_blocks(W, codes, m, n_lu)
    return W


def sample_hit_and_run(codes: np.ndarray, m: np.ndarray, n_lu: int,
                       K: int, rng: np.random.Generator,
                       burn_in: int = HR_BURN_IN, thin: int = HR_THIN) -> tuple:
    """Hit-and-run uniform sampler (cross-check sampler, kept in full alongside the primary sampler).

    Implementation: a random direction is projected onto the constraint
    nullspace, then a uniform point is drawn along the resulting chord
    within the non-negative orthant. For the one-hot structure of A, the
    nullspace projection delta = z - A^T(AA^T)^+ Az is **exactly equal to
    per-category mean subtraction** (the category indicator rows are
    mutually orthogonal and the all-ones row is dependent on them),
    computable in O(n).

    The starting point is the per-block uniform point (the polytope's
    centroid, a strict interior point); after each step, block masses are
    re-pinned (a numerical-hygiene step preventing drift over tens of
    thousands of steps from breaking the 1e-12 equality tolerance).
    Returns (W(K,n), count of degenerate steps).
    """
    n = len(codes)
    counts = np.bincount(codes, minlength=n_lu).astype(float)
    free_block = (counts >= 2) & (m > 0)          # category blocks with degrees of freedom
    free_mask = free_block[codes]                 # agent-level free mask
    n_free = int(free_mask.sum())

    # Starting point: per-block uniform (centroid); zero-mass / single-point blocks take their defined value
    w = np.zeros(n, dtype=float)
    for c in range(n_lu):
        if counts[c] > 0 and m[c] > 0:
            w[codes == c] = m[c] / counts[c]

    W = np.zeros((K, n), dtype=float)
    if n_free == 0:                               # no degrees of freedom: the level set degenerates to a single point
        W[:] = w
        return W, 0

    codes_free = codes[free_mask]
    counts_free = np.bincount(codes_free, minlength=n_lu).astype(float)
    counts_free[counts_free == 0] = 1.0           # avoid division by zero (non-free categories do not participate)

    degenerate_steps = 0
    n_collected = 0
    total_steps = burn_in + K * thin
    step = 0
    w_free = w[free_mask].copy()
    m_free = m.copy()

    while n_collected < K:
        step += 1
        if step > total_steps * 10:               # hard safety valve against excessive degenerate steps
            raise RuntimeError('hit-and-run: too many degenerate steps, chain cannot advance')

        # Direction: Gaussian -> nullspace projection (= per-category mean subtraction) -> normalise
        z = rng.standard_normal(n_free)
        block_mean = np.bincount(codes_free, weights=z, minlength=n_lu) / counts_free
        delta = z - block_mean[codes_free]
        norm = float(np.linalg.norm(delta))
        if norm < 1e-12:
            degenerate_steps += 1
            continue
        delta /= norm

        # Chord endpoints: w + t*delta >= 0 => t in [t_lo, t_hi]
        moving = np.abs(delta) > 1e-14
        d_mov = delta[moving]
        w_mov = w_free[moving]
        with np.errstate(divide='ignore'):
            neg = d_mov < 0
            pos = ~neg
            t_hi = float(np.min(w_mov[neg] / -d_mov[neg])) if neg.any() else np.inf
            t_lo = float(-np.min(w_mov[pos] / d_mov[pos])) if pos.any() else -np.inf
        if not (np.isfinite(t_lo) and np.isfinite(t_hi)) or t_hi <= t_lo:
            degenerate_steps += 1                 # degenerate chord at a boundary point: skip
            continue

        # Draw a uniform point on the chord + update + re-pin block masses (numerical hygiene)
        t = rng.uniform(t_lo, t_hi)
        w_free = np.maximum(w_free + t * delta, 0.0)
        block_sum = np.bincount(codes_free, weights=w_free, minlength=n_lu)
        block_sum[block_sum == 0] = 1.0
        w_free *= (m_free / block_sum)[codes_free]

        if step > burn_in and (step - burn_in) % thin == 0:
            w[free_mask] = w_free
            W[n_collected] = w
            n_collected += 1

    _pin_blocks(W, codes, m, n_lu)                # pin block masses exactly (np.sum convention) after sampling
    return W, degenerate_steps


# ════════════════════════════════════════════════════════════
# Unit-level sampling and evaluation (pure numpy — tests exactly re-run this function via rerun_bundle)
# ════════════════════════════════════════════════════════════

def _unit_rng_key(seed: int, loc_idx: int, base_idx: int,
                  sampler_idx: int, itl3_idx: int) -> list:
    """Deterministic per-unit RNG seed (derived from the fixed RNG_SEED, exactly reproducible)."""
    return [RNG_SEED, seed, loc_idx, base_idx, sampler_idx, itl3_idx]


def sample_unit(unit: dict, K: int = K_SAMPLES,
                samplers: tuple = ('dirichlet_exact', 'hit_and_run'),
                hr_burn_in: int = HR_BURN_IN, hr_thin: int = HR_THIN) -> dict:
    """Samples the level set for a single (seed x region x basis) unit and computes the substation RMSE distribution.

    unit (pure-numpy input, can be pickled as a snapshot for test re-runs):
        seed, loc_idx, base_idx : seed-derivation keys
        itl3_names   : [str]                 ITL3 order (fixed)
        codes_list   : [np.ndarray int]      per-ITL3 category codes (in region-local position order)
        m_list       : [np.ndarray float]    per-ITL3 target category mass vector
        D_list       : [float]               per-ITL3 total demand D_r
        assign_list  : [np.ndarray int]      per-ITL3 agent -> substation assignment
        n_subs       : int
        actual       : np.ndarray            true substation demand
        n_lu         : int

    Returns: {sampler: {'rmse': (K,), 'mixture_dev_max', 'min_weight',
                     'hr_degenerate_steps'}}
    """
    n_subs = unit['n_subs']
    actual = unit['actual']
    out = {}
    for sampler in samplers:
        sampler_idx = SAMPLERS.index(sampler)
        alloc = np.zeros((K, n_subs), dtype=float)
        mixture_dev_max = 0.0
        min_weight = np.inf
        degenerate_total = 0
        for j in range(len(unit['itl3_names'])):
            codes = unit['codes_list'][j]
            m = unit['m_list'][j]
            rng = np.random.default_rng(
                _unit_rng_key(unit['seed'], unit['loc_idx'], unit['base_idx'],
                              sampler_idx, j))
            if sampler == 'dirichlet_exact':
                W = sample_dirichlet_exact(codes, m, unit['n_lu'], K, rng)
            else:
                W, deg = sample_hit_and_run(codes, m, unit['n_lu'], K, rng,
                                            burn_in=hr_burn_in, thin=hr_thin)
                degenerate_total += deg
            # Exact level-set equality check (mixture vector matches the learned solution to within 1e-12)
            mixture_dev_max = max(mixture_dev_max,
                                  mixture_deviation(W, codes, m, unit['n_lu']))
            min_weight = min(min_weight, float(W.min()))
            # Aggregation: demand = w * D_r -> substation (fast bincount path;
            # equivalence with the shared module's per-substation aggregation
            # is verified against the learned solution in the main pipeline)
            D_r = unit['D_list'][j]
            assign_j = unit['assign_list'][j]
            for k in range(K):
                alloc[k] += np.bincount(assign_j, weights=W[k], minlength=n_subs) * D_r
        rmse = np.sqrt(np.mean((alloc - actual[None, :]) ** 2, axis=1))
        out[sampler] = {
            'rmse': rmse,
            'mixture_dev_max': mixture_dev_max,
            'min_weight': min_weight,
            'hr_degenerate_steps': degenerate_total,
        }
    return out


def midrank_percentile(samples: np.ndarray, value: float) -> float:
    """Percentile of value within the sample distribution (midrank convention, in [0,100])."""
    less = float(np.sum(samples < value))
    equal = float(np.sum(samples == value))
    return 100.0 * (less + 0.5 * equal) / len(samples)


# ════════════════════════════════════════════════════════════
# Main-pipeline helper: term-for-term verification of the loss structure
# (graph cache -> compact per-unit input)
# ════════════════════════════════════════════════════════════

def verify_and_extract_loc(loc: str, graph, grid_gdf, region_sub, subs_sub) -> dict:
    """Verifies for a single region that the land-use supervision matrix matches grid_gdf term-for-term, and extracts a compact input.

    Five checks are performed (each result is recorded as a boolean; any
    failure raises immediately):
      V1 star topology: each agent has exactly one source->agent edge;
      V2 one-hot: mapping_matrix has exactly one nonzero entry per edge, equal to 1;
      V3 column decoding: col//n_lu == the edge's source, and col%n_lu == the grid point's land-use category;
      V4 domain alignment: each edge's source ITL3 == the agent's ITL3;
      V5 target provenance: landuse_ratio == row-normalised region_sub {cat}_percent
         (within float32 storage tolerance < 1e-6).
    """
    import torch  # noqa: F401  local import: not needed on the test-only code path

    ei = graph['source', 'connects_to', 'agent'].edge_index.numpy()
    n_agents = int(graph['agent'].num_nodes)
    M = graph.landuse_mapping_matrix.numpy()
    ratio = graph.landuse_ratio.numpy()
    n_lu = ratio.shape[1]

    cats = sorted(grid_gdf['landuse'].unique())
    assert len(cats) == n_lu, f'{loc}: category count {len(cats)} != landuse_ratio column count {n_lu}'
    cat_to_idx = {c: i for i, c in enumerate(cats)}
    lu_codes_all = grid_gdf['landuse'].map(cat_to_idx).values.astype(np.int64)

    v1 = bool((np.bincount(ei[1], minlength=n_agents) == 1).all())
    nz_per_row = (M != 0).sum(axis=1)
    v2 = bool((nz_per_row == 1).all() and (M.sum(axis=1) == 1).all())
    col_idx = M.argmax(axis=1)
    v3 = bool(((col_idx // n_lu) == ei[0]).all()
              and ((col_idx % n_lu) == lu_codes_all[ei[1]]).all())
    sim = graph.source_index_map
    src_itl3 = region_sub.loc[sim.values, 'ITL3'].values
    v4 = bool((src_itl3[ei[0]] == grid_gdf['ITL3'].values[ei[1]]).all())
    pct_cols = [f'{c}_percent' for c in cats]
    vals = region_sub.loc[sim.values, pct_cols].values.astype(float)
    row_sums = vals.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    ratio_dev = float(np.abs(vals / row_sums - ratio).max())
    v5 = ratio_dev < 1e-6

    checks = {'V1_star': v1, 'V2_onehot': v2, 'V3_column_decode': v3,
              'V4_itl3_domain': v4, 'V5_target_source': v5,
              'landuse_ratio_max_dev_float32': ratio_dev, 'n_lu': n_lu}
    if not all([v1, v2, v3, v4, v5]):
        raise AssertionError(f'{loc}: loss structure verification failed {checks}')

    # Compact input: grouped by ITL3 (coverage must be 100%)
    region_info = region_sub.set_index('ITL3')
    itl3_groups = [(itl3, np.asarray(g.index))
                   for itl3, g in grid_gdf.groupby('ITL3')]
    covered = sum(len(idx) for itl3, idx in itl3_groups if itl3 in region_info.index)
    coverage = covered / n_agents
    itl3_groups = [(itl3, idx) for itl3, idx in itl3_groups
                   if itl3 in region_info.index]
    D_map = {itl3: float(region_info.loc[itl3, 'Demand (MVA)'])
             for itl3, _ in itl3_groups}

    return {
        'checks': checks, 'coverage': coverage, 'cats': cats,
        'lu_codes': lu_codes_all, 'itl3_groups': itl3_groups, 'D_map': D_map,
        'n_agents': n_agents, 'n_lu': n_lu,
        'gva_cols_present_unused': [c for c in region_sub.columns if c.endswith('_gva')],
    }


def find_test_fold(seed: int, loc: str) -> str:
    """Finds the fold in which loc is used as the test set, from kfold_splits.json (must be exactly one fold)."""
    with open(EXP0_DIR / f'seed_{seed}' / CONFIG / 'kfold_splits.json',
              encoding='utf-8') as f:
        splits = json.load(f)
    folds = [k for k, v in splits.items() if loc in v['test']]
    assert len(folds) == 1, f'seed={seed} loc={loc}: number of test folds {len(folds)} != 1'
    return folds[0].replace('fold_', 'fold')


def read_baseline_objective() -> dict:
    """Reads the baseline objective function directly from the training script's source via importlib (this field is derived, never written by hand)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('kfold_prior_training_005',
                                                  TRAINING_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return dict(mod.CONFIG_MAP[CONFIG]['objective_weights'])


# ════════════════════════════════════════════════════════════
# Main pipeline
# ════════════════════════════════════════════════════════════

def run(skip_hr: bool = False) -> None:
    import shared_correction_utils as scu

    samplers = ('dirichlet_exact',) if skip_hr else tuple(SAMPLERS)
    all_locations = scu.ALL_LOCATIONS

    # ── 0. Disclose the baseline objective function (read from source) ──
    objective = read_baseline_objective()
    objective_only_landuse = (objective == {'landuse_prediction_loss': 1.0})
    print(f'baseline objective function (read from training script source): {objective}')

    # ── 1. Load graph cache + verify loss structure term-for-term + extract compact inputs ──
    print(f'Loading graph cache: {GRAPH_CACHE}')
    with open(GRAPH_CACHE, 'rb') as f:
        cache = pickle.load(f)
    graphs, grids = cache['graphs'], cache['grids']
    region_dict, subs_dict = cache['region_dict'], cache['subs_dict']

    loc_data = {}
    loss_checks = {}
    for loc in all_locations:
        grid_gdf, _ = grids[loc]
        info = verify_and_extract_loc(loc, graphs[loc], grid_gdf,
                                      region_dict[loc], subs_dict[loc])
        loc_data[loc] = info
        loss_checks[loc] = info['checks']
        print(f'  {loc}: loss structure verification passed (n={info["n_agents"]}, '
              f'n_itl3={len(info["itl3_groups"])}, coverage={info["coverage"]:.4f})')
    del graphs, cache   # free the supervision-matrix memory; grids/region/subs are still needed

    # ── 2. Voronoi assignment (once per region, using the shared module's two-stage cache) ──
    voronoi_cache = {}
    for loc in all_locations:
        grid_gdf, _ = grids[loc]
        scu.compute_voronoi_assignment(grid_gdf, subs_dict[loc],
                                       cache=voronoi_cache, cache_key=loc)
        print(f'  {loc}: Voronoi assignment complete')

    # ── 3. rank / nullspace dimension (per region x ITL3, independent of seed) ──
    nullspace_rows = []
    for loc in all_locations:
        info = loc_data[loc]
        for itl3, idx in info['itl3_groups']:
            codes = info['lu_codes'][idx]
            rank, nsdim = rank_and_nullspace(codes, info['n_lu'])
            nullspace_rows.append({
                'location': loc, 'itl3': itl3, 'n_agents': len(idx),
                'n_cats_present': int(len(np.unique(codes))),
                'rank_A': rank, 'nullspace_dim': nsdim,
                'plan_expected_rank': PLAN_EXPECTED_RANK,
            })
    ns_df = pd.DataFrame(nullspace_rows)
    rank_equals_cats = bool((ns_df['rank_A'] == ns_df['n_cats_present']).all())

    # ── 4. 48 units x 2 bases x samplers ──
    summary_rows = []
    npz_arrays = {}
    anchor_dev_max = 0.0
    agg_rel_dev_max = 0.0
    mixture_dev_global = 0.0
    min_weight_global = np.inf
    msum_dev_max = 0.0
    sum_w_dev_prenorm_max = 0.0
    rerun_bundle = None

    for seed in SEEDS:
        kfold_csv = pd.read_csv(EXP0_DIR / f'seed_{seed}' / CONFIG / 'kfold_test_rmse.csv',
                                index_col=0)
        for loc_idx, loc in enumerate(all_locations):
            info = loc_data[loc]
            fold = find_test_fold(seed, loc)
            gd_path = (EXP0_DIR / f'seed_{seed}' / CONFIG / fold /
                       'grid_demands' / f'{loc}_grid_demands.pickle')
            with open(gd_path, 'rb') as f:
                grid_demands = pickle.load(f)
            d_raw = np.asarray(grid_demands['gnn_demand'], dtype=float)

            grid_gdf, _ = grids[loc]
            subs_sub = subs_dict[loc]
            assignment = voronoi_cache[loc]
            actual = subs_sub['Demand (MVA)'].values.astype(float)
            n_subs = len(subs_sub)

            # ── Anchor the learned solution (raw, un-normalised demand, matching the training pipeline's aggregation path) ──
            alloc_raw = np.zeros(n_subs)
            for itl3, idx in info['itl3_groups']:
                alloc_raw += np.bincount(assignment[idx], weights=d_raw[idx],
                                         minlength=n_subs)
            rmse_anchor_raw = float(np.sqrt(np.mean((alloc_raw - actual) ** 2)))
            # Shared-module per-substation aggregation convention (equivalence anchor)
            subs_result = scu.aggregate_by_assignment(subs_sub, assignment, d_raw)
            rmse_shared = float(scu.evaluate_allocation(subs_result)['rmse'])
            agg_rel_dev = abs(rmse_anchor_raw - rmse_shared) / rmse_shared
            agg_rel_dev_max = max(agg_rel_dev_max, agg_rel_dev)
            # Frozen CSV anchor (rounded to 4 decimal places)
            csv_val = float(kfold_csv.loc['voronoi_GNN', loc])
            anchor_dev = abs(rmse_anchor_raw - csv_val)
            anchor_dev_max = max(anchor_dev_max, anchor_dev)

            # ── Recover per-agent basis weights (normalised representative; the loss is scale-invariant, see module docstring item 5) ──
            w_norm = np.zeros_like(d_raw)
            sum_w_dev_prenorm = 0.0
            for itl3, idx in info['itl3_groups']:
                w_raw = d_raw[idx] / info['D_map'][itl3]
                s = float(np.sum(w_raw))
                sum_w_dev_prenorm = max(sum_w_dev_prenorm, abs(s - 1.0))
                w_norm[idx] = w_raw / s
            sum_w_dev_prenorm_max = max(sum_w_dev_prenorm_max, sum_w_dev_prenorm)

            for base_idx, base in enumerate(BASES):
                # Basis point and its category mass vector
                itl3_names, codes_list, m_list, D_list, assign_list = [], [], [], [], []
                w_base = np.zeros_like(d_raw)
                for itl3, idx in info['itl3_groups']:
                    codes = info['lu_codes'][idx]
                    if base == 'gnn':
                        w_b = w_norm[idx]
                    else:                       # per-ITL3 Uniform
                        w_b = np.full(len(idx), 1.0 / len(idx))
                    w_base[idx] = w_b
                    m = block_masses(codes, w_b, info['n_lu'])
                    msum_dev_max = max(msum_dev_max, abs(float(m.sum()) - 1.0))
                    itl3_names.append(itl3)
                    codes_list.append(codes)
                    m_list.append(m)
                    D_list.append(info['D_map'][itl3])
                    assign_list.append(assignment[idx])

                # RMSE at the basis point / at the level-set centroid (fast bincount path)
                def _rmse_of_weights(w_vec):
                    alloc = np.zeros(n_subs)
                    for j2, (itl3, idx) in enumerate(info['itl3_groups']):
                        alloc += np.bincount(assignment[idx],
                                             weights=w_vec[idx],
                                             minlength=n_subs) * D_list[j2]
                    return float(np.sqrt(np.mean((alloc - actual) ** 2)))

                rmse_base_point = _rmse_of_weights(w_base)
                w_bar = np.zeros_like(w_base)      # centroid = per-block uniform (under mass conservation)
                for j2, (itl3, idx) in enumerate(info['itl3_groups']):
                    codes = codes_list[j2]
                    counts = np.bincount(codes, minlength=info['n_lu']).astype(float)
                    counts[counts == 0] = 1.0
                    w_bar[idx] = (m_list[j2] / counts)[codes]
                rmse_barycenter = _rmse_of_weights(w_bar)

                unit = {
                    'seed': seed, 'loc_idx': loc_idx, 'base_idx': base_idx,
                    'itl3_names': itl3_names, 'codes_list': codes_list,
                    'm_list': m_list, 'D_list': D_list,
                    'assign_list': assign_list, 'n_subs': n_subs,
                    'actual': actual, 'n_lu': info['n_lu'],
                }
                res = sample_unit(unit, K=K_SAMPLES, samplers=samplers)

                # Snapshot for exact re-run tests (specific unit x gnn basis)
                if ((seed, loc) == RERUN_BUNDLE_UNIT and base == 'gnn'
                        and not skip_hr):
                    rerun_bundle = {
                        'unit': unit, 'K': K_SAMPLES,
                        'hr_burn_in': HR_BURN_IN, 'hr_thin': HR_THIN,
                        'rng_seed': RNG_SEED,
                        'results': {s: {'rmse': res[s]['rmse'].copy()}
                                    for s in res},
                    }

                # Zero-mass block / free-dimension statistics (face dimension = sum_{m_c>0} (|block|-1))
                face_dim = 0
                n_zero_mass_blocks = 0
                for j2 in range(len(itl3_names)):
                    counts = np.bincount(codes_list[j2], minlength=info['n_lu'])
                    for c in range(info['n_lu']):
                        if counts[c] == 0:
                            continue
                        if m_list[j2][c] > 0:
                            face_dim += int(counts[c]) - 1
                        else:
                            n_zero_mass_blocks += 1
                nullspace_total = int(
                    ns_df[ns_df['location'] == loc]['nullspace_dim'].sum())

                q_print, pct_print = None, None      # for progress printing (exact-sampler convention)
                for sampler in samplers:
                    r = res[sampler]
                    rmse = r['rmse']
                    mixture_dev_global = max(mixture_dev_global, r['mixture_dev_max'])
                    min_weight_global = min(min_weight_global, r['min_weight'])
                    npz_arrays[f'{seed}|{loc}|{base}|{sampler}'] = rmse
                    q = np.percentile(rmse, [5, 25, 50, 75, 95])
                    summary_rows.append({
                        'seed': seed, 'location': loc, 'fold': fold,
                        'base': base, 'sampler': sampler, 'K': len(rmse),
                        'n_agents': info['n_agents'],
                        'n_itl3': len(info['itl3_groups']), 'n_subs': n_subs,
                        'nullspace_dim_total': nullspace_total,
                        'face_dim_total': face_dim,
                        'n_zero_mass_blocks': n_zero_mass_blocks,
                        'rmse_base_point': rmse_base_point,
                        'rmse_barycenter': rmse_barycenter,
                        'rmse_anchor_raw': rmse_anchor_raw if base == 'gnn' else np.nan,
                        'anchor_dev_vs_kfold_csv': anchor_dev if base == 'gnn' else np.nan,
                        'rmse_min': float(rmse.min()), 'rmse_p5': float(q[0]),
                        'rmse_p25': float(q[1]), 'rmse_p50': float(q[2]),
                        'rmse_p75': float(q[3]), 'rmse_p95': float(q[4]),
                        'rmse_max': float(rmse.max()),
                        'rmse_range': float(rmse.max() - rmse.min()),
                        'rmse_p95_p5': float(q[4] - q[0]),
                        'rmse_std': float(rmse.std()),   # numpy ddof=0 convention
                        'percentile_base_point': midrank_percentile(rmse, rmse_base_point),
                        'percentile_barycenter': midrank_percentile(rmse, rmse_barycenter),
                        'mixture_dev_max': r['mixture_dev_max'],
                        'min_weight': r['min_weight'],
                        'hr_degenerate_steps': (r['hr_degenerate_steps']
                                                if sampler == 'hit_and_run' else 0),
                        'sum_w_dev_prenorm': (sum_w_dev_prenorm
                                              if base == 'gnn' else 0.0),
                    })
                    if sampler == 'dirichlet_exact':
                        q_print = q
                        pct_print = summary_rows[-1]['percentile_base_point']
                print(f'  seed={seed} {loc} [{base}]: '
                      f'RMSE(basis point)={rmse_base_point:.4f}, '
                      f'level-set P5-P95=[{q_print[0]:.4f}, {q_print[4]:.4f}], '
                      f'basis-point percentile={pct_print:.1f}%  (exact sampler)')

    summary_df = pd.DataFrame(summary_rows)
    # Derived columns (pure column-wise arithmetic, independently verifiable):
    #   learned_minus_p50   = RMSE difference between the basis point (a level-set
    #                         member) and the level set's typical member (P50) --
    #                         an empirical measure of how much RMSE can change
    #                         under a redistribution that leaves the loss unchanged
    #   span_incl_base_point = observed RMSE span across the level set, including the basis point
    summary_df['learned_minus_p50'] = (summary_df['rmse_base_point']
                                       - summary_df['rmse_p50'])
    summary_df['span_incl_base_point'] = (
        np.maximum(summary_df['rmse_max'], summary_df['rmse_base_point'])
        - np.minimum(summary_df['rmse_min'], summary_df['rmse_base_point']))

    # ── 5. Aggregate readouts and rule-derived conclusion fields ──
    def _agg_percentiles(df_sub: pd.DataFrame) -> dict:
        """Aggregates across units: first averages over seeds per region, then takes the median across the 16 regions;
        also reports the raw median across all 48 units."""
        per_loc = df_sub.groupby('location')['percentile_base_point'].mean()
        return {
            'median_across_locations_seed_mean': float(per_loc.median()),
            'median_across_48_units': float(df_sub['percentile_base_point'].median()),
            'iqr_across_48_units': [
                float(df_sub['percentile_base_point'].quantile(0.25)),
                float(df_sub['percentile_base_point'].quantile(0.75))],
            'min': float(df_sub['percentile_base_point'].min()),
            'max': float(df_sub['percentile_base_point'].max()),
            'n_units_below_all_samples': int((df_sub['percentile_base_point']
                                              <= 100.0 * 0.5 / K_SAMPLES).sum()),
        }

    gnn_exact = summary_df[(summary_df['base'] == 'gnn')
                           & (summary_df['sampler'] == 'dirichlet_exact')]
    uni_exact = summary_df[(summary_df['base'] == 'uniform')
                           & (summary_df['sampler'] == 'dirichlet_exact')]

    r21_pct = _agg_percentiles(gnn_exact)
    med_pct = r21_pct['median_across_locations_seed_mean']
    if med_pct <= 5.0:
        r21_verdict = 'learned_strongly_below_levelset'
    elif med_pct <= 25.0:
        r21_verdict = 'learned_below_typical'
    elif med_pct < 75.0:
        r21_verdict = 'learned_typical_member'
    else:
        r21_verdict = 'learned_worse_than_typical'

    # Antagonism magnitude (read directly from the frozen summary_rmse.csv, never written by hand)
    sm = pd.read_csv(SUMMARY_RMSE_CSV)
    sm_base = sm[sm['config'] == CONFIG].set_index('method')['mean']
    delta_np = float(sm_base['voronoi_ntl_prox_GNN'] - sm_base['voronoi_GNN'])
    delta_n = float(sm_base['voronoi_ntl_GNN'] - sm_base['voronoi_GNN'])
    delta_p = float(sm_base['voronoi_prox_GNN'] - sm_base['voronoi_GNN'])

    range_median = float(gnn_exact['rmse_range'].median())
    p95p5_median = float(gnn_exact['rmse_p95_p5'].median())
    std_median = float(gnn_exact['rmse_std'].median())
    learned_minus_p50_median = float(gnn_exact['learned_minus_p50'].median())
    learned_minus_p50_abs_median = float(gnn_exact['learned_minus_p50'].abs().median())
    span_median = float(gnn_exact['span_incl_base_point'].median())
    ratio_range = range_median / abs(delta_np)
    ratio_p95p5 = p95p5_median / abs(delta_np)
    ratio_learned_p50 = learned_minus_p50_abs_median / abs(delta_np)
    ratio_span = span_median / abs(delta_np)
    if ratio_p95p5 >= 1.0:
        r22_verdict = 'levelset_freedom_exceeds_antagonism'
    elif ratio_p95p5 >= 0.5:
        r22_verdict = 'levelset_freedom_comparable_to_antagonism'
    else:
        r22_verdict = 'levelset_freedom_below_antagonism'
    if ratio_learned_p50 >= 1.0:
        r22_verdict_learned = 'learned_to_typical_shift_exceeds_antagonism'
    elif ratio_learned_p50 >= 0.5:
        r22_verdict_learned = 'learned_to_typical_shift_comparable_to_antagonism'
    else:
        r22_verdict_learned = 'learned_to_typical_shift_below_antagonism'

    # HR vs. exact-sampler agreement (cross-check diagnostic)
    hr_agreement = None
    if 'hit_and_run' in samplers:
        rows_hr = []
        for (seed, loc, base), grp in summary_df.groupby(['seed', 'location', 'base']):
            g_ex = grp[grp['sampler'] == 'dirichlet_exact'].iloc[0]
            g_hr = grp[grp['sampler'] == 'hit_and_run'].iloc[0]
            a = npz_arrays[f'{seed}|{loc}|{base}|dirichlet_exact']
            b = npz_arrays[f'{seed}|{loc}|{base}|hit_and_run']
            # Two-sample KS statistic (computed manually to avoid a new dependency)
            allv = np.sort(np.concatenate([a, b]))
            cdf_a = np.searchsorted(np.sort(a), allv, side='right') / len(a)
            cdf_b = np.searchsorted(np.sort(b), allv, side='right') / len(b)
            ks = float(np.abs(cdf_a - cdf_b).max())
            iqr_ex = g_ex['rmse_p75'] - g_ex['rmse_p25']
            iqr_hr = g_hr['rmse_p75'] - g_hr['rmse_p25']
            rows_hr.append({'seed': int(seed), 'location': str(loc),
                            'base': str(base), 'ks_stat': ks,
                            'median_diff': float(g_hr['rmse_p50'] - g_ex['rmse_p50']),
                            'iqr_ratio_hr_over_exact':
                                float(iqr_hr / iqr_ex) if iqr_ex > 0 else float('nan')})
        hr_df = pd.DataFrame(rows_hr)
        hr_agreement = {
            'note': ('Hit-and-run is retained as a cross-check sampler; at '
                     '~50,000 dimensions, an MCMC chain of 9000 steps would '
                     'necessarily remain confined to a lower-dimensional '
                     'slice, which theoretically predicts a narrower spread '
                     '(iqr_ratio < 1). Primary readouts use the exact '
                     'i.i.d. uniform sampler (see note D2).'),
            'hr_burn_in': HR_BURN_IN, 'hr_thin': HR_THIN,
            'ks_stat_median': float(hr_df['ks_stat'].median()),
            'ks_stat_max': float(hr_df['ks_stat'].max()),
            'median_diff_median': float(hr_df['median_diff'].median()),
            'iqr_ratio_median': float(hr_df['iqr_ratio_hr_over_exact'].median()),
            'iqr_ratio_min': float(hr_df['iqr_ratio_hr_over_exact'].min()),
            'per_unit': rows_hr,
        }

    # ── 6. Precondition checks (expected/actual/status) ──
    preconditions = {
        'P14_1_loss_structure_verified': {
            'description': ('Term-for-term loss-structure verification (V1 star '
                            'topology / V2 one-hot / V3 column decoding / V4 ITL3 '
                            'domain / V5 target provenance), passing for all 16 regions'),
            'expected': True,
            'actual': all(all(v for k, v in c.items()
                              if k.startswith('V')) for c in loss_checks.values()),
            'status': _status(all(all(v for k, v in c.items() if k.startswith('V'))
                                  for c in loss_checks.values())),
        },
        'P14_2_objective_only_landuse': {
            'description': ('The baseline objective function consists only of '
                            'landuse_prediction_loss (read directly from the '
                            'training script source), so the level set is a '
                            'complete characterisation of that configuration\'s '
                            'solution space'),
            'expected': {'landuse_prediction_loss': 1.0},
            'actual': objective,
            'status': _status(objective_only_landuse),
        },
        'P14_3_mixture_equality': {
            'description': f'Level-set equality: maximum deviation of sampled category mass vs. target mixture < {TOL_MIXTURE_ABS:g}',
            'expected': f'< {TOL_MIXTURE_ABS:g}',
            'actual': mixture_dev_global,
            'status': _status(mixture_dev_global < TOL_MIXTURE_ABS),
        },
        'P14_4_nonnegativity': {
            'description': 'All sampled weights are non-negative',
            'expected': '>= 0',
            'actual': min_weight_global,
            'status': _status(min_weight_global >= 0.0),
        },
        'P14_5_msum_normalized': {
            'description': f'Normalised-representative deviation |sum_c m_c - 1| < {TOL_MSUM_ABS:g} (including the Uniform basis)',
            'expected': f'< {TOL_MSUM_ABS:g}',
            'actual': msum_dev_max,
            'status': _status(msum_dev_max < TOL_MSUM_ABS),
        },
        'P14_6_rank_verification': {
            'description': ('Directly computed rank(A): rank = number of '
                            'categories present (<=5, not 6, since the '
                            'all-ones row is a linear combination of the '
                            'one-hot rows; see note D1)'),
            'expected': 'rank_A == n_cats_present (all ITL3s)',
            'actual': {'all_equal': rank_equals_cats,
                       'rank_min': int(ns_df['rank_A'].min()),
                       'rank_max': int(ns_df['rank_A'].max())},
            'status': _status(rank_equals_cats),
        },
        'P14_7_anchor_learned_rmse': {
            'description': (f'Learned-solution RMSE (raw demand + bincount '
                            f'aggregation) anchored against the frozen '
                            f'kfold_test_rmse.csv, per region, to within '
                            f'{TOL_ANCHOR_ABS:g} '
                            '(accounting for 4-decimal-place CSV rounding '
                            '<=5e-5, plus float32 re-inference drift for '
                            'some folds\' regenerated grid_demands; see D5)'),
            'expected': f'< {TOL_ANCHOR_ABS:g}',
            'actual': anchor_dev_max,
            'status': _status(anchor_dev_max < TOL_ANCHOR_ABS),
        },
        'P14_8_bincount_equals_shared_module': {
            'description': (f'Relative deviation between the fast bincount '
                            f'aggregation and the shared module\'s '
                            f'per-substation aggregation (learned solution) '
                            f'< {TOL_AGG_REL:g}'),
            'expected': f'< {TOL_AGG_REL:g}',
            'actual': agg_rel_dev_max,
            'status': _status(agg_rel_dev_max < TOL_AGG_REL),
        },
    }

    results = {
        'meta': {
            'script': '033_exp_r21_r22_levelset.py',
            'plan_step': 'Identifiability and training-evaluation-misalignment analysis via level-set sampling',
            'config': CONFIG, 'seeds': SEEDS, 'n_locations': len(all_locations),
            'n_units': len(SEEDS) * len(all_locations),
            'K': K_SAMPLES, 'rng_seed': RNG_SEED,
            'rng_scheme': ('default_rng([RNG_SEED, seed, loc_idx, base_idx, '
                           'sampler_idx, itl3_idx]) -- deterministic per-unit derivation, exactly reproducible'),
            'samplers': list(samplers), 'primary_sampler': 'dirichlet_exact',
            'bases': BASES,
            'hr_burn_in': HR_BURN_IN, 'hr_thin': HR_THIN,
            'omp_threads': os.environ.get('OMP_NUM_THREADS'),
            'python': platform.python_version(), 'numpy': np.__version__,
            'pandas': pd.__version__,
            'timestamp': datetime.now().isoformat(timespec='seconds'),
        },
        'tolerances': {
            'mixture_abs': TOL_MIXTURE_ABS, 'msum_abs': TOL_MSUM_ABS,
            'agg_rel': TOL_AGG_REL, 'anchor_abs': TOL_ANCHOR_ABS,
        },
        'loss_verification': {
            'summary': ('LandusePredictionLoss reconstruction target = the '
                        'category one-hot aggregate mixture per ITL3 source '
                        '(sum of edge-weight mass for agents whose category '
                        '= argmax(lu_*_prop), then row-normalised, compared '
                        'via KL divergence against the normalised '
                        '{cat}_percent target); aggregation domain = ITL3; '
                        'star graph topology; no GVA component.'),
            'per_location_checks': loss_checks,
            'objective_weights_from_source': objective,
            'gva_columns_present_but_unused':
                loc_data[all_locations[0]]['gva_cols_present_unused'],
        },
        'deviations_registry': [
            ('D1: rank(A) = number of categories present (<=5), not 6 as '
             'would follow if the constraint rows were linearly '
             'independent -- the actual loss uses categorical one-hot '
             'encoding, so the all-ones row is a linear combination of '
             'the indicator rows; the nullspace dimension n - rank(A) is '
             'verified directly.'),
            ('D2: the one-hot structure implies the level set is a '
             'product of per-category rescaled simplices, which admits '
             'an exact i.i.d. uniform sampler (per-block '
             'Dirichlet(1,...,1)). Primary readouts use dirichlet_exact; '
             'hit-and-run sampling is retained in full as a cross-check, '
             'with agreement diagnostics recorded.'),
            ('D3: anchoring the learned solution uses raw, un-normalised '
             f'demand (float32 storage introduces a maximum sum(w)-1 '
             f'drift of {sum_w_dev_prenorm_max:.2e}); locating the level '
             'set uses a normalised representative, since the loss is '
             'invariant to an overall rescaling of w (see '
             'loss_verification).'),
            ('D4: KMP_DUPLICATE_LIB_OK=TRUE is set as an environment '
             'guard against a torch/MKL duplicate-OpenMP conflict.'),
            ('D5: the learned-solution anchoring tolerance is set to '
             '2e-4 (rather than the pure rounding bound of 6e-5), '
             'because for some folds the frozen grid_demands were '
             'regenerated after the training session by reloading the '
             'checkpoint and re-running inference (mtime evidence: '
             'fold3/rmse.csv 2026-03-10 vs. fold3/grid_demands '
             '2026-03-12); GPU float32 re-inference drift produces an '
             'RMSE difference of up to 8.7e-5 in practice; this drift is '
             'a pre-existing property of the frozen exp0 artifacts, not '
             'something introduced by this experiment.'),
        ],
        'preconditions': preconditions,
        'nullspace': {
            'note': ('Nullspace dimension = n_agents - rank(A), verified '
                     'directly per ITL3. Because of the one-hot structure, '
                     'rank = number of categories present (dimension = '
                     'n-5 when all 5 categories are present).'),
            'n_itl3_total': len(ns_df),
            'nullspace_dim_min': int(ns_df['nullspace_dim'].min()),
            'nullspace_dim_max': int(ns_df['nullspace_dim'].max()),
            'nullspace_dim_median': float(ns_df['nullspace_dim'].median()),
            'rank_distribution': {int(k): int(v) for k, v in
                                  ns_df['rank_A'].value_counts().items()},
        },
        'r21_identifiability': {
            'question': ('Identifiability: is w_sa identifiable given only '
                         'the aggregate land-use constraint? The readout is '
                         'the percentile of the learned-solution RMSE '
                         'within the uniform sample distribution over its '
                         'level set.'),
            'learned_percentile': r21_pct,
            'barycenter_percentile_median': float(
                gnn_exact['percentile_barycenter'].median()),
            'verdict': r21_verdict,
            'verdict_rule': ('median_across_locations_seed_mean: <=5 → '
                             'learned_strongly_below_levelset; <=25 → '
                             'learned_below_typical; <75 → learned_typical_member; '
                             'else learned_worse_than_typical'),
            'interpretation_low': ('A markedly low percentile implies that '
                                   'the parameterisation, graph structure, '
                                   'and/or training dynamics act as an '
                                   'implicit regulariser, so the learned '
                                   'solution is not a random member of the '
                                   'level set -- strengthening the '
                                   'identifiability argument.'),
            'interpretation_mid': ('A mid-range percentile means L_landuse '
                                   'does not pin down RMSE, and the '
                                   'argument falls back to the paired '
                                   'experimental design instead.'),
        },
        'r22_evaluation_freedom': {
            'question': ('Evaluation sensitivity: how much can a '
                         'redistribution that barely changes L_landuse '
                         'change substation RMSE? The readout is the RMSE '
                         'spread across the level set compared with the '
                         'antagonism magnitude reported elsewhere.'),
            'levelset_rmse_range_median': range_median,
            'levelset_rmse_p95_p5_median': p95p5_median,
            'levelset_rmse_std_median': std_median,
            'learned_minus_p50_median': learned_minus_p50_median,
            'learned_minus_p50_abs_median': learned_minus_p50_abs_median,
            'span_incl_base_point_median': span_median,
            'antagonism_delta_np_vs_base': delta_np,
            'delta_n_vs_base': delta_n,
            'delta_p_vs_base': delta_p,
            'antagonism_source': str(SUMMARY_RMSE_CSV.relative_to(SCRIPT_DIR)),
            'ratio_range_to_antagonism': ratio_range,
            'ratio_p95p5_to_antagonism': ratio_p95p5,
            'ratio_learned_p50_to_antagonism': ratio_learned_p50,
            'ratio_span_to_antagonism': ratio_span,
            'verdict': r22_verdict,
            'verdict_rule': ('ratio_p95p5_to_antagonism: >=1 → exceeds; '
                             '>=0.5 → comparable; else below'),
            'verdict_learned_to_typical': r22_verdict_learned,
            'verdict_learned_to_typical_rule': (
                'ratio_learned_p50_to_antagonism: >=1 → exceeds; '
                '>=0.5 → comparable; else below'),
            'reading_note': ('Two complementary readouts: p95_p5 is the '
                             'RMSE spread **among typical members** of the '
                             'level set (expected to be narrow, since a '
                             'uniform measure concentrates in high '
                             'dimensions); learned_minus_p50 is the RMSE '
                             'shift from the learned solution (itself one '
                             'level-set member) to a typical member -- an '
                             'empirically demonstrated lower bound on how '
                             'much RMSE can change under a redistribution '
                             'that leaves the loss exactly unchanged.'),
        },
        'uniform_control': {
            'note': ('Control: the Uniform basis is sampled the same way '
                     'on its own mixture\'s level set (a static-basis '
                     'solution-space context). The Uniform basis point '
                     'itself is the centroid of its level set.'),
            'percentile': _agg_percentiles(uni_exact),
            'rmse_range_median': float(uni_exact['rmse_range'].median()),
            'rmse_p95_p5_median': float(uni_exact['rmse_p95_p5'].median()),
        },
        'hr_vs_exact_agreement': hr_agreement,
        'diagnostics': {
            'sum_w_dev_prenorm_max': sum_w_dev_prenorm_max,
            'anchor_dev_max': anchor_dev_max,
            'agg_rel_dev_max': agg_rel_dev_max,
            'mixture_dev_global': mixture_dev_global,
            'min_weight_global': min_weight_global,
        },
    }

    # ── 7. Write outputs ──
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(OUTPUT_DIR / 'levelset_samples_summary.csv', index=False)
    ns_df.to_csv(OUTPUT_DIR / 'nullspace_dims.csv', index=False)
    np.savez_compressed(OUTPUT_DIR / 'rmse_samples.npz', **npz_arrays)
    with open(OUTPUT_DIR / 'identifiability_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    if rerun_bundle is not None:
        with open(OUTPUT_DIR / 'rerun_bundle.pickle', 'wb') as f:
            pickle.dump(rerun_bundle, f)

    # ── 8. Print summary ──
    print('\n' + '=' * 68)
    print('Identifiability / evaluation-sensitivity level-set sampling -- summary')
    print('=' * 68)
    for key, item in preconditions.items():
        print(f'  {key}: {item["status"]}')
    print(f'\nLearned-solution percentile (median across regions, exact sampler): {med_pct:.2f}%  -> {r21_verdict}')
    print(f'     (centroid percentile median = {results["r21_identifiability"]["barycenter_percentile_median"]:.2f}%)')
    print(f'Level-set RMSE spread median: range {range_median:.4f} / P95-P5 {p95p5_median:.4f} MVA')
    print(f'     Learned-to-typical shift |learned-P50| median: {learned_minus_p50_abs_median:.4f} MVA'
          f' (signed median {learned_minus_p50_median:+.4f})')
    print(f'     Antagonism magnitude (frozen artifacts): dNP={delta_np:+.4f}, dN={delta_n:+.4f}, dP={delta_p:+.4f}')
    print(f'     P95-P5 / |dNP| = {ratio_p95p5:.3f}  -> {r22_verdict}')
    print(f'     |learned-P50| / |dNP| = {ratio_learned_p50:.3f}  -> {r22_verdict_learned}')
    print(f'Nullspace dimension range: [{results["nullspace"]["nullspace_dim_min"]}, '
          f'{results["nullspace"]["nullspace_dim_max"]}], '
          f'median {results["nullspace"]["nullspace_dim_median"]:.0f}')
    print(f'Uniform control percentile median: '
          f'{results["uniform_control"]["percentile"]["median_across_locations_seed_mean"]:.2f}%')
    if hr_agreement:
        print(f'HR vs. exact sampler: KS median {hr_agreement["ks_stat_median"]:.3f}, '
              f'IQR ratio median {hr_agreement["iqr_ratio_median"]:.3f}')
    print(f'\nOutputs written to {OUTPUT_DIR}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Identifiability / evaluation-sensitivity level-set sampling experiment')
    parser.add_argument('--skip-hr', action='store_true',
                        help='Debug option: skip the hit-and-run cross-check sampler')
    args = parser.parse_args()
    run(skip_hr=args.skip_hr)
