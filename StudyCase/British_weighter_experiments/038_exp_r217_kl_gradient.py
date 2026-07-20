# -*- coding: utf-8 -*-
"""
Gradient-share measurement for the forward-KL prior loss off-support
(floor-level) target term.

Motivation: in the forward-KL prior implementation, the auxiliary value v(a) is
zeroed outside the RCI mask → log(1+·) → floored at 1e-8 → normalized per source
to obtain q; inside the KL term both q and w are floored at 1e-8 on both sides.
The analytical concern this experiment addresses: for off-support agents, the
floor-level target term q_a = floor/normalizer is not exactly zero, so it may
still inject gradient into the weight w_a of those agents — what is the actual
share of the total gradient this contributes? This experiment turns that concern
into a measurable quantity and records it to disk. Pure CPU post-processing, no
retraining.

═══ Theory (metric definitions) ═══
    L_prior = mean_{s∈valid} Σ_a q_a (log q_a − log w_a)
    ∂L/∂w_a = −q_a / w_a × 1{w_a ≥ floor} / n_valid       (q does not depend on w;
    1{w≥floor} comes from the clamp gradient gate; boundary behaviour is measured
    and asserted at runtime)
    For each source s (L1 gradient mass decomposition):
        G_off(s) = Σ_{a∈off(s)} q_a/w_a·1{w_a≥floor}
        G_all(s) = Σ_{a∈s}      q_a/w_a·1{w_a≥floor}
        ratio_off(s) = G_off(s) / G_all(s)
    off(s) = edges where the log-transformed auxiliary value triggers the floor
    (log(1+v·rci) < 1e-8, i.e. agents that are RCI-masked or have zero auxiliary
    value). Max/median/mean are reported across all sources, along with the
    source with the smallest w singled out.

═══ q construction verified against the training implementation, term by term ═══
Training-side implementation (cross-checked line by line against this script,
and further verified pointwise via autograd on the actual loss classes):
    NTLPriorLoss        SpatialAllocation/GNN/Layer/LossFunction/NTLPriorLoss.py
    ProximityPriorLoss  same directory, ProximityPriorLoss.py
  Pipeline (identical between the two losses, only the auxiliary quantity differs):
    1. edge_v = aux[a_indices]; if rci_mask is present: edge_v *= rci_mask[a].float()
       (RCI zeroing; rci_mask = lu_res+com+ind > 0.5, precomputed and injected
       upstream)
    2. edge_v = log(1 + clamp(edge_v, min=0))                      (log1p)
    3. edge_v = clamp(edge_v, min=1e-8)                            (floor)
    4. v_sum = scatter_add per source; q = edge_v/(v_sum[s]+1e-8)  (per-source
       normalization)
    5. t_safe = clamp(q, 1e-8); w_safe = clamp(w, 1e-8)            (two-sided KL
       floor)
    6. kl_per_s = scatter_add(t_safe·(log t_safe − log w_safe));
       valid = v_sum > 1e-6; loss = kl_per_s[valid].mean()         (masked-source
       exclusion)
  Metadata wiring: EdgeWeightSolver.py (agent_ntl = graph['agent'].ntl_values,
  agent_proximity = .proximity_scores, agent_rci_mask = .rci_mask, num_s =
  source.num_nodes); w = EdgeWeightLayer scatter_softmax output (per-source
  Σw=1).
  Alignment verification methods (run in full for every seed×location×signal
  unit):
    A. Take ∂L/∂w via float32 autograd on the actual loss class and compare
       pointwise against the analytical expression
       −t_safe/w_safe·1{w≥floor}·1{s∈valid}/n_valid (max relative deviation
       recorded to disk);
    B. Compare the float32 loss value from the actual loss class against this
       script's float64 recomputation (within the tolerance expected from
       differing summation order);
    C. Runtime probe of the clamp boundary gradient behaviour (gradient = 1 at
       x == min → gate uses ≥).

═══ Scope ═══
    Configuration = UK ntl_prox (dual-prior, λ_ntl = λ_prox = 0.05, read from
    the CONFIG_MAP defined in 005_kfold_prior_training.py; λ only rescales the
    prior gradient as a whole and does not affect the off/all ratio); seeds
    {42,123,456}; for each seed and location, the fold in which that location
    is the TEST set (find_test_fold, using the same protocol and loading path
    as 033_exp_r21_r22_levelset.py) supplies grid_demands['gnn_demand'], from
    which w_a = gnn_demand_a / D_ITL3 is recovered (star graph: each agent has
    exactly one edge, and Σw≈1 per source serves as the anchor check). q does
    not depend on the model or the seed and is identical under any
    configuration; w is taken from the converged dual-prior training output.
    The source pool = 3 seeds × 16 locations × the number of ITL3 areas per
    location (each seed/location pair appears exactly once).

Determinism: fixed inputs, no randomness, no timestamps; the primary metric
uses float64 numpy (deterministic np.bincount summation); the autograd check
uses float32 on CPU.

═══ Data (read-only) ═══
    grid_demands + kfold_splits.json from the exp0 output for the ntl_prox
    configuration across 3 seeds; graph cache cached_graphs.pickle (edge
    structure plus injected ntl/proximity/rci attributes).

Outputs (results/exp_r217_kl_gradient/):
    kl_gradient_decomposition.json   primary output artifact (verdict
                                      sentences are rule-generated from the
                                      metrics, not hand-authored)
    per_source_ratios.csv            per (signal × seed × location × source)
                                      breakdown

Usage:
    python 034_exp_r217_kl_gradient.py
"""

# ── Environment guards: must run before importing numpy/torch ──
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')   # torch and MKL both link OpenMP; avoids a duplicate-runtime conflict
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ.setdefault('MPLBACKEND', 'Agg')              # non-interactive backend (this script does not plot; set defensively)

import json
import pickle
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Windows console defaults to cp1252, which cannot encode non-ASCII output —
# force UTF-8 (does not affect file outputs)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent   # repository root (where the SpatialAllocation package lives)
for _p in (str(SCRIPT_DIR), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Actual training loss classes (used for alignment checks A/B — not
# reimplementations, these are the exact classes used in training)
from SpatialAllocation.GNN.Layer.LossFunction.NTLPriorLoss import NTLPriorLoss
from SpatialAllocation.GNN.Layer.LossFunction.ProximityPriorLoss import ProximityPriorLoss

# ════════════════════════════════════════════════════════════
# Constants
# ════════════════════════════════════════════════════════════

EXP0_DIR = SCRIPT_DIR / 'results' / 'exp0_kfold_prior'
GRAPH_CACHE = EXP0_DIR / 'graph_cache' / 'cached_graphs.pickle'
OUTPUT_DIR = SCRIPT_DIR / 'results' / 'exp_r217_kl_gradient'
TRAINING_SCRIPT = SCRIPT_DIR / '005_kfold_prior_training.py'

CONFIG = 'ntl_prox'              # dual-prior configuration under evaluation
SEEDS = [42, 123, 456]
FLOOR = 1e-8                     # matches the floor value used in the training implementation
VALID_THRESHOLD = 1e-6           # threshold for excluding masked sources
SIGNALS = ['ntl', 'prox']

# ── Tolerance specification ──
TOL_SUM_W = 5e-4                 # Anchor: per-source |Σw − 1|. float32
                                 # scatter_softmax accumulates in edge order
                                 # over up to ~5.2e4 edges per source, giving
                                 # drift on the order of O(n·eps32) (observed
                                 # max 1.39e-4, seed456/TLC1); on top of this,
                                 # some fold grid_demands files are the
                                 # product of re-inference after training, a
                                 # known source of float32 drift in the exp0
                                 # outputs. Any genuine mismatch (wrong
                                 # fold/column/domain) would show up as
                                 # O(0.01+), so 5e-4 is still a tight anchor
                                 # (roughly five parts in ten thousand); and
                                 # since the ratio metric is invariant to an
                                 # overall rescale of w per source, this
                                 # anchor is used only to verify correct
                                 # pairing.
TOL_AUTOGRAD_REL = 1e-5          # autograd vs analytical gradient (same float32 pipeline, expected to agree near machine precision)
TOL_LOSSVAL_REL = 5e-4           # float32 actual loss value vs float64 recomputation (differing summation order)
RATIO_INERT_THRESHOLD = 0.01     # decision threshold: a ratio < 1% is considered negligible


def _status(ok: bool) -> str:
    """Status string derived programmatically from the numeric checks, not hand-written."""
    return 'ok' if ok else 'fail'


# ════════════════════════════════════════════════════════════
# Alignment check C: probe of the clamp boundary gradient behaviour
# ════════════════════════════════════════════════════════════

def probe_clamp_boundary_grad() -> list:
    """Measure the gradient of torch.clamp(min=FLOOR) at [x==min, x>min, x<min].

    Determines the analytical gradient gate: gradient=[1,1,0] ⇒ the gate is 1{w ≥ floor}.
    """
    x = torch.tensor([FLOOR, 2.0 * FLOOR, 0.5 * FLOOR],
                     dtype=torch.float32, requires_grad=True)
    torch.clamp(x, min=FLOOR).sum().backward()
    return [float(v) for v in x.grad]


# ════════════════════════════════════════════════════════════
# q pipeline reimplementation (float64 for the primary metric; each op
# corresponds 1:1 with the training implementation)
# ════════════════════════════════════════════════════════════

def q_pipeline_np(aux64: np.ndarray, rci: np.ndarray, ei: np.ndarray,
                  num_s: int) -> dict:
    """Reproduce steps 1-5 of the training implementation's q side term for
    term (float64; q does not depend on w or the seed).

    Returns:
        t_safe : (E,) target distribution actually used inside the KL term
                 (includes the q-side floor)
        off    : (E,) bool, off-support edges (the log1p value triggers the floor)
        off_rci_masked / off_zero_aux : decomposition of the causes of "off"
        v_sum  : (num_s,) auxiliary mass (summed after flooring; used for the
                 valid criterion)
        valid  : (num_s,) bool, sources that participate in the loss
    """
    s_idx, a_idx = ei[0], ei[1]
    edge_v = aux64[a_idx] * rci[a_idx].astype(np.float64)        # step 1: zero outside RCI
    v_log = np.log1p(np.maximum(edge_v, 0.0))                    # step 2: log(1+clamp(·,0))
    off = v_log < FLOOR                                          # floor triggered = off-support
    v_cl = np.maximum(v_log, FLOOR)                              # step 3: floor at 1e-8
    v_sum = np.bincount(s_idx, weights=v_cl, minlength=num_s)    # step 4: scatter_add
    q = v_cl / (v_sum[s_idx] + FLOOR)                            # step 4: per-source normalization
    t_safe = np.maximum(q, FLOOR)                                # step 5: q-side floor
    valid = v_sum > VALID_THRESHOLD                              # step 6: masked-source exclusion
    return {
        't_safe': t_safe, 'off': off,
        'off_rci_masked': off & ~rci[a_idx],
        'off_zero_aux': off & rci[a_idx],
        'v_sum': v_sum, 'valid': valid,
    }


def per_source_gradient_decomposition(qp: dict, w: np.ndarray, ei: np.ndarray,
                                      num_s: int) -> dict:
    """Decompose the L1 gradient mass per source: G_off / G_all / ratio_off plus diagnostic quantities."""
    s_idx = ei[0]
    t_safe, off = qp['t_safe'], qp['off']
    w_safe = np.maximum(w, FLOOR)
    gate = (w >= FLOOR).astype(np.float64)   # clamp boundary gradient = 1 (per probe measurement) → gate uses ≥
    g = t_safe / w_safe * gate               # |∂KL_s/∂w_a| (1/n_valid cancels out in the ratio)

    G_all = np.bincount(s_idx, weights=g, minlength=num_s)
    G_off = np.bincount(s_idx, weights=g * off, minlength=num_s)
    n_edges = np.bincount(s_idx, minlength=num_s)
    n_off = np.bincount(s_idx, weights=off.astype(np.float64),
                        minlength=num_s).astype(np.int64)
    n_w_below_floor = np.bincount(s_idx, weights=(w < FLOOR).astype(np.float64),
                                  minlength=num_s).astype(np.int64)
    n_off_gated = np.bincount(s_idx, weights=(off & (w < FLOOR)).astype(np.float64),
                              minlength=num_s).astype(np.int64)

    min_w = np.full(num_s, np.inf)
    np.minimum.at(min_w, s_idx, w)
    max_t_off = np.zeros(num_s)
    if off.any():
        np.maximum.at(max_t_off, s_idx[off], t_safe[off])

    with np.errstate(invalid='ignore', divide='ignore'):
        ratio = np.where(G_all > 0, G_off / G_all, np.nan)
    return {
        'G_all': G_all, 'G_off': G_off, 'ratio_off': ratio,
        'n_edges': n_edges, 'n_off': n_off, 'min_w': min_w,
        'n_w_below_floor': n_w_below_floor, 'n_off_gated': n_off_gated,
        'max_t_off': max_t_off,
    }


# ════════════════════════════════════════════════════════════
# Alignment checks A/B: autograd on the actual loss class vs analytical
# gradient / float64 recomputation
# ════════════════════════════════════════════════════════════

_LOSS_MODULES = {'ntl': NTLPriorLoss(), 'prox': ProximityPriorLoss()}


def autograd_alignment_check(signal: str, aux32: torch.Tensor, rci_t: torch.Tensor,
                             ei_t: torch.Tensor, num_s: int,
                             w64: np.ndarray, qp: dict) -> dict:
    """A: pointwise comparison of float32 autograd gradients from the actual
    loss class against the analytical expression;
    B: comparison of the float32 actual loss value against the float64
    recomputed value."""
    w32 = torch.tensor(w64, dtype=torch.float32, requires_grad=True)
    metadata = {
        'agent_ntl': aux32 if signal == 'ntl' else None,
        'agent_proximity': aux32 if signal == 'prox' else None,
        'agent_rci_mask': rci_t,
        'num_s': num_s,
    }
    loss = _LOSS_MODULES[signal](w32, ei_t, metadata)
    loss.backward()
    g_auto = w32.grad.detach()

    # Analytical gradient (float32, using exactly the same ops as the loss's internal pipeline)
    with torch.no_grad():
        s, a = ei_t[0], ei_t[1]
        v = aux32[a] * rci_t[a].float()
        v = torch.log(1.0 + torch.clamp(v, min=0.0))
        v = torch.clamp(v, min=FLOOR)
        v_sum = torch.zeros(num_s, dtype=torch.float32).scatter_add_(0, s, v)
        t_safe32 = torch.clamp(v / (v_sum[s] + FLOOR), min=FLOOR)
        wd = w32.detach()
        gate = (wd >= FLOOR).float()
        valid = v_sum > VALID_THRESHOLD
        n_valid = valid.sum().float()
        g_ana = -(t_safe32 / torch.clamp(wd, min=FLOOR)) * gate \
            * valid[s].float() / n_valid
    denom = float(g_auto.abs().max())
    grad_rel_dev = float((g_auto - g_ana).abs().max()) / denom if denom > 0 else 0.0

    # B: float64 recomputed loss value (uses the same t_safe/valid as the primary metric)
    s_np = ei_t[0].numpy()
    w_safe64 = np.maximum(w64, FLOOR)
    kl_terms = qp['t_safe'] * (np.log(qp['t_safe']) - np.log(w_safe64))
    kl_per_s = np.bincount(s_np, weights=kl_terms, minlength=num_s)
    loss64 = float(kl_per_s[qp['valid']].mean())
    loss32 = float(loss.item())
    loss_rel_dev = abs(loss32 - loss64) / max(abs(loss64), 1e-12)

    return {'grad_rel_dev': grad_rel_dev, 'loss_rel_dev': loss_rel_dev,
            'loss_float32': loss32, 'loss_float64': loss64,
            'n_valid': int(valid.sum())}


# ════════════════════════════════════════════════════════════
# Data extraction and structural verification (graph cache → compact
# per-unit inputs)
# ════════════════════════════════════════════════════════════

def extract_and_verify_loc(loc: str, graph, grid_gdf, region_sub,
                           ntl_from_cache: np.ndarray) -> dict:
    """Extract the compact per-location inputs and verify the structural
    preconditions (raises on the first failure).

    P1 star topology: each agent has exactly one source→agent edge
       (precondition for recovering w);
    P2 index identity: both agent_index_map and grid_gdf.index are 0..n-1
       positional sequences (precondition for grid_demands array indices to
       match the graph's agent node indices);
    P3 domain alignment: each edge's source ITL3 == its agent's ITL3;
    P4 column decoding (same check as V3 in 033_exp_r21_r22_levelset.py):
       landuse_mapping_matrix columns ↔ (edge source, grid landuse category)
       — direct evidence that agent node order and grid row order correspond
       positionally;
    P5 injection consistency: graph['agent'].ntl_values == cached
       ntl_dict[loc] (float32 storage).
    """
    ei = graph['source', 'connects_to', 'agent'].edge_index.numpy()
    n_agents = int(graph['agent'].num_nodes)
    num_s = int(graph['source'].num_nodes)

    checks = {}
    checks['P1_star'] = bool((np.bincount(ei[1], minlength=n_agents) == 1).all())
    aim = np.asarray(graph.agent_index_map.values)
    checks['P2_index_identity'] = bool(
        (aim == np.arange(n_agents)).all()
        and (np.asarray(grid_gdf.index.values) == np.arange(n_agents)).all())
    sim = graph.source_index_map
    src_itl3 = region_sub.loc[sim.values, 'ITL3'].values
    checks['P3_itl3_domain'] = bool(
        (src_itl3[ei[0]] == grid_gdf['ITL3'].values[ei[1]]).all())
    M = graph.landuse_mapping_matrix.numpy()
    cats = sorted(grid_gdf['landuse'].unique())
    n_lu = len(cats)
    cat_to_idx = {c: i for i, c in enumerate(cats)}
    lu_codes = grid_gdf['landuse'].map(cat_to_idx).values.astype(np.int64)
    col_idx = M.argmax(axis=1)
    checks['P4_column_decode'] = bool(
        ((M != 0).sum(axis=1) == 1).all()
        and ((col_idx // n_lu) == ei[0]).all()
        and ((col_idx % n_lu) == lu_codes[ei[1]]).all())
    checks['P5_ntl_injection_match'] = bool(np.array_equal(
        graph['agent'].ntl_values.numpy(),
        ntl_from_cache.astype(np.float32)))
    if not all(checks.values()):
        raise AssertionError(f'{loc}: structural verification failed {checks}')

    return {
        'checks': checks,
        'ei': ei, 'n_agents': n_agents, 'num_s': num_s,
        'itl3_of_source': [str(x) for x in src_itl3],
        'D_src': region_sub.loc[sim.values, 'Demand (MVA)'].values.astype(np.float64),
        'aux64': {
            'ntl': graph['agent'].ntl_values.numpy().astype(np.float64),
            'prox': graph['agent'].proximity_scores.numpy().astype(np.float64),
        },
        'rci': graph['agent'].rci_mask.numpy().astype(bool),
        'aux32': {
            'ntl': graph['agent'].ntl_values.clone(),
            'prox': graph['agent'].proximity_scores.clone(),
        },
        'rci_t': graph['agent'].rci_mask.clone(),
        'ei_t': graph['source', 'connects_to', 'agent'].edge_index.clone(),
    }


def find_test_fold(seed: int, loc: str) -> str:
    """Find the fold in which loc is the test set, from kfold_splits.json
    (exactly one fold; same approach as 033_exp_r21_r22_levelset.py)."""
    with open(EXP0_DIR / f'seed_{seed}' / CONFIG / 'kfold_splits.json',
              encoding='utf-8') as f:
        splits = json.load(f)
    folds = [k for k, v in splits.items() if loc in v['test']]
    assert len(folds) == 1, f'seed={seed} loc={loc}: number of test folds {len(folds)} != 1'
    return folds[0].replace('fold_', 'fold')


def read_config_objective() -> dict:
    """Read the ntl_prox objective weights from 005_kfold_prior_training.py
    via importlib (a disclosure field, generated rather than hand-written)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('kfold_prior_training_005',
                                                  TRAINING_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return dict(mod.CONFIG_MAP[CONFIG]['objective_weights'])


# ════════════════════════════════════════════════════════════
# Main pipeline
# ════════════════════════════════════════════════════════════

def run() -> None:
    import shared_correction_utils as scu
    all_locations = scu.ALL_LOCATIONS

    # ── 0. Objective weights disclosure (read from source) + clamp boundary probe ──
    objective = read_config_objective()
    print(f'{CONFIG} objective weights (read from 005_kfold_prior_training.py): {objective}')
    clamp_probe = probe_clamp_boundary_grad()
    print(f'clamp(min=floor) boundary gradient probe [x==min, x>min, x<min]: {clamp_probe}')
    clamp_probe_ok = clamp_probe == [1.0, 1.0, 0.0]
    if not clamp_probe_ok:
        raise AssertionError(
            f'unexpected clamp boundary gradient behaviour {clamp_probe} — the analytical gate 1{{w≥floor}} no longer holds')

    # ── 1. Load graph cache + structural verification + compact extraction ──
    print(f'Loading graph cache: {GRAPH_CACHE}')
    with open(GRAPH_CACHE, 'rb') as f:
        cache = pickle.load(f)
    graphs, grids = cache['graphs'], cache['grids']
    ntl_dict, region_dict = cache['ntl_dict'], cache['region_dict']

    loc_data = {}
    structure_checks = {}
    for loc in all_locations:
        grid_gdf, _ = grids[loc]
        info = extract_and_verify_loc(loc, graphs[loc], grid_gdf,
                                      region_dict[loc], ntl_dict[loc])
        loc_data[loc] = info
        structure_checks[loc] = info['checks']
        print(f'  {loc}: structural verification passed (n_agents={info["n_agents"]}, '
              f'n_source={info["num_s"]})')
        del graphs[loc]   # free the large supervision-matrix tensors, etc.
    del graphs, cache, grids

    # ── 2. q-side pipeline (seed-independent, computed once per location × signal) ──
    q_side = {}
    for loc in all_locations:
        info = loc_data[loc]
        q_side[loc] = {sig: q_pipeline_np(info['aux64'][sig], info['rci'],
                                          info['ei'], info['num_s'])
                       for sig in SIGNALS}

    # ── 3. 48 units × 2 signals: recover w + gradient decomposition + alignment checks ──
    rows = []
    sum_w_dev_max = 0.0
    grad_rel_dev_max = {sig: 0.0 for sig in SIGNALS}
    loss_rel_dev_max = {sig: 0.0 for sig in SIGNALS}

    for seed in SEEDS:
        for loc in all_locations:
            info = loc_data[loc]
            ei = info['ei']
            fold = find_test_fold(seed, loc)
            gd_path = (EXP0_DIR / f'seed_{seed}' / CONFIG / fold /
                       'grid_demands' / f'{loc}_grid_demands.pickle')
            with open(gd_path, 'rb') as f:
                grid_demands = pickle.load(f)
            d_raw = np.asarray(grid_demands['gnn_demand'], dtype=np.float64)
            assert len(d_raw) == info['n_agents'], \
                f'seed={seed} {loc}: grid_demands length {len(d_raw)} != number of agents'

            # Recover w (star graph: each agent has exactly one edge); Σw≈1 anchor check
            w = d_raw[ei[1]] / info['D_src'][ei[0]]
            sums = np.bincount(ei[0], weights=w, minlength=info['num_s'])
            sum_w_dev_max = max(sum_w_dev_max, float(np.abs(sums - 1.0).max()))

            for sig in SIGNALS:
                qp = q_side[loc][sig]
                dec = per_source_gradient_decomposition(qp, w, ei, info['num_s'])
                chk = autograd_alignment_check(
                    sig, info['aux32'][sig], info['rci_t'], info['ei_t'],
                    info['num_s'], w, qp)
                grad_rel_dev_max[sig] = max(grad_rel_dev_max[sig],
                                            chk['grad_rel_dev'])
                loss_rel_dev_max[sig] = max(loss_rel_dev_max[sig],
                                            chk['loss_rel_dev'])

                for s in range(info['num_s']):
                    rows.append({
                        'signal': sig, 'seed': seed, 'location': loc,
                        'fold': fold, 'source_idx': s,
                        'itl3': info['itl3_of_source'][s],
                        'n_edges': int(dec['n_edges'][s]),
                        'n_off': int(dec['n_off'][s]),
                        'off_frac': float(dec['n_off'][s] / dec['n_edges'][s]),
                        'aux_mass_v_sum': float(qp['v_sum'][s]),
                        'valid': bool(qp['valid'][s]),
                        'G_all': float(dec['G_all'][s]),
                        'G_off': float(dec['G_off'][s]),
                        'ratio_off': float(dec['ratio_off'][s]),
                        'min_w': float(dec['min_w'][s]),
                        'n_w_below_floor': int(dec['n_w_below_floor'][s]),
                        'n_off_gradient_gated': int(dec['n_off_gated'][s]),
                        'max_t_off': float(dec['max_t_off'][s]),
                    })
            print(f'  seed={seed} {loc} [{fold}]: '
                  f'Σw deviation {float(np.abs(sums - 1.0).max()):.2e}, '
                  f'autograd check passed (cumulative max deviation '
                  f'ntl={grad_rel_dev_max["ntl"]:.2e}, '
                  f'prox={grad_rel_dev_max["prox"]:.2e})')

    df = pd.DataFrame(rows)

    # ── 4. Aggregate per signal (pool of sources with valid and G_all>0) ──
    def aggregate_signal(sig: str) -> dict:
        sub = df[df['signal'] == sig]
        pool = sub[sub['valid'] & (sub['G_all'] > 0)]
        excluded = sub[~sub['valid']]
        zero_grad = sub[sub['valid'] & (sub['G_all'] <= 0)]
        r = pool['ratio_off'].values

        # Single out the source with the smallest w (within the valid pool;
        # np.argmin breaks ties by taking the first in iteration order)
        i_minw = int(np.argmin(pool['min_w'].values))
        row_minw = pool.iloc[i_minw]
        # The source with the largest ratio (report the worst case as-is)
        i_worst = int(np.argmax(r))
        row_worst = pool.iloc[i_worst]

        def _src_block(row) -> dict:
            return {
                'seed': int(row['seed']), 'location': str(row['location']),
                'itl3': str(row['itl3']), 'source_idx': int(row['source_idx']),
                'ratio_off': float(row['ratio_off']),
                'min_w': float(row['min_w']),
                'n_edges': int(row['n_edges']), 'n_off': int(row['n_off']),
                'off_frac': float(row['off_frac']),
            }

        # Edge-level statistics (q side is seed-independent — use the slice for the first seed to avoid triple-counting)
        sub_one_seed = sub[sub['seed'] == SEEDS[0]]
        n_edges_total = int(sub_one_seed['n_edges'].sum())
        n_edges_off = int(sub_one_seed['n_off'].sum())

        return {
            'n_sources': int(len(pool)),
            'n_sources_excluded_by_valid_mask': int(len(excluded)),
            'n_sources_zero_gradient': int(len(zero_grad)),
            'ratio_off_max': float(np.max(r)),
            'ratio_off_median': float(np.median(r)),
            'ratio_off_mean': float(np.mean(r)),
            'ratio_off_p90': float(np.percentile(r, 90)),
            'ratio_off_p99': float(np.percentile(r, 99)),
            'global_ratio_off': float(pool['G_off'].sum() / pool['G_all'].sum()),
            'n_sources_ratio_ge_1pct': int((r >= RATIO_INERT_THRESHOLD).sum()),
            'per_seed': {
                str(seed): {
                    'ratio_off_max': float(np.max(g['ratio_off'].values)),
                    'ratio_off_median': float(np.median(g['ratio_off'].values)),
                    'ratio_off_mean': float(np.mean(g['ratio_off'].values)),
                }
                for seed, g in pool.groupby('seed')
            },
            'edge_stats_per_seed_scope': {
                'note': 'the off mask is determined by the q side and is seed-independent; figures here are for a single seed across 16 locations',
                'n_edges_total': n_edges_total,
                'n_edges_off': n_edges_off,
                'off_edge_share': float(n_edges_off / n_edges_total),
            },
            'n_off_edges_gradient_gated_pool': int(pool['n_off_gradient_gated'].sum()),
            'max_t_off_after_clamp': float(sub['max_t_off'].max()),
            'min_w_source': _src_block(row_minw),
            'worst_ratio_source': _src_block(row_worst),
        }

    per_signal = {sig: aggregate_signal(sig) for sig in SIGNALS}

    # Decompose the causes of "off" (q side; independent of the seed/w; computed once per signal)
    off_breakdown = {}
    for sig in SIGNALS:
        n_rci, n_zero, n_tot = 0, 0, 0
        for loc in all_locations:
            qp = q_side[loc][sig]
            n_rci += int(qp['off_rci_masked'].sum())
            n_zero += int(qp['off_zero_aux'].sum())
            n_tot += len(qp['off'])
        off_breakdown[sig] = {
            'n_edges_total': n_tot,
            'n_off_rci_masked': n_rci,
            'n_off_zero_aux_within_rci': n_zero,
        }

    # ── 5. Verdict sentences (rule-generated, not hand-authored) ──
    verdict_rule = (
        f'per_signal: ratio_off_max < {RATIO_INERT_THRESHOLD} → negligible (inert); '
        f'ratio_off_median < {RATIO_INERT_THRESHOLD} ≤ ratio_off_max → '
        'mostly negligible with some exceptions (mostly_inert); otherwise → '
        'non_negligible. '
        'overall: both signals inert → the floor term has no material effect; '
        'otherwise report the graded classification as-is.')

    def _classify(st: dict) -> str:
        if st['ratio_off_max'] < RATIO_INERT_THRESHOLD:
            return 'inert'
        if st['ratio_off_median'] < RATIO_INERT_THRESHOLD:
            return 'mostly_inert'
        return 'non_negligible'

    def _sentence(sig: str, st: dict) -> str:
        cls = _classify(st)
        if cls == 'inert':
            return (f'{sig}: the off-support floor-term gradient ratio is below 1% '
                    f'for all {st["n_sources"]} sources (max={st["ratio_off_max"]:.3e}, '
                    f'median={st["ratio_off_median"]:.3e}, '
                    f'mean={st["ratio_off_mean"]:.3e}) — negligible magnitude; '
                    'the floor term has no material effect on the training gradient.')
        if cls == 'mostly_inert':
            return (f'{sig}: the off-support floor-term gradient ratio has a median '
                    f'below 1% (median={st["ratio_off_median"]:.3e}) but '
                    f'{st["n_sources_ratio_ge_1pct"]}/{st["n_sources"]} sources exceed 1% '
                    f'(max={st["ratio_off_max"]:.3e}) — mostly negligible, with some '
                    'exceptions.')
        return (f'{sig}: the off-support floor-term gradient ratio is not negligible '
                f'(median={st["ratio_off_median"]:.3e}, '
                f'max={st["ratio_off_max"]:.3e}, '
                f'{st["n_sources_ratio_ge_1pct"]}/{st["n_sources"]} sources ≥1%) '
                '— the floor term contributes a measurable share of the training '
                'gradient for this signal and should be accounted for.')

    classes = {sig: _classify(per_signal[sig]) for sig in SIGNALS}
    if all(c == 'inert' for c in classes.values()):
        overall = ('For both signals (NTL/Prox), the off-support floor-term gradient '
                   'ratio is negligible (<1%) across all sources — the floor term '
                   'does not materially contaminate the training gradient.')
    else:
        overall = ('Per-signal classification: ' + ', '.join(f'{s}={classes[s]}' for s in SIGNALS)
                   + ' — a non-negligible component is present; reported as-is per '
                   'signal without downplaying it.')

    verdict = {
        'rule': verdict_rule,
        'per_signal_class': classes,
        'ntl': _sentence('ntl', per_signal['ntl']),
        'prox': _sentence('prox', per_signal['prox']),
        'overall': overall,
    }

    # ── 6. Preconditions (expected/actual/status) ──
    preconditions = {
        'P1_structure_all_locations': {
            'description': ('P1 star topology / P2 index identity / P3 ITL3 domain '
                            'alignment / P4 column decoding / P5 NTL injection '
                            'consistency pass for all 16 locations'),
            'expected': True,
            'actual': all(all(c.values()) for c in structure_checks.values()),
            'status': _status(all(all(c.values())
                                  for c in structure_checks.values())),
        },
        'P2_objective_dual_prior': {
            'description': 'the ntl_prox objective includes both priors (read from the CONFIG_MAP in 005_kfold_prior_training.py)',
            'expected': {'landuse_prediction_loss': 1.0,
                         'ntl_prior': 0.05, 'proximity_prior': 0.05},
            'actual': objective,
            'status': _status(objective == {'landuse_prediction_loss': 1.0,
                                            'ntl_prior': 0.05,
                                            'proximity_prior': 0.05}),
        },
        'P3_sum_w_anchor': {
            'description': (f'max per-source |Σw − 1| < {TOL_SUM_W:g}'
                            '(float32 scatter_softmax output plus re-inference drift in the cached outputs)'),
            'expected': f'< {TOL_SUM_W:g}',
            'actual': sum_w_dev_max,
            'status': _status(sum_w_dev_max < TOL_SUM_W),
        },
        'P4_clamp_boundary_probe': {
            'description': ('clamp(min=floor) gradient probe [x==min, x>min, x<min]'
                            ' — basis for the analytical gate 1{w≥floor}'),
            'expected': [1.0, 1.0, 0.0],
            'actual': clamp_probe,
            'status': _status(clamp_probe_ok),
        },
        'P5_autograd_alignment': {
            'description': (f'max relative deviation between autograd gradients from the '
                            f'actual loss class and the analytical expression '
                            f'< {TOL_AUTOGRAD_REL:g} (all 96 units, same float32 pipeline)'),
            'expected': f'< {TOL_AUTOGRAD_REL:g}',
            'actual': grad_rel_dev_max,
            'status': _status(max(grad_rel_dev_max.values()) < TOL_AUTOGRAD_REL),
        },
        'P6_loss_value_alignment': {
            'description': (f'max relative deviation between the float32 actual loss '
                            f'value and the float64 recomputation < {TOL_LOSSVAL_REL:g}'),
            'expected': f'< {TOL_LOSSVAL_REL:g}',
            'actual': loss_rel_dev_max,
            'status': _status(max(loss_rel_dev_max.values()) < TOL_LOSSVAL_REL),
        },
    }
    failed = [k for k, v in preconditions.items() if v['status'] != 'ok']
    if failed:
        raise AssertionError(f'precondition checks failed: {failed}')

    # ── 7. Assemble and save results ──
    results = {
        'config': CONFIG,
        'seeds': SEEDS,
        'floor': FLOOR,
        'valid_threshold': VALID_THRESHOLD,
        'scope': {
            'description': (
                'UK ntl_prox (dual-prior) configuration; for each seed and location, '
                'the fold in which that location is the TEST set (find_test_fold, '
                'using the same protocol and loading path as '
                "033_exp_r21_r22_levelset.py) supplies grid_demands['gnn_demand'], "
                'from which w_a = gnn_demand_a / D_ITL3 is recovered via the star '
                'graph (Σw≈1 per source serves as the anchor check). q is '
                'reconstructed term for term from the ntl_values/proximity_scores/'
                'rci_mask injected into the graph cache, and does not depend on the '
                'model or the seed. The source pool = 3 seeds × 16 locations × the '
                'number of ITL3 areas per location (each seed/location pair appears '
                'exactly once). w comes from the converged training output — this '
                'experiment answers "what share of the gradient near the '
                'convergence point comes from the floor term".'),
            'n_units': len(SEEDS) * len(all_locations),
            'locations': list(all_locations),
            'w_source': ('exp0_kfold_prior/seed_{seed}/ntl_prox/{test_fold}/'
                         'grid_demands/{loc}_grid_demands.pickle → gnn_demand'),
            'objective_weights_from_source': objective,
            'lambda_note': ('λ_ntl = λ_prox = 0.05 only rescales each prior '
                            'gradient as a whole, and does not affect the '
                            'off/all ratio within a single prior.'),
        },
        'metric_definition': {
            'gradient': '∂L/∂w_a = −q_a/w_a × 1{w_a≥floor} / n_valid (q does not depend on w)',
            'G_off': 'Σ_{a∈off(s)} q_a/w_a·1{w_a≥floor} (off = edges where the log1p auxiliary value triggers the floor)',
            'G_all': 'Σ_{a∈s} q_a/w_a·1{w_a≥floor}',
            'ratio_off': 'G_off(s)/G_all(s), per-source; q is t_safe as actually used inside the KL term',
            'precision': ('primary metric uses float64 numpy (deterministic '
                          'np.bincount summation); deviation from the float32 '
                          'training pipeline is bounded by the P5/P6 checks'),
        },
        'alignment': {
            'loss_implementation_files': {
                'ntl': 'SpatialAllocation/GNN/Layer/LossFunction/NTLPriorLoss.py',
                'prox': 'SpatialAllocation/GNN/Layer/LossFunction/ProximityPriorLoss.py',
                'metadata_wiring': 'SpatialAllocation/GNN/core/EdgeWeightSolver.py',
                'edge_weights': ('SpatialAllocation/GNN/Layer/EdgeWeightLayer.py '
                                 'scatter_softmax (Σw=1 per source)'),
            },
            'q_pipeline_steps_verbatim': [
                '1. edge_v = aux[a_indices]; edge_v *= rci_mask[a].float() (zero outside RCI)',
                '2. edge_v = log(1 + clamp(edge_v, min=0)) (log1p)',
                '3. edge_v = clamp(edge_v, min=1e-8) (floor)',
                '4. v_sum = scatter_add per source; q = edge_v/(v_sum[s]+1e-8) (per-source normalization)',
                '5. t_safe = clamp(q, 1e-8); w_safe = clamp(w, 1e-8) (two-sided KL floor)',
                '6. valid = v_sum > 1e-6; loss = kl_per_s[valid].mean() (masked-source exclusion)',
            ],
            'verification': {
                'A_autograd_grad_max_rel_dev': grad_rel_dev_max,
                'B_loss_value_max_rel_dev': loss_rel_dev_max,
                'C_clamp_boundary_probe': clamp_probe,
                'note': ('A/B are run in full over 96 units (= 3 seeds×16 '
                         'locations×2 signals): A compares ∂L/∂w from the actual '
                         'loss class (not a reimplementation), via float32 '
                         'autograd, pointwise against the analytical expression; '
                         'B cross-checks the primary metric pipeline using a '
                         'float64 recomputed loss value; C determines the clamp '
                         'boundary gradient gate operator.'),
            },
        },
        'preconditions': preconditions,
        'off_support_breakdown': off_breakdown,
        'per_signal': per_signal,
        'verdict': verdict,
        'diagnostics': {
            'sum_w_dev_max': sum_w_dev_max,
            'n_rows_per_source_csv': len(df),
        },
        'meta': {
            'script': '034_exp_r217_kl_gradient.py',
            'plan_step': 'off-support floor-term gradient contribution measurement',
            'determinism': 'fixed inputs, no randomness, no timestamps; primary metric uses float64 bincount',
            'python': platform.python_version(),
            'numpy': np.__version__, 'pandas': pd.__version__,
            'torch': torch.__version__,
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DIR / 'per_source_ratios.csv', index=False)
    with open(OUTPUT_DIR / 'kl_gradient_decomposition.json', 'w',
              encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # ── 8. Summary printout ──
    print('\n' + '=' * 68)
    print('Off-support floor-term gradient ratio — summary')
    print('=' * 68)
    for key, item in preconditions.items():
        print(f'  {key}: {item["status"]}')
    for sig in SIGNALS:
        st = per_signal[sig]
        print(f'\n[{sig}] n_sources={st["n_sources"]} '
              f'(excluded {st["n_sources_excluded_by_valid_mask"]}, '
              f'zero-gradient {st["n_sources_zero_gradient"]})')
        print(f'  ratio_off: max={st["ratio_off_max"]:.3e}  '
              f'median={st["ratio_off_median"]:.3e}  '
              f'mean={st["ratio_off_mean"]:.3e}  '
              f'p99={st["ratio_off_p99"]:.3e}')
        print(f'  global ratio ΣG_off/ΣG_all = {st["global_ratio_off"]:.3e}; '
              f'off-edge share = {st["edge_stats_per_seed_scope"]["off_edge_share"]:.3f}')
        mw = st['min_w_source']
        print(f'  source with smallest w: seed={mw["seed"]} {mw["location"]}/{mw["itl3"]} '
              f'min_w={mw["min_w"]:.3e} → ratio_off={mw["ratio_off"]:.3e}')
        wr = st['worst_ratio_source']
        print(f'  source with largest ratio: seed={wr["seed"]} {wr["location"]}/{wr["itl3"]} '
              f'→ ratio_off={wr["ratio_off"]:.3e} (off_frac={wr["off_frac"]:.3f})')
    print(f'\nVerdict: {verdict["overall"]}')
    print(f'\nOutputs written to {OUTPUT_DIR}')


if __name__ == '__main__':
    run()
