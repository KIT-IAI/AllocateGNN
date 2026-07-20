# -*- coding: utf-8 -*-
"""
025 - Condition proposition phase diagram

Theoretical proposition (derived in log space) for when multiplicative
vs. additive correction reduces error: let r be the vector of log
residual ratios and c the vector of centered log correction factors.
Multiplicative correction followed by renormalization reduces error if
and only if 2<r,c> > ||c||^2, i.e. **alpha = rho(r,c) > sigma_c / (2*sigma_r)**.
This experiment validates that boundary with a synthetic phase diagram
and overlays the 240 empirical combinations from the exp_r25 experiment.

=== Layer definitions ===
[Synthetic layer A - strict theoretical domain - quantitative assertions]
    Agent-level, no aggregation. Generates log-normal base residuals r
    (sigma_r fixed) and correction factors c (controlling sigma_c and
    corr(r,c)=alpha), applies multiplicative correction + renorm on an
    (alpha, sigma_c/sigma_r) grid, and averages over repeated draws to
    reduce noise while testing whether each cell's MSE improves. Three
    variants run in parallel:
      1. geo_log -- renorm = log-domain centering of c (geometric
         normalization, the strict implementation of "centering c" in
         the theoretical derivation), MSE in log space. **The theorem's
         strict domain**: the boundary is asserted to deviate from the
         theoretical line alpha* = sigma_ratio/2 by at most 2 grid steps.
      2. arith_log -- renorm = the real pipeline's arithmetic mass
         conservation (a positive scalar per region), MSE still in log
         space. A pipeline-faithful variant, also asserted to deviate by
         at most 2 grid steps (measured systematic offset ~= 0.07 <
         tolerance 0.1, arising from the Jensen gap between the
         arithmetic renorm constant log(sum(w*e^c)) ~= sigma_c^2/2 and
         geometric centering).
      3. arith_demand -- arithmetic renorm + demand-space MSE (the real
         evaluation convention). The boundary shape is preserved but
         shifts upward systematically at high sigma_ratio (measured
         maximum offset ~= 0.19, and the sigma_ratio=1.6 row has no
         help region within alpha <= 0.95) -- **recorded only, not
         asserted**; the aggregated version of this convention is
         covered qualitatively by layer B.
[Synthetic layer B - qualitative - no quantitative assertions]
    Generated the same way as layer A, but with an added Voronoi-style
    aggregation step: agents are grouped and summed according to the
    **true Voronoi group sizes of the 16 real regions** (the bincount
    of an assignment computed directly from frozen data), and help/hurt
    is judged from substation-level demand-space RMSE. The
    post-aggregation (alpha_meas, sigma_ratio_meas) coordinates are also
    measured, following exactly the same protocol used for the exp_r25
    empirical points (epsilon = total regional demand * 1e-6), to give
    the overlay of empirical points coordinate consistency. Only
    qualitative checks are performed: whether the boundary is
    non-decreasing, and the positional offset relative to the
    theoretical line, both recorded in the JSON output.
[Empirical placement]
    The exp_r25 experiment's 240 (base, region, signal) combinations are
    placed by (pearson_log, sigma_ratio) coordinates; the theoretical
    prediction help iff pearson_log > sigma_ratio/2 is compared against
    the actual delta-RMSE sign in a 2x2 confusion matrix; three
    propositions are checked numerically (verdict fields are always
    generated from numeric rules, never hand-written):
      P1 static arms (uniform+gpm, 96 rows) fall in the help region;
      P2 the combined GNN arm (gnn x NP, 48 rows) falls in the hurt
         region;
      P3 dual-signal variance superadditivity: for each (base_id,
         location), check sigma_c(NP)^2 > sigma_c(N)^2 + sigma_c(P)^2
         (80 combinations).
    Verdict rule: value >= 0.9 -> supported; >= 0.5 ->
    partially_supported; otherwise not_supported (value is defined per
    proposition field).

=== Correspondence with the theoretical boundary ===
The layer A/B phase-diagram coordinates are the generation parameters
(alpha, sigma_c/sigma_r); the empirical/layer-B measured coordinates are
substation-level (corr(log F, log rho), std(log F)/std(log rho)). The
two are identical at the agent level (log F = c - a constant, and both
correlation and std are invariant to constants); the post-aggregation
coordinate drift is quantified in layer B's coord_drift fields.

Output artifacts:
    results/exp_r26/phase_diagram_data.npz    -- full layer A/B grids + layer B scatter points
    results/exp_r26/empirical_points.csv      -- placement detail for the 240 empirical points
    results/exp_r26/proposition_checks.json   -- confusion matrix + verdicts for the three propositions

CPU-only, fixed seed (np.random.default_rng(20260714)). All pre-existing
output artifacts are treated as read-only inputs.

Usage:
    python 025_exp_r26_condition_phase_diagram.py
"""

import sys
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import shared_correction_utils as scu  # noqa: E402

# ════════════════════════════════════════════════════════════
# Paths and constants
# ════════════════════════════════════════════════════════════

S3_DIR = SCRIPT_DIR / 'results' / 'exp_r25'
OUTPUT_DIR = SCRIPT_DIR / 'results' / 'exp_r26'

RNG_SEED = 20260714

# Phase-diagram grid: alpha covers the empirical pearson_log range
# [-0.17, 0.93], sigma_ratio covers the empirical sigma_ratio range
# [0.09, 1.47]
ALPHA_STEP = 0.05
ALPHA_GRID = np.round(np.arange(-0.20, 0.951, ALPHA_STEP), 10)
SRATIO_STEP = 0.1
SRATIO_GRID = np.round(np.arange(0.1, 1.601, SRATIO_STEP), 10)

# Layer A: sigma_r fixed at 0.7 (~= the median sigma_r=0.719 of the GNN
# base in the exp_r25 experiment; the theoretical boundary depends only
# on the ratio sigma_c/sigma_r, so the value of sigma_r only affects the
# absolute scale, not the boundary); base heterogeneity sigma_b=1
SIGMA_R = 0.7
SIGMA_B = 1.0
N_AGENTS_A = 2000
N_REPS_A = 60

# Layer B: repeats per cell = 2 cycles x 16 real regions = 32 replicates
N_CYCLES_B = 2

# Assertion tolerance = grid step x 2 (task specification)
BOUNDARY_TOL = 2 * ALPHA_STEP

# Verdict thresholds for the three propositions (numeric rule, never hand-written)
VERDICT_TIERS = [(0.9, 'supported'), (0.5, 'partially_supported')]
VERDICT_RULE_TEXT = ('value >= 0.9 -> supported; value >= 0.5 -> '
                     'partially_supported; else not_supported')

EPS_RATIO = 1e-6            # same protocol as exp_r25: epsilon = total regional demand * 1e-6


# ════════════════════════════════════════════════════════════
# Shared helpers
# ════════════════════════════════════════════════════════════

def find_boundary(delta_grid: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    """For each sigma_ratio row, find the alpha where delta transitions
    from >=0 to <0 (linear interpolation between the first negative cell
    and the preceding cell).

    A row with no negative cell -> NaN (that row has no help region
    within the grid range).
    """
    out = []
    for row in delta_grid:
        idx = np.where(row < 0)[0]
        if len(idx) == 0:
            out.append(np.nan)
            continue
        i = idx[0]
        if i == 0:
            out.append(float(alphas[0]))
            continue
        x0, x1 = alphas[i - 1], alphas[i]
        y0, y1 = row[i - 1], row[i]
        out.append(float(x0 + (x1 - x0) * (0.0 - y0) / (y1 - y0)))
    return np.array(out)


def verdict_of(value: float) -> str:
    """Numeric rule -> verdict field (never hand-written)."""
    for threshold, label in VERDICT_TIERS:
        if value >= threshold:
            return label
    return 'not_supported'


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation (zero variance -> NaN; the caller is
    responsible for handling NaNs)."""
    sx, sy = x.std(), y.std()
    if sx == 0 or sy == 0:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])


# ════════════════════════════════════════════════════════════
# Synthetic layer A (strict theoretical domain)
# ════════════════════════════════════════════════════════════

def simulate_cell_layer_a(alpha: float, sratio: float, rng) -> dict:
    """One (alpha, sigma_ratio) grid cell: three-way delta-MSE over
    N_REPS_A repetitions.

    Generation: r = sigma_r * z1; c = sigma_c * (alpha*z1 +
    sqrt(1-alpha^2)*z2) (population corr = alpha); log d_base =
    sigma_b*z_b; d_true = d_base*e^r, then rescaled so that
    sum(d_true) = sum(d_base) (the mass-consistency precondition of the
    real pipeline: the base and the ground truth share the same
    regional total).
    """
    sigma_c = sratio * SIGMA_R
    z1 = rng.standard_normal((N_REPS_A, N_AGENTS_A))
    z2 = rng.standard_normal((N_REPS_A, N_AGENTS_A))
    zb = rng.standard_normal((N_REPS_A, N_AGENTS_A))

    r = SIGMA_R * z1
    c = sigma_c * (alpha * z1 + np.sqrt(1.0 - alpha ** 2) * z2)
    d_base = np.exp(SIGMA_B * zb)
    d_true_raw = d_base * np.exp(r)
    d_true = d_true_raw * (d_base.sum(1, keepdims=True)
                           / d_true_raw.sum(1, keepdims=True))
    r_eff = np.log(d_true / d_base)          # effective log residual after rescaling
    f = np.exp(c)

    # Variant 1, geo_log: renorm = log-domain centering of c (the
    # theorem's strict domain), log-space MSE
    c_centered = c - c.mean(1, keepdims=True)
    mse_log_base = (r_eff ** 2).mean(1)
    mse_log_geo = ((r_eff - c_centered) ** 2).mean(1)

    # Variants 2/3: arithmetic mass-conservation renorm (as defined by
    # the real pipeline)
    raw = d_base * f
    d_corr = raw * (d_base.sum(1, keepdims=True) / raw.sum(1, keepdims=True))
    mse_log_arith = (np.log(d_true / d_corr) ** 2).mean(1)
    mse_dem_base = ((d_base - d_true) ** 2).mean(1)
    mse_dem_arith = ((d_corr - d_true) ** 2).mean(1)

    d_geo = mse_log_geo - mse_log_base
    d_arith = mse_log_arith - mse_log_base
    d_dem = mse_dem_arith - mse_dem_base
    return {
        'delta_geo_log': float(d_geo.mean()),
        'delta_arith_log': float(d_arith.mean()),
        'delta_arith_demand': float(d_dem.mean()),
        'frac_improved_geo_log': float((d_geo < 0).mean()),
        'frac_improved_arith_log': float((d_arith < 0).mean()),
        'frac_improved_arith_demand': float((d_dem < 0).mean()),
    }


def run_layer_a(rng) -> dict:
    """Full grid for layer A + boundaries for all three variants +
    quantitative assertions (geo_log and arith_log)."""
    shape = (len(SRATIO_GRID), len(ALPHA_GRID))
    grids = {k: np.zeros(shape) for k in (
        'delta_geo_log', 'delta_arith_log', 'delta_arith_demand',
        'frac_improved_geo_log', 'frac_improved_arith_log',
        'frac_improved_arith_demand')}

    for i, sr in enumerate(SRATIO_GRID):
        for j, a in enumerate(ALPHA_GRID):
            cell = simulate_cell_layer_a(float(a), float(sr), rng)
            for k, v in cell.items():
                grids[k][i, j] = v
        print(f'  Layer A sigma_ratio={sr:.1f} complete ({i + 1}/{len(SRATIO_GRID)})')

    theory = SRATIO_GRID / 2.0
    boundaries = {
        'geo_log': find_boundary(grids['delta_geo_log'], ALPHA_GRID),
        'arith_log': find_boundary(grids['delta_arith_log'], ALPHA_GRID),
        'arith_demand': find_boundary(grids['delta_arith_demand'], ALPHA_GRID),
    }
    devs = {k: b - theory for k, b in boundaries.items()}

    # -- Quantitative assertions (task specification: boundary deviation
    # from the theoretical line must be <= grid step x 2) --
    for variant in ('geo_log', 'arith_log'):
        b = boundaries[variant]
        if np.isnan(b).any():
            raise RuntimeError(f'Layer A {variant}: some sigma_ratio rows have no '
                               f'help region, boundary is unidentifiable -- assertion failed')
        max_dev = float(np.abs(devs[variant]).max())
        if max_dev > BOUNDARY_TOL:
            raise RuntimeError(f'Layer A {variant}: max boundary deviation {max_dev:.4f} > '
                               f'tolerance {BOUNDARY_TOL} (= grid step x 2)')
        print(f'  Layer A {variant}: max boundary deviation {max_dev:.4f} <= {BOUNDARY_TOL} OK')

    return {'grids': grids, 'boundaries': boundaries, 'theory': theory,
            'devs': devs}


# ════════════════════════════════════════════════════════════
# Synthetic layer B (Voronoi-style aggregation, qualitative)
# ════════════════════════════════════════════════════════════

def load_real_group_sizes() -> dict:
    """Compute the true Voronoi group sizes for the 16 regions (agent
    count / substation) directly from the frozen data.

    Follows exactly the same code path as the exp_r25 experiment:
    scu.load_grid_and_subs + compute_voronoi_assignment (a frozen
    EPSG:3857 fact). Returns {loc: sizes array}; sizes includes groups
    with zero agents (an empty substation is a genuine part of the real
    structure and is kept).
    """
    sizes_by_loc = {}
    for loc in scu.ALL_LOCATIONS:
        grid_gdf, region_sub, subs_sub, _ = scu.load_grid_and_subs(loc)
        assert (grid_gdf.index == np.arange(len(grid_gdf))).all(), \
            f'{loc}: grid_gdf is not a RangeIndex'
        assignment = scu.compute_voronoi_assignment(grid_gdf, subs_sub)
        sizes = np.bincount(np.asarray(assignment), minlength=len(subs_sub))
        sizes_by_loc[loc] = sizes.astype(np.int64)
        print(f'  Group sizes {loc}: {len(grid_gdf)} agents -> {len(subs_sub)} groups '
              f'(empty groups {int((sizes == 0).sum())})')
    return sizes_by_loc


def simulate_replicate_layer_b(alpha: float, sigma_c: float,
                               group_idx: np.ndarray, n_groups: int,
                               n_agents: int, rng) -> dict:
    """One layer B replicate: agent-level generation -> arithmetic
    renorm -> group aggregation -> substation-level evaluation.

    Returns help/hurt (demand-space RMSE) and the post-aggregation
    measured coordinates, using the same protocol as the exp_r25
    experiment.
    """
    z1 = rng.standard_normal(n_agents)
    z2 = rng.standard_normal(n_agents)
    zb = rng.standard_normal(n_agents)

    r = SIGMA_R * z1
    c = sigma_c * (alpha * z1 + np.sqrt(1.0 - alpha ** 2) * z2)
    d_base = np.exp(SIGMA_B * zb)
    d_true_raw = d_base * np.exp(r)
    d_true = d_true_raw * (d_base.sum() / d_true_raw.sum())

    raw = d_base * np.exp(c)
    d_corr = raw * (d_base.sum() / raw.sum())     # arithmetic mass-conservation renorm

    # Voronoi-style aggregation (real group sizes; agents are
    # exchangeable, so sequential chunking is equivalent to random grouping)
    D_true = np.bincount(group_idx, weights=d_true, minlength=n_groups)
    D_base = np.bincount(group_idx, weights=d_base, minlength=n_groups)
    D_corr = np.bincount(group_idx, weights=d_corr, minlength=n_groups)

    rmse_base = float(np.sqrt(np.mean((D_base - D_true) ** 2)))
    rmse_corr = float(np.sqrt(np.mean((D_corr - D_true) ** 2)))

    # Post-aggregation measured coordinates (same protocol as exp_r25:
    # epsilon = total regional demand * 1e-6)
    eps = d_true.sum() * EPS_RATIO
    log_rho = np.log((D_true + eps) / (D_base + eps))
    log_f = np.log((D_corr + eps) / (D_base + eps))
    s_rho, s_f = log_rho.std(), log_f.std()
    return {
        'delta_rmse': rmse_corr - rmse_base,
        'help': rmse_corr < rmse_base,
        'alpha_meas': _corr(log_f, log_rho),
        'sratio_meas': float(s_f / s_rho) if s_rho > 0 else float('nan'),
    }


def run_layer_b(rng, sizes_by_loc: dict) -> dict:
    """Full grid for layer B (qualitative): frac_improved, mean
    delta-RMSE, measured coordinates, plus scatter-point records."""
    locs = list(scu.ALL_LOCATIONS)
    # Pre-build group indices (agents are exchangeable, so np.repeat
    # sequential chunking is equivalent to random grouping)
    group_idx_of = {loc: np.repeat(np.arange(len(s)), s)
                    for loc, s in sizes_by_loc.items()}

    shape = (len(SRATIO_GRID), len(ALPHA_GRID))
    frac_improved = np.zeros(shape)
    mean_delta_rmse = np.zeros(shape)
    meas_alpha_mean = np.zeros(shape)
    meas_sratio_mean = np.zeros(shape)

    pts = {k: [] for k in ('gen_alpha', 'gen_sratio', 'alpha_meas',
                           'sratio_meas', 'help', 'region_idx')}
    n_nan_meas = 0

    for i, sr in enumerate(SRATIO_GRID):
        sigma_c = float(sr) * SIGMA_R
        for j, a in enumerate(ALPHA_GRID):
            helps, deltas, alphas_m, sratios_m = [], [], [], []
            for t in range(N_CYCLES_B * len(locs)):
                loc = locs[t % len(locs)]
                sizes = sizes_by_loc[loc]
                rep = simulate_replicate_layer_b(
                    float(a), sigma_c, group_idx_of[loc], len(sizes),
                    int(sizes.sum()), rng)
                helps.append(rep['help'])
                deltas.append(rep['delta_rmse'])
                alphas_m.append(rep['alpha_meas'])
                sratios_m.append(rep['sratio_meas'])
                pts['gen_alpha'].append(float(a))
                pts['gen_sratio'].append(float(sr))
                pts['alpha_meas'].append(rep['alpha_meas'])
                pts['sratio_meas'].append(rep['sratio_meas'])
                pts['help'].append(bool(rep['help']))
                pts['region_idx'].append(t % len(locs))
            frac_improved[i, j] = np.mean(helps)
            mean_delta_rmse[i, j] = np.mean(deltas)
            with np.errstate(all='ignore'):
                meas_alpha_mean[i, j] = np.nanmean(alphas_m)
                meas_sratio_mean[i, j] = np.nanmean(sratios_m)
            n_nan_meas += int(np.isnan(alphas_m).sum() + np.isnan(sratios_m).sum())
        print(f'  Layer B sigma_ratio={sr:.1f} complete ({i + 1}/{len(SRATIO_GRID)})')

    # Qualitative boundary: frac_improved crossing 0.5 (primary) and
    # mean delta-RMSE crossing 0 (secondary)
    boundary_frac = find_boundary(0.5 - frac_improved, ALPHA_GRID)
    boundary_mean = find_boundary(mean_delta_rmse, ALPHA_GRID)

    theory = SRATIO_GRID / 2.0
    valid = ~np.isnan(boundary_frac)
    diffs = np.diff(boundary_frac[valid])
    offsets = (boundary_frac - theory)[valid]

    qualitative = {
        'boundary_frac_improved': [None if np.isnan(v) else round(float(v), 6)
                                   for v in boundary_frac],
        'boundary_mean_delta_rmse': [None if np.isnan(v) else round(float(v), 6)
                                     for v in boundary_mean],
        'n_rows_with_boundary': int(valid.sum()),
        'n_rows_total': int(len(SRATIO_GRID)),
        # Monotonicity: allow a pullback of up to 1 grid step of sampling noise
        'monotonic_nondecreasing_within_one_step':
            bool((diffs >= -ALPHA_STEP).all()) if len(diffs) else True,
        'max_decrease': float(-diffs.min()) if len(diffs) and diffs.min() < 0 else 0.0,
        'offset_vs_theory_median': float(np.median(offsets)) if len(offsets) else None,
        'offset_vs_theory_max_abs': float(np.abs(offsets).max()) if len(offsets) else None,
        # Drift from generation coordinates to post-aggregation measured
        # coordinates (basis for coordinate consistency when overlaying the plots)
        'coord_drift_alpha_mean_abs': float(np.nanmean(
            np.abs(np.array(pts['alpha_meas']) - np.array(pts['gen_alpha'])))),
        'coord_drift_sratio_mean_abs': float(np.nanmean(
            np.abs(np.array(pts['sratio_meas']) - np.array(pts['gen_sratio'])))),
        'n_nan_measured_values': n_nan_meas,
        'note': ('Layer B only checks qualitatively that the boundary shape is '
                'preserved (monotonicity, positional offset recorded); '
                'quantitative assertions are made only for layer A.'),
    }
    return {
        'frac_improved': frac_improved,
        'mean_delta_rmse': mean_delta_rmse,
        'meas_alpha_mean': meas_alpha_mean,
        'meas_sratio_mean': meas_sratio_mean,
        'boundary_frac': boundary_frac,
        'boundary_mean': boundary_mean,
        'points': {k: np.array(v) for k, v in pts.items()},
        'qualitative': qualitative,
    }


# ════════════════════════════════════════════════════════════
# Empirical placement (240 combinations from exp_r25)
# ════════════════════════════════════════════════════════════

def build_empirical_points(s3_df: pd.DataFrame) -> pd.DataFrame:
    """Convert the 240 rows from exp_r25 into an empirical-placement
    table: coordinates + theoretical prediction + actual sign."""
    cols = ['base_type', 'base_id', 'seed', 'fold', 'location', 'signal',
            'pearson_log', 'spearman_log', 'sigma_r', 'sigma_c', 'sigma_ratio',
            'rmse_base', 'rmse_corrected', 'delta_rmse']
    df = s3_df[cols].copy()
    df['theory_threshold'] = df['sigma_ratio'] / 2.0
    df['predicted_help'] = df['pearson_log'] > df['theory_threshold']
    df['actual_help'] = df['delta_rmse'] < 0
    df['prediction_correct'] = df['predicted_help'] == df['actual_help']
    return df


def confusion_matrix_of(df: pd.DataFrame) -> dict:
    """2x2 confusion matrix (theoretical prediction x actual delta-RMSE sign)."""
    p, a = df['predicted_help'].values, df['actual_help'].values
    cm = {
        'pred_help_actual_help': int((p & a).sum()),
        'pred_help_actual_hurt': int((p & ~a).sum()),
        'pred_hurt_actual_help': int((~p & a).sum()),
        'pred_hurt_actual_hurt': int((~p & ~a).sum()),
    }
    cm['total'] = int(sum(cm.values()))
    cm['accuracy'] = float((p == a).mean())
    return cm


def check_propositions(df: pd.DataFrame) -> dict:
    """Verdicts for the three propositions -- generated entirely from
    numeric rules, never hand-written."""
    # P1: static arms fall in the help region
    static = df[df['base_type'].isin(['uniform', 'gpm'])]
    p1_pred = float(static['predicted_help'].mean())
    p1_act = float(static['actual_help'].mean())
    p1_value = min(p1_pred, p1_act)

    # P2: the combined GNN arm (NP) falls in the hurt region
    gnn_np = df[(df['base_type'] == 'gnn') & (df['signal'] == 'NP')]
    p2_pred = float((~gnn_np['predicted_help']).mean())
    p2_act = float((~gnn_np['actual_help']).mean())
    p2_value = min(p2_pred, p2_act)

    # P3: dual-signal variance superadditivity: check
    # sigma_c(NP)^2 > sigma_c(N)^2 + sigma_c(P)^2 for each (base_id, location)
    piv = df.pivot_table(index=['base_type', 'base_id', 'location'],
                         columns='signal', values='sigma_c', aggfunc='first')
    assert not piv.isna().any().any(), 'P3: some combinations are missing one of the N/P/NP signals'
    var_np = piv['NP'] ** 2
    var_sum = piv['N'] ** 2 + piv['P'] ** 2
    superadd = var_np > var_sum
    ratio = var_np / var_sum
    p3_value = float(superadd.mean())
    p3_by_bt = {
        bt: {
            'n': int((piv.index.get_level_values('base_type') == bt).sum()),
            'frac_superadditive': float(
                superadd[piv.index.get_level_values('base_type') == bt].mean()),
            'median_var_ratio': float(
                ratio[piv.index.get_level_values('base_type') == bt].median()),
        } for bt in ('uniform', 'gpm', 'gnn')
    }

    return {
        'P1_static_in_help_region': {
            'description': 'Static arms (uniform+gpm) fall in the theoretical help region and actually improve',
            'n_rows': int(len(static)),
            'frac_predicted_help': p1_pred,
            'frac_actual_help': p1_act,
            'value': p1_value,
            'value_rule': 'value = min(frac_predicted_help, frac_actual_help)',
            'verdict': verdict_of(p1_value),
        },
        'P2_gnn_np_in_hurt_region': {
            'description': 'The combined GNN arm (gnn x NP) falls in the theoretical hurt region and actually worsens',
            'n_rows': int(len(gnn_np)),
            'frac_predicted_hurt': p2_pred,
            'frac_actual_hurt': p2_act,
            'value': p2_value,
            'value_rule': 'value = min(frac_predicted_hurt, frac_actual_hurt)',
            'verdict': verdict_of(p2_value),
        },
        'P3_dual_signal_var_superadditive': {
            'description': ('Dual-signal variance superadditivity: '
                           'sigma_c(NP)^2 > sigma_c(N)^2 + sigma_c(P)^2 '
                           'for each (base_id, location) combination'),
            'n_combos': int(len(piv)),
            'frac_superadditive': p3_value,
            'median_var_ratio': float(ratio.median()),
            'by_base_type': p3_by_bt,
            'value': p3_value,
            'value_rule': 'value = frac_superadditive',
            'verdict': verdict_of(p3_value),
        },
    }


# ════════════════════════════════════════════════════════════
# Main pipeline
# ════════════════════════════════════════════════════════════

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)

    # -- Input: exp_r25 artifacts (read-only) --
    # float_precision='round_trip': the default 'high' parser does not
    # guarantee round-trip within 1 ulp, which would break a bit-for-bit
    # consistency test confirming this script only transcribes (rather
    # than recomputes) the input values
    s3_csv = S3_DIR / 'alignment_per_region.csv'
    s3_df = pd.read_csv(s3_csv, dtype={'seed': str, 'fold': str},
                        float_precision='round_trip')
    assert len(s3_df) == 240, f'exp_r25 input row count {len(s3_df)} != 240'

    # -- Layer A (strict theoretical domain, quantitative assertions) --
    print('═══ Synthetic layer A (agent-level, no aggregation, strict theoretical domain) ═══')
    layer_a = run_layer_a(rng)

    # -- Layer B (real group sizes + Voronoi-style aggregation, qualitative) --
    print('═══ Synthetic layer B (Voronoi-style aggregation, qualitative) ═══')
    print('Loading the true group sizes for the 16 regions (computed directly from frozen data)...')
    sizes_by_loc = load_real_group_sizes()
    layer_b = run_layer_b(rng, sizes_by_loc)

    # -- Empirical placement --
    print('═══ Empirical placement (exp_r25, 240 combinations) ═══')
    emp = build_empirical_points(s3_df)
    cm_all = confusion_matrix_of(emp)
    cm_by_bt = {bt: confusion_matrix_of(emp[emp['base_type'] == bt])
                for bt in ('uniform', 'gpm', 'gnn')}
    props = check_propositions(emp)

    # -- Write npz to disk --
    locs = list(scu.ALL_LOCATIONS)
    sizes_concat = np.concatenate([sizes_by_loc[loc] for loc in locs])
    sizes_ptr = np.cumsum([0] + [len(sizes_by_loc[loc]) for loc in locs])
    np.savez_compressed(
        OUTPUT_DIR / 'phase_diagram_data.npz',
        alpha_grid=ALPHA_GRID,
        sratio_grid=SRATIO_GRID,
        sigma_r=np.float64(SIGMA_R),
        sigma_b=np.float64(SIGMA_B),
        theory_boundary=layer_a['theory'],
        # Layer A
        layerA_delta_geo_log=layer_a['grids']['delta_geo_log'],
        layerA_delta_arith_log=layer_a['grids']['delta_arith_log'],
        layerA_delta_arith_demand=layer_a['grids']['delta_arith_demand'],
        layerA_frac_improved_geo_log=layer_a['grids']['frac_improved_geo_log'],
        layerA_frac_improved_arith_log=layer_a['grids']['frac_improved_arith_log'],
        layerA_frac_improved_arith_demand=layer_a['grids']['frac_improved_arith_demand'],
        layerA_boundary_geo_log=layer_a['boundaries']['geo_log'],
        layerA_boundary_arith_log=layer_a['boundaries']['arith_log'],
        layerA_boundary_arith_demand=layer_a['boundaries']['arith_demand'],
        # Layer B
        layerB_frac_improved=layer_b['frac_improved'],
        layerB_mean_delta_rmse=layer_b['mean_delta_rmse'],
        layerB_meas_alpha_mean=layer_b['meas_alpha_mean'],
        layerB_meas_sratio_mean=layer_b['meas_sratio_mean'],
        layerB_boundary_frac=layer_b['boundary_frac'],
        layerB_boundary_mean=layer_b['boundary_mean'],
        layerB_points_gen_alpha=layer_b['points']['gen_alpha'],
        layerB_points_gen_sratio=layer_b['points']['gen_sratio'],
        layerB_points_alpha_meas=layer_b['points']['alpha_meas'],
        layerB_points_sratio_meas=layer_b['points']['sratio_meas'],
        layerB_points_help=layer_b['points']['help'],
        layerB_points_region_idx=layer_b['points']['region_idx'],
        layerB_group_sizes_concat=sizes_concat,
        layerB_group_sizes_ptr=sizes_ptr,
        layerB_region_names=np.array(locs),
    )

    # -- Write empirical_points.csv to disk --
    emp.to_csv(OUTPUT_DIR / 'empirical_points.csv', index=False)

    # -- Write proposition_checks.json to disk --
    layer_a_report = {
        'design': {
            'sigma_r': SIGMA_R, 'sigma_b': SIGMA_B,
            'n_agents': N_AGENTS_A, 'n_reps_per_cell': N_REPS_A,
            'alpha_grid': {'min': float(ALPHA_GRID[0]),
                           'max': float(ALPHA_GRID[-1]), 'step': ALPHA_STEP},
            'sratio_grid': {'min': float(SRATIO_GRID[0]),
                            'max': float(SRATIO_GRID[-1]), 'step': SRATIO_STEP},
            'mass_consistency': 'd_true is rescaled so that sum(d_true) = sum(d_base) (a structural precondition of the pipeline)',
        },
        'variants': {},
    }
    variant_notes = {
        'geo_log': 'renorm = log-domain centering of c (the theorem strict domain), log-space MSE -- quantitative assertion',
        'arith_log': ('renorm = arithmetic mass conservation (the real pipeline), '
                     'log-space MSE -- quantitative assertion; the systematic offset '
                     'comes from the Jensen gap between the renorm constant '
                     'log(sum(w*e^c)) and geometric centering'),
        'arith_demand': ('arithmetic renorm + demand-space MSE (the real evaluation '
                        'convention) -- recorded only, not asserted; the boundary '
                        'shape is preserved but shifts upward at high sigma_ratio; '
                        'the aggregated version is covered qualitatively by layer B'),
    }
    for variant, b in layer_a['boundaries'].items():
        dev = layer_a['devs'][variant]
        finite = ~np.isnan(b)
        entry = {
            'note': variant_notes[variant],
            'n_rows_with_boundary': int(finite.sum()),
            'boundary': [None if np.isnan(v) else round(float(v), 6) for v in b],
            'max_abs_dev_vs_theory': float(np.abs(dev[finite]).max()),
            'tolerance': BOUNDARY_TOL,
            'tolerance_rule': 'grid step x 2 (task specification)',
        }
        if variant in ('geo_log', 'arith_log'):
            entry['asserted'] = True
            entry['pass'] = bool(finite.all()
                                 and np.abs(dev[finite]).max() <= BOUNDARY_TOL)
        else:
            entry['asserted'] = False
        layer_a_report['variants'][variant] = entry

    checks = {
        'meta': {
            'script': '025_exp_r26_condition_phase_diagram.py',
            'purpose': 'Condition proposition phase diagram (synthetic layers A/B + empirical placement)',
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'rng_seed': RNG_SEED,
            'inputs': [str(s3_csv.relative_to(SCRIPT_DIR))],
            'theory': ('help iff rho(r,c) > sigma_c/(2*sigma_r); r = log residual '
                      'ratio, c = centered log correction factor (derived in log space)'),
            'theory_line': 'alpha* = sigma_ratio / 2',
            'verdict_rule': VERDICT_RULE_TEXT,
            'notes': [
                'Verdict fields (verdict/pass) are generated entirely from numeric rules, never hand-written.',
                ('Layer A coordinates = generation parameters; empirical coordinates '
                '= exp_r25 substation-level measurements (pearson_log, sigma_ratio). '
                'The two are identical at the agent level; aggregation-induced '
                'coordinate drift is recorded in layer_b.qualitative.coord_drift_*.'),
            ],
        },
        'layer_a': layer_a_report,
        'layer_b': {
            'design': {
                'group_sizes': ('true Voronoi group sizes for the 16 regions '
                               '(bincount of the assignment computed directly '
                               'from frozen data)'),
                'n_regions': len(locs),
                'n_reps_per_cell': N_CYCLES_B * len(locs),
                'total_agents_per_cycle': int(sizes_concat.sum()),
                'total_groups': int(len(sizes_concat)),
                'renorm': 'arithmetic mass conservation (single group per region, a per-ITL3 simplification of the real pipeline)',
                'metric': 'substation-level demand-space RMSE (the real evaluation convention)',
                'eps_rule': f'epsilon = total regional demand * {EPS_RATIO} (same protocol as exp_r25)',
            },
            'qualitative': layer_b['qualitative'],
        },
        'empirical': {
            'n_points': int(len(emp)),
            'prediction_rule': 'predicted_help iff pearson_log > sigma_ratio / 2',
            'actual_rule': 'actual_help iff delta_rmse < 0',
            'confusion_matrix': cm_all,
            'confusion_by_base_type': cm_by_bt,
            'propositions': props,
        },
    }
    with open(OUTPUT_DIR / 'proposition_checks.json', 'w', encoding='utf-8') as f:
        json.dump(checks, f, ensure_ascii=False, indent=2)

    # -- Summary --
    print('\n' + '=' * 70)
    for variant in ('geo_log', 'arith_log', 'arith_demand'):
        e = layer_a_report['variants'][variant]
        tail = ('assertion ' + ('passed' if e.get('pass') else 'failed')
                if e['asserted'] else 'recorded (not asserted)')
        print(f"Layer A {variant}: max boundary deviation {e['max_abs_dev_vs_theory']:.4f} "
              f"(tolerance {BOUNDARY_TOL}) -- {tail}")
    q = layer_b['qualitative']
    print(f"Layer B: identifiable boundary rows {q['n_rows_with_boundary']}/{q['n_rows_total']}, "
          f"monotonic (1-step tolerance)={q['monotonic_nondecreasing_within_one_step']}, "
          f"median offset from theory {q['offset_vs_theory_median']:+.4f}")
    print(f"Confusion matrix: help/help={cm_all['pred_help_actual_help']}, "
          f"help/hurt={cm_all['pred_help_actual_hurt']}, "
          f"hurt/help={cm_all['pred_hurt_actual_help']}, "
          f"hurt/hurt={cm_all['pred_hurt_actual_hurt']} "
          f"(total {cm_all['total']}, accuracy {cm_all['accuracy']:.3f})")
    for name, p in props.items():
        print(f"{name}: value={p['value']:.4f} -> {p['verdict']}")
    print(f'Artifacts: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
