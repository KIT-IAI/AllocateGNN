# -*- coding: utf-8 -*-
"""Australia case study - phase-diagram placement check (014).

Background: the prior statistical-evaluation step (013) found that multiplicative
correction **significantly improves** the AU GNN base allocation (GNNpostNP vs GNN
delta=-12.23, Holm p=0.006) -- the opposite direction from the UK case's antagonism
(delta=+1.93, i.e. a degradation). This step checks whether that "non-replication"
is actually predicted by a conditional theory: the theory line help <=> rho >
sigma_c/(2*sigma_r) (derived in log-space in the UK case's 025 script). If the AU
GNN x multiplicative combinations systematically land in the "help" region while
the UK ones land in the "hurt" region, the two case studies are simply **two sides
of the same theoretical line**, rather than the theory failing to hold.

=== Formulas (matching UK's 024_exp_r25_residual_alignment.py exactly) ===
- Per substation:
    rho_j = (d_true_j + eps) / (d_base_j + eps)   -- residual ratio
    F_j = (d_corr_j + eps) / (d_base_j + eps)     -- effective correction factor
  eps = total regional demand x 1e-6 (log divide-by-zero protection; the count of
  protected samples is recorded in the output).
- Alignment rho = Pearson corr(log F, log rho_resid); sigma_r = std(log rho_resid),
  sigma_c = std(log F) (numpy population standard deviation, ddof=0).
- Delta RMSE = RMSE(corrected) - RMSE(base), where RMSE is computed on the
  **raw** (non-eps-protected) allocation values via scu.evaluate_allocation
  (actual_col='peak_mw', the demand convention used throughout this case study).
- Theoretical prediction predicted_help <=> rho > sigma_c/(2*sigma_r); actual
  actual_help <=> delta RMSE < 0 (same rule as UK's build_empirical_points).

=== Combination universe (AU parameter substitution of the UK script) ===
Base in {Uniform, GPM, GNN x 3 seeds} (for GNN, each (seed, region) uses the fold
in which that region served as the test set -- consistent with the test-fold
convention used in 013) x 12 SA4 regions x signal in {N, P, NP}
(multiplicative apply_standard_multiplicative) -> 12 x 3 x (2 + 3) = 180 rows.

=== Built-in anchoring (guards against base/correction pipeline drift) ===
- The static bases and all 6 multiplicative arms' RMSE are anchored against the
  frozen au_static_metrics.csv (stored at full precision, deterministic pipeline
  in 011) with tolerance 1e-9. Note: the UK 024 script leaves the P-signal arm
  unanchored, whereas AU's UniP / GPMpostP have frozen rows available -- so this
  step anchors the P arm as well (an upgrade in coverage, not a deviation);
- The GNN base (voronoi_GNN) and multiplicative arms (voronoi_GNNpost{sig}) RMSE
  are anchored against the frozen kfold_test_rmse.csv (stored at 4 decimal
  places) with tolerance = half a ulp, 5e-5, plus 1e-9 (matching the 4.99e-5
  maximum deviation measured for the same reaggregation pipeline in 013).
Any anchor failure raises an error immediately (if the base is wrong, everything
downstream is invalid).

=== Merging with UK data points (draft material for a three-case-study phase
diagram figure) ===
Reads UK's exp_r26/empirical_points.csv (240 points, transcribed only, never
recomputed) and merges it with the AU 180 points under a common schema, tagging
each row with its country -> uk_au_phase_points_merged.csv (a third country can
be appended later once the Germany training run is complete).

=== Deviation log relative to the UK conventions (written to summary.deviations) ===
1. Demand convention = peak_mw (MW, active power) vs the UK case's
   'Demand (MVA)' (an established, case-internal-consistent choice);
2. proximity/factor inputs are inherited from 011's context object (B2
   precomputed npz, EPSG:7856) -- the compute_factors formula itself is
   unchanged;
3. The GNN base = 012's grid_demands['GNN'] (the UK script's equivalent is
   exp0's 'gnn_demand' -- same semantics: GNN demand on the test fold under the
   baseline configuration);
4. The UK script leaves the P arm unanchored (not_anchored); AU's P arm has a
   frozen row available, so it is anchored here.

All conclusion fields are generated programmatically from numeric rules rather
than hand-written (the verdict thresholds match the UK 025 script: value >= 0.9
-> supported; >= 0.5 -> partially_supported; otherwise not_supported). This step
has no randomness (deterministic post-processing); runs entirely on CPU (capped
at 4 threads). All frozen upstream artifacts are read-only.

Outputs:
    data/processed/evaluation/au_phase_points.csv           -- 180 data points
        (schema column-aligned with UK's exp_r26/empirical_points.csv)
    data/processed/evaluation/uk_au_phase_points_merged.csv -- 420 merged points
        (country column + UK schema; draft material for the three-case-study
        phase diagram figure)
    data/processed/evaluation/phase_placement_summary.json  -- confusion matrix +
        core test (au_gnn_mult_in_help_region) + UK contrast + anchor report

Usage:
    python 014_phase_placement_check.py    (allocategnn env)
"""

from __future__ import annotations

import os

# Thread caps must be set before numpy/scipy are first imported (the GPU is busy
# with the Germany training run, so this step is CPU-only, capped at 4 threads)
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import importlib.util
import json
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Force UTF-8 output on the Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# sys.path setup (repo root + UK experiment directory, so shared modules can be
# imported across case studies)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
_p = SCRIPT_DIR
while not (_p / "SpatialAllocation").exists():
    if _p.parent == _p:
        raise RuntimeError("Could not locate the repository root (SpatialAllocation package)")
    _p = _p.parent
PROJECT_ROOT = _p
UK_EXP_DIR = PROJECT_ROOT / "StudyCase" / "British_weighter_experiments"
for _extra in (str(UK_EXP_DIR), str(PROJECT_ROOT)):
    if _extra not in sys.path:
        sys.path.insert(0, _extra)

from scipy.stats import pearsonr, spearmanr                             # noqa: E402

import shared_correction_utils as scu                                   # noqa: E402  # shared post-hoc correction utilities (read-only import from the UK case study)


def _load_script_module(name: str, path: Path):
    """Load a numerically-prefixed script module via importlib, reusing its functions (011/013) instead of duplicating the implementation."""
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# 011 provides the static-base functions used here (compute_uniform_base/compute_gpm_base/loaders)
au011 = _load_script_module("au011", SCRIPT_DIR / "011_static_baselines.py")

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
PROCESSED = SCRIPT_DIR / "data" / "processed"
TRAIN_ROOT = PROCESSED / "training"
STATIC_CSV = PROCESSED / "static" / "au_static_metrics.csv"
STEP_TABLE = PROCESSED / "grid" / "grid_step_size_table.csv"
B2_REGISTRY = SCRIPT_DIR / "docs" / "b1b2" / "b2_features.json"
EVAL_DIR = PROCESSED / "evaluation"

UK_EMPIRICAL_CSV = UK_EXP_DIR / "results" / "exp_r26" / "empirical_points.csv"

POINTS_CSV = EVAL_DIR / "au_phase_points.csv"
MERGED_CSV = EVAL_DIR / "uk_au_phase_points_merged.csv"
SUMMARY_JSON = EVAL_DIR / "phase_placement_summary.json"

SEEDS = [42, 123, 456]
SIGNALS = ["N", "P", "NP"]
CONFIG = "baseline"                 # GNN base = the "baseline" training configuration (matching the UK case's convention)
EPS_RATIO = 1e-6                    # eps = total regional demand x EPS_RATIO (matching the UK case's protocol)
DEMAND_COL = "peak_mw"              # demand convention used throughout this case study (deviation log item 1)
N_SA4 = 12

# Anchor tolerances: static arms are stored at full precision (013 measured a
# recomputation deviation of 7.1e-15) -> 1e-9; GNN arm kfold csv is stored at 4
# decimal places -> half a ulp, 5e-5, plus floating-point slack (013 measured 4.99e-5)
STATIC_ANCHOR_TOL = 1e-9
GNN_ANCHOR_TOL = 5e-5 + 1e-9

# Static-arm anchor mapping: (base_type, signal) -> arm label in au_static_metrics.csv
STATIC_ARM_MAP = {
    ("uniform", None): "Uni",
    ("uniform", "N"): "UniN", ("uniform", "P"): "UniP", ("uniform", "NP"): "UniNP",
    ("gpm", None): "GPM",
    ("gpm", "N"): "GPMpostN", ("gpm", "P"): "GPMpostP", ("gpm", "NP"): "GPMpostNP",
}

# The 18-column schema of UK's empirical_points.csv (column-aligned; do not
# add, remove, or reorder columns)
UK_SCHEMA = [
    "base_type", "base_id", "seed", "fold", "location", "signal",
    "pearson_log", "spearman_log", "sigma_r", "sigma_c", "sigma_ratio",
    "rmse_base", "rmse_corrected", "delta_rmse",
    "theory_threshold", "predicted_help", "actual_help", "prediction_correct",
]

# Verdict thresholds (matching the UK 025 script; generated programmatically, not hand-written)
VERDICT_TIERS = [(0.9, "supported"), (0.5, "partially_supported")]
VERDICT_RULE_TEXT = ("value >= 0.9 -> supported; value >= 0.5 -> "
                     "partially_supported; else not_supported")


def verdict_of(value: float) -> str:
    """Numeric rule -> conclusion field (generated programmatically; same function as in the UK 025 script)."""
    for threshold, label in VERDICT_TIERS:
        if value >= threshold:
            return label
    return "not_supported"


# ---------------------------------------------------------------------------
# Alignment statistics (ported verbatim from the UK case's 024 alignment_stats
# implementation so the formulas match exactly)
# ---------------------------------------------------------------------------
def alignment_stats(d_true: np.ndarray, d_base: np.ndarray,
                    d_corr: np.ndarray, eps: float) -> dict:
    """Compute per-substation alignment statistics for one (base, region, signal) combination (matching the UK 024 script).

    The ratio-based quantities are computed in log space; the difference-based
    quantities are computed in the original value space. eps is only used for
    the ratio/log protection -- the difference d_true - d_base uses raw values.
    """
    d_true = np.asarray(d_true, dtype=float)
    d_base = np.asarray(d_base, dtype=float)
    d_corr = np.asarray(d_corr, dtype=float)
    n = len(d_true)
    assert n >= 3, f"Substation count {n} < 3, cannot compute correlation"

    # Count of eps-protected samples (values <= 0 before protection was applied)
    n_prot_true = int((d_true <= 0).sum())
    n_prot_base = int((d_base <= 0).sum())
    n_prot_corr = int((d_corr <= 0).sum())
    n_prot_any = int(((d_true <= 0) | (d_base <= 0) | (d_corr <= 0)).sum())

    # Ratio-based quantities (log space)
    log_rho = np.log((d_true + eps) / (d_base + eps))
    log_f = np.log((d_corr + eps) / (d_base + eps))

    if np.std(log_f) == 0 or np.std(log_rho) == 0:
        raise RuntimeError("log F or log rho has zero variance -- correlation is undefined, check this combination")

    pearson_log = float(pearsonr(log_f, log_rho)[0])
    spearman_log = float(spearmanr(log_f, log_rho)[0])

    # sigma_r / sigma_c (phase-diagram coordinates; ddof=0)
    sigma_r = float(np.std(log_rho))
    sigma_c = float(np.std(log_f))

    return {
        "n_substations": n,
        "eps": float(eps),
        "n_eps_protected_true": n_prot_true,
        "n_eps_protected_base": n_prot_base,
        "n_eps_protected_corrected": n_prot_corr,
        "n_eps_protected_any": n_prot_any,
        "pearson_log": pearson_log,
        "spearman_log": spearman_log,
        "sigma_r": sigma_r,
        "sigma_c": sigma_c,
        "sigma_ratio": sigma_c / sigma_r,
    }


# ---------------------------------------------------------------------------
# Single-base analysis (AU parameter substitution of UK's analyse_base)
# ---------------------------------------------------------------------------
def analyse_base(ctx: dict, d_true: np.ndarray, eps: float,
                 base_demand: np.ndarray, base_type: str, base_id: str,
                 seed, fold) -> list:
    """Compute alignment statistics and delta RMSE for one base across the three signals, returning a list of records."""
    grid_gdf = ctx["grid_gdf"]
    region_sub = ctx["region_sub"]
    subs_sub = ctx["subs_sub"]
    assignment = ctx["assignment"]

    subs_base = scu.aggregate_by_assignment(subs_sub, assignment, base_demand)
    d_base = subs_base["allocated_demand"].values.astype(float)
    m_base = scu.evaluate_allocation(subs_base, actual_col=DEMAND_COL)

    rows = []
    for signal in SIGNALS:
        corrected = scu.apply_standard_multiplicative(
            base_demand, ctx["factors"][signal], grid_gdf, region_sub)
        subs_corr = scu.aggregate_by_assignment(subs_sub, assignment, corrected)
        d_corr = subs_corr["allocated_demand"].values.astype(float)
        m_corr = scu.evaluate_allocation(subs_corr, actual_col=DEMAND_COL)

        stats = alignment_stats(d_true, d_base, d_corr, eps)
        rows.append({
            "base_type": base_type,
            "base_id": base_id,
            "seed": seed,
            "fold": fold,
            "location": ctx["loc"],
            "signal": signal,
            **stats,
            "rmse_base": float(m_base["rmse"]),
            "rmse_corrected": float(m_corr["rmse"]),
            "delta_rmse": float(m_corr["rmse"] - m_base["rmse"]),
        })
    return rows


# ---------------------------------------------------------------------------
# Frozen-artifact anchoring (guards against base-reconstruction drift)
# ---------------------------------------------------------------------------
def _check_anchor(dev: float, tol: float, what: str, anchor_log: list) -> None:
    """Record one anchor check and raise if the deviation exceeds tolerance."""
    anchor_log.append({"what": what, "abs_dev": dev, "tol": tol,
                       "status": "ok" if dev <= tol else "FAIL"})
    if dev > tol:
        raise RuntimeError(f"Anchor check failed: {what} deviation {dev:.3e} > tolerance {tol:.3e}")


_STATIC_DF: pd.DataFrame | None = None


def _static_frozen_value(arm: str, loc: str) -> float:
    """Read the frozen RMSE from au_static_metrics.csv (main table, station_set='all')."""
    global _STATIC_DF
    if _STATIC_DF is None:
        _STATIC_DF = pd.read_csv(STATIC_CSV, encoding="utf-8-sig")
    row = _STATIC_DF[(_STATIC_DF["arm"] == arm)
                     & (_STATIC_DF["metric"] == "rmse")
                     & (_STATIC_DF["station_set"] == "all")]
    assert len(row) == 1, f"Static arm {arm}/rmse/all row count {len(row)} != 1"
    return float(row[loc].iloc[0])


def anchor_static(loc: str, rows: list, anchor_log: list) -> None:
    """Anchor the static bases and all 6 multiplicative arms against the frozen au_static_metrics.csv (tolerance 1e-9).

    The UK 024 script leaves the P arm unanchored (not_anchored); AU's
    UniP/GPMpostP have frozen rows available, so the P arm is anchored here too
    (an upgrade in coverage, recorded as deviation item 4).
    """
    by = {(r["base_type"], r["signal"]): r for r in rows
          if r["base_type"] in ("uniform", "gpm")}
    for bt in ("uniform", "gpm"):
        base_arm = STATIC_ARM_MAP[(bt, None)]
        _check_anchor(
            abs(by[(bt, "N")]["rmse_base"] - _static_frozen_value(base_arm, loc)),
            STATIC_ANCHOR_TOL, f"{loc}/{bt} base vs au_static_metrics {base_arm}",
            anchor_log)
        for sig in SIGNALS:
            arm = STATIC_ARM_MAP[(bt, sig)]
            _check_anchor(
                abs(by[(bt, sig)]["rmse_corrected"] - _static_frozen_value(arm, loc)),
                STATIC_ANCHOR_TOL, f"{loc}/{bt}+{sig} vs au_static_metrics {arm}",
                anchor_log)


_KFOLD_CACHE: dict = {}


def _gnn_frozen_value(arm: str, seed: int, loc: str) -> float:
    """Read the frozen voronoi_{arm} row from kfold_test_rmse.csv (stored at 4 decimal places)."""
    if seed not in _KFOLD_CACHE:
        path = TRAIN_ROOT / f"seed_{seed}" / CONFIG / "kfold_test_rmse.csv"
        _KFOLD_CACHE[seed] = pd.read_csv(path, index_col=0)
    return float(_KFOLD_CACHE[seed].loc[f"voronoi_{arm}", loc])


def anchor_gnn(loc: str, seed: int, rows: list, anchor_log: list) -> None:
    """Anchor the GNN base and 3 multiplicative arms' RMSE against the frozen kfold_test_rmse.csv.

    Tolerance = half a ulp at 4 decimal places (5e-5) plus floating-point slack
    (013's reaggregation measured a maximum deviation of 4.99e-5 for the same
    pipeline); the purpose of this anchor is to catch seed/fold/region pickle
    mismatches (which produce deviations on the order of >= 1e-1).
    """
    by = {r["signal"]: r for r in rows}
    _check_anchor(
        abs(by["N"]["rmse_base"] - _gnn_frozen_value("GNN", seed, loc)),
        GNN_ANCHOR_TOL, f"{loc}/gnn seed{seed} base vs kfold voronoi_GNN",
        anchor_log)
    for sig in SIGNALS:
        _check_anchor(
            abs(by[sig]["rmse_corrected"]
                - _gnn_frozen_value(f"GNNpost{sig}", seed, loc)),
            GNN_ANCHOR_TOL,
            f"{loc}/gnn seed{seed}+{sig} vs kfold voronoi_GNNpost{sig}",
            anchor_log)


# ---------------------------------------------------------------------------
# Phase-point table + confusion matrix (matching UK's build_empirical_points /
# confusion_matrix_of)
# ---------------------------------------------------------------------------
def build_phase_points(df: pd.DataFrame) -> pd.DataFrame:
    """Detail rows -> phase-diagram point table: coordinates + theoretical prediction + actual sign (matching the UK 025 script's rule)."""
    out = df.copy()
    out["theory_threshold"] = out["sigma_ratio"] / 2.0
    out["predicted_help"] = out["pearson_log"] > out["theory_threshold"]
    out["actual_help"] = out["delta_rmse"] < 0
    out["prediction_correct"] = out["predicted_help"] == out["actual_help"]
    return out[UK_SCHEMA + ["n_substations", "n_eps_protected_any"]]


def confusion_matrix_of(df: pd.DataFrame) -> dict:
    """2x2 confusion matrix (theoretical prediction x actual delta-RMSE sign; same function as in the UK 025 script)."""
    p = df["predicted_help"].values.astype(bool)
    a = df["actual_help"].values.astype(bool)
    cm = {
        "pred_help_actual_help": int((p & a).sum()),
        "pred_help_actual_hurt": int((p & ~a).sum()),
        "pred_hurt_actual_help": int((~p & a).sum()),
        "pred_hurt_actual_hurt": int((~p & ~a).sum()),
    }
    cm["total"] = int(sum(cm.values()))
    cm["accuracy"] = float((p == a).mean())
    return cm


def region_cell(df: pd.DataFrame) -> dict:
    """One summary cell: placement fractions + coordinate medians (generated programmatically)."""
    return {
        "n_rows": int(len(df)),
        "frac_predicted_help": float(df["predicted_help"].mean()),
        "frac_actual_help": float(df["actual_help"].mean()),
        "frac_prediction_correct": float(df["prediction_correct"].mean()),
        "median_pearson_log": float(df["pearson_log"].median()),
        "median_sigma_r": float(df["sigma_r"].median()),
        "median_sigma_c": float(df["sigma_c"].median()),
        "median_sigma_ratio": float(df["sigma_ratio"].median()),
        "median_delta_rmse": float(df["delta_rmse"].median()),
        "mean_delta_rmse": float(df["delta_rmse"].mean()),
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    t0 = time.time()
    print("=" * 70)
    print("AU phase-diagram placement check (014) -- 180 combinations x (rho, sigma_c/sigma_r) + UK merged view")
    print("=" * 70)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # ── 0. Region universe and fold splits ──
    step_table = pd.read_csv(STEP_TABLE, encoding="utf-8-sig")
    all_locations = step_table["loc_key"].tolist()
    assert len(all_locations) == N_SA4, f"SA4 count {len(all_locations)} != {N_SA4}"

    # Fold split per seed: loc -> fold directory name (each region is the test
    # set exactly once per seed)
    fold_of: dict = {}
    for seed in SEEDS:
        splits_path = TRAIN_ROOT / f"seed_{seed}" / CONFIG / "kfold_splits.json"
        splits = json.loads(splits_path.read_text(encoding="utf-8"))
        mapping = {}
        for fold_key, info in splits.items():
            for loc in info["test"]:
                assert loc not in mapping, f"seed {seed}: {loc} appears in multiple test folds"
                mapping[loc] = fold_key.replace("_", "")    # fold_1 -> fold1
        assert set(mapping) == set(all_locations), \
            f"seed {seed}: test folds do not cover all {N_SA4} regions"
        fold_of[seed] = mapping

    # ── 1. Compute the 180 combinations, region by region ──
    b2 = json.loads(B2_REGISTRY.read_text(encoding="utf-8"))
    regions = au011.load_regions()
    stations = au011.load_stations(regions)

    assignment_cache: dict = {}
    records: list = []
    anchor_log: list = []

    for loc in all_locations:
        print(f"\n=== {loc} ===")
        ctx = au011.load_location_context(loc, regions, stations, b2,
                                          assignment_cache)
        grid_gdf, region_sub = ctx["grid_gdf"], ctx["region_sub"]
        d_true = ctx["subs_sub"][DEMAND_COL].values.astype(float)
        region_total = float(np.asarray(region_sub["Demand (MVA)"],
                                        dtype=float).sum())
        eps = region_total * EPS_RATIO

        # ── Static bases (reusing the same functions defined in 011) ──
        uni_base = au011.compute_uniform_base(grid_gdf, region_sub)
        gpm_base, _ = au011.compute_gpm_base(grid_gdf, region_sub, ctx["subs_sub"])

        loc_rows: list = []
        loc_rows += analyse_base(ctx, d_true, eps, uni_base,
                                 "uniform", "uniform", "-", "-")
        loc_rows += analyse_base(ctx, d_true, eps, gpm_base,
                                 "gpm", "gpm", "-", "-")
        anchor_static(loc, loc_rows, anchor_log)

        # ── GNN base (each seed, using the fold where this region was the test set -- matching the test-fold convention used in 013) ──
        for seed in SEEDS:
            fold_dir_name = fold_of[seed][loc]
            gd_path = (TRAIN_ROOT / f"seed_{seed}" / CONFIG / fold_dir_name
                       / "grid_demands" / f"{loc}_grid_demands.pickle")
            with open(gd_path, "rb") as fh:
                gd = pickle.load(fh)
            gnn_base = np.asarray(gd["GNN"], dtype=float)
            assert len(gnn_base) == len(grid_gdf), \
                f"{loc} seed{seed}: grid_demands length does not match grid-point count"

            gnn_rows = analyse_base(
                ctx, d_true, eps, gnn_base, "gnn",
                f"gnn_seed{seed}_{fold_dir_name}", str(seed), fold_dir_name)
            anchor_gnn(loc, seed, gnn_rows, anchor_log)
            loc_rows += gnn_rows

        records += loc_rows
        np_gnn = [r for r in loc_rows
                  if r["base_type"] == "gnn" and r["signal"] == "NP"]
        print(f"  gnn x NP: median pearson_log "
              f"{np.median([r['pearson_log'] for r in np_gnn]):.3f}, "
              f"median dRMSE {np.median([r['delta_rmse'] for r in np_gnn]):.3f} | "
              f"anchors so far: {len(anchor_log)}, all passed")

    raw_df = pd.DataFrame(records)
    emp = build_phase_points(raw_df)

    # ── 2. Row-count check + write au_phase_points.csv (UK schema) ──
    expected = {"uniform": N_SA4 * len(SIGNALS),
                "gpm": N_SA4 * len(SIGNALS),
                "gnn": len(SEEDS) * N_SA4 * len(SIGNALS)}
    expected["total"] = sum(expected.values())
    actual_counts = {bt: int((emp["base_type"] == bt).sum())
                     for bt in ("uniform", "gpm", "gnn")}
    actual_counts["total"] = int(len(emp))
    assert actual_counts == expected, \
        f"Row count mismatch: actual={actual_counts} vs expected={expected}"

    emp[UK_SCHEMA].to_csv(POINTS_CSV, index=False, encoding="utf-8-sig")
    print(f"\nau_phase_points.csv: {len(emp)} rows (schema column-aligned with "
          f"UK's empirical_points) -> {POINTS_CSV}")

    # ── 3. Merge with UK data points (transcribed only, never recomputed; float round_trip preserves bit-for-bit values) ──
    uk = pd.read_csv(UK_EMPIRICAL_CSV, dtype={"seed": str, "fold": str},
                     float_precision="round_trip")
    assert list(uk.columns) == UK_SCHEMA, \
        "UK empirical_points.csv columns do not match the expected schema -- check for upstream changes"
    assert len(uk) == 240, f"UK data point count {len(uk)} != 240"
    au_out = emp[UK_SCHEMA].copy()
    uk_out = uk.copy()
    au_out.insert(0, "country", "AU")
    uk_out.insert(0, "country", "UK")
    merged = pd.concat([uk_out, au_out], ignore_index=True)
    merged.to_csv(MERGED_CSV, index=False, encoding="utf-8-sig")
    print(f"uk_au_phase_points_merged.csv: {len(merged)} rows"
          f"(UK {len(uk_out)} + AU {len(au_out)}) -> {MERGED_CSV}")

    # ── 4. Confusion matrix + core test (generated programmatically, not hand-written) ──
    cm_all = confusion_matrix_of(emp)
    cm_by_bt = {bt: confusion_matrix_of(emp[emp["base_type"] == bt])
                for bt in ("uniform", "gpm", "gnn")}

    gnn = emp[emp["base_type"] == "gnn"]
    gnn_np = gnn[gnn["signal"] == "NP"]
    static = emp[emp["base_type"].isin(["uniform", "gpm"])]

    # Core test: do AU's GNN x multiplicative combinations systematically land in the "help" region (the mirror case of the UK "P2" finding)?
    au_gnn_value = float(min(gnn["predicted_help"].mean(),
                             gnn["actual_help"].mean()))
    au_gnn_np_value = float(min(gnn_np["predicted_help"].mean(),
                                gnn_np["actual_help"].mean()))
    au_static_value = float(min(static["predicted_help"].mean(),
                                static["actual_help"].mean()))

    # UK contrast (transcribed only): full GNN arm set + GNN x NP (the "hurt" side in the UK 025 finding)
    uk_gnn = uk[uk["base_type"] == "gnn"]
    uk_gnn_np = uk_gnn[uk_gnn["signal"] == "NP"]
    uk_gnn_np_hurt_value = float(min((~uk_gnn_np["predicted_help"]).mean(),
                                     (~uk_gnn_np["actual_help"]).mean()))

    # "Two sides of the same theory line": AU gnn x NP lands in the help region AND UK gnn x NP lands in the hurt region
    two_sides_value = float(min(au_gnn_np_value, uk_gnn_np_hurt_value))

    eps_prot = {
        "rule": f"eps = total regional demand x {EPS_RATIO}",
        "total_protected_any": int(raw_df["n_eps_protected_any"].sum()),
        "rows_with_any_protection": int((raw_df["n_eps_protected_any"] > 0).sum()),
        "by_base_type": {
            bt: {
                "protected_true": int(raw_df.loc[raw_df["base_type"] == bt,
                                                 "n_eps_protected_true"].sum()),
                "protected_base": int(raw_df.loc[raw_df["base_type"] == bt,
                                                 "n_eps_protected_base"].sum()),
                "protected_corrected": int(
                    raw_df.loc[raw_df["base_type"] == bt,
                               "n_eps_protected_corrected"].sum()),
                "protected_any": int(raw_df.loc[raw_df["base_type"] == bt,
                                                "n_eps_protected_any"].sum()),
            } for bt in ("uniform", "gpm", "gnn")
        },
    }

    anchors_failed = [a for a in anchor_log if a["status"] != "ok"]
    static_devs = [a["abs_dev"] for a in anchor_log
                   if a["tol"] == STATIC_ANCHOR_TOL]
    gnn_devs = [a["abs_dev"] for a in anchor_log if a["tol"] == GNN_ANCHOR_TOL]

    summary = {
        "meta": {
            "generated_by": "014_phase_placement_check.py",
            "purpose": "Phase-diagram placement check (AU 180 combinations + UK 240-point merged view)",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "theory": "help <=> rho(r,c) > sigma_c/(2*sigma_r); coordinates = per-substation "
                      "(corr(log F, log rho_resid), std(log F)/std(log rho_resid))"
                      "(matching the UK 024/025 scripts)",
            "theory_line": "alpha* = sigma_ratio / 2",
            "prediction_rule": "predicted_help <=> pearson_log > sigma_ratio / 2",
            "actual_rule": "actual_help <=> delta_rmse < 0",
            "verdict_rule": VERDICT_RULE_TEXT,
            "inputs": [
                "training/seed_*/baseline/fold*/grid_demands (GNN base, read-only)",
                "static/au_static_metrics.csv (anchor, read-only)",
                "training/seed_*/baseline/kfold_test_rmse.csv (anchor, read-only)",
                str(UK_EMPIRICAL_CSV.relative_to(PROJECT_ROOT)) + " (merge source, read-only)",
            ],
            "randomness": "This step has no randomness (deterministic post-processing)",
            "notes": [
                "Substation-level analysis only: agent (grid-point) level has no "
                "observed ground truth, so grid-point-level residuals are undefined "
                "(as established for the UK case; the same applies here).",
                "The GNN base = 012's grid_demands['GNN'] (baseline configuration); "
                "for each (seed, region) only the fold where that region served as "
                "the test set is used (matching the test-fold convention in 013).",
                "Conclusion fields (value/verdict) are always generated programmatically, never hand-written.",
                "Delta RMSE is computed on raw (non-eps-protected) allocation values (actual_col='peak_mw').",
            ],
            "deviations": [
                {"id": 1, "item": "Demand convention = peak_mw (MW) vs UK's 'Demand (MVA)'",
                 "detail": "An established, case-internal-consistent choice: cross-case comparisons only look at mechanism direction, not absolute values"},
                {"id": 2, "item": "proximity/factor inputs = B2 precomputed npz (EPSG:7856)",
                 "detail": "Inherited via 011's load_location_context "
                           "(the compute_factors formula itself is unchanged)"},
                {"id": 3, "item": "GNN base key name = 'GNN' (the UK script uses 'gnn_demand')",
                 "detail": "Same semantics: the GNN demand array on the test fold under the baseline configuration"},
                {"id": 4, "item": "The P arm is fully anchored here (the UK script records it as not_anchored)",
                 "detail": "AU's UniP/GPMpostP/GNNpostP all have frozen rows available -- an upgrade in anchor coverage"},
            ],
        },
        "row_counts": {"expected": expected, "actual": actual_counts},
        "eps_protection": eps_prot,
        "anchor_check": {
            "n_anchors": len(anchor_log),
            "n_failed": len(anchors_failed),
            "max_abs_dev_static_anchors": float(max(static_devs)),
            "static_tol": STATIC_ANCHOR_TOL,
            "max_abs_dev_gnn_anchors": float(max(gnn_devs)),
            "gnn_tol": GNN_ANCHOR_TOL,
            "gnn_tolerance_note": "kfold csv is stored at 4 decimal places -> half a ulp (5e-5) plus slack"
                                  "(013's reaggregation measured a maximum deviation of 4.99e-5 for the same pipeline)",
            "all_passed": len(anchors_failed) == 0,
        },
        "confusion_matrix": cm_all,
        "confusion_by_base_type": cm_by_bt,
        "au_gnn_mult_in_help_region": {
            "description": "Core test: do AU's GNN x multiplicative combinations "
                           "(all signals) land in the theoretical help region and "
                           "actually improve (the mirror case of the UK 025 finding)?",
            **region_cell(gnn),
            "value": au_gnn_value,
            "value_rule": "value = min(frac_predicted_help, frac_actual_help)",
            "verdict": verdict_of(au_gnn_value),
            "by_signal": {sig: region_cell(gnn[gnn["signal"] == sig])
                          for sig in SIGNALS},
        },
        "au_gnn_np_in_help_region": {
            "description": "Does the GNN x NP arm (the exact counterpart of the UK finding) land in the help region?",
            **region_cell(gnn_np),
            "value": au_gnn_np_value,
            "value_rule": "value = min(frac_predicted_help, frac_actual_help)",
            "verdict": verdict_of(au_gnn_np_value),
        },
        "au_static_in_help_region": {
            "description": "Do the static arms (uniform+gpm) land in the help region (the AU replication of the UK finding)?",
            **region_cell(static),
            "value": au_static_value,
            "value_rule": "value = min(frac_predicted_help, frac_actual_help)",
            "verdict": verdict_of(au_static_value),
        },
        "uk_contrast": {
            "note": "UK numbers are transcribed only from exp_r26/empirical_points.csv, never recomputed",
            "uk_gnn_all_signals": {
                "n_rows": int(len(uk_gnn)),
                "frac_predicted_help": float(uk_gnn["predicted_help"].mean()),
                "frac_actual_help": float(uk_gnn["actual_help"].mean()),
                "median_pearson_log": float(uk_gnn["pearson_log"].median()),
                "median_sigma_ratio": float(uk_gnn["sigma_ratio"].median()),
            },
            "uk_gnn_np": {
                "n_rows": int(len(uk_gnn_np)),
                "frac_predicted_hurt": float((~uk_gnn_np["predicted_help"]).mean()),
                "frac_actual_hurt": float((~uk_gnn_np["actual_help"]).mean()),
                "median_pearson_log": float(uk_gnn_np["pearson_log"].median()),
                "median_sigma_ratio": float(uk_gnn_np["sigma_ratio"].median()),
            },
            "au_vs_uk_gnn_medians": {
                "au_median_pearson_log": float(gnn["pearson_log"].median()),
                "au_median_sigma_ratio": float(gnn["sigma_ratio"].median()),
                "uk_median_pearson_log": float(uk_gnn["pearson_log"].median()),
                "uk_median_sigma_ratio": float(uk_gnn["sigma_ratio"].median()),
            },
        },
        "same_theory_line_two_sides": {
            "description": "Two-sides-of-the-same-theory-line narrative: AU gnn x NP "
                           "lands in the help side AND UK gnn x NP lands in the hurt "
                           "side (both expressed as min(predicted, actual))",
            "au_gnn_np_help_value": au_gnn_np_value,
            "uk_gnn_np_hurt_value": uk_gnn_np_hurt_value,
            "value": two_sides_value,
            "value_rule": "value = min(au_gnn_np_help_value, uk_gnn_np_hurt_value)",
            "verdict": verdict_of(two_sides_value),
        },
        "merged_output": {
            "file": MERGED_CSV.name,
            "n_uk": int(len(uk_out)), "n_au": int(len(au_out)),
            "n_total": int(len(merged)),
            "note": "Draft material for the three-case-study phase diagram figure (a third country can be appended once the Germany training run is complete)",
        },
    }
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                            encoding="utf-8")
    print(f"phase_placement_summary.json -> {SUMMARY_JSON}")

    # ── 5. Console summary ──
    print("\n" + "═" * 70)
    print("Core readout (full detail in phase_placement_summary.json)")
    print("═" * 70)
    print(f"{len(anchor_log)} anchor checks, all passed | max static deviation {max(static_devs):.3e} | "
          f"max GNN deviation {max(gnn_devs):.3e}")
    print(f"eps-protected samples: {eps_prot['total_protected_any']}"
          f"(affecting {eps_prot['rows_with_any_protection']} rows)")
    print(f"Confusion matrix: help/help={cm_all['pred_help_actual_help']}, "
          f"help/hurt={cm_all['pred_help_actual_hurt']}, "
          f"hurt/help={cm_all['pred_hurt_actual_help']}, "
          f"hurt/hurt={cm_all['pred_hurt_actual_hurt']} "
          f"(total {cm_all['total']}, accuracy {cm_all['accuracy']:.3f})")
    g = summary["au_gnn_mult_in_help_region"]
    print(f"AU GNN x multiplicative (all signals, {g['n_rows']} rows): predicted help "
          f"{g['frac_predicted_help']:.3f} / actual help {g['frac_actual_help']:.3f} "
          f"-> value={g['value']:.3f} [{g['verdict']}]")
    print(f"  Typical coordinates: median (rho, sigma_c/sigma_r) = ({g['median_pearson_log']:.3f}, "
          f"{g['median_sigma_ratio']:.3f})  vs UK GNN "
          f"({summary['uk_contrast']['uk_gnn_all_signals']['median_pearson_log']:.3f}, "
          f"{summary['uk_contrast']['uk_gnn_all_signals']['median_sigma_ratio']:.3f})")
    ts = summary["same_theory_line_two_sides"]
    print(f"Two sides of the same theory line: AU help side {ts['au_gnn_np_help_value']:.3f} AND "
          f"UK hurt side {ts['uk_gnn_np_hurt_value']:.3f} -> value="
          f"{ts['value']:.3f} [{ts['verdict']}]")
    print(f"\nOutput directory: {EVAL_DIR}")
    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
