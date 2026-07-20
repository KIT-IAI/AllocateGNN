# -*- coding: utf-8 -*-
"""Australia case study - statistical evaluation (013).

Statistical protocol follows the UK case study's `031_exp_r216_robust_stats.py`.
Statistical primitives are always imported from the shared `revision_statistics`
module rather than reimplemented, matching the pattern used for the shared
correction utilities in scripts 011/012.

=== Core hypotheses (comparison list = au_comparison_registry.csv, written to
disk before testing) ===
- H-AU1 multiplicative correction harms the GNN base: GNNpostNP/N/P vs GNN +
  GNNpostNP vs GNNpostP (the latter is the same definition as the UK
  antagonism-core comparison, exp_r216 #4);
- H-AU2 additive repair: GNNaddNP vs GNNpostNP, GNNaddP vs GNN (same
  definitions as UK exp_r12 #8/#4);
- H-AU3 static base gains: UniNP vs Uni, GPMpostNP vs GPM, GPMpostNP vs
  GPMpostP (the static antagonism control is exp_r216 #7, read from the
  static-baseline table, no seed branch);
- Cross-base interaction: GNNaddP vs GPMpostNP, GNNaddNP vs GPMpostNP (same
  cross-base comparison list as UK's exp_r12);
- prior-loss channel: GNNpriorNP vs GNN (the voronoi_GNN row of the ntl_prox
  training arm vs baseline).

=== Statistical protocol (mirrors the UK script 031 point-for-point, with
parameters adapted for AU) ===
(1) inference unit = SA4 region (n=12, corrected from the original plan's
figure of 13 once the actual region universe was confirmed); (2) main test =
seed-averaged -> paired 12-region differences -> exact 2^12 = 4096 sign-flip
permutation (static-vs-static comparisons use the no-seed branch); (3) CI =
region-level paired bootstrap, outer loop only, B=10^4 (not used to back out
p-values); (4) per-seed permutation p-values are all written to disk, with a
Cauchy combination provided for reference only (not used as a median); (5)
robustness check = mixed-effects model (region x seed crossed random
intercepts); (6) Holm correction = one family per metric (family = the 12
registered comparisons); (7) Moran's I diagnostic (primary queen-contiguity
weights plus a kNN(3) supplement; a significant result triggers a spatial
block-permutation substitute).

=== Sensitivity analyses (purely statistical post-processing, no retraining) ===
- SA3 x 34: station-level residuals are re-aggregated by SA3 and all
  comparisons are rerun. With 34 units this exceeds the exact-enumeration
  guard (_MAX_EXACT_N=20), so a Monte-Carlo sign-flip permutation with
  B=10^5 is used instead (the task brief originally specified B=10^4; the
  stricter B=10^5 required by the design spec is used instead and recorded
  in summary.deviations), with a fixed numpy default_rng seed and an
  add-one correction. The corr metric degenerates for SA3 units with few
  stations (n=1 is undefined, n=2 is always +/-1), so the corr sub-analysis
  is restricted to SA3 units with >=3 stations (the unit count is recorded).
- PV: all comparisons are rerun under an alternate protocol that excludes 5
  stations flagged for a day/night ratio anomaly (pv_level_fy2009 ==
  'suspected'). Static arms read the frozen excl_pv rows directly; GNN arms
  are re-aggregated from grid_demands (allocation/factors/assignment are not
  rerun -- this is purely a station-set filter applied at the statistics
  layer).

=== Re-aggregation pipeline anchoring ===
GNN arms: grid_demands (written to disk by 012, keyed by the same arm labels
used throughout the pipeline) -> the shared two-stage aggregation -> per-SA4
metrics must match the frozen kfold_test_*.csv (stored at 4 decimal places,
so the tolerance is half a storage unit, 5e-5). Static arms: recomputed using
the same definitions as 011 (importing 011's functions rather than
duplicating the implementation) -> per-SA4 metrics must match
au_static_metrics.csv (stored at full precision, tolerance 1e-9). Any anchor
failure is a hard failure.

=== Directional consistency with the UK results ===
UK reference values are read-only: exp_r12/full_matrix.csv (the 21-arm x
16-region matrix, source of Delta) plus exp_r216/robust_tests.csv and
exp_r12/new_comparisons.csv (frozen p-values/significance); the UK arm
corresponding to the prior channel (the ntl_prox config) is not in those
frozen tables, so Delta is instead computed from exp0_kfold_prior's
kfold_test_*.csv (read-only) and anchored against exp_r216 #8 (GNNpriorN vs
GNN, frozen), with the source recorded row-by-row in the uk_source column.

=== Deviations from the UK protocol (recorded here, read before writing up
results) ===
1. The task brief said "SA3 x 35"; the actual count is 34, corrected once
   the region universe was confirmed (see b1_regions.json; same convention
   used in scripts 011/012);
2. The task brief said "MC permutation B=10^4"; B=10^5 is used instead, per
   the design spec's stricter ">=10^5" requirement;
3. Moran kNN centroid CRS: revision_statistics.build_queen_weights returns
   centroids in EPSG:27700 (a value frozen for the UK case, not valid for
   AU) -> this script computes its own EPSG:7856 centroids (matching the CRS
   used elsewhere in the AU pipeline).

All conclusion fields are generated programmatically from numeric rules
rather than hand-written; everything runs on CPU (capped at 4 threads), and
all random sources use fixed seeds. Run with:
`python 013_statistical_evaluation.py` (allocategnn env).
"""

from __future__ import annotations

import os

# The thread cap must be set before numpy/scipy are first imported (the GPU is
# occupied by the Germany-case training run, so this script runs entirely on
# CPU with a 4-thread cap)
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import importlib.util
import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Force UTF-8 output on the Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# sys.path injection (repo root + UK experiment directory)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
_p = SCRIPT_DIR
while not (_p / "SpatialAllocation").exists():
    if _p.parent == _p:
        raise RuntimeError("Could not find the repository root (SpatialAllocation package)")
    _p = _p.parent
PROJECT_ROOT = _p
UK_EXP_DIR = PROJECT_ROOT / "StudyCase" / "British_weighter_experiments"
for _extra in (str(UK_EXP_DIR), str(PROJECT_ROOT)):
    if _extra not in sys.path:
        sys.path.insert(0, _extra)

from scipy.stats import pearsonr                                       # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error    # noqa: E402

import shared_correction_utils as scu                                  # noqa: E402  # shared post-hoc correction utilities (read-only import from the UK case study)
# Statistical primitives are always imported from the shared module rather than reimplemented
from revision_statistics import (                                      # noqa: E402
    exact_sign_flip_permutation, paired_region_bootstrap,
    cauchy_combination, morans_i, build_queen_weights, build_knn_weights,
    holm,
)

warnings.filterwarnings("ignore", category=FutureWarning)


def _load_script_module(name: str, path: Path):
    """Load a numerically-prefixed script module via importlib, so its functions can be reused directly instead of being duplicated."""
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# au011 = the static-baseline module's shared function definitions
# (compute_uniform_base/compute_gpm_base/loaders);
# uk031 = the UK robust-stats script's build_contiguity_blocks (spatial block
# permutation) and fit_crossed_mixedlm functions
au011 = _load_script_module("au011", SCRIPT_DIR / "011_static_baselines.py")
uk031 = _load_script_module("uk031", UK_EXP_DIR / "031_exp_r216_robust_stats.py")

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
PROCESSED = SCRIPT_DIR / "data" / "processed"
TRAIN_ROOT = PROCESSED / "training"
STATIC_CSV = PROCESSED / "static" / "au_static_metrics.csv"
EVAL_DIR = PROCESSED / "evaluation"

UK_FULL_MATRIX = UK_EXP_DIR / "results" / "exp_r12" / "full_matrix.csv"
UK_R216_CSV = UK_EXP_DIR / "results" / "exp_r216" / "robust_tests.csv"
UK_R12_CSV = UK_EXP_DIR / "results" / "exp_r12" / "new_comparisons.csv"
UK_EXP0_DIR = UK_EXP_DIR / "results" / "exp0_kfold_prior"
DE_STATIC_CSV = (PROJECT_ROOT / "StudyCase" / "Germany" / "results"
                 / "static_allocation" / "boerde_static_metrics.csv")

SEEDS = [42, 123, 456]
METRICS = ["rmse", "mae", "corr"]
CONFIGS = ["baseline", "ntl_prox"]
ALPHA = 0.05
DEMAND_COL = "peak_mw"              # primary demand column (annual peak, MW)

# Region universe (corrected once the actual data was confirmed; order fixed to match 012)
ALL_LOCATIONS = list(au011.__dict__.get("ALL_LOCATIONS", [])) or [
    "Central_Coast", "Hunter_Valley_exc_Newcastle",
    "Newcastle_and_Lake_Macquarie", "Sydney_City_and_Inner_South",
    "Sydney_Eastern_Suburbs", "Sydney_Inner_South_West", "Sydney_Inner_West",
    "Sydney_North_Sydney_and_Hornsby", "Sydney_Northern_Beaches",
    "Sydney_Parramatta", "Sydney_Ryde", "Sydney_Sutherland",
]
N_SA4 = 12
N_SA3 = 34                          # corrected once the actual data was confirmed (originally planned as 35)
SA3_CORR_MIN_STATIONS = 3           # min station count for the SA3 corr sub-analysis (n=1 undefined / n=2 always +/-1)

GNN_ARMS = ["GNN", "GNNpostN", "GNNpostP", "GNNpostNP",
            "GNNaddN", "GNNaddP", "GNNaddNP"]
SIGNALS = ["N", "P", "NP"]

# All random sources use fixed seeds (deterministic); 613xxx is this script's
# dedicated seed namespace, derived from row order
B_BOOT = 10000
MC_B = 100000                       # SA3 Monte-Carlo permutation count, >= 10^5 per the design spec
BOOT_SEED_BASE = 613001             # main-branch bootstrap
PV_BOOT_SEED_BASE = 613101          # PV-branch bootstrap
SA3_BOOT_SEED_BASE = 613201         # SA3-branch bootstrap
SA3_MC_SEED_BASE = 613301           # SA3-branch Monte-Carlo permutation
MORAN_SEED_QUEEN_BASE = 613401
MORAN_SEED_KNN_BASE = 613501
MORAN_N_PERM = 999

# GNN re-aggregation anchor tolerance: the kfold csv is stored at 4 decimal
# places, so the tolerance is half a storage unit
GNN_ANCHOR_TOL = 5e-5 + 1e-9
# Static re-aggregation anchor tolerance: au_static_metrics.csv is stored at
# full precision and the 011 pipeline is deterministic, so tolerance = 1e-9
STATIC_ANCHOR_TOL = 1e-9

# ---------------------------------------------------------------------------
# Comparison registry (written to comparison_registry.csv before testing)
# arm spec = (arm label, type 'gnn'|'static', config); uk_* = UK reference source
# ---------------------------------------------------------------------------
COMPARISONS = [
    dict(cid=1, label="GNNpostNP vs GNN", hyp="H-AU1",
         a=("GNNpostNP", "gnn", "baseline"), b=("GNN", "gnn", "baseline"),
         uk_frozen=None),
    dict(cid=2, label="GNNpostN vs GNN", hyp="H-AU1",
         a=("GNNpostN", "gnn", "baseline"), b=("GNN", "gnn", "baseline"),
         uk_frozen=("r216", 3)),
    dict(cid=3, label="GNNpostP vs GNN", hyp="H-AU1",
         a=("GNNpostP", "gnn", "baseline"), b=("GNN", "gnn", "baseline"),
         uk_frozen=("r216", 2)),
    dict(cid=4, label="GNNpostNP vs GNNpostP", hyp="H-AU1",
         a=("GNNpostNP", "gnn", "baseline"), b=("GNNpostP", "gnn", "baseline"),
         uk_frozen=("r216", 4)),
    dict(cid=5, label="GNNaddNP vs GNNpostNP", hyp="H-AU2",
         a=("GNNaddNP", "gnn", "baseline"), b=("GNNpostNP", "gnn", "baseline"),
         uk_frozen=("r12", 8)),
    dict(cid=6, label="GNNaddP vs GNN", hyp="H-AU2",
         a=("GNNaddP", "gnn", "baseline"), b=("GNN", "gnn", "baseline"),
         uk_frozen=("r12", 4)),
    dict(cid=7, label="UniNP vs Uni", hyp="H-AU3",
         a=("UniNP", "static", ""), b=("Uni", "static", ""),
         uk_frozen=None),
    dict(cid=8, label="GPMpostNP vs GPM", hyp="H-AU3",
         a=("GPMpostNP", "static", ""), b=("GPM", "static", ""),
         uk_frozen=None),
    dict(cid=9, label="GPMpostNP vs GPMpostP", hyp="H-AU3",
         a=("GPMpostNP", "static", ""), b=("GPMpostP", "static", ""),
         uk_frozen=("r216", 7)),
    dict(cid=10, label="GNNaddP vs GPMpostNP", hyp="cross_base",
         a=("GNNaddP", "gnn", "baseline"), b=("GPMpostNP", "static", ""),
         uk_frozen=("r12", 1)),
    dict(cid=11, label="GNNaddNP vs GPMpostNP", hyp="cross_base",
         a=("GNNaddNP", "gnn", "baseline"), b=("GPMpostNP", "static", ""),
         uk_frozen=("r12", 2)),
    dict(cid=12, label="GNNpriorNP vs GNN", hyp="prior_channel",
         a=("GNN", "gnn", "ntl_prox"), b=("GNN", "gnn", "baseline"),
         uk_frozen=None),   # the corresponding UK arm is not in the frozen tables -> read from exp0 (read-only), anchored against r216#8
]
HYP_CIDS = {"H-AU1": [1, 2, 3, 4], "H-AU2": [5, 6], "H-AU3": [7, 8, 9],
            "cross_base": [10, 11], "prior_channel": [12]}
N_COMPARISONS = len(COMPARISONS)


# ---------------------------------------------------------------------------
# AU-specific adaptation of the sign-flip permutation primitive: the shared
# module only has an exact 2^n version, and n=34 exceeds the exact-enumeration
# guard (_MAX_EXACT_N=20), so a Monte-Carlo version with an add-one correction
# is implemented locally here.
# ---------------------------------------------------------------------------
def mc_sign_flip_permutation(diffs, B: int = MC_B, seed: int = 0) -> dict:
    """Monte-Carlo sign-flip permutation test for paired differences (two-sided, add-one correction).

    p = (1 + #{|T_perm| >= |T_obs|}) / (B + 1); min_attainable_p = 1/(B+1).
    The test statistic and tolerance handling match exact_sign_flip_permutation
    (T = sum(+-d_i)).
    """
    d = np.asarray(diffs, dtype=float)
    if d.ndim != 1 or len(d) == 0:
        raise ValueError("diffs must be a non-empty 1D array")
    if not np.all(np.isfinite(d)):
        raise ValueError("diffs contains NaN/Inf")
    rng = np.random.default_rng(seed)
    signs = rng.integers(0, 2, size=(B, len(d))).astype(np.float64) * 2.0 - 1.0
    t_all = signs @ d
    t_obs = float(d.sum())
    tol = 1e-12 * max(1.0, float(np.abs(d).sum()))
    p = float((1 + np.sum(np.abs(t_all) >= abs(t_obs) - tol)) / (B + 1))
    return {"p": p, "t_obs": t_obs, "n": len(d), "B": int(B),
            "min_attainable_p": 1.0 / (B + 1), "seed": int(seed)}


# ---------------------------------------------------------------------------
# Loading frozen artifacts (input to the main branch)
# ---------------------------------------------------------------------------
_KFOLD_CACHE: dict = {}


def load_gnn_region_values(arm: str, config: str, metric: str, seed: int) -> np.ndarray:
    """Read the 12 region values for a GNN arm at a single seed (kfold_test_{metric}.csv, column order = ALL_LOCATIONS)."""
    key = (config, metric, seed)
    if key not in _KFOLD_CACHE:
        path = TRAIN_ROOT / f"seed_{seed}" / config / f"kfold_test_{metric}.csv"
        _KFOLD_CACHE[key] = pd.read_csv(path, index_col=0)
    df = _KFOLD_CACHE[key]
    return df.loc[f"voronoi_{arm}", ALL_LOCATIONS].values.astype(float)


_STATIC_DF: pd.DataFrame | None = None


def load_static_region_values(arm: str, metric: str, station_set: str = "all") -> np.ndarray:
    """Read the 12 region values for a static arm (frozen row in au_static_metrics.csv)."""
    global _STATIC_DF
    if _STATIC_DF is None:
        _STATIC_DF = pd.read_csv(STATIC_CSV, encoding="utf-8-sig")
    row = _STATIC_DF[(_STATIC_DF["arm"] == arm)
                     & (_STATIC_DF["metric"] == metric)
                     & (_STATIC_DF["station_set"] == station_set)]
    assert len(row) == 1, f"Static arm {arm}/{metric}/{station_set} row count {len(row)} != 1"
    return row[ALL_LOCATIONS].values.astype(float)[0]


# ---------------------------------------------------------------------------
# UK reference values (read-only)
# ---------------------------------------------------------------------------
def load_uk_references() -> dict:
    """UK Delta and frozen p-values: full_matrix (Delta) + r216/r12 (p, significance) + exp0 (prior-channel Delta)."""
    fm = pd.read_csv(UK_FULL_MATRIX)
    r216 = pd.read_csv(UK_R216_CSV)
    r12 = pd.read_csv(UK_R12_CSV)

    def fm_mean(arm: str, metric: str) -> float:
        row = fm[(fm["arm"] == arm) & (fm["metric"] == metric)]
        assert len(row) == 1, f"full_matrix missing {arm}/{metric}"
        return float(row["mean_16regions"].iloc[0])

    def frozen_row(src: str, cid: int, metric: str) -> dict | None:
        df = r216 if src == "r216" else r12
        hit = df[(df["comparison_id"] == cid) & (df["metric"] == metric)]
        if len(hit) != 1:
            return None
        h = hit.iloc[0]
        return {"mean_diff": float(h["mean_diff"]), "perm_p": float(h["perm_p"]),
                "holm_p": float(h["holm_p"]), "sig": bool(h["new_sig"]),
                "src": (f"exp_r216/robust_tests.csv #{cid}" if src == "r216"
                        else f"exp_r12/new_comparisons.csv #{cid}")}

    # prior channel (cid 12): exp0's ntl_prox vs baseline voronoi_GNN (supplemental read-only reference)
    prior_delta = {}
    for metric in METRICS:
        per_seed = []
        for s in SEEDS:
            a = pd.read_csv(UK_EXP0_DIR / f"seed_{s}" / "ntl_prox"
                            / f"kfold_test_{metric}.csv", index_col=0)
            b = pd.read_csv(UK_EXP0_DIR / f"seed_{s}" / "baseline"
                            / f"kfold_test_{metric}.csv", index_col=0)
            cols = [c for c in a.columns if c != "mean"]
            assert cols == [c for c in b.columns if c != "mean"], "exp0's two configs have inconsistent column order"
            assert "voronoi_GNN" in a.index and "voronoi_GNN" in b.index
            per_seed.append(a.loc["voronoi_GNN", cols].values.astype(float)
                            - b.loc["voronoi_GNN", cols].values.astype(float))
        prior_delta[metric] = float(np.mean(per_seed, axis=0).mean())

    refs = {}
    for comp in COMPARISONS:
        cid = comp["cid"]
        (la, ta, _), (lb, tb, _) = comp["a"], comp["b"]
        for metric in METRICS:
            entry = {"uk_delta": None, "uk_perm_p": None, "uk_holm_p": None,
                     "uk_sig": None, "uk_source": None}
            if cid == 12:
                entry["uk_delta"] = prior_delta[metric]
                entry["uk_source"] = ("exp0_kfold_prior ntl_prox-baseline "
                                      "voronoi_GNN (supplemental read-only); "
                                      "frozen anchor = exp_r216 #8 GNNpriorN vs GNN")
                anchor = frozen_row("r216", 8, metric)
                if anchor is not None:
                    entry["uk_prior_anchor_delta"] = anchor["mean_diff"]
                    entry["uk_prior_anchor_perm_p"] = anchor["perm_p"]
            else:
                frozen = (frozen_row(*comp["uk_frozen"], metric)
                          if comp["uk_frozen"] else None)
                if frozen is not None:
                    entry.update({"uk_delta": frozen["mean_diff"],
                                  "uk_perm_p": frozen["perm_p"],
                                  "uk_holm_p": frozen["holm_p"],
                                  "uk_sig": frozen["sig"],
                                  "uk_source": frozen["src"]})
                else:
                    entry["uk_delta"] = fm_mean(la, metric) - fm_mean(lb, metric)
                    entry["uk_source"] = "exp_r12/full_matrix.csv (Delta of mean_16regions)"
            refs[(cid, metric)] = entry
    return refs


# ---------------------------------------------------------------------------
# Station-level re-aggregation (SA3 sensitivity + PV alternate protocol + anchoring)
# ---------------------------------------------------------------------------
def per_sa3_metrics(subs_result: pd.DataFrame) -> dict:
    """Per-substation results table -> the three metrics for each SA3 (same formulas as evaluate_allocation).

    corr is only computed for SA3 units with >= SA3_CORR_MIN_STATIONS stations (otherwise recorded as NaN).
    """
    out = {}
    for code, grp in subs_result.groupby("ITL3"):
        actual = grp[DEMAND_COL].values.astype(float)
        alloc = grp["allocated_demand"].values.astype(float)
        rmse = float(np.sqrt(mean_squared_error(actual, alloc)))
        mae = float(mean_absolute_error(actual, alloc))
        corr = np.nan
        if len(grp) >= SA3_CORR_MIN_STATIONS:
            with np.errstate(all="ignore"):
                c, _ = pearsonr(actual, alloc)
            corr = float(c)
        out[str(code)] = {"rmse": rmse, "mae": mae, "corr": corr,
                          "n_stations": int(len(grp))}
    return out


def recompute_station_level() -> dict:
    """Station-level re-aggregation for the 14 static arms + 7 GNN arms x 2 configs x 3 seeds.

    Returns:
        gnn_sa4_all / gnn_sa4_excl : {(config, arm, metric, seed): {loc: v}}
        gnn_sa3                    : {(config, arm, metric, seed): {sa3: v}}
        static_sa4_all/_excl       : {(arm, metric): {loc: v}} (anchoring use only)
        static_sa3                 : {(arm, metric): {sa3: v}}
        sa3_n_stations             : {sa3: n}
        anchor_report              : dict
    """
    print("  Loading region/station/grid context (reusing the shared definitions from 011)...")
    b2 = json.loads((SCRIPT_DIR / "docs" / "b1b2" / "b2_features.json")
                    .read_text(encoding="utf-8"))
    regions = au011.load_regions()
    stations = au011.load_stations(regions)
    assert regions["SA3"].nunique() == N_SA3
    assignment_cache: dict = {}
    contexts = {}
    for loc in ALL_LOCATIONS:
        contexts[loc] = au011.load_location_context(
            loc, regions, stations, b2, assignment_cache)
        print(f"    {loc}: {len(contexts[loc]['grid_gdf'])} grid points / "
              f"{len(contexts[loc]['subs_sub'])} stations / assignment ready")

    sa3_n_stations = {}
    for loc in ALL_LOCATIONS:
        for code, grp in contexts[loc]["subs_sub"].groupby("ITL3"):
            sa3_n_stations[str(code)] = int(len(grp))
    assert len(sa3_n_stations) == N_SA3, \
        f"SA3 unit count {len(sa3_n_stations)} != {N_SA3}"

    def eval_station_table(ctx, demand_arr):
        """Demand array -> per-substation table + (all/excl_pv region metrics, per-SA3 metrics)."""
        subs_result = scu.aggregate_by_assignment(
            ctx["subs_sub"], ctx["assignment"], np.asarray(demand_arr, dtype=float))
        m_all = scu.evaluate_allocation(subs_result, actual_col=DEMAND_COL)
        m_excl = scu.evaluate_allocation(subs_result[~ctx["pv_mask"]],
                                         actual_col=DEMAND_COL)
        return m_all, m_excl, per_sa3_metrics(subs_result)

    # ── Static 14 arms (recomputed using the same definitions as 011: base-arm functions imported and reused + scu correction) ──
    print("  Recomputing station-level values for static arms (same pipeline as 011)...")
    static_sa4_all = {}
    static_sa4_excl = {}
    static_sa3 = {}
    for loc in ALL_LOCATIONS:
        ctx = contexts[loc]
        grid_gdf, region_sub = ctx["grid_gdf"], ctx["region_sub"]
        uni_base = au011.compute_uniform_base(grid_gdf, region_sub)
        gpm_base, _ = au011.compute_gpm_base(grid_gdf, region_sub, ctx["subs_sub"])
        for label, base_name, form, sig in au011.ARMS:
            base_arr = uni_base if base_name == "uniform" else gpm_base
            if form == "none":
                arr = np.asarray(base_arr, dtype=float)
            elif form == "mult":
                arr = scu.apply_standard_multiplicative(
                    base_arr, ctx["factors"][sig], grid_gdf, region_sub)
            else:
                arr = scu.apply_additive_correction(
                    base_arr, ctx["factors"][sig], grid_gdf, region_sub)
            m_all, m_excl, m_sa3 = eval_station_table(ctx, arr)
            for metric in METRICS:
                static_sa4_all.setdefault((label, metric), {})[loc] = m_all[metric]
                static_sa4_excl.setdefault((label, metric), {})[loc] = m_excl[metric]
                for code, mm in m_sa3.items():
                    static_sa3.setdefault((label, metric), {})[code] = mm[metric]

    # ── GNN arms (from grid_demands, using the test-fold convention: each region taken from the fold where it served as the test set) ──
    print("  Re-aggregating station-level values for GNN arms (test-fold convention from grid_demands)...")
    gnn_sa4_all = {}
    gnn_sa4_excl = {}
    gnn_sa3 = {}
    for config in CONFIGS:
        for seed in SEEDS:
            seed_dir = TRAIN_ROOT / f"seed_{seed}" / config
            splits = json.loads((seed_dir / "kfold_splits.json")
                                .read_text(encoding="utf-8"))
            loc2fold = {loc: fold_name for fold_name, fold in splits.items()
                        for loc in fold["test"]}
            assert sorted(loc2fold) == sorted(ALL_LOCATIONS)
            for loc in ALL_LOCATIONS:
                fold_idx = loc2fold[loc].split("_")[1]
                gd_path = (seed_dir / f"fold{fold_idx}" / "grid_demands"
                           / f"{loc}_grid_demands.pickle")
                with open(gd_path, "rb") as fh:
                    gd = pickle.load(fh)
                assert set(gd) == set(GNN_ARMS), \
                    f"{gd_path}: keys {sorted(gd)} != expected arm labels"
                ctx = contexts[loc]
                for arm, arr in gd.items():
                    arr = np.asarray(arr, dtype=float)
                    assert arr.shape == (len(ctx["grid_gdf"]),), \
                        f"{gd_path}/{arm}: length {arr.shape} != grid-point count"
                    m_all, m_excl, m_sa3 = eval_station_table(ctx, arr)
                    for metric in METRICS:
                        gnn_sa4_all.setdefault(
                            (config, arm, metric, seed), {})[loc] = m_all[metric]
                        gnn_sa4_excl.setdefault(
                            (config, arm, metric, seed), {})[loc] = m_excl[metric]
                        for code, mm in m_sa3.items():
                            gnn_sa3.setdefault(
                                (config, arm, metric, seed), {})[code] = mm[metric]
            print(f"    {config}/seed_{seed}: re-aggregation complete for 12 regions x 7 arms")

    # ── Anchoring (re-aggregation pipeline vs frozen artifacts) ──
    print("  Anchoring the re-aggregation pipeline against frozen artifacts...")
    anchor_report = {
        "gnn_anchor_rule": ("per-SA4 'all' metrics vs frozen kfold_test_*.csv; "
                            "frozen values are stored at 4 decimal places, so tolerance = half a storage unit (5e-5)"),
        "gnn_anchor_tol": GNN_ANCHOR_TOL,
        "static_anchor_rule": ("per-SA4 all/excl_pv metrics vs au_static_metrics.csv "
                               "(stored at full precision, 011 pipeline is deterministic) -> tolerance 1e-9"),
        "static_anchor_tol": STATIC_ANCHOR_TOL,
    }
    gnn_max_dev = 0.0
    for (config, arm, metric, seed), d in gnn_sa4_all.items():
        frozen = load_gnn_region_values(arm, config, metric, seed)
        dev = float(np.abs(np.array([d[loc] for loc in ALL_LOCATIONS]) - frozen).max())
        gnn_max_dev = max(gnn_max_dev, dev)
    anchor_report["gnn_max_abs_dev"] = gnn_max_dev
    anchor_report["gnn_passed"] = bool(gnn_max_dev <= GNN_ANCHOR_TOL)
    if not anchor_report["gnn_passed"]:
        raise RuntimeError(f"GNN re-aggregation anchor failed: max deviation {gnn_max_dev:.3e} > "
                           f"{GNN_ANCHOR_TOL:.1e} (fold selection or aggregation pipeline mismatch)")

    static_max_dev = {"all": 0.0, "excl_pv": 0.0}
    for sset, table in (("all", static_sa4_all), ("excl_pv", static_sa4_excl)):
        for (arm, metric), d in table.items():
            frozen = load_static_region_values(arm, metric, sset)
            dev = float(np.abs(np.array([d[loc] for loc in ALL_LOCATIONS])
                               - frozen).max())
            static_max_dev[sset] = max(static_max_dev[sset], dev)
    anchor_report["static_max_abs_dev"] = static_max_dev
    anchor_report["static_passed"] = bool(
        max(static_max_dev.values()) <= STATIC_ANCHOR_TOL)
    if not anchor_report["static_passed"]:
        raise RuntimeError(f"Static arm recomputation anchor failed: {static_max_dev} (011 pipeline mismatch)")
    print(f"    GNN anchor: max deviation {gnn_max_dev:.3e} (tolerance {GNN_ANCHOR_TOL:.1e}) passed")
    print(f"    Static anchor: max deviation all={static_max_dev['all']:.3e} / "
          f"excl_pv={static_max_dev['excl_pv']:.3e} (tolerance 1e-9) passed")

    return {"gnn_sa4_all": gnn_sa4_all, "gnn_sa4_excl": gnn_sa4_excl,
            "gnn_sa3": gnn_sa3, "static_sa4_all": static_sa4_all,
            "static_sa4_excl": static_sa4_excl, "static_sa3": static_sa3,
            "sa3_n_stations": sa3_n_stations, "anchor_report": anchor_report,
            "contexts": contexts, "regions": regions}


# ---------------------------------------------------------------------------
# Branch runner (shared by main / PV / SA3; mirrors the UK script's main loop structure)
# ---------------------------------------------------------------------------
def run_branch(branch: str, get_vec, units_by_metric: dict,
               exact: bool, boot_seed_base: int,
               mc_seed_base: int | None = None) -> pd.DataFrame:
    """Run the main significance test + CI + Holm correction for every (comparison x metric) pair.

    get_vec(spec, metric, seed) -> unit vector (order = units_by_metric[metric]);
    static arms ignore seed. exact=True uses the exact 2^n permutation, otherwise Monte-Carlo (B=10^5).
    """
    rows = []
    row_idx = 0
    for comp in COMPARISONS:
        for metric in METRICS:
            units = units_by_metric[metric]
            a_spec, b_spec = comp["a"], comp["b"]
            is_gnn = (a_spec[1] == "gnn") or (b_spec[1] == "gnn")

            if a_spec[1] == "gnn":
                a_per_seed = {s: get_vec(a_spec, metric, s) for s in SEEDS}
                a_avg = np.mean([a_per_seed[s] for s in SEEDS], axis=0)
            else:
                a_avg = get_vec(a_spec, metric, None)
            if b_spec[1] == "gnn":
                b_per_seed = {s: get_vec(b_spec, metric, s) for s in SEEDS}
                b_avg = np.mean([b_per_seed[s] for s in SEEDS], axis=0)
            else:
                b_avg = get_vec(b_spec, metric, None)

            diffs = a_avg - b_avg
            assert len(diffs) == len(units), \
                f"{branch}/{comp['label']}/{metric}: vector length != unit count"
            if exact:
                perm = exact_sign_flip_permutation(diffs)
            else:
                perm = mc_sign_flip_permutation(diffs, B=MC_B,
                                                seed=mc_seed_base + row_idx)
            boot = paired_region_bootstrap(diffs, B=B_BOOT,
                                           seed=boot_seed_base + row_idx)
            rows.append({
                "branch": branch,
                "comparison_id": comp["cid"], "comparison": comp["label"],
                "hypothesis": comp["hyp"], "metric": metric,
                "holm_family": f"{branch}_{metric}",
                "seed_branch": "seed_averaged" if is_gnn else "no_seed",
                "n_units": len(units),
                "mean_diff": float(np.mean(diffs)),
                "perm_p": perm["p"], "perm_t_obs": perm["t_obs"],
                "perm_min_attainable_p": perm["min_attainable_p"],
                "perm_kind": "exact_2^n" if exact else f"monte_carlo_B{MC_B}",
                "holm_p": None, "sig": None,
                "ci_lo": boot["ci_lo"], "ci_hi": boot["ci_hi"],
                "boot_B": boot["B"], "boot_seed": boot["seed"],
            })
            row_idx += 1
    df = pd.DataFrame(rows)
    # Holm correction: one family per metric (family = the 12 registered comparisons)
    for metric in METRICS:
        mask = df["metric"] == metric
        pmap = {int(r["comparison_id"]): float(r["perm_p"])
                for _, r in df[mask].iterrows()}
        adj = holm(pmap)
        df.loc[mask, "holm_p"] = df.loc[mask, "comparison_id"].map(adj)
    df["holm_p"] = df["holm_p"].astype(float)
    df["sig"] = df["holm_p"] < ALPHA
    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    t0 = time.time()
    print("=" * 70)
    print("Australia case study - statistical evaluation (013) -- protocol x 12 comparisons x 3 metrics x 3 branches")
    print("=" * 70)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # ── 0. Write the comparison registry to disk before testing ──
    registry_rows = []
    for comp in COMPARISONS:
        (la, ta, ca), (lb, tb, cb) = comp["a"], comp["b"]
        registry_rows.append({
            "comparison_id": comp["cid"], "comparison": comp["label"],
            "hypothesis": comp["hyp"],
            "arm_a": la, "arm_a_type": ta, "arm_a_config": ca,
            "arm_a_source": (f"training/seed_*/{ca}/kfold_test_*.csv"
                             if ta == "gnn" else "static/au_static_metrics.csv"),
            "arm_b": lb, "arm_b_type": tb, "arm_b_config": cb,
            "arm_b_source": (f"training/seed_*/{cb}/kfold_test_*.csv"
                             if tb == "gnn" else "static/au_static_metrics.csv"),
            "seed_branch": ("seed_averaged" if (ta == "gnn" or tb == "gnn")
                            else "no_seed"),
            "uk_frozen_ref": (f"{comp['uk_frozen'][0]}#{comp['uk_frozen'][1]}"
                              if comp["uk_frozen"] else
                              ("exp0_kfold_prior (supplemental read-only)" if comp["cid"] == 12
                               else "exp_r12/full_matrix.csv (Delta)")),
        })
    registry = pd.DataFrame(registry_rows)
    registry.to_csv(EVAL_DIR / "au_comparison_registry.csv",
                    index=False, encoding="utf-8-sig")
    print(f"Comparison registry: {len(registry)} rows -> au_comparison_registry.csv")

    # ── 1. UK reference values ──
    uk_refs = load_uk_references()

    # ── 2. Main branch (SA4 x 12, frozen-artifact input, exact 2^12 permutation) ──
    print("\n[Main branch] SA4 x 12 (exact 2^12 = 4096 permutation)...")
    sa4_units = {m: list(ALL_LOCATIONS) for m in METRICS}

    def main_vec(spec, metric, seed):
        label, typ, config = spec
        if typ == "gnn":
            return load_gnn_region_values(label, config, metric, seed)
        return load_static_region_values(label, metric, "all")

    main_df = run_branch("main_sa4", main_vec, sa4_units,
                         exact=True, boot_seed_base=BOOT_SEED_BASE)

    # ── 2a. Per-seed transparency + Cauchy combination and the mixed-effects model ──
    per_seed_rows = []
    mixed_models = {}
    for comp in COMPARISONS:
        a_spec, b_spec = comp["a"], comp["b"]
        if a_spec[1] != "gnn" and b_spec[1] != "gnn":
            continue
        for metric in METRICS:
            seed_ps = {}
            long_rows = []
            for s in SEEDS:
                a_s = main_vec(a_spec, metric, s)
                b_s = main_vec(b_spec, metric, s)
                d_s = a_s - b_s
                seed_ps[s] = exact_sign_flip_permutation(d_s)["p"]
                for r_i, loc in enumerate(ALL_LOCATIONS):
                    long_rows.append({"region": loc, "seed": s, "diff": d_s[r_i]})
            per_seed_rows.append({
                "comparison_id": comp["cid"], "comparison": comp["label"],
                "metric": metric,
                **{f"perm_p_seed_{s}": seed_ps[s] for s in SEEDS},
                "cauchy_combined_p": cauchy_combination(
                    [seed_ps[s] for s in SEEDS]),
                "cauchy_note": "for reference only, not the primary figure (the primary figure is the permutation p-value after seed averaging)",
            })
            mm = uk031.fit_crossed_mixedlm(pd.DataFrame(long_rows))
            perm_p = float(main_df[(main_df["comparison_id"] == comp["cid"])
                                   & (main_df["metric"] == metric)]["perm_p"].iloc[0])
            mm.update({
                "comparison_id": comp["cid"], "comparison": comp["label"],
                "metric": metric,
                "agree_with_perm_at_0.05": (
                    None if mm["p"] is None
                    else bool((mm["p"] < ALPHA) == (perm_p < ALPHA))),
            })
            mixed_models[f"{comp['cid']}_{metric}"] = mm
    per_seed_df = pd.DataFrame(per_seed_rows)
    per_seed_df.to_csv(EVAL_DIR / "au_per_seed_pvalues.csv",
                       index=False, encoding="utf-8-sig")

    # ── 2b. Moran's I diagnostic (primary queen-contiguity weights + kNN(3) supplement) ──
    print("[Main branch] Moran's I diagnostic (queen + kNN(3))...")
    stage = recompute_station_level()          # station-level re-aggregation (anchoring done internally)
    regions = stage["regions"]
    dissolved = (regions[["loc_key", "geometry"]].dissolve(by="loc_key")
                 .reset_index().rename(columns={"loc_key": "ITL2"}))
    queen_gpkg = EVAL_DIR / "au_sa4_queen.gpkg"
    dissolved.to_file(queen_gpkg, layer="sa4", driver="GPKG")
    W_queen, w_info = build_queen_weights(queen_gpkg, ALL_LOCATIONS)
    if w_info["zero_rows"]:
        print(f"  [warning] queen W has zero-neighbor rows: {w_info['zero_rows']} (kNN weights used as a fallback)")
    # This script computes its own kNN centroids in EPSG:7856 (build_queen_weights
    # returns centroids in EPSG:27700, a value frozen for the UK case and not
    # valid for AU)
    cent = (dissolved.set_index("ITL2").loc[ALL_LOCATIONS]
            .to_crs("EPSG:7856").geometry.centroid)
    centroids_7856 = np.column_stack([cent.x.values, cent.y.values])
    W_knn = build_knn_weights(centroids_7856, k=3)
    blocks = uk031.build_contiguity_blocks(w_info["binary"], ALL_LOCATIONS)
    blocks_named = [[ALL_LOCATIONS[i] for i in blk] for blk in blocks]

    morans_out = {
        "W_meta": {
            "gpkg": "evaluation/au_sa4_queen.gpkg (regions_sa3 dissolved by loc_key)",
            "region_ids": ALL_LOCATIONS,
            "neighbor_counts": w_info["neighbor_counts"],
            "zero_rows": w_info["zero_rows"],
            "knn_k": 3, "knn_centroid_crs": "EPSG:7856",
            "n_perm": MORAN_N_PERM,
            "blocks_greedy_matching": blocks_named,
            "note": "Moran's I is a diagnostic, not a hypothesis test, so no "
                    "multiple-comparison correction is applied; spatial_warning "
                    "is set when the two-sided queen-weights permutation p < 0.05",
        },
        "diagnostics": {},
    }
    moran_cols = {}
    for row_idx, (_, r) in enumerate(main_df.iterrows()):
        comp = next(c for c in COMPARISONS if c["cid"] == r["comparison_id"])
        metric = r["metric"]
        if comp["a"][1] == "gnn":
            a_avg = np.mean([main_vec(comp["a"], metric, s) for s in SEEDS], axis=0)
        else:
            a_avg = main_vec(comp["a"], metric, None)
        if comp["b"][1] == "gnn":
            b_avg = np.mean([main_vec(comp["b"], metric, s) for s in SEEDS], axis=0)
        else:
            b_avg = main_vec(comp["b"], metric, None)
        diffs = a_avg - b_avg
        mi_q = morans_i(diffs, W_queen, n_perm=MORAN_N_PERM,
                        seed=MORAN_SEED_QUEEN_BASE + row_idx)
        mi_k = morans_i(diffs, W_knn, n_perm=MORAN_N_PERM,
                        seed=MORAN_SEED_KNN_BASE + row_idx)
        warning = bool(mi_q["p_perm_two_sided"] < ALPHA)
        block_p = None
        if warning:
            block_sums = np.array([diffs[np.array(blk)].sum() for blk in blocks])
            block_p = exact_sign_flip_permutation(block_sums)["p"]
        key = f"{r['comparison_id']}_{metric}"
        morans_out["diagnostics"][key] = {
            "comparison": r["comparison"], "metric": metric,
            "queen": mi_q, "knn3": mi_k,
            "spatial_warning": warning, "block_perm_p": block_p,
            "n_blocks": len(blocks),
        }
        moran_cols[key] = {"morans_i_queen": mi_q["I"],
                           "morans_p_queen": mi_q["p_perm_two_sided"],
                           "morans_i_knn3": mi_k["I"],
                           "morans_p_knn3": mi_k["p_perm_two_sided"],
                           "spatial_warning": warning,
                           "block_perm_p": block_p}
    for col in ["morans_i_queen", "morans_p_queen", "morans_i_knn3",
                "morans_p_knn3", "spatial_warning", "block_perm_p"]:
        main_df[col] = [moran_cols[f"{r['comparison_id']}_{r['metric']}"][col]
                        for _, r in main_df.iterrows()]
    n_warn = sum(1 for d in morans_out["diagnostics"].values()
                 if d["spatial_warning"])
    morans_out["summary"] = {"n_tests": len(morans_out["diagnostics"]),
                             "n_spatial_warning_queen": n_warn}
    with open(EVAL_DIR / "au_morans_i.json", "w", encoding="utf-8") as f:
        json.dump(morans_out, f, ensure_ascii=False, indent=1)

    n_conv = sum(1 for m in mixed_models.values() if m.get("converged"))
    n_agree = sum(1 for m in mixed_models.values()
                  if m.get("agree_with_perm_at_0.05"))
    with open(EVAL_DIR / "au_mixed_model.json", "w", encoding="utf-8") as f:
        json.dump({
            "design_statement": (
                "Inference unit = SA4 region (n=12). Each (region, seed) "
                "observation comes from 012's 4-fold cross-evaluation: each "
                "region is evaluated exactly once as the test set for a given "
                "seed, and the fold split changes with the seed -- fold "
                "variability is absorbed into the region x seed unit, "
                "matching the argument used in the UK case's equivalent script "
                "(an AU-parameterized version of that script's design_statement)."),
            "model_spec": ("diff ~ 1 + (1|region) + (1|seed), crossed random "
                           "intercepts (statsmodels MixedLM vc_formula, REML); "
                           "only applies to comparisons involving a GNN arm"),
            "summary": {"n_models": len(mixed_models), "n_converged": n_conv,
                        "n_agree_with_perm_at_0.05": n_agree},
            "models": mixed_models,
        }, f, ensure_ascii=False, indent=1)

    # ── 3. PV alternate-protocol branch (excludes 5 stations, reruns all comparisons) ──
    print("\n[PV branch] Excluding 5 day/night-ratio-anomaly stations (exact 2^12 permutation)...")
    gnn_sa4_excl = stage["gnn_sa4_excl"]

    def pv_vec(spec, metric, seed):
        label, typ, config = spec
        if typ == "gnn":
            d = gnn_sa4_excl[(config, label, metric, seed)]
            return np.array([d[loc] for loc in ALL_LOCATIONS])
        return load_static_region_values(label, metric, "excl_pv")

    pv_df = run_branch("pv_excl", pv_vec, sa4_units,
                       exact=True, boot_seed_base=PV_BOOT_SEED_BASE)

    # ── 4. SA3 sensitivity branch (34 units, Monte-Carlo permutation B=10^5) ──
    print(f"\n[SA3 branch] {N_SA3} units (Monte-Carlo B={MC_B})...")
    gnn_sa3 = stage["gnn_sa3"]
    static_sa3 = stage["static_sa3"]
    sa3_all_units = sorted(stage["sa3_n_stations"])
    # corr unit set: >=3 stations and corr finite for every arm involved (deterministic rule, recorded to disk)
    corr_candidates = [u for u in sa3_all_units
                       if stage["sa3_n_stations"][u] >= SA3_CORR_MIN_STATIONS]
    needed_tables = []
    for comp in COMPARISONS:
        for spec in (comp["a"], comp["b"]):
            label, typ, config = spec
            if typ == "gnn":
                needed_tables += [gnn_sa3[(config, label, "corr", s)] for s in SEEDS]
            else:
                needed_tables.append(static_sa3[(label, "corr")])
    corr_units = [u for u in corr_candidates
                  if all(np.isfinite(t.get(u, np.nan)) for t in needed_tables)]
    corr_dropped_nonfinite = sorted(set(corr_candidates) - set(corr_units))
    sa3_units = {"rmse": sa3_all_units, "mae": sa3_all_units, "corr": corr_units}
    print(f"  rmse/mae units = {len(sa3_all_units)}; corr units = {len(corr_units)}"
          f" ({len(corr_candidates)} with >={SA3_CORR_MIN_STATIONS} stations, "
          f"{len(corr_dropped_nonfinite)} dropped as non-finite)")

    def sa3_vec(spec, metric, seed):
        label, typ, config = spec
        units = sa3_units[metric]
        if typ == "gnn":
            d = gnn_sa3[(config, label, metric, seed)]
        else:
            d = static_sa3[(label, metric)]
        return np.array([d[u] for u in units])

    sa3_df = run_branch("sa3_sensitivity", sa3_vec, sa3_units,
                        exact=False, boot_seed_base=SA3_BOOT_SEED_BASE,
                        mc_seed_base=SA3_MC_SEED_BASE)

    # ── 5. Sensitivity-flip determination (generated programmatically) ──
    def _key(df):
        return df.set_index(["comparison_id", "metric"])

    main_idx = _key(main_df)
    for df in (pv_df, sa3_df):
        idx = _key(df)
        sig_flip, dir_flip = [], []
        for (cid, metric), row in idx.iterrows():
            m = main_idx.loc[(cid, metric)]
            sig_flip.append(bool(row["sig"]) != bool(m["sig"]))
            dir_flip.append(np.sign(row["mean_diff"]) != np.sign(m["mean_diff"]))
        df["sig_flipped_vs_main"] = sig_flip
        df["direction_flipped_vs_main"] = dir_flip

    hyp_df = pd.concat([main_df, pv_df, sa3_df], ignore_index=True)
    hyp_df.to_csv(EVAL_DIR / "au_hypothesis_tests.csv",
                  index=False, encoding="utf-8-sig")
    print(f"\nau_hypothesis_tests.csv: {len(hyp_df)} rows"
          f" ({N_COMPARISONS} comparisons x 3 metrics x 3 branches)")

    # ── 6. Directional-consistency table vs the UK results ──
    def _better(delta: float, metric: str) -> bool:
        """Whether Delta = a - b indicates a is better (lower is better for rmse/mae; higher is better for corr)."""
        return bool(delta < 0) if metric in ("rmse", "mae") else bool(delta > 0)

    dir_rows = []
    for comp in COMPARISONS:
        for metric in METRICS:
            m = main_idx.loc[(comp["cid"], metric)]
            ref = uk_refs[(comp["cid"], metric)]
            au_delta = float(m["mean_diff"])
            uk_delta = ref["uk_delta"]
            match = (None if uk_delta is None
                     else bool(np.sign(au_delta) == np.sign(uk_delta)))
            dir_rows.append({
                "comparison_id": comp["cid"], "comparison": comp["label"],
                "hypothesis": comp["hyp"], "metric": metric,
                "au_delta": au_delta, "au_perm_p": float(m["perm_p"]),
                "au_holm_p": float(m["holm_p"]), "au_sig": bool(m["sig"]),
                "au_a_better": _better(au_delta, metric),
                "uk_delta": uk_delta,
                "uk_perm_p": ref["uk_perm_p"], "uk_holm_p": ref["uk_holm_p"],
                "uk_sig": ref["uk_sig"],
                "uk_a_better": (None if uk_delta is None
                                else _better(uk_delta, metric)),
                "direction_match_uk": match,
                "both_sig": (bool(m["sig"]) and bool(ref["uk_sig"])
                             if ref["uk_sig"] is not None else None),
                "uk_source": ref["uk_source"],
                "uk_prior_anchor_delta": ref.get("uk_prior_anchor_delta"),
                "uk_prior_anchor_perm_p": ref.get("uk_prior_anchor_perm_p"),
            })
    dir_df = pd.DataFrame(dir_rows)
    dir_df.to_csv(EVAL_DIR / "au_vs_uk_direction.csv",
                  index=False, encoding="utf-8-sig")
    n_match_by_metric = {
        metric: int(dir_df[(dir_df["metric"] == metric)
                           & (dir_df["direction_match_uk"] == True)  # noqa: E712
                           ].shape[0])
        for metric in METRICS}
    n_with_uk = int(dir_df["direction_match_uk"].notna().sum())
    n_match_total = int((dir_df["direction_match_uk"] == True).sum())  # noqa: E712

    # ── 7. Three-case comparison table (UK/DE/AU side by side, mean RMSE per arm) ──
    fm = pd.read_csv(UK_FULL_MATRIX)
    de = pd.read_csv(DE_STATIC_CSV).set_index("method")
    # DE method-name mapping (the DE static baseline is a port of the UK
    # static baseline; the NP arm uses a sequential-stacking convention, noted below)
    DE_METHOD_MAP = {
        "Uni": "voronoi", "UniN": "voronoi_ntl", "UniP": "voronoi_prox2",
        "UniNP": "voronoi_prox2_ntl",
        "GPM": "voronoi_gpm", "GPMpostN": "voronoi_ntl_gpm",
        "GPMpostP": "voronoi_prox2_gpm", "GPMpostNP": "voronoi_prox2_ntl_gpm",
    }
    three_rows = []
    for label, base, form, sig in au011.ARMS:                    # 14 static arms
        fm_row = fm[(fm["arm"] == label) & (fm["metric"] == "rmse")]
        de_method = DE_METHOD_MAP.get(label)
        de_val = (float(de.loc[de_method, "rmse"])
                  if de_method is not None and de_method in de.index else None)
        au_vec = load_static_region_values(label, "rmse", "all")
        three_rows.append({
            "arm": label, "base": base, "form": form, "signal": sig,
            "uk_rmse_mean_16regions": float(fm_row["mean_16regions"].iloc[0]),
            "de_rmse_boerde": de_val,
            "de_method_name": de_method,
            "de_status": "ok" if de_val is not None else "no corresponding arm (the DE static artifacts have no additive variant)",
            "au_rmse_mean_12regions": float(au_vec.mean()),
            "au_source": "static/au_static_metrics.csv (station_set='all')",
            "note": ("The DE NP arm uses a sequential-stacking convention "
                     "(ntl->prox2), not the single-factor NP convention used here"
                     if sig == "NP" and de_val is not None
                     else ""),
        })
    for arm in GNN_ARMS:                                         # 7 GNN arms
        fm_row = fm[(fm["arm"] == arm) & (fm["metric"] == "rmse")]
        per_region = np.mean(
            [load_gnn_region_values(arm, "baseline", "rmse", s) for s in SEEDS],
            axis=0)
        form = ("none" if arm == "GNN"
                else ("mult" if "post" in arm else "add"))
        signal = arm.replace("GNNpost", "").replace("GNNadd", "") \
            if arm != "GNN" else ""
        three_rows.append({
            "arm": arm, "base": "gnn", "form": form, "signal": signal,
            "uk_rmse_mean_16regions": float(fm_row["mean_16regions"].iloc[0]),
            "de_rmse_boerde": None,
            "de_method_name": None,
            "de_status": "pending (DE training in progress)",
            "au_rmse_mean_12regions": float(per_region.mean()),
            "au_source": "training/seed_*/baseline/kfold_test_rmse.csv (averaged over 3 seeds)",
            "note": "",
        })
    three_df = pd.DataFrame(three_rows)
    three_df.to_csv(EVAL_DIR / "three_case_comparison.csv",
                    index=False, encoding="utf-8-sig")
    print(f"three_case_comparison.csv: {len(three_df)} rows"
          f" (14 static arms + 7 GNN arms; DE covers a single region, Boerde, GNN arms pending)")

    # Write the anchoring report to disk
    with open(EVAL_DIR / "au_reaggregation_anchor.json", "w", encoding="utf-8") as f:
        json.dump(stage["anchor_report"], f, ensure_ascii=False, indent=1)

    # ── 8. evaluation_summary.json (conclusion fields generated programmatically, never hand-written) ──
    def row_of(cid: int, metric: str = "rmse", branch_df: pd.DataFrame = main_df):
        r = branch_df[(branch_df["comparison_id"] == cid)
                      & (branch_df["metric"] == metric)].iloc[0]
        return r

    def comp_readout(cid: int) -> dict:
        r = row_of(cid)
        comp = next(c for c in COMPARISONS if c["cid"] == cid)
        ref = uk_refs[(cid, "rmse")]
        delta = float(r["mean_diff"])
        sig = bool(r["sig"])
        direction = ("harms" if delta > 0 else "improves") if sig else "null"
        match = (None if ref["uk_delta"] is None
                 else bool(np.sign(delta) == np.sign(ref["uk_delta"])))
        return {
            "comparison": comp["label"],
            "delta_rmse": delta, "perm_p": float(r["perm_p"]),
            "holm_p": float(r["holm_p"]), "sig": sig,
            "direction": direction,
            "uk_delta_rmse": ref["uk_delta"], "direction_match_uk": match,
            "conclusion": (
                f"{comp['label']} (RMSE): Delta={delta:+.3f}, Holm p="
                f"{float(r['holm_p']):.3g} ({'significant' if sig else 'not significant'}), "
                f"direction = {'worse' if delta > 0 else 'better'}; UK Delta="
                f"{ref['uk_delta']:+.3f}, direction {'matches' if match else 'does not match'}"
                if ref["uk_delta"] is not None else
                f"{comp['label']} (RMSE): Delta={delta:+.3f}, Holm p="
                f"{float(r['holm_p']):.3g} ({'significant' if sig else 'not significant'})"),
        }

    h1 = {f"cid{c}": comp_readout(c) for c in HYP_CIDS["H-AU1"]}
    h2 = {f"cid{c}": comp_readout(c) for c in HYP_CIDS["H-AU2"]}
    h3 = {f"cid{c}": comp_readout(c) for c in HYP_CIDS["H-AU3"]}
    cross = {f"cid{c}": comp_readout(c) for c in HYP_CIDS["cross_base"]}
    prior = {f"cid{c}": comp_readout(c) for c in HYP_CIDS["prior_channel"]}

    r1 = row_of(1)
    h_au1_replicated = bool(float(r1["mean_diff"]) > 0 and bool(r1["sig"]))
    r5, r6 = row_of(5), row_of(6)
    h_au2_add_beats_mult = bool(float(r5["mean_diff"]) < 0 and bool(r5["sig"]))
    h_au2_add_improves_base = bool(float(r6["mean_diff"]) < 0 and bool(r6["sig"]))
    r7, r8 = row_of(7), row_of(8)
    h_au3_static_gain = bool(float(r7["mean_diff"]) < 0 and bool(r7["sig"])
                             and float(r8["mean_diff"]) < 0 and bool(r8["sig"]))

    def flips(df: pd.DataFrame) -> dict:
        f_sig = df[df["sig_flipped_vs_main"]]
        f_dir = df[df["direction_flipped_vs_main"]]
        return {
            "n_sig_flips": int(len(f_sig)),
            "sig_flips": [f"{r['comparison']}({r['metric']}): main sig="
                          f"{bool(main_idx.loc[(r['comparison_id'], r['metric']), 'sig'])}"
                          f" -> branch sig={bool(r['sig'])}"
                          for _, r in f_sig.iterrows()],
            "n_direction_flips": int(len(f_dir)),
            "direction_flips": [f"{r['comparison']}({r['metric']})"
                                for _, r in f_dir.iterrows()],
        }

    summary = {
        "meta": {
            "generated_by": "013_statistical_evaluation.py",
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "spec": "Statistical evaluation protocol, following the UK case "
                    "study's robust-statistics script (031)",
            "protocol": {
                "inference_unit_main": f"SA4 x {N_SA4}",
                "permutation_main": "exact 2^12 = 4096 (min p = 2/4096 ~= 4.88e-4)",
                "bootstrap": f"region-level paired, outer loop only, B={B_BOOT} (CI is not used to back out p)",
                "holm": f"one family per metric (family = {N_COMPARISONS} comparisons)",
                "sa3_sensitivity": f"SA3 x {N_SA3}, Monte-Carlo B={MC_B}"
                                   " (add-one correction, fixed seed)",
                "pv_sensitivity": "excludes 5 day/night-ratio-anomaly stations (statistics-layer filter only)",
                "seeds": SEEDS, "alpha": ALPHA,
            },
            "deviations": [
                {"id": 1, "item": "SA3 unit count is 34 (originally planned as 35)",
                 "detail": "Corrected once the actual data was confirmed (see "
                           "b1_regions.json; a placeholder station with no data, "
                           "Galston, drops SA3 11502/SA4 115 from the universe), "
                           "same convention as scripts 011/012"},
                {"id": 2, "item": f"SA3 Monte-Carlo permutation uses B={MC_B} (originally planned as 1e4)",
                 "detail": "The design spec requires '>=10^5'; the stricter value is used"},
                {"id": 3, "item": "Moran kNN centroid CRS = EPSG:7856",
                 "detail": "build_queen_weights returns centroids in EPSG:27700, "
                           "a value frozen for the UK case and not valid for AU; "
                           "the queen contiguity structure itself is CRS-independent"},
                {"id": 4, "item": f"SA3 corr sub-analysis covers {len(corr_units)} "
                                  f"SA3 units with >={SA3_CORR_MIN_STATIONS} stations",
                 "detail": "corr is undefined for n=1 and always +/-1 for n=2 (degenerate); rmse/mae cover all 34 units"},
            ],
        },
        "h_au1_multiplicative_on_gnn": {
            "hypothesis": "Multiplicative correction harms the GNN base (AU replication of the UK antagonism finding)",
            "replicated_harm": h_au1_replicated,
            "comparisons": h1,
        },
        "h_au2_additive_repair": {
            "hypothesis": "Additive correction repairs the multiplicative harm / improves the GNN base",
            "additive_beats_multiplicative": h_au2_add_beats_mult,
            "additive_improves_base": h_au2_add_improves_base,
            "comparisons": h2,
        },
        "h_au3_static_base_gain": {
            "hypothesis": "Static bases benefit from multiplicative correction (same direction as the UK results)",
            "both_static_gains_significant": h_au3_static_gain,
            "comparisons": h3,
        },
        "cross_base": cross,
        "prior_channel": prior,
        "uk_direction_consistency": {
            "n_rows_with_uk_delta": n_with_uk,
            "n_direction_match": n_match_total,
            "match_rate": (n_match_total / n_with_uk if n_with_uk else None),
            "per_metric_match": n_match_by_metric,
            "per_metric_total": {m: int(dir_df[(dir_df["metric"] == m)]
                                        ["direction_match_uk"].notna().sum())
                                 for m in METRICS},
        },
        "sa3_sensitivity": {
            "n_units_rmse_mae": len(sa3_all_units),
            "n_units_corr": len(corr_units),
            "corr_unit_rule": f"station count >= {SA3_CORR_MIN_STATIONS} and corr finite for every arm involved",
            "corr_units_dropped_nonfinite": corr_dropped_nonfinite,
            "mc_B": MC_B,
            **flips(sa3_df),
        },
        "pv_sensitivity": {
            "excluded_stations": sorted(au011.PV_SUSPECTED_EXPECTED),
            "n_excluded": len(au011.PV_SUSPECTED_EXPECTED),
            **flips(pv_df),
        },
        "moran_diagnostics": morans_out["summary"],
        "mixedlm": {"n_models": len(mixed_models), "n_converged": n_conv,
                    "n_agree_with_perm_at_0.05": n_agree},
        "reaggregation_anchor": {
            "gnn_max_abs_dev": stage["anchor_report"]["gnn_max_abs_dev"],
            "static_max_abs_dev": stage["anchor_report"]["static_max_abs_dev"],
            "passed": bool(stage["anchor_report"]["gnn_passed"]
                           and stage["anchor_report"]["static_passed"]),
        },
    }
    with open(EVAL_DIR / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print("evaluation_summary.json written")

    # ── 9. Console summary of key readouts ──
    print("\n" + "═" * 70)
    print("Key readouts (RMSE is the primary metric; see evaluation_summary.json for the full results)")
    print("═" * 70)
    for hyp, cids in HYP_CIDS.items():
        print(f"\n[{hyp}]")
        for cid in cids:
            ro = comp_readout(cid)
            print(f"  {ro['conclusion']}")
    print(f"\nH-AU1 multiplicative antagonism (harms the GNN base) replicated in AU: {h_au1_replicated}")
    print(f"H-AU2 additive beats multiplicative: {h_au2_add_beats_mult} | "
          f"additive improves the base: {h_au2_add_improves_base}")
    print(f"H-AU3 both static bases gain (UniNP and GPMpostNP both significant): {h_au3_static_gain}")
    print(f"\nDirection matches UK: {n_match_total}/{n_with_uk}"
          f" (rmse {n_match_by_metric['rmse']}/12, mae {n_match_by_metric['mae']}/12,"
          f" corr {n_match_by_metric['corr']}/12)")
    s_fl = flips(sa3_df)
    p_fl = flips(pv_df)
    print(f"SA3 sensitivity: significance flips {s_fl['n_sig_flips']} / direction flips "
          f"{s_fl['n_direction_flips']} ({len(sa3_all_units)} units, corr sub-analysis "
          f"{len(corr_units)} units)")
    print(f"PV sensitivity: significance flips {p_fl['n_sig_flips']} / direction flips "
          f"{p_fl['n_direction_flips']}")
    print(f"Moran queen spatial warnings: {n_warn}/{len(main_df)}; "
          f"mixed-model converged {n_conv}/{len(mixed_models)}, agrees with permutation test {n_agree}")
    print(f"\nOutput directory: {EVAL_DIR}")
    print(f"Statistical evaluation complete, elapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
