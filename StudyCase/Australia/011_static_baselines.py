# -*- coding: utf-8 -*-
"""Australia case study - static baselines and correction arms (011).

═══ Arm definitions (14 = 2 static bases x {none, multiplicative N/P/NP, additive N/P/NP}) ═══
- Bases (defined identically to the UK case's equivalent script):
  * Uniform -- per-SA3 uniform allocation: demand_i = region demand / number of grid points in the region;
  * GPM -- categorical land-use weighting (`weighter_registry('gpm')` one-hot(argmax lu_*))
    -> score = W @ pcts (the 5 percent columns from region_attributes, matching the
    UK case's compute_demand semantics) -> per-SA3 normalization: demand_i = region demand x score_i / sum(score).
- Region demand = sum of `peak_mw` over the usable stations in each SA3 (this case's primary
  demand convention; regions_sa3.gpkg already on disk).
- Correction always **imports the UK case's shared module shared_correction_utils directly**
  (a read-only cross-case dependency, injected onto sys.path from the repo root):
  factors = compute_factors' ntl_factor / prox_factor / their product (NP);
  multiplicative = apply_standard_multiplicative; additive = apply_additive_correction
  (alpha = base_std/offset_std, ddof=0). The numerical logic is unchanged from the shared module.
- Aggregation = two-stage Voronoi (compute_voronoi_assignment + aggregate_by_assignment,
  EPSG:3857, matching the UK case's convention) -> per-substation -> rmse/mae/corr;
  per-region metrics = cross-station comparison within each SA4 (12 evaluation regions).

═══ Uniform x additive structural degeneracy (an expected mathematical consequence, not a bug) ═══
additive's alpha = base_std/offset_std is supplied by the base's own within-region variance;
for the Uniform base, within-region base_std = 0 -> alpha = 0 -> the correction becomes exactly
equal to the baseline (leaving only floating-point noise from one idempotent renormalization).
This script asserts per-SA3 within-group ptp == 0 (a strict precondition) plus a metric-level
deviation <= 1e-9.

═══ PV sensitivity ═══
Primary table = all 143 usable stations; secondary table = excludes 5 stations flagged by an
anomalous day/night ratio (pv_level_fy2009 == 'suspected': Berowra / Jannali / Lake Munmorah /
Menai / Singleton North). The exclusion is a **pure statistical-layer station-set filter**
(rows for these 5 stations are dropped at evaluation time) -- allocation, factors, and the
Voronoi assignment are never rerun.

═══ Deviations from the UK case's conventions (all written to results_summary.json.deviations) ═══
1. Region universe = 34 SA3 units / 12 SA4 units (empirically corrected after data validation;
   see b1_regions.json; an earlier assumption had used 35/13).
2. The proximity score does not call scu.compute_prox_scores (its TARGET_CRS=EPSG:27700 is fixed
   to the UK case), and instead uses the precomputed `{loc}_proximity.npz` (same formula:
   gamma=2.0, clamp=0.01 km, only the CRS changes to EPSG:7856) -- compute_factors itself does
   not recompute proximity, so the factor formula is unchanged.
3. Column-name alignment (a pure rename, no change in numeric semantics): AU's SA3 code is
   aliased to an 'ITL3' column, and demand_peak_mw is aliased to 'Demand (MVA)', so that the
   shared module's groupby/lookup logic works unmodified.
4. Demand convention = peak_mw (MW, active power) vs. the UK case's MVA (apparent power) --
   this is a deliberate, self-consistent choice within this case; cross-case comparisons should
   focus on directional agreement, not absolute magnitudes.

This step has no randomness (no RNG calls); runs entirely on CPU.

Outputs:
    data/processed/static/au_static_metrics.csv -- 14 arms x 3 metrics x 2 station sets
        (primary table station_set='all' + PV secondary table 'excl_pv') x 12 regions + mean/std
    data/processed/static/results_summary.json  -- per-arm 12-region means +
        conservation/degeneracy/direction checks (generated programmatically, not hand-written)
        + deviation log

Usage:
    python 011_static_baselines.py
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Force UTF-8 console output on Windows (consistent with the other pipeline scripts)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# sys.path injection (repo root + UK experiment directory)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
_p = SCRIPT_DIR
while not (_p / "SpatialAllocation").exists():
    _p = _p.parent
PROJECT_ROOT = _p
UK_EXP_DIR = PROJECT_ROOT / "StudyCase" / "British_weighter_experiments"
for _extra in (str(UK_EXP_DIR), str(PROJECT_ROOT)):
    if _extra not in sys.path:
        sys.path.insert(0, _extra)

import geopandas as gpd  # noqa: E402
import shared_correction_utils as scu  # noqa: E402  # shared post-hoc correction utilities (read-only import from the UK case study)
from SpatialAllocation.Weighter import weighter_registry  # noqa: E402

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
PROCESSED = SCRIPT_DIR / "data" / "processed"
ASSEMBLED = PROCESSED / "features" / "assembled"
EXTRACTED = PROCESSED / "features" / "extracted"
STATIC_DIR = PROCESSED / "static"
STEP_TABLE = PROCESSED / "grid" / "grid_step_size_table.csv"
REGIONS_SA3 = PROCESSED / "regions_sa3.gpkg"
STATION_TABLE = PROCESSED / "station_table_fy2009.csv"
B2_REGISTRY = SCRIPT_DIR / "docs" / "b1b2" / "b2_features.json"

METRICS_CSV = STATIC_DIR / "au_static_metrics.csv"
SUMMARY_JSON = STATIC_DIR / "results_summary.json"

USABLE_STATUSES = {"matched", "matched_osm"}
N_USABLE = 143                      # usable count for FY2009, gated by upstream matching
EXPECTED_N_SA3 = 34                 # empirically corrected count (an earlier assumption used 35)
EXPECTED_N_SA4 = 12                 # empirically corrected count (an earlier assumption used 13)
DEMAND_COL = "peak_mw"              # this case's primary demand convention (annual peak, MW)

METRICS = ["rmse", "mae", "corr"]
SIGNALS = ["N", "P", "NP"]
STATION_SETS = ["all", "excl_pv"]

# 5 stations flagged by an anomalous day/night ratio (station_table's pv_level_fy2009 == 'suspected')
PV_SUSPECTED_EXPECTED = {"Berowra", "Jannali", "Lake Munmorah", "Menai",
                         "Singleton North"}

# Land-use columns and region percentage columns (same names/order as the UK case, one-to-one correspondence)
LU_COLS = ["lu_residential_prop", "lu_commercial_prop", "lu_industrial_prop",
           "lu_agricultural_prop", "lu_others_prop"]
PCT_COLS = ["residential_percent", "commercial_percent", "industrial_percent",
            "agricultural_percent", "others_percent"]

CONS_TOL_MW = 1e-6                  # per-SA3 conservation assertion tolerance (absolute, MW)
DEG_TOL = 1e-9                      # floating-point noise ceiling for the UniAdd == Uni idempotent renormalization

# Arm definitions (label, base, form, signal) -- labels follow the same naming as the UK case (for cross-case comparison)
ARMS = [
    ("Uni",       "uniform", "none", ""),
    ("UniN",      "uniform", "mult", "N"),
    ("UniP",      "uniform", "mult", "P"),
    ("UniNP",     "uniform", "mult", "NP"),
    ("UniAddN",   "uniform", "add",  "N"),
    ("UniAddP",   "uniform", "add",  "P"),
    ("UniAddNP",  "uniform", "add",  "NP"),
    ("GPM",       "gpm",     "none", ""),
    ("GPMpostN",  "gpm",     "mult", "N"),
    ("GPMpostP",  "gpm",     "mult", "P"),
    ("GPMpostNP", "gpm",     "mult", "NP"),
    ("GPMaddN",   "gpm",     "add",  "N"),
    ("GPMaddP",   "gpm",     "add",  "P"),
    ("GPMaddNP",  "gpm",     "add",  "NP"),
]
DEGENERATE_ARMS = ["UniAddN", "UniAddP", "UniAddNP"]   # structurally degenerate arms (see docstring above)


# ---------------------------------------------------------------------------
# Static bases (defined identically to the UK case's compute_uniform_base / compute_gpm_base)
# ---------------------------------------------------------------------------
def compute_uniform_base(grid_gdf: gpd.GeoDataFrame,
                         region_sub: gpd.GeoDataFrame) -> np.ndarray:
    """Uniform base: per-SA3 uniform allocation = total_demand / len(group)."""
    region_info = region_sub.set_index("ITL3")
    base = np.zeros(len(grid_gdf))
    for itl3, group in grid_gdf.groupby("ITL3"):
        if itl3 not in region_info.index:
            continue
        total_demand = region_info.loc[itl3, "Demand (MVA)"]
        base[group.index] = total_demand / len(group)
    return base


def compute_gpm_base(grid_gdf: gpd.GeoDataFrame,
                     region_sub: gpd.GeoDataFrame,
                     subs_sub: gpd.GeoDataFrame) -> tuple[np.ndarray, int]:
    """GPM base: reconstructs the UK case's 'landuse_demand' logic (categorical GPM -> per-SA3 normalization).

    score = W @ pcts (W = one-hot(argmax lu_*), pcts = the region's 5 percent
    columns, matching the UK case's compute_demand semantics). Returns
    (base array, number of groups that triggered the fallback).
    """
    gpm = weighter_registry.create(
        "gpm", config={"mode": "categorical", "proportion_columns": LU_COLS})
    gpm_res = gpm.compute(grid_gdf, target_gdf=subs_sub)
    W = gpm_res.weights

    region_info = region_sub.set_index("ITL3")
    base = np.zeros(len(grid_gdf))
    n_fallback = 0
    for itl3, group in grid_gdf.groupby("ITL3"):
        if itl3 not in region_info.index:
            continue
        total_demand = region_info.loc[itl3, "Demand (MVA)"]
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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_regions() -> gpd.GeoDataFrame:
    """Load the 34 source SA3 units (with demand + 5 percent columns), and inject the shared module's alias columns."""
    sa3 = gpd.read_file(REGIONS_SA3, layer="regions_sa3")
    assert len(sa3) == EXPECTED_N_SA3, f"SA3 count {len(sa3)} != {EXPECTED_N_SA3}"
    assert sa3["SA4"].nunique() == EXPECTED_N_SA4
    assert sa3[PCT_COLS + ["demand_peak_mw"]].notna().all().all()
    assert (sa3["demand_peak_mw"] > 0).all(), "Found non-positive region demand"
    # Column-name alias only (no change in numeric semantics): SA3->ITL3, demand_peak_mw->'Demand (MVA)',
    # so that shared_correction_utils' groupby('ITL3') / region_info lookups work unmodified
    sa3["ITL3"] = sa3["SA3"].astype(str)
    sa3["Demand (MVA)"] = sa3["demand_peak_mw"]
    return sa3


def load_stations(regions: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """FY2009 usable stations (143) -> sjoin to inherit SA3/SA4/loc_key membership."""
    st = pd.read_csv(STATION_TABLE, encoding="utf-8-sig")
    usable = st[st["status"].isin(USABLE_STATUSES)].copy()
    assert len(usable) == N_USABLE, f"Usable station count {len(usable)} != {N_USABLE}"
    pv = set(usable.loc[usable["pv_level_fy2009"] == "suspected", "station"])
    assert pv == PV_SUSPECTED_EXPECTED, \
        f"PV-suspected station set {sorted(pv)} != expected {sorted(PV_SUSPECTED_EXPECTED)}"
    pts = gpd.GeoDataFrame(
        usable,
        geometry=gpd.points_from_xy(usable["lon_wgs84"], usable["lat_wgs84"]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(pts, regions[["ITL3", "SA4", "loc_key", "geometry"]],
                       how="left", predicate="within")
    assert joined["ITL3"].notna().all(), "Found a usable station that does not fall within any SA3 footprint"
    assert len(joined) == N_USABLE, "sjoin produced duplicate rows (station on a boundary?)"
    return joined.drop(columns="index_right")


def load_location_context(loc: str, regions: gpd.GeoDataFrame,
                          stations: gpd.GeoDataFrame, b2: dict,
                          assignment_cache: dict) -> dict:
    """Load all shared quantities for one SA4 evaluation region (computed once per region)."""
    # Assemble the grid (RangeIndex discipline: shared_correction_utils uses group.index as a numpy positional index)
    with open(ASSEMBLED / f"{loc}_grid_points.pickle", "rb") as fh:
        grid_gdf, step_size_m = pickle.load(fh)
    grid_gdf = grid_gdf.reset_index(drop=True)
    assert (grid_gdf.index == np.arange(len(grid_gdf))).all()
    grid_gdf["ITL3"] = grid_gdf["SA3"].astype(str)   # column-name alias (deviation 3 in the module docstring)

    region_sub = regions[regions["loc_key"] == loc].reset_index(drop=True)
    assert len(region_sub) >= 1
    # Hard assertion: the grid's SA3 set and region_sub's SA3 set must cover each other exactly,
    # to prevent the shared module's "if itl3 not in region_info.index: continue" from silently zeroing rows
    assert set(grid_gdf["ITL3"].unique()) == set(region_sub["ITL3"]), \
        f"{loc}: grid SA3 set and region-table SA3 set do not cover each other"

    subs_sub = stations[stations["loc_key"] == loc].reset_index(drop=True)
    n_b2 = b2["regions"][loc]["proximity_stats"]["n_stations"]
    assert len(subs_sub) == n_b2, \
        f"{loc}: station count {len(subs_sub)} != registered count {n_b2} -- check for upstream data changes"

    # NTL (precomputed on disk; median of DMSP F162008+F162009)
    ntl_npz = np.load(EXTRACTED / f"{loc}_ntl.npz", allow_pickle=True)
    ntl_values = ntl_npz["data"][:, 0]
    assert ntl_values.shape == (len(grid_gdf),)
    assert np.isfinite(ntl_values).all() and (ntl_values >= 0).all()

    # proximity (deviation 2: uses the precomputed npz, EPSG:7856;
    # does not call scu.compute_prox_scores, since its TARGET_CRS=EPSG:27700 is fixed to the UK case)
    prox_npz = np.load(EXTRACTED / f"{loc}_proximity.npz", allow_pickle=True)
    prox_scores = prox_npz["data"][:, 0]
    assert prox_scores.shape == (len(grid_gdf),)
    assert np.isfinite(prox_scores).all() and (prox_scores > 0).all()

    # Factors (unchanged from the UK shared module: RCI mask = lu_r+c+i > 0.5, full log1p/median/epsilon chain)
    ntl_factor, prox_factor = scu.compute_factors(grid_gdf, ntl_values, prox_scores)
    factors = {"N": ntl_factor, "P": prox_factor, "NP": ntl_factor * prox_factor}

    # Voronoi assignment (EPSG:3857, matching the UK case's convention; computed once per region)
    assignment = scu.compute_voronoi_assignment(
        grid_gdf, subs_sub, cache=assignment_cache, cache_key=loc)

    # per-SA3 positional index (used by conservation assertions)
    sa3_groups = {code: group.index.to_numpy()
                  for code, group in grid_gdf.groupby("ITL3")}
    region_demand = region_sub.set_index("ITL3")["Demand (MVA)"].to_dict()

    return {"loc": loc, "grid_gdf": grid_gdf, "region_sub": region_sub,
            "subs_sub": subs_sub, "factors": factors, "assignment": assignment,
            "sa3_groups": sa3_groups, "region_demand": region_demand,
            "pv_mask": (subs_sub["pv_level_fy2009"] == "suspected").to_numpy()}


# ---------------------------------------------------------------------------
# Conservation + evaluation
# ---------------------------------------------------------------------------
def assert_sa3_conservation(ctx: dict, arm_label: str,
                            demand_arr: np.ndarray) -> float:
    """Per-SA3 conservation assertion: the sum of allocated demand within a region equals the region's demand. Returns the max deviation."""
    max_dev = 0.0
    for code, idx in ctx["sa3_groups"].items():
        dev = abs(float(demand_arr[idx].sum()) - ctx["region_demand"][code])
        max_dev = max(max_dev, dev)
    assert max_dev <= CONS_TOL_MW, \
        f"{ctx['loc']}/{arm_label}: per-SA3 conservation violated (max deviation {max_dev:.3e} MW)"
    return max_dev


def eval_both_sets(ctx: dict, demand_arr: np.ndarray) -> tuple[dict, float]:
    """Demand array -> Voronoi aggregation (cached assignment) -> three metrics for both the primary and secondary station sets.

    The secondary table excludes the PV-suspected stations at the evaluation layer only
    (pure post-processing: allocation is not rerun, rows for these 5 stations are simply
    dropped from the per-substation result table).
    Returns ({station_set: metrics}, total station-level conservation deviation).
    """
    subs_result = scu.aggregate_by_assignment(
        ctx["subs_sub"], ctx["assignment"], demand_arr)
    station_dev = abs(float(subs_result["allocated_demand"].sum())
                      - float(demand_arr.sum()))
    assert station_dev <= CONS_TOL_MW, \
        f"{ctx['loc']}: Voronoi aggregation violated total conservation (deviation {station_dev:.3e} MW)"
    out = {"all": scu.evaluate_allocation(subs_result, actual_col=DEMAND_COL)}
    sub = subs_result[~ctx["pv_mask"]]
    out["excl_pv"] = scu.evaluate_allocation(sub, actual_col=DEMAND_COL)
    return out, station_dev


def seven_variants(ctx: dict, base_demand: np.ndarray) -> tuple[dict, float]:
    """The 7 variants of a single base (none + multiplicative N/P/NP + additive N/P/NP).

    Returns ({(form, signal): {station_set: metrics}}, max per-SA3 conservation deviation);
    each variant's demand array passes the per-SA3 conservation assertion before aggregation
    (both multiplicative and additive corrections renormalize per region, so conservation
    should hold).
    """
    grid_gdf, region_sub = ctx["grid_gdf"], ctx["region_sub"]
    out = {}
    max_dev = 0.0
    variants = {("none", ""): np.asarray(base_demand, dtype=float)}
    for sig in SIGNALS:
        variants[("mult", sig)] = scu.apply_standard_multiplicative(
            base_demand, ctx["factors"][sig], grid_gdf, region_sub)
        variants[("add", sig)] = scu.apply_additive_correction(
            base_demand, ctx["factors"][sig], grid_gdf, region_sub)
    for key, arr in variants.items():
        assert np.isfinite(arr).all() and (arr >= 0).all(), \
            f"{ctx['loc']}/{key}: demand array contains non-finite or negative values"
        max_dev = max(max_dev, assert_sa3_conservation(ctx, str(key), arr))
        out[key], _ = eval_both_sets(ctx, arr)
    return out, max_dev


# ---------------------------------------------------------------------------
# Direction checks (generated programmatically from the numeric results, not hand-written)
# ---------------------------------------------------------------------------
def _better(a: float, b: float, metric: str) -> bool:
    """Whether a is better than b (lower is better for rmse/mae; higher is better for corr)."""
    return bool(a < b) if metric in ("rmse", "mae") else bool(a > b)


def build_direction_checks(arm_mean: dict) -> dict:
    """Generate boolean direction fields from each arm's 12-region mean.

    Core question: on the GPM base, does "multiplicative vs. additive" reproduce the
    static-base multiplicative-correction dominance seen in the UK case (checked per
    signal individually, plus an AND across all signals, with rmse as the primary metric)?
    The Uniform base's additive arms are structurally degenerate (== Uni) and are excluded
    from the dominance check.
    """
    checks = {"note": "Generated programmatically (not hand-written); based on the 12-region mean of the primary table (station_set='all')",
              "per_metric": {}}
    for m in METRICS:
        per_signal = {}
        for sig in SIGNALS:
            gpm_mult = arm_mean[f"GPMpost{sig}"][m]
            gpm_add = arm_mean[f"GPMadd{sig}"][m]
            per_signal[sig] = {
                "gpm_mult_mean": gpm_mult,
                "gpm_add_mean": gpm_add,
                "gpm_mult_beats_additive": _better(gpm_mult, gpm_add, m),
                "gpm_mult_improves_base": _better(gpm_mult, arm_mean["GPM"][m], m),
                "gpm_add_improves_base": _better(gpm_add, arm_mean["GPM"][m], m),
                "uniform_mult_improves_base": _better(
                    arm_mean[f"Uni{sig}"][m], arm_mean["Uni"][m], m),
            }
        checks["per_metric"][m] = {
            "per_signal": per_signal,
            "static_mult_dominance_all_signals": all(
                per_signal[s]["gpm_mult_beats_additive"] for s in SIGNALS),
        }
    checks["uk_s6_static_mult_dominance_replicated_rmse"] = \
        checks["per_metric"]["rmse"]["static_mult_dominance_all_signals"]
    return checks


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------
def main() -> None:
    t0 = time.time()
    print("=" * 70)
    print("Australia case study - static baselines and correction arms (011) -- 14 arms x 3 metrics x 2 station sets x 12 regions")
    print("=" * 70)
    STATIC_DIR.mkdir(parents=True, exist_ok=True)

    step_table = pd.read_csv(STEP_TABLE, encoding="utf-8-sig")
    all_locations = step_table["loc_key"].tolist()
    assert len(all_locations) == EXPECTED_N_SA4

    b2 = json.loads(B2_REGISTRY.read_text(encoding="utf-8"))
    regions = load_regions()
    stations = load_stations(regions)

    assignment_cache: dict = {}
    gpm_fallback_groups = 0
    cons_max_dev = 0.0            # global max per-SA3 conservation deviation
    n_cons_checks = 0

    # values[label][sset][metric][loc]
    values = {label: {ss: {m: {} for m in METRICS} for ss in STATION_SETS}
              for label, *_ in ARMS}

    # Uniform x additive degeneracy check quantities
    uni_base_group_ptp_max = 0.0
    uni_base_group_std_max = 0.0
    uni_add_demand_dev = {sig: 0.0 for sig in SIGNALS}

    for loc in all_locations:
        print(f"\n=== {loc} ===")
        ctx = load_location_context(loc, regions, stations, b2, assignment_cache)
        grid_gdf, region_sub = ctx["grid_gdf"], ctx["region_sub"]

        # Static bases
        uni_base = compute_uniform_base(grid_gdf, region_sub)
        gpm_base, n_fb = compute_gpm_base(grid_gdf, region_sub, ctx["subs_sub"])
        gpm_fallback_groups += n_fb

        # Degeneracy precondition: within-SA3-group values are exactly equal for the Uniform base (ptp == 0)
        for code, idx in ctx["sa3_groups"].items():
            g_vals = uni_base[idx]
            uni_base_group_ptp_max = max(
                uni_base_group_ptp_max, float(g_vals.max() - g_vals.min()))
            uni_base_group_std_max = max(
                uni_base_group_std_max, float(g_vals.std()))

        # Grid-point-level demand deviation under degeneracy (additive output vs. the Uniform base itself)
        for sig in SIGNALS:
            add_demand = scu.apply_additive_correction(
                uni_base, ctx["factors"][sig], grid_gdf, region_sub)
            uni_add_demand_dev[sig] = max(
                uni_add_demand_dev[sig],
                float(np.abs(add_demand - uni_base).max()))

        # 14 arms (seven_variants asserts per-SA3 conservation for each variant; aggregate the max deviation here)
        for base_name, base_arr in (("uniform", uni_base), ("gpm", gpm_base)):
            variants, dev = seven_variants(ctx, base_arr)
            cons_max_dev = max(cons_max_dev, dev)
            n_cons_checks += 7 * len(ctx["sa3_groups"])
            for label, base, form, sig in ARMS:
                if base != base_name:
                    continue
                for ss in STATION_SETS:
                    for m in METRICS:
                        values[label][ss][m][loc] = variants[(form, sig)][ss][m]

        print(f"  Uni={values['Uni']['all']['rmse'][loc]:.3f}  "
              f"GPM={values['GPM']['all']['rmse'][loc]:.3f}  "
              f"GPMpostNP={values['GPMpostNP']['all']['rmse'][loc]:.3f}  "
              f"GPMaddNP={values['GPMaddNP']['all']['rmse'][loc]:.3f}  (RMSE, all stations)")

    def vec(label: str, ss: str, metric: str) -> np.ndarray:
        return np.array([values[label][ss][metric][loc] for loc in all_locations])

    # ── Degeneracy verification: UniAdd* == Uni (metric-level <= 1e-9, both station sets) ──
    assert uni_base_group_ptp_max == 0.0, \
        "Uniform base within-group values are not exactly equal (ptp != 0) -- the degeneracy precondition is violated"
    deg_max = 0.0
    deg_detail = {}
    for label in DEGENERATE_ARMS:
        for ss in STATION_SETS:
            for m in METRICS:
                dev = float(np.abs(vec(label, ss, m) - vec("Uni", ss, m)).max())
                deg_detail[f"{label}_{ss}_{m}"] = dev
                deg_max = max(deg_max, dev)
    if deg_max > DEG_TOL:
        raise RuntimeError(f"UniAdd degeneracy deviation {deg_max:.3e} exceeds tolerance {DEG_TOL} -- "
                           f"inconsistent with the expected structural result, please investigate")
    print(f"\nDegeneracy verification: UniAdd* == Uni, max metric-level deviation {deg_max:.3e}"
          f" (max grid-point-level demand deviation {max(uni_add_demand_dev.values()):.3e})")
    print(f"per-SA3 conservation: all assertions passed, max deviation {cons_max_dev:.3e} MW"
          f" (tolerance {CONS_TOL_MW})")

    # ── Write the primary/secondary tables to disk ──
    matrix_rows = []
    for label, base, form, sig in ARMS:
        for ss in STATION_SETS:
            for m in METRICS:
                v = vec(label, ss, m)
                matrix_rows.append({
                    "arm": label, "base": base, "form": form, "signal": sig,
                    "metric": m, "station_set": ss,
                    "structural_degeneration": label in DEGENERATE_ARMS,
                    "n_regions": len(all_locations),
                    **{loc: values[label][ss][m][loc] for loc in all_locations},
                    "mean_12regions": float(v.mean()),
                    "std_ddof0_12regions": float(v.std()),
                })
    matrix = pd.DataFrame(matrix_rows)
    matrix.to_csv(METRICS_CSV, index=False, encoding="utf-8-sig")
    print(f"\nau_static_metrics.csv: {len(matrix)} rows"
          f" (14 arms x 3 metrics x 2 station sets) -> {METRICS_CSV}")

    # ── results_summary.json (per-arm 12-region means + rule-based checks + deviation log) ──
    arm_means = {}
    for ss in STATION_SETS:
        arm_means[ss] = {
            label: {m: {"mean": float(vec(label, ss, m).mean()),
                        "std_ddof0": float(vec(label, ss, m).std())}
                    for m in METRICS}
            for label, *_ in ARMS}
    arm_mean_flat = {label: {m: arm_means["all"][label][m]["mean"]
                             for m in METRICS} for label, *_ in ARMS}
    direction = build_direction_checks(arm_mean_flat)

    pv_affected = sorted(
        stations.loc[stations["pv_level_fy2009"] == "suspected", "loc_key"].unique())

    summary = {
        "meta": {
            "generated_by": "011_static_baselines.py",
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "spec": "Static baselines and correction arms for the Australia case study",
            "demand_col": DEMAND_COL,
            "region_order": all_locations,
            "n_regions": len(all_locations),
            "n_arms": len(ARMS),
            "n_usable_stations": N_USABLE,
            "gpm_fallback_groups": gpm_fallback_groups,
            "correction_module": "StudyCase/British_weighter_experiments/"
                                 "shared_correction_utils.py (read-only import from the UK case study)",
            "voronoi_crs": "EPSG:3857 (VoronoiAllocator's default working_crs, "
                           "matching the UK case's convention)",
            "randomness": "This step has no randomness (no RNG calls)",
        },
        "deviations": [
            {"id": 1, "item": "Region universe = 34 SA3 units / 12 SA4 units",
             "detail": "Empirically corrected after data validation (see b1_regions.json.deviation_from_b0; "
                       "an earlier assumption had used 35/13 -- a placeholder empty station at Galston "
                       "caused SA3 unit 11502 / SA4 unit 115 to be excluded)"},
            {"id": 2, "item": "Proximity score source",
             "detail": "Does not call scu.compute_prox_scores (its TARGET_CRS=EPSG:27700 "
                       "is fixed to the UK case), and instead uses the precomputed {loc}_proximity.npz -- "
                       "same formula (gamma=2.0, clamp=0.01 km), only the CRS changes to "
                       "EPSG:7856; compute_factors itself does not recompute proximity, so the factor formula is unchanged"},
            {"id": 3, "item": "Column-name alignment (a pure rename, no change in numeric semantics)",
             "detail": "AU's SA3 code -> 'ITL3' alias column, demand_peak_mw -> "
                       "'Demand (MVA)' alias column, so the shared module's groupby/lookup logic works unmodified"},
            {"id": 4, "item": "Demand convention peak_mw (MW, active power) vs. the UK case's MVA (apparent power)",
             "detail": "A deliberate, self-consistent choice within this case; cross-case comparisons should focus on directional agreement, not absolute magnitudes"},
            {"id": 5, "item": "The PV secondary table is a pure statistical-layer station-set filter",
             "detail": "Allocation, factors, and the Voronoi assignment are not rerun; "
                       "rows for the 5 suspected stations are simply dropped at evaluation time"},
        ],
        "conservation": {
            "per_sa3_max_abs_dev_mw": cons_max_dev,
            "n_checks": n_cons_checks,
            "tol_mw": CONS_TOL_MW,
            "passed": bool(cons_max_dev <= CONS_TOL_MW),
            "note": "Each (region, arm) demand array passes both the per-SA3 conservation assertion "
                    "and the station-level total conservation assertion; the global max deviation is recorded here",
        },
        "uniform_additive_degeneration": {
            "structural_degeneration": True,
            "affected_arms": DEGENERATE_ARMS,
            "reason": "additive's alpha = base_std/offset_std (ddof=0) is supplied by the base's own "
                      "variance; for the Uniform base, within-SA3-group base_std = 0 -> alpha = 0 -> "
                      "the correction becomes exactly equal to the baseline (leaving only floating-point "
                      "noise from one idempotent renormalization) -- an expected mathematical consequence, reproduced here in the AU case",
            "uniform_base_within_group_ptp_max": uni_base_group_ptp_max,
            "uniform_base_within_group_std_max": uni_base_group_std_max,
            "max_agent_demand_abs_dev_by_signal": {
                sig: uni_add_demand_dev[sig] for sig in SIGNALS},
            "metric_level_max_abs_dev": deg_detail,
            "metric_level_max_abs_dev_overall": deg_max,
            "tol": DEG_TOL,
            "all_within_tol": bool(deg_max <= DEG_TOL),
        },
        "pv_sensitivity": {
            "excluded_stations": sorted(PV_SUSPECTED_EXPECTED),
            "n_excluded": len(PV_SUSPECTED_EXPECTED),
            "affected_loc_keys": pv_affected,
            "note": "Secondary table station_set='excl_pv'; unaffected regions have identical values in both tables",
        },
        "arm_means_12regions": arm_means,
        "direction_checks": direction,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                            encoding="utf-8")
    print(f"results_summary.json -> {SUMMARY_JSON}")

    # ── Console summary ──
    print("\n──── Per-arm 12-region mean RMSE (primary table, all stations | PV secondary table) ────")
    for label, base, form, sig in ARMS:
        deg = "  [structurally degenerate, == Uni]" if label in DEGENERATE_ARMS else ""
        print(f"  {label:10s} {arm_means['all'][label]['rmse']['mean']:7.3f} +/- "
              f"{arm_means['all'][label]['rmse']['std_ddof0']:5.3f} | "
              f"{arm_means['excl_pv'][label]['rmse']['mean']:7.3f}{deg}")
    rep = direction["uk_s6_static_mult_dominance_replicated_rmse"]
    print(f"\nStatic-base multiplicative-correction dominance (GPM base, rmse, all signals) reproduced: {rep}")
    for sig in SIGNALS:
        d = direction["per_metric"]["rmse"]["per_signal"][sig]
        print(f"  {sig:3s}: GPMpost={d['gpm_mult_mean']:.3f} vs "
              f"GPMadd={d['gpm_add_mean']:.3f} -> multiplicative dominates = "
              f"{d['gpm_mult_beats_additive']}")
    print(f"\nDone, elapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
