"""Step B registered unit ``paper2_fixed_load_connection`` (revision plan section 5).

Fixed-load connection recomputation for the margin-criterion revision. It reads
only the frozen backup (``results/_backup/inputs``) and the upstream release code
archive (``code/upstream_release_69c4008732ed.zip``, commit 69c4008), imports the
upstream numerical primitives from that archive, and writes one run root
``results/_backup/recompute/<run_id>/``. It trains nothing and writes nothing
upstream.

Reused upstream primitives (unchanged):
  sglib.experiment.connection      stride_candidates, neighbourhood_members/sum, reference_field
  sglib.experiment.control_fields  build_control_arms (PERM/SMOOTH, RandomState(42); PERM-R/3 seed+R)
  sglib.experiment.connection_observations  rank_agreement, compiled_members
  sglib.experiment.conditional_bounds       loss_bounds (rectangular exchange bound)
  sglib.dataoverview...grid_bundle          load_grid_bundle

Paper-specific layer (this file): fixed X in {100,300,500} MW, monetised British
cost C_y = c*max(0, G_y+X-F_y) + T_y with T9 (T25 only for R=10 km, X=300 MW),
the D2 Ref screen, the leave-one-region-out budget per method/seed/radius, and the
four output families aligned with ``legacy/frozen_lu5``.

Every region is cross-checked bit-for-bit against the r2 release observation
(``surfaces.npz``: candidate rows, G, F, A_ref, Ghat of the five candidate fields,
the six PERM/SMOOTH fields) and against the r2 Ref pre-audit score.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import time
import zipfile

import numpy as np
import pandas as pd

# ----------------------------------------------------------------- registration

RUN_SCHEMA = "paper2_fixed_load_connection_v1"
UPSTREAM_ZIP = "code/upstream_release_69c4008732ed.zip"
UPSTREAM_COMMIT = "69c4008732ed83dfb876020504baacbfa526d1aa"
RELEASE_ID = "refactor_20260922_r2"

COUNTRIES = {
    "uk": {"directory": "1_UK", "working_crs": "EPSG:27700", "demand_unit": "MVA", "capacity_unit": "MVA",
           "demand_column": "Demand (MVA)", "capacity_column": "Firm Capacity (MVA)", "source_key": "ITL3",
           "region_demand_column": "Demand (MVA)", "regions_path": "data/datasets/2_derived/uk/bplus/regions.gpkg",
           "stations_path": "data/datasets/2_derived/uk/bplus/stations.gpkg",
           "cost_unit": "GBP", "c_per_mva": 430_000.0, "tariff": "T9"},
    "au": {"directory": "2_AU", "working_crs": "EPSG:7856", "demand_unit": "MW", "capacity_unit": "MVA",
           "demand_column": "G_fy2024_mw", "capacity_column": "F_mva", "source_key": "SA3",
           "region_demand_column": "demand_peak_mw", "regions_path": "data/datasets/2_derived/au/bplus/regions_sa3.gpkg",
           "stations_path": "data/datasets/2_derived/au/bplus/stations.gpkg",
           "cost_unit": "MVA_equivalent_PF1", "c_per_mva": 1.0, "tariff": None},
}
PROTOCOL = {
    "loads_mw": [100.0, 300.0, 500.0], "main_load_mw": 300.0, "main_radius_km": 10.0,
    "bound_radii_km": [10.0, 20.0], "scale_radii_km": [1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0],
    "candidate_count": 2000, "shortlist_k": 20, "tie_rule": "numpy argsort kind=stable on raw grid-row order",
    "neighbourhood": "closed ball d(i,y) <= R in the national projected working CRS; members restricted to the region's own grid/stations",
    "etas": [0.5, 0.8, 0.9, 1.0], "quantile_method": "linear (numpy)",
    "budget_domain": "all regions of the country with a valid 2000-candidate pool; not Ref-screened; per method/seed/radius; LOO over regions; budget independent of X",
    "ref_threshold": 0.25, "ref_radii_km": [10.0, 20.0],
    "ref_rule": "q_r = max_R mean_y|A_ref,R - G_R| / median_y G_R <= 0.25 with every denominator defined (median G > 0)",
    "annuity": {"rate": 0.035, "years": 20, "formula": "(1-(1+r)^-n)/r at full precision"},
    "uk_cost": "C_y = 430000 GBP/MVA * max(0, G_y + X - F_y) + X_MW*1000*t_y*AF",
    "au_cost": "C_y = 1 * max(0, G_y + X - F_y); T=0; PF=1 MVA-equivalent, not a monetary cost",
    "t25_sensitivity": {"country": "uk", "radius_km": 10.0, "load_mw": 300.0, "component": "T25 2026/27 Final Year Round (unfloored locational)"},
    "panel": ["Ref", "PERM-R/3", "PERM-R", "PERM-3R", "SMOOTH-0.5", "SMOOTH-1", "SMOOTH-2", "LU", "GNN"],
    "panel_gnn_seed": 42, "control_seed": 42,
    "methods": {"Uni": "Uni", "LU": "GPM (categorical)", "GNN": "GNN held-out TEST fold, seeds 42/123/456"},
    "zero_tolerance": "D14 (decided 2026-09-25): value <= 1e-9 * max(1, max_y C_y) counts as zero (L_S, bound_S)",
    "violation_tolerance": "1e-10 * max(1, L_E, B_E, L_S, B_S) as in upstream conditional_bounds",
    "aggregate_rank_primary": "Spearman(Ghat, G) against the station-register neighbourhood demand",
    "smooth_crosscheck_rel_tol": 1e-12,
    "tariff_unmatched_rule": "D12 (decided 2026-09-25): a position outside every DNO polygon takes the nearest polygon; position, zone and distance are listed in uk_tariff_mapping.csv",
    "aggregate_rank_secondary": "Spearman(Ghat, A_ref) against the VD-Ref aggregate (upstream panel definition)",
}
METHODS = [("Uni", "Uni", None), ("LU", "GPM", None), ("GNN", "GNN", 42), ("GNN", "GNN", 123), ("GNN", "GNN", 456)]
GSP_TO_TNUOS_ZONE = {"_A": "Eastern", "_B": "East Midlands", "_C": "London", "_D": "N Wales & Mersey", "_E": "Midlands",
                     "_F": "Northern", "_G": "North West", "_H": "Southern", "_J": "South East", "_K": "South Wales",
                     "_L": "South Western", "_M": "Yorkshire", "_N": "Southern Scotland", "_P": "Northern Scotland"}


def annuity_factor(rate=0.035, years=20):
    return (1.0 - (1.0 + rate) ** (-years)) / rate


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def import_upstream(runtime: Path):
    if str(runtime) not in sys.path:
        sys.path.insert(0, str(runtime))
    from sglib.experiment import connection, control_fields, connection_observations, conditional_bounds
    from sglib.dataoverview.processing.features.grid_bundle import load_grid_bundle
    return connection, control_fields, connection_observations, conditional_bounds, load_grid_bundle


# ----------------------------------------------------------------- tariffs

def read_tariffs(pricing: Path) -> dict:
    workbook = pricing / "tnuos_tariffs_2026_27.xlsx"
    t9 = pd.read_excel(workbook, sheet_name="T9", header=None)
    if str(t9.iat[2, 1]) != "Zone Name" or str(t9.iat[2, 2]) != "HH Demand Tariff (£/kW)":
        raise ValueError("T9 header layout changed")
    t9 = t9.iloc[3:17]
    t25 = pd.read_excel(workbook, sheet_name="T25", header=None)
    if str(t25.iat[2, 4]) != "2026/27 Final" or str(t25.iat[3, 5]) != "Year Round (£/kW)":
        raise ValueError("T25 header layout changed")
    t25 = t25.iloc[4:18]
    out = {"T9": dict(zip(t9[1].astype(str), t9[2].astype(float))),
           "T25": dict(zip(t25[1].astype(str), t25[5].astype(float))),
           "source": {"workbook": "inputs/connection_pricing/tnuos_tariffs_2026_27.xlsx", "sha256": sha256_file(workbook),
                      "T9": "sheet T9 rows 4-17, column 'HH Demand Tariff (£/kW)' (2026/27, zero-floored)",
                      "T25": "sheet T25 rows 5-18, '2026/27 Final' 'Year Round (£/kW)' (unfloored locational)"}}
    for name in ("T9", "T25"):
        if sorted(out[name]) != sorted(GSP_TO_TNUOS_ZONE.values()):
            raise ValueError(f"{name}: zone names do not match the 14 GSP groups")
    return out


def map_tariff_zones(xy: np.ndarray, zones_path: Path, working_crs: str, tariffs: dict, region_polygon) -> pd.DataFrame:
    """Point-in-polygon on the 14 DNO licence areas; unmatched and multiply matched points are recorded.

    A point outside every DNO polygon (coastline resolution) takes the zone of the nearest polygon, with the
    distance recorded; a point inside more than one polygon takes the first by GSP letter, also recorded.
    """
    import geopandas as gpd
    zones = gpd.read_file(zones_path).to_crs(working_crs)
    zones["tnuos_zone"] = zones["Name"].map(GSP_TO_TNUOS_ZONE)
    points = gpd.GeoDataFrame({"k": np.arange(len(xy))}, geometry=gpd.points_from_xy(xy[:, 0], xy[:, 1]), crs=working_crs)
    joined = gpd.sjoin(points, zones[["Name", "tnuos_zone", "geometry"]], how="left", predicate="within")
    counts = joined.groupby("k")["Name"].count()
    first = joined.sort_values(["k", "Name"]).groupby("k").first()
    result = pd.DataFrame({"n_zone_matches": counts.reindex(range(len(xy))).fillna(0).astype(int).to_numpy(),
                           "gsp_group": first["Name"].reindex(range(len(xy))).to_numpy(object)})
    unmatched = result.n_zone_matches.eq(0).to_numpy()
    result["match_rule"] = np.where(result.n_zone_matches.eq(1), "within", np.where(unmatched, "nearest_polygon", "within_multiple_first_gsp"))
    result["nearest_distance_m"] = 0.0
    if unmatched.any():
        nearest = gpd.sjoin_nearest(points[unmatched], zones[["Name", "geometry"]], how="left", distance_col="d")
        nearest = nearest.sort_values(["k", "Name"]).groupby("k").first()
        result.loc[unmatched, "gsp_group"] = nearest["Name"].reindex(np.flatnonzero(unmatched)).to_numpy(object)
        result.loc[unmatched, "nearest_distance_m"] = nearest["d"].reindex(np.flatnonzero(unmatched)).to_numpy(float)
    result["tnuos_zone"] = result.gsp_group.map(GSP_TO_TNUOS_ZONE)
    if result.tnuos_zone.isna().any():
        raise ValueError("tariff zone unresolved after nearest-polygon rule")
    result["t9_gbp_per_kw"] = result.tnuos_zone.map(tariffs["T9"]).astype(float)
    result["t25_gbp_per_kw"] = result.tnuos_zone.map(tariffs["T25"]).astype(float)
    inside = points.within(region_polygon) if region_polygon is not None else pd.Series(True, index=points.index)
    result["inside_region_polygon"] = inside.to_numpy(bool)
    return result


# ----------------------------------------------------------------- per-region worker

def cost_vectors(g, ghat, f, x, c, t):
    q, qhat = np.maximum(g + x - f, 0.0), np.maximum(ghat + x - f, 0.0)
    return c * q + t, c * qhat + t, q, qhat


def select(truth, estimate, k):
    return np.argsort(estimate, kind="stable")[:k], np.argsort(truth, kind="stable")[:k]


def losses(truth, estimate, k):
    sel, orc = select(truth, estimate, k)
    fixed = float(np.mean(np.abs(estimate - truth)))
    regret = float(np.mean(truth[sel]) - np.mean(truth[orc]))
    scale = max(1.0, float(np.max(np.abs(truth))))
    if regret < -1e-10 * scale:
        raise ValueError("negative selection regret beyond tolerance")
    return fixed, max(0.0, regret), sel, orc


def process_region(task: dict) -> dict:
    runtime, backup, country, region = Path(task["runtime"]), Path(task["backup"]), task["country"], task["region"]
    connection, control_fields, cobs, cbounds, load_grid_bundle = import_upstream(runtime)
    import geopandas as gpd
    spec = COUNTRIES[country]
    base = backup / "inputs" / "base"
    release = backup / "inputs" / "release_r2"
    started = time.monotonic()
    consumed = {}

    grid_dir = base / f"data/datasets/2_derived/{country}/grid_bplus"
    grid, step_m, grid_meta = load_grid_bundle(region, grid_dir)
    for name in ("grid_points.parquet", "grid_metadata.json"):
        p = grid_dir / "bundles" / region / name
        consumed[p.relative_to(backup).as_posix()] = sha256_file(p)
    working = grid.to_crs(spec["working_crs"])
    grid_xy = np.column_stack([working.geometry.x, working.geometry.y])
    grid_lonlat = np.column_stack([grid.geometry.x, grid.geometry.y])

    stations = gpd.read_file(base / spec["stations_path"])
    stations["station_id"] = stations["station_id"].astype(str)
    stations = stations.set_index("station_id", verify_integrity=True)
    vd_path = base / f"results/2_Generator/{spec['directory']}/static/assignments/{region}.npz"
    consumed[vd_path.relative_to(backup).as_posix()] = sha256_file(vd_path)
    with np.load(vd_path, allow_pickle=False) as vd:
        assignment, station_ids = vd["assignment"].copy(), vd["station_id"].astype(str)
    chosen = stations.loc[station_ids].to_crs(spec["working_crs"])
    station_xy = np.column_stack([chosen.geometry.x, chosen.geometry.y])
    demand = chosen[spec["demand_column"]].to_numpy(float)
    capacity = chosen[spec["capacity_column"]].to_numpy(float)

    regions_table = gpd.read_file(base / spec["regions_path"])
    source_order = list(map(str, grid_meta["source_key_order"]))
    sources = regions_table[regions_table[spec["source_key"]].astype(str).isin(source_order)]
    source_total = sum(dict(zip(sources[spec["source_key"]].astype(str), sources[spec["region_demand_column"]].astype(float))).values())

    index = json.loads((base / f"results/2_Generator/{spec['directory']}/candidates/candidate_index.json").read_text(encoding="utf-8"))
    fields = {}
    for method, label, seed in METHODS:
        entries = [e for e in index["entries"] if e["label"] == label and e["region"] == region and e.get("seed") == seed and not e.get("qa_only")]
        if len(entries) != 1:
            raise ValueError(f"{region}: candidate {label}/{seed} not unique")
        path = base / f"results/2_Generator/{spec['directory']}" / entries[0]["path"]
        digest = sha256_file(path)
        if digest != entries[0]["sha256"]:
            raise ValueError(f"{path}: candidate hash differs from its index")
        consumed[path.relative_to(backup).as_posix()] = digest
        with np.load(path, allow_pickle=False) as arrays:
            if not np.array_equal(arrays["grid_row"], np.arange(len(arrays["data"]))):
                raise ValueError(f"{path}: grid identity mismatch")
            fields[(method, seed)] = {"values": arrays["data"].astype(float), "sha256": digest, "fold": entries[0].get("fold")}
    for v in fields.values():
        if v["values"].shape != (len(grid_xy),):
            raise ValueError(f"{region}: field length differs from grid")

    rows = connection.stride_candidates(len(grid_xy), PROTOCOL["candidate_count"])
    queries = grid_xy[rows]
    reference = connection.reference_field(assignment, demand)
    pool_hash = hashlib.sha256(json.dumps({"grid_row": rows.tolist(), "xy": queries.tolist(), "working_crs": spec["working_crs"]},
                                          sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    # ---- bit-for-bit cross-check against the r2 release observation
    surf_path = release / f"3_Experiment/{spec['directory']}/observations/{region}/connection/surfaces.npz"
    consumed[surf_path.relative_to(backup).as_posix()] = sha256_file(surf_path)
    with np.load(surf_path, allow_pickle=False) as blob:
        upstream = {k: blob[k].copy() for k in blob.files}
    up_fields = list(zip(upstream["field_labels"].astype(str), upstream["field_seeds"].astype(int)))
    check_detail = {}
    check = {"candidate_id": bool(np.array_equal(upstream["candidate_id"], rows)),
             "source_total": bool(float(upstream["source_total"]) == float(source_total))}

    # ---- tariffs (UK)
    tariff_frame = None
    if spec["tariff"]:
        region_polygon = None
        tariff_frame = map_tariff_zones(queries, backup / "inputs/connection_pricing/dno_zones_20240503.geojson",
                                        spec["working_crs"], task["tariffs"], region_polygon)
        tariff_frame.insert(0, "candidate_id", rows)
        tariff_frame.insert(0, "region", region)
        tariff_frame.insert(0, "country", country)
    af = annuity_factor()
    c = spec["c_per_mva"]
    k = PROTOCOL["shortlist_k"]

    def t_vector(x, which):
        if tariff_frame is None:
            return np.zeros(len(rows))
        col = {"T9": "t9_gbp_per_kw", "T25": "t25_gbp_per_kw"}[which]
        return x * 1000.0 * tariff_frame[col].to_numpy(float) * af

    cand_records, cost_rows, panel_rows, t25_rows, surfaces = [], [], [], [], {}
    ref_rows = []
    for radius in PROTOCOL["bound_radii_km"]:
        smembers = cobs.compiled_members(station_xy, queries, radius * 1000.0)
        gmembers = cobs.compiled_members(grid_xy, queries, radius * 1000.0)
        g = connection.neighbourhood_sum(demand, smembers)
        f = connection.neighbourhood_sum(capacity, smembers)
        a_ref = connection.neighbourhood_sum(reference, gmembers)
        band = [float(r) for r in upstream["radii_km"]].index(radius)
        check[f"G_R{radius:g}"] = bool(np.array_equal(upstream["G"][band], g))
        check[f"F_R{radius:g}"] = bool(np.array_equal(upstream["F"][band], f))
        check[f"A_ref_R{radius:g}"] = bool(np.array_equal(upstream["A_ref"][band], a_ref))
        ghat = {}
        for key, v in fields.items():
            ghat[key] = connection.neighbourhood_sum(v["values"], gmembers)
            up_key = ("GPM" if key[0] == "LU" else key[0], key[1] or 0)
            check[f"Ghat_{key[0]}{key[1] or ''}_R{radius:g}"] = bool(np.array_equal(upstream["Ghat"][up_fields.index(up_key), band], ghat[key]))
        n_st = np.array([len(m) for m in smembers])
        median_g = float(np.median(g))
        ref_rows.append({"country": country, "region": region, "radius_km": radius, "median_G": median_g,
                         "ref_mean_abs_error": float(np.mean(np.abs(a_ref - g))),
                         "ref_score_radius": float(np.mean(np.abs(a_ref - g))) / median_g if median_g > 0 else None,
                         "zero_station_fraction": float(np.mean(n_st == 0))})
        surfaces[radius] = {"G": g, "F": f, "A_ref": a_ref, "Ghat": ghat, "gmembers": gmembers}

        estimates = {("Ref", None): a_ref, **ghat}
        for x in PROTOCOL["loads_mw"]:
            t9 = t_vector(x, "T9")
            for (method, seed), est in estimates.items():
                truth_c, est_c, q, qhat = cost_vectors(g, est, f, x, c, t9)
                l_e, l_s, sel, orc = losses(truth_c, est_c, k)
                sel_mask, orc_mask = np.zeros(len(rows), bool), np.zeros(len(rows), bool)
                sel_mask[sel] = True; orc_mask[orc] = True
                tie = est_c == est_c[sel[-1]]
                cost_rows.append({"country": country, "region": region, "method": method, "seed": seed, "radius_km": radius,
                                  "dc_mw": x, "tariff": spec["tariff"] or "none", "cost_unit": spec["cost_unit"],
                                  "L_E": l_e, "L_S": l_s, "n_candidates": len(rows), "shortlist_k": k,
                                  "true_zero_q_fraction": float(np.mean(q == 0)), "estimated_zero_q_fraction": float(np.mean(qhat == 0)),
                                  "cutoff_tie_size": int(tie.sum()), "selected_cutoff_tie_count": int(tie[sel].sum()),
                                  "selected_ids": "|".join(map(str, rows[sel])), "oracle_ids": "|".join(map(str, rows[orc])),
                                  "field_sha256": fields[(method, seed)]["sha256"] if method != "Ref" else "VD-Ref"})
                cand_records.append(pd.DataFrame({
                    "country": country, "region": region, "radius_km": radius, "dc_mw": x, "method": method,
                    "seed": np.full(len(rows), -1 if seed is None else seed), "candidate_id": rows,
                    "x": queries[:, 0], "y": queries[:, 1], "lon": grid_lonlat[rows, 0], "lat": grid_lonlat[rows, 1],
                    "n_stations": n_st, "G": g, "F": f, "Ghat": est, "T": t9, "C_true": truth_c, "C_est": est_c,
                    "selected": sel_mask, "oracle": orc_mask}))
            # ---- T25 single-setting sensitivity (UK, R=10, X=300)
            if (spec["tariff"] and radius == PROTOCOL["t25_sensitivity"]["radius_km"]
                    and x == PROTOCOL["t25_sensitivity"]["load_mw"]):
                t25 = t_vector(x, "T25")
                for (method, seed), est in estimates.items():
                    c9, e9, _, _ = cost_vectors(g, est, f, x, c, t9)
                    c25, e25, _, _ = cost_vectors(g, est, f, x, c, t25)
                    le9, ls9, s9, o9 = losses(c9, e9, k)
                    le25, ls25, s25, o25 = losses(c25, e25, k)
                    cen = lambda idx: queries[idx].mean(axis=0)
                    t25_rows.append({"country": country, "region": region, "method": method, "seed": seed, "radius_km": radius,
                                     "dc_mw": x, "L_E_T9": le9, "L_E_T25": le25, "L_S_T9": ls9, "L_S_T25": ls25,
                                     "selected_overlap": int(len(set(s9) & set(s25))), "oracle_overlap": int(len(set(o9) & set(o25))),
                                     "selected_shift_km": float(np.linalg.norm(cen(s9) - cen(s25)) / 1000.0),
                                     "oracle_shift_km": float(np.linalg.norm(cen(o9) - cen(o25)) / 1000.0),
                                     "n_tariff_zones_in_region": int(tariff_frame.tnuos_zone.nunique()),
                                     "selected_ids_T9": "|".join(map(str, rows[s9])), "selected_ids_T25": "|".join(map(str, rows[s25])),
                                     "oracle_ids_T9": "|".join(map(str, rows[o9])), "oracle_ids_T25": "|".join(map(str, rows[o25]))})

        # ---- nine-distribution panel (GNN seed 42 only)
        controls = control_fields.build_control_arms(reference, grid_xy / 1000.0, radius, seed=PROTOCOL["control_seed"])
        up_controls = upstream["control_fields"][band]
        for i, name in enumerate(upstream["control_labels"].astype(str)):
            # PERM is a pure permutation and must match exactly; SMOOTH passes sin/exp, whose last bit may differ
            # between the machine that sealed r2 and this one (registered tolerance, observed maximum recorded).
            rel = float(np.max(np.abs(up_controls[i] - controls[name]) / np.maximum(np.abs(up_controls[i]), 1e-300)))
            check[f"control_{name}_R{radius:g}"] = bool(np.array_equal(up_controls[i], controls[name])) if name.startswith("PERM") \
                else bool(rel <= PROTOCOL["smooth_crosscheck_rel_tol"])
            check_detail[f"control_{name}_R{radius:g}_max_rel_diff"] = rel
        panel_fields = {"Ref": reference, **controls, "LU": fields[("LU", None)]["values"],
                        "GNN": fields[("GNN", PROTOCOL["panel_gnn_seed"])]["values"]}
        for name in PROTOCOL["panel"]:
            value = panel_fields[name]
            est = a_ref if name == "Ref" else connection.neighbourhood_sum(value, gmembers)
            rho_cell = cobs.rank_agreement(value, reference)
            rho_g = cobs.rank_agreement(est, g)
            rho_a = cobs.rank_agreement(est, a_ref)
            for x in PROTOCOL["loads_mw"]:
                truth_c, est_c, _, _ = cost_vectors(g, est, f, x, c, t_vector(x, "T9"))
                l_e, l_s, _, _ = losses(truth_c, est_c, k)
                panel_rows.append({"country": country, "region": region, "distribution": name,
                                   "gnn_seed": PROTOCOL["panel_gnn_seed"] if name == "GNN" else None, "radius_km": radius,
                                   "dc_mw": x, "rho_cell": rho_cell, "rho_agg": rho_g, "rho_agg_vs_Aref": rho_a,
                                   "L_E": l_e, "L_S": l_s, "cost_unit": spec["cost_unit"]})

    # ---- Ref screen (D2)
    ref = pd.DataFrame(ref_rows)
    scores = ref.ref_score_radius.tolist()
    q_r = max(scores) if all(s is not None and np.isfinite(s) for s in scores) else None
    eligible = q_r is not None and q_r <= PROTOCOL["ref_threshold"]
    reason = "" if eligible else ("REF_ZERO_MEDIAN_DEMAND" if q_r is None else "REF_SCORE_ABOVE_THRESHOLD")
    ref["q_r"] = q_r; ref["ref_eligible"] = eligible; ref["ref_reason"] = reason
    pre = pd.read_csv(release / f"3_Experiment/{spec['directory']}/preflight/connection/connection_scenario_preflight.csv",
                      dtype={"region": str}, float_precision="round_trip")
    pre = pre[pre.region.eq(region)]
    up_score = pre.ref_score.iloc[0]
    check["ref_score_matches_r2_preflight"] = bool((q_r is None and pd.isna(up_score)) or (q_r is not None and q_r == float(up_score)))
    check["ref_eligible_matches_r2_preflight"] = bool(eligible == bool(pre.ref_eligible.iloc[0]))

    # ---- scale curves: 8 radii x {station, grid} centres, same centres/radii/candidates for both targets
    scale_rows = []
    for radius in PROTOCOL["scale_radii_km"]:
        for centre_kind, centres in (("station", station_xy), ("grid", queries)):
            if centre_kind == "grid" and radius in surfaces:
                gmem = surfaces[radius]["gmembers"]
            else:
                gmem = cobs.compiled_members(grid_xy, centres, radius * 1000.0)
            smem = cobs.compiled_members(station_xy, centres, radius * 1000.0)
            ledger = connection.neighbourhood_sum(demand, smem)
            vdref = connection.neighbourhood_sum(reference, gmem)
            n_st = np.array([len(m) for m in smem])
            for (method, seed), v in fields.items():
                est = connection.neighbourhood_sum(v["values"], gmem)
                scale_rows.append({"country": country, "region": region, "method": method, "seed": seed, "radius_km": radius,
                                   "centres": centre_kind, "mae_decision": float(np.mean(np.abs(est - ledger))),
                                   "mae_representation": float(np.mean(np.abs(est - vdref))),
                                   "rmse_decision": float(np.sqrt(np.mean((est - ledger) ** 2))),
                                   "rmse_representation": float(np.sqrt(np.mean((est - vdref) ** 2))),
                                   "mean_G": float(np.mean(ledger)), "mean_Aref": float(np.mean(vdref)),
                                   "zero_station_share": float(np.mean(n_st == 0)), "n_stations_mean": float(np.mean(n_st)),
                                   "n_stations_median": float(np.median(n_st)), "n_centres": len(centres)})

    packed = {"G": np.stack([surfaces[r]["G"] for r in PROTOCOL["bound_radii_km"]]),
              "F": np.stack([surfaces[r]["F"] for r in PROTOCOL["bound_radii_km"]]),
              "A_ref": np.stack([surfaces[r]["A_ref"] for r in PROTOCOL["bound_radii_km"]]),
              "Ghat": np.stack([np.stack([surfaces[r]["Ghat"][key] for r in PROTOCOL["bound_radii_km"]]) for key in fields]),
              "T9_per_mw": (np.zeros(len(rows)) if tariff_frame is None else 1000.0 * tariff_frame.t9_gbp_per_kw.to_numpy(float) * af)}
    support = {"country": country, "region": region, "n_grid": len(grid_xy), "grid_step_m": step_m, "n_stations": len(station_ids),
               "n_sources": len(sources), "source_total": source_total, "station_demand_total": float(demand.sum()),
               "n_candidates": len(rows), "candidate_pool_hash": pool_hash, "ref_q_r": q_r, "ref_eligible": eligible,
               "ref_reason": reason, "in_budget_domain": True,
               "fields": {f"{m}{'' if s is None else s}": {"sha256": v["sha256"], "fold": v["fold"]} for (m, s), v in fields.items()}}
    return {"country": country, "region": region, "consumed": consumed, "check": check, "check_detail": check_detail, "support": support,
            "ref": ref.to_dict("records"), "cost": cost_rows, "panel": panel_rows, "scale": scale_rows, "t25": t25_rows,
            "tariff": None if tariff_frame is None else tariff_frame.to_dict("records"),
            "candidates": pd.concat(cand_records, ignore_index=True), "packed": packed,
            "field_keys": [f"{m}{'' if s is None else s}" for (m, s) in fields], "elapsed_s": time.monotonic() - started}


# ----------------------------------------------------------------- country-level budget and bounds

def adequacy(country: str, results: list[dict], runtime: Path) -> tuple[pd.DataFrame, dict]:
    _, _, _, cbounds, _ = import_upstream(runtime)
    spec = COUNTRIES[country]
    c, k, af = spec["c_per_mva"], PROTOCOL["shortlist_k"], annuity_factor()
    by_region = {r["region"]: r for r in results}
    regions = sorted(by_region)
    keys = results[0]["field_keys"]
    norm = {}
    for name, r in by_region.items():
        total = r["support"]["source_total"]
        if total <= 0:
            raise ValueError(f"{name}: non-positive source total in the budget domain")
        for fi, key in enumerate(keys):
            for bi, radius in enumerate(PROTOCOL["bound_radii_km"]):
                norm[name, key, radius] = float(np.max(np.abs(r["packed"]["Ghat"][fi, bi] - r["packed"]["G"][bi]))) / total
    rows, violations_realised, violations_unrealised = [], 0, 0
    for name in regions:
        r = by_region[name]
        total = r["support"]["source_total"]
        for fi, key in enumerate(keys):
            method = "Uni" if key == "Uni" else "LU" if key == "LU" else "GNN"
            seed = None if method != "GNN" else int(key[3:])
            for bi, radius in enumerate(PROTOCOL["bound_radii_km"]):
                others = np.array([norm[o, key, radius] for o in regions if o != name])
                current = norm[name, key, radius]
                g, f, ghat = r["packed"]["G"][bi], r["packed"]["F"][bi], r["packed"]["Ghat"][fi, bi]
                for eta in PROTOCOL["etas"]:
                    b_norm = float(np.quantile(others, eta, method="linear"))
                    budget = b_norm * total
                    realised = current <= b_norm + 1e-12 * max(1.0, current, b_norm)
                    for x in PROTOCOL["loads_mw"]:
                        t = x * r["packed"]["T9_per_mw"]
                        truth = c * np.maximum(g + x - f, 0.0) + t
                        est = c * np.maximum(ghat + x - f, 0.0) + t
                        lo = c * np.maximum(ghat - budget + x - f, 0.0) + t
                        hi = c * np.maximum(ghat + budget + x - f, 0.0) + t
                        v = cbounds.loss_bounds(truth, est, lo, hi, k=k)
                        tol = 1e-10 * max(1.0, *v.values())
                        violated = v["L_E"] > v["B_E"] + tol or v["L_S"] > v["B_S"] + tol
                        if violated and realised:
                            violations_realised += 1
                        elif violated:
                            violations_unrealised += 1
                        ztol = 1e-9 * max(1.0, float(np.max(np.abs(truth))))
                        rows.append({"country": country, "region": name, "method": method, "seed": seed, "radius_km": radius,
                                     "dc_mw": x, "eta": eta, "cost_unit": spec["cost_unit"], "source_total": total,
                                     "budget_normalized": b_norm, "e_budget": budget, "eps_max": current * total,
                                     "realized_normalized_error": current, "budget_realised": bool(realised),
                                     "L_E": v["L_E"], "L_E_bound": v["B_E"], "L_S": v["L_S"], "L_S_bound": v["B_S"],
                                     "slack_E": v["B_E"] - v["L_E"], "slack_S": v["B_S"] - v["L_S"],
                                     "valid_E": bool(v["L_E"] <= v["B_E"] + tol), "valid_S": bool(v["L_S"] <= v["B_S"] + tol),
                                     "zero_tolerance": ztol, "L_S_zero": bool(v["L_S"] <= ztol), "L_S_bound_zero": bool(v["B_S"] <= ztol),
                                     "n_calibration": len(others), "calibration_regions": "|".join(o for o in regions if o != name)})
    frame = pd.DataFrame(rows)
    summary = {"units_total": len(frame), "units_budget_realised": int(frame.budget_realised.sum()),
               "violations_budget_realised": violations_realised, "violations_budget_not_realised": violations_unrealised}
    if violations_realised:
        raise ValueError(f"{country}: {violations_realised} conditional bound violations")
    return frame, summary


# ----------------------------------------------------------------- driver

def write_csv(frame: pd.DataFrame, path: Path, outputs: dict, root: Path):
    frame.to_csv(path, index=False, float_format="%.17g", lineterminator="\n")
    outputs[path.relative_to(root).as_posix()] = {"sha256": sha256_file(path), "rows": len(frame)}


def main() -> None:
    here = Path(__file__).resolve()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backup", type=Path, default=here.parents[2] / "results" / "_backup")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--countries", nargs="+", default=["uk", "au"])
    ap.add_argument("--regions", nargs="*", default=None, help="smoke subset only; a formal run uses every region")
    ap.add_argument("--root", type=Path, default=None, help="override output root (smoke tests outside recompute/)")
    args = ap.parse_args()
    backup = args.backup.resolve()
    root = args.root.resolve() if args.root else backup / "recompute" / args.run_id
    if root.exists() and any(root.iterdir()):
        raise SystemExit(f"{root} already has content; choose a new run_id")
    root.mkdir(parents=True, exist_ok=True)
    log_path = root / "run.log"
    log = log_path.open("w", encoding="utf-8")

    def say(message):
        line = f"{datetime.now(timezone.utc).isoformat()} {message}"
        print(line, flush=True); log.write(line + "\n"); log.flush()

    freeze = sorted((backup / "manifests").glob("active_input_freeze_*.json"))
    passing = [p for p in freeze if json.loads(p.read_text(encoding="utf-8"))["conclusion"] == "PASS"]
    if not passing:
        raise SystemExit("no PASS active-input freeze receipt; run close_active_inputs.py first")
    freeze_receipt = passing[-1]
    say(f"input freeze: {freeze_receipt.name}")

    zip_path = backup / UPSTREAM_ZIP
    runtime = Path(tempfile.mkdtemp(prefix="sg69c4008_"))
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(runtime)
    say(f"upstream code {UPSTREAM_COMMIT} extracted from {UPSTREAM_ZIP} (sha256 {sha256_file(zip_path)})")
    tariffs = read_tariffs(backup / "inputs/connection_pricing")

    code_dir = root / "code"; code_dir.mkdir()
    shutil.copy2(here, code_dir / here.name)
    upstream_modules = {m: sha256_file(runtime / m) for m in (
        "sglib/experiment/connection.py", "sglib/experiment/control_fields.py", "sglib/experiment/connection_observations.py",
        "sglib/experiment/conditional_bounds.py", "sglib/dataoverview/processing/features/grid_bundle.py")}
    config = {"schema": RUN_SCHEMA, "unit": "paper2_fixed_load_connection", "run_id": args.run_id,
              "release_id": RELEASE_ID, "upstream_commit": UPSTREAM_COMMIT, "protocol": PROTOCOL,
              "countries": {cc: COUNTRIES[cc] for cc in args.countries}, "annuity_factor_full_precision": annuity_factor(),
              "tariffs": tariffs, "distance_implementation": {
                  "connection_and_scale": "projected planar distance in the national working CRS (UK EPSG:27700, AU EPSG:7856), cKDTree closed ball",
                  "siting_and_sizing": "reused C4 products (haversine distance), not recomputed here"},
              "input_freeze_receipt": {"path": f"manifests/{freeze_receipt.name}", "sha256": sha256_file(freeze_receipt)}}
    (root / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    tasks = []
    for cc in args.countries:
        regs = json.loads((backup / f"inputs/base/results/2_Generator/{COUNTRIES[cc]['directory']}/static/assignments/index.json")
                          .read_text(encoding="utf-8"))["regions"]
        tasks += [{"runtime": str(runtime), "backup": str(backup), "country": cc, "region": r["region"],
                   "tariffs": {"T9": tariffs["T9"], "T25": tariffs["T25"]}} for r in regs
                  if args.regions is None or r["region"] in args.regions]
    say(f"{len(tasks)} region tasks, {args.workers} workers")
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(process_region, tasks):
            bad = [k for k, v in result["check"].items() if not v]
            say(f"{result['country']}/{result['region']}: {result['elapsed_s']:.1f}s crosscheck_fail={bad} "
                f"ref_q={result['support']['ref_q_r']} eligible={result['support']['ref_eligible']}")
            results.append(result)

    outputs, checks, consumed = {}, {"crosscheck": {}, "adequacy": {}}, {}
    (root / "candidates").mkdir()
    for cc in args.countries:
        part = [r for r in results if r["country"] == cc]
        for r in part:
            consumed.update(r["consumed"])
            checks["crosscheck"][f"{cc}/{r['region']}"] = r["check"]
            checks.setdefault("crosscheck_detail", {})[f"{cc}/{r['region']}"] = r["check_detail"]
        cand = pd.concat([r["candidates"] for r in part], ignore_index=True)
        path = root / "candidates" / f"{cc}_candidates.parquet"
        cand.to_parquet(path, index=False)
        outputs[path.relative_to(root).as_posix()] = {"sha256": sha256_file(path), "rows": len(cand)}
        write_csv(pd.DataFrame([r["support"] | {"fields": json.dumps(r["support"]["fields"])} for r in part]),
                  root / f"{cc}_region_support.csv", outputs, root)
        write_csv(pd.DataFrame([row for r in part for row in r["ref"]]), root / f"{cc}_ref_eligibility.csv", outputs, root)
        write_csv(pd.DataFrame([row for r in part for row in r["cost"]]), root / f"{cc}_cost_oof.csv", outputs, root)
        write_csv(pd.DataFrame([row for r in part for row in r["panel"]]), root / f"{cc}_controls.csv", outputs, root)
        scale = pd.DataFrame([row for r in part for row in r["scale"]])
        write_csv(scale[scale.centres.eq("station")], root / f"{cc}_scale_station.csv", outputs, root)
        write_csv(scale[scale.centres.eq("grid")], root / f"{cc}_scale_grid.csv", outputs, root)
        if part[0]["tariff"] is not None:
            write_csv(pd.DataFrame([row for r in part for row in r["tariff"]]), root / f"{cc}_tariff_mapping.csv", outputs, root)
            write_csv(pd.DataFrame([row for r in part for row in r["t25"]]), root / f"{cc}_t25_sensitivity.csv", outputs, root)
        adq, summary = adequacy(cc, part, runtime)
        write_csv(adq, root / f"{cc}_adequacy.csv", outputs, root)
        checks["adequacy"][cc] = summary
        say(f"{cc} adequacy: {summary}")

    failing = {k: [n for n, v in c.items() if not v] for k, c in checks["crosscheck"].items() if not all(c.values())}
    checks["crosscheck_all_pass"] = not failing
    checks["crosscheck_failures"] = failing
    checks["zero_violation_all_pass"] = all(s["violations_budget_realised"] == 0 for s in checks["adequacy"].values())
    (root / "checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    outputs["checks.json"] = {"sha256": sha256_file(root / "checks.json")}
    with (root / "inputs_manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["backup_relative_path", "sha256"])
        for rel in sorted(consumed):
            writer.writerow([rel, consumed[rel]])
        for extra in ("inputs/connection_pricing/tnuos_tariffs_2026_27.xlsx", "inputs/connection_pricing/dno_zones_20240503.geojson", UPSTREAM_ZIP):
            writer.writerow([extra, sha256_file(backup / extra)])
    receipt = {"schema": RUN_SCHEMA, "run_id": args.run_id, "completed_utc": datetime.now(timezone.utc).isoformat(),
               "status": "PASS" if checks["crosscheck_all_pass"] and checks["zero_violation_all_pass"] else "FAIL",
               "config_sha256": sha256_file(root / "config.json"), "inputs_manifest_sha256": sha256_file(root / "inputs_manifest.csv"),
               "code": {"script": f"code/{here.name}", "script_sha256": sha256_file(code_dir / here.name),
                        "upstream_zip": UPSTREAM_ZIP, "upstream_zip_sha256": sha256_file(zip_path), "upstream_modules": upstream_modules,
                        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
               "outputs": outputs}
    (root / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    say(f"status {receipt['status']}; crosscheck failures {failing}")
    log.close()
    shutil.rmtree(runtime, ignore_errors=True)
    if receipt["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
