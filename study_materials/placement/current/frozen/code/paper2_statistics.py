"""Step C: task exports and the paper's statistics for one recompute run (plan sections 6, 5.3-5.5).

Reads ``recompute/<run_id>/`` (Step B products) and the r2 release per-seed task
products in ``inputs/release_r2/3_Experiment`` (reconstruction, siting, sizing:
reused C4, not recomputed). Writes ``<run>/tasks/`` and ``<run>/stats/``.

Statistics protocol (manuscript 6ff3af5, Sec. 4 protocol + App. C):
  * region is the unit; GNN seeds are averaged within region first, then paired with LU;
  * primary descriptive effect: median regional percentage change (undefined where the baseline is zero);
  * aggregate relative effect sum(d)/sum(b) with a paired-region percentile bootstrap
    (B=10,000, PCG64 seed 42, regions resampled, never seeds);
  * exact two-sided sign-flip over all 2^n assignments on raw regional differences;
  * one four-module Holm family per country (reconstruction, siting, sizing many-to-one, connection L_E at
    R=10 km, X=300 MW). The upstream nine-member C4 family enters only the disclosure table.
The upstream primitives ``sign_flip``, ``holm`` and ``_bootstrap`` are imported unchanged from the
release archive (commit 69c4008).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import zipfile

import numpy as np
import pandas as pd

UPSTREAM_ZIP = "code/upstream_release_69c4008732ed.zip"
BOOT_SEED, BOOT_B = 42, 10_000
ZERO_REL = 1e-9  # D9: scale-curve numerical zero, relative to the regional total D_r
COUNTRY_DIR = {"uk": "1_UK", "au": "2_AU"}
UNITS = {"uk": {"recon": "MVA", "connection": "GBP"}, "au": {"recon": "MW", "connection": "MVA_equivalent_PF1"}}
MAIN_R, MAIN_X, MAIN_ETA = 10.0, 300.0, 0.9
PANEL = ["Ref", "PERM-R/3", "PERM-R", "PERM-3R", "SMOOTH-0.5", "SMOOTH-1", "SMOOTH-2", "LU", "GNN"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def upstream(backup: Path):
    runtime = Path(tempfile.mkdtemp(prefix="sg69c4008_stats_"))
    with zipfile.ZipFile(backup / UPSTREAM_ZIP) as archive:
        archive.extractall(runtime)
    sys.path.insert(0, str(runtime))
    from sglib.analysis import paired_inference
    from sglib.experiment.connection_observations import rank_agreement
    return paired_inference, rank_agreement


# ----------------------------------------------------------------- C.1 task exports

def export_tasks(backup: Path, cc: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    exp = backup / "inputs/release_r2/3_Experiment" / COUNTRY_DIR[cc]
    analysis = backup / "inputs/release_r2/4_Analysis" / COUNTRY_DIR[cc]
    regions = [p.name for p in sorted((exp / "observations").iterdir()) if p.is_dir()]
    rows, consumed = [], {}
    for region in regions:
        path = exp / "observations" / region / "reconstruction/metrics.csv"
        consumed[path.relative_to(backup).as_posix()] = sha256_file(path)
        m = pd.read_csv(path, float_precision="round_trip")
        m = m[m.candidate.isin(["Uni", "GPM", "GNN"]) & m.allocator.eq("VD") & m.metric.isin(["rmse", "mae"])]
        for r in m.itertuples():
            rows.append({"country": cc, "region": region, "module": "reconstruction", "metric": r.metric, "matching": "",
                         "method": "LU" if r.candidate == "GPM" else r.candidate, "seed": None if pd.isna(r.seed) else int(r.seed),
                         "fold": None if pd.isna(r.fold) else int(r.fold), "value": float(r.value), "unit": r.unit,
                         "status": r.status, "n_valid": int(r.n_targets), "n_requested": int(r.n_targets),
                         "source": path.relative_to(backup).as_posix()})
        for cand in ("Uni", "GPM", "GNN"):
            for seed_dir in sorted((exp / "planning" / region / cand).iterdir()):
                path = seed_dir / "metrics.csv"
                consumed[path.relative_to(backup).as_posix()] = sha256_file(path)
                consumed[(seed_dir / "decisions.csv").relative_to(backup).as_posix()] = sha256_file(seed_dir / "decisions.csv")
                m = pd.read_csv(path, float_precision="round_trip")
                keep = (m.task.eq("siting") & m.metric.eq("WSD")) | (m.task.eq("sizing") & m.metric.isin(["RSD_median", "RSD_mean"]))
                for r in m[keep].itertuples():
                    rows.append({"country": cc, "region": region, "module": r.task, "metric": r.metric,
                                 "matching": "" if r.matching == "not_applicable" else r.matching,
                                 "method": "LU" if cand == "GPM" else cand, "seed": None if cand != "GNN" else int(r.seed),
                                 "fold": None if pd.isna(r.fold) else int(r.fold), "value": float(r.value), "unit": r.unit,
                                 "status": r.status, "n_valid": int(r.n_valid_matches), "n_requested": int(r.n_requested_matches),
                                 "source": path.relative_to(backup).as_posix()})
    seeds = pd.DataFrame(rows)
    if not seeds.status.eq("VALID").all():
        raise ValueError(f"{cc}: non-VALID task rows")
    keys = ["country", "region", "module", "metric", "matching", "method"]
    region = seeds.groupby(keys, dropna=False, sort=True).agg(value=("value", "mean"), n_seeds=("value", "size"),
                                                               unit=("unit", "first")).reset_index()
    # cross-check against the r2 C4 seed-averaged region values
    path = analysis / "C4/region_metrics.csv"
    consumed[path.relative_to(backup).as_posix()] = sha256_file(path)
    c4 = pd.read_csv(path, float_precision="round_trip")
    check = {}
    for module, metric, matching in (("reconstruction", "rmse", ""), ("siting", "WSD", ""), ("sizing", "RSD_median", "many_to_one")):
        ours = region[(region.module == module) & (region.metric == metric) & (region.matching == matching) & region.method.isin(["LU", "GNN"])]
        theirs = c4[(c4.task_coordinate == module) & (c4.metric == metric)].assign(method=lambda f: f.candidate.replace({"GPM": "LU"}))
        merged = ours.merge(theirs[["region", "method", "value"]], on=["region", "method"], suffixes=("", "_c4"))
        rel = float(np.max(np.abs(merged.value - merged.value_c4) / np.abs(merged.value_c4))) if len(merged) else None
        check[f"{module}:{metric}:{matching or '-'}"] = {"n": len(merged), "max_rel_diff_vs_C4": rel}
    # C4 region_metrics carries sizing under many-to-one only; one-to-one is checked against the country means
    path = analysis / "support/C4_matching_summary.csv"
    consumed[path.relative_to(backup).as_posix()] = sha256_file(path)
    summary = pd.read_csv(path, float_precision="round_trip")
    for matching in ("many_to_one", "one_to_one"):
        for method, cand in (("LU", "GPM"), ("GNN", "GNN"), ("Uni", "Uni")):
            ours = region[(region.module == "sizing") & (region.metric == "RSD_median") & (region.matching == matching) & (region.method == method)]
            theirs = summary[(summary.candidate == cand) & (summary.metric == "RSD_median") & (summary.matching == matching)]
            if len(theirs) == 1:
                check[f"sizing:RSD_median:{matching}:{method}:country_mean"] = {
                    "n": len(ours), "rel_diff_vs_C4_matching_summary": float(abs(ours.value.mean() - theirs["mean"].iloc[0]) / abs(theirs["mean"].iloc[0]))}
    return seeds, region, {"consumed": consumed, "c4_crosscheck": check}


# ----------------------------------------------------------------- inference primitives

def paired(pi, lu: np.ndarray, gnn: np.ndarray) -> dict:
    lu, gnn = np.asarray(lu, float), np.asarray(gnn, float)
    d = gnn - lu
    defined = lu != 0
    boot = pi._bootstrap(gnn, lu, [[i] for i in range(len(lu))], seed=BOOT_SEED, repetitions=BOOT_B)
    flip = pi.sign_flip(d)
    return {"n": len(d), "lu_mean": float(lu.mean()), "gnn_mean": float(gnn.mean()),
            "median_rel_pct": float(np.median(100 * d[defined] / lu[defined])) if defined.any() else None,
            "n_pct_defined": int(defined.sum()), "wins": int((d < 0).sum()), "losses": int((d > 0).sum()),
            "ties": int((d == 0).sum()), "p_raw": flip["p"], "p_min": flip["p_min"],
            "rel_effect_pct": float(100 * d.sum() / lu.sum()) if lu.sum() else None,
            "rel_ci_lo": boot["relative_ci"][0] if boot["relative_ci"] else None,
            "rel_ci_hi": boot["relative_ci"][1] if boot["relative_ci"] else None,
            "zero_denominator_draws": boot["zero_denominator_draws"], "bootstrap_indices_sha256": boot["indices_sha256"]}


def module_rows(pi, region: pd.DataFrame, seeds: pd.DataFrame, cost: pd.DataFrame, cc: str, eligible: set) -> tuple[pd.DataFrame, pd.DataFrame]:
    """All module contrasts (family and descriptive) plus per-seed rows."""
    specs = [("reconstruction", "rmse", "", True), ("siting", "WSD", "", True), ("sizing", "RSD_median", "many_to_one", True),
             ("sizing", "RSD_median", "one_to_one", False), ("reconstruction", "mae", "", False)]
    rows, seed_rows = [], []
    for module, metric, matching, family in specs:
        sub = region[(region.module == module) & (region.metric == metric) & (region.matching == matching)]
        piv = sub.pivot(index="region", columns="method", values="value").dropna(subset=["LU", "GNN"])
        rows.append({"country": cc, "module": module, "metric": metric, "matching": matching, "radius_km": None, "dc_mw": None,
                     "in_family": family, "unit": sub.unit.iloc[0], "regions": "|".join(piv.index), "excluded": "",
                     **paired(pi, piv.LU.to_numpy(), piv.GNN.to_numpy())})
        s = seeds[(seeds.module == module) & (seeds.metric == metric) & (seeds.matching == matching)]
        lu = s[s.method.eq("LU")].set_index("region").value
        for seed in (42, 123, 456):
            g = s[s.method.eq("GNN") & s.seed.eq(seed)].set_index("region").value.reindex(lu.index)
            seed_rows.append({"country": cc, "module": module, "metric": metric, "matching": matching, "seed": seed,
                              "in_family_module": family, **paired(pi, lu.to_numpy(), g.to_numpy())})
    for radius in (10.0, 20.0):
        for x in (100.0, 300.0, 500.0):
            for loss in ("L_E", "L_S"):
                sub = cost[(cost.radius_km == radius) & (cost.dc_mw == x)]
                lu = sub[sub.method.eq("LU")].set_index("region")[loss]
                gnn = sub[sub.method.eq("GNN")].groupby("region")[loss].mean().reindex(lu.index)
                family = loss == "L_E" and radius == MAIN_R and x == MAIN_X
                rows.append({"country": cc, "module": "connection", "metric": loss, "matching": "", "radius_km": radius, "dc_mw": x,
                             "in_family": family, "unit": UNITS[cc]["connection"], "regions": "|".join(lu.index), "excluded": "",
                             **paired(pi, lu.to_numpy(), gnn.to_numpy())})
                for seed in (42, 123, 456):
                    g = sub[sub.method.eq("GNN") & sub.seed.eq(seed)].set_index("region")[loss].reindex(lu.index)
                    seed_rows.append({"country": cc, "module": "connection", "metric": loss, "matching": "", "radius_km": radius,
                                      "dc_mw": x, "seed": seed, "in_family_module": family, **paired(pi, lu.to_numpy(), g.to_numpy())})
    # descriptive only: the connection comparison restricted to the D2 Ref-eligible regions (open protocol question)
    for loss in ("L_E", "L_S"):
        sub = cost[(cost.radius_km == MAIN_R) & (cost.dc_mw == MAIN_X) & cost.region.isin(eligible)]
        lu = sub[sub.method.eq("LU")].set_index("region")[loss]
        gnn = sub[sub.method.eq("GNN")].groupby("region")[loss].mean().reindex(lu.index)
        rows.append({"country": cc, "module": "connection_refscreen", "metric": loss, "matching": "", "radius_km": MAIN_R, "dc_mw": MAIN_X,
                     "in_family": False, "unit": UNITS[cc]["connection"], "regions": "|".join(lu.index),
                     "excluded": "|".join(sorted(set(cost.region) - set(eligible))), **paired(pi, lu.to_numpy(), gnn.to_numpy())})
    table = pd.DataFrame(rows)
    fam = table.in_family
    if fam.sum() != 4:
        raise ValueError("four-module family is not four rows")
    table["holm_p_4"] = None
    table.loc[fam, "holm_p_4"] = pi.holm(table.loc[fam, "p_raw"].to_numpy())
    table["holm_family"] = np.where(fam, f"{cc}_four_module", "")
    return table, pd.DataFrame(seed_rows)


# ----------------------------------------------------------------- RQ1 panel

def panel_stats(rank_agreement, pi, controls: pd.DataFrame, eligibility: pd.DataFrame, cc: str):
    eligible = eligibility.groupby("region").ref_eligible.first()
    per_region, summary = [], []
    for (radius, x), block in controls.groupby(["radius_km", "dc_mw"]):
        for region in sorted(block.region.unique()):
            g = block[block.region.eq(region)].set_index("distribution").reindex(PANEL)
            rec = {"country": cc, "region": region, "radius_km": radius, "dc_mw": x, "ref_eligible": bool(eligible[region])}
            for agg_col, tag in (("rho_agg", "agg"), ("rho_agg_vs_Aref", "aggAref")):
                for loss in ("L_E", "L_S"):
                    rec[f"assoc_{tag}_{loss}"] = None
            for loss in ("L_E", "L_S"):
                cell = g.rho_cell.to_numpy(float) if g.rho_cell.notna().all() else None
                rec[f"assoc_cell_{loss}"] = rank_agreement(cell, g[loss].to_numpy(float)) if cell is not None else None
                for agg_col, tag in (("rho_agg", "agg"), ("rho_agg_vs_Aref", "aggAref")):
                    agg = g[agg_col].to_numpy(float) if g[agg_col].notna().all() else None
                    rec[f"assoc_{tag}_{loss}"] = rank_agreement(agg, g[loss].to_numpy(float)) if agg is not None else None
                rec[f"loss_constant_{loss}"] = bool(np.ptp(g[loss].to_numpy(float)) == 0)
            per_region.append(rec)
    per_region = pd.DataFrame(per_region)
    for (radius, x), block in per_region.groupby(["radius_km", "dc_mw"]):
        el = block[block.ref_eligible]
        for tag in ("agg", "aggAref"):
            for loss in ("L_E", "L_S"):
                a, c = f"assoc_{tag}_{loss}", f"assoc_cell_{loss}"
                ok = el[el[a].notna() & el[c].notna()]
                d = (ok[a] - ok[c]).to_numpy(float)
                row = {"country": cc, "radius_km": radius, "dc_mw": x, "aggregate_target": "G" if tag == "agg" else "A_ref",
                       "loss": loss, "n_total": len(block), "n_eligible": len(el), "n_corr": len(ok),
                       "regions_corr": "|".join(ok.region), "regions_undefined": "|".join(sorted(set(el.region) - set(ok.region))),
                       "median_assoc_cell": float(ok[c].median()) if len(ok) else None,
                       "median_assoc_agg": float(ok[a].median()) if len(ok) else None,
                       "median_abs_assoc_cell": float(ok[c].abs().median()) if len(ok) else None,
                       "median_abs_assoc_agg": float(ok[a].abs().median()) if len(ok) else None,
                       "n_agg_more_negative": int((d < 0).sum()), "n_agg_stronger_abs": int((ok[a].abs() > ok[c].abs()).sum()),
                       "p_signflip_d": pi.sign_flip(d)["p"] if len(d) else None, "p_min": pi.sign_flip(d)["p_min"] if len(d) else None}
                summary.append(row)
    levels = (controls.merge(eligible.rename("ref_eligible"), left_on="region", right_index=True)
              .query("ref_eligible").groupby(["radius_km", "dc_mw", "distribution"])
              .agg(n_regions=("region", "nunique"), rho_cell_mean=("rho_cell", "mean"), rho_agg_mean=("rho_agg", "mean"),
                   rho_agg_vs_Aref_mean=("rho_agg_vs_Aref", "mean"), L_E_mean=("L_E", "mean"), L_S_mean=("L_S", "mean"))
              .reset_index())
    levels.insert(0, "country", cc)
    levels["order"] = levels.distribution.map({n: i for i, n in enumerate(PANEL)})
    return per_region, pd.DataFrame(summary), levels.sort_values(["radius_km", "dc_mw", "order"]).drop(columns="order")


# ----------------------------------------------------------------- scale curves

def scale_stats(pi, scale: pd.DataFrame, cc: str, centres: str, totals: pd.Series) -> pd.DataFrame:
    """Scale curves under D9: seed-average MAE within region, then treat each method's error <= 1e-9 * D_r as numerical
    zero. The percentage is undefined where LU is zero; wins/losses/ties and the sign-flip use the zeroed paired errors
    (LU zero with GNN positive is a loss for GNN, both zero is a tie). Raw errors stay in the frozen scale tables."""
    rows = []
    for (radius), block in scale.groupby("radius_km"):
        for target, col in (("ledger_G", "mae_decision"), ("VD_Ref", "mae_representation")):
            piv = block.groupby(["region", "method"])[col].mean().unstack()
            eps = ZERO_REL * totals.reindex(piv.index)
            lu = piv.LU.where(piv.LU.abs() > eps, 0.0)
            gnn = piv.GNN.where(piv.GNN.abs() > eps, 0.0)
            defined = lu != 0
            rel = (100 * (gnn - lu) / lu)[defined]
            d = (gnn - lu).to_numpy(float)
            flip = pi.sign_flip(d)
            rows.append({"country": cc, "centres": centres, "target": target, "radius_km": radius, "n": len(piv),
                         "n_pct_defined": int(defined.sum()), "regions_pct_undefined": "|".join(piv.index[~defined]),
                         "n_lu_zero": int((lu == 0).sum()), "n_gnn_zero": int((gnn == 0).sum()),
                         "median_rel_pct": float(np.median(rel)) if len(rel) else None,
                         "q25": float(np.percentile(rel, 25)) if len(rel) else None,
                         "q75": float(np.percentile(rel, 75)) if len(rel) else None,
                         "wins": int((d < 0).sum()), "losses": int((d > 0).sum()), "ties": int((d == 0).sum()), "p_signflip": flip["p"],
                         "zero_rule": "D9: error <= 1e-9 * D_r is numerical zero",
                         "zero_station_share_median": float(block.groupby("region").zero_station_share.first().median())})
    return pd.DataFrame(rows)


def voronoi_radius(backup: Path, cc: str) -> dict:
    import geopandas as gpd
    from scipy.spatial import cKDTree
    spec = {"uk": ("data/datasets/2_derived/uk/bplus/stations.gpkg", "EPSG:27700"),
            "au": ("data/datasets/2_derived/au/bplus/stations.gpkg", "EPSG:7856")}[cc]
    st = gpd.read_file(backup / "inputs/base" / spec[0]).to_crs(spec[1])
    st["station_id"] = st.station_id.astype(str)
    st = st.set_index("station_id")
    folder = backup / f"inputs/base/results/2_Generator/{COUNTRY_DIR[cc]}/static/assignments"
    nn, pooled = [], []
    for path in sorted(folder.glob("*.npz")):
        with np.load(path, allow_pickle=False) as vd:
            ids = vd["station_id"].astype(str)
        sub = st.loc[ids]
        xy = np.column_stack([sub.geometry.x, sub.geometry.y])
        d, _ = cKDTree(xy).query(xy, k=2)
        nn.append(np.median(d[:, 1] / 1000.0))
        pooled.extend(d[:, 1] / 1000.0)
    # D10: regions weigh equally (median over regions of the regional median), as in the reviewed manuscript
    return {"n_regions": len(nn), "n_stations": len(pooled), "median_of_regional_median_nn_km": float(np.median(nn)),
            "voronoi_equivalent_radius_km": float(np.median(nn) / 2),
            "pooled_station_radius_km_for_reference": float(np.median(pooled) / 2),
            "definition": "D10: median over regions of the within-region median nearest-neighbour station distance, halved (working CRS)"}


# ----------------------------------------------------------------- adequacy

def adequacy_stats(adq: pd.DataFrame, cc: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    realised = adq[adq.budget_realised]
    at = realised[realised.eta.eq(MAIN_ETA)]

    def add(stat, value, **kw):
        rows.append({"country": cc, "statistic": stat, "value": value, **kw})
    add("units_total", len(adq)); add("units_budget_realised", len(realised)); add("units_realised_eta0.9", len(at))
    add("violations_realised", int((~realised.valid_E | ~realised.valid_S).sum()))
    add("violations_not_realised", int((~adq[~adq.budget_realised].valid_E | ~adq[~adq.budget_realised].valid_S).sum()))
    for eta, g in adq.groupby("eta"):
        for method, gm in g.groupby("method"):
            add("alpha_obs", float(gm.budget_realised.mean()), eta=eta, method=method)
        add("alpha_obs", float(g.budget_realised.mean()), eta=eta, method="all")
    add("zero_bound_S_share", float(at.L_S_bound_zero.mean()), eta=MAIN_ETA, method="all")
    add("zero_regret_share", float(at.L_S_zero.mean()), eta=MAIN_ETA, method="all")
    add("zero_bound_S_share_all_eta", float(realised.L_S_bound_zero.mean()), method="all")
    add("zero_regret_share_all_eta", float(realised.L_S_zero.mean()), method="all")
    for x, g in at.groupby("dc_mw"):
        add("zero_bound_S_share", float(g.L_S_bound_zero.mean()), eta=MAIN_ETA, dc_mw=x, method="all")
        add("zero_regret_share", float(g.L_S_zero.mean()), eta=MAIN_ETA, dc_mw=x, method="all")
    for method, g in at.groupby("method"):
        add("zero_bound_S_share", float(g.L_S_bound_zero.mean()), eta=MAIN_ETA, method=method)
        add("zero_regret_share", float(g.L_S_zero.mean()), eta=MAIN_ETA, method=method)
        for loss, bound in (("L_E", "L_E_bound"), ("L_S", "L_S_bound")):
            pos = g[g[loss] > g.zero_tolerance]
            add(f"bound_over_actual_{loss}_median", float((pos[bound] / pos[loss]).median()) if len(pos) else None,
                eta=MAIN_ETA, method=method, n=len(pos))
    for eta, g in realised.groupby("eta"):
        add("median_slack_E", float(g.slack_E.median()), eta=eta, method="all")
        add("median_slack_S", float(g.slack_S.median()), eta=eta, method="all")
        for method, gm in g.groupby("method"):
            add("median_slack_S", float(gm.slack_S.median()), eta=eta, method=method)
            add("median_slack_E", float(gm.slack_E.median()), eta=eta, method=method)
    # tolerance pass-rate curves (Figure 5 source): P(bound <= tau | realised, eta=0.9) by method
    unit = 1e6 if cc == "uk" else 1.0
    taus = np.logspace(np.log10(0.3), np.log10(1000.0), 160) * unit
    curves = []
    for method, g in at.groupby("method"):
        for bound in ("L_S_bound", "L_E_bound"):
            v = g[bound].to_numpy(float)
            curves += [{"country": cc, "method": method, "bound": bound, "tau": float(t), "tau_display": float(t / unit),
                        "display_unit": "GBP_m" if cc == "uk" else "MVA", "n": len(v), "pass_rate": float((v <= t).mean())} for t in taus]
    return pd.DataFrame(rows), pd.DataFrame(curves)


# ----------------------------------------------------------------- driver

def main() -> None:
    here = Path(__file__).resolve()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backup", type=Path, default=here.parents[2] / "results" / "_backup")
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()
    backup = args.backup.resolve()
    root = backup / "recompute" / args.run_id
    if json.loads((root / "receipt.json").read_text(encoding="utf-8"))["status"] != "PASS":
        raise SystemExit("Step B receipt is not PASS")
    stats_dir, task_dir = root / "stats", root / "tasks"
    for d in (stats_dir, task_dir):
        if d.exists():
            raise SystemExit(f"{d} exists; statistics are written once per run")
        d.mkdir()
    pi, rank_agreement = upstream(backup)
    outputs, meta = {}, {"consumed": {}, "c4_crosscheck": {}, "voronoi": {}}

    def write(frame, path):
        frame.to_csv(path, index=False, float_format="%.17g", lineterminator="\n")
        outputs[path.relative_to(root).as_posix()] = {"sha256": sha256_file(path), "rows": len(frame)}

    modules, per_seed, panels, levels, scales, adequacy, curves, endpoints, t25s = [], [], [], [], [], [], [], [], []
    per_region_panels = []
    for cc in ("uk", "au"):
        seeds, region, info = export_tasks(backup, cc)
        meta["consumed"].update(info["consumed"]); meta["c4_crosscheck"][cc] = info["c4_crosscheck"]
        write(seeds, task_dir / f"{cc}_task_seed_values.csv")
        write(region, task_dir / f"{cc}_task_region_values.csv")
        cost = pd.read_csv(root / f"{cc}_cost_oof.csv")
        elig = pd.read_csv(root / f"{cc}_ref_eligibility.csv")
        table, seed_table = module_rows(pi, region, seeds, cost, cc, set(elig[elig.ref_eligible].region))
        modules.append(table); per_seed.append(seed_table)
        pr, ps, lv = panel_stats(rank_agreement, pi, pd.read_csv(root / f"{cc}_controls.csv"), elig, cc)
        per_region_panels.append(pr); panels.append(ps); levels.append(lv)
        for centres in ("station", "grid"):
            scales.append(scale_stats(pi, pd.read_csv(root / f"{cc}_scale_{centres}.csv"), cc, centres,
                                      pd.read_csv(root / f"{cc}_region_support.csv").set_index("region").source_total))
        meta["voronoi"][cc] = voronoi_radius(backup, cc)
        a, cv = adequacy_stats(pd.read_csv(root / f"{cc}_adequacy.csv"), cc)
        adequacy.append(a); curves.append(cv)
        # endpoint normalisation Pi (Ref-dependent: D2-eligible regions only)
        ok = set(elig[elig.ref_eligible].region)
        for (radius, x), g in cost[cost.region.isin(ok)].groupby(["radius_km", "dc_mw"]):
            for loss in ("L_E", "L_S"):
                m = g.groupby(["region", "method"])[loss].mean().unstack().mean()
                span = m["Uni"] - m["Ref"]
                endpoints.append({"country": cc, "radius_km": radius, "dc_mw": x, "loss": loss, "n_regions": g.region.nunique(),
                                  "Uni": m["Uni"], "LU": m["LU"], "GNN": m["GNN"], "Ref": m["Ref"],
                                  "endpoint_span_rel_to_Uni": span / m["Uni"] if m["Uni"] else None,
                                  # D13: Pi is reported only where the Uni-Ref span is well conditioned (not Australia)
                                  "pi_reported": cc == "uk",
                                  "Pi_LU_pct": 100 * (m["Uni"] - m["LU"]) / span if span else None,
                                  "Pi_GNN_pct": 100 * (m["Uni"] - m["GNN"]) / span if span else None})
        if cc == "uk":
            t25 = pd.read_csv(root / "uk_t25_sensitivity.csv")
            t25["method_seedavg"] = t25.method
            for method, g in t25.groupby("method"):
                gg = g.groupby("region")[["L_S_T9", "L_S_T25", "selected_overlap", "oracle_overlap", "oracle_shift_km"]].mean()
                t25s.append({"country": cc, "method": method, "n_regions": len(gg),
                             "regions_oracle_changed": int((gg.oracle_overlap < 20).sum()),
                             "regions_selection_changed": int((gg.selected_overlap < 20).sum()),
                             "regions_L_S_changed": int((~np.isclose(gg.L_S_T9, gg.L_S_T25, rtol=0, atol=1e-6)).sum()),
                             "median_L_S_T9": float(gg.L_S_T9.median()), "median_L_S_T25": float(gg.L_S_T25.median()),
                             "mean_L_S_T9": float(gg.L_S_T9.mean()), "mean_L_S_T25": float(gg.L_S_T25.mean()),
                             "max_oracle_shift_km": float(gg.oracle_shift_km.max())})

    modules = pd.concat(modules, ignore_index=True)
    write(modules, stats_dir / "module_contrasts.csv")
    write(modules[modules.in_family].reset_index(drop=True), stats_dir / "table3_four_module.csv")
    write(pd.concat(per_seed, ignore_index=True), stats_dir / "per_seed_contrasts.csv")
    write(pd.concat(per_region_panels, ignore_index=True), stats_dir / "panel_region_associations.csv")
    write(pd.concat(panels, ignore_index=True), stats_dir / "panel_summary.csv")
    write(pd.concat(levels, ignore_index=True), stats_dir / "panel_levels.csv")
    write(pd.concat(scales, ignore_index=True), stats_dir / "scale_curves.csv")
    write(pd.concat(adequacy, ignore_index=True), stats_dir / "adequacy_summary.csv")
    write(pd.concat(curves, ignore_index=True), stats_dir / "adequacy_pass_curves.csv")
    write(pd.DataFrame(endpoints), stats_dir / "connection_endpoints.csv")
    write(pd.DataFrame(t25s), stats_dir / "t25_summary.csv")

    # Holm disclosure: four-module family here vs the upstream nine-member C4 family
    disclosure = []
    for cc in ("uk", "au"):
        c4 = pd.read_csv(backup / "inputs/release_r2/4_Analysis" / COUNTRY_DIR[cc] / "C4/contrasts.csv", float_precision="round_trip")
        mine = modules[(modules.country == cc) & modules.in_family]
        for r in mine.itertuples():
            coord = r.module if r.module != "connection" else None
            up = c4[c4.task_coordinate.eq(coord) & c4.metric.eq(r.metric if r.module != "sizing" else "RSD_median")] if coord else c4.iloc[0:0]
            disclosure.append({"country": cc, "module": r.module, "metric": r.metric, "matching": r.matching, "n": r.n,
                               "p_raw_this_paper": r.p_raw, "holm_p_four_module": r.holm_p_4,
                               "upstream_c4_family_size": int(up.family_size.iloc[0]) if len(up) else None,
                               "upstream_c4_contrast": up.contrast_id.iloc[0] if len(up) else "not in C4 (fixed-X connection is paper-specific)",
                               "upstream_p_raw": float(up.p.iloc[0]) if len(up) else None,
                               "upstream_holm_p_nine": float(up.holm_p.iloc[0]) if len(up) else None,
                               "upstream_members": "|".join(c4.contrast_id + ":" + c4.task_coordinate + ":" + c4.metric)})
    write(pd.DataFrame(disclosure), stats_dir / "holm_family_disclosure.csv")
    meta["protocol"] = {"bootstrap": {"B": BOOT_B, "seed": BOOT_SEED, "rng": "numpy PCG64 (upstream paired_inference._bootstrap)"},
                        "sign_flip": "upstream paired_inference.sign_flip, exact 2^n, two-sided, tolerance 1e-12*max(1,sum|d|)",
                        "holm": "upstream paired_inference.holm over the four family rows per country",
                        "seed_aggregation": "mean over GNN seeds within region, then paired with LU",
                        "panel": "Spearman (average ranks) across the nine distributions within each Ref-eligible region; n_corr counts regions where both associations are defined",
                        "bound_over_actual": "median over realised eta=0.9 units with loss > zero tolerance",
                        "main_setting": {"radius_km": MAIN_R, "dc_mw": MAIN_X, "eta": MAIN_ETA}}
    (stats_dir / "stats_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    outputs["stats/stats_meta.json"] = {"sha256": sha256_file(stats_dir / "stats_meta.json")}
    receipt = {"schema": "paper2_statistics_v1", "run_id": args.run_id, "completed_utc": datetime.now(timezone.utc).isoformat(),
               "script_sha256": sha256_file(here), "step_b_receipt_sha256": sha256_file(root / "receipt.json"), "outputs": outputs}
    (stats_dir / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(meta["c4_crosscheck"], indent=1))
    print(modules[modules.in_family][["country", "module", "metric", "n", "lu_mean", "gnn_mean", "median_rel_pct", "wins", "p_raw", "holm_p_4", "rel_effect_pct", "rel_ci_lo", "rel_ci_hi"]].to_string())


if __name__ == "__main__":
    main()
