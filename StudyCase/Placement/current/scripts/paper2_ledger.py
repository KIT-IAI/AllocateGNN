"""Claim ledger for the margin-criterion revision (plan section 9).

Every reportable number becomes one row: ``claim_id`` bound to its run file, the
row filter, the statistic column, the full-precision value, unit, denominator and
display format. The ledger is a pure function of a run root, so running it on
``frozen/<run_id>`` reproduces the frozen ledger byte for byte:

    python scripts/paper2_ledger.py --root <.../frozen/<run_id>> --out <dir>

Display conventions: ``display_scale`` multiplies the stored value before
printing (1e-6 for GBP -> GBP m, 100 for shares -> percent); the checker compares
a printed token with ``abs(value * display_scale)`` at the printed precision.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import pandas as pd

COLUMNS = ["claim_id", "file", "filter", "statistic", "value", "unit", "denominator", "display_format",
           "display_scale", "display_example", "manuscript_location", "note"]


class Ledger:
    def __init__(self, root: Path):
        self.root, self.rows, self.ids = root, [], set()

    def add(self, claim_id, file, filt, statistic, value, unit, denominator="", fmt="{:.2f}", scale=1.0, note=""):
        if claim_id in self.ids:
            raise ValueError(f"duplicate claim {claim_id}")
        self.ids.add(claim_id)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            text, example = "", "undefined"
        else:
            value = float(value)
            integer = fmt.endswith("d}")
            if integer and not value.is_integer():
                raise ValueError(f"{claim_id}: integer display for a non-integer value")
            text = str(int(value)) if integer else repr(value)
            example = fmt.format(int(value) if integer else value * scale)
        self.rows.append({"claim_id": claim_id, "file": file, "filter": filt, "statistic": statistic, "value": text,
                          "unit": unit, "denominator": denominator, "display_format": fmt, "display_scale": repr(float(scale)),
                          "display_example": example, "manuscript_location": "", "note": note})

    def read(self, rel):
        return pd.read_csv(self.root / rel, keep_default_na=True)


def q(**kw) -> str:
    return " & ".join(f"{k}=={v!r}" for k, v in kw.items())


def build(root: Path) -> list[dict]:
    L = Ledger(root)
    money = {"uk": ("GBP", 1e-6, "£m"), "au": ("MVA_equivalent_PF1", 1.0, "MVA")}
    label = {"uk": "GB", "au": "AU"}

    # ---- case facts
    for cc in ("uk", "au"):
        sup = L.read(f"{cc}_region_support.csv")
        f = f"{cc}_region_support.csv"
        L.add(f"DATA.{label[cc]}.n_regions", f, "all rows", "count(region)", len(sup), "regions", fmt="{:d}")
        L.add(f"DATA.{label[cc]}.n_stations", f, "all rows", "sum(n_stations)", int(sup.n_stations.sum()), "stations", fmt="{:,d}")
        L.add(f"DATA.{label[cc]}.n_grid", f, "all rows", "sum(n_grid)", int(sup.n_grid.sum()), "cells", fmt="{:,d}")
        for stat in ("min", "median", "max"):
            L.add(f"DATA.{label[cc]}.grid_step_m.{stat}", f, "all rows", f"{stat}(grid_step_m)", float(getattr(sup.grid_step_m, stat)()),
                  "m", fmt="{:.0f}", note="region-specific target ground step (bounded equal-area budget)")
        for r in sup.itertuples():
            L.add(f"DATA.{label[cc]}.grid_step_m.{r.region}", f, q(region=r.region), "grid_step_m", r.grid_step_m, "m", fmt="{:.0f}")
            L.add(f"DATA.{label[cc]}.n_stations.{r.region}", f, q(region=r.region), "n_stations", r.n_stations, "stations", fmt="{:d}")
        L.add(f"DATA.{label[cc]}.source_total", f, "all rows", "sum(source_total)", float(sup.source_total.sum()),
              "MVA" if cc == "uk" else "MW", fmt="{:,.0f}")
        L.add(f"REF.{label[cc]}.n_eligible", f, "ref_eligible", "count", int(sup.ref_eligible.sum()), "regions",
              denominator=f"{len(sup)} regions", fmt="{:d}", note="D2: max over R in {10,20} km of mean|A_ref-G|/median G <= 0.25")
        for r in sup.itertuples():
            L.add(f"REF.{label[cc]}.q_r.{r.region}", f, q(region=r.region), "ref_q_r", r.ref_q_r, "ratio", fmt="{:.3f}",
                  note=r.ref_reason if isinstance(r.ref_reason, str) else "")
    cfg = json.loads((root / "config.json").read_text(encoding="utf-8"))
    L.add("COST.annuity_factor", "config.json", "annuity_factor_full_precision", "value", cfg["annuity_factor_full_precision"],
          "years-equivalent", fmt="{:.2f}", note="3.5 %, 20 years; 14.21 is display only")
    tm = L.read("uk_tariff_mapping.csv")
    L.add("COST.GB.tariff_candidates", "uk_tariff_mapping.csv", "all rows", "count", len(tm), "evaluation positions", fmt="{:,d}")
    L.add("COST.GB.tariff_nearest_polygon", "uk_tariff_mapping.csv", "match_rule=='nearest_polygon'", "count",
          int(tm.match_rule.eq("nearest_polygon").sum()), "evaluation positions", fmt="{:d}")
    L.add("COST.GB.tariff_multiple", "uk_tariff_mapping.csv", "n_zone_matches>1", "count", int((tm.n_zone_matches > 1).sum()),
          "evaluation positions", fmt="{:d}")

    # ---- Table 3 and all module contrasts
    mc = L.read("stats/module_contrasts.csv")
    for r in mc.itertuples():
        cc = r.country
        tag = f"{r.module}.{r.metric}" + (f".{r.matching}" if isinstance(r.matching, str) and r.matching else "")
        if r.module.startswith("connection"):
            tag += f".R{r.radius_km:g}.X{r.dc_mw:g}"
        base = f"{'T3' if r.in_family else 'MOD'}.{label[cc]}.{tag}"
        filt = q(country=cc, module=r.module, metric=r.metric) + (f" & matching=={r.matching!r}" if isinstance(r.matching, str) and r.matching else "") \
            + (f" & radius_km=={r.radius_km} & dc_mw=={r.dc_mw}" if r.module.startswith("connection") else "")
        unit = r.unit
        scale, fmt = (money[cc][1], "{:.3f}") if r.module.startswith("connection") else (1.0, "{:.3f}")
        den = f"{r.n} regions"
        L.add(f"{base}.lu_mean", "stats/module_contrasts.csv", filt, "lu_mean", r.lu_mean, unit, den, fmt, scale)
        L.add(f"{base}.gnn_mean", "stats/module_contrasts.csv", filt, "gnn_mean", r.gnn_mean, unit, den, fmt, scale)
        L.add(f"{base}.median_rel_pct", "stats/module_contrasts.csv", filt, "median_rel_pct", r.median_rel_pct, "%",
              f"{r.n_pct_defined} regions with nonzero LU", "{:+.2f}")
        L.add(f"{base}.wins", "stats/module_contrasts.csv", filt, "wins", r.wins, "regions", den, "{:d}")
        L.add(f"{base}.n", "stats/module_contrasts.csv", filt, "n", r.n, "regions", "", "{:d}")
        L.add(f"{base}.p_raw", "stats/module_contrasts.csv", filt, "p_raw", r.p_raw, "p", den, "{:.3g}")
        L.add(f"{base}.rel_effect_pct", "stats/module_contrasts.csv", filt, "rel_effect_pct", r.rel_effect_pct, "%", den, "{:+.2f}")
        L.add(f"{base}.rel_ci_lo", "stats/module_contrasts.csv", filt, "rel_ci_lo", r.rel_ci_lo, "%", "B=10,000 region bootstrap", "{:+.1f}")
        L.add(f"{base}.rel_ci_hi", "stats/module_contrasts.csv", filt, "rel_ci_hi", r.rel_ci_hi, "%", "B=10,000 region bootstrap", "{:+.1f}")
        if r.in_family:
            L.add(f"{base}.holm_p_4", "stats/module_contrasts.csv", filt, "holm_p_4", r.holm_p_4, "p", f"four-module family, {label[cc]}", "{:.3g}")
    ps = L.read("stats/per_seed_contrasts.csv")
    ps = ps[ps.in_family_module]
    for r in ps.itertuples():
        tag = f"{r.module}.{r.metric}" + (f".{r.matching}" if isinstance(r.matching, str) and r.matching else "")
        filt = q(country=r.country, module=r.module, metric=r.metric, seed=r.seed) + (" & in_family_module" )
        for stat, fmt, unit in (("median_rel_pct", "{:+.2f}", "%"), ("wins", "{:d}", "regions"), ("p_raw", "{:.3g}", "p")):
            L.add(f"SEED.{label[r.country]}.{tag}.s{r.seed}.{stat}", "stats/per_seed_contrasts.csv", filt, stat, getattr(r, stat), unit,
                  f"{r.n} regions", fmt)

    # ---- Holm disclosure
    hd = L.read("stats/holm_family_disclosure.csv")
    for r in hd.itertuples():
        base = f"HOLM.{label[r.country]}.{r.module}"
        filt = q(country=r.country, module=r.module)
        L.add(f"{base}.p_raw", "stats/holm_family_disclosure.csv", filt, "p_raw_this_paper", r.p_raw_this_paper, "p", f"{r.n} regions", "{:.6g}")
        L.add(f"{base}.holm_4", "stats/holm_family_disclosure.csv", filt, "holm_p_four_module", r.holm_p_four_module, "p", "four-module family", "{:.6g}")
        if pd.notna(r.upstream_holm_p_nine):
            L.add(f"{base}.holm_9_upstream", "stats/holm_family_disclosure.csv", filt, "upstream_holm_p_nine", r.upstream_holm_p_nine, "p",
                  f"upstream C4 family of {int(r.upstream_c4_family_size)}", "{:.6g}")

    # ---- endpoints
    ep = L.read("stats/connection_endpoints.csv")
    for r in ep[(ep.radius_km == 10) & (ep.dc_mw == 300)].itertuples():
        cc = r.country
        filt = q(country=cc, radius_km=10.0, dc_mw=300.0, loss=r.loss)
        for col in ("Uni", "LU", "GNN", "Ref"):
            L.add(f"END.{label[cc]}.{r.loss}.{col}", "stats/connection_endpoints.csv", filt, col, getattr(r, col), money[cc][0],
                  f"{r.n_regions} Ref-eligible regions", "{:.3f}", money[cc][1])
        for col in (("Pi_LU_pct", "Pi_GNN_pct", "endpoint_span_rel_to_Uni") if r.pi_reported else ("endpoint_span_rel_to_Uni",)):
            L.add(f"END.{label[cc]}.{r.loss}.{col}", "stats/connection_endpoints.csv", filt, col, getattr(r, col), "%" if col != "endpoint_span_rel_to_Uni" else "ratio",
                  f"{r.n_regions} Ref-eligible regions", "{:.1f}" if col != "endpoint_span_rel_to_Uni" else "{:.3f}")

    # ---- RQ1 panel
    pl = L.read("stats/panel_levels.csv")
    for r in pl[(pl.dc_mw == 300)].itertuples():
        cc = r.country
        filt = q(country=cc, radius_km=r.radius_km, dc_mw=300.0, distribution=r.distribution)
        d = r.distribution.replace("/", "_")
        for col, unit, fmt, scale in (("rho_cell_mean", "rho", "{:.3f}", 1.0), ("rho_agg_mean", "rho", "{:.3f}", 1.0),
                                      ("L_E_mean", money[cc][0], "{:.3f}", money[cc][1]), ("L_S_mean", money[cc][0], "{:.3f}", money[cc][1])):
            L.add(f"PANEL.{label[cc]}.R{r.radius_km:g}.{d}.{col}", "stats/panel_levels.csv", filt, col, getattr(r, col), unit,
                  f"{r.n_regions} Ref-eligible regions", fmt, scale)
    pm = L.read("stats/panel_summary.csv")
    for r in pm[pm.dc_mw == 300].itertuples():
        cc = r.country
        base = f"PANEL.{label[cc]}.R{r.radius_km:g}.{r.aggregate_target}.{r.loss}"
        filt = q(country=cc, radius_km=r.radius_km, dc_mw=300.0, aggregate_target=r.aggregate_target, loss=r.loss)
        for col, unit, fmt in (("n_total", "regions", "{:d}"), ("n_eligible", "regions", "{:d}"), ("n_corr", "regions", "{:d}"),
                               ("median_assoc_cell", "rho", "{:.3f}"), ("median_assoc_agg", "rho", "{:.3f}"),
                               ("median_abs_assoc_cell", "rho", "{:.3f}"), ("median_abs_assoc_agg", "rho", "{:.3f}"),
                               ("n_agg_more_negative", "regions", "{:d}"), ("n_agg_stronger_abs", "regions", "{:d}"),
                               ("p_signflip_d", "p", "{:.3g}")):
            den = f"n_corr={r.n_corr}" if col not in ("n_total", "n_eligible", "n_corr") else ""
            L.add(f"{base}.{col}", "stats/panel_summary.csv", filt, col, getattr(r, col), unit, den, fmt)

    # ---- scale curves (Figure 4)
    sc = L.read("stats/scale_curves.csv")
    for r in sc[sc.centres == "station"].itertuples():
        base = f"SCALE.{label[r.country]}.{r.target}.R{r.radius_km:g}"
        filt = q(country=r.country, centres="station", target=r.target, radius_km=r.radius_km)
        for col, unit, fmt in (("median_rel_pct", "%", "{:+.2f}"), ("q25", "%", "{:+.1f}"), ("q75", "%", "{:+.1f}"),
                               ("wins", "regions", "{:d}"), ("losses", "regions", "{:d}"), ("ties", "regions", "{:d}"),
                               ("n_pct_defined", "regions", "{:d}"), ("p_signflip", "p", "{:.3g}")):
            L.add(f"{base}.{col}", "stats/scale_curves.csv", filt, col, getattr(r, col), unit, f"{r.n} regions", fmt)
    meta = json.loads((root / "stats/stats_meta.json").read_text(encoding="utf-8"))
    for cc, v in meta["voronoi"].items():
        L.add(f"SCALE.{label[cc]}.voronoi_radius_km", "stats/stats_meta.json", f"voronoi.{cc}", "voronoi_equivalent_radius_km",
              v["voronoi_equivalent_radius_km"], "km", f"{v['n_stations']} stations", "{:.2f}", note=v["definition"])

    # ---- adequacy (Figure 5, RQ3)
    ad = L.read("stats/adequacy_summary.csv")
    for r in ad.itertuples():
        cc = r.country
        parts = [r.statistic]
        if pd.notna(r.eta):
            parts.append(f"eta{r.eta:g}")
        if isinstance(r.method, str):
            parts.append(r.method)
        if pd.notna(r.dc_mw):
            parts.append(f"X{r.dc_mw:g}")
        cid = f"ADQ.{label[cc]}." + ".".join(parts)
        filt = q(country=cc, statistic=r.statistic) + (f" & eta=={r.eta}" if pd.notna(r.eta) else "") + \
            (f" & method=={r.method!r}" if isinstance(r.method, str) else "") + (f" & dc_mw=={r.dc_mw}" if pd.notna(r.dc_mw) else "")
        if r.statistic.startswith(("units", "violations")):
            unit, fmt, scale = "evaluations", "{:,d}", 1.0
        elif "share" in r.statistic or r.statistic == "alpha_obs":
            unit, fmt, scale = "share", "{:.3f}", 1.0
        elif r.statistic.startswith("bound_over_actual"):
            unit, fmt, scale = "ratio", "{:.1f}", 1.0
        else:
            unit, fmt, scale = money[cc][0], "{:.2f}", money[cc][1]
        den = "" if r.statistic.startswith("units") else ("realised budget evaluations" if "share" in r.statistic or "slack" in r.statistic
                                                           else f"n={int(r.n)}" if pd.notna(r.n) else "")
        L.add(cid, "stats/adequacy_summary.csv", filt, "value", r.value, unit, den, fmt, scale)

    # ---- T25 single-setting sensitivity
    t25 = L.read("stats/t25_summary.csv")
    for r in t25.itertuples():
        filt = q(country="uk", method=r.method)
        for col, unit, fmt, scale in (("regions_oracle_changed", "regions", "{:d}", 1.0), ("regions_selection_changed", "regions", "{:d}", 1.0),
                                      ("regions_L_S_changed", "regions", "{:d}", 1.0), ("median_L_S_T9", "GBP", "{:.3f}", 1e-6),
                                      ("median_L_S_T25", "GBP", "{:.3f}", 1e-6), ("max_oracle_shift_km", "km", "{:.1f}", 1.0)):
            L.add(f"T25.GB.{r.method}.{col}", "stats/t25_summary.csv", filt, col, getattr(r, col), unit,
                  f"{r.n_regions} regions, R=10 km, X=300 MW", fmt, scale)
    return L.rows


def main() -> None:
    here = Path(__file__).resolve()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True, help="run root (recompute/<run_id> or frozen/<run_id>)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows = build(args.root.resolve())
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "claim_ledger.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    print(f"{len(rows)} claims -> {args.out / 'claim_ledger.csv'}")


if __name__ == "__main__":
    main()
