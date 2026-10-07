"""Substation dataset facts quoted in the manuscript (the dataset table ``tab:cases`` and the text of the experimental
setup and of the Australian substation dataset appendix ``app:australia``), generated from the frozen inputs.

The claim ledger covers the evaluation outputs; this script covers the descriptive facts of the two
substation datasets that the manuscript describes (capacity totals, spare capacity, zero available
capacity). It reads only hash-frozen inputs under ``results/_backup/inputs/base`` and the frozen run's
region-support table, and writes ``figures_new/register_facts.json`` (values at full precision, with
their source files and SHA-256). ``check_manuscript_numbers.py`` accepts these values as generated facts.

    python scripts/paper2_register_facts.py --run-id fixedload_20260925_r2
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

X_MW = 300.0  # reference incoming load; PF = 1 for the MVA comparison


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    here = Path(__file__).resolve()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--backup", type=Path, default=here.parents[2] / "results" / "_backup")
    ap.add_argument("--out", type=Path, default=here.parents[1] / "figures_new" / "register_facts.json")
    args = ap.parse_args()
    backup = args.backup.resolve()
    frozen = backup / "frozen" / args.run_id
    base = backup / "inputs/base"
    used: dict[str, str] = {}

    def use(path: Path) -> Path:
        used[path.relative_to(backup).as_posix()] = sha(path)
        return path

    facts: dict[str, dict] = {}

    # Great Britain: full dataset and the stations of the 16 analysis regions
    uk = gpd.read_file(use(base / "data/datasets/2_derived/uk/bplus/stations.gpkg"))
    uk["station_id"] = uk.station_id.astype(str)
    spare = uk["Firm Capacity (MVA)"] - uk["Demand (MVA)"]
    support = pd.read_csv(use(frozen / "uk_region_support.csv"))
    ids = []
    for region in support.region:
        with np.load(use(base / f"results/2_Generator/1_UK/static/assignments/{region}.npz"), allow_pickle=False) as z:
            ids.extend(z["station_id"].astype(str).tolist())
    if len(ids) != len(set(ids)) or len(ids) != int(support.n_stations.sum()):
        raise ValueError("analysis-region station sets overlap or disagree with the support table")
    sub = uk.set_index("station_id").loc[ids]
    facts["GB"] = {
        "register_stations": int(len(uk)),
        "register_median_spare_mva": float(spare.median()),
        "register_stations_spare_ge_X": int((spare >= X_MW).sum()),
        "analysis_stations": int(len(sub)),
        "analysis_share_of_register_pct": 100.0 * len(sub) / len(uk),
        "analysis_peak_mva": float(sub["Demand (MVA)"].sum()),
        "analysis_firm_mva": float(sub["Firm Capacity (MVA)"].sum()),
        "analysis_peak_to_firm_pct": 100.0 * float(sub["Demand (MVA)"].sum() / sub["Firm Capacity (MVA)"].sum()),
    }

    # Australia: combined dataset (peak in MW, N-1 available capacity A in MVA, firm = peak + A at PF = 1)
    au = gpd.read_file(use(base / "data/datasets/2_derived/au/bplus/stations.gpkg"))
    if not np.allclose(au.F_mva, au.G_fy2024_mw + au.A_mva, rtol=0, atol=1e-6):
        raise ValueError("AU firm capacity is not peak + available capacity")
    facts["AU"] = {
        "register_stations": int(len(au)),
        "peak_field": "G_fy2024_mw (financial year 2024)",
        "stations_zero_available": int((au.A_mva == 0).sum()),
        "median_available_mva": float(au.A_mva.median()),
        "stations_available_ge_X": int((au.A_mva >= X_MW).sum()),
        "analysis_peak_mw": float(au.G_fy2024_mw.sum()),
        "analysis_firm_mva_equivalent": float(au.F_mva.sum()),
        "analysis_peak_to_firm_pct": 100.0 * float(au.G_fy2024_mw.sum() / au.F_mva.sum()),
    }
    for cc, key in (("uk", "GB"), ("au", "AU")):
        sup = pd.read_csv(use(frozen / f"{cc}_region_support.csv"))
        facts[key].update({"source_areas_total": int(sup.n_sources.sum()), "source_areas_min": int(sup.n_sources.min()),
                           "source_areas_max": int(sup.n_sources.max()), "cells_per_region_min": int(sup.n_grid.min()),
                           "cells_per_region_max": int(sup.n_grid.max())})
    # D12: evaluation positions outside every DNO polygon take the nearest tariff zone
    tm = pd.read_csv(use(frozen / "uk_tariff_mapping.csv"))
    near = tm[tm.match_rule == "nearest_polygon"].sort_values("candidate_id")
    facts["GB"]["tariff_nearest_positions"] = [
        {"region": r.region, "candidate_id": int(r.candidate_id), "zone": r.tnuos_zone, "distance_m": float(r.nearest_distance_m),
         "t9_gbp_per_kw": float(r.t9_gbp_per_kw)} for r in near.itertuples()]
    record = {"run_id": args.run_id, "script_sha256": sha(here), "X_mw": X_MW, "facts": facts,
              "inputs_sha256": dict(sorted(used.items()))}
    args.out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(facts, indent=1))


if __name__ == "__main__":
    main()
