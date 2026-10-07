"""Publish FY2024 AU demand onto the frozen AU12/SA3-34 geography.

The spatial aggregation functions are pure.  ``publish_au_fy2024_demand`` is
the sole filesystem boundary and receives the repository and both pipeline
configurations explicitly, so it can be called directly from a notebook.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import geopandas as gpd
import pandas as pd

from sglib.core.infra.hashing import sha256_file
from sglib.core.infra.artifacts import atomic_text

from ...config import CountryPipelineContext


EXPECTED_N_SA3 = 34
EXPECTED_N_SA4 = 12
EXPECTED_VINTAGE = "FY2024"


def fixed_source_order(country_config: Mapping[str, Any]) -> list[str]:
    """Return and validate the frozen SA3-34 order from one AU config."""

    country = country_config.get("country", {})
    if not isinstance(country, Mapping) or country.get("code") != "au":
        raise ValueError("FY2024 demand publication requires country='au'")
    order = [
        str(key)
        for region in country_config.get("regions", {}).get("items", [])
        for key in region.get("source_key_order", [])
    ]
    if len(order) != EXPECTED_N_SA3 or len(set(order)) != EXPECTED_N_SA3:
        raise RuntimeError("AU country config is not the frozen exact SA3-34 universe")
    return order


def aggregate_fy2024_demand(
    regions: gpd.GeoDataFrame,
    stations: pd.DataFrame,
    source_order: list[str],
) -> tuple[gpd.GeoDataFrame, dict]:
    """Aggregate usable stations while reporting, not folding, out-of-scope rows."""

    required_region = {"sa3_code", "loc_key", "geometry"}
    required_station = {
        "station",
        "status",
        "lon_wgs84",
        "lat_wgs84",
        "peak_mw",
        "energy_gwh",
    }
    if not required_region <= set(regions):
        raise ValueError(f"regions missing columns: {sorted(required_region - set(regions))}")
    if not required_station <= set(stations):
        raise ValueError(f"stations missing columns: {sorted(required_station - set(stations))}")
    if regions.crs is None:
        raise ValueError("regions CRS is missing")
    if len(regions) != EXPECTED_N_SA3 or regions["loc_key"].nunique() != EXPECTED_N_SA4:
        raise ValueError("regions are not the frozen AU12/SA3-34 geography")
    observed_order = regions["sa3_code"].astype(str).tolist()
    if set(observed_order) != set(source_order):
        raise ValueError("regions and frozen country-config source keys differ")

    usable = stations.loc[stations["status"].eq("usable")].copy()
    numeric = ["lon_wgs84", "lat_wgs84", "peak_mw", "energy_gwh"]
    if usable.empty or usable[numeric].isna().any().any():
        raise ValueError("FY2024 usable station rows are empty or incomplete")
    if (usable[["peak_mw", "energy_gwh"]] <= 0).any().any():
        raise ValueError("FY2024 usable station demand must be positive")

    points = gpd.GeoDataFrame(
        usable,
        geometry=gpd.points_from_xy(usable["lon_wgs84"], usable["lat_wgs84"]),
        crs="EPSG:4326",
    )
    scope = regions[["sa3_code", "loc_key", "geometry"]].to_crs("EPSG:4326")
    joined = gpd.sjoin(points, scope, how="left", predicate="within")
    if len(joined) != len(points):
        raise RuntimeError("FY2024 station spatial join produced duplicate rows")

    inside = joined.loc[joined["sa3_code"].notna()].copy()
    outside = joined.loc[joined["sa3_code"].isna()].copy()
    grouped = inside.groupby(inside["sa3_code"].astype(str), sort=False).agg(
        n_stations_usable=("station", "size"),
        demand_peak_mw=("peak_mw", "sum"),
        demand_energy_gwh=("energy_gwh", "sum"),
    )
    missing = sorted(set(source_order) - set(grouped.index))
    if missing:
        raise RuntimeError(f"FY2024 fixed scope has source regions without stations: {missing}")

    updated = regions.copy()
    keys = updated["sa3_code"].astype(str)
    for column in ("n_stations_usable", "demand_peak_mw", "demand_energy_gwh"):
        updated[column] = keys.map(grouped[column])
    updated["demand_vintage"] = EXPECTED_VINTAGE
    updated["station_vintage"] = EXPECTED_VINTAGE

    outside_rows = [
        {
            "station": str(row.station),
            "peak_mw": float(row.peak_mw),
            "energy_gwh": float(row.energy_gwh),
        }
        for row in outside.sort_values("station").itertuples()
    ]
    return updated, {
        "schema": "au_fy2024_region_demand_report_v1",
        "vintage": EXPECTED_VINTAGE,
        "spatial_scope": "frozen_AU12_SA3_34",
        "n_sa4": int(updated["loc_key"].nunique()),
        "n_sa3": int(len(updated)),
        "n_usable_stations_all": int(len(usable)),
        "n_usable_stations_in_scope": int(len(inside)),
        "n_usable_stations_outside_scope": int(len(outside)),
        "outside_scope_stations": outside_rows,
        "sum_peak_mw_in_scope": float(inside["peak_mw"].sum()),
        "sum_energy_gwh_in_scope": float(inside["energy_gwh"].sum()),
        "source_key_order": list(source_order),
    }


def update_dependent_region_tables(
    updated_sa3: gpd.GeoDataFrame,
    attributes: pd.DataFrame,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """Return FY2024-consistent SA4 geometry and non-spatial SA3 attributes."""

    required_sa3 = {
        "sa3_code",
        "sa4_code",
        "sa4_name",
        "loc_key",
        "n_stations_usable",
        "demand_peak_mw",
        "demand_energy_gwh",
        "population_erp_2009",
        "demand_vintage",
        "station_vintage",
        "geometry",
    }
    missing = required_sa3 - set(updated_sa3)
    if missing:
        raise ValueError(f"updated_sa3 missing columns: {sorted(missing)}")
    if "sa3_code" not in attributes:
        raise ValueError("attributes missing column: sa3_code")
    if attributes["sa3_code"].astype(str).duplicated().any():
        raise ValueError("attributes.sa3_code must be unique")

    sa4 = (
        updated_sa3.dissolve(
            by="sa4_code",
            aggfunc={
                "sa4_name": "first",
                "loc_key": "first",
                "n_stations_usable": "sum",
                "demand_peak_mw": "sum",
                "demand_energy_gwh": "sum",
                "population_erp_2009": "sum",
                "sa3_code": "count",
                "demand_vintage": "first",
                "station_vintage": "first",
            },
        )
        .rename(columns={"sa3_code": "n_sa3"})
        .reset_index()
    )
    if len(sa4) != EXPECTED_N_SA4:
        raise RuntimeError(f"FY2024 SA4 result must contain {EXPECTED_N_SA4} rows")

    updated_attributes = attributes.copy()
    by_code = updated_sa3.set_index(updated_sa3["sa3_code"].astype(str))
    keys = updated_attributes["sa3_code"].astype(str)
    if set(keys) != set(by_code.index):
        raise ValueError("attributes and updated_sa3 source keys differ")
    for column in ("n_stations_usable", "demand_peak_mw", "demand_energy_gwh"):
        updated_attributes[column] = keys.map(by_code[column])
    updated_attributes["demand_vintage"] = EXPECTED_VINTAGE
    updated_attributes["station_vintage"] = EXPECTED_VINTAGE
    return sa4, updated_attributes


def publish_au_fy2024_demand(
    *,
    context: CountryPipelineContext,
) -> dict:
    """Atomically publish FY2024 demand to all dependent AU region artifacts."""

    if context.country_code != "au":
        raise ValueError("publish_au_fy2024_demand only accepts the AU country config")
    if context.merged["temporal"].get("protocol") != "FY2024_same_year":
        raise ValueError("AU country config must freeze temporal_protocol=FY2024_same_year")

    derived = context.derived_root
    regions_path = derived / "regions_sa3.gpkg"
    regions_sa4_path = derived / "regions_sa4.gpkg"
    attributes_path = derived / "region_attributes.csv"
    stations_path = derived / "ledger" / "station_table_fy2024.csv"
    report_path = derived / "ledger" / "fy2024_region_demand_report.json"
    required = (regions_path, regions_sa4_path, attributes_path, stations_path)
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"AU FY2024 publication inputs missing: {missing}")

    before_sha = {
        "regions_sa3": sha256_file(regions_path),
        "regions_sa4": sha256_file(regions_sa4_path),
        "region_attributes": sha256_file(attributes_path),
    }
    stations_sha = sha256_file(stations_path)
    regions = gpd.read_file(regions_path, layer="regions_sa3")
    stations = pd.read_csv(stations_path, encoding="utf-8-sig")
    attributes = pd.read_csv(attributes_path, encoding="utf-8-sig")
    updated, report = aggregate_fy2024_demand(
        regions,
        stations,
        fixed_source_order(context.merged),
    )
    updated_sa4, updated_attributes = update_dependent_region_tables(updated, attributes)

    region_part = regions_path.with_name(f".{regions_path.name}.fy2024.part.gpkg")
    region_part.unlink(missing_ok=True)
    updated.to_file(region_part, layer="regions_sa3", driver="GPKG", index=False)
    if region_part.stat().st_size == 0:
        raise RuntimeError("FY2024 region GeoPackage staging file is empty")
    region_part.replace(regions_path)

    sa4_part = regions_sa4_path.with_name(f".{regions_sa4_path.name}.fy2024.part.gpkg")
    sa4_part.unlink(missing_ok=True)
    updated_sa4.to_file(sa4_part, layer="regions_sa4", driver="GPKG", index=False)
    sa4_part.replace(regions_sa4_path)

    attrs_part = attributes_path.with_name(f".{attributes_path.name}.fy2024.part.csv")
    updated_attributes.to_csv(attrs_part, index=False, encoding="utf-8-sig")
    attrs_part.replace(attributes_path)

    report.update(
        {
            "temporal_protocol": "FY2024_same_year",
            "historical_raw_vintage": "FY2009",
            "task_truth_vintage": "FY2024",
            "station_table": stations_path.relative_to(context.repo_root).as_posix(),
            "station_table_sha256": stations_sha,
            "regions_path": regions_path.relative_to(context.repo_root).as_posix(),
            "artifacts_before_sha256": before_sha,
            "regions_after_sha256": sha256_file(regions_path),
            "regions_sa4_after_sha256": sha256_file(regions_sa4_path),
            "region_attributes_after_sha256": sha256_file(attributes_path),
        }
    )
    # Keep the native trailing newline of the original Path.write_text report.
    atomic_text(report_path, json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + os.linesep)
    return report


__all__ = [
    "aggregate_fy2024_demand",
    "fixed_source_order",
    "publish_au_fy2024_demand",
    "update_dependent_region_tables",
]
