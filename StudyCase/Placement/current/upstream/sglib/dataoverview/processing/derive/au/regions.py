from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

from ...config import CountryPipelineContext

from ..common import atomic_geofile
from . import au_regions


def sync_region_tables(context: CountryPipelineContext) -> list[Path]:
    (context.derived_root / "bplus").mkdir(parents=True, exist_ok=True)
    sa3 = gpd.read_file(context.derived_root / "regions_sa3.gpkg", layer="regions_sa3")
    sa4 = gpd.read_file(context.derived_root / "regions_sa4.gpkg", layer="regions_sa4")
    outputs = [
        atomic_geofile(sa3, (context.derived_root / "bplus") / "regions_sa3.gpkg", layer="regions_sa3"),
        atomic_geofile(sa4, (context.derived_root / "bplus") / "regions_sa4.gpkg", layer="regions_sa4"),
    ]
    attributes = pd.read_csv(context.derived_root / "region_attributes.csv")
    attribute_path = (context.derived_root / "bplus") / "region_attributes.csv"
    partial = attribute_path.with_name(f".{attribute_path.name}.part")
    attributes.to_csv(partial, index=False)
    partial.replace(attribute_path)
    outputs.append(attribute_path)
    return outputs


def canonical_station_table(context: CountryPipelineContext) -> Path:
    station_path = context.derived_root / "ledger" / "station_table_fy2024.csv"
    capacity_path = context.derived_root / "ledger" / "au_firm_capacity.csv"
    region_path = (context.derived_root / "bplus") / "regions_sa3.gpkg"
    missing = [path for path in (station_path, capacity_path, region_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"AU canonical station inputs missing: {missing}")
    stations = pd.read_csv(station_path)
    capacity = pd.read_csv(capacity_path)
    stations = stations.loc[stations["status"].eq("usable")].merge(
        capacity[["station", "F_mva", "A_mva", "kv_primary"]], on="station", how="left", validate="one_to_one"
    )
    if stations[["F_mva", "lon_wgs84", "lat_wgs84"]].isna().any().any():
        raise RuntimeError("AU canonical station table has missing capacity or geometry")
    points = gpd.GeoDataFrame(
        stations,
        geometry=gpd.points_from_xy(stations["lon_wgs84"], stations["lat_wgs84"]),
        crs="EPSG:4326",
    )
    regions = gpd.read_file(region_path, layer="regions_sa3")
    joined = gpd.sjoin(points, regions[["SA3", "SA4", "loc_key", "geometry"]], how="left", predicate="within")
    if len(joined) != len(points):
        raise RuntimeError("AU station-to-region join is not one-to-one")
    points["SA3"] = joined["SA3"].to_numpy()
    points["SA4"] = joined["SA4"].to_numpy()
    points["loc_key"] = joined["loc_key"].to_numpy()
    points = points.loc[points["SA3"].notna()].copy().reset_index(drop=True)
    points["capacity_basis"] = "reconstructed_firm_n1"
    points["G_fy2024_mw"] = points["peak_mw"]
    points["station_id"] = "au:" + points["station"].astype(str)
    if points["station_id"].duplicated().any():
        raise RuntimeError("AU station_id is not unique")
    return atomic_geofile(points, (context.derived_root / "bplus") / "stations.gpkg")


def run(context: CountryPipelineContext, *, force: bool = False) -> list[Path]:
    au_regions.derive_au_regions(context, force=force)
    outputs = sync_region_tables(context)
    outputs.append(canonical_station_table(context))
    return outputs
