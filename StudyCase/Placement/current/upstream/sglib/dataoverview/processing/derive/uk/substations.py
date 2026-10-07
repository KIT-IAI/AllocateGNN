from __future__ import annotations

from pathlib import Path
import hashlib

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from ...config import CountryPipelineContext

from ..common import atomic_geofile


def build(context: CountryPipelineContext, regions: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    regions = regions.drop(columns=["Demand (MVA)"], errors="ignore").copy()
    path = context.raw_root / "HDRah-Data-PS-GB-1f63a32" / "GB_PS_data_extend.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    table = pd.read_csv(path)
    coordinates = table["Geo(Long,Lat)"].map(lambda value: [float(item) for item in str(value).split(",")])
    table["geometry"] = [Point(value[0], value[1]) for value in coordinates]
    stations = gpd.GeoDataFrame(table, geometry="geometry", crs="EPSG:4326")
    numeric = stations.select_dtypes(include=np.number).columns.tolist()
    non_numeric = [column for column in stations.select_dtypes(exclude=np.number).columns if column != "geometry"]
    aggregation = {column: "sum" for column in numeric} | {column: "first" for column in non_numeric}
    stations = gpd.GeoDataFrame(stations.groupby("geometry", as_index=False).agg(aggregation), geometry="geometry", crs="EPSG:4326")
    stations["Demand (MVA)"] = pd.to_numeric(stations["Demand (MVA)"], errors="coerce")
    stations["Firm Capacity (MVA)"] = pd.to_numeric(stations["Firm Capacity (MVA)"], errors="coerce")
    stations["RegName"] = stations["RegName"].str.replace(" ", "", regex=False)
    stations = stations[["PS Name", "Demand (MVA)", "Firm Capacity (MVA)", "RegName", "geometry", "RegID"]]
    joined = gpd.sjoin(stations, regions, how="left", predicate="within")
    stations["ITL3"] = joined["ITL3"].to_numpy()
    stations["ITL2"] = joined["ITL2"].to_numpy()
    demand = joined.groupby("ITL3", dropna=True)["Demand (MVA)"].sum()
    regions["Demand (MVA)"] = regions["ITL3"].map(demand)
    london = ["TLI3", "TLI4", "TLI5", "TLI6", "TLI7"]
    regions.loc[regions["ITL2"].isin(london), "ITL2"] = "London"
    stations.loc[stations["ITL2"].isin(london), "ITL2"] = "London"
    stations["capacity_basis"] = "firm_n1"
    stations["station_id"] = [
        "uk:" + hashlib.sha256(
            f"{name}|{geometry.x:.8f}|{geometry.y:.8f}".encode("utf-8")
        ).hexdigest()[:20]
        for name, geometry in zip(stations["PS Name"].astype(str), stations.geometry, strict=True)
    ]
    if stations["station_id"].duplicated().any():
        raise RuntimeError("UK station_id is not unique")
    return stations, regions


def run(context: CountryPipelineContext, *, force: bool = False) -> tuple[Path, Path]:
    region_path = (context.derived_root / "bplus") / "regions.gpkg"
    station_path = (context.derived_root / "bplus") / "stations.gpkg"
    if station_path.is_file() and region_path.is_file() and not force:
        existing = gpd.read_file(station_path, rows=1)
        if {"capacity_basis", "station_id"} <= set(existing.columns):
            return station_path, region_path
    if not region_path.is_file():
        from .regions import run as run_regions

        run_regions(context, force=force)
    regions = gpd.read_file(region_path)
    stations, updated_regions = build(context, regions)
    atomic_geofile(updated_regions, region_path)
    atomic_geofile(stations, station_path)
    return station_path, region_path
