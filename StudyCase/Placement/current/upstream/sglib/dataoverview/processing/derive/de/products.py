from __future__ import annotations

from ...config import CountryPipelineContext

from pathlib import Path

import geopandas as gpd

from . import legacy
from ..common import atomic_geofile


def run(context: CountryPipelineContext, *, force: bool = False) -> list[Path]:
    expected = [context.derived_root / name for name in legacy.OUTPUTS]
    if not all(path.is_file() for path in expected) or force:
        legacy.derive_de(context, force=force)
    formal = context.derived_root / "bplus"
    regions = gpd.read_file(context.derived_root / "source_regions.gpkg")
    stations = gpd.read_file(context.derived_root / "substations.gpkg")
    changed = False
    if "capacity_basis" not in stations:
        stations["capacity_basis"] = "not_applicable"
        changed = True
    if "station_id" not in stations:
        stations["station_id"] = "de:" + stations["Kennzeichen"].astype(str)
        changed = True
    if stations["station_id"].duplicated().any():
        raise RuntimeError("DE station_id is not unique")
    region_path = atomic_geofile(regions, formal / "regions.gpkg")
    station_path = atomic_geofile(stations, formal / "stations.gpkg")
    return [region_path, station_path]
