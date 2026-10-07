"""AU query helpers used by the AU transform adapter."""

from __future__ import annotations

import json
from pathlib import Path


from .queries.arcgis import run as run_arcgis
from .queries.overpass import run as run_overpass

LOCATIONS_URL = (
    "https://portal.data.nsw.gov.au/arcgis/rest/services/Hosted/"
    "Ausgrid_UHC_Data/FeatureServer/0/query?where=year%3D2025&outFields=*&f=geojson"
)
LOCATIONS_RELPATH = "locations_sample/ausgrid_uhc_full_2025.geojson"
OSM_BBOX = (-34.25, 150.5, -32.3, 152.3)
OSM_BBOX_EXT = (-34.6, 150.0, -31.8, 152.8)
OSM_BBOX_RELPATH = "osm/overpass_power_substations_bbox.json"
OSM_BBOX_EXT_RELPATH = "osm/overpass_power_substations_bbox_ext.json"


def land_locations_layer(target: Path) -> dict:
    path = run_arcgis({"url": LOCATIONS_URL, "filename": LOCATIONS_RELPATH}, target)[0]
    return json.loads(path.read_text(encoding="utf-8"))


def fetch_osm_substation_dump(
    bbox: tuple[float, float, float, float],
    cache_rel: str,
    label: str,
    target: Path,
) -> dict:
    south, west, north, east = bbox
    query = (
        "[out:json][timeout:45];("
        f'node["power"="substation"]({south},{west},{north},{east});'
        f'way["power"="substation"]({south},{west},{north},{east});'
        f'relation["power"="substation"]({south},{west},{north},{east});'
        ");out center tags;"
    )
    path = run_overpass(
        {
            "endpoint": "https://overpass-api.de/api/interpreter",
            "query": query,
            "filename": cache_rel,
        },
        target,
    )[0]
    return json.loads(path.read_text(encoding="utf-8"))
