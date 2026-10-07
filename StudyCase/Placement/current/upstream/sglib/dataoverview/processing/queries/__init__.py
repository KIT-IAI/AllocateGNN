"""Protocol adapters for query-driven sources."""

from . import arcgis, gee, geoserver, ogcapi, overpass

PROTOCOLS = {
    "arcgis": arcgis.run,
    "gee": gee.run,
    "geoserver": geoserver.run,
    "ogcapi": ogcapi.run,
    "overpass": overpass.run,
}

__all__ = ["PROTOCOLS"]
