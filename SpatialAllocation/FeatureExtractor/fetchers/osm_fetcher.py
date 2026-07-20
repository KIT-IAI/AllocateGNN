"""
OsmFetcher - OpenStreetMap data fetching (with caching)

Logic adapted from SpatialAllocation/utils/GetOsmData.py
Added: md5-based caching keyed on bounds + tags
"""
import hashlib
import logging
import os
import pickle
import threading
import time
from typing import Any

import osmnx as ox

from .registry import fetcher_registry
from .base import BaseFetcher

logger = logging.getLogger(__name__)


@fetcher_registry.register("osm", description="OpenStreetMap data fetching")
class OsmFetcher(BaseFetcher):
    """
    Fetches vector data such as land use from OpenStreetMap.

    Config fields:
        tags: OSM query tags (default {"landuse": true})
        buffer_m: boundary buffer distance in meters, default 0
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.tags = config.get("tags", {"landuse": True})
        self.buffer_m = config.get("buffer_m", 0)

    def fetch(self, region_bounds: Any, cache_dir: str) -> str:
        """
        Download OSM data to local storage and return the pickle file path.
        Skips the download if the file already exists.

        Args:
            region_bounds: a Shapely geometry (with a .bounds attribute)
            cache_dir: cache root directory

        Returns:
            the pickle file path
        """
        # Compute the cache path
        cache_key = self.get_cache_key(region_bounds)
        osm_cache_dir = os.path.join(cache_dir, "osm")
        os.makedirs(osm_cache_dir, exist_ok=True)
        cache_path = os.path.join(osm_cache_dir, f"{cache_key}.pickle")

        # Cache hit
        if os.path.exists(cache_path):
            logger.info(f"OSM cache hit: {cache_path}")
            return cache_path

        # Extract bounds values
        if hasattr(region_bounds, "bounds"):
            minx, miny, maxx, maxy = region_bounds.bounds
        else:
            minx, miny, maxx, maxy = region_bounds

        # Download data from OSM (enable osmnx console logging + background progress thread)
        logger.info(f"Downloading data from OSM: bounds=({minx:.4f}, {miny:.4f}, {maxx:.4f}, {maxy:.4f}), tags={self.tags}")
        ox.settings.log_console = True

        stop_event = threading.Event()

        def _progress_reporter():
            t0 = time.time()
            interval = 15
            while not stop_event.wait(interval):
                print(f"  [osm] waited {time.time() - t0:.0f}s, download still in progress...")

        reporter = threading.Thread(target=_progress_reporter, daemon=True)
        reporter.start()
        try:
            landuse_gdf = ox.features.features_from_bbox(
                bbox=(minx, miny, maxx, maxy),
                tags=self.tags,
            )
        finally:
            stop_event.set()
            reporter.join(timeout=1)
            ox.settings.log_console = False

        if landuse_gdf.empty:
            logger.warning("No OSM data found in the specified region")

        # Save to cache
        with open(cache_path, "wb") as f:
            pickle.dump(landuse_gdf, f)

        logger.info(f"OSM data cached: {cache_path} ({len(landuse_gdf)} records)")
        return cache_path

    def get_cache_key(self, region_bounds: Any) -> str:
        """Generate the cache key from bounds + tags"""
        if hasattr(region_bounds, "bounds"):
            bounds_tuple = region_bounds.bounds
        else:
            bounds_tuple = tuple(region_bounds)
        key_str = f"{bounds_tuple}_{self.tags}"
        return hashlib.md5(key_str.encode()).hexdigest()[:12]
