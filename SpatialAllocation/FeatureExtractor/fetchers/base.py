"""
BaseFetcher base class - unified interface for all Fetcher plugins
"""
import hashlib
from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseFetcher(ABC):
    """
    Base class for data fetchers.
    Each Fetcher is responsible for downloading raw data from a specific data source
    (GEE, OSM, etc.) to local storage and managing the cache
    (skip the download if the file already exists).
    """

    def __init__(self, config: dict):
        """
        Args:
            config: the configuration block for this fetcher in the JSON config
        """
        self.config = config

    @abstractmethod
    def fetch(self, region_bounds: Any, cache_dir: str) -> str:
        """
        Download raw data to local storage and return the file path.
        If the cached file already exists, return its path directly (skip the download).

        Args:
            region_bounds: region bounds (a Shapely geometry or an
                (minx, miny, maxx, maxy) tuple, EPSG:4326)
            cache_dir: cache root directory

        Returns:
            local file path (GeoTIFF / pickle / etc.)
        """
        raise NotImplementedError

    def get_cache_key(self, region_bounds: Any) -> str:
        """
        Generate a cache filename hash from the region bounds and configuration.

        Args:
            region_bounds: region bounds

        Returns:
            an MD5 hash string (first 12 characters)
        """
        # Normalize bounds into a tuple
        if hasattr(region_bounds, 'bounds'):
            bounds_tuple = region_bounds.bounds
        else:
            bounds_tuple = tuple(region_bounds)

        key_str = f"{bounds_tuple}_{self.config}"
        return hashlib.md5(key_str.encode()).hexdigest()[:12]
