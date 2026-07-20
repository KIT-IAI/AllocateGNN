"""
Fetchers subpackage - imports all fetcher modules to trigger registration
"""
from .registry import fetcher_registry, FetcherRegistry
from .base import BaseFetcher

# Import concrete fetcher modules to trigger the @fetcher_registry.register decorator
# Add the import here whenever a new fetcher is introduced
from . import osm_fetcher  # noqa: F401
from . import sentinel2_fetcher  # noqa: F401
from . import worldcover_fetcher  # noqa: F401
from . import ntl_fetcher  # noqa: F401

__all__ = ["fetcher_registry", "FetcherRegistry", "BaseFetcher"]
