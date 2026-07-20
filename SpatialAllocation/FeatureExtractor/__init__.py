"""
FeatureExtractor modular feature pipeline

Modeled after the LossFunction Registry plugin pattern:
- fetchers/ - data fetching layer (GEE, OSM, etc.)
- extractors/ - feature computation layer
- orchestrator - the orchestrator

Usage:
    from SpatialAllocation.FeatureExtractor import FeatureOrchestrator
    config = json.load(open("configs/default.json"))
    orchestrator = FeatureOrchestrator(config)
    grid_gdf, schema, results = orchestrator.run(region_gdf)
"""

# Import subpackages (triggers fetcher/extractor/corrector registration)
from . import fetchers  # noqa: F401
from . import extractors  # noqa: F401
from . import correctors  # noqa: F401

# Export the public API
from .orchestrator import FeatureOrchestrator
from .grid_generator import calculate_step_size, generate_base_grid
from .fetchers import fetcher_registry, FetcherRegistry, BaseFetcher
from .extractors import extractor_registry, ExtractorRegistry, BaseExtractor, ExtractorResult
from .correctors import corrector_registry, CorrectorRegistry, BaseCorrector, CorrectorResult

__all__ = [
    "FeatureOrchestrator",
    "calculate_step_size",
    "generate_base_grid",
    "fetcher_registry",
    "FetcherRegistry",
    "BaseFetcher",
    "extractor_registry",
    "ExtractorRegistry",
    "BaseExtractor",
    "ExtractorResult",
    "corrector_registry",
    "CorrectorRegistry",
    "BaseCorrector",
    "CorrectorResult",
]
