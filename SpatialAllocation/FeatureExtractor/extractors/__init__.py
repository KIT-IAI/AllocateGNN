"""
Extractors subpackage — imports all extractor modules to trigger registration
"""
from .registry import extractor_registry, ExtractorRegistry
from .base import BaseExtractor, ExtractorResult

# Import concrete extractor modules to trigger the @extractor_registry.register decorator
# Add new imports here as new extractors are added
from . import landuse_extractor  # noqa: F401
from . import spectral_extractor  # noqa: F401
from . import worldcover_extractor  # noqa: F401
from . import ntl_extractor  # noqa: F401

__all__ = ["extractor_registry", "ExtractorRegistry", "BaseExtractor", "ExtractorResult"]
