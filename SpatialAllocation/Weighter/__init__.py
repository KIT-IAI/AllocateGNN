"""
Weighter weight computation subpackage

Follows the FeatureExtractor Registry plugin pattern:
- methods/ — weight computation methods (uniform / gpm / hetero_gnn)

Usage:
    from SpatialAllocation.Weighter import weighter_registry
    weighter = weighter_registry.create("uniform")
    result = weighter.compute(grid_gdf, target_gdf)
"""

# Import subpackage (triggers weighter registration)
from . import methods  # noqa: F401

# Export public API
from .methods import weighter_registry, WeighterRegistry, BaseWeighter, WeightResult

__all__ = [
    "weighter_registry",
    "WeighterRegistry",
    "BaseWeighter",
    "WeightResult",
]
