"""
Allocator spatial allocation subpackage

Follows the FeatureExtractor Registry plugin pattern:
- methods/ — allocation methods (voronoi / civd)
- clustering/ — clustering algorithms (moved in from voronoi/clustering/)

Usage:
    from SpatialAllocation.Allocator import allocator_registry
    allocator = allocator_registry.create("voronoi")
    result = allocator.allocate(grid_gdf, target_gdf)
"""

# Import subpackage (triggers allocator registration)
from . import methods  # noqa: F401

# Export public API
from .methods import allocator_registry, AllocatorRegistry, BaseAllocator, AllocationResult

__all__ = [
    "allocator_registry",
    "AllocatorRegistry",
    "BaseAllocator",
    "AllocationResult",
]
