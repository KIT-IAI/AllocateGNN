"""
Allocator methods subpackage — imports all allocator modules to trigger registration
"""
from .registry import allocator_registry, AllocatorRegistry
from .base import BaseAllocator, AllocationResult

# Import concrete allocator modules to trigger the @allocator_registry.register decorator
from . import voronoi_allocator  # noqa: F401
from . import civd_allocator  # noqa: F401

__all__ = ["allocator_registry", "AllocatorRegistry", "BaseAllocator", "AllocationResult"]
