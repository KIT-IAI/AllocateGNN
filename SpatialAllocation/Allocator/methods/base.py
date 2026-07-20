"""
BaseAllocator base class + AllocationResult data structure
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import geopandas as gpd


@dataclass
class AllocationResult:
    """
    Unified output structure for spatial allocators.

    Attributes:
        assignment: (N,) array of target indices indicating which target each grid point is assigned to
        assignment_column_name: column name used when writing to a GeoDataFrame
        voronoi_gdf: Voronoi polygon GeoDataFrame (if applicable)
        assignment_gdf: grid point GeoDataFrame with assignment labels
        confidence: (N,) array of assignment confidence scores (if applicable)
        metadata: additional metadata (method parameters, computation statistics, etc.)
    """
    assignment: np.ndarray
    assignment_column_name: str = "assigned_target"
    voronoi_gdf: Optional[gpd.GeoDataFrame] = None
    assignment_gdf: Optional[gpd.GeoDataFrame] = None
    confidence: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAllocator(ABC):
    """
    Base class for spatial allocators.
    All Allocator implementations must inherit from this class and implement the allocate method.
    """

    def __init__(self, config: dict):
        """
        Args:
            config: configuration dictionary for the allocator
        """
        self.config = config

    def fit(
        self,
        grid_gdf: gpd.GeoDataFrame,
        target_gdf: gpd.GeoDataFrame,
        weights: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        """
        Optional training/fitting step (no-op by default).
        Allocators that require pretraining should override this method.
        """
        pass

    @abstractmethod
    def allocate(
        self,
        grid_gdf: gpd.GeoDataFrame,
        target_gdf: gpd.GeoDataFrame,
        weights: Optional[np.ndarray] = None,
        source_gdf: Optional[gpd.GeoDataFrame] = None,
        **kwargs,
    ) -> AllocationResult:
        """
        Assign grid points to targets.

        Args:
            grid_gdf: grid point GeoDataFrame (N rows, containing a geometry column)
            target_gdf: target point GeoDataFrame (substations, etc.)
            weights: (N,) weight array (provided by a Weighter, optional)
            source_gdf: source region GeoDataFrame (optional)
            **kwargs: method-specific additional parameters

        Returns:
            AllocationResult
        """
        raise NotImplementedError

    @property
    def requires_fit(self) -> bool:
        """Whether fit() must be called first"""
        return False

    @property
    def requires_weights(self) -> bool:
        """Whether weight input is required"""
        return False
