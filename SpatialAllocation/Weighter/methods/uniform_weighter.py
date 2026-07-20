"""
Uniform weighting — equal weight for all grid points
"""
import numpy as np
import geopandas as gpd
from typing import Optional

from .base import BaseWeighter, WeightResult
from .registry import weighter_registry


@weighter_registry.register("uniform", description="Assigns the same weight to all grid points")
class UniformWeighter(BaseWeighter):
    """
    The simplest weighting strategy: every grid point gets the same weight (1/N).
    Typically used as a baseline for comparison.
    """

    def compute(
        self,
        grid_gdf: gpd.GeoDataFrame,
        target_gdf: gpd.GeoDataFrame,
        source_gdf: Optional[gpd.GeoDataFrame] = None,
        **kwargs,
    ) -> WeightResult:
        n = len(grid_gdf)
        weights = np.ones(n) / n
        return WeightResult(
            weights=weights,
            weight_column_name="weight",
            normalized=True,
            metadata={"method": "uniform", "n_points": n},
        )
