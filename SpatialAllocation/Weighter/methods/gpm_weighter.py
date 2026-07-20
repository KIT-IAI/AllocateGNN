"""
Grid Point Model (GPM) weighting — based on land-use proportions

Supports two modes:
- proportional: directly uses the continuous proportion of each landuse type as the weight matrix (N, K)
- categorical: winner-take-all — each grid point takes the one-hot encoding of its dominant landuse type (N, K)

Downstream, when grouping by ITL3, these are combined with regional percentages to compute the final demand.
"""
import numpy as np
import geopandas as gpd
from typing import Optional

from .base import BaseWeighter, WeightResult
from .registry import weighter_registry


@weighter_registry.register("gpm", description="Land-use proportion weighting: proportional / categorical dual mode")
class GPMWeighter(BaseWeighter):
    """
    Grid point weight calculator based on land-use proportions.

    Configuration parameters:
        mode: "proportional" (default) or "categorical"
        proportion_columns: list of proportion column names in the grid GeoDataFrame (required)
    """

    def compute(
        self,
        grid_gdf: gpd.GeoDataFrame,
        target_gdf: gpd.GeoDataFrame,
        source_gdf: Optional[gpd.GeoDataFrame] = None,
        **kwargs,
    ) -> WeightResult:
        mode = self.config.get("mode", "proportional")
        proportion_columns = self.config.get("proportion_columns")

        if not proportion_columns:
            raise ValueError("GPMWeighter requires 'proportion_columns' to be configured (list of landuse proportion column names)")

        missing = [c for c in proportion_columns if c not in grid_gdf.columns]
        if missing:
            raise ValueError(f"grid_gdf is missing the following proportion columns: {missing}")

        # Extract the proportion matrix P (N, K)
        P = grid_gdf[proportion_columns].values.astype(float)

        if mode == "proportional":
            W = P
        elif mode == "categorical":
            # Winner-take-all: one-hot(argmax)
            n, k = P.shape
            dominant = np.argmax(P, axis=1)
            W = np.zeros((n, k), dtype=float)
            W[np.arange(n), dominant] = 1.0
        else:
            raise ValueError(f"Unsupported mode: '{mode}', options: 'proportional', 'categorical'")

        return WeightResult(
            weights=W,
            weight_column_name="weight",
            normalized=False,
            metadata={
                "method": "gpm",
                "mode": mode,
                "proportion_columns": proportion_columns,
                "n_grid": len(grid_gdf),
                "n_types": len(proportion_columns),
            },
        )
