"""Formal source-aware Uniform weighting on C/U/Z support."""

from __future__ import annotations

from typing import Optional

import geopandas as gpd
import numpy as np

from ..support import compose_support_field
from ..base import BaseWeighter, WeightResult
from ..registry import weighter_registry


@weighter_registry.register("uniform", description="Source-conserving C/U/Z Uniform field")
class UniformWeighter(BaseWeighter):
    def compute(
        self,
        grid_gdf: gpd.GeoDataFrame,
        target_gdf: gpd.GeoDataFrame,
        source_gdf: Optional[gpd.GeoDataFrame] = None,
        **kwargs,
    ) -> WeightResult:
        if source_gdf is None:
            raise ValueError("UniformWeighter requires source_gdf with source demand")
        source_column = self.config.get("source_column", "ITL3")
        demand_column = self.config.get("demand_column", "Demand (MVA)")
        required_grid = {source_column, "covered_mask", "unknown_mask", "built_fraction"}
        missing_grid = sorted(required_grid - set(grid_gdf.columns))
        missing_source = sorted({source_column, demand_column} - set(source_gdf.columns))
        if missing_grid or missing_source:
            raise ValueError(
                f"UniformWeighter missing grid={missing_grid} source={missing_source} columns"
            )
        demand = dict(
            zip(
                source_gdf[source_column].astype(str),
                source_gdf[demand_column].astype(float),
                strict=True,
            )
        )
        field = compose_support_field(
            grid_gdf[source_column].astype(str).to_numpy(),
            demand,
            grid_gdf["covered_mask"].to_numpy(bool),
            grid_gdf["unknown_mask"].to_numpy(bool),
            grid_gdf["built_fraction"].to_numpy(float),
            np.ones(len(grid_gdf), dtype=float),
        )
        return WeightResult(
            weights=field,
            weight_column_name="uniform_demand",
            normalized=False,
            metadata={"method": "uniform", "support": "C/U/Z"},
        )


