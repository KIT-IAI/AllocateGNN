"""
Voronoi nearest-neighbor allocator

Absorbs the core logic of SimpleVoronoi + NearestAssignment from voronoi/core/,
unified under the BaseAllocator interface.
"""
import numpy as np
import geopandas as gpd
import pandas as pd
from typing import Optional

from .base import BaseAllocator, AllocationResult
from .registry import allocator_registry


@allocator_registry.register("voronoi", description="Nearest-neighbor-based Voronoi spatial allocation")
class VoronoiAllocator(BaseAllocator):
    """
    Nearest-neighbor Voronoi allocator.
    Assigns each grid point to the nearest target point, then generates
    Voronoi polygons via dissolve + convex_hull.

    Configuration parameters:
        sub_columns: grouping column mapping (optional), format {"target_col": "grid_col"}
        working_crs: projected CRS used for computation (default "EPSG:3857")
        output_crs: output CRS (default "EPSG:4326")
    """

    def allocate(
        self,
        grid_gdf: gpd.GeoDataFrame,
        target_gdf: gpd.GeoDataFrame,
        weights: Optional[np.ndarray] = None,
        source_gdf: Optional[gpd.GeoDataFrame] = None,
        **kwargs,
    ) -> AllocationResult:
        sub_columns = self.config.get("sub_columns")
        working_crs = self.config.get("working_crs", "EPSG:3857")
        output_crs = self.config.get("output_crs", "EPSG:4326")

        # Project to the working CRS
        grid_proj = grid_gdf.to_crs(working_crs).copy()
        target_proj = target_gdf.to_crs(working_crs).copy()

        # Nearest-neighbor assignment
        if sub_columns:
            assignment_gdf = self._grouped_nearest(
                grid_proj, target_proj, sub_columns
            )
        else:
            assignment_gdf = gpd.sjoin_nearest(
                grid_proj[["geometry"]],
                target_proj[["geometry"]],
                how="left",
            )

        # Extract assignment indices
        assignment = assignment_gdf["index_right"].values.astype(int)

        # Generate Voronoi polygons (dissolve + convex_hull)
        assignment_gdf["assigned_target"] = assignment
        voronoi_gdf = assignment_gdf.dissolve(by="assigned_target")
        voronoi_gdf["geometry"] = voronoi_gdf.convex_hull
        voronoi_gdf = voronoi_gdf.to_crs(output_crs).reset_index()

        # Convert the output assignment_gdf back to the output CRS as well
        result_gdf = assignment_gdf.to_crs(output_crs)

        return AllocationResult(
            assignment=assignment,
            assignment_column_name="assigned_target",
            voronoi_gdf=voronoi_gdf[["assigned_target", "geometry"]],
            assignment_gdf=result_gdf,
            metadata={
                "method": "voronoi",
                "n_grid": len(grid_gdf),
                "n_target": len(target_gdf),
                "n_voronoi_regions": len(voronoi_gdf),
                "sub_columns": sub_columns,
            },
        )

    @staticmethod
    def _grouped_nearest(
        grid_proj: gpd.GeoDataFrame,
        target_proj: gpd.GeoDataFrame,
        sub_columns: dict,
    ) -> gpd.GeoDataFrame:
        """Grouped nearest-neighbor assignment"""
        target_col = list(sub_columns.keys())[0]
        grid_col = list(sub_columns.values())[0]

        common_groups = (
            set(grid_proj[grid_col].unique())
            & set(target_proj[target_col].unique())
        )

        all_mappings = []
        for group_id in common_groups:
            grid_sub = grid_proj[grid_proj[grid_col] == group_id]
            target_sub = target_proj[target_proj[target_col] == group_id]
            if grid_sub.empty or target_sub.empty:
                continue
            mapping = gpd.sjoin_nearest(grid_sub, target_sub, how="left")
            all_mappings.append(mapping)

        return pd.concat(all_mappings, ignore_index=True)
