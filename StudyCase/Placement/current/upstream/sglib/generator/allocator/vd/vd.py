"""
VD（Voronoi disaggregation）最近邻分配器

吸收 voronoi/core/ 中 SimpleVoronoi + NearestAssignment 的核心逻辑，
统一为 BaseAllocator 接口。
"""
import numpy as np
import geopandas as gpd
import pandas as pd
from typing import Optional

from ..base import BaseAllocator, AllocationResult
from ..registry import allocator_registry


def compute_vd_assignment(
    grid_gdf: gpd.GeoDataFrame,
    target_gdf: gpd.GeoDataFrame,
    *,
    config: dict | None = None,
) -> np.ndarray:
    """Return only the VD assignment vector for metric/serialization consumers."""

    return VDAllocator(config or {}).allocate(grid_gdf, target_gdf).assignment


def aggregate_by_assignment(
    assignment: np.ndarray,
    values: np.ndarray,
    n_targets: int,
) -> np.ndarray:
    """Aggregate one grid value vector to target ordinals."""

    assignment = np.asarray(assignment, dtype=np.int64)
    values = np.asarray(values, dtype=float)
    if assignment.shape != values.shape or assignment.ndim != 1:
        raise ValueError("assignment and values must be aligned vectors")
    if n_targets < 0 or (len(assignment) and ((assignment < 0).any() or (assignment >= n_targets).any())):
        raise ValueError("assignment contains an out-of-range target ordinal")
    result = np.zeros(int(n_targets), dtype=float)
    np.add.at(result, assignment, values)
    return result


@allocator_registry.register("vd", description="Voronoi disaggregation nearest assignment")
class VDAllocator(BaseAllocator):
    """
    最近邻 Voronoi 分配器。
    将每个网格点分配到最近的目标点，然后通过 dissolve + convex_hull
    生成 Voronoi 多边形。

    配置参数:
        sub_columns: 分组列映射（可选），格式 {"target_col": "grid_col"}
        working_crs: 计算用投影坐标系（默认 "EPSG:3857"）
        output_crs: 输出坐标系（默认 "EPSG:4326"）
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

        # 投影到工作坐标系
        grid_proj = grid_gdf.to_crs(working_crs).copy()
        target_proj = target_gdf.to_crs(working_crs).copy()

        # 最近邻分配
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

        # 提取分配索引
        assignment = assignment_gdf["index_right"].values.astype(int)

        # 生成 Voronoi 多边形（dissolve + convex_hull）
        assignment_gdf["assigned_target"] = assignment
        voronoi_gdf = assignment_gdf.dissolve(by="assigned_target")
        voronoi_gdf["geometry"] = voronoi_gdf.convex_hull
        voronoi_gdf = voronoi_gdf.to_crs(output_crs).reset_index()

        # 输出的 assignment_gdf 也转回输出坐标系
        result_gdf = assignment_gdf.to_crs(output_crs)

        return AllocationResult(
            assignment=assignment,
            assignment_column_name="assigned_target",
            voronoi_gdf=voronoi_gdf[["assigned_target", "geometry"]],
            assignment_gdf=result_gdf,
            metadata={
                "method": "vd",
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
        """分组最近邻分配"""
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

