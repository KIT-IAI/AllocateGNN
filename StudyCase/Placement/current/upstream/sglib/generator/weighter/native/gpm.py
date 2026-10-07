"""
网格点权重 (Grid Point Model) — 基于土地利用比例

支持两种模式:
- proportional: 直接使用各 landuse 类型的连续比例作为权重矩阵 (N, K)
- categorical: 赢者通吃 — 每个格点取主导 landuse 类型的 one-hot 编码 (N, K)

下游按 ITL3 分组时，结合区域百分比计算最终需求。
"""
import numpy as np
import geopandas as gpd
from typing import Optional

from ..base import BaseWeighter, WeightResult
from ..registry import weighter_registry
from ..support import compose_support_field


@weighter_registry.register("gpm", description="土地利用比例权重：proportional / categorical 双模式")
class GPMWeighter(BaseWeighter):
    """
    基于土地利用比例的网格点权重计算器。

    配置参数:
        mode: "proportional"（默认）或 "categorical"
        proportion_columns: 网格 GeoDataFrame 中的比例列名列表（必需）
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
            raise ValueError("GPMWeighter 需要配置 'proportion_columns'（landuse 比例列名列表）")

        missing = [c for c in proportion_columns if c not in grid_gdf.columns]
        if missing:
            raise ValueError(f"grid_gdf 缺少以下比例列: {missing}")

        # 提取比例矩阵 P (N, K)
        P = grid_gdf[proportion_columns].values.astype(float)

        if mode == "proportional":
            W = P
        elif mode == "categorical":
            # 赢者通吃: one-hot(argmax)
            n, k = P.shape
            dominant = np.argmax(P, axis=1)
            W = np.zeros((n, k), dtype=float)
            W[np.arange(n), dominant] = 1.0
        else:
            raise ValueError(f"不支持的 mode: '{mode}'，可选: 'proportional', 'categorical'")

        if source_gdf is not None:
            source_column = self.config.get("source_column", "ITL3")
            demand_column = self.config.get("demand_column", "Demand (MVA)")
            feature_columns = self.config.get("source_feature_columns")
            if not feature_columns:
                raise ValueError("GPM final field requires source_feature_columns")
            missing_source = sorted(
                {source_column, demand_column, *feature_columns} - set(source_gdf.columns)
            )
            required_grid = {
                source_column, "covered_mask", "unknown_mask", "built_fraction"
            }
            missing_grid = sorted(required_grid - set(grid_gdf.columns))
            if missing_source or missing_grid:
                raise ValueError(
                    f"GPM final field missing grid={missing_grid} source={missing_source} columns"
                )
            info = source_gdf.set_index(source_column)
            scores = np.zeros(len(grid_gdf), dtype=float)
            for source, group in grid_gdf.groupby(source_column, sort=False):
                percentages = info.loc[source, feature_columns].to_numpy(dtype=float)
                scores[group.index.to_numpy(dtype=int)] = W[group.index.to_numpy(dtype=int)] @ percentages
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
                scores,
            )
            return WeightResult(
                weights=field,
                weight_column_name="gpm_demand",
                normalized=False,
                metadata={
                    "method": "gpm",
                    "mode": mode,
                    "support": "C/U/Z",
                    "proportion_columns": proportion_columns,
                    "source_feature_columns": feature_columns,
                },
            )

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


