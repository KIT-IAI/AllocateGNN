"""
NtlExtractor — VIIRS 夜间灯光逐点采样特征

从 NTL GeoTIFF 中提取每个网格点的辐射值。
NTL 原生分辨率 ~500m，使用 rasterio.sample() 逐点采样，
不需要 patch 裁切 + 聚合的重模式。

两种模式：
- 纯点采样（radius_m=0，默认）：直接读取每个 grid 点所在像素值
- 邻域聚合（radius_m>0）：以 radius_m 为半径计算均值/标准差
"""
import logging
from typing import List, Optional

import numpy as np
import geopandas as gpd

from .registry import extractor_registry
from .base import BaseExtractor, ExtractorResult

logger = logging.getLogger(__name__)


@extractor_registry.register("ntl", description="VIIRS 夜间灯光逐点采样")
class NtlExtractor(BaseExtractor):
    """
    从 NTL GeoTIFF 提取辐射值特征。

    配置字段:
        fetcher: 依赖的 fetcher 名称（"ntl"）
        radius_m: 邻域半径（默认 0 = 纯点采样）
        aggregation: 当 radius_m > 0 时使用的聚合方法（默认 ["mean", "std"]）
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.radius_m: float = config.get("radius_m", 0)
        self.aggregation: List[str] = config.get("aggregation", ["mean", "std"])

    def extract(
        self,
        grid_gdf: gpd.GeoDataFrame,
        step_size_m: float,
        fetched_path: Optional[str] = None,
    ) -> ExtractorResult:
        """
        对每个网格点提取 NTL 辐射值。

        模式 A（radius_m == 0）：rasterio.sample() 逐点采样
        模式 B（radius_m > 0）：读取整张 raster，逐点邻域聚合

        返回:
            ExtractorResult:
            - numerical_columns: {"ntl_value": (N,)} 或 {"ntl_mean": ..., "ntl_std": ...}
        """
        if fetched_path is None:
            raise ValueError("NtlExtractor 需要 fetched_path（GeoTIFF 路径）")

        import rasterio

        n_points = len(grid_gdf)

        if self.radius_m <= 0:
            # 模式 A：纯点采样
            numerical_columns = self._extract_point_sample(
                grid_gdf, fetched_path, n_points
            )
        else:
            # 模式 B：邻域聚合
            numerical_columns = self._extract_neighborhood(
                grid_gdf, fetched_path, n_points
            )

        return ExtractorResult(
            numerical_columns=numerical_columns,
            metadata={
                "mode": "point_sample" if self.radius_m <= 0 else "neighborhood",
                "radius_m": self.radius_m,
                "n_points": n_points,
            },
        )

    def _extract_point_sample(
        self, grid_gdf: gpd.GeoDataFrame, fetched_path: str, n_points: int
    ) -> dict:
        """纯点采样：rasterio.sample() 直接读取每个 grid 点所在像素值"""
        import rasterio

        with rasterio.open(fetched_path) as src:
            grid_proj = grid_gdf.to_crs(src.crs)
            coords = list(zip(grid_proj.geometry.x, grid_proj.geometry.y))
            samples = np.array(list(src.sample(coords)))  # (N, 1)
            values = samples[:, 0].astype(np.float64)  # (N,)

        # NaN/负值 → 0（VIIRS 暗区可能有仪器噪声负值）
        valid_mask = np.isfinite(values) & (values >= 0)
        values = np.where(valid_mask, values, 0.0)

        n_invalid = n_points - np.sum(valid_mask)
        if n_invalid > 0:
            logger.info(f"NTL 点采样: {n_invalid}/{n_points} 个点为无效值，已置零")

        return {"ntl_value": values}

    def _extract_neighborhood(
        self, grid_gdf: gpd.GeoDataFrame, fetched_path: str, n_points: int
    ) -> dict:
        """邻域聚合：读取整张 raster，逐点以 radius_m 为半径计算统计量"""
        import rasterio

        with rasterio.open(fetched_path) as src:
            grid_proj = grid_gdf.to_crs(src.crs)
            raster_data = src.read(1).astype(np.float64)  # (H, W)
            transform = src.transform
            pixel_size = src.res[0]  # 假设正方形像素

        # 邻域半径对应的像素数
        radius_px = max(1, int(round(self.radius_m / pixel_size)))

        xs = grid_proj.geometry.x.values
        ys = grid_proj.geometry.y.values
        h, w = raster_data.shape

        # 预分配结果
        results = {agg: np.zeros(n_points, dtype=np.float64) for agg in self.aggregation}

        for i in range(n_points):
            # 点坐标转像素坐标
            col = int((xs[i] - transform.c) / transform.a)
            row = int((ys[i] - transform.f) / transform.e)

            # 裁切邻域窗口
            r0 = max(0, row - radius_px)
            r1 = min(h, row + radius_px + 1)
            c0 = max(0, col - radius_px)
            c1 = min(w, col + radius_px + 1)

            patch = raster_data[r0:r1, c0:c1]

            # 过滤无效值
            valid = patch[np.isfinite(patch) & (patch >= 0)]

            if len(valid) == 0:
                for agg in self.aggregation:
                    results[agg][i] = 0.0
                continue

            for agg in self.aggregation:
                if agg == "mean":
                    results[agg][i] = np.mean(valid)
                elif agg == "std":
                    results[agg][i] = np.std(valid)
                elif agg == "median":
                    results[agg][i] = np.median(valid)
                elif agg == "max":
                    results[agg][i] = np.max(valid)
                elif agg == "min":
                    results[agg][i] = np.min(valid)
                else:
                    logger.warning(f"未知聚合方法: {agg}，跳过")

        return {f"ntl_{agg}": results[agg] for agg in self.aggregation}

    def get_output_schema(self) -> dict:
        """从 radius_m 和 aggregation 动态生成 schema"""
        if self.radius_m <= 0:
            return {
                "numerical": ["ntl_value"],
                "categorical": {},
            }
        else:
            return {
                "numerical": [f"ntl_{agg}" for agg in self.aggregation],
                "categorical": {},
            }
