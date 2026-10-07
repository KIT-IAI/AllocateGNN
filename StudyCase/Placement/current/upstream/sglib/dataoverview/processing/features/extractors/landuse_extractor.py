"""
LanduseExtractor — OSM 土地利用类别 + 面积占比

逻辑 copy 自 SpatialGranularity/utils/CalcuLanduse.py
关键改动: 类别映射从 JSON 配置读取，不再硬编码
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

from .registry import extractor_registry
from .base import BaseExtractor, ExtractorResult

logger = logging.getLogger(__name__)


@extractor_registry.register("landuse", description="OSM 土地利用类别 + 面积占比")
class LanduseExtractor(BaseExtractor):
    """
    从 OSM 数据提取土地利用特征。

    配置字段:
        fetcher: 依赖的 fetcher 名称（"osm"）
        categories: 类别映射字典 {"category_name": ["osm_tag1", "osm_tag2", ...]}
        default_category: 默认类别（未匹配到任何类别时使用）
        mode: 特征模式 "proportions"（面积占比）或 "point"（点查询） ，默认 "proportions"
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.categories: Dict[str, List[str]] = config.get("categories", {})
        self.default_category: str = config.get("default_category", "others")
        self.mode: str = config.get("mode", "proportions")
        self.target_grid_crs: str = config.get("target_grid_crs", "EPSG:3857")

        # 构建反向查找表: osm_tag → category
        self._tag_to_category: Dict[str, str] = {}
        for category, tags in self.categories.items():
            for tag in tags:
                self._tag_to_category[tag] = category

    def _classify(self, landuse_value) -> str:
        """
        将 OSM landuse 标签值映射到配置的类别。

        相当于原 CalcuLanduse.py 中的 get_category()，
        但类别映射从 JSON 配置读取。
        """
        if pd.isna(landuse_value):
            return self.default_category

        category = self._tag_to_category.get(landuse_value)
        if category is not None:
            return category

        logger.debug(f"Landuse '{landuse_value}' 未匹配，归入 '{self.default_category}'")
        return self.default_category

    def extract(
        self,
        grid_gdf: gpd.GeoDataFrame,
        step_size_m: float,
        fetched_path: Optional[str] = None,
    ) -> ExtractorResult:
        """
        从 OSM 数据计算土地利用特征。

        参数:
            grid_gdf: 网格点 GeoDataFrame（CRS=EPSG:4326）
            step_size_m: 网格步长（米）
            fetched_path: OsmFetcher 输出的 safe GeoParquet 路径

        返回:
            ExtractorResult，包含:
            - categorical_columns: {"landuse": (N,) str}  点查询模式
            - numerical_columns: {"lu_{cat}_prop": (N,) float}  占比模式
        """
        if fetched_path is None:
            raise ValueError("LanduseExtractor 需要 fetched_path（OSM safe GeoParquet 路径）")

        from ..fetchers.osm_fetcher import OSM_SAFE_COLUMNS

        source = Path(fetched_path)
        if source.suffix.lower() != ".parquet":
            raise ValueError("formal OSM extraction only accepts safe GeoParquet")
        osm_gdf = gpd.read_parquet(source)
        return self.extract_frame(grid_gdf, step_size_m, osm_gdf)

    def extract_frame(
        self,
        grid_gdf: gpd.GeoDataFrame,
        step_size_m: float,
        osm_gdf: gpd.GeoDataFrame,
    ) -> ExtractorResult:
        """Extract from an already safe-read frame (migration replay/audit API)."""

        from ..fetchers.osm_fetcher import OSM_SAFE_COLUMNS

        if tuple(osm_gdf.columns) != OSM_SAFE_COLUMNS:
            raise ValueError("OSM GeoParquet columns differ from the exact safe schema")

        if self.mode == "point":
            return self._extract_point(grid_gdf, osm_gdf)
        else:
            return self._extract_proportions(grid_gdf, osm_gdf, step_size_m)

    def _extract_point(
        self, grid_gdf: gpd.GeoDataFrame, osm_gdf: gpd.GeoDataFrame
    ) -> ExtractorResult:
        """
        点查询模式: 每个网格点所在的 landuse 类别。
        Copy 自 CalcuLanduse.fetch_landuse_data() 的 sjoin 逻辑。
        """
        osm_4326 = osm_gdf.to_crs("EPSG:4326")
        landuse_slim = osm_4326[["landuse", "geometry"]].dropna(subset=["landuse"])

        grid_4326 = grid_gdf.to_crs("EPSG:4326") if grid_gdf.crs != "EPSG:4326" else grid_gdf.copy()
        joined = gpd.sjoin(grid_4326, landuse_slim, how="left", predicate="within")
        joined = joined[~joined.index.duplicated(keep="first")]

        landuse_values = joined["landuse"].apply(self._classify).values

        return ExtractorResult(
            categorical_columns={"landuse": landuse_values},
        )

    def _extract_proportions(
        self, grid_gdf: gpd.GeoDataFrame, osm_gdf: gpd.GeoDataFrame, step_size_m: float
    ) -> ExtractorResult:
        """
        面积占比模式: 每个网格单元中各类别的面积占比。
        Copy 自 CalcuLanduse.calculate_landuse_proportions() 的 overlay 逻辑。
        """
        # 准备 OSM 数据
        osm_gdf = osm_gdf[["landuse", "geometry"]].dropna(subset=["landuse"])
        osm_gdf = osm_gdf.copy()
        osm_gdf["landuse"] = osm_gdf["landuse"].apply(self._classify)
        osm_m = osm_gdf.to_crs(self.target_grid_crs)

        # 只保留 Polygon/MultiPolygon（OSM 数据常混合点/线/面）
        osm_m = osm_m[osm_m.geom_type.isin(["Polygon", "MultiPolygon"])]

        # 去除重复 geometry（同一多边形只保留一条，避免 overlay 重复计数）
        osm_m = osm_m.drop_duplicates(subset=["geometry"])

        # dissolve 同类别多边形，消除同类空间重叠（避免 prop > 1.0）
        osm_m = osm_m.dissolve(by="landuse").explode(index_parts=False).reset_index()

        if osm_m.empty:
            logger.warning("OSM 数据过滤后为空，所有比例设为 0")
            return self._empty_proportions(len(grid_gdf))

        # 将网格点转换为网格单元（正方形多边形）
        grid_m = grid_gdf.to_crs(self.target_grid_crs).copy()
        half_step = step_size_m / 2
        grid_m["geometry"] = grid_m["geometry"].apply(
            lambda pt: Polygon([
                (pt.x - half_step, pt.y - half_step),
                (pt.x + half_step, pt.y - half_step),
                (pt.x + half_step, pt.y + half_step),
                (pt.x - half_step, pt.y + half_step),
            ])
        )
        grid_m["grid_id"] = grid_m.index

        # overlay 计算交集
        intersection_gdf = gpd.overlay(grid_m[["grid_id", "geometry"]], osm_m, how="intersection")
        intersection_gdf["intersection_area"] = intersection_gdf.geometry.area

        # 聚合
        landuse_areas = intersection_gdf.pivot_table(
            index="grid_id",
            columns="landuse",
            values="intersection_area",
            aggfunc="sum",
        ).fillna(0)

        # 计算占比
        cell_area = step_size_m ** 2
        landuse_proportions = landuse_areas / cell_area

        # 确保所有配置类别都有列
        category_names = list(self.categories.keys())
        result_numerical = {}
        for cat in category_names:
            col_name = f"lu_{cat}_prop"
            if cat in landuse_proportions.columns:
                # 按 grid_id 对齐到原始 grid_gdf 索引
                values = landuse_proportions[cat].reindex(grid_gdf.index, fill_value=0.0).values
            else:
                values = np.zeros(len(grid_gdf))
            result_numerical[col_name] = values

        return ExtractorResult(
            numerical_columns=result_numerical,
            metadata={
                "mode": "proportions",
                "target_grid_crs": self.target_grid_crs,
                "num_osm_records": len(osm_gdf),
                "num_intersections": len(intersection_gdf) if not osm_m.empty else 0,
            },
        )

    def _empty_proportions(self, n: int) -> ExtractorResult:
        """OSM 数据为空时返回全零比例"""
        category_names = list(self.categories.keys())
        return ExtractorResult(
            numerical_columns={f"lu_{cat}_prop": np.zeros(n) for cat in category_names},
            metadata={"mode": "proportions", "num_osm_records": 0},
        )

    def get_output_schema(self) -> dict:
        """从配置的 categories 动态生成 schema"""
        category_names = list(self.categories.keys())

        if self.mode == "point":
            return {
                "numerical": [],
                "categorical": {"landuse": category_names},
            }
        else:
            return {
                "numerical": [f"lu_{cat}_prop" for cat in category_names],
                "categorical": {},
            }
