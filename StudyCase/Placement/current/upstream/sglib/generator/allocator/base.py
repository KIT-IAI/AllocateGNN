"""
BaseAllocator 基类 + AllocationResult 数据结构
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import geopandas as gpd


@dataclass
class AllocationResult:
    """
    空间分配器的统一输出结构。

    属性:
        assignment: (N,) 目标索引数组，表示每个网格点分配到的目标
        assignment_column_name: 写入 GeoDataFrame 时使用的列名
        voronoi_gdf: Voronoi 多边形 GeoDataFrame（如适用）
        assignment_gdf: 带分配标签的网格点 GeoDataFrame
        confidence: (N,) 分配置信度数组（如适用）
        metadata: 额外元数据（方法参数、计算统计等）
    """
    assignment: np.ndarray
    assignment_column_name: str = "assigned_target"
    voronoi_gdf: Optional[gpd.GeoDataFrame] = None
    assignment_gdf: Optional[gpd.GeoDataFrame] = None
    confidence: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAllocator(ABC):
    """
    空间分配器基类。
    所有 Allocator 实现必须继承此类并实现 allocate 方法。
    """

    def __init__(self, config: dict):
        """
        参数:
            config: 分配器的配置字典
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
        可选的训练/拟合步骤（默认无操作）。
        需要预训练的 Allocator 应重写此方法。
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
        将网格点分配到目标。

        参数:
            grid_gdf: 网格点 GeoDataFrame（N 行，包含 geometry 列）
            target_gdf: 目标点 GeoDataFrame（变电站等）
            weights: (N,) 权重数组（由 Weighter 提供，可选）
            source_gdf: 源区域 GeoDataFrame（可选）
            **kwargs: 方法特定的额外参数

        返回:
            AllocationResult
        """
        raise NotImplementedError

    @property
    def requires_fit(self) -> bool:
        """是否需要先调用 fit()"""
        return False

    @property
    def requires_weights(self) -> bool:
        """是否需要权重输入"""
        return False


