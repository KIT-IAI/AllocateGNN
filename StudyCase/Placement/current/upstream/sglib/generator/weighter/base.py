"""
BaseWeighter 基类 + WeightResult 数据结构
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import geopandas as gpd


@dataclass
class WeightResult:
    """
    权重计算器的统一输出结构。

    属性:
        weights: (N,) 标量权重数组，或 (N, K) 多类型权重矩阵
        weight_column_name: 写入 GeoDataFrame 时使用的列名
        normalized: 是否已归一化（每个 source 组 sum=1）
        metadata: 额外元数据（方法参数、计算统计等）
    """
    weights: np.ndarray
    weight_column_name: str = "weight"
    normalized: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseWeighter(ABC):
    """
    权重计算器基类。
    所有 Weighter 实现必须继承此类并实现 compute 方法。
    """

    def __init__(self, config: dict):
        """
        参数:
            config: 权重计算器的配置字典
        """
        self.config = config

    def fit(
        self,
        grid_gdf: gpd.GeoDataFrame,
        target_gdf: gpd.GeoDataFrame,
        source_gdf: Optional[gpd.GeoDataFrame] = None,
        **kwargs,
    ) -> None:
        """
        可选的训练/拟合步骤（默认无操作）。
        需要预训练的 Weighter 应重写此方法。
        """
        pass

    @abstractmethod
    def compute(
        self,
        grid_gdf: gpd.GeoDataFrame,
        target_gdf: gpd.GeoDataFrame,
        source_gdf: Optional[gpd.GeoDataFrame] = None,
        **kwargs,
    ) -> WeightResult:
        """
        计算每个网格点的权重。

        参数:
            grid_gdf: 网格点 GeoDataFrame（N 行，包含 geometry 列）
            target_gdf: 目标点 GeoDataFrame（变电站等）
            source_gdf: 源区域 GeoDataFrame（可选）
            **kwargs: 方法特定的额外参数

        返回:
            WeightResult
        """
        raise NotImplementedError

    @property
    def requires_fit(self) -> bool:
        """是否需要先调用 fit()"""
        return False


