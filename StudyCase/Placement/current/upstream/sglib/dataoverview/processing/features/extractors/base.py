"""
BaseExtractor 基类 + ExtractorResult 数据结构
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import geopandas as gpd


@dataclass
class ExtractorResult:
    """
    特征提取器的统一输出结构。

    属性:
        numerical_columns: 数值列 {列名: (N,) ndarray}，会合并到 grid_gdf
        categorical_columns: 分类列 {列名: (N,) str ndarray}，会合并到 grid_gdf
        array_features: 高维特征矩阵 (N, d)，不加入 GDF，单独保存/传递
        array_feature_names: array_features 各列的名称
        metadata: 额外元数据（如有效像素数、处理统计等）
    """
    numerical_columns: Dict[str, np.ndarray] = field(default_factory=dict)
    categorical_columns: Dict[str, np.ndarray] = field(default_factory=dict)
    array_features: Optional[np.ndarray] = None
    array_feature_names: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseExtractor(ABC):
    """
    特征提取器基类。
    每个 Extractor 从已下载的数据（由 Fetcher 提供的本地文件）
    对每个网格点计算特征。
    """

    def __init__(self, config: dict):
        """
        参数:
            config: JSON 配置中对应 extractor 的配置块
        """
        self.config = config

    @abstractmethod
    def extract(
        self,
        grid_gdf: gpd.GeoDataFrame,
        step_size_m: float,
        fetched_path: Optional[str] = None,
    ) -> ExtractorResult:
        """
        对每个网格点计算特征。

        参数:
            grid_gdf: 网格点 GeoDataFrame（包含 geometry 列）
            step_size_m: 网格步长（米）
            fetched_path: 对应 Fetcher 输出的本地文件路径，无依赖则为 None

        返回:
            ExtractorResult
        """
        raise NotImplementedError

    @abstractmethod
    def get_output_schema(self) -> dict:
        """
        声明输出的列名和类型，用于下游 preprocess_features。

        返回:
            {
                "numerical": ["col1", "col2", ...],
                "categorical": {"col_name": ["member1", "member2", ...]}
            }
        """
        raise NotImplementedError

    @property
    def fetcher_name(self) -> Optional[str]:
        """
        声明依赖的 Fetcher 名称，从 config["fetcher"] 读取。
        无 Fetcher 依赖的 Extractor 返回 None。
        """
        return self.config.get("fetcher")
