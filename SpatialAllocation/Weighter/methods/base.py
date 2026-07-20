"""
BaseWeighter base class + WeightResult data structure
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import geopandas as gpd


@dataclass
class WeightResult:
    """
    Unified output structure for weight calculators.

    Attributes:
        weights: (N,) scalar weight array, or (N, K) multi-type weight matrix
        weight_column_name: column name used when writing to a GeoDataFrame
        normalized: whether normalized (sum=1 within each source group)
        metadata: additional metadata (method parameters, computation statistics, etc.)
    """
    weights: np.ndarray
    weight_column_name: str = "weight"
    normalized: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseWeighter(ABC):
    """
    Base class for weight calculators.
    All Weighter implementations must inherit from this class and implement the compute method.
    """

    def __init__(self, config: dict):
        """
        Args:
            config: configuration dictionary for the weight calculator
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
        Optional training/fitting step (no-op by default).
        Weighters that require pretraining should override this method.
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
        Compute the weight for each grid point.

        Args:
            grid_gdf: grid point GeoDataFrame (N rows, containing a geometry column)
            target_gdf: target point GeoDataFrame (substations, etc.)
            source_gdf: source region GeoDataFrame (optional)
            **kwargs: method-specific additional parameters

        Returns:
            WeightResult
        """
        raise NotImplementedError

    @property
    def requires_fit(self) -> bool:
        """Whether fit() must be called first"""
        return False
