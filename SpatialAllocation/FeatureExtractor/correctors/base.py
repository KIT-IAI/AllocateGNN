"""
BaseCorrector abstract base class + CorrectorResult data class
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict

import geopandas as gpd
import numpy as np


@dataclass
class CorrectorResult:
    """Correction result"""
    corrected_col: str              # column name written to grid_gdf
    metadata: Dict[str, Any] = field(default_factory=dict)  # per-ITL3 statistics


class BaseCorrector(ABC):
    """
    Abstract base class for demand correctors.

    Every Corrector takes precomputed scores (N,), applies a multiplicative
    correction to base_demand, and renormalizes per-ITL3 to preserve demand conservation.
    """

    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def correct(
        self,
        grid_gdf: gpd.GeoDataFrame,
        region_sub: gpd.GeoDataFrame,
        base_demand_col: str,
        scores: np.ndarray,
        corrected_col: str,
    ) -> CorrectorResult:
        """
        Correct grid_gdf in place, adding the corrected_col column, preserving per-ITL3 demand conservation.

        Parameters
        ----------
        grid_gdf : grid GeoDataFrame (modified in place)
        region_sub : region GeoDataFrame, containing 'ITL3', 'Demand (MVA)' columns
        base_demand_col : base demand column name
        scores : (N,) array of correction signals
        corrected_col : corrected demand column name

        Returns
        -------
        CorrectorResult
        """
