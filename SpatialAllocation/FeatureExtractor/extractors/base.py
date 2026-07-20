"""
BaseExtractor base class + ExtractorResult data structure
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import geopandas as gpd


@dataclass
class ExtractorResult:
    """
    Unified output structure for feature extractors.

    Attributes:
        numerical_columns: numerical columns {column_name: (N,) ndarray}, merged into grid_gdf
        categorical_columns: categorical columns {column_name: (N,) str ndarray}, merged into grid_gdf
        array_features: high-dimensional feature matrix (N, d), not added to the GDF, saved/passed separately
        array_feature_names: names of the columns in array_features
        metadata: additional metadata (e.g. valid pixel count, processing statistics)
    """
    numerical_columns: Dict[str, np.ndarray] = field(default_factory=dict)
    categorical_columns: Dict[str, np.ndarray] = field(default_factory=dict)
    array_features: Optional[np.ndarray] = None
    array_feature_names: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseExtractor(ABC):
    """
    Base class for feature extractors.
    Each Extractor computes features for every grid point from already-downloaded
    data (local files provided by a Fetcher).
    """

    def __init__(self, config: dict):
        """
        Args:
            config: the config block for this extractor in the JSON config
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
        Compute features for every grid point.

        Args:
            grid_gdf: grid point GeoDataFrame (contains a geometry column)
            step_size_m: grid step size (meters)
            fetched_path: local file path produced by the corresponding Fetcher, None if there is no dependency

        Returns:
            ExtractorResult
        """
        raise NotImplementedError

    @abstractmethod
    def get_output_schema(self) -> dict:
        """
        Declare the output column names and types, used downstream by preprocess_features.

        Returns:
            {
                "numerical": ["col1", "col2", ...],
                "categorical": {"col_name": ["member1", "member2", ...]}
            }
        """
        raise NotImplementedError

    @property
    def fetcher_name(self) -> Optional[str]:
        """
        Declare the name of the Fetcher this extractor depends on, read from config["fetcher"].
        Extractors with no Fetcher dependency return None.
        """
        return self.config.get("fetcher")
