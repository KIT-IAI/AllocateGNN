"""
FeatureOrchestrator - orchestrates the complete Fetcher + Extractor pipeline

Modeled after the CombinedLoss pattern: select/combine modules by configuration
and execute them uniformly.
"""
import logging
from typing import Any, Dict, Optional, Tuple

import geopandas as gpd
from shapely.ops import unary_union

from .fetchers import fetcher_registry, BaseFetcher
from .extractors import extractor_registry, BaseExtractor, ExtractorResult
from .grid_generator import calculate_step_size, generate_base_grid

logger = logging.getLogger(__name__)


class FeatureOrchestrator:
    """
    Feature pipeline orchestrator.

    Accepts a region GeoDataFrame + JSON configuration and executes the
    following flow:
    1. Compute step_size -> generate grid points
    2. Iterate over the enabled fetchers -> download/cache data
    3. Iterate over the enabled extractors -> compute features
    4. Merge all features into grid_gdf
    """

    def __init__(self, config: dict):
        """
        Args:
            config: the full JSON configuration dict, including
                grid/cache_dir/fetchers/extractors and other fields
        """
        self.config = config
        self.grid_config = config.get("grid", {})
        self.cache_dir = config.get("cache_dir", "./cache")

        # Instantiate the enabled fetchers
        self.fetchers: Dict[str, BaseFetcher] = {}
        fetchers_config = config.get("fetchers", {})
        for name, fc in fetchers_config.items():
            if not fc.get("enabled", False):
                continue
            fetcher_cls = fetcher_registry.get_fetcher(name)
            if fetcher_cls is None:
                logger.warning(f"Fetcher '{name}' is enabled but not registered in the registry, skipping")
                continue
            self.fetchers[name] = fetcher_cls(fc)

        # Instantiate the enabled extractors
        self.extractors: Dict[str, BaseExtractor] = {}
        extractors_config = config.get("extractors", {})
        for name, ec in extractors_config.items():
            if not ec.get("enabled", False):
                continue
            extractor_cls = extractor_registry.get_extractor(name)
            if extractor_cls is None:
                logger.warning(f"Extractor '{name}' is enabled but not registered in the registry, skipping")
                continue
            self.extractors[name] = extractor_cls(ec)

    def run(
        self,
        region_gdf: gpd.GeoDataFrame,
    ) -> Tuple[gpd.GeoDataFrame, dict, Dict[str, ExtractorResult]]:
        """
        Executes the complete feature pipeline.

        Args:
            region_gdf: the region polygon GeoDataFrame (e.g. ITL3 regions)

        Returns:
            (grid_gdf, feature_schema, results)
            - grid_gdf: the grid point GeoDataFrame with all feature columns
            - feature_schema: feature metadata {step_size_m, numerical_col_names,
              categorical_col_members, array_features}
            - results: {extractor_name: ExtractorResult} the raw results dict
        """
        # === 1. Generate the grid ===
        target_points = self.grid_config.get("target_points", 50000)
        min_step = self.grid_config.get("min_step_size", 10)
        crs_project = self.grid_config.get("crs_project", "EPSG:3857")

        step_size_m = calculate_step_size(target_points, region_gdf, min_step)
        logger.info(f"Grid step size: {step_size_m}m")

        grid_gdf = generate_base_grid(region_gdf, step_size_m, crs_project)
        logger.info(f"Generated grid points: {len(grid_gdf)}")

        # === 2. Run the Fetchers ===
        # Compute the region boundary (EPSG:4326)
        region_4326 = region_gdf.to_crs("EPSG:4326")
        region_bounds = unary_union(region_4326["geometry"])

        fetched_paths: Dict[str, str] = {}
        for name, fetcher in self.fetchers.items():
            logger.info(f"Fetcher '{name}' starting data fetch...")
            path = fetcher.fetch(region_bounds, self.cache_dir)
            fetched_paths[name] = path
            logger.info(f"Fetcher '{name}' complete: {path}")

        # === 3. Run the Extractors ===
        results: Dict[str, ExtractorResult] = {}
        all_numerical_cols = []
        all_categorical_members = {}

        for name, extractor in self.extractors.items():
            # Find the output path of the corresponding fetcher
            dep_fetcher = extractor.fetcher_name
            fetched_path = fetched_paths.get(dep_fetcher) if dep_fetcher else None

            logger.info(f"Extractor '{name}' starting feature computation...")
            result = extractor.extract(grid_gdf, step_size_m, fetched_path)
            results[name] = result

            # Merge numerical columns into grid_gdf
            for col_name, col_data in result.numerical_columns.items():
                grid_gdf[col_name] = col_data

            # Merge categorical columns into grid_gdf
            for col_name, col_data in result.categorical_columns.items():
                grid_gdf[col_name] = col_data

            # Collect schema
            schema = extractor.get_output_schema()
            all_numerical_cols.extend(schema.get("numerical", []))
            all_categorical_members.update(schema.get("categorical", {}))

            logger.info(f"Extractor '{name}' complete")

        # === 4. Build feature_schema ===
        feature_schema = {
            "step_size_m": step_size_m,
            "numerical_col_names": all_numerical_cols,
            "categorical_col_members": all_categorical_members,
            "array_features": {
                name: {
                    "shape": r.array_features.shape if r.array_features is not None else None,
                    "names": r.array_feature_names,
                }
                for name, r in results.items()
                if r.array_features is not None
            },
        }

        return grid_gdf, feature_schema, results
