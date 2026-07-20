"""
LanduseExtractor — OSM land-use categories + area proportions

Logic ported from SpatialAllocation/utils/CalcuLanduse.py
Key change: category mapping is now read from JSON config instead of being hardcoded
"""
import logging
import pickle
from typing import Dict, List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

from .registry import extractor_registry
from .base import BaseExtractor, ExtractorResult

logger = logging.getLogger(__name__)


@extractor_registry.register("landuse", description="OSM land-use categories + area proportions")
class LanduseExtractor(BaseExtractor):
    """
    Extracts land-use features from OSM data.

    Config fields:
        fetcher: name of the fetcher this depends on ("osm")
        categories: category mapping dict {"category_name": ["osm_tag1", "osm_tag2", ...]}
        default_category: default category (used when no category matches)
        mode: feature mode, "proportions" (area proportion) or "point" (point query), default "proportions"
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.categories: Dict[str, List[str]] = config.get("categories", {})
        self.default_category: str = config.get("default_category", "others")
        self.mode: str = config.get("mode", "proportions")

        # Build the reverse lookup table: osm_tag → category
        self._tag_to_category: Dict[str, str] = {}
        for category, tags in self.categories.items():
            for tag in tags:
                self._tag_to_category[tag] = category

    def _classify(self, landuse_value) -> str:
        """
        Map an OSM landuse tag value to the configured category.

        Equivalent to get_category() in the original CalcuLanduse.py,
        but the category mapping is now read from JSON config.
        """
        if pd.isna(landuse_value):
            return self.default_category

        category = self._tag_to_category.get(landuse_value)
        if category is not None:
            return category

        logger.debug(f"Landuse '{landuse_value}' did not match any category, assigned to '{self.default_category}'")
        return self.default_category

    def extract(
        self,
        grid_gdf: gpd.GeoDataFrame,
        step_size_m: float,
        fetched_path: Optional[str] = None,
    ) -> ExtractorResult:
        """
        Compute land-use features from OSM data.

        Args:
            grid_gdf: grid point GeoDataFrame (CRS=EPSG:4326)
            step_size_m: grid step size (meters)
            fetched_path: pickle path produced by OsmFetcher

        Returns:
            ExtractorResult, containing:
            - categorical_columns: {"landuse": (N,) str}  point query mode
            - numerical_columns: {"lu_{cat}_prop": (N,) float}  proportion mode
        """
        if fetched_path is None:
            raise ValueError("LanduseExtractor requires fetched_path (OSM pickle path)")

        # Load OSM data
        with open(fetched_path, "rb") as f:
            osm_gdf = pickle.load(f)

        if self.mode == "point":
            return self._extract_point(grid_gdf, osm_gdf)
        else:
            return self._extract_proportions(grid_gdf, osm_gdf, step_size_m)

    def _extract_point(
        self, grid_gdf: gpd.GeoDataFrame, osm_gdf: gpd.GeoDataFrame
    ) -> ExtractorResult:
        """
        Point query mode: the landuse category the grid point falls within.
        Ported from the sjoin logic in CalcuLanduse.fetch_landuse_data().
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
        Area proportion mode: the area proportion of each category within every grid cell.
        Ported from the overlay logic in CalcuLanduse.calculate_landuse_proportions().
        """
        # Prepare OSM data
        osm_gdf = osm_gdf[["landuse", "geometry"]].dropna(subset=["landuse"])
        osm_gdf = osm_gdf.copy()
        osm_gdf["landuse"] = osm_gdf["landuse"].apply(self._classify)
        osm_m = osm_gdf.to_crs("EPSG:3857")

        # Keep only Polygon/MultiPolygon (OSM data often mixes points/lines/polygons)
        osm_m = osm_m[osm_m.geom_type.isin(["Polygon", "MultiPolygon"])]

        # Remove duplicate geometries (keep only one copy of the same polygon, to avoid double-counting in overlay)
        osm_m = osm_m.drop_duplicates(subset=["geometry"])

        # Dissolve polygons of the same category to eliminate same-category spatial overlap (avoids prop > 1.0)
        osm_m = osm_m.dissolve(by="landuse").explode(index_parts=False).reset_index()

        if osm_m.empty:
            logger.warning("OSM data is empty after filtering, all proportions set to 0")
            return self._empty_proportions(len(grid_gdf))

        # Convert grid points into grid cells (square polygons)
        grid_m = grid_gdf.to_crs("EPSG:3857").copy()
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

        # overlay to compute intersections
        intersection_gdf = gpd.overlay(grid_m[["grid_id", "geometry"]], osm_m, how="intersection")
        intersection_gdf["intersection_area"] = intersection_gdf.geometry.area

        # Aggregate
        landuse_areas = intersection_gdf.pivot_table(
            index="grid_id",
            columns="landuse",
            values="intersection_area",
            aggfunc="sum",
        ).fillna(0)

        # Compute proportions
        cell_area = step_size_m ** 2
        landuse_proportions = landuse_areas / cell_area

        # Ensure every configured category has a column
        category_names = list(self.categories.keys())
        result_numerical = {}
        for cat in category_names:
            col_name = f"lu_{cat}_prop"
            if cat in landuse_proportions.columns:
                # Align to the original grid_gdf index by grid_id
                values = landuse_proportions[cat].reindex(grid_gdf.index, fill_value=0.0).values
            else:
                values = np.zeros(len(grid_gdf))
            result_numerical[col_name] = values

        return ExtractorResult(
            numerical_columns=result_numerical,
            metadata={
                "mode": "proportions",
                "num_osm_records": len(osm_gdf),
                "num_intersections": len(intersection_gdf) if not osm_m.empty else 0,
            },
        )

    def _empty_proportions(self, n: int) -> ExtractorResult:
        """Return all-zero proportions when OSM data is empty"""
        category_names = list(self.categories.keys())
        return ExtractorResult(
            numerical_columns={f"lu_{cat}_prop": np.zeros(n) for cat in category_names},
            metadata={"mode": "proportions", "num_osm_records": 0},
        )

    def get_output_schema(self) -> dict:
        """Dynamically generate the schema from the configured categories"""
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
