"""
NtlExtractor — VIIRS nighttime lights per-point sampled features

Extracts the radiance value at each grid point from the NTL GeoTIFF.
NTL has a native resolution of ~500m, so rasterio.sample() is used for
per-point sampling instead of the heavier patch-crop-and-aggregate pattern.

Two modes:
- Pure point sampling (radius_m=0, default): read the pixel value directly at each grid point
- Neighborhood aggregation (radius_m>0): compute mean/std within a radius_m neighborhood
"""
import logging
from typing import List, Optional

import numpy as np
import geopandas as gpd

from .registry import extractor_registry
from .base import BaseExtractor, ExtractorResult

logger = logging.getLogger(__name__)


@extractor_registry.register("ntl", description="VIIRS nighttime lights per-point sampling")
class NtlExtractor(BaseExtractor):
    """
    Extracts radiance value features from the NTL GeoTIFF.

    Config fields:
        fetcher: name of the fetcher this depends on ("ntl")
        radius_m: neighborhood radius (default 0 = pure point sampling)
        aggregation: aggregation methods used when radius_m > 0 (default ["mean", "std"])
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
        Extract the NTL radiance value for every grid point.

        Mode A (radius_m == 0): per-point sampling via rasterio.sample()
        Mode B (radius_m > 0): read the entire raster, then per-point neighborhood aggregation

        Returns:
            ExtractorResult:
            - numerical_columns: {"ntl_value": (N,)} or {"ntl_mean": ..., "ntl_std": ...}
        """
        if fetched_path is None:
            raise ValueError("NtlExtractor requires fetched_path (GeoTIFF path)")

        import rasterio

        n_points = len(grid_gdf)

        if self.radius_m <= 0:
            # Mode A: pure point sampling
            numerical_columns = self._extract_point_sample(
                grid_gdf, fetched_path, n_points
            )
        else:
            # Mode B: neighborhood aggregation
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
        """Pure point sampling: rasterio.sample() reads the pixel value directly at each grid point"""
        import rasterio

        with rasterio.open(fetched_path) as src:
            grid_proj = grid_gdf.to_crs(src.crs)
            coords = list(zip(grid_proj.geometry.x, grid_proj.geometry.y))
            samples = np.array(list(src.sample(coords)))  # (N, 1)
            values = samples[:, 0].astype(np.float64)  # (N,)

        # NaN/negative values → 0 (VIIRS dark regions may contain negative instrument noise)
        valid_mask = np.isfinite(values) & (values >= 0)
        values = np.where(valid_mask, values, 0.0)

        n_invalid = n_points - np.sum(valid_mask)
        if n_invalid > 0:
            logger.info(f"NTL point sampling: {n_invalid}/{n_points} points were invalid, set to zero")

        return {"ntl_value": values}

    def _extract_neighborhood(
        self, grid_gdf: gpd.GeoDataFrame, fetched_path: str, n_points: int
    ) -> dict:
        """Neighborhood aggregation: read the entire raster, then compute statistics within a radius_m neighborhood at each point"""
        import rasterio

        with rasterio.open(fetched_path) as src:
            grid_proj = grid_gdf.to_crs(src.crs)
            raster_data = src.read(1).astype(np.float64)  # (H, W)
            transform = src.transform
            pixel_size = src.res[0]  # assumes square pixels

        # Number of pixels corresponding to the neighborhood radius
        radius_px = max(1, int(round(self.radius_m / pixel_size)))

        xs = grid_proj.geometry.x.values
        ys = grid_proj.geometry.y.values
        h, w = raster_data.shape

        # Pre-allocate results
        results = {agg: np.zeros(n_points, dtype=np.float64) for agg in self.aggregation}

        for i in range(n_points):
            # Convert point coordinates to pixel coordinates
            col = int((xs[i] - transform.c) / transform.a)
            row = int((ys[i] - transform.f) / transform.e)

            # Crop the neighborhood window
            r0 = max(0, row - radius_px)
            r1 = min(h, row + radius_px + 1)
            c0 = max(0, col - radius_px)
            c1 = min(w, col + radius_px + 1)

            patch = raster_data[r0:r1, c0:c1]

            # Filter out invalid values
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
                    logger.warning(f"Unknown aggregation method: {agg}, skipping")

        return {f"ntl_{agg}": results[agg] for agg in self.aggregation}

    def get_output_schema(self) -> dict:
        """Dynamically generate the schema from radius_m and aggregation"""
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
