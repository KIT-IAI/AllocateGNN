"""Area-conserving GHSL Built-S extraction on the native Mollweide grid."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np

from .base import BaseExtractor, ExtractorResult
from .registry import extractor_registry


logger = logging.getLogger(__name__)


@extractor_registry.register("ghsl_built_s", description="native GHSL Built-S fraction")
class BuiltSurfaceExtractor(BaseExtractor):
    """Integrate native extensive Built-S mass over each square agent cell.

    Agent squares are first built in their actual country grid CRS, then
    transformed to the equal-area native Mollweide CRS.  Every intersecting
    native 100 m pixel contributes ``built_surface * overlap_area / 10000``.
    No reprojection/resampling of the extensive source band occurs.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.source_pixel_area_m2 = float(config.get("source_pixel_area_m2", 10_000.0))
        self.target_grid_crs = config.get("target_grid_crs")
        if not np.isfinite(self.source_pixel_area_m2) or self.source_pixel_area_m2 <= 0.0:
            raise ValueError("source_pixel_area_m2 must be finite and positive")
        if self.source_pixel_area_m2 != 10_000.0:
            raise ValueError("C/U/Z schema v1 requires the native 100 m GHSL pixel area")
        if not self.target_grid_crs:
            raise ValueError("target_grid_crs is required for exact agent-cell geometry")

    def extract(
        self,
        grid_gdf: gpd.GeoDataFrame,
        step_size_m: float,
        fetched_path: Optional[str] = None,
    ) -> ExtractorResult:
        if fetched_path is None:
            raise ValueError("BuiltSurfaceExtractor requires a native GHSL GeoTIFF path")
        if not np.isfinite(step_size_m) or step_size_m <= 0.0:
            raise ValueError("step_size_m must be finite and positive")
        if len(grid_gdf) == 0:
            raise ValueError("grid_gdf must not be empty")

        import rasterio
        from rasterio.crs import CRS
        from pyproj import Transformer
        import shapely
        from shapely.geometry import box
        from shapely.ops import transform as transform_geometry

        raster_path = Path(fetched_path).resolve()
        built_fraction = np.zeros(len(grid_gdf), dtype=np.float64)
        half = float(step_size_m) / 2.0
        max_closure_error = 0.0
        max_closure_tolerance_fraction = 0.0
        roundoff_count = 0
        max_roundoff_excess = 0.0

        with rasterio.open(raster_path) as src:
            transform = src.transform
            expected_crs = CRS.from_string("ESRI:54009")
            if (
                src.count != 3
                or src.dtypes != ("uint16", "uint16", "uint16")
                or src.nodata is not None
                or src.crs != expected_crs
                or transform.b != 0.0
                or transform.d != 0.0
                or transform.a != 100.0
                or transform.e != -100.0
            ):
                raise ValueError("GHSL crop must be the validated native 100 m uint16 grid")
            values = src.read(1, masked=False)
            written_from_tile = src.read(2, masked=False)
            official_valid = src.read(3, masked=False)
            from ..fetchers.ghsl_built_s_fetcher import OFFICIAL_NODATA

            if (
                values.dtype != np.uint16
                or np.any((official_valid == 1) & (values > 10_000))
                or np.any((written_from_tile != 0) & (written_from_tile != 1))
                or np.any((official_valid != 0) & (official_valid != 1))
                or np.any(official_valid > written_from_tile)
                or np.any((official_valid == 0) & (values != OFFICIAL_NODATA))
            ):
                raise ValueError("GHSL built_surface must be uint16 within [0, 10000]")
            grid_projected = grid_gdf.to_crs(self.target_grid_crs)
            xs = grid_projected.geometry.x.to_numpy(dtype=np.float64)
            ys = grid_projected.geometry.y.to_numpy(dtype=np.float64)
            if not np.isfinite(xs).all() or not np.isfinite(ys).all():
                raise ValueError("grid coordinates are non-finite in target_grid_crs")
            transformer = Transformer.from_crs(
                grid_projected.crs, src.crs, always_xy=True
            )

            for idx, (x, y) in enumerate(zip(xs, ys)):
                target_country = box(x - half, y - half, x + half, y + half)
                target_native = transform_geometry(transformer.transform, target_country)
                if target_native.is_empty or not target_native.is_valid:
                    raise ValueError(f"grid cell {idx} has invalid native geometry")
                target_area = float(target_native.area)
                if not np.isfinite(target_area) or target_area <= 0.0:
                    raise ValueError(f"grid cell {idx} has invalid native area")
                minx, miny, maxx, maxy = target_native.bounds
                if (
                    minx < src.bounds.left
                    or maxx > src.bounds.right
                    or miny < src.bounds.bottom
                    or maxy > src.bounds.top
                ):
                    raise ValueError(f"grid cell {idx} is outside the buffered GHSL crop")

                # Explicit outward snap: floor the first source index and ceil
                # the exclusive stop.  No round_offsets/round_lengths ambiguity.
                col0 = max(int(np.floor((minx - src.bounds.left) / 100.0)), 0)
                col1 = min(int(np.ceil((maxx - src.bounds.left) / 100.0)), src.width)
                row0 = max(int(np.floor((src.bounds.top - maxy) / 100.0)), 0)
                row1 = min(int(np.ceil((src.bounds.top - miny) / 100.0)), src.height)
                if col0 >= col1 or row0 >= row1:
                    raise ValueError(f"grid cell {idx} has no native GHSL overlap")

                cols, rows = np.meshgrid(
                    np.arange(col0, col1, dtype=np.float64),
                    np.arange(row0, row1, dtype=np.float64),
                )
                pixel_left = src.bounds.left + cols * 100.0
                pixel_top = src.bounds.top - rows * 100.0
                pixel_boxes = shapely.box(
                    pixel_left,
                    pixel_top - 100.0,
                    pixel_left + 100.0,
                    pixel_top,
                )
                overlap = shapely.area(shapely.intersection(pixel_boxes, target_native))
                overlap = np.asarray(overlap, dtype=np.float64)
                # GEOS overlay error scales with the coordinate magnitude as
                # well as polygon size.  This binary64-derived bound remains
                # orders of magnitude below one native pixel, so a missed edge
                # window cannot be hidden by it.
                coordinate_scale = max(abs(minx), abs(miny), abs(maxx), abs(maxy), 1.0)
                perimeter_scale = max(float(target_native.length), 1.0)
                tolerance = (
                    np.finfo(np.float64).eps
                    * max(overlap.size, 1)
                    * coordinate_scale
                    * perimeter_scale
                    * 128.0
                )
                closure_error = abs(float(np.sum(overlap, dtype=np.float64)) - target_area)
                if closure_error > tolerance:
                    raise ValueError(
                        f"grid cell {idx}: overlap closure error {closure_error} > {tolerance}"
                    )
                max_closure_error = max(max_closure_error, closure_error)
                max_closure_tolerance_fraction = max(
                    max_closure_tolerance_fraction, closure_error / tolerance
                )
                block = values[row0:row1, col0:col1].astype(np.float64, copy=False)
                block_written = written_from_tile[row0:row1, col0:col1]
                block_valid = official_valid[row0:row1, col0:col1]
                if np.any((block_written == 0) & (overlap > 0.0)):
                    raise ValueError(
                        f"grid cell {idx} intersects an unwritten GHSL crop cell"
                    )
                if np.any((block_valid == 0) & (overlap > 0.0)):
                    raise ValueError(
                        f"grid cell {idx} intersects official GHSL nodata"
                    )
                built_area = float(
                    np.sum(block * (overlap / self.source_pixel_area_m2), dtype=np.float64)
                )
                fraction = built_area / target_area
                if fraction > 1.0:
                    excess = fraction - 1.0
                    fraction_tolerance = (
                        tolerance / target_area + np.finfo(np.float64).eps * 128.0
                    )
                    if excess > fraction_tolerance:
                        raise ValueError(
                            f"grid cell {idx}: built_fraction {fraction} exceeds one "
                            f"beyond geometric tolerance {fraction_tolerance}"
                        )
                    roundoff_count += 1
                    max_roundoff_excess = max(max_roundoff_excess, excess)
                    fraction = 1.0
                built_fraction[idx] = fraction

        tolerance = np.finfo(np.float64).eps * 128.0
        if (
            not np.isfinite(built_fraction).all()
            or (built_fraction < 0.0).any()
            or (built_fraction > 1.0 + tolerance).any()
        ):
            raise ValueError(
                f"built_fraction outside [0,1]: min={built_fraction.min()}, "
                f"max={built_fraction.max()}"
            )
        return ExtractorResult(
            numerical_columns={"ghsl_built_fraction": built_fraction},
            metadata={
                "source_pixel_area_m2": self.source_pixel_area_m2,
                "aggregation": "native_extensive_exact_overlap",
                "target_grid_crs": str(self.target_grid_crs),
                "n_positive": int(np.count_nonzero(built_fraction > 0.0)),
                "upper_bound_roundoff_count": roundoff_count,
                "max_upper_bound_roundoff_excess": max_roundoff_excess,
                "max_overlap_closure_error_m2": float(max_closure_error),
                "max_overlap_tolerance_fraction": float(
                    max_closure_tolerance_fraction
                ),
            },
        )

    def get_output_schema(self) -> dict:
        return {"numerical": ["ghsl_built_fraction"], "categorical": {}}
