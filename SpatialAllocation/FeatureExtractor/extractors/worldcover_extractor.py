"""
WorldCoverExtractor — ESA WorldCover pixel proportion features

For each grid point, crops a patch from the WorldCover GeoTIFF,
tallies the per-category pixel proportion, and aggregates via aggregation_mapping.

Performance optimizations:
- A single src.read call reads the bounding box region, followed by in-memory numpy slicing
- np.bincount + clip replaces a per-category loop
- A pre-built aggregation matrix replaces a triple-nested loop via matrix multiplication
"""
import logging
from typing import Dict, List, Optional

import numpy as np
import geopandas as gpd

from .registry import extractor_registry
from .base import BaseExtractor, ExtractorResult

logger = logging.getLogger(__name__)


@extractor_registry.register("worldcover", description="WorldCover pixel proportion")
class WorldCoverExtractor(BaseExtractor):
    """
    Extracts pixel proportion features from the WorldCover GeoTIFF.

    Config fields:
        fetcher: name of the fetcher this depends on ("worldcover")
        classes: WorldCover class mapping {"pixel_value": "class_name"}
        aggregation_mapping: aggregation rules {"group_name": [pixel_value_1, ...]}
    """

    def __init__(self, config: dict):
        super().__init__(config)
        # classes: {"10": "tree_cover", "20": "shrubland", ...}
        self.classes: Dict[str, str] = config.get("classes", {})
        # aggregation_mapping: {"built_up": [50], "agricultural": [10, 20, 30, 40], ...}
        self.aggregation_mapping: Dict[str, List[int]] = config.get("aggregation_mapping", {})

        # Build the pixel_value → class_name lookup table (integer keys)
        self._value_to_class: Dict[int, str] = {
            int(k): v for k, v in self.classes.items()
        }
        # All known pixel values
        self._all_values: List[int] = sorted(self._value_to_class.keys())

        # Pre-build the pixel_value → column index mapping
        self._value_to_col: Dict[int, int] = {v: j for j, v in enumerate(self._all_values)}

        # Pre-build the aggregation matrix (n_classes, n_groups), replacing a loop with matrix multiplication
        n_classes = len(self._all_values)
        group_names = list(self.aggregation_mapping.keys())
        n_groups = len(group_names)
        self._agg_matrix = np.zeros((n_classes, n_groups), dtype=np.float64)
        for g, group in enumerate(group_names):
            for val in self.aggregation_mapping[group]:
                if val in self._value_to_col:
                    self._agg_matrix[self._value_to_col[val], g] = 1.0

        # bincount's minlength: covers all known pixel values + 1
        self._bincount_minlen = max(self._all_values) + 1 if self._all_values else 1

    def extract(
        self,
        grid_gdf: gpd.GeoDataFrame,
        step_size_m: float,
        fetched_path: Optional[str] = None,
    ) -> ExtractorResult:
        """
        Crop a patch for each grid point and compute the WorldCover per-category pixel proportion.

        Optimization strategy:
        1. A single src.read call reads the bounding box into memory
        2. numpy slicing extracts each point's patch
        3. np.bincount + clip performs a single-pass tally
        4. Matrix multiplication aggregation

        Returns:
            ExtractorResult:
            - numerical_columns: {f"wc_{group}_ratio": (N,)} aggregated proportions
            - array_features: (N, len(classes)) raw per-category proportions
            - array_feature_names: list of class names
        """
        if fetched_path is None:
            raise ValueError("WorldCoverExtractor requires fetched_path (GeoTIFF path)")

        n_points = len(grid_gdf)
        n_classes = len(self._all_values)

        # Raw per-category proportions (N, n_classes)
        raw_proportions = np.zeros((n_points, n_classes), dtype=np.float64)

        import rasterio
        from rasterio.windows import Window

        with rasterio.open(fetched_path) as src:
            tif_crs = src.crs
            grid_proj = grid_gdf.to_crs(tif_crs)

            t = src.transform
            pixel_size_x = src.res[0]
            n_bands = src.count

            xs = grid_proj.geometry.x.values
            ys = grid_proj.geometry.y.values

            half = step_size_m / 2.0
            patch_pixels = int(round(step_size_m / pixel_size_x))

            # Compute the bounding box covering all points
            all_minx = xs.min() - half
            all_maxy = ys.max() + half
            all_maxx = xs.max() + half
            all_miny = ys.min() - half

            region_col_off = (all_minx - t.c) / t.a
            region_row_off = (all_maxy - t.f) / t.e
            region_col_size = (all_maxx - all_minx) / abs(t.a)
            region_row_size = (all_maxy - all_miny) / abs(t.e)

            region_window = Window(region_col_off, region_row_off, region_col_size, region_row_size)

            # Single I/O call: read the entire bounding box
            region_data = src.read(
                window=region_window,
                boundless=True,
                fill_value=0,
            )
            # region_data: (C, H_region, W_region) — WorldCover is usually C=1

            # Pixel offset of each point
            point_col_starts = ((xs - half - all_minx) / abs(t.a)).astype(np.int64)
            point_row_starts = ((all_maxy - (ys + half)) / abs(t.e)).astype(np.int64)

            _, region_h, region_w = region_data.shape

            # Take the first band (WorldCover is single-band)
            if region_data.shape[0] >= 1:
                region_band = region_data[0]
            else:
                region_band = region_data.reshape(region_h, region_w)

            for i in range(n_points):
                r0 = int(point_row_starts[i])
                c0 = int(point_col_starts[i])
                r1 = r0 + patch_pixels
                c1 = c0 + patch_pixels

                if r0 < 0 or c0 < 0 or r1 > region_h or c1 > region_w:
                    # Boundary fallback: per-point window read
                    minx_i = xs[i] - half
                    maxy_i = ys[i] + half
                    col_off_i = (minx_i - t.c) / t.a
                    row_off_i = (maxy_i - t.f) / t.e
                    col_size_i = step_size_m / abs(t.a)
                    row_size_i = step_size_m / abs(t.e)
                    win_i = Window(col_off_i, row_off_i, col_size_i, row_size_i)
                    try:
                        patch_data = src.read(
                            window=win_i, boundless=True, fill_value=0,
                            out_shape=(n_bands, patch_pixels, patch_pixels),
                        )
                        patch = patch_data[0]
                    except Exception as e:
                        logger.warning(f"Failed to crop point {i} ({xs[i]:.1f}, {ys[i]:.1f}): {e}")
                        continue
                else:
                    patch = region_band[r0:r1, c0:c1]

                total_pixels = patch.size
                if total_pixels < 10:
                    logger.warning(f"Point {i}: insufficient valid pixels ({total_pixels})")
                    continue

                # clip to [0, bincount_minlen-1] to prevent dirty data (e.g. nodata=255) from inflating bincount
                flat = np.clip(patch.ravel(), 0, self._bincount_minlen - 1).astype(np.intp)
                counts = np.bincount(flat, minlength=self._bincount_minlen)
                for val, j in self._value_to_col.items():
                    raw_proportions[i, j] = counts[val] / total_pixels

        # Matrix multiplication aggregation: (N, n_classes) @ (n_classes, n_groups) → (N, n_groups)
        agg_proportions = raw_proportions @ self._agg_matrix

        # Build the result
        group_names = list(self.aggregation_mapping.keys())
        numerical_columns = {
            f"wc_{group}_ratio": agg_proportions[:, g]
            for g, group in enumerate(group_names)
        }

        class_names = [self._value_to_class[v] for v in self._all_values]

        return ExtractorResult(
            numerical_columns=numerical_columns,
            array_features=raw_proportions,
            array_feature_names=class_names,
            metadata={
                "n_classes": n_classes,
                "n_groups": len(group_names),
            },
        )

    def get_output_schema(self) -> dict:
        """Dynamically generate the schema from aggregation_mapping"""
        group_names = list(self.aggregation_mapping.keys())
        return {
            "numerical": [f"wc_{group}_ratio" for group in group_names],
            "categorical": {},
        }
