"""
SpectralExtractor — Sentinel-2 spectral index statistical features

For each grid point, crops a patch from the Sentinel-2 GeoTIFF,
computes spectral indices, then spatially aggregates them (configurable mean/std/median, etc.).

Performance optimizations:
- A single src.read call reads the bounding box of all points, followed by in-memory numpy slicing
- Formulas are precompiled in __init__ to avoid eval() parsing overhead in the hot loop
- Normalization detection is done only once (at the file level), not per patch
- valid_mask checks all bands
"""
import logging
from types import CodeType
from typing import Dict, List, Optional

import numpy as np
import geopandas as gpd

from .registry import extractor_registry
from .base import BaseExtractor, ExtractorResult

logger = logging.getLogger(__name__)


def _safe_normalized_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Safely compute the normalized difference index (a-b)/(a+b)"""
    numerator = a - b
    denominator = a + b
    return np.where(np.abs(denominator) < 1e-10, 0.0, numerator / denominator)


@extractor_registry.register("spectral", description="Sentinel-2 spectral index statistics")
class SpectralExtractor(BaseExtractor):
    """
    Extracts spectral index statistical features from Sentinel-2 GeoTIFF.

    Config fields:
        fetcher: name of the fetcher this depends on ("sentinel2")
        indices: index definitions {"NDVI": {"formula": "(B8-B4)/(B8+B4)"}, ...}
        aggregation: list of aggregation methods ["mean", "std", "median"]
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.indices: dict = config.get("indices", {})
        self.aggregation: List[str] = config.get("aggregation", ["mean", "std"])

        # Precompile formulas to avoid repeated eval() string parsing in the hot loop
        self._compiled_formulas: Dict[str, CodeType] = {}
        for idx_name, idx_conf in self.indices.items():
            formula = idx_conf.get("formula", "")
            try:
                self._compiled_formulas[idx_name] = compile(formula, f"<formula:{idx_name}>", "eval")
            except SyntaxError as e:
                logger.warning(f"Failed to compile formula '{idx_name}': {formula} → {e}")

    def _eval_compiled(self, idx_name: str, band_map: Dict[str, np.ndarray]) -> np.ndarray:
        """Execute the formula using the precompiled code object"""
        code_obj = self._compiled_formulas.get(idx_name)
        if code_obj is None:
            return np.zeros_like(next(iter(band_map.values())))

        safe_env = {"__builtins__": {"__import__": __import__}}
        result = eval(code_obj, safe_env, band_map)

        if isinstance(result, np.ndarray):
            result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        return result

    def extract(
        self,
        grid_gdf: gpd.GeoDataFrame,
        step_size_m: float,
        fetched_path: Optional[str] = None,
    ) -> ExtractorResult:
        """
        Crop a Sentinel-2 patch for each grid point, compute spectral indices, and aggregate.

        Optimization strategy:
        1. A single src.read call reads the bounding box region into memory
        2. numpy slicing extracts each point's patch, with zero additional disk I/O
        3. Normalization detection is done only once
        4. Formulas are precompiled

        Returns:
            ExtractorResult:
            - array_features: (N, n_indices × n_aggregations)
            - array_feature_names: [f"{idx}_{agg}" for ...]
        """
        if fetched_path is None:
            raise ValueError("SpectralExtractor requires fetched_path (GeoTIFF path)")

        n_points = len(grid_gdf)
        n_indices = len(self.indices)
        n_aggs = len(self.aggregation)
        feature_dim = n_indices * n_aggs

        features = np.zeros((n_points, feature_dim), dtype=np.float64)
        valid_counts = np.zeros(n_points, dtype=np.int32)

        import rasterio
        from rasterio.windows import Window
        index_names = list(self.indices.keys())

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

            # Convert to pixel coordinates
            region_col_off = (all_minx - t.c) / t.a
            region_row_off = (all_maxy - t.f) / t.e
            region_col_size = (all_maxx - all_minx) / abs(t.a)
            region_row_size = (all_maxy - all_miny) / abs(t.e)

            region_window = Window(region_col_off, region_row_off, region_col_size, region_row_size)

            # Single I/O call: read the entire bounding box region
            region_data = src.read(
                window=region_window,
                boundless=True,
                fill_value=np.nan,
            )
            # region_data: (C, H_region, W_region)

            # Normalization detection: done only once (at the file level)
            needs_rescale = False
            sample = region_data[:, ::10, ::10]  # sparse sampling for detection
            valid_sample = sample[np.isfinite(sample)]
            if len(valid_sample) > 0 and np.max(valid_sample) > 1.0:
                needs_rescale = True

            # Pixel offset of each point (relative to the region's top-left corner)
            # World coordinate of point i's patch top-left corner = (xs[i] - half, ys[i] + half)
            # Pixel offset relative to the region's top-left corner
            point_col_starts = ((xs - half - all_minx) / abs(t.a)).astype(np.int64)
            point_row_starts = ((all_maxy - (ys + half)) / abs(t.e)).astype(np.int64)

            _, region_h, region_w = region_data.shape

            for i in range(n_points):
                r0 = int(point_row_starts[i])
                c0 = int(point_col_starts[i])
                r1 = r0 + patch_pixels
                c1 = c0 + patch_pixels

                # Boundary check: cases requiring padding
                if r0 < 0 or c0 < 0 or r1 > region_h or c1 > region_w:
                    # Fall back to a per-point window read (boundary cases are rare)
                    minx_i = xs[i] - half
                    maxy_i = ys[i] + half
                    col_off_i = (minx_i - t.c) / t.a
                    row_off_i = (maxy_i - t.f) / t.e
                    col_size_i = step_size_m / abs(t.a)
                    row_size_i = step_size_m / abs(t.e)
                    win_i = Window(col_off_i, row_off_i, col_size_i, row_size_i)
                    try:
                        patch_data = src.read(
                            window=win_i, boundless=True, fill_value=np.nan,
                            out_shape=(n_bands, patch_pixels, patch_pixels),
                        )
                    except Exception as e:
                        logger.warning(f"Failed to crop point {i} ({xs[i]:.1f}, {ys[i]:.1f}): {e}")
                        continue
                else:
                    patch_data = region_data[:, r0:r1, c0:c1]

                # (C, H, W) → (H, W, C)
                patch = np.transpose(patch_data, (1, 2, 0)).astype(np.float64)

                # Check valid pixels across all bands
                valid_mask = np.all(np.isfinite(patch), axis=2)
                n_valid = int(np.sum(valid_mask))
                valid_counts[i] = n_valid

                if n_valid < 10:
                    continue

                # Normalize (file-level flag, no longer detected per patch)
                if needs_rescale:
                    patch = patch / 10000.0

                # Build the band map
                band_map = {
                    "B2": patch[:, :, 0],
                    "B3": patch[:, :, 1],
                    "B4": patch[:, :, 2],
                    "B8": patch[:, :, 3],
                    "B11": patch[:, :, 4],
                    "B12": patch[:, :, 5],
                }

                # Compute spectral indices (precompiled formulas)
                feat_vec = []
                for idx_name in index_names:
                    idx_data = self._eval_compiled(idx_name, band_map)

                    # _eval_compiled already applies nan_to_num, no need to filter finite values again
                    flat = idx_data.ravel()

                    if len(flat) < 10:
                        feat_vec.extend([0.0] * n_aggs)
                        continue

                    for agg in self.aggregation:
                        if agg == "mean":
                            feat_vec.append(float(np.mean(flat)))
                        elif agg == "std":
                            feat_vec.append(float(np.std(flat)))
                        elif agg == "median":
                            feat_vec.append(float(np.median(flat)))
                        elif agg == "q25":
                            feat_vec.append(float(np.percentile(flat, 25)))
                        elif agg == "q75":
                            feat_vec.append(float(np.percentile(flat, 75)))
                        else:
                            logger.warning(f"Unknown aggregation method: {agg}")
                            feat_vec.append(0.0)

                features[i] = feat_vec

        # Generate feature names
        feature_names = [
            f"{idx_name}_{agg}"
            for idx_name in index_names
            for agg in self.aggregation
        ]

        return ExtractorResult(
            array_features=features,
            array_feature_names=feature_names,
            metadata={
                "n_indices": n_indices,
                "n_aggregations": n_aggs,
                "feature_dim": feature_dim,
                "valid_pixel_counts": valid_counts.tolist(),
            },
        )

    def get_output_schema(self) -> dict:
        """Dynamically generate the schema from indices × aggregation"""
        index_names = list(self.indices.keys())
        return {
            "numerical": [],
            "categorical": {},
            "array_features": [
                f"{idx}_{agg}"
                for idx in index_names
                for agg in self.aggregation
            ],
        }
