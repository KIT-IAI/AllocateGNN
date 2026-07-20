# -*- coding: utf-8 -*-
"""
Spectral index computation and spatial aggregation module

This module provides a complete pipeline for extracting spectral index
features from multi-band Sentinel-2 GeoTIFFs, including GeoTIFF quality
validation, per-pixel spectral index computation, spatial aggregation, and
batch feature extraction.

Spectral index definitions (per pixel):
    NDVI = (B8 - B4) / (B8 + B4)      Normalized Difference Vegetation Index
    NDBI = (B11 - B8) / (B11 + B8)    Normalized Difference Built-up Index
    NDWI = (B3 - B8) / (B3 + B8)      Normalized Difference Water Index
    BSI  = ((B11+B4) - (B8+B2)) / ((B11+B4) + (B8+B2))  Bare Soil Index
    UI   = (B12 - B8) / (B12 + B8)    Urban Index

Band order convention: (H, W, 6) -> [B2, B3, B4, B8, B11, B12]
"""

from typing import Optional, Tuple, Dict, Any, Union
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# GeoTIFF quality validation
# =============================================================================

def validate_mosaic(mosaic_path: str) -> Dict[str, Any]:
    """
    Validate the quality of a GeoTIFF mosaic.

    Checks the CRS, band count, pixel value range, and nodata pixel ratio.

    Args:
        mosaic_path: GeoTIFF file path

    Returns:
        Dict[str, Any]: Validation result dictionary, containing:
            - crs_valid (bool): Whether the CRS is a projected CRS
            - crs_name (str): CRS name
            - bands_valid (bool): Whether the band count is 6
            - band_count (int): Actual band count
            - values_valid (bool): Whether pixel values fall within [0, 1] (excluding nodata)
            - value_min (float): Minimum value among valid pixels
            - value_max (float): Maximum value among valid pixels
            - nodata_ratio (float): Nodata pixel ratio (0-1)
            - all_valid (bool): Whether all checks passed
    """
    import rasterio

    result = {
        'crs_valid': False,
        'crs_name': '',
        'bands_valid': False,
        'band_count': 0,
        'values_valid': False,
        'value_min': float('nan'),
        'value_max': float('nan'),
        'nodata_ratio': float('nan'),
        'all_valid': False,
    }

    with rasterio.open(mosaic_path) as src:
        # Check whether the CRS is a projected CRS
        crs = src.crs
        result['crs_name'] = str(crs)
        result['crs_valid'] = crs.is_projected

        # Check the band count
        result['band_count'] = src.count
        result['bands_valid'] = (src.count == 6)

        # Read the data to check the pixel value range
        data = src.read()  # (C, H, W)
        valid_mask = np.isfinite(data)
        total_pixels = data.size
        nodata_pixels = np.sum(~valid_mask)

        result['nodata_ratio'] = float(nodata_pixels / total_pixels) if total_pixels > 0 else 0.0

        if valid_mask.any():
            valid_data = data[valid_mask]
            result['value_min'] = float(np.min(valid_data))
            result['value_max'] = float(np.max(valid_data))
            # Check whether values fall within [0, 1] (allowing a small floating-point tolerance)
            result['values_valid'] = (result['value_min'] >= -0.01 and result['value_max'] <= 1.01)

    result['all_valid'] = (
        result['crs_valid'] and
        result['bands_valid'] and
        result['values_valid']
    )

    return result


# =============================================================================
# Per-pixel spectral index computation
# =============================================================================

def _safe_normalized_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Safely compute a normalized difference index (a - b) / (a + b).

    Returns 0 when the denominator is zero.

    Args:
        a: Array for the positive term of the numerator
        b: Array for the negative term of the numerator

    Returns:
        np.ndarray: Normalized difference index, value range [-1, 1]
    """
    numerator = a - b
    denominator = a + b
    # Return 0 when the denominator is zero
    result = np.where(
        np.abs(denominator) < 1e-10,
        0.0,
        numerator / denominator
    )
    return result


def compute_spectral_indices(bands_array: np.ndarray) -> np.ndarray:
    """
    Compute five spectral indices per pixel.

    Takes a six-band array and computes NDVI, NDBI, NDWI, BSI, and UI per pixel.

    Args:
        bands_array: NumPy array of shape (H, W, 6)
            Band order: B2(blue), B3(green), B4(red), B8(NIR), B11(SWIR1), B12(SWIR2)

    Returns:
        np.ndarray: Array of shape (H, W, 5)
            Channel order: NDVI, NDBI, NDWI, BSI, UI
            Value range [-1, 1]; 0 when the denominator is zero

    Note:
        - If the input values are > 1.0, they are automatically divided by
          10000 and a warning is printed (a safeguard against forgetting to normalize)
        - NaN values in the input are preserved in the output
    """
    bands = bands_array.astype(np.float64)

    # Automatically detect and correct un-normalized DN values
    valid_values = bands[np.isfinite(bands)]
    if len(valid_values) > 0 and np.max(valid_values) > 1.0:
        warnings.warn(
            f"Input value range [{np.min(valid_values):.1f}, {np.max(valid_values):.1f}] exceeds 1.0; "
            f"automatically dividing by 10000 to normalize. Please confirm whether these are raw Sentinel-2 DN values.",
            UserWarning,
            stacklevel=2,
        )
        bands = bands / 10000.0

    # Extract each band
    B2 = bands[:, :, 0]   # Blue
    B3 = bands[:, :, 1]   # Green
    B4 = bands[:, :, 2]   # Red
    B8 = bands[:, :, 3]   # Near-infrared (NIR)
    B11 = bands[:, :, 4]  # Short-wave infrared (SWIR1)
    B12 = bands[:, :, 5]  # Short-wave infrared (SWIR2)

    H, W = bands.shape[:2]
    indices = np.empty((H, W, 5), dtype=np.float64)

    # NDVI = (B8 - B4) / (B8 + B4)
    indices[:, :, 0] = _safe_normalized_difference(B8, B4)

    # NDBI = (B11 - B8) / (B11 + B8)
    indices[:, :, 1] = _safe_normalized_difference(B11, B8)

    # NDWI = (B3 - B8) / (B3 + B8)
    indices[:, :, 2] = _safe_normalized_difference(B3, B8)

    # BSI = ((B11 + B4) - (B8 + B2)) / ((B11 + B4) + (B8 + B2))
    indices[:, :, 3] = _safe_normalized_difference(B11 + B4, B8 + B2)

    # UI = (B12 - B8) / (B12 + B8)
    indices[:, :, 4] = _safe_normalized_difference(B12, B8)

    return indices


# =============================================================================
# Spatial aggregation
# =============================================================================

def aggregate_spectral_patch(
    indices_array: np.ndarray,
    extended: bool = False
) -> Tuple[np.ndarray, int]:
    """
    Spatially aggregate a spectral index patch into a feature vector.

    Computes the mean and standard deviation of each spectral index, with
    optional q25/q75 percentiles.

    Args:
        indices_array: Array of shape (H, W, 5) (NaN is allowed)
        extended: Whether to output extended features
            - False: returns a length-10 vector (5 means + 5 std devs)
            - True: returns a length-20 vector (5 means + 5 std devs + 5 q25 + 5 q75)

    Returns:
        Tuple[np.ndarray, int]:
            - feature_vector: A 1D feature vector
            - n_valid_pixels: Number of valid (non-NaN) pixels

    Note:
        - Uses nanmean/nanstd to ignore NaN values
        - Returns an all-zero vector and issues a warning when the number
          of valid pixels is below 10
    """
    n_indices = indices_array.shape[2]

    # Compute the number of valid pixels (based on the first index channel)
    valid_mask = np.isfinite(indices_array[:, :, 0])
    n_valid = int(np.sum(valid_mask))

    # Insufficient valid pixels
    if n_valid < 10:
        dim = 20 if extended else 10
        warnings.warn(
            f"Number of valid pixels ({n_valid}) is below 10; returning an all-zero feature vector.",
            UserWarning,
            stacklevel=2,
        )
        return np.zeros(dim, dtype=np.float64), n_valid

    # Reshape to (N_pixels, 5) for easier aggregation
    flat = indices_array.reshape(-1, n_indices)

    # Compute the statistics
    means = np.nanmean(flat, axis=0)    # (5,)
    stds = np.nanstd(flat, axis=0)      # (5,)

    if extended:
        # Compute both percentiles in a single batch call to avoid an extra pass
        q25, q75 = np.nanquantile(flat, [0.25, 0.75], axis=0)  # each (5,)
        feature_vector = np.concatenate([means, stds, q25, q75])
    else:
        feature_vector = np.concatenate([means, stds])

    # Ensure there are no NaNs (should not occur in theory; defensive handling)
    feature_vector = np.nan_to_num(feature_vector, nan=0.0)

    return feature_vector, n_valid


# =============================================================================
# Mosaic-based feature extraction
# =============================================================================

def extract_spectral_features_from_mosaic(
    mosaic_path: str,
    center_x: float,
    center_y: float,
    ground_size_m: float,
    extended: bool = False
) -> Tuple[np.ndarray, int]:
    """
    Extract the spectral index features for a single point from a mosaic.

    Combines the full pipeline: extract_patch_from_mosaic ->
    compute_spectral_indices -> aggregate_spectral_patch.

    Args:
        mosaic_path: GeoTIFF file path
        center_x: Center X coordinate (must match the GeoTIFF CRS)
        center_y: Center Y coordinate
        ground_size_m: Ground-coverage side length of the patch (meters)
        extended: Whether to use extended aggregation (20D vs 10D)

    Returns:
        Tuple[np.ndarray, int]:
            - feature_vector: Spectral feature vector (10D or 20D)
            - n_valid_pixels: Number of valid pixels
    """
    # Lazy import to avoid a circular dependency
    from .ImageFetcher import extract_patch_from_mosaic

    # 1. Crop the patch from the mosaic
    patch = extract_patch_from_mosaic(mosaic_path, center_x, center_y, ground_size_m)

    # 2. Compute the spectral indices
    indices = compute_spectral_indices(patch)

    # 3. Spatial aggregation
    feature_vector, n_valid = aggregate_spectral_patch(indices, extended=extended)

    return feature_vector, n_valid


def _extract_spectral_from_open_src(
    src, cx: float, cy: float, gs: float, extended: bool,
    transform, pixel_size_x: float, n_bands: int,
) -> Tuple[np.ndarray, int]:
    """Extract spectral features for a single point from an already-open dataset (internal function)."""
    from .ImageFetcher import extract_patch_from_open_dataset

    patch = extract_patch_from_open_dataset(
        src, cx, cy, gs,
        transform=transform, pixel_size_x=pixel_size_x, n_bands=n_bands,
    )
    indices = compute_spectral_indices(patch)
    return aggregate_spectral_patch(indices, extended=extended)


def _extract_patches_bulk(
    src, xs: np.ndarray, ys: np.ndarray, ground_size_m: float,
    transform, pixel_size_x: float, n_bands: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Batch-extract patches from an already-open dataset: a single I/O read of
    the bounding box, followed by in-memory slicing.

    Only applicable to scenarios where all points use the same ground_size_m.

    Args:
        src: An already-open rasterio DatasetReader
        xs, ys: Coordinate arrays (N,)
        ground_size_m: The uniform patch size
        transform, pixel_size_x, n_bands: Pre-cached parameters

    Returns:
        (region_data, point_row_starts, point_col_starts):
            - region_data: (C, H_region, W_region) data for the entire bounding box
            - point_row_starts: (N,) row offset of each point relative to the region
            - point_col_starts: (N,) column offset of each point relative to the region
    """
    from rasterio.windows import Window

    t = transform
    half = ground_size_m / 2.0

    all_minx = xs.min() - half
    all_maxy = ys.max() + half
    all_maxx = xs.max() + half
    all_miny = ys.min() - half

    region_col_off = (all_minx - t.c) / t.a
    region_row_off = (all_maxy - t.f) / t.e
    region_col_size = (all_maxx - all_minx) / abs(t.a)
    region_row_size = (all_maxy - all_miny) / abs(t.e)

    region_window = Window(region_col_off, region_row_off, region_col_size, region_row_size)

    region_data = src.read(
        window=region_window,
        boundless=True,
        fill_value=np.nan,
    )

    point_col_starts = ((xs - half - all_minx) / abs(t.a)).astype(np.int64)
    point_row_starts = ((all_maxy - (ys + half)) / abs(t.e)).astype(np.int64)

    return region_data, point_row_starts, point_col_starts


def _slice_patch_from_region(
    region_data: np.ndarray, row_start: int, col_start: int, patch_pixels: int,
) -> Optional[np.ndarray]:
    """
    Slice a patch out of region_data. Returns None if out of bounds.

    Returns: (H, W, C) float64 or None
    """
    _, region_h, region_w = region_data.shape
    r0, c0 = row_start, col_start
    r1, c1 = r0 + patch_pixels, c0 + patch_pixels

    if r0 < 0 or c0 < 0 or r1 > region_h or c1 > region_w:
        return None

    return np.transpose(region_data[:, r0:r1, c0:c1], (1, 2, 0)).astype(np.float64)


def _process_chunk(args) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process one chunk of points (used for multiprocessing).

    Each worker independently opens the mosaic once.
    If ground_size is uniform, uses bulk read + in-memory slicing.
    """
    import rasterio
    from rasterio.windows import Window

    mosaic_path, xs, ys, ground_sizes, extended, chunk_indices = args
    feature_dim = 20 if extended else 10
    n = len(xs)

    features = np.zeros((n, feature_dim), dtype=np.float64)
    valid_counts = np.zeros(n, dtype=np.int32)

    uniform_gs = ground_sizes[0] if np.all(ground_sizes == ground_sizes[0]) else None

    with rasterio.open(mosaic_path) as src:
        transform = src.transform
        pixel_size_x = src.res[0]
        n_bands = src.count

        if uniform_gs is not None:
            # Bulk-read mode
            patch_pixels = int(round(uniform_gs / pixel_size_x))
            region_data, row_starts, col_starts = _extract_patches_bulk(
                src, xs, ys, uniform_gs, transform, pixel_size_x, n_bands,
            )
            _, region_h, region_w = region_data.shape

            for i in range(n):
                try:
                    patch = _slice_patch_from_region(
                        region_data, int(row_starts[i]), int(col_starts[i]), patch_pixels,
                    )
                    if patch is None:
                        # Boundary fallback
                        from .ImageFetcher import extract_patch_from_open_dataset
                        patch = extract_patch_from_open_dataset(
                            src, xs[i], ys[i], uniform_gs,
                            transform=transform, pixel_size_x=pixel_size_x, n_bands=n_bands,
                        )
                    indices = compute_spectral_indices(patch)
                    feat, n_valid = aggregate_spectral_patch(indices, extended=extended)
                    features[i] = feat
                    valid_counts[i] = n_valid
                except Exception:
                    features[i] = 0.0
                    valid_counts[i] = 0
        else:
            # Variable-size mode: per-point window read
            from .ImageFetcher import extract_patch_from_open_dataset
            for i in range(n):
                try:
                    patch = extract_patch_from_open_dataset(
                        src, xs[i], ys[i], ground_sizes[i],
                        transform=transform, pixel_size_x=pixel_size_x, n_bands=n_bands,
                    )
                    indices = compute_spectral_indices(patch)
                    feat, n_valid = aggregate_spectral_patch(indices, extended=extended)
                    features[i] = feat
                    valid_counts[i] = n_valid
                except Exception:
                    features[i] = 0.0
                    valid_counts[i] = 0

    return features, valid_counts


def batch_extract_from_mosaic(
    mosaic_path: str,
    coords_df: pd.DataFrame,
    ground_size_m_col_or_value: Union[float, int, str],
    output_path: str,
    extended: bool = False,
    n_workers: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Batch-extract the spectral features of all points from a mosaic.

    Optimization strategy:
    - When the patch size is uniform: a single I/O read of the bounding
      box, followed by numpy slicing in memory
    - When patch sizes vary: a single open + per-point window read
    - Supports multiprocessing (n_workers > 1)

    Args:
        mosaic_path: GeoTIFF file path
        coords_df: DataFrame containing 'x'/'y' columns (coordinates must match the GeoTIFF CRS)
        ground_size_m_col_or_value: Patch size parameter:
            - Scalar (float/int): uses the same patch size for all points
            - String: a DataFrame column name; each row uses a different patch size
        output_path: Output .npy file path (without extension)
        extended: Whether to use extended aggregation (20D vs 10D)
        n_workers: Number of parallel workers (default 1 = single process, for backward compatibility)

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - features: Feature matrix of shape (N, 10) or (N, 20)
            - valid_counts: (N,) number of valid pixels for each point

    Side effects:
        Saves {output_path}.npy (feature matrix) and
        {output_path}_valid_pixels.npy (valid pixel counts)
    """
    n_points = len(coords_df)
    feature_dim = 20 if extended else 10

    features = np.zeros((n_points, feature_dim), dtype=np.float64)
    valid_counts = np.zeros(n_points, dtype=np.int32)

    # Pre-extract the coordinate arrays (avoids row-by-row iloc)
    xs = coords_df['x'].values.astype(np.float64)
    ys = coords_df['y'].values.astype(np.float64)

    # Determine whether ground_size_m is a scalar or a column name
    is_uniform = not isinstance(ground_size_m_col_or_value, str)
    if isinstance(ground_size_m_col_or_value, str):
        ground_sizes = coords_df[ground_size_m_col_or_value].values.astype(np.float64)
    else:
        ground_sizes = np.full(n_points, float(ground_size_m_col_or_value))

    if n_workers > 1:
        # Multiprocessing mode: split into chunks and process in parallel
        from concurrent.futures import ProcessPoolExecutor
        chunk_size = max(1, n_points // (n_workers * 4))
        chunk_size = min(chunk_size, 5000)

        chunks = []
        for start in range(0, n_points, chunk_size):
            end = min(start + chunk_size, n_points)
            chunks.append((
                mosaic_path,
                xs[start:end],
                ys[start:end],
                ground_sizes[start:end],
                extended,
                (start, end),
            ))

        print(f"  Parallel mode: {n_workers} workers, {len(chunks)} chunks, chunk_size~={chunk_size}")

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(_process_chunk, chunks))

        offset = 0
        for chunk_feats, chunk_counts in results:
            n = len(chunk_feats)
            features[offset:offset + n] = chunk_feats
            valid_counts[offset:offset + n] = chunk_counts
            offset += n

        print(f"  Progress: {n_points}/{n_points}")

    else:
        # Single-process mode
        import rasterio

        with rasterio.open(mosaic_path) as src:
            transform = src.transform
            pixel_size_x = src.res[0]
            n_bands = src.count

            if is_uniform:
                # Bulk-read mode: a single I/O read
                uniform_gs = float(ground_sizes[0])
                patch_pixels = int(round(uniform_gs / pixel_size_x))
                region_data, row_starts, col_starts = _extract_patches_bulk(
                    src, xs, ys, uniform_gs, transform, pixel_size_x, n_bands,
                )

                for i in range(n_points):
                    try:
                        patch = _slice_patch_from_region(
                            region_data, int(row_starts[i]), int(col_starts[i]), patch_pixels,
                        )
                        if patch is None:
                            # Boundary fallback
                            from .ImageFetcher import extract_patch_from_open_dataset
                            patch = extract_patch_from_open_dataset(
                                src, xs[i], ys[i], uniform_gs,
                                transform=transform, pixel_size_x=pixel_size_x, n_bands=n_bands,
                            )
                        indices = compute_spectral_indices(patch)
                        feat, n_valid = aggregate_spectral_patch(indices, extended=extended)
                        features[i] = feat
                        valid_counts[i] = n_valid
                    except Exception as e:
                        warnings.warn(
                            f"Extraction failed for point {i} ({xs[i]:.1f}, {ys[i]:.1f}): {e}. Using an all-zero vector.",
                            UserWarning,
                            stacklevel=2,
                        )
                        features[i] = 0.0
                        valid_counts[i] = 0

                    if (i + 1) % 1000 == 0 or (i + 1) == n_points:
                        print(f"  Progress: {i + 1}/{n_points}")
            else:
                # Variable-size mode: per-point window read
                from .ImageFetcher import extract_patch_from_open_dataset

                for i in range(n_points):
                    try:
                        patch = extract_patch_from_open_dataset(
                            src, xs[i], ys[i], ground_sizes[i],
                            transform=transform, pixel_size_x=pixel_size_x, n_bands=n_bands,
                        )
                        indices = compute_spectral_indices(patch)
                        feat, n_valid = aggregate_spectral_patch(indices, extended=extended)
                        features[i] = feat
                        valid_counts[i] = n_valid
                    except Exception as e:
                        warnings.warn(
                            f"Extraction failed for point {i} ({xs[i]:.1f}, {ys[i]:.1f}): {e}. Using an all-zero vector.",
                            UserWarning,
                            stacklevel=2,
                        )
                        features[i] = 0.0
                        valid_counts[i] = 0

                    if (i + 1) % 1000 == 0 or (i + 1) == n_points:
                        print(f"  Progress: {i + 1}/{n_points}")

    # Save the results
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    features_path = str(output_file) if output_file.suffix == '.npy' else str(output_file) + '.npy'
    np.save(features_path, features)

    valid_path = str(output_file).replace('.npy', '') + '_valid_pixels.npy'
    np.save(valid_path, valid_counts)

    print(f"Features saved: {features_path} (shape: {features.shape})")
    print(f"Valid pixel counts saved: {valid_path}")

    return features, valid_counts
