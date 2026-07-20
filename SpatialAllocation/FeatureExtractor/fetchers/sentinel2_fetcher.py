"""
Sentinel2Fetcher - Sentinel-2 multispectral image GeoTIFF download

Logic adapted from ImageFetcher.download_region_mosaic(),
including ImageCollection filtering -> composite -> /10000 + clamp -> tiled download -> merge.
"""
import hashlib
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np

from .registry import fetcher_registry
from .base import BaseFetcher

logger = logging.getLogger(__name__)

# Lazy import of the GEE module
ee = None
_gee_initialized = False


def _initialize_gee() -> None:
    """Initialize the GEE connection (copied from ImageFetcher._initialize_gee)"""
    global ee, _gee_initialized
    if _gee_initialized:
        return

    import ee as earth_engine
    ee = earth_engine

    from dotenv import load_dotenv
    load_dotenv()

    key_path = os.getenv("GEE_SERVICE_ACCOUNT_KEY_PATH")
    if not key_path:
        raise ValueError("Environment variable GEE_SERVICE_ACCOUNT_KEY_PATH is not set")

    key_file = Path(key_path)
    if not key_file.is_absolute():
        project_root = Path(__file__).parent.parent.parent.parent
        key_file = project_root / key_path

    if not key_file.exists():
        raise ValueError(f"GEE key file does not exist: {key_file}")

    import json
    with open(key_file, "r") as f:
        key_data = json.load(f)

    email = key_data.get("client_email")
    if not email:
        raise ValueError("Key file is missing the client_email field")

    credentials = ee.ServiceAccountCredentials(email, str(key_file))
    ee.Initialize(credentials)
    _gee_initialized = True


def _estimate_region_size(
    region_bounds, n_bands: int, scale_m: int
) -> Tuple[int, Tuple[float, float, float, float]]:
    """Estimate the download size for a region, returning (byte count, EPSG:27700 bbox)"""
    import pyproj

    if isinstance(region_bounds, tuple) and len(region_bounds) == 4:
        minx, miny, maxx, maxy = region_bounds
    else:
        minx, miny, maxx, maxy = region_bounds.bounds

    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
    x1, y1 = transformer.transform(minx, miny)
    x2, y2 = transformer.transform(maxx, maxy)

    proj_bbox = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
    width_px = int(math.ceil((proj_bbox[2] - proj_bbox[0]) / scale_m))
    height_px = int(math.ceil((proj_bbox[3] - proj_bbox[1]) / scale_m))
    # Overestimating is safer than underestimating: computed as Float64 (8 bytes)
    # to ensure the GEE limit is not exceeded
    estimated_bytes = width_px * height_px * n_bands * 8

    return estimated_bytes, proj_bbox


def _compute_tile_grid(
    projected_bbox: Tuple[float, float, float, float],
    n_bands: int,
    scale_m: int,
    target_bytes: int = 24_000_000,
) -> List[Tuple[float, float, float, float]]:
    """Split the projected region into a tile grid"""
    # Consistent with _estimate_region_size: overestimate using Float64 (8 bytes)
    # to ensure tiles don't exceed the GEE limit
    tile_px = int(math.floor(math.sqrt(target_bytes / (n_bands * 8))))
    tile_size_m = tile_px * scale_m

    bminx, bminy, bmaxx, bmaxy = projected_bbox
    n_cols = int(math.ceil((bmaxx - bminx) / tile_size_m))
    n_rows = int(math.ceil((bmaxy - bminy) / tile_size_m))

    tiles = []
    for row in range(n_rows):
        for col in range(n_cols):
            tx_min = bminx + col * tile_size_m
            ty_min = bminy + row * tile_size_m
            tx_max = min(tx_min + tile_size_m, bmaxx)
            ty_max = min(ty_min + tile_size_m, bmaxy)
            tiles.append((tx_min, ty_min, tx_max, ty_max))

    return tiles


def _download_and_merge_tiles(
    composite, tile_bboxes, bands, scale_m, crs, output_path
) -> str:
    """Download the GEE image tile by tile and merge into a single GeoTIFF"""
    import requests
    import rasterio
    from rasterio.merge import merge
    import pyproj
    from shapely.geometry import box, mapping
    import shutil

    output_file = Path(output_path)
    temp_dir = output_file.parent / f"_tiles_tmp_{output_file.stem}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    transformer_to_4326 = pyproj.Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)

    tile_paths = []
    max_retries = 3

    for i, tile_bbox in enumerate(tile_bboxes):
        tile_path = temp_dir / f"tile_{i:04d}.tif"

        if tile_path.exists() and tile_path.stat().st_size > 1024:
            tile_paths.append(tile_path)
            continue

        tx_min, ty_min, tx_max, ty_max = tile_bbox
        lon1, lat1 = transformer_to_4326.transform(tx_min, ty_min)
        lon2, lat2 = transformer_to_4326.transform(tx_max, ty_max)
        tile_geojson = mapping(box(
            min(lon1, lon2), min(lat1, lat2),
            max(lon1, lon2), max(lat1, lat2),
        ))
        ee_region = ee.Geometry(tile_geojson)

        for attempt in range(max_retries):
            try:
                url = composite.getDownloadURL({
                    "region": ee_region,
                    "scale": scale_m,
                    "crs": crs,
                    "format": "GEO_TIFF",
                    "bands": bands,
                })

                response = requests.get(url, timeout=300, stream=True)
                response.raise_for_status()

                with open(tile_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                logger.info(f"  Tile {i+1}/{len(tile_bboxes)}: OK ({tile_path.stat().st_size / 1024 / 1024:.1f} MB)")
                tile_paths.append(tile_path)
                break

            except Exception as e:
                if attempt < max_retries - 1:
                    wait_sec = 5 * (3 ** attempt)
                    logger.warning(f"  Tile {i+1} retry {attempt+1}/{max_retries} (waiting {wait_sec}s)... {e}")
                    time.sleep(wait_sec)
                else:
                    raise RuntimeError(f"Tile {i+1} download failed (after {max_retries} retries): {e}") from e

    # Merge tiles
    old_stderr_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, 2)
    try:
        if len(tile_paths) == 1:
            shutil.move(str(tile_paths[0]), str(output_file))
        else:
            datasets = [rasterio.open(p) for p in tile_paths]
            try:
                merged, merged_transform = merge(datasets, method="first")
                profile = datasets[0].profile.copy()
                profile.update({
                    "height": merged.shape[1],
                    "width": merged.shape[2],
                    "transform": merged_transform,
                    "photometric": "MINISBLACK",
                })
                with rasterio.open(str(output_file), "w", **profile) as dst:
                    dst.write(merged)
            finally:
                for ds in datasets:
                    ds.close()
    finally:
        os.dup2(old_stderr_fd, 2)
        os.close(devnull_fd)
        os.close(old_stderr_fd)

    if temp_dir.exists():
        shutil.rmtree(str(temp_dir), ignore_errors=True)

    return str(output_file)


@fetcher_registry.register("sentinel2", description="Sentinel-2 multispectral image download")
class Sentinel2Fetcher(BaseFetcher):
    """
    Downloads a Sentinel-2 multi-band median composite GeoTIFF from GEE.

    Config fields:
        gee_collection: GEE ImageCollection ID
        bands: band list
        scale_m: output resolution in meters
        composite_method: composite method ("median" / "mean")
        max_cloud_cover: maximum cloud cover percentage
        date_range: [start_date, end_date]
        output_crs: output CRS
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.gee_collection = config.get("gee_collection", "COPERNICUS/S2_SR_HARMONIZED")
        self.bands = config.get("bands", ["B2", "B3", "B4", "B8", "B11", "B12"])
        self.scale_m = config.get("scale_m", 10)
        self.composite_method = config.get("composite_method", "median")
        self.max_cloud_cover = config.get("max_cloud_cover", 20)
        self.date_range = config.get("date_range", ["2022-01-01", "2023-12-31"])
        self.output_crs = config.get("output_crs", "EPSG:27700")

    def fetch(self, region_bounds: Any, cache_dir: str) -> str:
        """
        Download the Sentinel-2 composite GeoTIFF and return the file path.
        Adapted from ImageFetcher.download_region_mosaic().
        """
        cache_key = self.get_cache_key(region_bounds)
        s2_cache_dir = os.path.join(cache_dir, "sentinel2")
        os.makedirs(s2_cache_dir, exist_ok=True)
        cache_path = os.path.join(s2_cache_dir, f"{cache_key}_mosaic.tif")

        if os.path.exists(cache_path) and os.path.getsize(cache_path) > 1024:
            logger.info(f"Sentinel-2 cache hit: {cache_path}")
            return cache_path

        # Initialize GEE
        _initialize_gee()

        # Build the GEE geometry
        from shapely.geometry import box, mapping
        if isinstance(region_bounds, tuple) and len(region_bounds) == 4:
            geometry_geojson = mapping(box(*region_bounds))
        else:
            geometry_geojson = mapping(region_bounds)

        ee_geometry = ee.Geometry(geometry_geojson)

        # Build the ImageCollection -> composite
        start_date, end_date = self.date_range
        collection = (
            ee.ImageCollection(self.gee_collection)
            .filterBounds(ee_geometry)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", self.max_cloud_cover))
            .select(self.bands)
        )

        if self.composite_method == "median":
            composite = collection.median()
        elif self.composite_method == "mean":
            composite = collection.mean()
        else:
            raise ValueError(f"Unsupported composite method: {self.composite_method}")

        # Normalize to 0-1
        composite = composite.divide(10000).clamp(0, 1)

        # Estimate the size to decide whether tiling is needed
        estimated_bytes, proj_bbox = _estimate_region_size(
            region_bounds, len(self.bands), self.scale_m
        )

        # Corner-based bbox transformation underestimates the actual pixel count
        # by roughly 2x, so the threshold is set to 20 MB (corresponding to ~40 MB actual)
        single_threshold = 20_000_000

        if estimated_bytes <= single_threshold:
            logger.info(f"Sentinel-2 region is small (~{estimated_bytes / 1e6:.0f} MB), downloading directly...")
            import requests

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    url = composite.getDownloadURL({
                        "region": ee_geometry,
                        "scale": self.scale_m,
                        "crs": self.output_crs,
                        "format": "GEO_TIFF",
                        "bands": self.bands,
                    })
                    response = requests.get(url, timeout=300, stream=True)
                    response.raise_for_status()
                    with open(cache_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_sec = 5 * (3 ** attempt)
                        logger.warning(f"Download failed, retrying {attempt+1}/{max_retries}... {e}")
                        time.sleep(wait_sec)
                    else:
                        raise RuntimeError(f"Sentinel-2 download failed (after {max_retries} retries): {e}") from e
        else:
            tile_bboxes = _compute_tile_grid(proj_bbox, len(self.bands), self.scale_m)
            logger.info(f"Sentinel-2 region is large (~{estimated_bytes / 1e6:.0f} MB), downloading in {len(tile_bboxes)} tiles...")
            _download_and_merge_tiles(
                composite, tile_bboxes, self.bands, self.scale_m, self.output_crs, cache_path
            )

        size_mb = os.path.getsize(cache_path) / 1024 / 1024
        logger.info(f"Sentinel-2 download complete: {cache_path} ({size_mb:.1f} MB)")
        return cache_path

    def get_cache_key(self, region_bounds: Any) -> str:
        if hasattr(region_bounds, "bounds"):
            bounds_tuple = region_bounds.bounds
        else:
            bounds_tuple = tuple(region_bounds)
        key_str = (
            f"{bounds_tuple}_{self.gee_collection}_{self.bands}_"
            f"{self.composite_method}_{self.max_cloud_cover}_{self.date_range}"
        )
        return hashlib.md5(key_str.encode()).hexdigest()[:12]
