"""
NtlFetcher - VIIRS nighttime lights GeoTIFF download

Downloads a VIIRS DNB monthly composite image from GEE, single band avg_rad.
Native resolution is ~500m; the entire UK is only ~19MB, so most scenes
don't need tiling.
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
    """Initialize the GEE connection (copied from sentinel2_fetcher._initialize_gee)"""
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
    region_bounds, n_bands: int, scale_m: int, bytes_per_pixel: int = 8
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
    estimated_bytes = width_px * height_px * n_bands * bytes_per_pixel

    return estimated_bytes, proj_bbox


def _compute_tile_grid(
    projected_bbox: Tuple[float, float, float, float],
    n_bands: int,
    scale_m: int,
    target_bytes: int = 24_000_000,
) -> List[Tuple[float, float, float, float]]:
    """Split the projected region into a tile grid"""
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


@fetcher_registry.register("ntl", description="VIIRS nighttime lights image download")
class NtlFetcher(BaseFetcher):
    """
    Downloads a VIIRS DNB monthly composite nighttime lights GeoTIFF from GEE.

    Config fields:
        gee_collection: GEE ImageCollection ID
        band: band name (default avg_rad)
        scale_m: output resolution in meters (default 500)
        composite_method: composite method ("median" / "mean")
        date_range: [start_date, end_date]
        output_crs: output CRS
        buffer_m: region boundary buffer in meters (default 1000, roughly 2 pixel widths)
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.gee_collection = config.get(
            "gee_collection", "NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG"
        )
        self.band = config.get("band", "avg_rad")
        self.scale_m = config.get("scale_m", 500)
        self.composite_method = config.get("composite_method", "median")
        self.date_range = config.get("date_range", ["2022-01-01", "2023-12-31"])
        self.output_crs = config.get("output_crs", "EPSG:27700")
        self.buffer_m = config.get("buffer_m", 1000)

    def fetch(self, region_bounds: Any, cache_dir: str) -> str:
        """
        Download the VIIRS NTL composite GeoTIFF and return the file path.

        Flow:
        1. Cache check
        2. Build ee_geometry with buffer expansion
        3. Filter ImageCollection + median composite (no cloud filtering)
        4. Clip at zero with .max(0) (no /10000 normalization)
        5. Estimate size -> most likely a single download
        """
        cache_key = self.get_cache_key(region_bounds)
        ntl_cache_dir = os.path.join(cache_dir, "ntl")
        os.makedirs(ntl_cache_dir, exist_ok=True)
        cache_path = os.path.join(ntl_cache_dir, f"{cache_key}_ntl.tif")

        if os.path.exists(cache_path) and os.path.getsize(cache_path) > 1024:
            logger.info(f"NTL cache hit: {cache_path}")
            return cache_path

        # Initialize GEE
        _initialize_gee()

        # Build the GEE geometry, adding a buffer to ensure edge grid points have data
        from shapely.geometry import box, mapping
        if isinstance(region_bounds, tuple) and len(region_bounds) == 4:
            geom = box(*region_bounds)
        else:
            geom = region_bounds

        # The buffer must be applied in a projected coordinate system (EPSG:4326 buffer units are degrees)
        import geopandas as gpd
        gdf_tmp = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")
        gdf_proj = gdf_tmp.to_crs(self.output_crs)
        buffered_geom = gdf_proj.geometry[0].buffer(self.buffer_m)
        gdf_buffered = gpd.GeoDataFrame(geometry=[buffered_geom], crs=self.output_crs)
        gdf_4326 = gdf_buffered.to_crs("EPSG:4326")
        geometry_geojson = mapping(gdf_4326.geometry[0])

        ee_geometry = ee.Geometry(geometry_geojson)

        # Build the ImageCollection -> composite (no cloud filtering; monthly composites are already de-clouded)
        start_date, end_date = self.date_range
        collection = (
            ee.ImageCollection(self.gee_collection)
            .filterBounds(ee_geometry)
            .filterDate(start_date, end_date)
            .select([self.band])
        )

        if self.composite_method == "median":
            composite = collection.median()
        elif self.composite_method == "mean":
            composite = collection.mean()
        else:
            raise ValueError(f"Unsupported composite method: {self.composite_method}")

        # Clip at zero: remove negative instrument noise, no normalization applied
        composite = composite.max(ee.Image(0))

        # Estimate the size to decide whether tiling is needed
        # (single band + 500m resolution, usually well below the threshold)
        estimated_bytes, proj_bbox = _estimate_region_size(
            region_bounds, 1, self.scale_m, bytes_per_pixel=8
        )

        single_threshold = 20_000_000
        bands = [self.band]

        if estimated_bytes <= single_threshold:
            logger.info(f"NTL region is small (~{estimated_bytes / 1e6:.1f} MB), downloading directly...")
            import requests

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    url = composite.getDownloadURL({
                        "region": ee_geometry,
                        "scale": self.scale_m,
                        "crs": self.output_crs,
                        "format": "GEO_TIFF",
                        "bands": bands,
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
                        logger.warning(f"NTL download failed, retrying {attempt+1}/{max_retries}... {e}")
                        time.sleep(wait_sec)
                    else:
                        raise RuntimeError(f"NTL download failed (after {max_retries} retries): {e}") from e
        else:
            tile_bboxes = _compute_tile_grid(proj_bbox, 1, self.scale_m)
            logger.info(f"NTL region is large (~{estimated_bytes / 1e6:.1f} MB), downloading in {len(tile_bboxes)} tiles...")
            _download_and_merge_tiles(
                composite, tile_bboxes, bands, self.scale_m, self.output_crs, cache_path
            )

        size_mb = os.path.getsize(cache_path) / 1024 / 1024
        logger.info(f"NTL download complete: {cache_path} ({size_mb:.1f} MB)")
        return cache_path

    def get_cache_key(self, region_bounds: Any) -> str:
        if hasattr(region_bounds, "bounds"):
            bounds_tuple = region_bounds.bounds
        else:
            bounds_tuple = tuple(region_bounds)
        key_str = (
            f"{bounds_tuple}_{self.gee_collection}_{self.band}_"
            f"{self.composite_method}_{self.date_range}_{self.buffer_m}"
        )
        return hashlib.md5(key_str.encode()).hexdigest()[:12]
