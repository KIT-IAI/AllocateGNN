# -*- coding: utf-8 -*-
"""
Remote sensing imagery acquisition module

This module provides functionality for fetching Sentinel-2 satellite imagery
from Google Earth Engine (GEE), used for visual feature extraction in the
spatial load allocation model.

Main features:
- Single-image download
- Batch image download
- Image index file generation
"""

from typing import Optional, Tuple, Dict, Any, Union, List
import math
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from dotenv import load_dotenv

# The GEE module is imported lazily to avoid requiring credentials at module load time
ee = None

# Load environment variables
load_dotenv()


# =============================================================================
# Private helper functions
# =============================================================================

def _latlon_to_tile_coords(lat: float, lon: float, zoom: int) -> Tuple[int, int]:
    """
    Convert WGS84 latitude/longitude to Web Mercator tile coordinates.

    Uses the standard Web Mercator projection formula to convert geographic
    coordinates into tile X/Y coordinates at the given zoom level.

    Args:
        lat: Latitude (between -85.05 and 85.05)
        lon: Longitude (between -180 and 180)
        zoom: Zoom level (0-20)

    Returns:
        Tuple[int, int]: (x, y) tile coordinates

    Note:
        The Web Mercator projection produces infinite values in polar
        regions (|lat| > 85.05 deg), so this function clamps the latitude
        to a valid range.
    """
    # Clamp the latitude to the valid Web Mercator range
    lat = max(-85.05112878, min(85.05112878, lat))

    # Compute the number of tiles
    n = 2.0 ** zoom

    # Convert longitude to the X coordinate
    x = int((lon + 180.0) / 360.0 * n)

    # Convert latitude to the Y coordinate (requires the Mercator projection formula)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)

    return (x, y)


def _latlon_to_bbox(
    center_lat: float,
    center_lon: float,
    size_meters: float
) -> Tuple[float, float, float, float]:
    """
    Compute a bounding box from a center point and side length.

    Computes a square bounding box of the given side length (meters)
    centered on the given latitude/longitude. Uses a simplified spherical
    approximation formula, suitable for small-area calculations at
    low-to-mid latitudes.

    Args:
        center_lat: Center point latitude (WGS84 CRS)
        center_lon: Center point longitude (WGS84 CRS)
        size_meters: Bounding box side length (meters), i.e. the side of the square

    Returns:
        Tuple[float, float, float, float]: (west, south, east, north) bounding box
            - west: western boundary longitude
            - south: southern boundary latitude
            - east: eastern boundary longitude
            - north: northern boundary latitude

    Note:
        - For high-latitude regions (e.g. northern UK), the distance error
          in the longitude direction increases
        - In GEE, an explicit projection transform is recommended for
          higher precision
        - The Earth's radius is taken as 6378137 meters (WGS84 ellipsoid
          semi-major axis)
    """
    # WGS84 ellipsoid semi-major axis (meters)
    EARTH_RADIUS = 6378137.0

    # Half side length
    half_size = size_meters / 2.0

    # Latitude direction: 1 degree is approximately EARTH_RADIUS * pi / 180 meters
    # lat_delta = half_size / (EARTH_RADIUS * pi / 180)
    lat_delta = (half_size / EARTH_RADIUS) * (180.0 / math.pi)

    # Longitude direction: the distance of 1 degree varies with latitude.
    # At latitude lat, 1 degree of longitude ~= cos(lat) * EARTH_RADIUS * pi / 180 meters
    # lon_delta = half_size / (cos(lat) * EARTH_RADIUS * pi / 180)
    lat_rad = math.radians(center_lat)
    lon_delta = (half_size / (EARTH_RADIUS * math.cos(lat_rad))) * (180.0 / math.pi)

    # Compute the bounding box
    west = center_lon - lon_delta
    east = center_lon + lon_delta
    south = center_lat - lat_delta
    north = center_lat + lat_delta

    return (west, south, east, north)


# GEE initialization status flag
_gee_initialized = False


def _initialize_gee() -> None:
    """
    Initialize the Google Earth Engine connection.

    Reads the service account JSON key file path from an environment
    variable and uses that credential to initialize GEE. This function is
    idempotent — repeated calls only perform initialization on the first call.

    Environment variables:
        GEE_SERVICE_ACCOUNT_KEY_PATH: Path to the JSON key file (relative to the project root)

    Raises:
        ValueError: When the environment variable is not set or the key file does not exist
        RuntimeError: When GEE initialization fails
    """
    global ee, _gee_initialized

    if _gee_initialized:
        return

    # Lazily import the ee module
    import ee as earth_engine
    ee = earth_engine

    # Read the environment variable
    key_path = os.getenv('GEE_SERVICE_ACCOUNT_KEY_PATH')

    if not key_path:
        raise ValueError(
            "Environment variable GEE_SERVICE_ACCOUNT_KEY_PATH is not set.\n"
            "Create a .env file in the project root, following .env.example:\n"
            "  GEE_SERVICE_ACCOUNT_KEY_PATH=secrets/gee-service-account.json\n"
            "and place the Google Cloud service account key file in the secrets/ directory."
        )

    # Check whether the key file exists
    key_file = Path(key_path)
    if not key_file.is_absolute():
        # If it is a relative path, resolve it relative to the project root
        project_root = Path(__file__).parent.parent.parent
        key_file = project_root / key_path

    if not key_file.exists():
        raise ValueError(
            f"GEE key file not found: {key_file}\n"
            "Make sure the Google Cloud service account key file has been placed at the specified location.\n"
            "Steps to obtain the key file:\n"
            "  1. Visit the Google Cloud Console\n"
            "  2. Create or select a service account\n"
            "  3. Register that service account in Earth Engine\n"
            "  4. Download the key file in JSON format"
        )

    try:
        # Read the JSON file to get the service account email
        import json
        with open(key_file, 'r') as f:
            key_data = json.load(f)

        service_account_email = key_data.get('client_email')
        if not service_account_email:
            raise ValueError(
                f"Malformed key file: missing the client_email field\n"
                f"File path: {key_file}"
            )

        # Initialize using the service account credentials
        credentials = ee.ServiceAccountCredentials(
            service_account_email,
            str(key_file)
        )
        ee.Initialize(credentials)
        _gee_initialized = True

    except ee.EEException as e:
        raise RuntimeError(
            f"GEE initialization failed: {str(e)}\n"
            "Please check:\n"
            "  1. Whether the service account is registered in Earth Engine\n"
            "  2. Whether the key file is valid\n"
            "  3. Whether the network connection is working"
        ) from e


def _get_default_date_range() -> Tuple[str, str]:
    """
    Get the default imagery time range.

    Reads GEE_DEFAULT_START_DATE and GEE_DEFAULT_END_DATE from environment
    variables. If not set, returns the default year 2023.

    Returns:
        Tuple[str, str]: (start_date, end_date) in 'YYYY-MM-DD' format
    """
    start_date = os.getenv('GEE_DEFAULT_START_DATE', '2023-01-01')
    end_date = os.getenv('GEE_DEFAULT_END_DATE', '2023-12-31')
    return (start_date, end_date)


def _get_max_cloud_cover() -> int:
    """
    Get the maximum cloud cover threshold.

    Reads GEE_MAX_CLOUD_COVER from an environment variable; returns the
    default value 20 if not set.

    Returns:
        int: Maximum cloud cover percentage (0-100)
    """
    cloud_cover_str = os.getenv('GEE_MAX_CLOUD_COVER', '20')
    try:
        return int(cloud_cover_str)
    except ValueError:
        return 20


def _get_image_size_meters() -> float:
    """
    Get the ground-coverage side length of the imagery.

    Reads GEE_IMAGE_SIZE_METERS from an environment variable; returns the
    default value 500 meters if not set.

    Returns:
        float: Ground-coverage side length (meters)
    """
    size_str = os.getenv('GEE_IMAGE_SIZE_METERS', '500')
    try:
        return float(size_str)
    except ValueError:
        return 500.0


def _get_image_dimension() -> int:
    """
    Get the pixel dimension of the output image.

    Reads GEE_IMAGE_DIMENSION from an environment variable; returns the
    default value 256 if not set.

    Returns:
        int: Pixel dimension (square side length)
    """
    dim_str = os.getenv('GEE_IMAGE_DIMENSION', '256')
    try:
        return int(dim_str)
    except ValueError:
        return 256


def _get_output_dir() -> str:
    """
    Get the satellite imagery save directory.

    Reads GEE_OUTPUT_DIR from an environment variable; returns the default
    value 'data/satellite_images' if not set. If it is a relative path, it
    is resolved relative to the project root.

    Returns:
        str: Absolute path of the output directory
    """
    output_dir = os.getenv('GEE_OUTPUT_DIR', 'data/satellite_images')

    output_path = Path(output_dir)
    if not output_path.is_absolute():
        # Relative to the project root
        project_root = Path(__file__).parent.parent.parent
        output_path = project_root / output_dir

    return str(output_path)


# =============================================================================
# Public interface functions
# =============================================================================

def fetch_satellite_image(
    center_lat: float,
    center_lon: float,
    grid_id: str,
    output_dir: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    force_download: bool = False,
    image_size_meters: Optional[float] = None,
    image_dimension: Optional[int] = None
) -> Dict[str, Any]:
    """
    Fetch the satellite image for a given coordinate point.

    Downloads a Sentinel-2 satellite image centered on the given
    latitude/longitude from GEE. The image is cropped to a fixed-size
    square region and saved in PNG format.

    Args:
        center_lat: Center point latitude (WGS84 CRS)
        center_lon: Center point longitude (WGS84 CRS)
        grid_id: Grid unique identifier, used to build the file name
        output_dir: Output directory path; if None, read from the
                   GEE_OUTPUT_DIR environment variable
        start_date: Imagery time range start date, format 'YYYY-MM-DD';
                   if None, the default is read from an environment variable
        end_date: Imagery time range end date, format 'YYYY-MM-DD';
                 if None, the default is read from an environment variable
        force_download: Whether to force re-download (ignore the cache), default False
        image_size_meters: Ground-coverage side length of the imagery (meters);
                          if None, read from the GEE_IMAGE_SIZE_METERS environment variable
        image_dimension: Pixel dimension of the output image;
                        if None, read from the GEE_IMAGE_DIMENSION environment variable

    Returns:
        Dict[str, Any]: Download result info, containing:
            - file_path: Full path of the saved image file
            - system_index: GEE image unique ID
            - acquisition_date: Image acquisition date
            - cloud_cover: Cloud cover percentage
            - cached: Whether the cache was used

    Raises:
        ValueError: When GEE credentials are not configured or the coordinates are invalid
        RuntimeError: When the GEE API call fails or no imagery is available
    """
    import requests
    import json

    # Read the default configuration from environment variables
    if output_dir is None:
        output_dir = _get_output_dir()
    if image_size_meters is None:
        image_size_meters = _get_image_size_meters()
    if image_dimension is None:
        image_dimension = _get_image_dimension()

    # Ensure the output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Build the output file path
    file_path = output_path / f"{grid_id}.png"
    metadata_path = output_path / f"{grid_id}.meta.json"

    # Minimum cache file size threshold (to avoid corrupted files)
    MIN_FILE_SIZE = 1024  # 1KB

    # Check the cache
    if not force_download and file_path.exists():
        if file_path.stat().st_size >= MIN_FILE_SIZE:
            # Try to read the metadata
            metadata = {}
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                except (json.JSONDecodeError, IOError):
                    pass

            return {
                'file_path': str(file_path),
                'system_index': metadata.get('system_index', ''),
                'acquisition_date': metadata.get('acquisition_date', ''),
                'cloud_cover': metadata.get('cloud_cover', -1),
                'cached': True
            }

    # Initialize GEE
    _initialize_gee()

    # Get the default time range
    if start_date is None or end_date is None:
        default_start, default_end = _get_default_date_range()
        start_date = start_date or default_start
        end_date = end_date or default_end

    # Get the cloud cover threshold
    max_cloud_cover = _get_max_cloud_cover()

    # Create the center point geometry object
    point = ee.Geometry.Point([center_lon, center_lat])

    # Compute the buffer region (using a metric buffer).
    # Note: buffer() is computed in spherical coordinates; use a projection
    # transform for high-latitude regions
    half_size = image_size_meters / 2.0
    region = point.buffer(half_size).bounds()

    # Get the Sentinel-2 image collection
    collection = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(point)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', max_cloud_cover))
        .sort('CLOUDY_PIXEL_PERCENTAGE')
    )

    # Get the collection size
    collection_size = collection.size().getInfo()

    if collection_size == 0:
        raise RuntimeError(
            f"No available imagery found under the specified conditions.\n"
            f"Coordinates: ({center_lat}, {center_lon})\n"
            f"Time range: {start_date} to {end_date}\n"
            f"Max cloud cover: {max_cloud_cover}%\n"
            "Suggestion: try widening the time range or raising the cloud cover threshold."
        )

    # Get the image with the lowest cloud cover
    best_image = collection.first()

    # Get the image metadata
    image_info = best_image.getInfo()
    properties = image_info.get('properties', {})

    system_index = properties.get('system:index', '')
    acquisition_date = properties.get('DATATAKE_IDENTIFIER', '')[:8] if properties.get('DATATAKE_IDENTIFIER') else ''
    cloud_cover = properties.get('CLOUDY_PIXEL_PERCENTAGE', -1)

    # Select the RGB bands and apply visualization parameters.
    # Sentinel-2 RGB: B4(red), B3(green), B2(blue)
    # Raw reflectance values range 0-10000 and require visualization scaling
    rgb_image = best_image.select(['B4', 'B3', 'B2'])

    # Visualization parameters
    vis_params = {
        'min': 0,
        'max': 3000,
        'bands': ['B4', 'B3', 'B2'],
        'region': region,
        'dimensions': image_dimension,
        'format': 'png'
    }

    # Get the thumbnail URL
    thumb_url = rgb_image.getThumbURL(vis_params)

    # Download the image
    try:
        response = requests.get(thumb_url, timeout=60)
        response.raise_for_status()

        # Save the image
        with open(file_path, 'wb') as f:
            f.write(response.content)

        # Save the metadata
        metadata = {
            'system_index': system_index,
            'acquisition_date': acquisition_date,
            'cloud_cover': cloud_cover,
            'center_lat': center_lat,
            'center_lon': center_lon,
            'start_date': start_date,
            'end_date': end_date,
            'image_size_meters': image_size_meters
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    except requests.RequestException as e:
        raise RuntimeError(
            f"Failed to download imagery: {str(e)}\n"
            f"URL: {thumb_url[:100]}..."
        ) from e

    return {
        'file_path': str(file_path),
        'system_index': system_index,
        'acquisition_date': acquisition_date,
        'cloud_cover': cloud_cover,
        'cached': False
    }


def batch_fetch_images(
    grid_gdf: gpd.GeoDataFrame,
    output_dir: Optional[str] = None,
    skip_existing: bool = True,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    max_retries: int = 3,
    delay_seconds: float = 0.5,
    log_file: Optional[str] = None
) -> pd.DataFrame:
    """
    Batch-fetch satellite imagery for multiple grid points.

    Iterates over all grid points in the input GeoDataFrame, downloading
    the corresponding satellite imagery for each in turn. Supports
    resumable downloads (skipping already-existing files) and a
    failure-retry mechanism.

    Args:
        grid_gdf: GeoDataFrame containing the grid center points;
                 must include 'grid_id' and 'geometry' (Point type) columns
        output_dir: Output directory path; if None, read from the
                   GEE_OUTPUT_DIR environment variable
        skip_existing: Whether to skip already-existing files, default True
        start_date: Imagery time range start date, format 'YYYY-MM-DD'
        end_date: Imagery time range end date, format 'YYYY-MM-DD'
        max_retries: Number of failure retries, default 3
        delay_seconds: Delay after each download (seconds), used to avoid
                       GEE rate limiting, default 0.5
        log_file: Failure log file path, defaults to output_dir/failed_downloads.log

    Returns:
        pd.DataFrame: Download result summary table, containing the following columns:
            - grid_id: Grid ID
            - status: Download status ('success' / 'failed' / 'skipped')
            - file_path: File save path (on success)
            - error_message: Error message (on failure)
            - system_index: GEE image unique ID (on success)
            - acquisition_date: Image acquisition date (on success)
            - cloud_cover: Cloud cover percentage (on success)
    """
    import time
    import logging
    from datetime import datetime

    # Validate the input GeoDataFrame
    if 'grid_id' not in grid_gdf.columns:
        raise ValueError("The input GeoDataFrame must contain a 'grid_id' column")
    if 'geometry' not in grid_gdf.columns:
        raise ValueError("The input GeoDataFrame must contain a 'geometry' column")

    # Read the default output directory from an environment variable
    if output_dir is None:
        output_dir = _get_output_dir()

    # Ensure the output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Configure logging
    if log_file is None:
        log_file = output_path / 'failed_downloads.log'
    else:
        log_file = Path(log_file)

    # Initialize the results list
    results = []
    total_count = len(grid_gdf)
    success_count = 0
    failed_count = 0
    skipped_count = 0

    # Progress reporting interval (report every 10% processed)
    progress_interval = max(1, total_count // 10)

    print(f"Starting batch download of {total_count} images...")
    print(f"Output directory: {output_dir}")

    for idx, row in grid_gdf.iterrows():
        grid_id = str(row['grid_id'])
        geometry = row['geometry']

        # Get the center point coordinates
        if hasattr(geometry, 'centroid'):
            center = geometry.centroid
        else:
            center = geometry

        center_lon = center.x
        center_lat = center.y

        # Check whether to skip an already-existing file
        file_path = output_path / f"{grid_id}.png"
        if skip_existing and file_path.exists() and file_path.stat().st_size >= 1024:
            results.append({
                'grid_id': grid_id,
                'status': 'skipped',
                'file_path': str(file_path),
                'error_message': '',
                'system_index': '',
                'acquisition_date': '',
                'cloud_cover': -1
            })
            skipped_count += 1

            # Progress report
            processed = len(results)
            if processed % progress_interval == 0:
                print(f"Progress: {processed}/{total_count} ({processed*100//total_count}%)")
            continue

        # Attempt the download, with a retry mechanism
        last_error = None
        for attempt in range(max_retries):
            try:
                result = fetch_satellite_image(
                    center_lat=center_lat,
                    center_lon=center_lon,
                    grid_id=grid_id,
                    output_dir=output_dir,
                    start_date=start_date,
                    end_date=end_date,
                    force_download=not skip_existing
                )

                results.append({
                    'grid_id': grid_id,
                    'status': 'success',
                    'file_path': result['file_path'],
                    'error_message': '',
                    'system_index': result['system_index'],
                    'acquisition_date': result['acquisition_date'],
                    'cloud_cover': result['cloud_cover']
                })
                success_count += 1
                last_error = None
                break

            except Exception as e:
                last_error = str(e)

                # For HTTP 429 errors, apply exponential backoff
                if '429' in last_error or 'Too Many Requests' in last_error:
                    wait_time = 30 * (2 ** attempt)  # 30s, 60s, 120s
                    print(f"Rate-limited; waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                elif attempt < max_retries - 1:
                    time.sleep(delay_seconds * (attempt + 1))

        # If all retries failed
        if last_error is not None:
            results.append({
                'grid_id': grid_id,
                'status': 'failed',
                'file_path': '',
                'error_message': last_error,
                'system_index': '',
                'acquisition_date': '',
                'cloud_cover': -1
            })
            failed_count += 1

            # Record the failure in the log
            with open(log_file, 'a', encoding='utf-8') as f:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"[{timestamp}] {grid_id}: {last_error}\n")

        # Delay to avoid GEE rate limiting
        time.sleep(delay_seconds)

        # Progress report
        processed = len(results)
        if processed % progress_interval == 0:
            print(f"Progress: {processed}/{total_count} ({processed*100//total_count}%)")

    # Print summary statistics
    print(f"\nDownload complete!")
    print(f"  Succeeded: {success_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Skipped: {skipped_count}")

    if failed_count > 0:
        print(f"Failure details recorded to: {log_file}")

    return pd.DataFrame(results)


def generate_image_index(image_dir: str) -> pd.DataFrame:
    """
    Generate the image index file.

    Scans all image files under the specified directory and generates an
    index CSV file containing metadata. The index file can be used by the
    subsequent visual feature extraction pipeline.

    Args:
        image_dir: Image directory path

    Returns:
        pd.DataFrame: Image index table, containing the following columns:
            - grid_id: Grid ID parsed from the file name
            - image_file_path: File path relative to the project root
            - file_size_bytes: File size (bytes)
            - download_time: File modification time
            - system_index: GEE image unique ID (if recorded)
            - acquisition_date: Image acquisition date (if recorded)
            - cloud_cover: Cloud cover percentage (if recorded)

    Side effects:
        Generates an 'image_index.csv' file under the image_dir directory
    """
    import json
    from datetime import datetime

    image_path = Path(image_dir)

    if not image_path.exists():
        raise ValueError(f"Image directory does not exist: {image_dir}")

    # Get the project root directory (used to compute relative paths)
    project_root = Path(__file__).parent.parent.parent

    # Scan for all PNG files
    image_files = list(image_path.glob('*.png'))

    if not image_files:
        print(f"Warning: no PNG files found in directory {image_dir}")
        return pd.DataFrame(columns=[
            'grid_id', 'image_file_path', 'file_size_bytes',
            'download_time', 'system_index', 'acquisition_date', 'cloud_cover'
        ])

    # Collect the index data
    index_data = []

    for image_file in image_files:
        # Parse grid_id from the file name (strip the .png extension)
        grid_id = image_file.stem

        # Get the file info
        file_stat = image_file.stat()
        file_size = file_stat.st_size
        download_time = datetime.fromtimestamp(file_stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')

        # Compute the relative path
        try:
            relative_path = image_file.relative_to(project_root)
        except ValueError:
            relative_path = image_file

        # Try to read the metadata file
        metadata_file = image_path / f"{grid_id}.meta.json"
        system_index = ''
        acquisition_date = ''
        cloud_cover = -1

        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                system_index = metadata.get('system_index', '')
                acquisition_date = metadata.get('acquisition_date', '')
                cloud_cover = metadata.get('cloud_cover', -1)
            except (json.JSONDecodeError, IOError):
                pass

        index_data.append({
            'grid_id': grid_id,
            'image_file_path': str(relative_path),
            'file_size_bytes': file_size,
            'download_time': download_time,
            'system_index': system_index,
            'acquisition_date': acquisition_date,
            'cloud_cover': cloud_cover
        })

    # Create the DataFrame
    df = pd.DataFrame(index_data)

    # Sort by grid_id
    df = df.sort_values('grid_id').reset_index(drop=True)

    # Save as CSV
    index_file = image_path / 'image_index.csv'
    df.to_csv(index_file, index=False, encoding='utf-8')

    print(f"Index file generated: {index_file}")
    print(f"  {len(df)} records total")

    return df


# =============================================================================
# Region-level GeoTIFF download and patch cropping (Phase A1 extension)
# =============================================================================

# Sentinel-2 six-band configuration (B2 blue, B3 green, B4 red, B8 near-infrared, B11 SWIR1, B12 SWIR2)
_S2_BANDS = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']

def _estimate_region_size(
    region_bounds,
    n_bands: int,
    scale_m: int
) -> Tuple[int, Tuple[float, float, float, float]]:
    """
    Estimate the region download size (in bytes) and return the EPSG:27700
    projected bbox.

    Args:
        region_bounds: A Shapely geometry, or an (minx, miny, maxx, maxy)
            bounding box (EPSG:4326)
        n_bands: Number of bands
        scale_m: Pixel resolution (meters)

    Returns:
        Tuple[int, Tuple]: (estimated byte count, (minx_27700, miny_27700, maxx_27700, maxy_27700))
    """
    import pyproj

    # Parse the bounding box
    if isinstance(region_bounds, tuple) and len(region_bounds) == 4:
        minx, miny, maxx, maxy = region_bounds
    else:
        minx, miny, maxx, maxy = region_bounds.bounds

    # Convert to EPSG:27700 to obtain metric dimensions
    transformer = pyproj.Transformer.from_crs(
        'EPSG:4326', 'EPSG:27700', always_xy=True
    )
    x1, y1 = transformer.transform(minx, miny)
    x2, y2 = transformer.transform(maxx, maxy)

    proj_bbox = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

    # Compute the pixel count and estimated size.
    # GEE's .divide() promotes the data to Float64 (8 bytes/pixel/band)
    width_m = proj_bbox[2] - proj_bbox[0]
    height_m = proj_bbox[3] - proj_bbox[1]
    width_px = int(math.ceil(width_m / scale_m))
    height_px = int(math.ceil(height_m / scale_m))
    estimated_bytes = width_px * height_px * n_bands * 8

    return estimated_bytes, proj_bbox


def _compute_tile_grid(
    projected_bbox: Tuple[float, float, float, float],
    n_bands: int,
    scale_m: int,
    target_bytes: int = 24_000_000
) -> List[Tuple[float, float, float, float]]:
    """
    Split the projected region into a tile grid (EPSG:27700 coordinates).

    Each tile targets 24MB (50% of the 48MB GEE limit), corresponding to
    roughly 7km x 7km @10m/6bands/Float64.

    Args:
        projected_bbox: (minx, miny, maxx, maxy) EPSG:27700 coordinates
        n_bands: Number of bands
        scale_m: Pixel resolution (meters)
        target_bytes: Target size per tile (bytes), default 24MB

    Returns:
        List[Tuple]: List of tile bboxes, each (minx, miny, maxx, maxy) in EPSG:27700
    """
    # Compute the tile side length (meters).
    # GEE's .divide() outputs Float64 -> 8 bytes/pixel/band
    # target_bytes = tile_px^2 * n_bands * 8 -> tile_px = sqrt(target / (n_bands*8))
    tile_px = int(math.floor(math.sqrt(target_bytes / (n_bands * 8))))
    tile_size_m = tile_px * scale_m  # ~7070m ~= 7km

    bminx, bminy, bmaxx, bmaxy = projected_bbox
    width_m = bmaxx - bminx
    height_m = bmaxy - bminy

    n_cols = int(math.ceil(width_m / tile_size_m))
    n_rows = int(math.ceil(height_m / tile_size_m))

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
    composite,
    tile_bboxes: List[Tuple[float, float, float, float]],
    bands: List[str],
    scale_m: int,
    crs: str,
    output_path: str
) -> str:
    """
    Download GEE imagery tile by tile and merge into a single GeoTIFF.

    Each tile is downloaded via getDownloadURL, with automatic retry up to
    3 times (exponential backoff) on failure. Once all tiles are downloaded,
    they are merged using rasterio.merge.

    Args:
        composite: The GEE Image object (already divided/clamped)
        tile_bboxes: List of tile bboxes (EPSG:27700 coordinates)
        bands: List of band names
        scale_m: Pixel resolution (meters)
        crs: Output CRS string (e.g. 'EPSG:27700')
        output_path: Final output file path

    Returns:
        str: The output file path
    """
    import time
    import requests
    import rasterio
    from rasterio.merge import merge
    import pyproj
    from shapely.geometry import box, mapping

    output_file = Path(output_path)
    # Place temp files in the same directory as the output file (avoids Windows cross-drive issues)
    temp_dir = output_file.parent / f'_tiles_tmp_{output_file.stem}'
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Coordinate transformer: EPSG:27700 -> EPSG:4326 (GEE requires 4326 coordinates)
    transformer_to_4326 = pyproj.Transformer.from_crs(
        'EPSG:27700', 'EPSG:4326', always_xy=True
    )

    tile_paths = []
    max_retries = 3

    for i, tile_bbox in enumerate(tile_bboxes):
        tile_path = temp_dir / f'tile_{i:04d}.tif'

        # Skip already-downloaded tiles (> 1KB is considered a valid file)
        if tile_path.exists() and tile_path.stat().st_size > 1024:
            size_mb = tile_path.stat().st_size / 1024 / 1024
            print(f"  Tile {i+1}/{len(tile_bboxes)}: already exists ({size_mb:.1f} MB)")
            tile_paths.append(tile_path)
            continue

        tx_min, ty_min, tx_max, ty_max = tile_bbox

        # Convert back to EPSG:4326 for GEE
        lon1, lat1 = transformer_to_4326.transform(tx_min, ty_min)
        lon2, lat2 = transformer_to_4326.transform(tx_max, ty_max)
        tile_geojson = mapping(box(
            min(lon1, lon2), min(lat1, lat2),
            max(lon1, lon2), max(lat1, lat2)
        ))
        ee_region = ee.Geometry(tile_geojson)

        # Download with retries
        for attempt in range(max_retries):
            try:
                url = composite.getDownloadURL({
                    'region': ee_region,
                    'scale': scale_m,
                    'crs': crs,
                    'format': 'GEO_TIFF',
                    'bands': bands,
                })

                response = requests.get(url, timeout=300, stream=True)
                response.raise_for_status()

                with open(tile_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                size_mb = tile_path.stat().st_size / 1024 / 1024
                print(f"  Tile {i+1}/{len(tile_bboxes)}: OK ({size_mb:.1f} MB)")
                tile_paths.append(tile_path)
                break

            except Exception as e:
                if attempt < max_retries - 1:
                    wait_sec = 5 * (3 ** attempt)  # 5s -> 15s -> 45s
                    print(
                        f"  Tile {i+1}/{len(tile_bboxes)}: "
                        f"retry {attempt+1}/{max_retries} (waiting {wait_sec}s)... {e}"
                    )
                    time.sleep(wait_sec)
                else:
                    raise RuntimeError(
                        f"Tile {i+1}/{len(tile_bboxes)} download failed (retried {max_retries} times): {e}"
                    ) from e

    # Merge all tiles.
    # The 6-band GeoTIFF output by GEE lacks the correct Photometric tag;
    # libtiff emits a large number of C-level warnings when reading it, so
    # stderr is temporarily redirected to silence them
    import shutil
    old_stderr_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, 2)
    try:
        if len(tile_paths) == 1:
            shutil.move(str(tile_paths[0]), str(output_file))
        else:
            datasets = [rasterio.open(p) for p in tile_paths]
            try:
                merged, merged_transform = merge(datasets, method='first')
                profile = datasets[0].profile.copy()
                profile.update({
                    'height': merged.shape[1],
                    'width': merged.shape[2],
                    'transform': merged_transform,
                    'photometric': 'MINISBLACK',
                })
                with rasterio.open(str(output_file), 'w', **profile) as dst:
                    dst.write(merged)
            finally:
                for ds in datasets:
                    ds.close()
    finally:
        os.dup2(old_stderr_fd, 2)
        os.close(devnull_fd)
        os.close(old_stderr_fd)

    # Only clean up the temp directory after a successful merge
    if temp_dir.exists():
        shutil.rmtree(str(temp_dir), ignore_errors=True)


def download_region_mosaic(
    region_bounds: Any,
    output_path: str,
    bands: Optional[List[str]] = None,
    scale_m: int = 10,
    composite_method: str = 'median',
    max_cloud_cover: int = 20,
    date_range: Tuple[str, str] = ('2022-01-01', '2023-12-31')
) -> str:
    """
    Download a multi-band median composite GeoTIFF covering the entire
    region from GEE.

    Uses getDownloadURL plus a spatial-tiling strategy. Internal workflow:
    1. Estimate the region size -> if <=45MB, download directly in a single request
    2. If larger, split into ~10km x 10km tiles -> download tile by tile -> merge with rasterio.merge
    3. Automatic retry up to 3 times on failure (exponential backoff)

    Args:
        region_bounds: Region boundary, supports the following formats:
            - A Shapely geometry object (Polygon/MultiPolygon)
            - An (minx, miny, maxx, maxy) bounding box tuple (EPSG:4326 coordinates)
        output_path: Output GeoTIFF file path
        bands: Band list, default ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
        scale_m: Output resolution (meters), fixed at 10m
        composite_method: Compositing method, 'median' or 'mean'
        max_cloud_cover: Maximum cloud cover percentage
        date_range: Time range tuple (start_date, end_date)

    Returns:
        str: The output file path

    Note:
        - The output CRS is EPSG:27700 (British National Grid projected CRS)
        - Pixel values are Surface Reflectance / 10000 (0-1 float)
        - B11/B12 have a native resolution of 20m; GEE automatically resamples to 10m
        - The single-download limit is 48MB; larger regions are automatically tiled
    """
    import time
    import requests
    from shapely.geometry import box, mapping

    # Initialize GEE
    _initialize_gee()

    if bands is None:
        bands = list(_S2_BANDS)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Output CRS: EPSG:27700 British National Grid
    crs = 'EPSG:27700'

    # --- (1) Parse the region boundary and build the GEE composite image ---
    if isinstance(region_bounds, tuple) and len(region_bounds) == 4:
        minx, miny, maxx, maxy = region_bounds
        geometry_geojson = mapping(box(minx, miny, maxx, maxy))
    else:
        geometry_geojson = mapping(region_bounds)

    ee_geometry = ee.Geometry(geometry_geojson)

    # Build the Sentinel-2 image collection
    start_date, end_date = date_range
    collection = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(ee_geometry)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', max_cloud_cover))
        .select(bands)
    )

    # Pixel-wise compositing
    if composite_method == 'median':
        composite = collection.median()
    elif composite_method == 'mean':
        composite = collection.mean()
    else:
        raise ValueError(
            f"Unsupported compositing method: {composite_method}. Use 'median' or 'mean'"
        )

    # Normalize to 0-1 (Surface Reflectance / 10000)
    composite = composite.divide(10000).clamp(0, 1)

    # --- (2) Estimate the region size and decide whether to tile ---
    estimated_bytes, proj_bbox = _estimate_region_size(
        region_bounds, len(bands), scale_m
    )

    # Single-download threshold of 40MB (leaves headroom under the 48MB GEE
    # limit, accounting for GeoTIFF metadata overhead)
    single_download_threshold = 40_000_000

    if estimated_bytes <= single_download_threshold:
        # --- (3a) Small region: download directly in a single request ---
        print(f"Region is small (~{estimated_bytes / 1e6:.0f} MB); downloading directly...")
        max_retries = 3
        for attempt in range(max_retries):
            try:
                url = composite.getDownloadURL({
                    'region': ee_geometry,
                    'scale': scale_m,
                    'crs': crs,
                    'format': 'GEO_TIFF',
                    'bands': bands,
                })

                response = requests.get(url, timeout=300, stream=True)
                response.raise_for_status()

                with open(output_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                break

            except Exception as e:
                if attempt < max_retries - 1:
                    wait_sec = 5 * (3 ** attempt)
                    print(f"  Download failed, retry {attempt+1}/{max_retries} (waiting {wait_sec}s)... {e}")
                    time.sleep(wait_sec)
                else:
                    raise RuntimeError(
                        f"Download failed (retried {max_retries} times): {e}"
                    ) from e
    else:
        # --- (3b) Large region: tiled download + merge ---
        tile_bboxes = _compute_tile_grid(proj_bbox, len(bands), scale_m)
        print(
            f"Region is large (~{estimated_bytes / 1e6:.0f} MB); "
            f"downloading in {len(tile_bboxes)} tiles..."
        )
        _download_and_merge_tiles(
            composite, tile_bboxes, bands, scale_m, crs, str(output_file)
        )

    size_mb = output_file.stat().st_size / 1024 / 1024
    print(f"Download complete: {output_file} ({size_mb:.1f} MB)")
    return str(output_file)


def extract_patch_from_mosaic(
    mosaic_path: str,
    center_x: float,
    center_y: float,
    ground_size_m: float
) -> np.ndarray:
    """
    Crop a patch of the given size from an already-downloaded GeoTIFF.

    Uses rasterio's windowed reading to crop a ground_size_m x ground_size_m
    rectangular region centered on (center_x, center_y). Does not load the
    entire mosaic into memory.

    Args:
        mosaic_path: GeoTIFF file path
        center_x: Center X coordinate (must match the GeoTIFF CRS, e.g. EPSG:27700 in meters)
        center_y: Center Y coordinate (same as above)
        ground_size_m: Ground-coverage side length of the patch (meters)

    Returns:
        np.ndarray: NumPy array of shape (H, W, C)
            - H, W are determined by ground_size_m / pixel_size
            - C = number of bands (default 6)
            - Pixels beyond the GeoTIFF extent at the boundary are filled with NaN

    Note:
        - Windowed reading is implemented via rasterio.windows.from_bounds
        - The entire mosaic is never loaded into memory
        - The coordinates (center_x, center_y) must match the GeoTIFF's CRS
    """
    import rasterio
    from rasterio.windows import Window

    half = ground_size_m / 2.0
    minx = center_x - half
    maxy = center_y + half  # Note: the GeoTIFF coordinate system's Y axis runs top to bottom

    with rasterio.open(mosaic_path) as src:
        pixel_size_x = src.res[0]
        pixel_size_y = src.res[1]
        expected_size = int(round(ground_size_m / pixel_size_x))

        # Manually compute the window position (avoids from_bounds compatibility issues).
        # transform: affine.Affine(a, b, c, d, e, f)
        # x = c + col * a + row * b
        # y = f + col * d + row * e
        t = src.transform
        # Invert: col = (x - c) / a, row = (y - f) / e
        col_off = (minx - t.c) / t.a
        row_off = (maxy - t.f) / t.e
        col_size = ground_size_m / abs(t.a)
        row_size = ground_size_m / abs(t.e)

        window = Window(col_off, row_off, col_size, row_size)

        # Read with boundless=True to allow reading beyond the extent (filled with nodata)
        data = src.read(
            window=window,
            boundless=True,
            fill_value=np.nan,
            out_shape=(src.count, expected_size, expected_size)
        )

        # data shape: (C, H, W) -> convert to (H, W, C)
        patch = np.transpose(data, (1, 2, 0)).astype(np.float64)

    return patch


def extract_patch_from_open_dataset(
    src,
    center_x: float,
    center_y: float,
    ground_size_m: float,
    transform=None,
    pixel_size_x: float = None,
    n_bands: int = None,
) -> np.ndarray:
    """
    Crop a patch from an already-open rasterio dataset (avoids repeated
    rasterio.open calls).

    Args:
        src: An already-open rasterio DatasetReader
        center_x: Center X coordinate
        center_y: Center Y coordinate
        ground_size_m: Ground-coverage side length of the patch (meters)
        transform: Pre-cached src.transform (optional, avoids repeated access)
        pixel_size_x: Pre-cached pixel resolution (optional)
        n_bands: Pre-cached band count (optional)

    Returns:
        np.ndarray: (H, W, C) array
    """
    from rasterio.windows import Window

    if transform is None:
        transform = src.transform
    if pixel_size_x is None:
        pixel_size_x = src.res[0]
    if n_bands is None:
        n_bands = src.count

    half = ground_size_m / 2.0
    minx = center_x - half
    maxy = center_y + half
    expected_size = int(round(ground_size_m / pixel_size_x))

    t = transform
    col_off = (minx - t.c) / t.a
    row_off = (maxy - t.f) / t.e
    col_size = ground_size_m / abs(t.a)
    row_size = ground_size_m / abs(t.e)

    window = Window(col_off, row_off, col_size, row_size)

    data = src.read(
        window=window,
        boundless=True,
        fill_value=np.nan,
        out_shape=(n_bands, expected_size, expected_size),
    )

    patch = np.transpose(data, (1, 2, 0)).astype(np.float64)
    return patch
