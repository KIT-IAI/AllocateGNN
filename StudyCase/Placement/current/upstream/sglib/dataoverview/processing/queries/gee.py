"""Country-neutral Google Earth Engine GeoTIFF acquisition."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import zipfile

import geopandas as gpd
import requests

from sglib.core.infra.artifacts import atomic_json
from sglib.core.infra.hashing import sha256_file
from sglib.core.infra.paths import find_repo_root


class GeeAcquisitionError(RuntimeError):
    pass


def _target(target_root: Path, output: object) -> Path:
    root = target_root.resolve()
    target = (root / str(output)).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise GeeAcquisitionError("GEE output escapes the country raw root") from exc
    return target


def _validate_raster(target: Path, query: dict) -> dict:
    import rasterio

    bands = list(map(str, query["bands"]))
    with rasterio.open(target) as source:
        if source.count != len(bands):
            raise GeeAcquisitionError(
                f"GEE raster has {source.count} bands, expected {len(bands)}"
            )
        if source.crs is None:
            raise GeeAcquisitionError("GEE raster has no CRS")
        expected_crs = str(query.get("output_crs", "")).strip()
        if expected_crs and source.crs.to_string() != expected_crs:
            raise GeeAcquisitionError(
                f"GEE raster CRS is {source.crs}, expected {expected_crs}"
            )
        return {
            "bands": bands,
            "count": source.count,
            "crs": source.crs.to_string(),
            "shape": [source.height, source.width],
            "bounds": list(source.bounds),
        }


def _receipt_current(target: Path, receipt: Path) -> bool:
    if not target.is_file() or target.stat().st_size == 0 or not receipt.is_file():
        return False
    try:
        document = json.loads(receipt.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    artifacts = document.get("artifacts", [])
    return any(
        item.get("sha256") == sha256_file(target)
        and int(item.get("bytes", -1)) == target.stat().st_size
        for item in artifacts
        if isinstance(item, dict)
    )


def _bounds(repo_root: Path, query: dict) -> list[float]:
    configured = query.get("bounds_source")
    if not configured:
        raise GeeAcquisitionError("GEE query must declare bounds_source")
    path = (repo_root / str(configured)).resolve()
    try:
        path.relative_to(repo_root)
    except ValueError as exc:
        raise GeeAcquisitionError("GEE bounds_source escapes repository") from exc
    if not path.is_file():
        raise FileNotFoundError(f"GEE bounds source is missing: {path}")
    layer = query.get("bounds_layer")
    regions = gpd.read_file(path, layer=str(layer)) if layer else gpd.read_file(path)
    if regions.empty or regions.crs is None:
        raise GeeAcquisitionError("GEE bounds source is empty or has no CRS")
    return list(map(float, regions.to_crs("EPSG:4326").total_bounds))


def _credentials(repo_root: Path, credential_env: str):
    import ee

    configured = os.environ.get(credential_env)
    if not configured:
        raise GeeAcquisitionError(
            f"missing GEE credential environment variable: {credential_env}"
        )
    key_path = Path(configured)
    if not key_path.is_absolute():
        key_path = (repo_root / key_path).resolve()
    key_data = json.loads(key_path.read_text(encoding="utf-8"))
    email = key_data.get("client_email")
    if not isinstance(email, str) or not email:
        raise GeeAcquisitionError("GEE service-account file has no client_email")
    ee.Initialize(ee.ServiceAccountCredentials(email, str(key_path)))
    return ee


def _download(url: str, target: Path) -> None:
    download = target.with_name(f".{target.name}.download")
    extracted = target.with_name(f".{target.name}.part")
    target.parent.mkdir(parents=True, exist_ok=True)
    for temporary in (download, extracted):
        temporary.unlink(missing_ok=True)
    try:
        with requests.get(url, stream=True, timeout=(30, 600)) as response:
            response.raise_for_status()
            expected = response.headers.get("Content-Length")
            size = 0
            with download.open("xb") as output:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        output.write(chunk)
                        size += len(chunk)
                output.flush()
                os.fsync(output.fileno())
            if expected is not None and size != int(expected):
                raise GeeAcquisitionError(
                    f"GEE download length mismatch: expected={expected}, got={size}"
                )
        if zipfile.is_zipfile(download):
            with zipfile.ZipFile(download) as archive:
                broken = archive.testzip()
                if broken is not None:
                    raise GeeAcquisitionError(f"GEE ZIP CRC failed: {broken}")
                members = [
                    item
                    for item in archive.infolist()
                    if item.filename.lower().endswith(".tif")
                ]
                if len(members) != 1:
                    raise GeeAcquisitionError(
                        "GEE archive must contain exactly one GeoTIFF"
                    )
                with archive.open(members[0]) as source, extracted.open("xb") as output:
                    shutil.copyfileobj(source, output)
                    output.flush()
                    os.fsync(output.fileno())
        else:
            os.replace(download, extracted)
        os.replace(extracted, target)
    finally:
        download.unlink(missing_ok=True)
        extracted.unlink(missing_ok=True)


def run(query: dict, target_root: Path, *, refresh: bool = False) -> list[Path]:
    required = {"output", "dataset_id", "date_range", "bands", "composite_method"}
    missing = required - set(query)
    if missing:
        raise GeeAcquisitionError(f"GEE query is missing {sorted(missing)}")
    target = _target(target_root, query["output"])
    receipt = target.parent / "acquisition_receipt.json"
    if target.is_file() and not refresh:
        details = _validate_raster(target, query)
        if not _receipt_current(target, receipt):
            atomic_json(
                {
                    "schema_version": "sg_gee_acquisition_receipt_v1",
                    "status": "PASS",
                    "dataset_id": str(query["dataset_id"]),
                    "date_range": list(query["date_range"]),
                    "reused": True,
                    "details": details,
                    "artifacts": [
                        {
                            "path": target.name,
                            "bytes": target.stat().st_size,
                            "sha256": sha256_file(target),
                        }
                    ],
                },
                receipt,
            )
        return [target, receipt]

    repo_root = find_repo_root(__file__)
    credential_env = str(
        query.get("credential_env", "GEE_SERVICE_ACCOUNT_KEY_PATH")
    )
    ee = _credentials(repo_root, credential_env)
    collection = ee.ImageCollection(str(query["dataset_id"])).filterDate(
        *list(map(str, query["date_range"]))
    )
    method = str(query["composite_method"])
    if method == "median":
        image = collection.median()
    elif method == "mean":
        image = collection.mean()
    elif method == "single_image":
        image = ee.Image(collection.first())
    else:
        raise GeeAcquisitionError(f"unsupported GEE composite method: {method}")
    bands = list(map(str, query["bands"]))
    image = image.select(bands).max(ee.Image(0))
    bounds = _bounds(repo_root, query)
    region = ee.Geometry.Rectangle(bounds, proj="EPSG:4326", geodesic=False)
    request = {
        "name": f"ntl_{target_root.name}",
        "format": "GEO_TIFF",
        "bands": bands,
        "region": region,
        "crs": str(query.get("output_crs", "EPSG:4326")),
        "scale": int(query.get("scale_m", 500)),
    }
    _download(image.getDownloadURL(request), target)
    details = _validate_raster(target, query)
    atomic_json(
        {
            "schema_version": "sg_gee_acquisition_receipt_v1",
            "status": "PASS",
            "dataset_id": str(query["dataset_id"]),
            "date_range": list(query["date_range"]),
            "reused": False,
            "details": details,
            "artifacts": [
                {
                    "path": target.name,
                    "bytes": target.stat().st_size,
                    "sha256": sha256_file(target),
                }
            ],
        },
        receipt,
    )
    return [target, receipt]


__all__ = ["GeeAcquisitionError", "run"]
