"""下载 GHSL Built-S 原生瓦片并生成不重采样的区域裁剪。"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from pathlib import Path
import time
from typing import Any
import zipfile

import numpy as np

from .base import BaseFetcher
from .cache_paths import cache_file, replace_file
from .registry import fetcher_registry


logger = logging.getLogger(__name__)

DEFAULT_PRODUCT_ID = "GHS_BUILT_S_E2020_GLOBE_R2023A_54009_100_V1_0"
DEFAULT_TILE_SCHEMA_URL = (
    "https://ghsl.jrc.ec.europa.eu/download/GHSL_data_54009_shapefile.zip"
)
DEFAULT_TILE_BASE_URL = (
    "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/"
    "GHS_BUILT_S_GLOBE_R2023A/GHS_BUILT_S_E2020_GLOBE_R2023A_54009_100/"
    "V1-0/tiles"
)
NATIVE_CRS = "ESRI:54009"
NATIVE_RESOLUTION_M = 100.0
NATIVE_X_ORIGIN = -18_041_000.0
NATIVE_Y_ORIGIN = 9_000_000.0
OFFICIAL_NODATA = 65_535


def _normalise_bounds(region_bounds: Any) -> tuple[float, float, float, float]:
    raw = region_bounds.bounds if hasattr(region_bounds, "bounds") else region_bounds
    bounds = tuple(float(value) for value in raw)
    if len(bounds) != 4 or not np.isfinite(bounds).all():
        raise ValueError("region_bounds 必须是四个有限的 EPSG:4326 坐标")
    if bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
        raise ValueError(f"无效的 EPSG:4326 bounds: {bounds}")
    return bounds


@fetcher_registry.register("ghsl_built_s", description="GHSL Built-S 2020 原生瓦片")
class GhslBuiltSurfaceFetcher(BaseFetcher):
    """复用共享原生瓦片，并为每个区域生成三通道原生裁剪。"""

    def __init__(self, config: dict):
        super().__init__(config)
        self.product_id = str(config.get("product_id", DEFAULT_PRODUCT_ID))
        self.tile_schema_url = str(
            config.get("tile_schema_url", DEFAULT_TILE_SCHEMA_URL)
        )
        self.tile_base_url = str(config.get("tile_base_url", DEFAULT_TILE_BASE_URL))
        self.resolution_m = int(config.get("resolution_m", 100))
        if self.product_id != DEFAULT_PRODUCT_ID:
            raise ValueError(f"不支持的 GHSL product_id: {self.product_id}")
        if self.resolution_m != 100:
            raise ValueError("C/U/Z 只支持 GHSL Built-S 原生 100 m 分辨率")

    def _request(self, region_bounds: Any) -> dict:
        return {
            "bounds_epsg4326": list(_normalise_bounds(region_bounds)),
            "product_id": self.product_id,
            "tile_schema_url": self.tile_schema_url,
            "tile_base_url": self.tile_base_url,
        }

    def get_cache_key(self, region_bounds: Any) -> str:
        payload = json.dumps(
            self._request(region_bounds), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.md5(payload, usedforsecurity=False).hexdigest()[:16]

    def get_cache_path(self, region_bounds: Any, cache_dir: str) -> Path:
        return cache_file(
            cache_dir,
            "ghsl_built_s",
            "crops",
            f"{self.get_cache_key(region_bounds)}.tif",
        )

    @staticmethod
    def _download(url: str, destination: Path) -> None:
        if destination.exists():
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(f".{destination.name}.part")
        if partial.exists():
            partial.unlink()
        import requests

        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = requests.get(url, timeout=300, stream=True)
                response.raise_for_status()
                expected = response.headers.get("Content-Length")
                with partial.open("wb") as stream:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            stream.write(chunk)
                if expected is not None and partial.stat().st_size != int(expected):
                    raise RuntimeError(f"下载长度不一致: {url}")
                replace_file(partial, destination)
                return
            except Exception as error:  # noqa: BLE001 - 有界网络重试
                last_error = error
                if partial.exists():
                    partial.unlink()
                if attempt < 2:
                    time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"下载失败: {url}") from last_error

    def _cache_root(self, cache_dir: str) -> Path:
        root = Path(cache_dir).resolve() / "ghsl_built_s"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _load_tile_schema(self, root: Path):
        import geopandas as gpd
        from pyproj import CRS

        schema_zip = root / "raw" / "GHSL_data_54009_shapefile.zip"
        self._download(self.tile_schema_url, schema_zip)
        with zipfile.ZipFile(schema_zip) as archive:
            broken = archive.testzip()
            if broken is not None:
                raise RuntimeError(f"GHSL tile schema ZIP CRC 失败: {broken}")
        schema = gpd.read_file(f"zip://{schema_zip}")
        expected = {"tile_id", "left", "top", "right", "bottom", "geometry"}
        if set(schema.columns) != expected or schema.empty:
            raise RuntimeError("GHSL tile schema 字段不符合预期")
        if schema.crs is None or schema.crs != CRS.from_user_input(NATIVE_CRS):
            raise RuntimeError(f"GHSL tile schema CRS 不一致: {schema.crs}")
        return schema

    def _tile_url(self, tile_id: str) -> str:
        return f"{self.tile_base_url}/{self.product_id}_{tile_id}.zip"

    def _materialise_tile(self, root: Path, tile_id: str, record) -> Path:
        import rasterio
        from rasterio.crs import CRS

        tile_dir = root / "raw" / "tiles"
        raw_zip = tile_dir / f"{self.product_id}_{tile_id}.zip"
        tile_path = tile_dir / f"{self.product_id}_{tile_id}.tif"
        self._download(self._tile_url(tile_id), raw_zip)
        expected_member = tile_path.name
        with zipfile.ZipFile(raw_zip) as archive:
            broken = archive.testzip()
            if broken is not None:
                raise RuntimeError(f"{raw_zip.name}: ZIP CRC 失败: {broken}")
            tif_members = [name for name in archive.namelist() if name.lower().endswith(".tif")]
            if tif_members != [expected_member]:
                raise RuntimeError(f"{raw_zip.name}: TIFF 成员不唯一")
            if not tile_path.exists():
                tile_path.parent.mkdir(parents=True, exist_ok=True)
                partial = tile_path.with_name(f".{tile_path.name}.part")
                if partial.exists():
                    partial.unlink()
                with archive.open(expected_member) as source, partial.open("wb") as target:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        target.write(chunk)
                replace_file(partial, tile_path)

        with rasterio.open(tile_path) as src:
            transform = src.transform
            if (
                src.count != 1
                or src.dtypes != ("uint16",)
                or src.nodata != float(OFFICIAL_NODATA)
                or src.crs != CRS.from_string(NATIVE_CRS)
                or transform.b != 0.0
                or transform.d != 0.0
                or transform.a != NATIVE_RESOLUTION_M
                or transform.e != -NATIVE_RESOLUTION_M
            ):
                raise RuntimeError(f"{tile_path.name}: 原生栅格元数据不一致")
            schema_bounds = (
                float(record.left),
                float(record.bottom),
                float(record.right),
                float(record.top),
            )
            if not (
                src.bounds.left >= schema_bounds[0]
                and src.bounds.bottom >= schema_bounds[1]
                and src.bounds.right <= schema_bounds[2]
                and src.bounds.top <= schema_bounds[3]
            ):
                raise RuntimeError(f"{tile_path.name}: 栅格范围超出 tile schema")
            x_phase = (transform.c - NATIVE_X_ORIGIN) / NATIVE_RESOLUTION_M
            y_phase = (NATIVE_Y_ORIGIN - transform.f) / NATIVE_RESOLUTION_M
            phase_tolerance = (
                np.finfo(np.float64).eps
                * max(abs(x_phase), abs(y_phase), 1.0)
                * 128.0
            )
            if (
                abs(x_phase - round(x_phase)) > phase_tolerance
                or abs(y_phase - round(y_phase)) > phase_tolerance
            ):
                raise RuntimeError(f"{tile_path.name}: 原生栅格 lattice phase 不一致")
        return tile_path

    @staticmethod
    def _projected_bbox(bounds: tuple[float, float, float, float], target_crs):
        import geopandas as gpd
        from shapely.geometry import box

        return gpd.GeoSeries([box(*bounds)], crs="EPSG:4326").to_crs(target_crs).iloc[0]

    def _build_crop(
        self,
        output: Path,
        tile_paths: list[Path],
        projected_bbox,
    ) -> None:
        import rasterio
        from rasterio.transform import from_origin
        from rasterio.windows import Window

        minx, miny, maxx, maxy = projected_bbox.bounds
        left = NATIVE_X_ORIGIN + math.floor((minx - NATIVE_X_ORIGIN) / 100.0) * 100.0
        right = NATIVE_X_ORIGIN + math.ceil((maxx - NATIVE_X_ORIGIN) / 100.0) * 100.0
        top = NATIVE_Y_ORIGIN - math.floor((NATIVE_Y_ORIGIN - maxy) / 100.0) * 100.0
        bottom = NATIVE_Y_ORIGIN - math.ceil((NATIVE_Y_ORIGIN - miny) / 100.0) * 100.0
        width = int(round((right - left) / 100.0))
        height = int(round((top - bottom) / 100.0))
        if width <= 0 or height <= 0:
            raise RuntimeError("GHSL crop 尺寸无效")

        crop = np.full((height, width), OFFICIAL_NODATA, dtype=np.uint16)
        written = np.zeros((height, width), dtype=np.uint16)
        valid = np.zeros((height, width), dtype=np.uint16)
        for tile_path in tile_paths:
            with rasterio.open(tile_path) as src:
                overlap_left = max(left, src.bounds.left)
                overlap_right = min(right, src.bounds.right)
                overlap_bottom = max(bottom, src.bounds.bottom)
                overlap_top = min(top, src.bounds.top)
                if overlap_left >= overlap_right or overlap_bottom >= overlap_top:
                    continue
                src_col = int(round((overlap_left - src.bounds.left) / 100.0))
                src_row = int(round((src.bounds.top - overlap_top) / 100.0))
                n_cols = int(round((overlap_right - overlap_left) / 100.0))
                n_rows = int(round((overlap_top - overlap_bottom) / 100.0))
                dst_col = int(round((overlap_left - left) / 100.0))
                dst_row = int(round((top - overlap_top) / 100.0))
                block = src.read(1, window=Window(src_col, src_row, n_cols, n_rows))
                rows = slice(dst_row, dst_row + n_rows)
                cols = slice(dst_col, dst_col + n_cols)
                if np.any(written[rows, cols] != 0):
                    raise RuntimeError("GHSL 原生瓦片发生正面积重叠")
                crop[rows, cols] = block
                written[rows, cols] = 1
                valid[rows, cols] = (block != OFFICIAL_NODATA).astype(np.uint16)

        output.parent.mkdir(parents=True, exist_ok=True)
        partial = output.with_name(f".{output.name}.part")
        if partial.exists():
            partial.unlink()
        transform = from_origin(left, top, 100.0, 100.0)
        try:
            with rasterio.open(
                partial,
                "w",
                driver="GTiff",
                width=width,
                height=height,
                count=3,
                dtype="uint16",
                crs=NATIVE_CRS,
                transform=transform,
                nodata=None,
                compress="DEFLATE",
            ) as dst:
                dst.write(crop, 1)
                dst.write(written, 2)
                dst.write(valid, 3)
            replace_file(partial, output)
        finally:
            if partial.exists():
                partial.unlink()

    def fetch(self, region_bounds: Any, cache_dir: str, **_kwargs) -> str:
        output = self.get_cache_path(region_bounds, cache_dir)
        if output.exists():
            return str(output)
        root = self._cache_root(cache_dir)
        schema = self._load_tile_schema(root)
        bounds = _normalise_bounds(region_bounds)
        projected_bbox = self._projected_bbox(bounds, schema.crs)
        selected = schema[schema.geometry.intersects(projected_bbox)].sort_values("tile_id")
        if selected.empty:
            raise RuntimeError("没有 GHSL 原生瓦片覆盖请求区域")
        tiles = [
            self._materialise_tile(root, str(row.tile_id), row)
            for row in selected.itertuples(index=False)
        ]
        self._build_crop(output, tiles, projected_bbox)
        return str(output)

    def validate_cache(self, region_bounds: Any, cache_dir: str) -> str:
        """按 query key 查找缓存，不校验任何存储哈希。"""

        output = self.get_cache_path(region_bounds, cache_dir)
        if not output.is_file():
            raise FileNotFoundError(f"GHSL 缓存不存在: {output}")
        return str(output)
