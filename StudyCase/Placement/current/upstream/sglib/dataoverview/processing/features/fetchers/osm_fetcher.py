"""从已落地的 Geofabrik PBF 快照派生并缓存 OSM 土地利用 GeoParquet。"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS

from .base import BaseFetcher
from .cache_paths import cache_file, replace_file
from .registry import fetcher_registry


logger = logging.getLogger(__name__)

OSM_SAFE_COLUMNS = ("element_type", "osmid", "landuse", "geometry")


class OsmCacheValidationError(RuntimeError):
    """OSM 缓存缺失或无法读取。"""


@fetcher_registry.register("osm", description="OpenStreetMap GeoParquet")
class OsmFetcher(BaseFetcher):
    """把 PBF 快照中的 OSM 土地利用区域子集保存为不可执行的 GeoParquet。"""

    def __init__(self, config: dict):
        super().__init__(config)
        self.tags = config.get("tags", {"landuse": True})
        self.buffer_m = config.get("buffer_m", 0)
        self.last_cache_status: str | None = None
        self._canonical_tags = json.loads(
            json.dumps(self.tags, ensure_ascii=False, sort_keys=True)
        )

    @staticmethod
    def _bounds(region_bounds: Any) -> tuple[float, float, float, float]:
        values = region_bounds.bounds if hasattr(region_bounds, "bounds") else region_bounds
        bounds = tuple(float(value) for value in values)
        if len(bounds) != 4 or not np.isfinite(bounds).all():
            raise ValueError("OSM query bounds 必须是四个有限值")
        minx, miny, maxx, maxy = bounds
        if not (-180 <= minx < maxx <= 180 and -90 <= miny < maxy <= 90):
            raise ValueError("OSM query bounds 必须是有序 EPSG:4326 坐标")
        return bounds

    def get_cache_key(self, region_bounds: Any) -> str:
        bounds = self._bounds(region_bounds)
        key_text = f"{bounds}_{self._canonical_tags}"
        return hashlib.md5(key_text.encode(), usedforsecurity=False).hexdigest()[:12]

    def get_cache_path(self, region_bounds: Any, cache_dir: str | os.PathLike) -> Path:
        return cache_file(
            cache_dir,
            "osm",
            self.get_cache_key(region_bounds),
            "osm_landuse.parquet",
        )

    def get_manifest_path(self, region_bounds: Any, cache_dir: str | os.PathLike) -> Path:
        """兼容旧调用；新缓存不再生成 manifest。"""

        return self.get_cache_path(region_bounds, cache_dir).with_name("manifest.json")

    def get_legacy_cache_path(
        self, region_bounds: Any, cache_dir: str | os.PathLike
    ) -> Path:
        return Path(cache_dir) / "osm" / f"{self.get_cache_key(region_bounds)}.pickle"

    @staticmethod
    def _validate_frame(frame: gpd.GeoDataFrame) -> None:
        if tuple(frame.columns) != OSM_SAFE_COLUMNS:
            raise OsmCacheValidationError(
                f"OSM GeoParquet 列必须是 {OSM_SAFE_COLUMNS}"
            )
        if frame.crs is None or not CRS.from_user_input(frame.crs).equals(
            CRS.from_epsg(4326)
        ):
            raise OsmCacheValidationError("OSM GeoParquet CRS 必须是 EPSG:4326")
        if frame.empty:
            raise OsmCacheValidationError("OSM GeoParquet 为空")

    def _normalise_response(self, payload: object) -> gpd.GeoDataFrame:
        if not isinstance(payload, gpd.GeoDataFrame) or payload.empty:
            raise OsmCacheValidationError("OSM 响应必须是非空 GeoDataFrame")
        if payload.crs is None or not CRS.from_user_input(payload.crs).equals(
            CRS.from_epsg(4326)
        ):
            raise OsmCacheValidationError("OSM 响应 CRS 必须是 EPSG:4326")

        frame = payload.reset_index()
        element_source = next(
            (name for name in ("element_type", "element") if name in frame.columns),
            None,
        )
        id_source = next((name for name in ("osmid", "id") if name in frame.columns), None)
        if element_source is None or id_source is None or "landuse" not in frame.columns:
            raise OsmCacheValidationError("OSM 响应缺少 element/id/landuse")
        frame = frame[[element_source, id_source, "landuse", payload.geometry.name]].copy()
        frame.columns = list(OSM_SAFE_COLUMNS)
        for column in ("element_type", "osmid"):
            if frame[column].isna().any():
                raise OsmCacheValidationError(f"OSM {column} 不能为空")
            frame[column] = frame[column].astype("string")
            if frame[column].str.len().eq(0).any():
                raise OsmCacheValidationError(f"OSM {column} 不能为空字符串")
        invalid_landuse = frame["landuse"].dropna().map(
            lambda value: not isinstance(value, str)
        )
        if bool(invalid_landuse.any()):
            raise OsmCacheValidationError("OSM landuse 必须是字符串或空值")
        frame["landuse"] = frame["landuse"].astype("string")
        result = gpd.GeoDataFrame(frame, geometry="geometry", crs="EPSG:4326")
        if (
            result.geometry.isna().any()
            or result.geometry.is_empty.any()
            or not result.geometry.is_valid.all()
        ):
            raise OsmCacheValidationError("OSM geometry 必须非空且有效")
        duplicate_rows = result.duplicated(
            subset=["element_type", "osmid"], keep=False
        )
        for identity, group in result.loc[duplicate_rows].groupby(
            ["element_type", "osmid"], sort=False
        ):
            first_landuse = group["landuse"].iloc[0]
            first_missing = bool(pd.isna(first_landuse))
            if any(
                bool(pd.isna(value)) != first_missing
                or (not first_missing and value != first_landuse)
                for value in group["landuse"].iloc[1:]
            ):
                raise OsmCacheValidationError(
                    f"OSM duplicate identity {identity!r} 的 landuse 冲突"
                )
            first_geometry = group.geometry.iloc[0]
            if any(
                not first_geometry.equals(geometry)
                or first_geometry.wkb != geometry.wkb
                for geometry in group.geometry.iloc[1:]
            ):
                raise OsmCacheValidationError(
                    f"OSM duplicate identity {identity!r} 的 geometry 冲突"
                )
        result = result.drop_duplicates(
            subset=["element_type", "osmid"], keep="first", ignore_index=True
        )
        result = result.sort_values(
            ["element_type", "osmid"], kind="mergesort", ignore_index=True
        )
        self._validate_frame(result)
        return result

    def _write_frame(self, path: Path, frame: gpd.GeoDataFrame) -> str:
        if path.exists():
            self.last_cache_status = "hit"
            return str(path)
        partial = path.with_name(f".{path.name}.part")
        if partial.exists():
            partial.unlink()
        frame.to_parquet(partial, index=False)
        replace_file(partial, path)
        self.last_cache_status = "miss"
        return str(path)

    def validate_cache(self, region_bounds: Any, cache_dir: str | os.PathLike) -> str:
        """按 query key 查找缓存，不校验任何存储哈希。"""

        path = self.get_cache_path(region_bounds, cache_dir)
        if not path.is_file():
            raise OsmCacheValidationError(f"OSM 缓存不存在: {path}")
        return str(path)

    def validate_snapshot_cache(
        self,
        region_bounds: Any,
        cache_dir: str | os.PathLike,
        expected_source: dict,
    ) -> str:
        return self.validate_cache(region_bounds, cache_dir)

    def fetch(self, region_bounds: Any, cache_dir: str) -> str:
        """返回由 PBF 快照写出的区域缓存；不联网查询。

        OSM 土地利用只从已落地的 Geofabrik 快照派生（``fetch_from_pbf`` /
        ``fetch_many_from_pbf``）；缓存缺失时报错，而不是在线补查。
        """

        return self.validate_cache(region_bounds, cache_dir)

    @staticmethod
    def _convert_pbf_layer(frame: gpd.GeoDataFrame, layer: str) -> gpd.GeoDataFrame:
        if "landuse" not in frame or frame.crs is None:
            raise OsmCacheValidationError(f"OSM {layer} 缺少 landuse/CRS")
        if layer == "multipolygons":
            if "osm_id" not in frame or "osm_way_id" not in frame:
                raise OsmCacheValidationError("OSM multipolygons 缺少 identity")
            is_way = frame["osm_way_id"].notna()
            is_relation = frame["osm_id"].notna()
            if bool((is_way == is_relation).any()):
                raise OsmCacheValidationError(
                    "OSM multipolygon 必须只有 way 或 relation identity"
                )
            element = is_way.map({True: "way", False: "relation"})
            osmid = frame["osm_way_id"].where(is_way, frame["osm_id"])
        else:
            if "osm_id" not in frame:
                raise OsmCacheValidationError(f"OSM {layer} 缺少 osm_id")
            element_type = {
                "points": "node",
                "lines": "way",
                "multilinestrings": "relation",
                "other_relations": "relation",
            }[layer]
            element = pd.Series(element_type, index=frame.index, dtype="string")
            osmid = frame["osm_id"]
        return gpd.GeoDataFrame(
            {
                "element_type": element.astype("string"),
                "osmid": osmid.astype("string"),
                "landuse": frame["landuse"].astype("string"),
            },
            geometry=frame.geometry,
            crs=frame.crs,
        ).to_crs("EPSG:4326")

    @staticmethod
    def _repair_pbf_geometries(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        from shapely import make_valid

        if frame.geometry.isna().any() or frame.geometry.is_empty.any():
            raise OsmCacheValidationError("OSM PBF 包含空 geometry")
        invalid = ~frame.geometry.is_valid
        if bool(invalid.any()):
            frame = frame.copy()
            frame.loc[invalid, "geometry"] = list(
                make_valid(frame.loc[invalid].geometry.array)
            )
        if not frame.geometry.is_valid.all():
            raise OsmCacheValidationError("OSM PBF geometry 修复后仍无效")
        return frame

    @staticmethod
    def _read_pbf_region(
        pbf_path: str | os.PathLike,
        bounds: tuple[float, float, float, float],
        osm_config_path: str | os.PathLike | None = None,
    ) -> gpd.GeoDataFrame:
        """通过 GDAL spatial filter 仅读取区域内 landuse，避免全国量入内存。"""

        import pyogrio

        pbf = Path(pbf_path).resolve()
        if not pbf.is_file():
            raise FileNotFoundError(f"OSM PBF 不存在: {pbf}")
        config = (
            Path(osm_config_path).resolve()
            if osm_config_path is not None
            else Path(__file__).with_name("osmconf_landuse.ini").resolve()
        )
        if not config.is_file():
            raise FileNotFoundError(f"OSM_CONFIG_FILE 不存在: {config}")

        old_config = pyogrio.get_gdal_config_option("OSM_CONFIG_FILE")
        pieces: list[gpd.GeoDataFrame] = []
        try:
            pyogrio.set_gdal_config_options({"OSM_CONFIG_FILE": str(config)})
            expected_layers = [
                "points",
                "lines",
                "multilinestrings",
                "multipolygons",
                "other_relations",
            ]
            layers = [str(item[0]) for item in pyogrio.list_layers(pbf)]
            if layers != expected_layers:
                raise OsmCacheValidationError(f"OSM PBF layer 顺序不一致: {layers}")
            for layer in expected_layers:
                frame = pyogrio.read_dataframe(
                    pbf,
                    layer=layer,
                    where="landuse IS NOT NULL",
                    bbox=bounds,
                    use_arrow=True,
                    INTERLEAVED_READING="YES",
                )
                if frame.empty:
                    continue
                pieces.append(OsmFetcher._convert_pbf_layer(frame, layer))
        finally:
            pyogrio.set_gdal_config_options({"OSM_CONFIG_FILE": old_config})
        if not pieces:
            raise OsmCacheValidationError("区域内没有 OSM landuse 要素")
        combined = gpd.GeoDataFrame(
            pd.concat(pieces, ignore_index=True),
            geometry="geometry",
            crs="EPSG:4326",
        )
        return OsmFetcher._repair_pbf_geometries(combined)

    def fetch_from_pbf(
        self,
        region_bounds: Any,
        cache_dir: str | os.PathLike,
        pbf_path: str | os.PathLike,
    ) -> str:
        """从国家级 PBF 的区域子集生成普通 GeoParquet 缓存。"""

        output = self.get_cache_path(region_bounds, cache_dir)
        if output.exists():
            self.last_cache_status = "hit"
            return str(output)
        bounds = self._bounds(region_bounds)
        source = self._read_pbf_region(pbf_path, bounds)
        from shapely.geometry import box

        subset = source.loc[source.geometry.intersects(box(*bounds))].copy()
        if subset.empty:
            raise OsmCacheValidationError("区域内没有 OSM landuse 要素")
        return self._write_frame(output, self._normalise_response(subset))

    def fetch_many_from_pbf(
        self,
        region_bounds: dict[str, Any],
        cache_dir: str | os.PathLike,
        pbf_path: str | os.PathLike,
    ) -> dict[str, str]:
        """按 layer 流式拆分一个国家 PBF，避免逐区域重复扫描或全国量驻内存。"""

        import pyogrio
        from shapely.geometry import box
        from tempfile import TemporaryDirectory

        outputs = {
            name: self.get_cache_path(bounds, cache_dir)
            for name, bounds in region_bounds.items()
        }
        pending = {name: path for name, path in outputs.items() if not path.exists()}
        if not pending:
            return {name: str(path) for name, path in outputs.items()}
        bounds = {name: self._bounds(region_bounds[name]) for name in pending}
        boxes = {name: box(*values) for name, values in bounds.items()}
        union_bbox = (
            min(value[0] for value in bounds.values()),
            min(value[1] for value in bounds.values()),
            max(value[2] for value in bounds.values()),
            max(value[3] for value in bounds.values()),
        )
        pbf = Path(pbf_path).resolve()
        if not pbf.is_file():
            raise FileNotFoundError(f"OSM PBF 不存在: {pbf}")
        config = Path(__file__).with_name("osmconf_landuse.ini").resolve()
        fragment_parent = Path(cache_dir).resolve() / "osm"
        fragment_parent.mkdir(parents=True, exist_ok=True)
        legacy_fragments = fragment_parent / ".fragments"
        if legacy_fragments.exists():
            shutil.rmtree(legacy_fragments)
        layers = [
            "points",
            "lines",
            "multilinestrings",
            "multipolygons",
            "other_relations",
        ]
        with TemporaryDirectory(dir=fragment_parent, prefix=".fragments-") as temporary:
            fragment_root = Path(temporary)
            old_config = pyogrio.get_gdal_config_option("OSM_CONFIG_FILE")
            try:
                pyogrio.set_gdal_config_options({"OSM_CONFIG_FILE": str(config)})
                observed_layers = [str(item[0]) for item in pyogrio.list_layers(pbf)]
                if observed_layers != layers:
                    raise OsmCacheValidationError(
                        f"OSM PBF layer 顺序不一致: {observed_layers}"
                    )
                for layer in layers:
                    raw = pyogrio.read_dataframe(
                        pbf,
                        layer=layer,
                        where="landuse IS NOT NULL",
                        bbox=union_bbox,
                        use_arrow=True,
                        INTERLEAVED_READING="YES",
                    )
                    if raw.empty:
                        continue
                    frame = self._repair_pbf_geometries(
                        self._convert_pbf_layer(raw, layer)
                    )
                    spatial_index = frame.sindex
                    for name, geometry in boxes.items():
                        indices = spatial_index.query(geometry, predicate="intersects")
                        if len(indices) == 0:
                            continue
                        fragment = fragment_root / (
                            f"{self.get_cache_key(region_bounds[name])}.{layer}.parquet"
                        )
                        frame.iloc[indices].to_parquet(fragment, index=False)
            finally:
                pyogrio.set_gdal_config_options({"OSM_CONFIG_FILE": old_config})

            for name, output in pending.items():
                key = self.get_cache_key(region_bounds[name])
                fragments = sorted(fragment_root.glob(f"{key}.*.parquet"))
                if not fragments:
                    raise OsmCacheValidationError(
                        f"{name}: 区域内没有 OSM landuse 要素"
                    )
                pieces = [gpd.read_parquet(path) for path in fragments]
                combined = gpd.GeoDataFrame(
                    pd.concat(pieces, ignore_index=True),
                    geometry="geometry",
                    crs="EPSG:4326",
                )
                self._write_frame(output, self._normalise_response(combined))
        return {name: str(path) for name, path in outputs.items()}

    def promote_snapshot_payload(
        self,
        region_bounds: Any,
        cache_dir: str | os.PathLike,
        payload: gpd.GeoDataFrame,
        snapshot_source: dict | None = None,
        *,
        snapshot_geometry_transform: dict | None = None,
        _fault_hook=None,
    ) -> str:
        """把国家级 PBF 的区域子集写入普通 OSM 缓存。"""

        output = self.get_cache_path(region_bounds, cache_dir)
        return self._write_frame(output, self._normalise_response(payload))
