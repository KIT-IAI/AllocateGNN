"""Read-only, stage-internal DataOverview loaders.

Downstream stages must use :mod:`sglib.dataoverview.handoff` instead.
"""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Mapping

import geopandas as gpd
import pandas as pd


class DataOverviewReadError(ValueError):
    pass


@lru_cache(maxsize=128)
def _read(path_text: str, mtime_ns: int, size: int, layer: str | None) -> Any:
    path = Path(path_text)
    suffix = path.suffix.lower()
    if suffix in {".gpkg", ".geojson", ".shp"}:
        return gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".geoparquet"}:
        return gpd.read_parquet(path)
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    raise DataOverviewReadError(f"unsupported DataOverview artifact: {path}")


def read_artifact(path: Path | str, *, layer: str | None = None, copy: bool = True):
    source = Path(path).resolve()
    if not source.is_file() or source.stat().st_size == 0:
        raise FileNotFoundError(f"DataOverview artifact is missing or empty: {source}")
    stat = source.stat()
    value = _read(str(source), stat.st_mtime_ns, stat.st_size, layer)
    if not copy:
        return value
    return value.copy() if hasattr(value, "copy") else json.loads(json.dumps(value))


def canonical_paths(
    repo_root: Path | str,
    country: str,
    *,
    configuration: Mapping[str, Any],
) -> dict[str, Path]:
    root = Path(repo_root).resolve()
    configured = configuration.get("canonical")
    if not isinstance(configured, Mapping) or set(configured) != {"regions", "stations"}:
        raise DataOverviewReadError(
            f"{country}: canonical configuration must declare regions and stations"
        )
    result: dict[str, Path] = {}
    for name in ("regions", "stations"):
        path = (root / str(configured[name])).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise DataOverviewReadError(
                f"{country}: canonical {name} path escapes repository"
            ) from exc
        result[name] = path
    return result
