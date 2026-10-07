"""GeoParquet grid bundles governed by the B+ spatial-resolution contract."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS, Geod
from scipy.spatial import cKDTree

from .fetchers.cache_paths import _atomic_write, canonical_json, replace_file
from .grid_generator import GridDesign, regenerate_grid_reference

GRID_STORAGE_CRS = "EPSG:4326"
GRID_GENERATION_CRS = "EPSG:3857"


class GridBundleError(RuntimeError):
    pass


def grid_paths(region: str, grid_dir: str | os.PathLike) -> tuple[Path, Path]:
    root = Path(grid_dir).resolve() / "bundles" / region
    return root / "grid_points.parquet", root / "grid_metadata.json"


def _columns(source_key: str, storage_columns: Sequence[str] | None) -> tuple[str, ...]:
    requested = tuple(storage_columns or ("index_region", source_key, "geometry"))
    required = {"index_region", source_key, "geometry"}
    if not required.issubset(requested) or len(requested) != len(set(requested)):
        raise GridBundleError(f"grid columns must uniquely contain {sorted(required)}")
    return requested


def _normalise_frame(
    frame: gpd.GeoDataFrame,
    *,
    source_key: str,
    storage_columns: Sequence[str] | None = None,
) -> gpd.GeoDataFrame:
    if not isinstance(frame, gpd.GeoDataFrame) or frame.empty:
        raise GridBundleError("grid must be a non-empty GeoDataFrame")
    if frame.crs is None or not CRS.from_user_input(frame.crs).equals(CRS.from_epsg(4326)):
        raise GridBundleError("grid storage CRS must be EPSG:4326")
    columns = _columns(source_key, storage_columns)
    missing = set(columns) - set(frame.columns)
    if missing:
        raise GridBundleError(f"grid is missing columns: {sorted(missing)}")
    result = frame.loc[:, columns].copy().reset_index(drop=True)
    result["index_region"] = pd.to_numeric(result["index_region"], errors="raise").astype(np.int64)
    if result[source_key].isna().any():
        raise GridBundleError(f"grid source key {source_key} contains nulls")
    result[source_key] = result[source_key].astype("string")
    result = gpd.GeoDataFrame(result, geometry="geometry", crs=GRID_STORAGE_CRS)
    if result.geometry.isna().any() or result.geometry.is_empty.any():
        raise GridBundleError("grid geometry contains empty values")
    if not result.geom_type.eq("Point").all():
        raise GridBundleError("grid geometry must contain only points")
    return result


def _ground_spacing(frame: gpd.GeoDataFrame, generation_crs: str) -> dict[str, float]:
    projected = frame.to_crs(generation_crs)
    coordinates = np.column_stack([projected.geometry.x, projected.geometry.y])
    if len(coordinates) < 2:
        return {"min": 0.0, "median": 0.0, "max": 0.0}
    sample_index = np.linspace(0, len(coordinates) - 1, min(len(coordinates), 2048), dtype=int)
    _, neighbour = cKDTree(coordinates).query(coordinates[sample_index], k=2)
    neighbour_index = neighbour[:, 1]
    left = frame.geometry.iloc[sample_index]
    right = frame.geometry.iloc[neighbour_index]
    geod = Geod(ellps="WGS84")
    _, _, distance = geod.inv(left.x.to_numpy(), left.y.to_numpy(), right.x.to_numpy(), right.y.to_numpy())
    return {
        "min": float(np.min(distance)),
        "median": float(np.median(distance)),
        "max": float(np.max(distance)),
    }


def _metadata(
    frame: gpd.GeoDataFrame,
    design: GridDesign,
    *,
    source_key: str,
    extra: Mapping[str, object] | None = None,
) -> dict:
    source_order = frame[source_key].drop_duplicates().astype(str).tolist()
    document = {
        "schema_version": "sg_grid_bundle_v2",
        "columns": list(frame.columns),
        "grid_policy": design.grid_policy,
        "target_points": design.target_points,
        "area_m2": design.area_m2,
        "area_crs": design.area_crs,
        "generation_crs": design.generation_crs,
        "storage_crs": design.storage_crs,
        "working_crs": design.generation_crs,
        "target_ground_step_m": design.target_ground_step_m,
        "projected_step_m": design.projected_step_m,
        "scale_factor_method": design.scale_factor_method,
        "reference_latitude": design.reference_latitude,
        "clamp_branch": design.clamp_branch,
        "n_cells": int(len(frame)),
        "n_sources": int(len(source_order)),
        "source_key": source_key,
        "source_key_order": source_order,
        "cells_per_source": float(len(frame) / len(source_order)),
        "ground_neighbour_distance_m": _ground_spacing(frame, design.generation_crs),
    }
    if extra:
        overlap = set(document) & set(extra)
        if overlap:
            raise GridBundleError(f"metadata extras may not replace contract keys: {sorted(overlap)}")
        document.update(extra)
    return document


def write_grid_bundle(
    region: str,
    grid_dir: str | os.PathLike,
    frame: gpd.GeoDataFrame,
    design: GridDesign,
    *,
    source_key: str,
    storage_columns: Sequence[str] | None = None,
    metadata_extra: Mapping[str, object] | None = None,
) -> tuple[Path, Path]:
    artifact, metadata_path = grid_paths(region, grid_dir)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    normalised = _normalise_frame(frame, source_key=source_key, storage_columns=storage_columns)
    metadata = _metadata(normalised, design, source_key=source_key, extra=metadata_extra)
    partial = artifact.with_name(f".{artifact.name}.part")
    partial.unlink(missing_ok=True)
    normalised.to_parquet(partial, index=False)
    replace_file(partial, artifact)
    _atomic_write(metadata_path, canonical_json(metadata))
    return artifact, metadata_path


def load_grid_bundle(region: str, grid_dir: str | os.PathLike) -> tuple[gpd.GeoDataFrame, float, dict]:
    artifact, metadata_path = grid_paths(region, grid_dir)
    if not artifact.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"{region}: grid bundle does not exist")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != "sg_grid_bundle_v2":
        raise GridBundleError(f"{region}: obsolete grid bundle schema")
    frame = _normalise_frame(
        gpd.read_parquet(artifact),
        source_key=str(metadata["source_key"]),
        storage_columns=metadata["columns"],
    )
    required = {
        "grid_policy",
        "target_ground_step_m",
        "projected_step_m",
        "scale_factor_method",
        "reference_latitude",
        "clamp_branch",
        "area_crs",
        "working_crs",
        "ground_neighbour_distance_m",
    }
    missing = required - set(metadata)
    if missing:
        raise GridBundleError(f"{region}: grid metadata missing {sorted(missing)}")
    if int(metadata["n_cells"]) != len(frame):
        raise GridBundleError(f"{region}: grid n_cells mismatch")
    observed_order = frame[str(metadata["source_key"])].drop_duplicates().astype(str).tolist()
    if observed_order != list(metadata["source_key_order"]):
        raise GridBundleError(f"{region}: source order mismatch")
    return frame, float(metadata["target_ground_step_m"]), metadata


__all__ = [
    "GRID_GENERATION_CRS",
    "GRID_STORAGE_CRS",
    "GridBundleError",
    "grid_paths",
    "load_grid_bundle",
    "regenerate_grid_reference",
    "write_grid_bundle",
]
