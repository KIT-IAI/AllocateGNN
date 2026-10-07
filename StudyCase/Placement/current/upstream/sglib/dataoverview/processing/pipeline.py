"""Country-neutral B+ grid and feature processing."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import shutil

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

from sglib.core.infra.artifacts import atomic_json
from sglib.core.infra.hashing import sha256_file, sha256_json
from sglib.core.infra.paths import portable_path
from sglib.dataoverview.engineering_admission import load_engineering_admission

from .config import CountryPipelineContext
from .features import extractor_registry, fetcher_registry
from .features.cuz_support import (
    OSM_LANDUSE_COLUMNS,
    atomic_savez,
    build_cuz_support,
    save_cuz_support,
    validate_cuz_grid_crs,
)
from .features.grid_bundle import GridBundleError, grid_paths, load_grid_bundle, write_grid_bundle
from .features.grid_generator import regenerate_grid_reference


def _source_regions(context: CountryPipelineContext) -> gpd.GeoDataFrame:
    regions = context.merged["regions"]
    path = context.path(str(regions["source_path"]), label="regions.source_path")
    if not path.is_file():
        raise FileNotFoundError(f"{context.country_code}: derive regions first: {path}")
    layer = regions.get("source_layer")
    return gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)


def selected_regions(context: CountryPipelineContext) -> dict[str, gpd.GeoDataFrame]:
    source = _source_regions(context)
    group_key = context.merged["regions"].get("group_key")
    output: dict[str, gpd.GeoDataFrame] = {}
    for record in context.region_items:
        name = str(record["id"])
        selected = source.loc[source[str(group_key)].astype("string") == name].copy() if group_key else source.copy()
        if selected.empty:
            raise ValueError(f"{context.country_code}/{name}: source region is empty")
        output[name] = selected
    return output


def _bounds(context: CountryPipelineContext, region: gpd.GeoDataFrame, *, union: bool):
    working_crs = str(context.merged["crs"]["working"])
    buffer_m = float(context.merged["grid"]["query_buffer_m"])
    projected = region.to_crs(working_crs)
    geometry = projected.geometry.union_all().buffer(buffer_m) if union else box(*projected.total_bounds).buffer(buffer_m)
    return gpd.GeoSeries([geometry], crs=working_crs).to_crs("EPSG:4326").iloc[0]


def _load_grid(context: CountryPipelineContext, region: str) -> tuple[gpd.GeoDataFrame, float, dict]:
    frame, _, metadata = load_grid_bundle(region, context.grid_root)
    return frame, float(metadata["projected_step_m"]), metadata


def _engineering_admission_metadata(context: CountryPipelineContext) -> dict[str, Any]:
    """Return disclosure-only B+ metadata when a country has a v2 contract."""

    directory = str(context.merged["country"]["directory"])
    country_stage = (
        context.repo_root
        / "casestudy"
        / "1_DataOverview"
        / directory
    )
    contracts = sorted(country_stage.glob("admission*.toml"))
    if not contracts:
        return {}
    if len(contracts) != 1:
        raise RuntimeError(
            f"{context.country_code}: expected one engineering admission contract, "
            f"found {[path.name for path in contracts]}"
        )
    contract = contracts[0]
    admission = load_engineering_admission(context.repo_root, contract)
    return {
        "engineering_admission": {
            "authority_schema_version": str(admission.authority["schema_version"]),
            "authority_sha256": admission.authority_sha256,
            "min_cells_per_source_hard": int(admission.grid["min_cells_per_source_hard"]),
            "min_cells_per_target_hard": int(admission.grid["min_cells_per_target_hard"]),
            "cells_per_source_oversampling_disclosure": int(
                admission.grid["cells_per_source_oversampling_disclosure"]
            ),
            "cells_per_target_oversampling_disclosure": int(
                admission.grid["cells_per_target_oversampling_disclosure"]
            ),
            "oversampling_binding": "report_only",
            "dense_dry_run_binding": str(admission.memory["dense_dry_run_binding"]),
            "dense_formal_binding": str(admission.memory["dense_formal_binding"]),
        }
    }


def run_grid(context: CountryPipelineContext, *, force: bool = False) -> dict[str, Any]:
    grid_config = context.merged["grid"]
    summary: dict[str, Any] = {}
    diagnostic = context.repo_root / "results" / "_diagnostic" / "legacy_dataoverview" / context.country_code / "grid"
    legacy_layout = context.derived_root / "features" / "grid"
    legacy_diagnostic = context.repo_root / "results" / "_diagnostic" / "legacy_dataoverview" / context.country_code / "grid_legacy_layout"
    if legacy_layout.is_dir() and not legacy_diagnostic.exists():
        shutil.copytree(legacy_layout, legacy_diagnostic)
    if context.grid_root.is_dir() and not diagnostic.exists():
        shutil.copytree(context.grid_root, diagnostic)
    item_by_name = {str(item["id"]): item for item in context.region_items}
    admission_metadata = _engineering_admission_metadata(context)
    for name, polygons in selected_regions(context).items():
        if not force:
            try:
                frame, _, metadata = _load_grid(context, name)
            except (FileNotFoundError, GridBundleError, ValueError):
                pass
            else:
                summary[name] = {"n_cells": len(frame), "reused": True, **metadata}
                continue
        frame, design = regenerate_grid_reference(
            polygons,
            target_points=int(grid_config["target_points"]),
            min_ground_step_m=float(grid_config["min_ground_step_m"]),
            max_ground_step_m=float(grid_config["max_ground_step_m"]),
            area_crs=str(context.merged["crs"]["area"]),
            generation_crs=str(grid_config["generation_crs"]),
        )
        source_key = str(context.merged["regions"]["source_key"])
        source_order = list(map(str, item_by_name[name]["source_key_order"]))
        rank = {key: index for index, key in enumerate(source_order)}
        frame["_source_order"] = frame[source_key].astype(str).map(rank)
        if frame["_source_order"].isna().any():
            unknown = sorted(set(frame.loc[frame["_source_order"].isna(), source_key].astype(str)))
            raise RuntimeError(f"{context.country_code}/{name}: unregistered source keys {unknown}")
        frame = frame.sort_values("_source_order", kind="stable").drop(columns="_source_order").reset_index(drop=True)
        write_grid_bundle(
            name,
            context.grid_root,
            frame,
            design,
            source_key=source_key,
            storage_columns=context.merged["grid_columns"]["names"],
            metadata_extra={
                "query_buffer_m": float(grid_config["query_buffer_m"]),
                "query_buffer_unit": str(grid_config["query_buffer_unit"]),
                **admission_metadata,
            },
        )
        _, _, metadata = _load_grid(context, name)
        summary[name] = {"n_cells": len(frame), "reused": False, **metadata}
    return summary


def _osm_snapshot(context: CountryPipelineContext) -> Path:
    specs = context.merged["datasets"]["osm_snapshot"]["files"]
    if len(specs) != 1:
        raise ValueError("osm_snapshot must contain exactly one file")
    path = (context.raw_root / str(specs[0]["filename"])).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"OSM snapshot is missing: {path}")
    return path


def _fetch_feature_cache(context: CountryPipelineContext, regions: dict[str, gpd.GeoDataFrame], *, force: bool) -> None:
    config = context.runtime_feature_config()
    osm = fetcher_registry.get_fetcher("osm")(config["fetchers"]["osm"])
    ghsl = fetcher_registry.get_fetcher("ghsl_built_s")(config["fetchers"]["ghsl_built_s"])
    pbf = _osm_snapshot(context)
    osm_bounds = {name: _bounds(context, polygons, union=False) for name, polygons in regions.items()}
    if force:
        for bounds in osm_bounds.values():
            osm.get_cache_path(bounds, context.cache_root).unlink(missing_ok=True)
    osm.fetch_many_from_pbf(osm_bounds, context.cache_root, pbf)
    for polygons in regions.values():
        bounds = _bounds(context, polygons, union=True)
        if force:
            ghsl.get_cache_path(bounds, str(context.cache_root)).unlink(missing_ok=True)
        ghsl.fetch(bounds, str(context.cache_root))


def _save_extractor_result(path: Path, result) -> None:
    columns = tuple(result.numerical_columns)
    data = np.column_stack([result.numerical_columns[name] for name in columns]).astype(np.float64, copy=False)
    atomic_savez(path, data=np.ascontiguousarray(data), columns=np.asarray(columns, dtype=np.str_))


def _artifact(context: CountryPipelineContext, region: str, kind: str) -> Path:
    suffix = {
        "landuse": "landuse",
        "built_surface": "ghsl_built_s",
        "cuz_support": "cuz_support",
        "ntl": "ntl",
    }[kind]
    return context.artifact_root / f"{region}_{suffix}.npz"


def _extract_cuz(
    context: CountryPipelineContext,
    regions: dict[str, gpd.GeoDataFrame],
    *,
    force: bool,
) -> dict[str, Any]:
    config = context.runtime_feature_config()
    validate_cuz_grid_crs(
        config["grid"]["generation_crs"],
        config["extractors"]["landuse"]["target_grid_crs"],
        config["extractors"]["ghsl_built_s"]["target_grid_crs"],
    )
    osm = fetcher_registry.get_fetcher("osm")(config["fetchers"]["osm"])
    ghsl = fetcher_registry.get_fetcher("ghsl_built_s")(config["fetchers"]["ghsl_built_s"])
    landuse = extractor_registry.get_extractor("landuse")(config["extractors"]["landuse"])
    built = extractor_registry.get_extractor("ghsl_built_s")(config["extractors"]["ghsl_built_s"])
    source_key = str(context.merged["regions"]["source_key"])
    summary: dict[str, Any] = {}
    context.artifact_root.mkdir(parents=True, exist_ok=True)
    for name, polygons in regions.items():
        grid, projected_step, _ = _load_grid(context, name)
        landuse_result = landuse.extract(grid, projected_step, osm.validate_cache(_bounds(context, polygons, union=False), context.cache_root))
        built_result = built.extract(grid, projected_step, ghsl.validate_cache(_bounds(context, polygons, union=True), str(context.cache_root)))
        landuse_path = _artifact(context, name, "landuse")
        built_path = _artifact(context, name, "built_surface")
        support_path = _artifact(context, name, "cuz_support")
        if force or not landuse_path.exists():
            _save_extractor_result(landuse_path, landuse_result)
        if force or not built_path.exists():
            _save_extractor_result(built_path, built_result)
        landuse_matrix = np.column_stack([landuse_result.numerical_columns[column] for column in OSM_LANDUSE_COLUMNS]).astype(np.float64, copy=False)
        built_fraction = np.asarray(built_result.numerical_columns["ghsl_built_fraction"], dtype=np.float64)
        support = build_cuz_support(landuse_matrix, built_fraction, grid[source_key].values)
        save_cuz_support(support_path, support, overwrite=force)
        zero_nonzero = int(np.count_nonzero(support.features[support.zero_mask]))
        if zero_nonzero:
            raise RuntimeError(f"{context.country_code}/{name}: Z cells are not exact zero")
        summary[name] = {
            "n_cells": len(grid),
            "covered": int(support.covered_mask.sum()),
            "unknown": int(support.unknown_mask.sum()),
            "zero": int(support.zero_mask.sum()),
            "zero_nonzero_values": zero_nonzero,
        }
    return summary


def _extract_ntl(context: CountryPipelineContext, regions: dict[str, gpd.GeoDataFrame], *, force: bool) -> dict[str, Any]:
    config = context.runtime_feature_config()
    ntl_path = Path(fetcher_registry.get_fetcher("ntl")(config["fetchers"]["ntl"]).fetch())
    extractor = extractor_registry.get_extractor("ntl")(config["extractors"]["ntl"])
    summary: dict[str, Any] = {}
    for name in regions:
        grid, projected_step, _ = _load_grid(context, name)
        path = _artifact(context, name, "ntl")
        if force or not path.exists():
            _save_extractor_result(path, extractor.extract(grid, projected_step, str(ntl_path)))
        summary[name] = {"n_cells": len(grid), "path": portable_path(path, context.repo_root), "sha256": sha256_file(path)}
    return summary


def _region_features_current(context: CountryPipelineContext, name: str) -> bool:
    try:
        grid, _, metadata = _load_grid(context, name)
        paths = [_artifact(context, name, kind) for kind in ("landuse", "built_surface", "cuz_support", "ntl")]
        if not all(path.is_file() for path in paths):
            return False
        with np.load(paths[0], allow_pickle=False) as landuse, np.load(paths[1], allow_pickle=False) as built, np.load(paths[2], allow_pickle=False) as support, np.load(paths[3], allow_pickle=False) as ntl:
            if any(archive["data"].shape[0] != len(grid) for archive in (landuse, built, ntl)):
                return False
            source_key = str(metadata["source_key"])
            return support["features"].shape[0] == len(grid) and np.array_equal(
                support["source_keys"].astype(str), grid[source_key].astype(str).to_numpy()
            )
    except (FileNotFoundError, KeyError, OSError, ValueError):
        return False


def run_features(context: CountryPipelineContext, *, force: bool = False) -> dict[str, Any]:
    receipt = context.derived_root / "features_bplus" / "features_receipt.json"
    all_regions = selected_regions(context)
    pending = all_regions if force else {
        name: polygons
        for name, polygons in all_regions.items()
        if not _region_features_current(context, name)
    }
    formal_rebuild = bool(pending)
    diagnostic = context.repo_root / "results" / "_diagnostic" / "legacy_dataoverview" / context.country_code / "features_extracted"
    if context.artifact_root.is_dir() and not diagnostic.exists() and formal_rebuild:
        shutil.copytree(context.artifact_root, diagnostic)
    regions = all_regions
    for name in regions:
        _load_grid(context, name)
    if pending:
        _fetch_feature_cache(context, pending, force=force)
        result = {
            "cuz": _extract_cuz(context, pending, force=True),
            "ntl": _extract_ntl(context, pending, force=True),
        }
    else:
        result = {"cuz": {}, "ntl": {}, "reused": sorted(regions)}
    grid_metadata = {}
    grid_row_identity = {}
    for name in regions:
        _, metadata_path = grid_paths(name, context.grid_root)
        grid_metadata[name] = sha256_file(metadata_path)
        grid, _, metadata = _load_grid(context, name)
        grid_row_identity[name] = sha256_json(grid[str(metadata["source_key"])].astype(str).tolist())
    artifacts = expected_feature_artifacts(context)
    atomic_json(
        {
            "schema_version": "sg_dataoverview_features_receipt_v1",
            "country": context.country_code,
            "grid_metadata_sha256": grid_metadata,
            "grid_row_identity_sha256": grid_row_identity,
            "artifacts": [
                {"path": path.relative_to(context.repo_root).as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size}
                for path in artifacts
            ],
        },
        receipt,
    )
    return result


def expected_feature_artifacts(context: CountryPipelineContext) -> list[Path]:
    return [
        _artifact(context, str(item["id"]), kind)
        for item in context.region_items
        for kind in ("landuse", "built_surface", "cuz_support", "ntl")
    ]
