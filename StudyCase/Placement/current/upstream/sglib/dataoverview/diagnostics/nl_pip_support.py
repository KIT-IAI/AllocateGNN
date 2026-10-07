"""NL PiP spatial-support counterfactual.

This module deliberately does not call any DataOverview runner.  It reads the
already-materialised NL B+/audit/authority inputs, replaces the spatial support
key of ``conflict`` equipment with its unique point-in-polygon (PiP) buurt, and
replays the pure partition/grid/admission calculations into a diagnostic-only
root.  The declared code is retained as lineage.  Nothing produced here is an
admission decision or formal authority.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import tomllib
from typing import Any, Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from sglib.core.algorithms.grid_adjacency import build_grid_adjacency_indices
from sglib.core.infra.artifacts import atomic_json
from sglib.core.infra.hashing import sha256_file, sha256_json
from sglib.dataoverview.processing.derive.common import atomic_geofile
from sglib.dataoverview.processing.derive.nl.pipeline import (
    _chunked_proximity,
    deterministic_analysis_regions,
)
from sglib.dataoverview.processing.features.grid_bundle import write_grid_bundle
from sglib.dataoverview.processing.features.grid_generator import regenerate_grid_reference


SCHEMA_VERSION = "sg_nl_pip_support_counterfactual_v1"
DIAGNOSTIC_RELATIVE_ROOT = Path("results/_diagnostic/004_nl_pip_support")


def diagnostic_root(repo_root: Path | str) -> Path:
    """Return the only permitted output root for a real diagnostic run."""

    return Path(repo_root).resolve() / DIAGNOSTIC_RELATIVE_ROOT


def _assert_output_root(repo_root: Path, output_root: Path) -> None:
    expected = diagnostic_root(repo_root)
    if output_root.resolve() != expected:
        raise ValueError(f"NL PiP diagnostic output must be exactly {expected}")


def _read_admission(repo_root: Path) -> tuple[Path, dict[str, Any]]:
    path = repo_root / "casestudy/1_DataOverview/4_NL/admission.toml"
    with path.open("rb") as handle:
        document = tomllib.load(handle)
    if document.get("schema_version") != "sg_nl_engineering_admission_proposal_v1":
        raise ValueError("unsupported NL engineering-admission proposal")
    return path, document


def _atomic_csv(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.part")
    partial.unlink(missing_ok=True)
    frame.to_csv(partial, index=False)
    os.replace(partial, path)
    return path


def _atomic_geoparquet(frame: gpd.GeoDataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.part")
    partial.unlink(missing_ok=True)
    frame.to_parquet(partial, index=False)
    os.replace(partial, path)
    return path


def _files_snapshot(repo_root: Path) -> dict[str, dict[str, Any]]:
    """Hash the formal surface that this diagnostic promises not to mutate."""

    derived = repo_root / "data/datasets/2_derived/nl"
    files: list[Path] = []
    for name in ("bplus", "audit", "authority"):
        files.extend(path for path in (derived / name).rglob("*") if path.is_file())
    files.extend(
        [
            repo_root / "casestudy/1_DataOverview/4_NL/admission.toml",
            repo_root / "casestudy/1_DataOverview/4_NL/nl.toml",
        ]
    )
    return {
        path.relative_to(repo_root).as_posix(): {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(files)
    }


def counterfactual_equipment(
    equipment: gpd.GeoDataFrame,
    crosswalk: pd.DataFrame,
) -> gpd.GeoDataFrame:
    """Apply the diagnostic support-key rule while retaining formal lineage."""

    if equipment["equipment_id"].duplicated().any() or crosswalk["equipment_id"].duplicated().any():
        raise ValueError("equipment_id must be unique in equipment and crosswalk")
    audit_columns = [
        "equipment_id",
        "declared_buurt_code",
        "pip_buurt_codes",
        "assigned_buurt_code",
        "crosswalk_class",
    ]
    missing = set(audit_columns) - set(crosswalk)
    if missing:
        raise ValueError(f"crosswalk columns missing: {sorted(missing)}")
    audit = crosswalk[audit_columns].rename(columns={"crosswalk_class": "audit_crosswalk_class"})
    result = equipment.merge(audit, on="equipment_id", how="left", validate="one_to_one")
    if result["audit_crosswalk_class"].isna().any():
        raise ValueError("crosswalk does not cover every equipment row")
    if not result["crosswalk_class"].astype(str).eq(result["audit_crosswalk_class"].astype(str)).all():
        raise ValueError("equipment and crosswalk classes disagree")

    result["formal_buurt_code"] = result["buurt_code"].astype("string")
    result["formal_analysis_region"] = result["analysis_region"].astype("string")
    result["declared_buurt_code_lineage"] = result["declared_buurt_code"].astype("string")
    result["pip_buurt_code"] = result["pip_buurt_codes"].astype("string")

    conflict = result["crosswalk_class"].eq("conflict")
    invalid_pip = conflict & (
        result["pip_buurt_code"].eq("")
        | result["pip_buurt_code"].str.contains("|", regex=False, na=False)
    )
    if invalid_pip.any():
        raise ValueError("counterfactual requires one unique PiP buurt for every conflict row")
    if not result.loc[conflict, "formal_buurt_code"].astype(str).eq(
        result.loc[conflict, "declared_buurt_code_lineage"].astype(str)
    ).all():
        raise ValueError("formal conflict rows no longer follow declared-code priority")

    outside = result["crosswalk_class"].eq("outside_all_polygons")
    support = result["formal_buurt_code"].copy()
    support.loc[conflict] = result.loc[conflict, "pip_buurt_code"]
    support.loc[outside] = pd.NA
    result["counterfactual_buurt_code"] = support.astype("string")
    result["buurt_code"] = result["counterfactual_buurt_code"]
    result["generator_eligible"] = (~outside & result["buurt_code"].notna()).astype(bool)
    result["analysis_region"] = pd.Series(pd.NA, index=result.index, dtype="string")
    result["support_policy"] = np.where(
        conflict,
        "unique_pip_for_conflict",
        np.where(outside, "outside_excluded", "formal_support_unchanged"),
    )
    return gpd.GeoDataFrame(result, geometry="geometry", crs=equipment.crs)


def _normalise_polygon_reference(polygons: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    code = "buurt_code" if "buurt_code" in polygons else "buurtcode"
    if code not in polygons or polygons.crs is None:
        raise ValueError("polygon reference lacks buurt code or CRS")
    result = polygons.rename(columns={code: "buurt_code"})[["buurt_code", "geometry"]].copy()
    result["buurt_code"] = result["buurt_code"].astype("string").str.strip()
    result = result.loc[result["buurt_code"].str.startswith("BU", na=False)].copy()
    if result["buurt_code"].duplicated().any():
        raise ValueError("polygon reference buurt codes must be unique")
    return result.to_crs("EPSG:4326").reset_index(drop=True)


def build_counterfactual_sources(
    formal_sources: gpd.GeoDataFrame,
    equipment: gpd.GeoDataFrame,
    polygon_reference: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Build the feature-free source table for counterfactual partitioning."""

    eligible = equipment.loc[equipment["generator_eligible"].astype(bool)].copy()
    if eligible.empty:
        raise ValueError("counterfactual has no eligible equipment")
    codes = sorted(eligible["buurt_code"].astype(str).unique())
    formal_geometry = formal_sources[["buurt_code", "geometry"]].copy().to_crs("EPSG:4326")
    formal_geometry["buurt_code"] = formal_geometry["buurt_code"].astype("string")
    formal_geometry = formal_geometry.loc[formal_geometry["buurt_code"].isin(codes)].copy()
    formal_geometry["support_geometry_origin"] = "formal_bplus_source_regions"

    missing_codes = sorted(set(codes) - set(formal_geometry["buurt_code"].astype(str)))
    reference = _normalise_polygon_reference(polygon_reference)
    supplemental = reference.loc[reference["buurt_code"].isin(missing_codes)].copy()
    supplemental["support_geometry_origin"] = "raw_pdok_reference_for_new_pip_support"
    if set(missing_codes) != set(supplemental["buurt_code"].astype(str)):
        absent = sorted(set(missing_codes) - set(supplemental["buurt_code"].astype(str)))
        raise ValueError(f"PiP source geometries unavailable: {absent[:10]}")

    source = gpd.GeoDataFrame(
        pd.concat([formal_geometry, supplemental], ignore_index=True),
        geometry="geometry",
        crs="EPSG:4326",
    )
    source = source.sort_values("buurt_code", kind="stable").reset_index(drop=True)
    if len(source) != len(codes) or source["buurt_code"].duplicated().any():
        raise RuntimeError("counterfactual source geometry is not one-to-one")

    counts = (
        eligible.groupby(["buurt_code", "operational_stratum"], observed=True)
        .size()
        .rename("count")
        .reset_index()
        .sort_values(
            ["buurt_code", "count", "operational_stratum"],
            ascending=[True, False, True],
            kind="stable",
        )
        .drop_duplicates("buurt_code")
    )
    demand = eligible.groupby("buurt_code", observed=True)["peak_kw"].sum()
    source = source.merge(
        counts[["buurt_code", "operational_stratum"]],
        on="buurt_code",
        how="left",
        validate="one_to_one",
    )
    source["demand_peak_kw"] = source["buurt_code"].astype(str).map(demand).astype(float)
    source["feature_scope"] = "geometry_and_demand_only"
    if source[["operational_stratum", "demand_peak_kw"]].isna().any().any():
        raise RuntimeError("counterfactual source aggregation is incomplete")
    return source


def boundary_mismatch(
    equipment: gpd.GeoDataFrame,
    polygon_reference: gpd.GeoDataFrame,
    *,
    support_key: str,
) -> pd.Series:
    """Return point-outside-assigned-support for eligible equipment."""

    reference = _normalise_polygon_reference(polygon_reference)
    geometry_by_code = reference.set_index("buurt_code").geometry
    eligible = equipment["generator_eligible"].astype(bool)
    equipment_work = equipment.to_crs(reference.crs)
    support_geometry = equipment.loc[eligible, support_key].astype(str).map(geometry_by_code)
    if support_geometry.isna().any():
        missing = sorted(set(equipment.loc[eligible, support_key].astype(str)) - set(geometry_by_code.index.astype(str)))
        raise ValueError(f"support geometry missing for boundary check: {missing[:10]}")
    polygons = gpd.GeoSeries(support_geometry, index=support_geometry.index, crs=reference.crs)
    result = pd.Series(False, index=equipment.index, dtype=bool)
    result.loc[eligible] = ~equipment_work.loc[eligible].geometry.within(polygons)
    return result


def _failed_check_counts(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        for name, value in record["checks"].items():
            if not bool(value):
                counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))


def _evaluate_regions(
    *,
    sources: gpd.GeoDataFrame,
    equipment: gpd.GeoDataFrame,
    region_records: list[dict[str, Any]],
    contract: dict[str, Any],
    output_root: Path,
    crs: dict[str, str],
    grid_config: dict[str, Any],
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    limits = contract["limits"]
    graph = contract["graph_estimate"]
    records_by_id = {str(item["id"]): item for item in region_records}
    reports: list[dict[str, Any]] = []
    occupancy_rows: list[dict[str, Any]] = []
    for region in sorted(records_by_id):
        authority = records_by_id[region]
        source = sources.loc[sources["analysis_region"].eq(region)].copy()
        targets = equipment.loc[
            equipment["generator_eligible"].astype(bool) & equipment["analysis_region"].eq(region)
        ].copy()
        if source.empty or targets.empty:
            raise RuntimeError(f"{region}: empty source or target table")
        grid, design = regenerate_grid_reference(
            source,
            target_points=int(limits["target_grid_cells"]),
            min_ground_step_m=float(grid_config["min_ground_step_m"]),
            max_ground_step_m=float(grid_config["max_ground_step_m"]),
            area_crs=crs["area"],
            generation_crs=grid_config["generation_crs"],
        )
        order = {code: index for index, code in enumerate(authority["source_key_order"])}
        grid["_order"] = grid["buurt_code"].astype(str).map(order)
        if grid["_order"].isna().any():
            raise RuntimeError(f"{region}: grid contains an unregistered source")
        grid = grid.sort_values("_order", kind="stable").drop(columns="_order").reset_index(drop=True)
        grid["analysis_region"] = region

        grid_work = grid.to_crs(crs["working"])
        target_work = targets.to_crs(crs["working"])
        grid_xy = np.column_stack([grid_work.geometry.x, grid_work.geometry.y])
        target_xy = np.column_stack([target_work.geometry.x, target_work.geometry.y])
        tree = cKDTree(target_xy)
        nearest_distance, nearest = tree.query(grid_xy, k=1)
        if len(targets) >= 2:
            two_distance, _ = tree.query(grid_xy, k=2)
            tie_count = int(
                np.isclose(two_distance[:, 0], two_distance[:, 1], rtol=0.0, atol=1e-8).sum()
            )
        else:
            tie_count = 0
        target_counts = np.bincount(nearest.astype(np.int64), minlength=len(targets))
        adjacency = build_grid_adjacency_indices(
            grid_xy,
            mode="neumann",
            step_size=float(design.projected_step_m),
        )
        agent_edges = int(adjacency.shape[1])
        source_agent_edges = int(graph["source_agent_edges_per_cell"]) * len(grid)
        total_edges = agent_edges + source_agent_edges
        estimated_ram = (
            (len(grid) + len(source)) * int(graph["bytes_per_node_conservative"])
            + total_edges * int(graph["bytes_per_edge_conservative"])
        )
        proximity_ok, proximity_peak = _chunked_proximity(
            grid_xy,
            target_xy,
            gamma=2.0,
            clamp_m=10.0,
        )
        cdist_bytes = len(grid) * len(targets) * np.dtype(np.float64).itemsize
        dense_bytes = len(grid) * len(source) * 5 * np.dtype(np.float32).itemsize
        checks = {
            "cells_limit": len(grid) <= int(limits["max_cells"]),
            "active_agent_limit": len(grid) <= int(limits["max_active_agents"]),
            "source_limit": len(source) <= int(limits["max_sources"]),
            "target_limit": len(targets) <= int(limits["max_targets"]),
            "planning_k_covered": len(targets) <= int(limits["planning_k_cap"]),
            "canonical_target_occupancy": bool((target_counts > 0).all()),
            "canonical_nearest_tie_free": tie_count == 0,
            "directed_source_agent_edge_limit": source_agent_edges
            <= int(limits["max_directed_source_agent_edges"]),
            "dense_intermediate_limit": dense_bytes <= int(limits["max_dense_intermediate_bytes"]),
            "cdist_intermediate_limit": cdist_bytes <= int(limits["max_cdist_intermediate_bytes"]),
            "graph_ram_limit": estimated_ram <= int(limits["max_estimated_graph_ram_bytes"]),
            "proximity_finite_positive": proximity_ok,
            "idr_realization_limit": 41 * 3 <= int(limits["max_idr_realizations_per_region"]),
        }
        violated = sorted(name for name, value in checks.items() if not value)
        artifact, metadata = write_grid_bundle(
            region,
            output_root / "grid_bplus",
            grid,
            design,
            source_key="buurt_code",
            storage_columns=["index_region", "buurt_code", "analysis_region", "geometry"],
            metadata_extra={
                "authority_scope": "diagnostic_only_non_authoritative",
                "support_policy": "unique_pip_for_conflict",
                "formal_admission_claim": False,
            },
        )
        reports.append(
            {
                "region": region,
                "diagnostic_disposition": (
                    "WITHIN_PROPOSED_BOUNDS" if not violated else "VIOLATES_PROPOSED_BOUNDS"
                ),
                "violated_checks": violated,
                "checks": checks,
                "n_cells": len(grid),
                "n_sources": len(source),
                "n_targets": len(targets),
                "cells_per_source": len(grid) / len(source),
                "cells_per_target": len(grid) / len(targets),
                "canonical_empty_targets": int((target_counts == 0).sum()),
                "canonical_nearest_tie_count": tie_count,
                "canonical_nearest_distance_max_m": float(nearest_distance.max()),
                "directed_agent_edges": agent_edges,
                "directed_source_agent_edges": source_agent_edges,
                "directed_graph_edges": total_edges,
                "estimated_graph_ram_bytes": estimated_ram,
                "estimated_dense_intermediate_bytes": dense_bytes,
                "estimated_full_cdist_bytes": cdist_bytes,
                "proximity_chunk_peak_bytes": proximity_peak,
                "working_crs": crs["working"],
                "grid_artifact": artifact.relative_to(output_root).as_posix(),
                "grid_metadata": metadata.relative_to(output_root).as_posix(),
            }
        )
        for index, (_, target) in enumerate(targets.iterrows()):
            occupancy_rows.append(
                {
                    "equipment_id": str(target["equipment_id"]),
                    "analysis_region": region,
                    "declared_buurt_code_lineage": str(target["declared_buurt_code_lineage"]),
                    "formal_buurt_code": str(target["formal_buurt_code"]),
                    "counterfactual_buurt_code": str(target["counterfactual_buurt_code"]),
                    "crosswalk_class": str(target["crosswalk_class"]),
                    "canonical_vd_cells": int(target_counts[index]),
                    "canonical_empty": bool(target_counts[index] == 0),
                }
            )
    return reports, pd.DataFrame(occupancy_rows)


def _engineering_summary(
    records: list[dict[str, Any]],
    *,
    boundary_mismatch_devices: int,
) -> dict[str, Any]:
    failed = _failed_check_counts(records)
    return {
        "region_count": len(records),
        "source_count": int(sum(int(item["n_sources"]) for item in records)),
        "target_count": int(sum(int(item["n_targets"]) for item in records)),
        "grid_cell_count": int(sum(int(item["n_cells"]) for item in records)),
        "canonical_empty_targets": int(
            sum(int(item["canonical_empty_targets"]) for item in records)
        ),
        "boundary_mismatch_devices": int(boundary_mismatch_devices),
        "regions_violating_any_proposed_check": int(
            sum(any(not bool(value) for value in item["checks"].values()) for item in records)
        ),
        "failure_counts_by_check": failed,
        "dense_limit_failure_regions": int(failed.get("dense_intermediate_limit", 0)),
        "cdist_limit_failure_regions": int(failed.get("cdist_intermediate_limit", 0)),
        "edge_limit_failure_regions": int(failed.get("directed_source_agent_edge_limit", 0)),
        "graph_ram_limit_failure_regions": int(failed.get("graph_ram_limit", 0)),
        "directed_graph_edges_total": int(
            sum(int(item["directed_graph_edges"]) for item in records)
        ),
        "directed_graph_edges_max_region": int(
            max(int(item["directed_graph_edges"]) for item in records)
        ),
        "estimated_dense_intermediate_bytes_max_region": int(
            max(int(item["estimated_dense_intermediate_bytes"]) for item in records)
        ),
        "estimated_full_cdist_bytes_max_region": int(
            max(int(item["estimated_full_cdist_bytes"]) for item in records)
        ),
    }


def build_comparison(
    original_records: list[dict[str, Any]],
    counterfactual_records: list[dict[str, Any]],
    *,
    original_boundary_mismatch: int,
    counterfactual_boundary_mismatch: int,
) -> dict[str, Any]:
    original = _engineering_summary(
        original_records,
        boundary_mismatch_devices=original_boundary_mismatch,
    )
    counterfactual = _engineering_summary(
        counterfactual_records,
        boundary_mismatch_devices=counterfactual_boundary_mismatch,
    )
    delta_keys = (
        "region_count",
        "source_count",
        "target_count",
        "grid_cell_count",
        "canonical_empty_targets",
        "boundary_mismatch_devices",
        "dense_limit_failure_regions",
        "cdist_limit_failure_regions",
        "edge_limit_failure_regions",
        "graph_ram_limit_failure_regions",
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": "diagnostic_only_non_authoritative",
        "decision": "NONE",
        "formal_admission_claim": False,
        "counterfactual": (
            "conflict equipment uses its unique PiP buurt as spatial support; "
            "declared exact code remains lineage; outside-all remains excluded"
        ),
        "original": original,
        "counterfactual_result": counterfactual,
        "counterfactual_minus_original": {
            key: int(counterfactual[key]) - int(original[key]) for key in delta_keys
        },
    }


def _write_manifest(
    *,
    repo_root: Path,
    output_root: Path,
    formal_snapshot: dict[str, dict[str, Any]],
    supplemental_inputs: list[Path],
    contract: dict[str, Any],
) -> Path:
    artifacts = []
    for path in sorted(item for item in output_root.rglob("*") if item.is_file()):
        if path.name == "manifest.json" or path.name.startswith("."):
            continue
        artifacts.append(
            {
                "path": path.relative_to(output_root).as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    document = {
        "schema_version": f"{SCHEMA_VERSION}_manifest",
        "scope": "diagnostic_only_non_authoritative",
        "decision": "NONE",
        "formal_admission_claim": False,
        "output_root": DIAGNOSTIC_RELATIVE_ROOT.as_posix(),
        "formal_inputs": formal_snapshot,
        "supplemental_read_only_inputs": {
            path.relative_to(repo_root).as_posix(): {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in supplemental_inputs
        },
        "algorithm_sources": {
            path.relative_to(repo_root).as_posix(): sha256_file(path)
            for path in (
                repo_root / "sglib/dataoverview/processing/derive/nl/pipeline.py",
                repo_root / "sglib/dataoverview/processing/features/grid_generator.py",
                repo_root / "sglib/core/algorithms/grid_adjacency.py",
                Path(__file__).resolve(),
            )
        },
        "proposed_contract": contract,
        "artifacts": artifacts,
    }
    document["manifest_payload_sha256"] = sha256_json(document)
    return atomic_json(document, output_root / "manifest.json")


def run(repo_root: Path | str) -> Path:
    """Execute the diagnostic and return its manifest path."""

    repo = Path(repo_root).resolve()
    output = diagnostic_root(repo)
    _assert_output_root(repo, output)
    formal_before = _files_snapshot(repo)
    admission_path, contract = _read_admission(repo)

    derived = repo / "data/datasets/2_derived/nl"
    raw_polygons_path = repo / "data/datasets/1_raw/nl/pdok_buurten_2025.geojson"
    equipment = gpd.read_file(derived / "bplus/equipment_register.gpkg", layer="equipment_register")
    formal_sources = gpd.read_file(derived / "bplus/source_regions.gpkg", layer="source_regions")
    crosswalk = pd.read_csv(
        derived / "audit/crosswalk.csv", dtype="string", keep_default_na=False
    )
    original_engineering = json.loads(
        (derived / "audit/engineering_admission.json").read_text(encoding="utf-8")
    )
    polygon_reference = gpd.read_file(raw_polygons_path)

    counter_equipment = counterfactual_equipment(equipment, crosswalk)
    counter_sources = build_counterfactual_sources(
        formal_sources,
        counter_equipment,
        polygon_reference,
    )
    original_boundary = boundary_mismatch(
        equipment,
        polygon_reference,
        support_key="buurt_code",
    )
    counter_boundary = boundary_mismatch(
        counter_equipment,
        polygon_reference,
        support_key="counterfactual_buurt_code",
    )
    counter_equipment["formal_boundary_mismatch"] = original_boundary.to_numpy(bool)
    counter_equipment["counterfactual_boundary_mismatch"] = counter_boundary.to_numpy(bool)

    limits = contract["limits"]
    counter_sources, eligible_partitioned, region_records = deterministic_analysis_regions(
        counter_sources,
        counter_equipment.loc[counter_equipment["generator_eligible"].astype(bool)].copy(),
        working_crs="EPSG:28992",
        limits=limits,
    )
    region_by_equipment = eligible_partitioned.set_index("equipment_id")["analysis_region"]
    counter_equipment["analysis_region"] = counter_equipment["equipment_id"].map(region_by_equipment).astype("string")
    analysis_regions = (
        counter_sources[["analysis_region", "geometry"]]
        .dissolve(by="analysis_region")
        .reset_index()
    )

    output.mkdir(parents=True, exist_ok=True)
    atomic_geofile(
        counter_sources,
        output / "counterfactual_source_regions.gpkg",
        layer="source_regions_diagnostic",
    )
    atomic_geofile(
        analysis_regions,
        output / "counterfactual_analysis_regions.gpkg",
        layer="analysis_regions_diagnostic",
    )
    equipment_columns = [
        "equipment_id",
        "station_id",
        "peak_kw",
        "operational_stratum",
        "crosswalk_class",
        "declared_buurt_code_lineage",
        "pip_buurt_code",
        "formal_buurt_code",
        "counterfactual_buurt_code",
        "formal_analysis_region",
        "analysis_region",
        "generator_eligible",
        "support_policy",
        "formal_boundary_mismatch",
        "counterfactual_boundary_mismatch",
        "geometry",
    ]
    _atomic_geoparquet(
        counter_equipment[equipment_columns],
        output / "counterfactual_equipment_lineage.parquet",
    )
    partition_document = {
        "schema_version": f"{SCHEMA_VERSION}_partition",
        "scope": "diagnostic_only_non_authoritative",
        "formal_admission_claim": False,
        "working_crs": "EPSG:28992",
        "partition_algorithm": "formal deterministic_analysis_regions function",
        "partition_algorithm_sha256": sha256_file(
            repo / "sglib/dataoverview/processing/derive/nl/pipeline.py"
        ),
        "proposed_contract": admission_path.relative_to(repo).as_posix(),
        "proposed_contract_sha256": sha256_file(admission_path),
        "n_regions": len(region_records),
        "regions": region_records,
    }
    atomic_json(partition_document, output / "counterfactual_partition_inventory.json")

    reports, occupancy = _evaluate_regions(
        sources=counter_sources,
        equipment=counter_equipment,
        region_records=region_records,
        contract=contract,
        output_root=output,
        crs={"working": "EPSG:28992", "area": "EPSG:28992"},
        grid_config={
            "min_ground_step_m": 100.0,
            "max_ground_step_m": 500.0,
            "generation_crs": "EPSG:3857",
        },
    )
    metric_rows = []
    for report in reports:
        flat = {key: value for key, value in report.items() if key != "checks"}
        flat["violated_checks"] = "|".join(report["violated_checks"])
        for name, value in report["checks"].items():
            flat[f"check__{name}"] = bool(value)
        metric_rows.append(flat)
    _atomic_csv(pd.DataFrame(metric_rows), output / "counterfactual_region_metrics.csv")
    _atomic_csv(occupancy, output / "counterfactual_target_occupancy.csv")

    comparison = build_comparison(
        original_engineering["regions"],
        reports,
        original_boundary_mismatch=int(original_boundary.sum()),
        counterfactual_boundary_mismatch=int(counter_boundary.sum()),
    )
    comparison["lineage_counts"] = {
        "equipment_rows": len(counter_equipment),
        "eligible_equipment_rows": int(counter_equipment["generator_eligible"].sum()),
        "outside_excluded_rows": int(
            counter_equipment["crosswalk_class"].eq("outside_all_polygons").sum()
        ),
        "conflict_rows_switched_to_pip": int(
            counter_equipment["crosswalk_class"].eq("conflict").sum()
        ),
        "eligible_analysis_region_membership_changed": int(
            (
                counter_equipment.loc[
                    counter_equipment["generator_eligible"].astype(bool),
                    "formal_analysis_region",
                ].astype(str)
                != counter_equipment.loc[
                    counter_equipment["generator_eligible"].astype(bool),
                    "analysis_region",
                ].astype(str)
            ).sum()
        ),
        "conflict_analysis_region_membership_changed": int(
            (
                counter_equipment.loc[
                    counter_equipment["crosswalk_class"].eq("conflict"),
                    "formal_analysis_region",
                ].astype(str)
                != counter_equipment.loc[
                    counter_equipment["crosswalk_class"].eq("conflict"),
                    "analysis_region",
                ].astype(str)
            ).sum()
        ),
        "source_support_keys_added": len(
            set(counter_sources["buurt_code"].astype(str))
            - set(formal_sources["buurt_code"].astype(str))
        ),
        "source_support_keys_removed": len(
            set(formal_sources["buurt_code"].astype(str))
            - set(counter_sources["buurt_code"].astype(str))
        ),
        "counterfactual_source_geometries_from_formal_bplus": int(
            counter_sources["support_geometry_origin"].eq("formal_bplus_source_regions").sum()
        ),
        "counterfactual_source_geometries_from_raw_pdok_reference": int(
            counter_sources["support_geometry_origin"].eq(
                "raw_pdok_reference_for_new_pip_support"
            ).sum()
        ),
    }
    atomic_json(comparison, output / "original_vs_counterfactual.json")

    formal_after = _files_snapshot(repo)
    if formal_after != formal_before:
        raise RuntimeError("formal NL data/config/authority changed during diagnostic")
    manifest = _write_manifest(
        repo_root=repo,
        output_root=output,
        formal_snapshot=formal_before,
        supplemental_inputs=[raw_polygons_path],
        contract=contract,
    )
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="repository root; output remains fixed under results/_diagnostic",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = run(args.repo_root)
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SCHEMA_VERSION",
    "boundary_mismatch",
    "build_comparison",
    "build_counterfactual_sources",
    "counterfactual_equipment",
    "diagnostic_root",
    "run",
]
