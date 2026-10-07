"""NZ core-9 adapter and pre-HPC engineering gates.

The handoff tables are rebuilt only from official fresh landings.  A rebuild
may drift from earlier research materialisations; downstream stages consume
the rebuild, so it is checked for internal consistency and contract semantics
rather than against frozen counts or frozen research tables.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from sglib.core.infra.artifacts import atomic_json
from sglib.core.infra.hashing import sha256_file
from sglib.dataoverview.engineering_admission import (
    EngineeringAdmission,
    chunked_nearest_assignment,
    evaluate_region,
    load_engineering_admission,
)
from sglib.dataoverview.processing.derive.common import atomic_geofile
from sglib.dataoverview.processing.features.grid_generator import regenerate_grid_reference


CORE_EDBS = ("Vector Lines", "Orion NZ", "Wellington Electricity")
DECLARED_SECURE_CLASSES = ("N-1", "N-1 switched")
REGION_ORDER = (
    "Orion_NZ__historical_spatial_1",
    "Orion_NZ__historical_spatial_2",
    "Orion_NZ__historical_spatial_3",
    "Vector_Lines__historical_spatial_1",
    "Vector_Lines__historical_spatial_2",
    "Vector_Lines__historical_spatial_3",
    "Wellington_Electricity__historical_spatial_1",
    "Wellington_Electricity__historical_spatial_2",
    "Wellington_Electricity__historical_spatial_3",
)
SOURCE_FEATURES = (
    "residential_percent",
    "commercial_percent",
    "industrial_percent",
    "agricultural_percent",
    "others_percent",
)


class NZCore9Error(ValueError):
    """Raised when the NZ core-9 contract is violated."""


@dataclass(frozen=True)
class Core9Artifacts:
    ledger: Path
    sites: Path
    sources: Path
    analysis_regions: Path
    gate: Path
    size_gate: Path | None = None


def _atomic_csv(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.part")
    partial.unlink(missing_ok=True)
    try:
        pd.DataFrame(frame.drop(columns="geometry", errors="ignore")).to_csv(
            partial, index=False
        )
        os.replace(partial, path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return path


def _site_id(edb: object, name: object) -> str:
    token = re.sub(r"[^a-z0-9]+", "-", f"{edb}-{name}".casefold()).strip("-")
    if not token:
        raise NZCore9Error("site identity normalised to an empty token")
    return f"nz:{token}"


def _require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = columns - set(frame)
    if missing:
        raise NZCore9Error(f"{label} is missing columns {sorted(missing)}")


def _canonical_ledger(ledger: gpd.GeoDataFrame, source_region: dict[str, str]) -> gpd.GeoDataFrame:
    result = ledger.copy()
    if result.empty:
        raise NZCore9Error("section ledger is empty")
    if set(result["edb"].astype(str)) != set(CORE_EDBS):
        raise NZCore9Error("section ledger is not exactly the three core EDBs")
    if set(result["security_class"].astype(str)) - set(DECLARED_SECURE_CLASSES):
        raise NZCore9Error("section ledger contains a non-declared-secure class")
    if not result["disc_yr"].eq(2024).all():
        raise NZCore9Error("section ledger contains a non-2024 disclosure row")
    for column in ("actual_peak_mva", "firm_capacity_mva"):
        values = pd.to_numeric(result[column], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise NZCore9Error(f"section ledger has invalid {column}")
    for column in ("longitude", "latitude"):
        values = pd.to_numeric(result[column], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all():
            raise NZCore9Error(f"section ledger has invalid {column}")
    result = result.sort_values(
        ["edb", "matched_d5_name", "sub_category"], kind="stable"
    ).reset_index(drop=True)
    result["lineage_id"] = [f"NZ2024-SECTION-{index:03d}" for index in range(1, len(result) + 1)]
    result["station_id"] = [
        _site_id(edb, name)
        for edb, name in zip(result["edb"], result["matched_d5_name"], strict=True)
    ]
    result["cluster_analysis_region"] = result["analysis_region"].astype(str)
    result["analysis_region"] = result["SA22023_V1_00"].astype(str).map(source_region)
    if result["analysis_region"].isna().any():
        raise NZCore9Error("section ledger has a source outside the core-9 inventory")
    result["capacity_basis"] = "declared_security_class"
    result["assignment_eligible"] = False
    result["truth_role"] = "lineage_and_section_sensitivity_only"
    result["demand_vintage"] = "2024_actual_annual_peak"
    result["capacity_vintage"] = "2024_installed_firm_capacity"
    result["geometry_vintage"] = "2025_comcom_d5"
    return result


def _canonical_sites(rebuilt: gpd.GeoDataFrame, source_region: dict[str, str]) -> gpd.GeoDataFrame:
    result = rebuilt.copy().sort_values(["edb", "matched_d5_name"], kind="stable").reset_index(drop=True)
    result["station_id"] = [
        _site_id(edb, name)
        for edb, name in zip(result["edb"], result["matched_d5_name"], strict=True)
    ]
    result["cluster_analysis_region"] = result["analysis_region"].astype(str)
    result["analysis_region"] = result["SA22023_V1_00"].astype(str).map(source_region)
    if result["analysis_region"].isna().any():
        raise NZCore9Error("site table has a source outside the core-9 inventory")
    result["security_class"] = result["security_classes"].map(
        lambda value: str(value) if " | " not in str(value) else "mixed_declared_secure"
    )
    result["strict_n1_eligible"] = result["security_classes"].eq("N-1")
    result["capacity_basis"] = "declared_security_class"
    result["capacity_basis_detail"] = np.where(
        result["security_class"].eq("mixed_declared_secure"),
        "sum_sections_mixed_declared_secure_classes",
        "sum_sections_single_declared_security_class",
    )
    result["assignment_eligible"] = True
    result["truth_role"] = "formal_generator_site_truth"
    result["demand_vintage"] = "2024_actual_annual_peak"
    result["capacity_vintage"] = "2024_installed_firm_capacity"
    result["geometry_vintage"] = "2025_comcom_d5"
    if result.empty or not result["station_id"].is_unique:
        raise NZCore9Error("site table must contain unique assignment targets")
    return result


def _canonical_sources(source: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    result = source.copy()
    _require_columns(
        result,
        {"SA22023_V1_00", "source_id", "analysis_region", "demand_peak_mva", *SOURCE_FEATURES},
        "source regions",
    )
    if result.empty or not result["source_id"].astype(str).is_unique:
        raise NZCore9Error("source table must contain unique SA2 rows")
    if result.crs is None or result.geometry.isna().any() or result.geometry.is_empty.any():
        raise NZCore9Error("source table contains invalid geometry")
    if not result.geometry.is_valid.all():
        raise NZCore9Error("source table contains topologically invalid geometry")
    for column in ("demand_peak_mva", *SOURCE_FEATURES):
        values = pd.to_numeric(result[column], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise NZCore9Error(f"source table has invalid {column}")
    for column in SOURCE_FEATURES:
        totals = result.groupby("analysis_region", sort=False)[column].sum()
        if not np.allclose(totals.to_numpy(float), 1.0, rtol=1e-10, atol=1e-10):
            raise NZCore9Error(f"{column} does not sum to one within every region")
    rank = {region: index for index, region in enumerate(REGION_ORDER)}
    result["_region_rank"] = result["analysis_region"].map(rank)
    if result["_region_rank"].isna().any():
        raise NZCore9Error("source table contains an unregistered analysis region")
    result["SA22023_V1_00"] = result["SA22023_V1_00"].astype(str)
    result = result.sort_values(["_region_rank", "SA22023_V1_00"], kind="stable").drop(columns="_region_rank").reset_index(drop=True)
    result["demand_vintage"] = "2024_actual_annual_peak"
    result["feature_vintage"] = "2023_census_sa2"
    result["source_geography_vintage"] = "2023_sa2_generalised"
    return result


def _analysis_regions(
    sources: gpd.GeoDataFrame,
    sites: gpd.GeoDataFrame,
    ledger: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    result = sources.dissolve(by=["edb", "analysis_region"], as_index=False)[
        ["edb", "analysis_region", "geometry"]
    ]
    rank = {region: index for index, region in enumerate(REGION_ORDER)}
    result["_rank"] = result["analysis_region"].map(rank)
    result = result.sort_values("_rank").drop(columns="_rank").reset_index(drop=True)
    if tuple(result["analysis_region"]) != REGION_ORDER:
        raise NZCore9Error("rebuilt analysis regions differ from the registered region inventory")
    source_counts = sources.groupby("analysis_region").size()
    target_counts = sites.groupby("analysis_region").size()
    section_counts = ledger.groupby("analysis_region").size()
    strict_counts = sites.groupby("analysis_region")["strict_n1_eligible"].sum()
    result["n_source_rows"] = result["analysis_region"].map(source_counts).astype(int)
    result["n_site_targets"] = result["analysis_region"].map(target_counts).fillna(0).astype(int)
    result["n_section_rows"] = result["analysis_region"].map(section_counts).fillna(0).astype(int)
    result["n_strict_n1_sites"] = result["analysis_region"].map(strict_counts).fillna(0).astype(int)
    return result


SECTION_RECONCILIATION = "audit/core9_section_reconciliation.csv"
SOURCE_RECONCILIATION = "audit/core9_source_reconciliation.csv"
CANDIDATE_SA3_COVERAGE = "audit/core9_candidate_sa3_coverage.csv"


def _section_keys(frame: pd.DataFrame) -> list[tuple[str, str, str]]:
    return sorted(map(tuple, frame[["edb", "network", "sub_category"]].astype(str).to_numpy()))


def _reconcile(
    sections: pd.DataFrame,
    landed_sa2: pd.DataFrame,
    coverage: pd.DataFrame,
    ledger: gpd.GeoDataFrame,
    sources: gpd.GeoDataFrame,
) -> tuple[pd.DataFrame, dict[str, bool], dict[str, Any]]:
    """Reconcile the handoff with every landed raw record instead of historical counts."""

    included = sections["status"].eq("included")
    reasons = sections["exclusion_reasons"].fillna("").astype(str)
    lineage = ledger[["edb", "network", "sub_category", "lineage_id", "station_id"]].astype(str)
    sections = sections.merge(
        lineage, on=["edb", "network", "sub_category"], how="left", validate="one_to_one"
    )
    source_reason = landed_sa2["exclusion_reason"].fillna("").astype(str)
    source_included = landed_sa2["status"].eq("included")
    orphans = landed_sa2.loc[source_reason.eq("no_sa3_parent")]
    checks = {
        "sections_reconciled": bool(
            sections["status"].isin(["included", "excluded"]).all()
            and reasons.eq("").eq(included).all()
        ),
        "ledger_is_included_sections": _section_keys(sections.loc[included]) == _section_keys(ledger)
        and bool(sections.loc[included, "lineage_id"].notna().all()),
        "sources_reconciled": bool(
            landed_sa2["SA22023_V1_00"].is_unique
            and source_reason.eq("").eq(source_included).all()
            and sorted(
                zip(landed_sa2.loc[source_included, "edb"].astype(str),
                    landed_sa2.loc[source_included, "SA22023_V1_00"].astype(str))
            )
            == sorted(zip(sources["edb"].astype(str), sources["SA22023_V1_00"].astype(str)))
        ),
        "candidate_sa3_tiled_by_landed_sa2": bool(
            len(coverage)
            and set(coverage["SA32023_V1_00"].astype(str))
            == set(ledger["SA32023_V1_00"].astype(str))
            and np.isclose(coverage["coverage"].to_numpy(float), 1.0, rtol=0.0, atol=1e-6).all()
        ),
        "unparented_sa2_outside_candidate_sa3": bool(
            orphans["dominant_sa3"].notna().all() and not orphans["dominant_sa3_is_candidate"].any()
        ),
    }
    excluded_reasons = reasons[~included].str.split(";").explode()
    summary = {
        "sections": {
            "landed_rows": len(sections),
            "included": int(included.sum()),
            "excluded": int((~included).sum()),
            "failed_rules": {key: int(value) for key, value in excluded_reasons.value_counts().sort_index().items()},
            "table": SECTION_RECONCILIATION,
        },
        "sources": {
            "landed_sa2": len(landed_sa2),
            "included": int(source_included.sum()),
            "excluded": {
                key: int(value)
                for key, value in source_reason[~source_included].value_counts().sort_index().items()
            },
            "table": SOURCE_RECONCILIATION,
        },
        "candidate_sa3": {
            "count": len(coverage),
            "min_coverage": float(coverage["coverage"].min()),
            "max_coverage": float(coverage["coverage"].max()),
            "table": CANDIDATE_SA3_COVERAGE,
        },
    }
    return sections, checks, summary


def _gate_document(
    repo_root: Path,
    sources: gpd.GeoDataFrame,
    sites: gpd.GeoDataFrame,
    ledger: gpd.GeoDataFrame,
    analysis: gpd.GeoDataFrame,
    *,
    fresh_source_manifest: Path,
    d6_truth_audit: dict[str, Any] | None = None,
    reconciliation: tuple[dict[str, bool], dict[str, Any]] = ({}, {}),
) -> dict[str, Any]:
    source_total = float(sources["demand_peak_mva"].sum())
    site_total = float(sites["actual_peak_mva"].sum())
    section_total = float(ledger["actual_peak_mva"].sum())
    checks = {
        "core_edbs_exact": set(sites["edb"].astype(str)) == set(CORE_EDBS),
        "analysis_regions_registered": tuple(analysis["analysis_region"]) == REGION_ORDER,
        "site_ids_unique": bool(sites["station_id"].is_unique),
        "source_ids_unique": bool(sources["source_id"].astype(str).is_unique),
        "sites_aggregate_every_section": set(ledger["station_id"]) == set(sites["station_id"]),
        "demand_conservation": bool(np.isclose(source_total, site_total, rtol=1e-12, atol=1e-9) and np.isclose(site_total, section_total, rtol=1e-12, atol=1e-9)),
        "formal_truth_excludes_section_ledger": bool((~ledger["assignment_eligible"]).all() and sites["assignment_eligible"].all()),
        "forecast_2026_excluded": bool(ledger["disc_yr"].eq(2024).all()),
        "source_features_complete": bool(np.isfinite(sources[list(SOURCE_FEATURES)].to_numpy(float)).all()),
        **reconciliation[0],
    }
    failures = sorted(name for name, passed in checks.items() if not passed)
    if failures:
        raise NZCore9Error(f"NZ core-9 gate failed: {failures}")
    document = {
        "schema_version": "sg_nz_core9_gate_v1",
        "status": "PASS",
        "country": "nz",
        "scope": "core9",
        "truth_granularity": "site",
        "formal_truth_rows": len(sites),
        "section_lineage_rows": len(ledger),
        "source_rows": len(sources),
        "analysis_regions": len(analysis),
        "strict_n1_robustness_rows": int(sites["strict_n1_eligible"].sum()),
        "noncoincident_aggregate_rows": int(sites["noncoincident_conservative"].sum()),
        "demand_totals_mva": {"source": source_total, "site": site_total, "section": section_total},
        "capacity_basis": "declared_security_class",
        "mixed_security_site_rule": "sum section capacities; label mixed_declared_secure; exclude from strict N-1 robustness",
        "noncoincident_rule": "sum section annual peaks and set noncoincident_conservative=true",
        "temporal_protocol": "2024 actual peak + 2024 Installed Firm Capacity + 2025 geospatial + 2023 Census/SA2",
        "forecast_2026_role": "prototype_only_excluded_from_truth",
        "feature_layer": "2023 Census real; OSM/GHSL/NTL pending formal HPC build",
        "formal_hpc_completion": False,
        "checks": checks,
        "source_mode": "fresh_official_downloads",
        "fresh_source_manifest": {
            "path": fresh_source_manifest.relative_to(repo_root).as_posix(),
            "sha256": sha256_file(fresh_source_manifest),
            "bytes": fresh_source_manifest.stat().st_size,
        },
        "d6_truth_2024": d6_truth_audit or {},
        "reconciliation": reconciliation[1],
    }
    return document


def build_core9_dataoverview_from_fresh(
    repo_root: Path | str,
    output_root: Path | str,
    *,
    force: bool = False,
) -> Core9Artifacts:
    """Build formal NZ core-9 products exclusively from official fresh landings.

    A verified fresh-source manifest is mandatory.  Earlier research
    materialisations are neither read nor compared: downstream stages consume
    this rebuild as it is.
    """

    from .fresh_sources import (
        FreshSourcePaths,
        derive_core9_frames,
        verify_fresh_manifest,
    )

    repository = Path(repo_root).resolve()
    destination = Path(output_root).resolve()
    ledger_path = destination / "lineage" / "station_ledger_2024.gpkg"
    sites_path = destination / "bplus" / "stations.gpkg"
    sources_path = destination / "bplus" / "regions.gpkg"
    analysis_path = destination / "bplus" / "analysis_regions.gpkg"
    gate_path = destination / "audit" / "nz_gate.json"
    truth_audit_path = destination / "audit" / "d6_truth_2024.json"
    dropped_path = destination / "audit" / "dropped_suppressed_zero_demand_sa2.csv"
    expected = (ledger_path, sites_path, sources_path, analysis_path, gate_path)
    if not force and all(path.is_file() and path.stat().st_size > 0 for path in expected):
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        if gate.get("status") == "PASS" and gate.get("source_mode") == "fresh_official_downloads":
            return Core9Artifacts(ledger_path, sites_path, sources_path, analysis_path, gate_path)

    fresh_paths = FreshSourcePaths.from_repo(repository)
    verify_fresh_manifest(fresh_paths)
    frames = derive_core9_frames(fresh_paths)
    fresh_sources = _canonical_sources(frames.sources)
    source_region = dict(
        zip(
            fresh_sources["SA22023_V1_00"].astype(str),
            fresh_sources["analysis_region"].astype(str),
            strict=True,
        )
    )
    ledger = _canonical_ledger(frames.ledger, source_region)
    sites = _canonical_sites(frames.sites, source_region)
    analysis = _analysis_regions(fresh_sources, sites, ledger)
    sections, reconciliation_checks, reconciliation_summary = _reconcile(
        frames.section_reconciliation,
        frames.source_reconciliation,
        frames.candidate_sa3_coverage,
        ledger,
        fresh_sources,
    )
    gate = _gate_document(
        repository,
        fresh_sources,
        sites,
        ledger,
        analysis,
        fresh_source_manifest=fresh_paths.manifest,
        d6_truth_audit=frames.truth_audit,
        reconciliation=(reconciliation_checks, reconciliation_summary),
    )

    atomic_geofile(ledger, ledger_path, layer="station_ledger_2024")
    _atomic_csv(ledger, ledger_path.with_suffix(".csv"))
    atomic_geofile(sites, sites_path, layer="station_sites")
    _atomic_csv(sites, sites_path.with_suffix(".csv"))
    atomic_geofile(fresh_sources, sources_path, layer="source_regions")
    _atomic_csv(fresh_sources, sources_path.with_suffix(".csv"))
    atomic_geofile(analysis, analysis_path, layer="analysis_regions")
    _atomic_csv(analysis, analysis_path.with_suffix(".csv"))
    _atomic_csv(frames.dropped_suppressed, dropped_path)
    _atomic_csv(sections, destination / SECTION_RECONCILIATION)
    _atomic_csv(frames.source_reconciliation, destination / SOURCE_RECONCILIATION)
    _atomic_csv(frames.candidate_sa3_coverage, destination / CANDIDATE_SA3_COVERAGE)
    atomic_json(frames.truth_audit, truth_audit_path)
    atomic_json(gate, gate_path)
    return Core9Artifacts(ledger_path, sites_path, sources_path, analysis_path, gate_path)


def _edge_count(grid: gpd.GeoDataFrame, working_crs: str) -> int:
    projected = grid.to_crs(working_crs)
    coordinates = np.column_stack([projected.geometry.x, projected.geometry.y])
    if len(coordinates) < 2:
        return 0
    tree = cKDTree(coordinates)
    nearest, _ = tree.query(coordinates, k=2)
    radius = float(np.median(nearest[:, 1]) * 1.5)
    return int(len(tree.query_pairs(radius)) * 2)


def _bounded_target_assignment(
    grid: gpd.GeoDataFrame,
    targets: gpd.GeoDataFrame,
    admission: EngineeringAdmission,
) -> tuple[np.ndarray, int, int]:
    working_crs = str(admission.country["working_crs"])
    grid_projected = grid.to_crs(working_crs)
    target_projected = targets.to_crs(working_crs)
    return chunked_nearest_assignment(
        np.column_stack([grid_projected.geometry.x, grid_projected.geometry.y]),
        np.column_stack([target_projected.geometry.x, target_projected.geometry.y]),
        max_workspace_mib=float(admission.memory["max_cdist_workspace_mib"]),
    )


def _region_admission_record(
    source: gpd.GeoDataFrame,
    target: gpd.GeoDataFrame,
    *,
    region: str,
    admission: EngineeringAdmission,
) -> dict[str, Any]:
    if source.empty or target.empty:
        raise NZCore9Error(f"{region}: admission candidate has an empty node class")
    grid_contract = admission.grid
    working_crs = str(admission.country["working_crs"])
    generation_crs = str(admission.country["generation_crs"])
    grid, design = regenerate_grid_reference(
        source,
        target_points=int(grid_contract["target_points"]),
        min_ground_step_m=float(grid_contract["min_ground_step_m"]),
        max_ground_step_m=float(grid_contract["max_ground_step_m"]),
        area_crs=working_crs,
        generation_crs=generation_crs,
    )
    assignment, cdist_peak_bytes, cdist_chunk_rows = _bounded_target_assignment(
        grid, target, admission
    )
    target_counts = np.bincount(assignment, minlength=len(target))
    source_key = source["SA22023_V1_00"].astype(str)
    source_counts = (
        grid.assign(_source_key=grid["SA22023_V1_00"].astype(str))
        .groupby("_source_key")
        .size()
        .reindex(source_key, fill_value=0)
    )
    agent_edges = _edge_count(grid, working_crs)
    source_agent_edges = 2 * len(grid)
    total_edges = int(agent_edges + source_agent_edges)
    full_cdist_bytes = len(grid) * len(target) * np.dtype(np.float64).itemsize
    metrics = {
        "n_sources": len(source),
        "n_targets": len(target),
        "n_cells": len(grid),
        "min_cells_per_source": int(source_counts.min()),
        "min_cells_per_target": int(target_counts.min()),
        "total_edges_directed": total_edges,
        "active_agent_nodes_upper_bound": len(grid),
        "cdist_peak_workspace_bytes": cdist_peak_bytes,
    }
    decision = evaluate_region(admission, metrics)
    return {
        "region": region,
        "working_crs": working_crs,
        "generation_crs": generation_crs,
        **{key: int(metrics[key]) for key in ("n_sources", "n_targets", "n_cells")},
        "min_cells_per_source": int(metrics["min_cells_per_source"]),
        "min_cells_per_target": int(metrics["min_cells_per_target"]),
        "empty_targets": int((target_counts == 0).sum()),
        "source_agent_edges_directed": source_agent_edges,
        "agent_adjacency_edges_directed": agent_edges,
        "total_edges_directed": total_edges,
        "active_agent_nodes_upper_bound": len(grid),
        "dense_tensor_upper_bound_mib": float(decision["memory"]["dense_tensor_mib"]),
        "cdist_full_matrix_mib": full_cdist_bytes / 2**20,
        "cdist_peak_workspace_mib": cdist_peak_bytes / 2**20,
        "cdist_chunk_rows": cdist_chunk_rows,
        "cdist_strategy": str(admission.memory["cdist_strategy"]),
        "target_ground_step_m": design.target_ground_step_m,
        "projected_step_m": design.projected_step_m,
        "clamp_branch": design.clamp_branch,
        "hard_checks": decision["hard_checks"],
        "oversampling_disclosure": decision["oversampling_disclosure"],
        "memory_advisory": decision["memory"],
        "status": "PASS" if decision["hard_pass"] else "FAIL",
    }


def _candidate_regions_per_edb(
    sources: gpd.GeoDataFrame,
    sites: gpd.GeoDataFrame,
    *,
    groups_per_edb: int,
    admission: EngineeringAdmission,
) -> list[tuple[str, gpd.GeoDataFrame, gpd.GeoDataFrame]]:
    """Build k=1/2 deterministic minimality candidates without changing core-9."""

    if groups_per_edb not in {1, 2}:
        raise NZCore9Error("minimality evidence only evaluates k=1 and k=2 per EDB")
    working_crs = str(admission.country["working_crs"])
    candidates: list[tuple[str, gpd.GeoDataFrame, gpd.GeoDataFrame]] = []
    for edb in sorted(sources["edb"].astype(str).unique()):
        edb_sources = sources.loc[sources["edb"].astype(str).eq(edb)].copy()
        edb_sites = (
            sites.loc[sites["edb"].astype(str).eq(edb)]
            .drop_duplicates("station_id")
            .sort_values("station_id", kind="stable")
            .to_crs(working_crs)
        )
        coordinates = np.column_stack([edb_sites.geometry.x, edb_sites.geometry.y])
        spread = np.ptp(coordinates, axis=0)
        axis = int(np.argmax(spread))
        other = 1 - axis
        ordered = np.lexsort(
            (
                edb_sites["station_id"].astype(str).to_numpy(),
                coordinates[:, other],
                coordinates[:, axis],
            )
        )
        groups = [np.asarray(group) for group in np.array_split(ordered, groups_per_edb)]
        centres = np.asarray([coordinates[group].mean(axis=0) for group in groups])
        centre_order = np.lexsort((centres[:, 0], -centres[:, 1]))
        centres = centres[centre_order]
        source_points = edb_sources.to_crs(working_crs).representative_point()
        labels, _, _ = chunked_nearest_assignment(
            np.column_stack([source_points.x, source_points.y]),
            centres,
            max_workspace_mib=float(admission.memory["max_cdist_workspace_mib"]),
        )
        for ordinal in range(groups_per_edb):
            source = edb_sources.iloc[np.flatnonzero(labels == ordinal)].copy()
            source_keys = set(source["SA22023_V1_00"].astype(str))
            target = sites.loc[
                sites["edb"].astype(str).eq(edb)
                & sites["SA22023_V1_00"].astype(str).isin(source_keys)
            ].copy()
            region = (
                f"{edb.replace(' ', '_')}__minimality_k{groups_per_edb}_"
                f"{ordinal + 1}"
            )
            candidates.append((region, source.reset_index(drop=True), target.reset_index(drop=True)))
    return candidates


def build_size_gate(
    repo_root: Path | str,
    output_root: Path | str,
    *,
    force: bool = False,
) -> Path:
    """Run the NZ lightweight grid/static/graph-size admission dry-run."""

    repository = Path(repo_root).resolve()
    destination = Path(output_root).resolve()
    gate_path = destination / "audit" / "nz_size_gate.json"
    admission = load_engineering_admission(
        repository,
        "casestudy/1_DataOverview/5_NZ/admission_contract.toml",
    )
    if gate_path.is_file() and not force:
        document = json.loads(gate_path.read_text(encoding="utf-8"))
        recorded = document.get("admission_contract", {})
        shared = document.get("shared_authority", {})
        if (
            document.get("schema_version") == "sg_nz_size_gate_evidence_v2"
            and recorded.get("sha256") == admission.country_sha256
            and shared.get("sha256") == admission.authority_sha256
        ):
            return gate_path
    artifacts = build_core9_dataoverview_from_fresh(repository, destination, force=force)
    sources = gpd.read_file(artifacts.sources, layer="source_regions")
    sites = gpd.read_file(artifacts.sites, layer="station_sites")
    records: list[dict[str, Any]] = []
    for region in REGION_ORDER:
        source = sources.loc[sources["analysis_region"].eq(region)].copy().reset_index(drop=True)
        source_keys = set(source["SA22023_V1_00"].astype(str))
        target = sites.loc[sites["SA22023_V1_00"].astype(str).isin(source_keys)].copy().reset_index(drop=True)
        records.append(
            _region_admission_record(
                source,
                target,
                region=region,
                admission=admission,
            )
        )

    smaller_candidates: list[dict[str, Any]] = []
    for groups_per_edb in (1, 2):
        candidate_records = [
            _region_admission_record(
                source,
                target,
                region=region,
                admission=admission,
            )
            for region, source, target in _candidate_regions_per_edb(
                sources,
                sites,
                groups_per_edb=groups_per_edb,
                admission=admission,
            )
        ]
        candidate_failures = [
            {
                "region": record["region"],
                "failed_hard_checks": sorted(
                    key for key, value in record["hard_checks"].items() if not value
                ),
            }
            for record in candidate_records
            if record["status"] != "PASS"
        ]
        smaller_candidates.append(
            {
                "regions_per_edb": groups_per_edb,
                "total_regions": len(candidate_records),
                "status": "PASS" if not candidate_failures else "FAIL",
                "failures": candidate_failures,
                "regions": candidate_records,
            }
        )

    contract = admission.country
    realizations = (
        int(contract["idr_matched"]["formal_candidates"])
        * int(contract["idr_matched"]["seeds"])
        * len(REGION_ORDER)
    )
    idr_pass = realizations <= int(contract["idr_matched"]["max_realizations_all_regions"])
    failures = [record["region"] for record in records if record["status"] != "PASS"]
    if not idr_pass:
        failures.append("idr_matched_realizations")
    smaller_partition_passed = any(
        candidate["status"] == "PASS" for candidate in smaller_candidates
    )
    if smaller_partition_passed:
        failures.append("smaller_partition_requires_authority_review")
    admitted = not failures
    core_gate = json.loads(artifacts.gate.read_text(encoding="utf-8"))
    document = {
        "schema_version": "sg_nz_size_gate_evidence_v2",
        "status": "PASS" if admitted else "REVIEW_REQUIRED",
        "admission_decision": "ADMITTED" if admitted else "NOT_MADE",
        "authority_status": "FROZEN",
        "country": "nz",
        "scope": "core9",
        "profile": "lightweight_grid_static_graph_size_dry_run",
        "formal_feature_layers_read": False,
        "region_authority": "frozen core-9 research materialisation; no model/evaluation outcomes read",
        "admission_contract": {
            "path": admission.country_path.relative_to(repository).as_posix(),
            "sha256": admission.country_sha256,
            "schema_version": str(contract["schema_version"]),
        },
        "shared_authority": {
            "path": admission.authority_path.relative_to(repository).as_posix(),
            "sha256": admission.authority_sha256,
            "schema_version": str(admission.authority["schema_version"]),
        },
        "formal_core9_source_mode": core_gate["source_mode"],
        "regions": records,
        "minimum_count_evidence": {
            "algorithm": str(contract["region_algorithm"]),
            "rule": str(contract["region_count_rule"]),
            "smaller_candidates": smaller_candidates,
            "all_smaller_candidates_fail": not smaller_partition_passed,
            "selected_regions_per_edb": 3,
            "selected_total_regions": len(REGION_ORDER),
            "selected_status": "PASS" if all(record["status"] == "PASS" for record in records) else "FAIL",
            "core9_preserved": True,
        },
        "idr_matched": {
            "formal_candidates": int(contract["idr_matched"]["formal_candidates"]),
            "seeds": int(contract["idr_matched"]["seeds"]),
            "regions": len(REGION_ORDER),
            "realizations": realizations,
            "max_realizations_all_regions": int(contract["idr_matched"]["max_realizations_all_regions"]),
            "pass": idr_pass,
        },
        "hard_gate_failures": failures,
        "limitations": [
            "dense-tensor values are conservative advisory upper bounds because OSM/GHSL formal features were not read",
            "formal training submission must fail closed against observed active cells and the 32 MiB dense-tensor authority",
            "worker RSS cannot be established by a geometry-only dry-run",
        ],
    }
    atomic_json(document, gate_path)
    return gate_path


__all__ = [
    "CORE_EDBS",
    "Core9Artifacts",
    "DECLARED_SECURE_CLASSES",
    "NZCore9Error",
    "REGION_ORDER",
    "SOURCE_FEATURES",
    "build_core9_dataoverview_from_fresh",
    "build_size_gate",
]
