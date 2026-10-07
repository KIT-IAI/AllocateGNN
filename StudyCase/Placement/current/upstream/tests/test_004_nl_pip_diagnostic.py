from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, box

from sglib.dataoverview.diagnostics.nl_pip_support import (
    boundary_mismatch,
    build_comparison,
    build_counterfactual_sources,
    counterfactual_equipment,
    diagnostic_root,
)

import pytest
pytestmark = pytest.mark.produce


def _equipment() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "equipment_id": ["conflict", "exact", "outside"],
            "station_id": ["s1", "s2", "s3"],
            "peak_kw": [5.0, 6.0, 7.0],
            "operational_stratum": ["A", "A", "A"],
            "buurt_code": ["BU_OLD", "BU_EXACT", pd.NA],
            "crosswalk_class": ["conflict", "exact_code", "outside_all_polygons"],
            "analysis_region": ["OLD_R", "OLD_R", pd.NA],
            "generator_eligible": [True, True, False],
        },
        geometry=[Point(2.5, 0.5), Point(1.5, 0.5), Point(9.0, 9.0)],
        crs="EPSG:4326",
    )


def _crosswalk() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "equipment_id": ["conflict", "exact", "outside"],
            "declared_buurt_code": ["BU_OLD", "BU_EXACT", "NO_CODE"],
            "pip_buurt_codes": ["BU_PIP", "BU_EXACT", ""],
            "assigned_buurt_code": ["BU_OLD", "BU_EXACT", ""],
            "crosswalk_class": ["conflict", "exact_code", "outside_all_polygons"],
        }
    )


def _polygons() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"buurt_code": ["BU_OLD", "BU_EXACT", "BU_PIP"]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1), box(2, 0, 3, 1)],
        crs="EPSG:4326",
    )


def test_conflict_switches_to_unique_pip_and_outside_stays_excluded() -> None:
    result = counterfactual_equipment(_equipment(), _crosswalk()).set_index("equipment_id")
    assert result.at["conflict", "buurt_code"] == "BU_PIP"
    assert result.at["conflict", "declared_buurt_code_lineage"] == "BU_OLD"
    assert result.at["conflict", "formal_analysis_region"] == "OLD_R"
    assert result.at["exact", "buurt_code"] == "BU_EXACT"
    assert pd.isna(result.at["outside", "buurt_code"])
    assert not bool(result.at["outside", "generator_eligible"])


def test_boundary_mismatch_changes_from_declared_to_pip() -> None:
    formal = _equipment()
    counter = counterfactual_equipment(formal, _crosswalk())
    before = boundary_mismatch(formal, _polygons(), support_key="buurt_code")
    after = boundary_mismatch(counter, _polygons(), support_key="counterfactual_buurt_code")
    assert before.tolist() == [True, False, False]
    assert after.tolist() == [False, False, False]


def test_new_pip_source_geometry_is_supplemented_without_features() -> None:
    counter = counterfactual_equipment(_equipment(), _crosswalk())
    formal_sources = gpd.GeoDataFrame(
        {"buurt_code": ["BU_EXACT"]},
        geometry=[box(1, 0, 2, 1)],
        crs="EPSG:4326",
    )
    result = build_counterfactual_sources(formal_sources, counter, _polygons()).set_index(
        "buurt_code"
    )
    assert set(result.index) == {"BU_EXACT", "BU_PIP"}
    assert result.at["BU_EXACT", "support_geometry_origin"] == "formal_bplus_source_regions"
    assert (
        result.at["BU_PIP", "support_geometry_origin"]
        == "raw_pdok_reference_for_new_pip_support"
    )
    assert result.at["BU_PIP", "demand_peak_kw"] == 5.0
    assert result["feature_scope"].eq("geometry_and_demand_only").all()


def test_comparison_quantifies_requested_failure_classes_without_pass_claim() -> None:
    base = {
        "n_sources": 2,
        "n_targets": 2,
        "n_cells": 10,
        "canonical_empty_targets": 1,
        "directed_graph_edges": 30,
        "estimated_dense_intermediate_bytes": 100,
        "estimated_full_cdist_bytes": 80,
        "checks": {
            "canonical_target_occupancy": False,
            "dense_intermediate_limit": False,
            "cdist_intermediate_limit": False,
            "directed_source_agent_edge_limit": True,
            "graph_ram_limit": True,
        },
    }
    changed = {
        **base,
        "canonical_empty_targets": 0,
        "checks": {**base["checks"], "canonical_target_occupancy": True},
    }
    comparison = build_comparison(
        [base],
        [changed],
        original_boundary_mismatch=1,
        counterfactual_boundary_mismatch=0,
    )
    assert comparison["decision"] == "NONE"
    assert comparison["formal_admission_claim"] is False
    assert comparison["original"]["dense_limit_failure_regions"] == 1
    assert comparison["original"]["cdist_limit_failure_regions"] == 1
    assert comparison["original"]["edge_limit_failure_regions"] == 0
    assert comparison["counterfactual_minus_original"]["canonical_empty_targets"] == -1
    assert comparison["counterfactual_minus_original"]["boundary_mismatch_devices"] == -1


def test_real_output_root_is_fixed_to_diagnostic_tree() -> None:
    root = Path("repo").resolve()
    assert diagnostic_root(root) == root / "results/_diagnostic/004_nl_pip_support"
