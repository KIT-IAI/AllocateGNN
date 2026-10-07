from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, box

from sglib.dataoverview.processing.derive.nl.pipeline import (
    aggregate_buurt_targets,
    merge_low_equipment_strata,
)
from sglib.dataoverview.engineering_admission import load_engineering_admission

pytestmark = pytest.mark.produce


REPO_ROOT = Path(__file__).resolve().parents[1]


def _lineage() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "equipment_id": ["a", "b", "c"],
            "buurt_code": ["BU00010000", "BU00010000", "BU00010100"],
            "wijk_code": ["WK000100", "WK000100", "WK000101"],
            "peak_kw": [3.0, 5.0, 0.0],
            "capacity_kw": [1.0, 3.0, 0.0],
            "nbl_limit_kw": [1.0, 3.0, 0.0],
            "operational_stratum": ["A", "A", "A"],
            "operational_stratum_original": ["A", "A", "A"],
            "generator_eligible": [True, True, True],
        },
        geometry=[Point(0, 0), Point(4, 0), Point(10, 2)],
        crs="EPSG:28992",
    )


def test_buurt_target_aggregation_conserves_mass_and_uses_capacity_centroid() -> None:
    targets = aggregate_buurt_targets(
        _lineage(),
        working_crs="EPSG:28992",
        capacity_basis="nominal_normal_state",
    ).set_index("buurt_code")
    first = targets.loc["BU00010000"]
    assert first["station_id"] == "nl:buurt:BU00010000"
    assert first["peak_kw"] == 8.0
    assert first["capacity_kw"] == 4.0
    assert first["n_equipment"] == 2
    assert np.isclose(first.geometry.x, 3.0)
    assert np.isclose(first.geometry.y, 0.0)
    assert bool(first["over_capacity"])
    assert bool(first["noncoincident_conservative"])
    assert targets["peak_kw"].sum() == _lineage()["peak_kw"].sum()
    assert targets["capacity_kw"].sum() == _lineage()["capacity_kw"].sum()


def test_zero_capacity_buurt_uses_declared_deterministic_centroid_fallback() -> None:
    target = aggregate_buurt_targets(
        _lineage().loc[lambda frame: frame["buurt_code"].eq("BU00010100")],
        working_crs="EPSG:28992",
        capacity_basis="nominal_normal_state",
    ).iloc[0]
    assert target["centroid_method"] == "unweighted_rd_new_zero_capacity_fallback"
    assert bool(target["zero_capacity"])
    assert target.geometry.equals(Point(10, 2))


def test_low_equipment_stratum_merges_to_longest_adjacent_high_stratum() -> None:
    equipment = gpd.GeoDataFrame(
        {
            "equipment_id": ["a1", "a2", "a3", "b1", "b2", "b3", "tiny"],
            "buurt_code": ["BUA1", "BUA1", "BUA1", "BUB1", "BUB1", "BUB1", "BUT1"],
            "operational_stratum_original": ["A", "A", "A", "B", "B", "B", "TINY"],
            "peak_kw": [1.0] * 7,
            "generator_eligible": [True] * 7,
        },
        geometry=[Point(0.2, 0.2)] * 3 + [Point(2.2, 0.2)] * 3 + [Point(1.2, 0.2)],
        crs="EPSG:28992",
    )
    polygons = gpd.GeoDataFrame(
        {
            "buurt_code": ["BUA1", "BUT1", "BUB1"],
            "wijk_code": ["WKA", "WKT", "WKB"],
        },
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1), box(2, 0, 3, 1)],
        crs="EPSG:28992",
    )
    mapping, evidence = merge_low_equipment_strata(
        equipment,
        polygons,
        minimum_equipment=2,
        working_crs="EPSG:28992",
    )
    # Both borders are length one, so the authority tie-break chooses A.
    assert mapping["TINY"] == "A"
    assert evidence[0]["tie_break"].endswith("stratum_lexicographic")


def test_nl_contract_references_shared_authority_without_duplicate_sections() -> None:
    admission = load_engineering_admission(
        REPO_ROOT,
        "casestudy/1_DataOverview/4_NL/admission.toml",
    )
    assert not ({"grid", "graph", "memory"} & set(admission.country))
    assert admission.country["partition"]["minimum_analysis_regions"] == 14
    assert admission.country["partition"]["maximum_analysis_regions"] == 16
    assert admission.country["crosswalk"]["conflict_policy"].startswith(
        "unique_or_lexicographic_pip"
    )


@pytest.mark.local_data
def test_real_nl_v2_artifacts_are_wijk_to_buurt_and_admitted() -> None:
    derived = REPO_ROOT / "data/datasets/2_derived/nl"
    if not (derived / "bplus/source_regions.gpkg").is_file():
        pytest.skip("formal NL DataOverview artifacts are not present in this checkout")
    source = gpd.read_file(derived / "bplus/source_regions.gpkg", layer="source_regions")
    targets = gpd.read_file(
        derived / "bplus/buurt_pseudo_stations.gpkg", layer="buurt_pseudo_stations"
    )
    lineage = gpd.read_file(
        derived / "lineage/equipment_register.gpkg", layer="equipment_lineage"
    )
    gate = json.loads((derived / "audit/gate_a.json").read_text(encoding="utf-8"))
    engineering = json.loads(
        (derived / "audit/engineering_admission.json").read_text(encoding="utf-8")
    )
    inventory = json.loads(
        (derived / "authority/analysis_region_inventory.json").read_text(encoding="utf-8")
    )
    assert gate["status"] == "PASS"
    assert engineering["status"] == "ADMITTED"
    assert inventory["status"] == "FROZEN_PASS"
    assert len(source) == 1283
    assert len(targets) == 5592
    assert len(lineage) == 31367
    assert inventory["n_regions"] == 16
    assert engineering["idr_matched"] == {
        "maximum": 2000,
        "pass": True,
        "realizations": 1824,
    }
    assert sum(row["canonical_empty_targets"] for row in engineering["regions"]) == 0
    assert min(row["min_cells_per_target"] for row in engineering["regions"]) >= 4
    assert min(row["min_cells_per_source"] for row in engineering["regions"]) >= 4
    assert np.isclose(targets["peak_kw"].sum(), source["demand_peak_kw"].sum())
    assert not lineage["formal_evaluation_eligible"].astype(bool).any()


@pytest.mark.local_results
def test_nl_smoke_is_explicitly_nonformal_train_verify_infer() -> None:
    root = REPO_ROOT / "results/_smoke/2_Generator/4_NL"
    if not (root / "pre_hpc_smoke.json").is_file():
        pytest.skip("NL Generator smoke artifacts are not present in this checkout")
    receipt = json.loads((root / "pre_hpc_smoke.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "PASS"
    assert receipt["formal"] is False
    assert receipt["source_level"] == "wijk"
    assert receipt["target_level"] == "buurt_pseudo_station"
    assert {row["family"] for row in receipt["training"]} == {"gnn", "mlp"}
    for row in receipt["training"]:
        assert row["formal"] is False
        assert row["epochs"] == 2
        training = json.loads((root / row["completion"]).read_text(encoding="utf-8"))
        inference = json.loads(
            (root / row["inference_completion"]).read_text(encoding="utf-8")
        )
        training_verify = json.loads(
            (root / row["training_verify"]).read_text(encoding="utf-8")
        )
        inference_verify = json.loads(
            (root / row["inference_verify"]).read_text(encoding="utf-8")
        )
        assert training["execution_identity"]["formal"] is False
        assert inference["execution_identity"]["formal"] is False
        assert training_verify["status"] == "PASS" and training_verify["formal"] is False
        assert inference_verify["status"] == "PASS" and inference_verify["formal"] is False
        assert inference["n_regions"] == 4
