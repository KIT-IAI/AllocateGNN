from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, box

from sglib.dataoverview.processing.derive.nl import PRODUCT_RUNNERS
from sglib.dataoverview.processing.derive.nl.pipeline import (
    classify_crosswalk,
    deterministic_analysis_regions,
)

import pytest
pytestmark = pytest.mark.produce


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_nl_product_runners_are_country_local_and_complete():
    assert set(PRODUCT_RUNNERS) == {
        "regions",
        "stations",
        "gate_a",
        "grid_skeleton",
        "grid",
        "features",
        "inventory",
    }


def test_nl_display_notebooks_are_numbered_chain_steps_without_side_writes():
    notebooks = sorted((REPO_ROOT / "casestudy/1_DataOverview/4_NL").glob("*.ipynb"))
    assert [path.name for path in notebooks] == [
        "02_regions_stations.ipynb",
        "03_grid.ipynb",
        "05_features_overview.ipynb",
        "06_inventory.ipynb",
    ]
    for path in notebooks:
        document = json.loads(path.read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in document["cells"]
            if cell.get("cell_type") == "code"
        )
        assert 'COUNTRY = "nl"' in code
        if path.name == "06_inventory.ipynb":
            assert "from sglib.dataoverview.manifest import run_gate" in code
            assert "report = run_gate(context)" in code
            assert "stage.run_step(" not in code
            assert "load_bundle(" not in code
        else:
            assert "stage.run_step(" in code or "stage.require_done(" in code
        assert not any(
            token in code
            for token in ("to_csv", "to_file", "write_text", "savefig", "run_training", "download")
        )
    inventory = json.loads((REPO_ROOT / "casestudy/1_DataOverview/4_NL/06_inventory.ipynb").read_text(encoding="utf-8"))
    code = "\n".join("".join(cell.get("source", [])) for cell in inventory["cells"] if cell.get("cell_type") == "code")
    assert 'if report["status"] == "PASS":' in code
    assert 'inventory_path.read_text(encoding="utf-8")' in code


def test_crosswalk_four_classes_are_exhaustive():
    polygons = gpd.GeoDataFrame(
        {"buurt_code": ["BU00000001", "BU00000002"]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
        crs="EPSG:4326",
    )
    cbs = pd.DataFrame(
        {
            "gwb_code_10": ["BU00000001", "BU00000002"],
            "recs": ["Buurt", "Buurt"],
        }
    )
    equipment = pd.DataFrame(
        {
            "CONDUCTINGEQUIPMENT_NAME": ["exact", "fallback", "conflict", "outside"],
            "STREETDETAIL_CODE": ["BU00000001", "BU99999999", "BU00000001", "BU99999998"],
            "POSITIONPOINT_XPOSITION": [0.5, 1.5, 1.5, 3.0],
            "POSITIONPOINT_YPOSITION": [0.5, 0.5, 0.5, 3.0],
        }
    )
    audit, assigned = classify_crosswalk(equipment, cbs, polygons)
    assert audit["crosswalk_class"].tolist() == [
        "exact_code",
        "pip_fallback",
        "conflict",
        "outside_all_polygons",
    ]
    assert assigned["buurt_code"].astype("string").tolist()[:3] == [
        "BU00000001",
        "BU00000002",
        "BU00000002",
    ]
    assert pd.isna(assigned.loc[3, "buurt_code"])


def test_analysis_region_partition_is_deterministic_under_row_reordering():
    sources = gpd.GeoDataFrame(
        {
            "buurt_code": [f"BU{i:08d}" for i in range(8)],
            "operational_stratum": ["A"] * 4 + ["B"] * 4,
        },
        geometry=[Point(i * 1000.0, (i % 2) * 100.0) for i in range(8)],
        crs="EPSG:28992",
    )
    equipment_rows = []
    for index, source in sources.iterrows():
        for member in range(3):
            equipment_rows.append(
                {
                    "equipment_id": f"e{index}-{member}",
                    "buurt_code": source["buurt_code"],
                    "geometry": Point(source.geometry.x + member, source.geometry.y),
                }
            )
    equipment = gpd.GeoDataFrame(equipment_rows, geometry="geometry", crs="EPSG:28992")
    limits = {
        "target_grid_cells": 100,
        "max_sources": 10,
        "provisional_max_sources": 2,
        "max_targets": 10,
        "provisional_max_targets": 6,
        "planning_k_cap": 10,
    }
    first, _, first_records = deterministic_analysis_regions(
        sources, equipment, working_crs="EPSG:28992", limits=limits
    )
    second, _, second_records = deterministic_analysis_regions(
        sources.sample(frac=1, random_state=9).reset_index(drop=True),
        equipment.sample(frac=1, random_state=7).reset_index(drop=True),
        working_crs="EPSG:28992",
        limits=limits,
    )
    first_map = dict(zip(first["buurt_code"], first["analysis_region"]))
    second_map = dict(zip(second["buurt_code"], second["analysis_region"]))
    assert first_map == second_map
    assert [item["id"] for item in first_records] == [item["id"] for item in second_records]
