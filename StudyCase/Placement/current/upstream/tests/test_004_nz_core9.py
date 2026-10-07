from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from sglib.core.infra.config import load_dataoverview_config
from sglib.core.infra.schema import validate_artifact
from sglib.core.infra.terms import load_country_profile
from sglib.dataoverview.processing.derive.nz import PRODUCT_RUNNERS
from sglib.dataoverview.processing.derive.nz.fresh_sources import (
    NZFreshSourceError,
    _section_reconciliation,
)
from sglib.dataoverview.processing.derive.nz.core9 import (
    CORE_EDBS,
    REGION_ORDER,
    build_core9_dataoverview_from_fresh,
    build_size_gate,
)

pytestmark = pytest.mark.produce


ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = ROOT / "casestudy/1_DataOverview/schemas"


def test_nz_profile_overlay_and_product_adapter_contract() -> None:
    profile = load_country_profile(ROOT / "casestudy/config/countries/nz.toml")
    assert profile.code == "nz"
    assert profile.directory == "5_NZ"
    assert profile.crs == {
        "native": "EPSG:2193",
        "area": "EPSG:2193",
        "generation": "EPSG:3857",
        "storage": "EPSG:4326",
        "working": "EPSG:2193",
    }
    # Counts of one materialisation are not contract terms; the gate records observed counts.
    assert not {"formal_truth_rows", "section_lineage_rows"} & set(profile.station_contract)
    assert profile.station_contract["core_edbs"] == list(CORE_EDBS)
    loaded = load_dataoverview_config(
        ROOT / "casestudy/1_DataOverview/general/general.toml",
        ROOT / "casestudy/config/countries/nz.toml",
        ROOT / "casestudy/1_DataOverview/5_NZ/nz.toml",
    )
    assert tuple(item["id"] for item in loaded.values["regions"]["items"]) == REGION_ORDER
    assert set(loaded.values["canonical"]) == {"regions", "stations"}
    assert set(PRODUCT_RUNNERS) == {
        "station_ledger_2024",
        "station_sites",
        "analysis_regions",
        "admission_gate",
    }


@pytest.mark.local_data
def test_core9_rebuild_is_site_truth_and_section_lineage(tmp_path: Path) -> None:
    # The rebuild is consumed as it is; it is checked for internal consistency and
    # contract semantics, never against counts or tables of an earlier materialisation.
    artifacts = build_core9_dataoverview_from_fresh(ROOT, tmp_path, force=True)
    for artifact, schema in (
        (artifacts.sources, "regions_nz.toml"),
        (artifacts.sites, "stations_nz.toml"),
        (artifacts.ledger, "station_ledger_nz.toml"),
        (artifacts.analysis_regions, "analysis_regions_nz.toml"),
        (artifacts.gate, "nz_gate.toml"),
    ):
        validate_artifact(artifact, SCHEMAS / schema)
    sites = gpd.read_file(artifacts.sites, layer="station_sites")
    ledger = gpd.read_file(artifacts.ledger, layer="station_ledger_2024")
    sources = gpd.read_file(artifacts.sources, layer="source_regions")
    analysis = gpd.read_file(artifacts.analysis_regions, layer="analysis_regions")
    assert set(sites["edb"]) == set(CORE_EDBS)
    assert "WEL Networks" not in set(sites["edb"])
    assert tuple(analysis["analysis_region"]) == REGION_ORDER
    assert sites["station_id"].is_unique
    assert sources["source_id"].is_unique
    assert set(ledger["station_id"]) == set(sites["station_id"])
    assert len(ledger) >= len(sites)
    assert sites["assignment_eligible"].all()
    assert not ledger["assignment_eligible"].any()
    sections = ledger.groupby("station_id").size()
    assert (sites.set_index("station_id")["noncoincident_conservative"] == sections.reindex(sites["station_id"]).gt(1).to_numpy()).all()
    assert np.isclose(sites["actual_peak_mva"].sum(), ledger["actual_peak_mva"].sum())
    assert np.isclose(sites["actual_peak_mva"].sum(), sources["demand_peak_mva"].sum())
    gate = json.loads(artifacts.gate.read_text(encoding="utf-8"))
    assert gate["status"] == "PASS"
    assert gate["source_mode"] == "fresh_official_downloads"
    assert gate["formal_truth_rows"] == len(sites)
    assert gate["section_lineage_rows"] == len(ledger)
    assert gate["formal_hpc_completion"] is False
    assert all(gate["checks"].values())
    assert "anchor_files" not in gate
    reconciliation = gate["reconciliation"]
    sections = pd.read_csv(tmp_path / reconciliation["sections"]["table"], keep_default_na=False)
    included = sections["status"].eq("included")
    assert reconciliation["sections"]["landed_rows"] == len(sections)
    assert reconciliation["sections"]["included"] == int(included.sum()) == len(ledger)
    assert sections.loc[~included, "exclusion_reasons"].ne("").all()
    assert sections.loc[included, "exclusion_reasons"].eq("").all()
    assert set(sections.loc[included, "lineage_id"]) == set(ledger["lineage_id"])
    landed = pd.read_csv(tmp_path / reconciliation["sources"]["table"], dtype={"SA22023_V1_00": str}, keep_default_na=False)
    assert landed["SA22023_V1_00"].is_unique
    assert set(landed.loc[landed["status"].eq("included"), "SA22023_V1_00"]) == set(sources["SA22023_V1_00"].astype(str))
    coverage = pd.read_csv(tmp_path / reconciliation["candidate_sa3"]["table"])
    assert np.isclose(coverage["coverage"], 1.0, atol=1e-6).all()


def _truth_rows(**overrides):
    rows = pd.DataFrame(
        {
            "edb": ["Orion NZ", "Orion NZ", "Orion NZ"],
            "network": ["All", "All", "All"],
            "disc_yr": [2024.0, 2024.0, 2024.0],
            "sub_category": ["Kept", "Weak", "Lost"],
            "match_method": ["exact_normalised", "exact_normalised", "unmatched"],
            "matched_d5_name": ["Kept", "Weak", None],
            "actual_peak_mva": [10.0, 5.0, 3.0],
            "firm_capacity_mva": [20.0, 0.0, 8.0],
            "security_class": ["N-1", "N", "N-1"],
            "coordinate_matched": [True, True, False],
        }
    )
    rows["four_task_immediate"] = [True, False, False]
    return rows.assign(**overrides)


def test_section_reconciliation_lists_every_failing_rule_and_every_landed_row() -> None:
    focus = _truth_rows()
    table = _section_reconciliation(focus, focus)
    assert table.set_index("sub_category")["exclusion_reasons"].to_dict() == {
        "Kept": "",
        "Lost": "coordinate_not_matched",
        "Weak": "firm_capacity_nonpositive;security_class_not_declared_secure",
    }
    landed = pd.concat([focus, focus.iloc[[0]].assign(sub_category="Dropped upstream")])
    with pytest.raises(NZFreshSourceError, match="does not cover the landed D6 rows"):
        _section_reconciliation(landed, focus)
    with pytest.raises(NZFreshSourceError, match="differ from the truth filter"):
        _section_reconciliation(focus, _truth_rows(four_task_immediate=[True, True, False]))


@pytest.mark.slow
@pytest.mark.local_data
def test_size_gate_uses_frozen_v2_authority_and_proves_core9_minimality(tmp_path: Path) -> None:
    path = build_size_gate(ROOT, tmp_path, force=True)
    validate_artifact(path, SCHEMAS / "engineering_admission.toml")
    gate = json.loads(path.read_text(encoding="utf-8"))
    assert gate["schema_version"] == "sg_nz_size_gate_evidence_v2"
    assert gate["status"] == "PASS"
    assert gate["admission_decision"] == "ADMITTED"
    assert gate["authority_status"] == "FROZEN"
    assert len(gate["regions"]) == 9
    assert all(record["empty_targets"] == 0 for record in gate["regions"])
    assert all(record["min_cells_per_source"] >= 4 for record in gate["regions"])
    assert all(record["min_cells_per_target"] >= 4 for record in gate["regions"])
    assert all(record["status"] == "PASS" for record in gate["regions"])
    assert all(all(record["hard_checks"].values()) for record in gate["regions"])
    assert all(record["cdist_peak_workspace_mib"] <= 128 for record in gate["regions"])
    assert any(not record["memory_advisory"]["dense_tensor_limit_met"] for record in gate["regions"])
    minimum = gate["minimum_count_evidence"]
    assert minimum["all_smaller_candidates_fail"] is True
    assert minimum["core9_preserved"] is True
    assert [item["regions_per_edb"] for item in minimum["smaller_candidates"]] == [1, 2]
    assert [item["total_regions"] for item in minimum["smaller_candidates"]] == [3, 6]
    assert {item["status"] for item in minimum["smaller_candidates"]} == {"FAIL"}
    assert gate["idr_matched"] == {
        "formal_candidates": 38,
        "seeds": 3,
        "regions": 9,
        "realizations": 1026,
        "max_realizations_all_regions": 2000,
        "pass": True,
    }
    assert gate["hard_gate_failures"] == []
