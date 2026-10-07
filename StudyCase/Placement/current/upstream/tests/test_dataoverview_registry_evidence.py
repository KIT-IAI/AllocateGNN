from __future__ import annotations
from pathlib import Path
import pytest
from sglib.core.infra.config import load_dataoverview_config
from sglib.dataoverview.evidence import load_handoff_evidence
from sglib.dataoverview.processing.registry import build_registry
pytestmark = pytest.mark.gate
ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "casestudy" / "1_DataOverview"
PROFILES = ROOT / "casestudy" / "config" / "countries"


def test_registry_is_toml_discovered_and_acyclic() -> None:
    configs = {}
    for country, directory in (
        ("uk", "1_UK"),
        ("au", "2_AU"),
        ("de", "3_DE"),
        ("nl", "4_NL"),
        ("nz", "5_NZ"),
    ):
        configs[country] = dict(
            load_dataoverview_config(
                STAGE / "general/general.toml",
                PROFILES / f"{country}.toml",
                STAGE / directory / f"{country}.toml",
            ).values
        )
    registry = build_registry(ROOT, configs)
    assert "uk.grid.derive" in registry
    assert "au.fy2024.derive" in registry
    assert "de.inventory.overview" in registry
    assert registry["general.matrix.overview"].depends_on == ()


def test_ogcapi_source_done_contract_includes_pagination_audit() -> None:
    configs = {
        "nl": {
            "datasets": {
                "boundaries": {
                    "category": "boundaries_grid",
                    "query_protocol": "ogcapi",
                    "query": {
                        "url": "https://example.test/items",
                        "filename": "boundaries.geojson",
                    },
                }
            },
            "products": {
                "inventory": {
                    "category": "features_cuz",
                    "depends_on": [],
                    "produces": [],
                }
            },
        }
    }
    registry = build_registry(ROOT, configs)
    outputs = registry["nl.boundaries.download"].produces
    assert tuple(path.name for path in outputs) == (
        "boundaries.geojson",
        "boundaries.geojson.pagination.json",
    )


def test_handoff_evidence_is_scoped_hashed_and_status_typed(tmp_path: Path) -> None:
    gate = tmp_path / "gate.json"
    gate.write_text('{"status":"PROVISIONAL_REVIEW_REQUIRED"}', encoding="utf-8")
    evidence = load_handoff_evidence(
        tmp_path,
        {
            "handoff_artifacts": {
                "engineering_admission": {
                    "path": "gate.json",
                    "formal_required": True,
                    "required_status": "PASS",
                }
            }
        },
    )["engineering_admission"]
    assert evidence.repo_relative == "gate.json"
    assert evidence.document == {"status": "PROVISIONAL_REVIEW_REQUIRED"}
    assert evidence.formal_required is True
    assert evidence.required_status == "PASS"
    assert len(evidence.sha256) == 64


def test_required_handoff_gate_is_not_done_until_status_passes(tmp_path: Path) -> None:
    gate = tmp_path / "gate.json"
    gate.write_text('{"status":"PROVISIONAL_REVIEW_REQUIRED"}', encoding="utf-8")
    configs = {
        "nz": {
            "canonical": {},
            "station_contract": {},
            "datasets": {},
            "handoff_artifacts": {
                "admission": {
                    "path": "gate.json",
                    "formal_required": True,
                    "required_status": "PASS",
                }
            },
            "products": {
                "admission": {
                    "category": "boundaries_grid",
                    "depends_on": [],
                    "produces": ["gate.json"],
                },
                "inventory": {
                    "category": "features_cuz",
                    "depends_on": ["nz.admission.derive"],
                    "produces": ["inventory.json"],
                },
            },
        }
    }
    unit = build_registry(tmp_path, configs)["nz.admission.derive"]
    assert unit.done() is False
    gate.write_text('{"status":"PASS"}', encoding="utf-8")
    assert unit.done() is True
