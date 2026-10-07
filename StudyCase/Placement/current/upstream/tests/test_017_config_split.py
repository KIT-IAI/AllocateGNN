"""DataOverview CLI and five-country configuration invariants."""

from __future__ import annotations

from pathlib import Path
import tomllib

from sglib.core.infra.config import load_dataoverview_config

import pytest
pytestmark = pytest.mark.gate


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_OVERVIEW = REPO_ROOT / "casestudy" / "1_DataOverview"


def test_dataoverview_has_one_resumable_cli_and_no_run_initializer():
    runner = DATA_OVERVIEW / "run_all.py"
    assert runner.is_file()
    assert "sglib.dataoverview.cli" in runner.read_text(encoding="utf-8")
    assert not (DATA_OVERVIEW / "000_initialize_run.ipynb").exists()
    assert not (DATA_OVERVIEW / "999_stage_gate.ipynb").exists()


def test_country_configs_are_independent_and_temporally_explicit():
    expected = {
        "1_UK": ("uk", "four_task_chain", 16),
        "2_AU": ("au", "four_task_chain", 12),
        "3_DE": ("de", "dataoverview_only", 1),
        "4_NL": ("nl", "four_task_chain", 16),
        "5_NZ": ("nz", "four_task_chain", 9),
    }
    profiles = REPO_ROOT / "casestudy/config/countries"
    for directory, (country, scope, regions) in expected.items():
        loaded = load_dataoverview_config(
            DATA_OVERVIEW / "general/general.toml",
            profiles / f"{country}.toml",
            DATA_OVERVIEW / directory / f"{country}.toml",
        )
        assert loaded.country_profile.code == country
        assert loaded.country_profile.evaluation_scope == scope
        assert len(loaded.values["regions"]["items"]) == regions

    with (profiles / "au.toml").open("rb") as stream:
        au = tomllib.load(stream)
    assert au["temporal"]["protocol"] == "FY2024_same_year"

