from __future__ import annotations
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
import pytest
from sglib.core.infra.config import ConfigError, load_dataoverview_config
from sglib.core.infra.schema import (
    SchemaValidationError,
    validate_artifact,
    validate_table,
)
pytestmark = pytest.mark.gate
ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "casestudy" / "1_DataOverview"
PROFILES = ROOT / "casestudy" / "config" / "countries"


@pytest.mark.parametrize(
    "country,directory,count",
    [
        ("uk", "1_UK", 16),
        ("au", "2_AU", 12),
        ("de", "3_DE", 1),
        ("nl", "4_NL", 16),
        ("nz", "5_NZ", 9),
    ],
)
def test_strict_toml_layers_load(country: str, directory: str, count: int) -> None:
    loaded = load_dataoverview_config(
        STAGE / "general/general.toml",
        PROFILES / f"{country}.toml",
        STAGE / directory / f"{country}.toml",
    )
    assert loaded.country_profile.code == country
    assert len(loaded.values["regions"]["items"]) == count
    assert loaded.values["grid"]["grid_policy"] == "bounded_equal_area_budget_v1"
    assert all(source.is_file() for source in loaded.sources.values())


def test_duplicate_layer_key_is_rejected(tmp_path: Path) -> None:
    profile = tmp_path / "uk.toml"
    profile.write_text((PROFILES / "uk.toml").read_text(encoding="utf-8"), encoding="utf-8")
    country = tmp_path / "country.toml"
    country.write_text(
        'schema_version = "sg_dataoverview_country_v1"\n[regions]\nitems=[]\n[country]\ncode="uk"\n',
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="unknown country overlay"):
        load_dataoverview_config(STAGE / "general/general.toml", profile, country)


def test_npz_schema_rejects_wrong_shape(tmp_path: Path) -> None:
    artifact = tmp_path / "landuse.npz"
    np.savez(artifact, data=np.ones((3, 4)), columns=np.asarray(["a", "b", "c", "d"]))
    with pytest.raises(SchemaValidationError, match="shape"):
        validate_artifact(artifact, STAGE / "schemas/landuse.toml")


def test_table_schema_supports_rows_boolean_and_numeric_bounds() -> None:
    schema = {
        "table": {
            "exact_rows": 2,
            "allow_extra_columns": False,
            "columns": [
                {
                    "name": "value",
                    "dtype": "number",
                    "nullable": False,
                    "minimum": 0,
                    "maximum": 2,
                },
                {
                    "name": "flag",
                    "dtype": "boolean",
                    "nullable": False,
                },
            ],
        }
    }
    validate_table(pd.DataFrame({"value": [0.0, 2.0], "flag": [True, False]}), schema)
    with pytest.raises(SchemaValidationError, match="expected exactly"):
        validate_table(pd.DataFrame({"value": [1.0], "flag": [True]}), schema)
    with pytest.raises(SchemaValidationError, match="above"):
        validate_table(pd.DataFrame({"value": [0.0, 3.0], "flag": [True, False]}), schema)


def test_installed_module_runs_outside_checkout(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "sglib"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=True,
    )
    assert result.stdout.strip().startswith("sglib 0.1")
