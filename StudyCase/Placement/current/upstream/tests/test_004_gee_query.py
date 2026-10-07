from __future__ import annotations

from pathlib import Path

import pytest

from sglib.core.infra.config import load_dataoverview_config
from sglib.dataoverview.processing.queries import gee
from sglib.dataoverview.processing.registry import build_registry

pytestmark = pytest.mark.produce


ROOT = Path(__file__).resolve().parents[1]


def _loaded(country: str, directory: str):
    return load_dataoverview_config(
        ROOT / "casestudy/1_DataOverview/general/general.toml",
        ROOT / f"casestudy/config/countries/{country}.toml",
        ROOT / f"casestudy/1_DataOverview/{directory}/{country}.toml",
    )


def _gee_query(country: str, directory: str) -> dict:
    values = _loaded(country, directory).values
    return {
        **dict(values["ntl"]),
        **dict(values["datasets"]["ntl"]["query"]),
    }


@pytest.mark.local_data
def test_existing_nl_gee_landing_is_validated_and_receipted(tmp_path) -> None:
    import shutil
    query = _gee_query("nl", "4_NL")
    raw = ROOT / "data/datasets/1_raw/nl"
    if not (raw / query["output"]).is_file():
        pytest.skip("local NL GEE landing is not present in this checkout")
    destination = tmp_path / query["output"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(raw / query["output"], destination)
    target, receipt = gee.run(query, tmp_path)
    assert target.name == "viirs_2024.tif"
    assert receipt.name == "acquisition_receipt.json"
    assert target.is_file() and receipt.is_file()
    assert gee._validate_raster(target, query)["count"] == 2


def test_gee_missing_credential_fails_before_network(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("GEE_SERVICE_ACCOUNT_KEY_PATH", raising=False)
    query = _gee_query("nz", "5_NZ")
    with pytest.raises(gee.GeeAcquisitionError, match="missing GEE credential"):
        gee.run(query, tmp_path)


def test_gee_output_may_not_escape_country_raw_root(tmp_path: Path) -> None:
    query = _gee_query("nz", "5_NZ")
    query["output"] = "../escape.tif"
    with pytest.raises(gee.GeeAcquisitionError, match="escapes"):
        gee.run(query, tmp_path)


def test_gee_registry_done_contract_includes_acquisition_receipt() -> None:
    loaded = _loaded("nz", "5_NZ")
    registry = build_registry(ROOT, {"nz": dict(loaded.values)})
    outputs = registry["nz.ntl.download"].produces
    assert tuple(path.name for path in outputs) == (
        "viirs_dnb_annual_v22_2024_nz.tif",
        "acquisition_receipt.json",
    )
