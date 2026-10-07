from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import shutil
from typing import Any

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, Polygon

from sglib.dataoverview.processing.derive.nz import PRODUCT_RUNNERS
from sglib.dataoverview.processing.derive.nz.fresh_sources import (
    CORE_EDBS,
    FreshSourcePaths,
    NZFreshSourceError,
    acquire_nz_fresh_sources,
    fetch_arcgis_complete,
    parse_d5,
    select_d6_truth_2024,
    verify_fresh_manifest,
)

pytestmark = pytest.mark.produce


ROOT = Path(__file__).resolve().parents[1]


class _Response:
    def __init__(self, *, document: dict[str, Any] | None = None, content: bytes = b""):
        self._document = document
        self._content = content
        self.headers = {"Content-Length": str(len(content))} if content else {}

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        assert self._document is not None
        return self._document

    def iter_content(self, chunk_size: int):
        for start in range(0, len(self._content), chunk_size):
            yield self._content[start : start + chunk_size]


class _ArcGISRequester:
    def __init__(self, identities: list[int]):
        self.identities = identities
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, **kwargs: Any) -> _Response:
        params = kwargs["params"]
        self.calls.append(dict(params))
        if params.get("returnIdsOnly") == "true":
            return _Response(document={"objectIds": list(reversed(self.identities))})
        requested = [int(value) for value in params["objectIds"].split(",")]
        return _Response(
            document={
                "features": [
                    {"attributes": {"OBJECTID": identity, "value": identity * 10}}
                    for identity in requested
                ]
            }
        )


def _parquet_bytes(frame: pd.DataFrame) -> bytes:
    stream = BytesIO()
    frame.to_parquet(stream, index=False)
    return stream.getvalue()


def _d5_fixture() -> bytes:
    rows = []
    for index, edb in enumerate((*CORE_EDBS, "WEL Networks"), start=1):
        rows.append(
            {
                "edb": edb,
                "geom_kind": "POINT",
                "name": f"STATION {index}",
                "feeders": None,
                "input_voltage": "33",
                "output_voltage": "11",
                "extra": None,
                "geometry": Point(1_600_000 + index, 5_200_000 + index).wkb,
            }
        )
    rows.append(
        {
            "edb": "Vector Lines",
            "geom_kind": "POLYGON",
            "name": "SERVICE AREA",
            "feeders": None,
            "input_voltage": None,
            "output_voltage": None,
            "extra": None,
            "geometry": Polygon(
                [(1_599_999, 5_199_999), (1_600_001, 5_199_999), (1_600_001, 5_200_001)]
            ).wkb,
        }
    )
    return _parquet_bytes(pd.DataFrame(rows))


def _d6_fixture() -> bytes:
    rows = []
    for edb in CORE_EDBS:
        for description, value, text in (
            ("Current Peak Load (MVA)", 10.0, None),
            ("Installed Firm Capacity (MVA)", 20.0, None),
            ("Security of Supply Classification (type)", None, "N-1"),
        ):
            rows.append(
                {
                    "edb": edb,
                    "network": "All",
                    "disc_yr": 2024.0,
                    "section": "12b(i): System Growth - Zone Substations",
                    "category": "Existing Zone Substation",
                    "sub_category": f"{edb} station",
                    "description": description,
                    "value": value,
                    "text_input": text,
                    "source": "Year beginning",
                    "version": "2024.05.1",
                }
            )
    rows.append(
        {
            **rows[0],
            "disc_yr": 2026.0,
            "value": 999.0,
            "version": "2026.1",
        }
    )
    return _parquet_bytes(pd.DataFrame(rows))


class _AcquisitionRequester:
    def __init__(self, static_content: dict[str, bytes]):
        self.static_content = static_content

    def get(self, url: str, **kwargs: Any) -> _Response:
        if "params" not in kwargs:
            return _Response(content=self.static_content[url])
        params = kwargs["params"]
        if params.get("returnIdsOnly") == "true":
            return _Response(document={"objectIds": [1]})
        identity = int(params["objectIds"])
        if "Statistical_Area_2" in url:
            feature = {
                "type": "Feature",
                "properties": {
                    "OBJECTID": identity,
                    "SA22023_V1_00": "100001",
                    "SA22023_V1_00_NAME": "Fixture SA2",
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[170.0, -44.0], [171.0, -44.0], [171.0, -43.0], [170.0, -44.0]]],
                },
            }
        elif "Statistical_Area_3" in url:
            feature = {
                "type": "Feature",
                "properties": {
                    "OBJECTID": identity,
                    "SA32023_V1_00": "60001",
                    "SA32023_V1_00_NAME": "Fixture SA3",
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[170.0, -44.0], [171.0, -44.0], [171.0, -43.0], [170.0, -44.0]]],
                },
            }
        elif "FeatureServer/1" in url:
            feature = {
                "attributes": {
                    "OBJECTID": identity,
                    "SA22023_V1_00": "100001",
                    "SA22023_V1_00_NAME": "Fixture SA2",
                    "VAR_1_3": 10,
                }
            }
        else:
            feature = {
                "attributes": {
                    "OBJECTID": identity,
                    "SA22023_V1_00": "100001",
                    "SA22023_V1_00_NAME": "Fixture SA2",
                    **{f"VAR_2_{number}": 1 for number in range(591, 611)},
                    **{f"VAR_2_{number}": 1 for number in range(657, 677)},
                }
            }
        return _Response(document={"features": [feature]})


class _NoNetworkRequester:
    def get(self, url: str, **kwargs: Any) -> _Response:
        raise AssertionError(f"verified landing unexpectedly used the network: {url}")


def test_arcgis_object_id_pagination_is_complete_and_reproducible(tmp_path: Path) -> None:
    requester = _ArcGISRequester([1, 2, 3, 4, 5])
    target = tmp_path / "attributes.json"
    receipt = fetch_arcgis_complete(
        configured_url=(
            "https://example.test/FeatureServer/0/query?where=1%3D1"
            "&outFields=OBJECTID%2Cvalue&returnGeometry=false&f=json"
        ),
        target=target,
        identity_field="OBJECTID",
        requester=requester,
        geometry=False,
        page_size=2,
    )
    audit = json.loads(
        target.with_suffix(".json.pagination.json").read_text(encoding="utf-8")
    )
    assert receipt["identity_count"] == 5
    assert audit["complete"] is True
    assert [page["returned"] for page in audit["pages"]] == [2, 2, 1]
    output_ids = [
        item["attributes"]["OBJECTID"]
        for item in json.loads(target.read_text(encoding="utf-8"))["features"]
    ]
    assert output_ids == [1, 2, 3, 4, 5]


def test_d5_parser_selects_points_and_records_coordinate_identity(tmp_path: Path) -> None:
    path = tmp_path / "d5.parquet"
    path.write_bytes(_d5_fixture())
    frame = parse_d5(path)
    assert len(frame) == 4
    assert set(frame["geom_kind"]) == {"POINT"}
    assert frame["source_row_identity"].is_unique
    assert frame.crs.to_epsg() == 4326


def test_d6_selects_only_2024_truth_and_records_lineage(tmp_path: Path) -> None:
    path = tmp_path / "d6.parquet"
    path.write_bytes(_d6_fixture())
    truth, audit = select_d6_truth_2024(path)
    assert len(truth) == 3
    assert truth["disc_yr"].eq(2024).all()
    assert truth["actual_peak_mva"].eq(10).all()
    assert truth["firm_capacity_mva"].eq(20).all()
    assert set(truth["security_class"]) == {"N-1"}
    assert audit["long_rows"] == 9
    assert audit["forecast_2026_excluded"] is True
    assert len(audit["semantic_identity_sha256"]) == 64


def test_full_acquisition_writes_anchor_free_relocatable_manifest(tmp_path: Path) -> None:
    overlay = tmp_path / "casestudy/1_DataOverview/5_NZ"
    overlay.mkdir(parents=True)
    shutil.copy2(ROOT / "casestudy/1_DataOverview/5_NZ/nz.toml", overlay / "nz.toml")
    real_overlay = (ROOT / "casestudy/1_DataOverview/5_NZ/nz.toml").read_bytes()
    # URLs remain the authority; map each static URL to deterministic fixture bytes.
    import tomllib

    parsed = tomllib.loads(real_overlay.decode("utf-8"))
    static_content: dict[str, bytes] = {}
    for dataset, content in (
        ("comcom_geospatial", _d5_fixture()),
        ("comcom_disclosure", _d6_fixture()),
        ("determination_2023", b"%PDF-1.4 fixture 2023"),
        ("determination_2026", b"%PDF-1.4 fixture 2026"),
    ):
        static_content[parsed["datasets"][dataset]["files"][0]["url"]] = content
    requester = _AcquisitionRequester(static_content)
    paths = acquire_nz_fresh_sources(
        tmp_path,
        requester=requester,
        acquired_at_utc="2026-09-02T00:00:00+00:00",
    )
    document = verify_fresh_manifest(paths)
    assert document["complete"] is True
    assert document["anchor_used"] is False
    assert document["d6_truth_2024"]["wide_rows"] == 3
    assert all(not Path(item["path"]).is_absolute() for item in document["sources"].values())
    assert all(
        item["fresh_official_download"] is True
        for item in document["sources"].values()
    )
    reused = acquire_nz_fresh_sources(
        tmp_path,
        requester=_NoNetworkRequester(),
        acquired_at_utc="2026-09-02T00:15:00+00:00",
    )
    reused_document = verify_fresh_manifest(reused)
    assert {
        item["status"] for item in reused_document["sources"].values()
    } == {"reused_verified_landing"}


def test_manifest_is_mandatory_even_when_raw_files_exist(tmp_path: Path) -> None:
    paths = FreshSourcePaths.from_repo(tmp_path)
    paths.d5.parent.mkdir(parents=True)
    paths.d5.write_bytes(_d5_fixture())
    with pytest.raises(NZFreshSourceError, match="manifest is missing"):
        verify_fresh_manifest(paths)


def test_formal_product_runner_is_bound_to_fresh_builder() -> None:
    runner = PRODUCT_RUNNERS["station_ledger_2024"]
    assert "build_core9_dataoverview_from_fresh" in runner.__globals__
    assert runner.__globals__["build_core9_dataoverview_from_fresh"].__name__.endswith(
        "from_fresh"
    )
