from __future__ import annotations
import json
from pathlib import Path
import geopandas as gpd
import pytest
from shapely.geometry import box
from sglib.dataoverview.processing.features.grid_bundle import load_grid_bundle, write_grid_bundle
from sglib.dataoverview.processing.features.grid_generator import design_grid, generate_base_grid
from sglib.dataoverview.processing.queries import ogcapi
pytestmark = pytest.mark.produce
ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "casestudy" / "1_DataOverview"
PROFILES = ROOT / "casestudy" / "config" / "countries"


def test_bounded_equal_area_grid_contract(tmp_path: Path) -> None:
    region = gpd.GeoDataFrame({"source": ["a"]}, geometry=[box(0, 0, 10_000, 10_000)], crs="EPSG:3857")
    design = design_grid(50_000, region, area_crs="EPSG:3857", min_ground_step_m=100, max_ground_step_m=500)
    assert design.target_ground_step_m == pytest.approx(100.0)
    assert design.clamp_branch == "minimum"
    grid = generate_base_grid(region, design.projected_step_m)
    write_grid_bundle("r", tmp_path, grid, design, source_key="source")
    loaded, step, metadata = load_grid_bundle("r", tmp_path)
    assert len(loaded) == len(grid)
    assert step == pytest.approx(100.0)
    assert metadata["area_crs"] == "EPSG:3857"
    assert metadata["ground_neighbour_distance_m"]["median"] == pytest.approx(100.0, rel=0.02)


def test_ogcapi_pagination_writes_complete_audit(tmp_path: Path, monkeypatch) -> None:
    class Response:
        def __init__(self, document):
            self.document = document

        def raise_for_status(self):
            return None

        def json(self):
            return self.document

    pages = {
        "https://example.test/items": {
            "type": "FeatureCollection",
            "numberMatched": 3,
            "numberReturned": 2,
            "features": [{"id": "a"}, {"id": "b"}],
            "links": [{"rel": "next", "href": "?offset=2"}],
        },
        "https://example.test/items?offset=2": {
            "type": "FeatureCollection",
            "numberMatched": 3,
            "numberReturned": 1,
            "features": [{"id": "c"}],
            "links": [],
        },
    }
    monkeypatch.setattr(
        ogcapi.requests,
        "get",
        lambda url, timeout: Response(pages[url]),
    )
    target, audit_path = ogcapi.run(
        {"url": "https://example.test/items", "filename": "items.geojson"},
        tmp_path,
    )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert json.loads(target.read_text(encoding="utf-8"))["features"][-1]["id"] == "c"
    assert audit["complete"] is True
    assert audit["expected_total"] == audit["total"] == 3
    assert [item["cumulative_count"] for item in audit["pages"]] == [2, 3]


def test_ogcapi_pagination_rejects_incomplete_collection(tmp_path: Path, monkeypatch) -> None:
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "numberMatched": 2,
                "numberReturned": 1,
                "features": [{"id": "a"}],
                "links": [],
            }

    monkeypatch.setattr(ogcapi.requests, "get", lambda url, timeout: Response())
    with pytest.raises(RuntimeError, match="pagination incomplete"):
        ogcapi.run(
            {"url": "https://example.test/items", "filename": "items.geojson"},
            tmp_path,
        )
    assert not (tmp_path / "items.geojson").exists()
