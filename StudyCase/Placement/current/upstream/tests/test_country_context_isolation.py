"""Country transforms consume their supplied roots, including repeated runs."""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, box

from sglib.dataoverview.processing.config import CountryPipelineContext
from sglib.dataoverview.processing.derive.au import au, au_ledger, au_pv
from sglib.dataoverview.processing.derive.de import PRODUCT_RUNNERS as DE_RUNNERS
from sglib.dataoverview.processing.derive.uk import PRODUCT_RUNNERS as UK_RUNNERS

pytestmark = pytest.mark.produce


def context(root: Path, country: str) -> CountryPipelineContext:
    return CountryPipelineContext(root.resolve(), {"country": {"code": country}})


def test_uk_context_reads_and_writes_two_roots_without_changing_demand(tmp_path):
    outputs = []
    for name, demand in (("first", 4.0), ("second", 19.0)):
        ctx = context(tmp_path / name, "uk")
        raw = ctx.raw_root / "HDRah-Data-PS-GB-1f63a32"
        raw.mkdir(parents=True)
        pd.DataFrame({"PS Name": ["Alpha", "Beta"], "Geo(Long,Lat)": ["0.25,0.25", "0.75,0.75"],
                      "Demand (MVA)": [demand, 2 * demand], "Firm Capacity (MVA)": [40.0, 50.0],
                      "RegName": ["North", "North"], "RegID": [1, 2]}).to_csv(raw / "GB_PS_data_extend.csv", index=False)
        derived = ctx.derived_root / "bplus"
        derived.mkdir(parents=True)
        regions = gpd.GeoDataFrame({"ITL3": ["r1"], "ITL2": ["r"], "area_crs": ["EPSG:27700"]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:4326")
        regions.to_file(derived / "regions.gpkg")
        station_path, region_path = UK_RUNNERS["substations"](ctx)
        actual = gpd.read_file(station_path)
        assert sorted(actual["Demand (MVA)"]) == [demand, 2 * demand]
        assert list(gpd.read_file(region_path)["Demand (MVA)"]) == [3 * demand]
        assert station_path.is_relative_to(ctx.repo_root)
        assert set(actual.capacity_basis) == {"firm_n1"}
        outputs.append(station_path)
    assert list(gpd.read_file(outputs[0])["Demand (MVA)"]) != list(gpd.read_file(outputs[1])["Demand (MVA)"])
    assert list(gpd.read_file(outputs[0])["Demand (MVA)"]) == [4.0, 8.0]


def test_au_runtime_paths_and_packaged_registry_are_independent(tmp_path):
    records = []
    for name, available in (("first", 5.0), ("second", 13.0)):
        ctx = context(tmp_path / name, "au")
        path = ctx.raw_root / "locations_sample/ausgrid_uhc_full_2025.geojson"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"features": [{"properties": {"substation": "Alpha", "available_capacity__load__at_n_": available, "voltage_level_primary": 33}, "geometry": {"coordinates": [151.0, -33.0]}}]}))
        _, coordinates = au.build_layer_index(ctx)
        assert coordinates == {"Alpha": (151.0, -33.0)}
        table = pd.DataFrame({"station": ["Alpha"], "status": ["usable"], "matched_name": ["Alpha"], "peak_mw": [2.0], "lon_wgs84": [151.0], "lat_wgs84": [-33.0], "peak_flagged": [False], "missing_rate": [0.0]})
        capacities, _ = au_ledger.build_firm_capacity(ctx, table)
        assert capacities.F_mva.tolist() == [available + 2.0]
        # Cached derived outputs must also use this context, not package location.
        output = ctx.derived_root / "a4_pv_screening.json"
        output.parent.mkdir(parents=True)
        output.write_text(json.dumps({"root": name}))
        assert au_pv.screen_pv(ctx) == output
        records.append(output)
    assert json.loads(records[0].read_text())["root"] == "first"
    assert json.loads(records[1].read_text())["root"] == "second"
    assert (au.REGISTRY_DIR / "a3_matching.json").is_file() or any(au.REGISTRY_DIR.glob("*.json"))
    assert not au.REGISTRY_DIR.is_relative_to(tmp_path)


def test_de_context_canonicalizes_each_existing_product_root(tmp_path):
    outputs = []
    for name, demand in (("first", 3.0), ("second", 17.0)):
        ctx = context(tmp_path / name, "de")
        ctx.derived_root.mkdir(parents=True)
        (ctx.derived_root / "nama_10r_3gva.xlsx").write_bytes(b"existing intermediate")
        gpd.GeoDataFrame({"Name": ["source"]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:4326").to_file(ctx.derived_root / "source_regions.gpkg")
        gpd.GeoDataFrame({"Kennzeichen": ["alpha"], "p_mw": [demand]}, geometry=[Point(.5, .5)], crs="EPSG:4326").to_file(ctx.derived_root / "substations.gpkg")
        _, station_path = DE_RUNNERS["substations"](ctx)
        actual = gpd.read_file(station_path)
        assert actual.p_mw.tolist() == [demand]
        assert actual.station_id.tolist() == ["de:alpha"]
        assert actual.capacity_basis.tolist() == ["not_applicable"]
        assert station_path.is_relative_to(ctx.repo_root)
        outputs.append(station_path)
    assert gpd.read_file(outputs[0]).p_mw.tolist() == [3.0]
    assert gpd.read_file(outputs[1]).p_mw.tolist() == [17.0]
