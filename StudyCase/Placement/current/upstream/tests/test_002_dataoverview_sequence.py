"""Gates for the numbered per-country DataOverview chain (01–06, five countries)."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

from sglib.dataoverview import stage  # noqa: E402
from sglib.dataoverview.overview import figures, tables  # noqa: E402

pytestmark = pytest.mark.gate

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "casestudy/1_DataOverview"
COUNTRIES = {"uk": "1_UK", "au": "2_AU", "de": "3_DE", "nl": "4_NL", "nz": "5_NZ"}

EXPECTED_CHAIN = [
    "01_download.py",
    "02_regions_stations.ipynb",
    "03_grid.ipynb",
    "04_features.py",
    "05_features_overview.ipynb",
    "06_inventory.ipynb",
]
NOTEBOOKS = [name for name in EXPECTED_CHAIN if name.endswith(".ipynb")]
FORBIDDEN_IN_NOTEBOOKS = ("to_file(", "to_csv(", "to_parquet(", "train(", "download(", "write_text(")


def _code(path: Path) -> str:
    document = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", [])) for cell in document["cells"] if cell["cell_type"] == "code"
    )


@pytest.mark.parametrize("country", sorted(COUNTRIES))
def test_chain_is_numbered_and_complete(country: str) -> None:
    folder = STAGE / COUNTRIES[country]
    numbered = sorted(path.name for path in folder.iterdir() if path.name[:2].isdigit())
    assert numbered == EXPECTED_CHAIN
    assert not list(folder.glob("0[1-7]_*_cuz.ipynb"))


@pytest.mark.parametrize("country", sorted(COUNTRIES))
def test_notebooks_use_stage_or_inventory_gate_and_keep_outputs_empty(country: str) -> None:
    folder = STAGE / COUNTRIES[country]
    for name in NOTEBOOKS:
        path = folder / name
        document = json.loads(path.read_text(encoding="utf-8"))
        code = _code(path)
        if name == "06_inventory.ipynb":
            assert "report = run_gate(context)" in code
            assert "raise SystemExit(1)" in code
            assert not any(token in code for token in ("run_step", "require_done", "load_bundle"))
        else:
            assert "stage.run_step(" in code or "stage.require_done(" in code
        assert f'COUNTRY = "{country}"' in code
        assert not any(token in code for token in FORBIDDEN_IN_NOTEBOOKS), name
        for cell in document["cells"]:
            if cell["cell_type"] == "code":
                assert cell.get("outputs", []) == [], f"{country}/{name} must keep outputs=[]"
    assert "figures_root" in _code(folder / "02_regions_stations.ipynb")


@pytest.mark.parametrize("country", sorted(COUNTRIES))
def test_scripts_delegate_to_stage(country: str) -> None:
    folder = STAGE / COUNTRIES[country]
    for name in ("01_download.py", "04_features.py"):
        source = (folder / name).read_text(encoding="utf-8")
        assert "stage.run_step(" in source
        assert f'COUNTRY = "{country}"' in source
    assert '["ntl"]' in (folder / "04_features.py").read_text(encoding="utf-8")


def test_navigation_lists_the_numbered_chain_for_every_country() -> None:
    root_readme = (ROOT / "README.md").read_text(encoding="utf-8")
    stage_readme = (STAGE / "README.md").read_text(encoding="utf-8")
    assert "casestudy/1_DataOverview/README.md" in root_readme
    for directory in COUNTRIES.values():
        for name in NOTEBOOKS:
            assert f"{directory}/{name}" in stage_readme
        assert f"{directory}/07_features_cuz.ipynb" not in stage_readme
    for name in EXPECTED_CHAIN:
        assert f"`{name}`" in stage_readme
    assert "general/00_data_matrix.ipynb" in stage_readme


def test_stage_selection_maps_members_to_units_and_matrix_has_no_dependencies() -> None:
    units, _ = stage.build_units(ROOT)
    assert units["general.matrix.overview"].depends_on == ()
    assert stage.select_units(units, None, ["matrix"]) == ["general.matrix.overview"]
    assert stage.select_units(units, "uk", ["regions", "substations"]) == [
        "uk.regions.derive",
        "uk.substations.derive",
    ]
    downloads = stage.select_units(units, "uk", ["download"])
    assert downloads and all(unit.endswith(".download") for unit in downloads)
    with pytest.raises(ValueError):
        stage.select_units(units, "uk", ["no_such_member"])


def test_stage_fails_closed_when_predecessors_are_missing() -> None:
    units, _ = stage.build_units(ROOT)
    fake = dict(units)
    fake["uk.features.derive"] = replace(units["uk.features.derive"], done_check=lambda: False)
    missing = stage._missing_predecessors(fake, ["uk.inventory.overview"])
    assert "uk.features.derive" in missing
    assert "uk.inventory.overview" not in missing


@pytest.mark.parametrize("country", sorted(COUNTRIES))
def test_ntl_download_depends_on_the_producer_of_its_bounds(country: str) -> None:
    from sglib.dataoverview.processing.registry import gee_query

    units, configurations = stage.build_units(ROOT)
    unit = units[f"{country}.ntl.download"]
    assert len(unit.depends_on) == 1 and unit.depends_on[0].endswith(".derive")
    config = configurations[country]
    query = gee_query(config, config["datasets"]["ntl"])
    assert query["bounds_source"]
    assert query["output_crs"]
    for unit_id in stage.select_units(units, country, ["download"]):
        if unit_id != unit.id:
            assert units[unit_id].depends_on == ()


def test_uk_gee_defaults_come_from_canonical_regions_and_native_crs() -> None:
    from sglib.dataoverview.processing.registry import gee_query

    _, configurations = stage.build_units(ROOT)
    config = configurations["uk"]
    query = gee_query(config, config["datasets"]["ntl"])
    assert query["bounds_source"] == "data/datasets/2_derived/uk/bplus/regions.gpkg"
    assert query["output_crs"] == "EPSG:27700"
    nl = configurations["nl"]
    assert gee_query(nl, nl["datasets"]["ntl"])["output_crs"] == "EPSG:4326"


def test_every_skip_states_its_reason() -> None:
    units, configurations = stage.build_units(ROOT)
    for unit in units.values():
        config = configurations.get(unit.country) if unit.country else None
        lines = stage.skip_reason(ROOT, unit, config)
        assert any(line.startswith("reason: ") for line in lines), unit.id
        if unit.step == "download":
            assert any(line.startswith("already downloaded: ") for line in lines), unit.id
        else:
            assert any(line.startswith("output: ") for line in lines), unit.id


def test_regions_and_stations_figure_draws_only_assigned_stations() -> None:
    import geopandas as gpd
    from shapely.geometry import Point, box

    regions = gpd.GeoDataFrame({"ITL2": ["A"]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:4326")
    stations = gpd.GeoDataFrame(
        {"ITL3": ["A1", None, "A2"]},
        geometry=[Point(0.2, 0.2), Point(5, 5), Point(0.8, 0.8)],
        crs="EPSG:4326",
    )
    fig = figures.plot_regions_and_stations(regions, stations, title="t", region_column="ITL3")
    labels = [text.get_text() for text in fig.axes[0].get_legend().get_texts()]
    assert all(len(label.split()) <= 3 for label in labels), labels
    title = fig.axes[0].get_title()
    assert "2 stations" in title and "1 outside regions omitted" in title, title
    fig.clear()
    zoom = figures.plot_group_zoom(regions, stations, group_column=None, group_value="all", region_column="ITL3")
    assert "2 stations" in zoom.axes[0].get_title()
    zoom.clear()


def test_display_helpers_use_the_required_sections_vocabulary() -> None:
    general = (STAGE / "general/general.toml").read_text(encoding="utf-8")
    assert "required_sections" in general
    config = stage.country_config(ROOT, "uk")
    ledger = tables.dataset_ledger(config.values, categories=["station_register"])
    assert list(ledger.columns) == ["dataset", "category", "vintage", "license", "acquisition", "semantics"]
    assert ledger["dataset"].tolist() == ["gb_ps_info"]
    facts = tables.unit_and_vintage(config)
    assert {"units", "temporal", "crs", "station_contract", "country"} <= set(facts["section"])
    assert callable(figures.save_figure) and callable(figures.plot_cell_categories)
