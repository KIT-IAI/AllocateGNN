from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point

from sglib.generator.materialize.idr import (
    materialize_idr_fixed,
    materialize_idr_matched,
    public_activity_field,
)

pytestmark = pytest.mark.consume


PUBLIC_COLUMNS = (
    "residential_percent",
    "commercial_percent",
    "industrial_percent",
    "agricultural_percent",
    "others_percent",
)


def _sources(demands: tuple[float, ...]) -> gpd.GeoDataFrame:
    count = len(demands)
    data: dict[str, list[float] | list[str]] = {
        "source": [chr(ord("A") + index) for index in range(count)],
        "Demand (MVA)": list(demands),
    }
    for column in PUBLIC_COLUMNS:
        data[column] = [
            (index + 1) / (10.0 if column == "residential_percent" else 20.0)
            for index in range(count)
        ]
    return gpd.GeoDataFrame(
        data,
        geometry=[Point(float(index), 0.0) for index in range(count)],
        crs="EPSG:3857",
    )


def _geometry() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, np.ndarray]:
    grid = gpd.GeoDataFrame(
        geometry=[Point(float(index), 0.0) for index in range(4)],
        crs="EPSG:3857",
    )
    stations = gpd.GeoDataFrame(
        {"station_id": ["S"]}, geometry=[Point(1.5, 0.0)], crs="EPSG:3857"
    )
    return grid, stations, np.zeros(4, dtype=np.int64)


def test_public_activity_mixed_sources_keeps_zero_demand_strictly_zero() -> None:
    sources = _sources((10.0, 0.0))
    # A's third cell represents exact-zero (Z) support in its otherwise
    # positive-demand block; B is an entirely zero-demand source.
    keys = np.asarray(["A", "A", "A", "B", "B"])
    stored = np.asarray([7.5, 2.5, 0.0, 0.0, 0.0])

    field, invariance = public_activity_field(
        stored, keys, sources, source_column="source"
    )

    activity_a = float(sources.loc[0, list(PUBLIC_COLUMNS)].sum())
    np.testing.assert_allclose(
        field[:3], activity_a * np.asarray([0.75, 0.25, 0.0])
    )
    np.testing.assert_array_equal(field[3:], np.zeros(2))
    assert field[2] == 0.0
    assert field[keys == "A"].sum() == pytest.approx(activity_a)
    assert field[keys == "B"].sum() == 0.0
    assert invariance <= 1e-12


@pytest.mark.parametrize(
    ("demands", "stored", "message"),
    (
        ((0.0,), np.asarray([1e-30, 0.0]), "source conservation"),
        ((1.0,), np.asarray([0.0, 0.0]), "source conservation"),
    ),
)
def test_public_activity_rejects_nonconserving_zero_boundaries(
    demands: tuple[float, ...], stored: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        public_activity_field(
            stored,
            np.asarray(["A", "A"]),
            _sources(demands),
            source_column="source",
        )


def test_all_zero_sources_are_legal_public_field_but_idr_fails_closed() -> None:
    field, invariance = public_activity_field(
        np.zeros(4),
        np.asarray(["A", "A", "B", "B"]),
        _sources((0.0, 0.0)),
        source_column="source",
    )
    np.testing.assert_array_equal(field, np.zeros(4))
    assert invariance == 0.0

    grid, stations, canonical = _geometry()
    with pytest.raises(ValueError, match="zero total mass; refusing to fabricate"):
        materialize_idr_fixed(
            grid,
            stations,
            field,
            canonical,
            working_crs="EPSG:3857",
            b_tv=0.10,
        )
    with pytest.raises(ValueError, match="zero total mass; refusing to fabricate"):
        materialize_idr_matched(
            grid,
            stations,
            {("candidate", 42): field},
            canonical,
            working_crs="EPSG:3857",
            b_tv=0.10,
        )


def test_mixed_zero_nonzero_field_passes_fixed_and_matched_gates() -> None:
    field, _ = public_activity_field(
        np.asarray([7.5, 2.5, 0.0, 0.0]),
        np.asarray(["A", "A", "B", "B"]),
        _sources((10.0, 0.0)),
        source_column="source",
    )
    grid, stations, canonical = _geometry()

    fixed = materialize_idr_fixed(
        grid,
        stations,
        field,
        canonical,
        working_crs="EPSG:3857",
        b_tv=0.10,
    )
    matched = materialize_idr_matched(
        grid,
        stations,
        {("candidate", 42): np.asarray([3.0, 7.0, 0.0, 0.0])},
        canonical,
        working_crs="EPSG:3857",
        b_tv=0.10,
    )[("candidate", 42)]

    for result in (fixed, matched):
        assert result["g0_pass"] is True
        assert result["g1_pass"] is True
        assert result["selected_mode"] == "matched_idr"
        assert result["tv_mass"] == 0.0
        assert result["canonical_total_mass"] > 0.0
        assert result["raw_total_mass"] == pytest.approx(
            result["canonical_total_mass"]
        )
