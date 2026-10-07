from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pytest
from scipy.spatial.distance import cdist
from shapely.geometry import Point

from sglib.generator.config import GeneratorConfigError, load_generator_config
from sglib.generator.inputs import assemble_from_handoff, compute_proximity_scores

pytestmark = pytest.mark.consume


ROOT = Path(__file__).resolve().parents[1]
GENERAL = ROOT / "casestudy/2_Generator/general/generator.toml"


def test_chunked_proximity_matches_the_previous_full_matrix_formula() -> None:
    grid = gpd.GeoDataFrame(
        geometry=[Point(float(index), float(index % 7)) for index in range(211)],
        crs="EPSG:3857",
    )
    stations = gpd.GeoDataFrame(
        geometry=[Point(float(index * 11), float(index % 5)) for index in range(37)],
        crs="EPSG:3857",
    )
    observed, diagnostics = compute_proximity_scores(
        grid,
        stations,
        working_crs="EPSG:3857",
        gamma=2.0,
        clamp_km=0.01,
        max_workspace_bytes=4096,
        return_diagnostics=True,
    )
    grid_xy = np.column_stack([grid.geometry.x, grid.geometry.y])
    station_xy = np.column_stack([stations.geometry.x, stations.geometry.y])
    full = np.maximum(cdist(grid_xy, station_xy) / 1000.0, 0.01)
    expected = np.sum(full ** -2.0, axis=1)
    np.testing.assert_allclose(observed, expected, rtol=1e-14, atol=0.0)
    assert diagnostics["chunk_rows"] < len(grid)
    assert diagnostics["peak_workspace_bytes"] <= 4096


def test_generator_config_resolves_only_the_hash_guarded_shared_budget() -> None:
    loaded = load_generator_config(
        ROOT,
        GENERAL,
        ROOT / "casestudy/config/countries/nz.toml",
        ROOT / "casestudy/2_Generator/5_NZ/nz.toml",
    )
    authorities = loaded.values["authorities"]
    workspace = loaded.values["execution"]["distance_workspace"]
    assert set(authorities) >= {
        "engineering_admission",
        "engineering_admission_sha256",
    }
    assert workspace == {
        "strategy": "country_neutral_chunked_rows_v1",
        "max_workspace_mib": 128.0,
        "max_workspace_bytes": 128 * 2**20,
        "authority_sha256": authorities["engineering_admission_sha256"],
    }


def test_generator_config_fails_closed_on_engineering_authority_hash_drift(
    tmp_path: Path,
) -> None:
    payload = GENERAL.read_text(encoding="utf-8")
    marker = 'engineering_admission_sha256 = "'
    start = payload.index(marker) + len(marker)
    end = payload.index('"', start)
    drifted = payload[:start] + "0" * 64 + payload[end:]
    general = tmp_path / "generator.toml"
    general.write_text(drifted, encoding="utf-8")
    with pytest.raises(GeneratorConfigError, match="authority hash mismatch: engineering_admission"):
        load_generator_config(
            ROOT,
            general,
            ROOT / "casestudy/config/countries/nz.toml",
            ROOT / "casestudy/2_Generator/5_NZ/nz.toml",
        )


def test_assemble_uses_and_records_the_resolved_workspace_budget() -> None:
    source_key = "source_code"
    grid = gpd.GeoDataFrame(
        {source_key: ["S"] * 4},
        geometry=[Point(0, 0), Point(10, 0), Point(20, 0), Point(30, 0)],
        crs="EPSG:3857",
    )
    support = {
        "features": np.ones((4, 6), dtype=float),
        "covered_mask": np.ones(4, dtype=bool),
        "unknown_mask": np.zeros(4, dtype=bool),
        "zero_mask": np.zeros(4, dtype=bool),
        "built_fraction": np.ones(4, dtype=float),
        "source_key_order": np.asarray(["S"]),
        "source_keys": np.asarray(["S"] * 4),
    }
    region = SimpleNamespace(
        region="R",
        grid=grid,
        grid_metadata={"projected_step_m": 10.0, "target_ground_step_m": 10.0},
        landuse={},
        built_surface={},
        cuz_support=support,
        ntl={"data": np.ones((4, 1), dtype=float)},
    )
    handoff = SimpleNamespace(
        country="xx",
        profile=SimpleNamespace(
            station_contract={
                "source_key": source_key,
                "region_demand_column": "demand",
                "region_column": source_key,
            },
            crs={"working": "EPSG:3857"},
        ),
        regions_table=gpd.GeoDataFrame(
            {source_key: ["S"], "demand": [1.0]},
            geometry=[Point(0, 0)],
            crs="EPSG:3857",
        ),
        stations_table=gpd.GeoDataFrame(
            {source_key: ["S"], "station_id": ["T"]},
            geometry=[Point(0, 0)],
            crs="EPSG:3857",
        ),
        regions=(region,),
        evidence={},
    )
    budget = 64
    config = {
        "regions": ["R"],
        "corrections": {
            "rci_threshold": 0.5,
            "proximity_gamma": 2.0,
            "dist_clamp_km": 0.01,
        },
        "execution": {
            "distance_workspace": {
                "strategy": "country_neutral_chunked_rows_v1",
                "max_workspace_bytes": budget,
                "authority_sha256": "a" * 64,
            }
        },
    }
    bundle = assemble_from_handoff(handoff, config)
    proximity = bundle.region_inputs["R"].metadata["proximity"]
    assert proximity["max_workspace_bytes"] == budget
    assert proximity["peak_workspace_bytes"] <= budget
    assert proximity["authority_sha256"] == "a" * 64
