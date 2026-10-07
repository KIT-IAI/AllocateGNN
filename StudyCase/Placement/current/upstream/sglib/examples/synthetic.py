"""A fully synthetic UK handoff and static Generator run; no network or models."""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, box

from sglib.core.infra.artifacts import atomic_json
from sglib.core.infra.terms import load_country_profile
from sglib.dataoverview.handoff import DataOverviewBundle, RegionData
from sglib.dataoverview.processing.config import CountryPipelineContext
from sglib.dataoverview.processing.derive.uk import PRODUCT_RUNNERS
from sglib.generator.config import load_generator_config, training_params
from sglib.generator.execution import prepare_inputs, generate_static_component
from .common import check_smoke_output, write_fixture_inventory


def run(repo_root: Path, output_root: Path, *, demand_scale: float = 1.0) -> Path:
    repository = Path(repo_root).resolve()
    output = Path(output_root).resolve()
    if not np.isfinite(demand_scale) or demand_scale <= 0:
        raise ValueError("demand_scale must be positive and finite")
    check_smoke_output(repository, output)
    context = CountryPipelineContext(output, {"country": {"code": "uk"}})
    raw = context.raw_root / "HDRah-Data-PS-GB-1f63a32"
    raw.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"PS Name": ["Alpha", "Beta"], "Geo(Long,Lat)": ["0.25,52.25", "0.75,52.25"],
                  "Demand (MVA)": [4 * demand_scale, 8 * demand_scale], "Firm Capacity (MVA)": [40.0, 50.0],
                  "RegName": ["Fixture", "Fixture"], "RegID": [1, 2]}).to_csv(raw / "GB_PS_data_extend.csv", index=False)
    derived = context.derived_root / "bplus"
    derived.mkdir(parents=True)
    regions = gpd.GeoDataFrame({"ITL3": ["fixture_1", "fixture_2"], "ITL2": ["Fixture", "Fixture"], "area_crs": ["EPSG:27700", "EPSG:27700"],
                               **{f"{sector}_percent": [.5, .5] for sector in ("residential", "commercial", "industrial", "agricultural", "others")}},
                              geometry=[box(0, 52, .5, 52.5), box(.5, 52, 1, 52.5)], crs="EPSG:4326")
    regions.to_file(derived / "regions.gpkg")
    stations_path, regions_path = PRODUCT_RUNNERS["substations"](context)
    stations = gpd.read_file(stations_path)
    regions = gpd.read_file(regions_path)
    grid = gpd.GeoDataFrame({"ITL3": np.repeat(["fixture_1", "fixture_2"], 4), "index_region": np.arange(8)},
                            geometry=[Point(x, y) for x in (.24, .26, .74, .76) for y in (52.24, 52.26)], crs="EPSG:4326")
    features = np.tile([.2, .2, .2, .2, .2, 0.0], (8, 1))
    region = RegionData("Fixture", grid,
                        {"schema_version": "sg_grid_bundle_v2", "projected_step_m": 100.0, "target_ground_step_m": 100.0, "source_key": "ITL3", "source_key_order": ["fixture_1", "fixture_2"], "fixture": True},
                        {"data": features[:, :5]}, {"data": np.ones((8, 1))},
                        {"features": features, "built_fraction": np.ones(8), "covered_mask": np.ones(8, bool), "unknown_mask": np.zeros(8, bool), "zero_mask": np.zeros(8, bool), "source_keys": grid.ITL3.to_numpy(dtype=str), "source_key_order": np.array(["fixture_1", "fixture_2"])},
                        {"data": np.arange(1, 9, dtype=float).reshape(-1, 1)})
    profile = load_country_profile(repository / "casestudy/config/countries/uk.toml")
    handoff = DataOverviewBundle("uk", profile, regions, stations, (region,), {"schema_version": "sg_dataoverview_inventory_v1", "formal": False, "synthetic": True}, {}, None)
    generator_root = output / "results/2_Generator/1_UK"
    handoff, inventory_path = write_fixture_inventory(handoff, generator_root)
    loaded = load_generator_config(repository, repository / "casestudy/2_Generator/general/generator.toml", profile.source_path, repository / "casestudy/2_Generator/1_UK/uk.toml")
    values = dict(loaded.values)
    values["regions"] = ["Fixture"]
    prepare_inputs(handoff, values, generator_root, worker_params=training_params(values), require_formal_evidence=False, inventory_path=inventory_path)
    for component in ("assignments", "uniform", "gpm", "proximity", "public_activity"):
        generate_static_component(generator_root, component)
    with np.load(generator_root / "static/uniform/Fixture.npz", allow_pickle=False) as field:
        total = float(field["data"].sum())
    expected = 12 * demand_scale
    if not np.isclose(total, expected, rtol=0, atol=1e-10):
        raise RuntimeError(f"synthetic field mass {total} differs from source demand {expected}")
    return atomic_json({"status": "PASS", "formal": False, "synthetic": True, "country": "uk", "n_cells": 8, "n_sources": 2, "n_stations": 2, "demand_total": total, "demand_scale": demand_scale, "training_executed": False, "hpc_submission_authorized": False}, output / "smoke_receipt.json")
