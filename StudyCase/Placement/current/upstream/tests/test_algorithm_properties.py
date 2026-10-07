"""Self-contained numerical properties of ported planning, weighting and CIVD kernels.

These fixtures replace the retired SpatialGranularity kernel tests: each
expected value follows from the documented formula, not from the old code.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point

from sglib.core.algorithms import planning_geometry as geometry
from sglib.core.algorithms import planning_metrics as metrics
from sglib.core.algorithms import planning_sizing as sizing
from sglib.core.algorithms.pmedian import assign_demand_points, solve_pmedian_greedy
from sglib.experiment.control_fields import block_permute, build_control_arms, smooth_multiplier
from sglib.generator.allocator.civd.clustering import hdbscan_station_clusters
from sglib.generator.weighter import weighter_registry
from sglib.generator.weighter.learned.common import kfold_splits
from sglib.generator.weighter.learned.models.gnn.graph import preprocess_features
from sglib.generator.weighter.learned.models.gnn.losses import CombinedLoss
from sglib.generator.weighter.native.equal import materialize_equal_grid
from sglib.generator.weighter.support import support_block_mass

pytestmark = pytest.mark.consume


ROOT = Path(__file__).resolve().parents[1]


def _generator_regions(country: str, directory: str) -> list[str]:
    with (ROOT / f"casestudy/2_Generator/{directory}/{country}.toml").open("rb") as stream:
        return tomllib.load(stream)["regions"]


def test_haversine_matrix_vector_and_chord_agree():
    a = np.array([[0.0, 50.0], [0.5, 50.5]])
    b = np.array([[1.0, 51.0], [0.0, 50.0]])
    matrix = geometry.haversine_distance_matrix(a, b)
    vector = geometry.haversine_vector(a[:, 0], a[:, 1], b[:, 0], b[:, 1])
    assert np.allclose(np.diag(matrix), vector)
    assert matrix[0, 1] == 0.0
    xa = geometry.to_ecef_km(a[:, 0], a[:, 1])
    xb = geometry.to_ecef_km(b[:, 0], b[:, 1])
    assert np.isclose(np.linalg.norm(xa[0] - xb[0]), geometry.chord_km(matrix[0, 0]), rtol=1e-12)


def test_block_permute_preserves_every_block_sum():
    rng = np.random.RandomState(42)
    xy = rng.rand(200, 2) * 10
    weights = rng.rand(200)
    out = block_permute(weights, xy, block_km=5.0, rng=np.random.RandomState(1))
    block = np.floor(xy[:, 0] / 5.0).astype(np.int64) * 1_000_003 + np.floor(xy[:, 1] / 5.0).astype(np.int64)
    for value in np.unique(block):
        assert np.isclose(out[block == value].sum(), weights[block == value].sum())
    assert not np.allclose(out, weights)


def test_smooth_multiplier_is_positive_and_deterministic():
    xy = np.random.RandomState(0).rand(50, 2) * 100
    first = smooth_multiplier(xy, 50.0, 0.5, np.random.RandomState(7))
    second = smooth_multiplier(xy, 50.0, 0.5, np.random.RandomState(7))
    assert (first > 0).all() and np.array_equal(first, second)


def test_control_arm_family_is_complete_deterministic_and_mass_preserving():
    rng = np.random.RandomState(4)
    xy = rng.rand(300, 2) * 100
    field = rng.rand(300)
    first = build_control_arms(field, xy, 10.0, seed=42)
    second = build_control_arms(field, xy, 10.0, seed=42)
    assert set(first) == {"PERM-R", "PERM-3R", "PERM-R/3", "SMOOTH-0.5", "SMOOTH-1", "SMOOTH-2"}
    for name, values in first.items():
        assert np.array_equal(values, second[name])
        assert np.isclose(values.sum(), field.sum())
        assert (values >= 0).all()


def test_pmedian_small_instance_is_optimal_and_deterministic():
    demand = np.array([[0.0, 50.0], [0.05, 50.0], [-0.05, 50.0], [2.0, 50.0], [2.05, 50.0], [1.95, 50.0]])
    facilities = np.array([[0.0, 50.0], [2.0, 50.0], [1.0, 50.0], [0.5, 50.5]])
    weights = np.ones(6) / 6
    config = {"max_iter": 50, "random_restarts": 2}
    first = solve_pmedian_greedy(demand, facilities, weights, 2, config)
    second = solve_pmedian_greedy(demand, facilities, weights, 2, config)
    assert set(first[0]) == {0, 1}
    assert np.array_equal(first[0], second[0]) and np.array_equal(first[1], second[1])
    assert np.array_equal(np.sort(np.unique(first[1])), np.sort(first[0]))


def test_pmedian_edge_cases():
    demand = np.array([[0.0, 50.0], [1.0, 50.0]])
    selected, _ = solve_pmedian_greedy(demand, demand, np.ones(2), 5, {})
    assert set(selected) == {0, 1}
    none, assignment = solve_pmedian_greedy(demand, demand, np.ones(2), 0, {})
    assert len(none) == 0 and assignment.tolist() == [0, 0]
    assert assign_demand_points(demand, demand).tolist() == [0, 1]


def test_substation_demand_prediction_conserves_total():
    assignment = np.array([0, 0, 1, 1, 1])
    weights = np.array([0.1, 0.1, 0.2, 0.3, 0.3])
    demand = sizing.predict_substation_demand(assignment, weights, 100.0, 2)
    assert np.allclose(demand, [20.0, 80.0]) and np.isclose(demand.sum(), 100.0)
    with pytest.raises(ValueError):
        sizing.predict_substation_demand(assignment, weights, 0.0, 2)


def test_recommended_capacity_rounds_up_to_the_ladder():
    ladder = [15.0, 30.0, 60.0]
    assert sizing.recommend_capacity(np.array([10.0, 33.0]), 1.5, discretise_to=ladder).tolist() == [15.0, 60.0]
    assert sizing.recommend_capacity(np.array([100.0]), 1.5, discretise_to=ladder).tolist() == [180.0]


def test_sizing_match_and_metric_identities():
    recommended = np.array([[0.1, 50.0], [1.1, 50.0]])
    stations = np.array([[0.1, 50.0], [1.1, 50.0]])
    actual, firm, _distance, too_far = sizing.match_to_real_substations(
        recommended, stations, np.array([40.0, 55.0]), np.array([60.0, 80.0])
    )
    assert np.allclose(actual, [40.0, 55.0]) and not too_far.any()
    result = sizing.compute_sizing_metrics(actual * 1.5, actual, firm)
    assert result["n_matched"] == 2
    assert np.isclose(result["TUR_mean"], 100.0 / 1.5)
    assert np.isclose(result["CE_mean"], 50.0)
    assert result["UPR"] == 0.0 and result["OPR"] == 0.0


def test_unique_matching_reports_collision_rate():
    recommended = np.array([[0.1, 50.0], [0.1, 50.001]])
    stations = np.array([[0.1, 50.0], [5.0, 55.0]])
    *_, collision = sizing.match_to_real_substations_unique(
        recommended, stations, np.array([40.0, 55.0]), np.array([60.0, 80.0])
    )
    assert collision == 0.5


def test_siting_and_sizing_metric_toys():
    points = np.array([[0.0, 0.0], [1.0, 0.0]])
    siting = metrics.compute_all_siting_metrics(points, points, np.array([0, 1]), np.ones(2), dcr_radii=[5.0])
    assert siting["WSD"] == pytest.approx(0.0)
    assert siting["LBI"] == pytest.approx(0.0)
    assert siting["DCR_5km"] == pytest.approx(1.0)
    config = {"safety_margin": 1.2, "standard_sizes_mva": [10, 20, 40]}
    predicted, actual = np.array([10.0, 30.0]), np.array([12.0, 35.0])
    _tur, recommended = metrics.compute_tur(predicted, actual, config)
    assert list(recommended) == [20.0, 40.0]
    sizing_metrics = metrics.compute_all_sizing_metrics(predicted, actual, config)
    assert sizing_metrics["UPR"] == 0.0 and sizing_metrics["OPR"] == 0.0
    firm = metrics.compute_all_sizing_metrics(predicted, actual, config, firm_capacity_mva=np.array([20.0, 63.0]))
    assert firm["FC_match_rate"] == pytest.approx(0.5)


def test_kfold_splits_are_frozen_complete_and_deterministic():
    uk = _generator_regions("uk", "1_UK")
    au = _generator_regions("au", "2_AU")
    assert kfold_splits(uk, 42, 4)[0][1] == ["London", "TLH2", "TLF2", "TLD3"]
    assert kfold_splits(au, 42, 4)[0][1] == ["Central_Coast", "Sydney_Parramatta", "Sydney_Ryde"]
    for regions in (uk, au):
        for seed in (42, 123, 456):
            splits = kfold_splits(regions, seed, 4)
            assert splits == kfold_splits(regions, seed, 4)
            assert sorted(region for _train, test in splits for region in test) == sorted(regions)


def _support_fixture():
    grid = gpd.GeoDataFrame(
        {
            "ITL3": ["A", "A", "A", "B", "B", "B"],
            "covered_mask": [True, True, False, True, False, False],
            "unknown_mask": [False, False, True, False, True, False],
            "built_fraction": [1.0, 1.0, 2.0, 1.0, 3.0, 0.0],
            "lu_residential_prop": [0.8, 0.2, 0.0, 0.1, 0.0, 0.0],
            "lu_commercial_prop": [0.2, 0.8, 0.0, 0.9, 0.0, 0.0],
        },
        geometry=[Point(float(i), 0.0) for i in range(6)],
        crs="EPSG:4326",
    )
    sources = gpd.GeoDataFrame(
        {
            "ITL3": ["A", "B"],
            "Demand (MVA)": [40.0, 60.0],
            "residential_percent": [0.75, 0.25],
            "commercial_percent": [0.25, 0.75],
        },
        geometry=[Point(0, 0), Point(1, 0)],
        crs="EPSG:4326",
    )
    return grid, sources


def test_uniform_is_source_conserving_on_cuz_support():
    grid, sources = _support_fixture()
    field = weighter_registry.create("uniform").compute(grid, sources, source_gdf=sources).weights
    assert np.allclose(field, [10.0, 10.0, 20.0, 15.0, 45.0, 0.0])
    assert field.sum() == 100.0


def test_gpm_reorders_covered_cells_but_keeps_uniform_block_mass():
    grid, sources = _support_fixture()
    uniform = weighter_registry.create("uniform").compute(grid, sources, source_gdf=sources).weights
    gpm = weighter_registry.create(
        "gpm",
        {
            "mode": "proportional",
            "proportion_columns": ["lu_residential_prop", "lu_commercial_prop"],
            "source_feature_columns": ["residential_percent", "commercial_percent"],
        },
    ).compute(grid, sources, source_gdf=sources).weights
    keys = grid["ITL3"].to_numpy()
    covered, unknown = grid["covered_mask"].to_numpy(), grid["unknown_mask"].to_numpy()
    uniform_mass = support_block_mass(uniform, keys, covered, unknown)
    gpm_mass = support_block_mass(gpm, keys, covered, unknown)
    assert uniform_mass.keys() == gpm_mass.keys()
    assert np.allclose(list(uniform_mass.values()), list(gpm_mass.values()))
    assert gpm[5] == 0.0
    assert not np.allclose(gpm[:2], uniform[:2])


def test_equal_grid_splits_block_mass_equally_across_active_vd_cells():
    grid = gpd.GeoDataFrame(
        {
            "SRC": ["S"] * 5,
            "covered_mask": [True, True, True, False, False],
            "unknown_mask": [False, False, False, True, False],
            "zero_mask": [False, False, False, False, True],
        },
        geometry=[Point(i, 0) for i in range(5)],
        crs="EPSG:3857",
    )
    uniform = np.array([2.0, 2.0, 2.0, 2.0, 0.0])
    result = materialize_equal_grid(grid, uniform, np.array([0, 0, 1, 0, 1]), source_column="SRC")
    assert np.allclose(result, [1.5, 1.5, 3.0, 2.0, 0.0])
    assert result.sum() == uniform.sum() and result[-1] == 0.0


def test_hdbscan_station_clusters_are_deterministic_and_contiguous():
    stations = gpd.GeoDataFrame(
        geometry=[Point(0, 0), Point(0.001, 0), Point(1, 1), Point(1.001, 1)], crs="EPSG:4326"
    )
    first = hdbscan_station_clusters(stations, min_cluster_size=2)
    second = hdbscan_station_clusters(stations, min_cluster_size=2)
    assert np.array_equal(first.labels, second.labels)
    assert set(first.labels) == set(range(first.n_clusters))


def test_fixed_onehot_schema_keeps_absent_categories():
    columns = ["lu_residential_prop", "lu_commercial_prop", "lu_industrial_prop", "lu_agricultural_prop", "lu_others_prop"]
    categories = ["residential", "commercial", "industrial", "agricultural", "others"]
    present = categories[:-1]
    values = np.eye(5)[: len(present)]
    frame = gpd.GeoDataFrame(
        {**{column: values[:, i] for i, column in enumerate(columns)}, "landuse": present},
        geometry=[Point(i, 0) for i in range(len(present))],
        crs="EPSG:3857",
    )
    result = preprocess_features(
        frame, numerical_col_names_all=columns, categorical_col_members_all={"landuse": categories}
    )
    features = result["final_features"]
    assert features.shape == (len(present), 10)
    np.testing.assert_array_equal(features.iloc[:, 5:9].to_numpy(), np.eye(4))
    np.testing.assert_array_equal(features.iloc[:, 9].to_numpy(), np.zeros(4))
    assert result["mapping"]["landuse"] == (5, 10, "categorical")


def test_combined_loss_accepts_the_active_landuse_prediction_loss():
    combined = CombinedLoss({"landuse_prediction_loss": 1.0}, learnable=False)
    assert set(combined.loss_functions) == {"landuse_prediction_loss"}
