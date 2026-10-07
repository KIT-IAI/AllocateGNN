from __future__ import annotations
from pathlib import Path
import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point
import torch
from sglib.generator.allocator.civd import influence_matrix
from sglib.generator.allocator.idr import allocate_idr_g01
from sglib.generator.allocator.vd import compute_vd_assignment
from sglib.generator.weighter.correction import apply_additive, apply_standard_multiplicative, compute_factors
from sglib.generator.weighter.learned.models.gnn.losses import NTLPriorLoss, ProximityPriorLoss
from sglib.generator.weighter.learned.training.policy import source_mean, uniform_region_mean, validate_region_schedule
pytestmark = pytest.mark.consume
ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "x,activity,budget,canonical,expected,mode,reason,tv_mass",
    [
        pytest.param(
            [0, 1, 9], [1, 2, 3], 0.10, [0, 0, 1], [0, 0, 1],
            "matched_idr", "", 0.0, id="unchanged-allocation",
        ),
        pytest.param(
            [0, 4.9, 5.1, 10], [100, 100, 1, 1], 0.10,
            [0, 0, 1, 1], [0, 0, 1, 1], "canonical_vd",
            "g1_transport_budget_exceeded", 100 / 202, id="budget-rejected",
        ),
        pytest.param(
            [0, 4.9, 5.1, 10], [100, 100, 1, 1], 0.50,
            [0, 0, 1, 1], [0, 1, 1, 1], "matched_idr", "",
            100 / 202, id="budget-accepted",
        ),
        pytest.param(
            [0, 1], [1, 2], 0.10, [0, 0], [0, 0],
            "canonical_vd", "g0_failed", 0.0, id="empty-station",
        ),
    ],
)
def test_vd_and_idr_have_self_contained_regression_cases(
    x, activity, budget, canonical, expected, mode, reason, tv_mass,
) -> None:
    grid = gpd.GeoDataFrame(geometry=[Point(value, 0) for value in x], crs="EPSG:3857")
    stations = gpd.GeoDataFrame(geometry=[Point(0, 0), Point(10, 0)], crs="EPSG:3857")
    assignment = compute_vd_assignment(grid, stations, config={"working_crs": "EPSG:3857"})
    assert assignment.tolist() == canonical
    grid_xy = np.array([[value, 0.0] for value in x], dtype=np.float64)
    station_xy = np.array([[0.0, 0.0], [10.0, 0.0]])
    result = allocate_idr_g01(
        grid_xy, station_xy, np.array(activity, dtype=np.float64), assignment,
        transport_budget=budget,
    )
    assert result.assignment.tolist() == expected
    assert result.selected_mode == mode
    assert result.fallback_reason == reason
    # Reassigning the second cell moves 100 of 202 units of public mass.
    assert result.tv_mass == pytest.approx(tv_mass, rel=0, abs=1e-15)
    assert result.canonical_total_mass == pytest.approx(sum(activity))
    assert result.raw_total_mass == pytest.approx(sum(activity))


def test_civd_influence_is_deterministic() -> None:
    grid = np.array([[0.0, 0.0], [5.0, 0.0]])
    targets = np.array([[0.0, 0.0], [10.0, 0.0]])
    first = influence_matrix(grid, targets, np.array([0, 1]), np.array([1.0, 2.0]))
    second = influence_matrix(grid, targets, np.array([0, 1]), np.array([1.0, 2.0]))
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])


def _correction_fixture() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "ITL3": ["a", "a", "b", "b"],
            "lu_residential_prop": [1.0, 0.0, 0.8, 0.0],
            "lu_commercial_prop": [0.0, 0.0, 0.0, 0.0],
            "lu_industrial_prop": [0.0, 0.0, 0.0, 0.0],
            "covered_mask": [True, True, True, True],
            "unknown_mask": [False, False, False, False],
        },
        geometry=[Point(i, 0) for i in range(4)],
        crs="EPSG:3857",
    )


def test_correction_factors_follow_the_per_source_log_ratio_formula() -> None:
    grid = _correction_fixture()
    ntl = np.array([1.0, 2.0, 4.0, 8.0])
    proximity = np.array([2.0, 3.0, 5.0, 7.0])
    ntl_factor, proximity_factor = compute_factors(grid, ntl, proximity, source_key="ITL3")
    # Only rows 0 and 2 are RCI (> 0.5), so each source's epsilon and median come from one cell:
    # source a: epsilon = median = 1; source b: epsilon = median = 4.
    expected_ntl = [np.log(3) / np.log(2), np.log(4) / np.log(2), np.log(9) / np.log(5), np.log(13) / np.log(5)]
    expected_proximity = [1.0, np.log(4) / np.log(3), 1.0, np.log(8) / np.log(6)]
    np.testing.assert_allclose(ntl_factor, expected_ntl, rtol=1e-12)
    np.testing.assert_allclose(proximity_factor, expected_proximity, rtol=1e-12)


def test_post_and_additive_corrections_conserve_source_block_mass() -> None:
    grid = _correction_fixture()
    base = np.array([2.0, 3.0, 5.0, 7.0])
    ntl_factor, proximity_factor = compute_factors(
        grid, np.array([1.0, 2.0, 4.0, 8.0]), np.array([2.0, 3.0, 5.0, 7.0]), source_key="ITL3"
    )
    sources = grid["ITL3"].to_numpy()
    for factor in (ntl_factor, proximity_factor, ntl_factor * proximity_factor):
        post = apply_standard_multiplicative(base, factor, grid, source_key="ITL3")
        add = apply_additive(base, factor, grid, source_key="ITL3")
        for source in ("a", "b"):
            block = sources == source
            assert post[block].sum() == pytest.approx(base[block].sum(), rel=1e-12)
            assert add[block].sum() == pytest.approx(base[block].sum(), rel=1e-12)
            raw = base[block] * factor[block]
            np.testing.assert_allclose(post[block], base[block].sum() * raw / raw.sum(), rtol=1e-12)
        assert (add >= 0).all()
    constant = np.full(4, 3.0)
    np.testing.assert_allclose(apply_standard_multiplicative(base, constant, grid, source_key="ITL3"), base, rtol=1e-12)
    np.testing.assert_allclose(apply_additive(base, constant, grid, source_key="ITL3"), base, rtol=1e-12)


def _forward_kl(weights: np.ndarray, sources: np.ndarray, values: np.ndarray) -> float:
    prior = np.log1p(values)
    per_source = []
    for source in np.unique(sources):
        block = sources == source
        target = prior[block] / prior[block].sum()
        per_source.append(float(np.sum(target * np.log(target / weights[block]))))
    return float(np.mean(per_source))


@pytest.mark.parametrize(
    "loss_class,metadata_key",
    [(NTLPriorLoss, "agent_ntl"), (ProximityPriorLoss, "agent_proximity")],
)
def test_registered_prior_losses_are_per_source_forward_kl(loss_class, metadata_key) -> None:
    weights = np.array([0.6, 0.4, 0.3, 0.7])
    sources = np.array([0, 0, 1, 1])
    values = np.array([1.0, 2.0, 3.0, 4.0])
    edges = torch.tensor(np.vstack([sources, np.arange(4)]))
    metadata = {metadata_key: torch.tensor(values, dtype=torch.float32), "agent_rci_mask": torch.ones(4, dtype=torch.bool), "num_s": 2}
    loss = loss_class()(torch.tensor(weights, dtype=torch.float32), edges, metadata)
    assert float(loss) == pytest.approx(_forward_kl(weights, sources, values), rel=1e-5)
    target = np.concatenate([np.log1p(values[:2]) / np.log1p(values[:2]).sum(), np.log1p(values[2:]) / np.log1p(values[2:]).sum()])
    assert float(loss_class()(torch.tensor(target, dtype=torch.float32), edges, metadata)) == pytest.approx(0.0, abs=1e-6)
    with pytest.raises(ValueError):
        loss_class()(torch.tensor(weights, dtype=torch.float32), edges, {"num_s": 2})


def test_weighting_policy_is_cell_and_region_size_invariant() -> None:
    values = np.array([1.0, 3.0, 10.0])
    sources = np.array([0, 0, 1])
    replicated_values = np.repeat(values, 5)
    replicated_sources = np.repeat(sources, 5)
    assert source_mean(values, sources) == pytest.approx(source_mean(replicated_values, replicated_sources))
    assert uniform_region_mean(np.array([2.0, 8.0])) == 5.0
    validate_region_schedule(["a", "b"], ["a", "b"])
    with pytest.raises(ValueError):
        validate_region_schedule(["a", "b"], ["a", "a"])
