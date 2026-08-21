                                                                 
from __future__ import annotations

from itertools import product
import numpy as np
import pytest

from SpatialPlacement.core.bounds import (
    connection_cost,
    cost_interval,
    fixed_site_error_and_bound,
    selection_regret_and_rectangular_bound,
)
from SpatialPlacement.core.sizing import compute_sizing_metrics
from SpatialPlacement.core.stats import exact_sign_flip_p
from SpatialPlacement import reproduce_paper_numbers as paper


def test_exact_sign_flip_is_two_sided_and_scale_symmetric() -> None:
    ones = np.ones(16)
    assert exact_sign_flip_p(ones)["p"] == pytest.approx(2.0 / 2**16)
    large = np.arange(1.0, 17.0) * 14_000_000.0
    assert exact_sign_flip_p(large)["p"] == exact_sign_flip_p(-large)["p"]


def test_rsd_applies_the_same_rule_to_estimate_and_benchmark() -> None:
    predicted = np.array([80.0, 120.0])
    observed = np.array([100.0, 100.0])
    expected = np.array([20.0, 20.0])
    for gamma in (1.1, 1.5, 2.0):
        metrics = compute_sizing_metrics(
            gamma * predicted, observed, gamma=gamma
        )
        assert metrics["RSD_mean"] == pytest.approx(expected.mean())
        assert metrics["RSD_median"] == pytest.approx(np.median(expected))


def test_pf1_reinforcement_identity() -> None:
    measured_peak = np.array([20.0, 75.0, 130.0])
    available = np.array([10.0, 0.0, 55.0])
    firm = measured_peak + available
    for incoming in (100.0, 300.0, 500.0):
        full = connection_cost(measured_peak, firm, incoming)
        closed = np.maximum(0.0, incoming - available)
        np.testing.assert_allclose(full, closed, rtol=0.0, atol=1e-12)


def test_fixed_site_bound_contains_realised_error() -> None:
    observed_demand = np.array([40.0, 90.0, 160.0])
    estimated_demand = np.array([50.0, 75.0, 145.0])
    firm = np.array([120.0, 150.0, 210.0])
    estimated, lower, upper = cost_interval(
        estimated_demand, 20.0, firm, incoming_load=100.0
    )
    observed = connection_cost(observed_demand, firm, incoming_load=100.0)
    realised, bound = fixed_site_error_and_bound(observed, estimated, lower, upper)
    assert realised <= bound + 1e-12


def test_rectangular_selection_bound_matches_endpoint_bruteforce() -> None:
    observed = np.array([3.0, 7.0, 5.0, 11.0])
    estimated = np.array([4.0, 2.0, 8.0, 10.0])
    lower = np.array([1.0, 1.5, 4.0, 8.0])
    upper = np.array([6.0, 8.0, 9.0, 13.0])
    _realised, bound = selection_regret_and_rectangular_bound(
        observed, estimated, lower, upper, top_fraction=0.5
    )
    selected = np.argsort(estimated, kind="stable")[:2]
    worst = 0.0
    for choices in product((0, 1), repeat=len(lower)):
        costs = np.where(np.asarray(choices, bool), upper, lower)
        optimum = np.argsort(costs, kind="stable")[:2]
        worst = max(worst, float(costs[selected].mean() - costs[optimum].mean()))
    assert bound == pytest.approx(worst)


def test_manifest_and_headline_contracts() -> None:
    paper.verify_manifest()
    data = paper.reproduce()
    paper.verify_headlines(data)
    assert data["tasks"]["AU"]["connection"]["p_holm_four_module"] > 0.05
    assert data["tasks"]["AU"]["sizing"]["p_holm_four_module"] < 0.05


def test_result_allowlist_excludes_non_manuscript_artifacts() -> None:
    actual = {path.name for path in paper.RESULTS.iterdir() if path.is_file()}
    forbidden = {
        "au_vintage_recon.csv",
        "au_vintage_summary.json",
        "au_controls.csv",
        "uk_scale_grid.csv",
        "au_scale_grid.csv",
    }
    assert not actual.intersection(forbidden)
    assert len(actual) == 12
