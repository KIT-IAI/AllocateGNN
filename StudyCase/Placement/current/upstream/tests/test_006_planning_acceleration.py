"""可证行筛除必须包住原浮点 delta，并逐位保持原选址 / assignment。"""

import numpy as np
import pytest

from sglib.core.algorithms.pmedian import (
    _compute_nn_sn, _eval_all_swaps_for_jin, certified_swap_lower_bounds, solve_pmedian_greedy,
)

pytestmark = pytest.mark.consume


@pytest.mark.parametrize("scale", [1e-8, 1., 1e8])
def test_certificates_bound_original_floating_swap_rows(scale):
    rng = np.random.default_rng(42)
    for n, m, k in ((30, 25, 1), (60, 45, 7), (120, 100, 35)):
        distances = rng.uniform(0, 100, size=(n, m)).astype(np.float64) * scale
        weights = rng.random(n).astype(np.float64)
        weights /= weights.sum()
        selected = rng.choice(m, k, replace=False)
        unselected = np.array([i for i in range(m) if i not in selected])
        nearest, nn, _, sn = _compute_nn_sn(distances, selected)
        lower = certified_swap_lower_bounds(nearest, nn, sn, distances, weights, unselected, k)
        for j in range(k):
            exact = _eval_all_swaps_for_jin(j, unselected, nearest, nn, sn, distances, weights)
            assert np.all(lower[j] <= exact)


@pytest.mark.parametrize("seed", [0, 1, 42, 123, 456])
def test_pruning_preserves_unaccelerated_results_bitwise(seed):
    rng = np.random.default_rng(seed)
    points = rng.random((35, 2)) * [.7, .3] + [4., 52.]
    weights = rng.random(35)
    for k in (1, 4, 12):
        config = {"max_iter": 40, "random_restarts": 3, "restart_seed": 42}
        old = solve_pmedian_greedy(points, points, weights, k, {**config, "certified_row_pruning": False})
        new = solve_pmedian_greedy(points, points, weights, k, {**config, "certified_row_pruning": True})
        assert all(np.array_equal(a, b) for a, b in zip(old, new, strict=True))


def test_symmetric_ties_keep_original_argmin():
    xy = np.array([(x, y) for x in range(5) for y in range(5)], dtype=float) * .01 + [4., 52.]
    config = {"max_iter": 30, "random_restarts": 3, "restart_seed": 42}
    a = solve_pmedian_greedy(xy, xy, np.ones(25), 7, {**config, "certified_row_pruning": False})
    b = solve_pmedian_greedy(xy, xy, np.ones(25), 7, {**config, "certified_row_pruning": True})
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
