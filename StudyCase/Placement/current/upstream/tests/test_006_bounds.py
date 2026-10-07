"""C6 矩形集固定 k 上界的穷举验收。"""

from itertools import product

import numpy as np
import pytest

from sglib.experiment.conditional_bounds import loss_bounds
from sglib.experiment.connection_observations import rank_agreement, selection_losses
from sglib.experiment.connection_observations import compiled_members
from sglib.experiment.connection import neighbourhood_members, neighbourhood_sum

pytestmark = pytest.mark.consume


def test_rectangle_bound_equals_exhaustive_worst_regret_with_ties():
    lower = np.array([1., 2., 0., 3.])
    upper = np.array([5., 7., 8., 9.])
    estimate = np.array([3., 3., 4., 6.])
    worst = 0.
    for choice in product((0, 1), repeat=4):
        truth = np.where(choice, upper, lower)
        result = loss_bounds(truth, estimate, lower, upper, k=2)
        assert result["L_E"] <= result["B_E"] + 1e-12
        assert result["L_S"] <= result["B_S"] + 1e-12
        worst = max(worst, result["L_S"])
    assert worst == pytest.approx(result["B_S"])


def test_rank_keeps_zero_ties_and_constant_is_not_assessable():
    assert rank_agreement([0, 0, 1, 2], [0, 0, 2, 4]) == pytest.approx(1.)
    assert rank_agreement([0, 0, 0], [0, 1, 2]) is None
    values, selected = selection_losses([1., 1., 1., 1.], [0., 0., 0., 0.], [5., 5., 5., 5.], 1., k=2)
    assert selected.tolist() == [0, 1]
    assert values["cutoff_tie_size"] == 4 and values["L_S"] == 0.


def test_index_compilation_is_elementwise_and_sum_exact_equivalent():
    rng = np.random.default_rng(42)
    points, queries = rng.normal(size=(1000, 2)), rng.normal(size=(40, 2))
    original = neighbourhood_members(points, queries, 2.)
    compiled = compiled_members(points, queries, 2.)
    assert all(np.array_equal(a, b) for a, b in zip(original, compiled, strict=True))
    for values in (rng.random(1000), rng.random(1000) * 1e8, np.zeros(1000)):
        assert np.array_equal(neighbourhood_sum(values, original), neighbourhood_sum(values, compiled))
