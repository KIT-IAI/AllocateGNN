"""006 W0：固定候选、负荷、单位与 Ref 准入的合成合同测试。"""

import numpy as np
import pytest

from sglib.experiment.connection import (
    ConnectionRegion, neighbourhood_members, reference_field,
    reference_surface, scenario_preflight, stride_candidates,
)

pytestmark = pytest.mark.consume


def region(capacity=(10., 20.), demand=(2., 4.)):
    xy = np.array([[0., 0.], [100., 0.], [20000., 0.], [20100., 0.]])
    return ConnectionRegion("fixture", "r", "kW", "EPSG:28992", "nominal_normal_state",
                            xy, xy[[0, 2]], np.array(["a", "b"]), np.array(demand),
                            np.array(capacity), np.array([0, 0, 1, 1]))


def test_stride_keeps_raw_grid_order_and_refuses_short_pool():
    assert np.array_equal(stride_candidates(9, 4), [0, 2, 4, 6])
    with pytest.raises(ValueError, match="不足"):
        stride_candidates(1999)


def test_closed_radius_and_reference_conservation():
    assert neighbourhood_members([[0., 0.], [3., 4.]], [[0., 0.]], 5.) == [[0, 1]]
    assert np.array_equal(reference_field([0, 0, 1], [4., 3.]), [2., 2., 3.])
    with pytest.raises(ValueError, match="无支持"):
        reference_field([0, 0], [4., 3.])


def test_x_uses_r0_for_both_radii_and_native_unit_is_preserved():
    source = region()
    surface, members = reference_surface(source, (10., 30.), 4)
    result = scenario_preflight(source, surface)
    assert surface.candidate_pool_hash.nunique() == 1
    assert len(members) == 12
    assert result.unit.eq("kW").all()
    for load, group in result.groupby("lambda"):
        assert np.allclose(group.X, load * 15.)
    assert result.ref_eligible.all()
    assert result.c4_c5_expected_eligible.all()


def test_ref_failure_retains_c6_and_all_rows_without_epsilon():
    source = region()
    surface, _ = reference_surface(source, (10., 30.), 4)
    surface["A_ref"] = surface.G * 2.
    result = scenario_preflight(source, surface)
    assert len(result) == 6
    assert not result.ref_eligible.any()
    assert not result.c4_c5_expected_eligible.any()
    assert result.c6_expected_eligible.all()
    assert result.ref_reason.eq("REF_SCORE_ABOVE_THRESHOLD").all()


def test_zero_capacity_or_demand_is_an_explicit_scientific_state():
    source = region(capacity=(0., 0.), demand=(0., 0.))
    surface, _ = reference_surface(source, (10., 30.), 4)
    result = scenario_preflight(source, surface)
    assert not result.scenario_defined.any()
    assert not result.ref_eligible.any()
    assert result.ref_score.isna().all()
    assert not result.c6_expected_eligible.any()
    assert result.ledger_zero_q_fraction.eq(1.).all()
