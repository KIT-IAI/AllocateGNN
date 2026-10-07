"""新增国别分析的合成消费合同；不读取正式预测或执行 HPC 作业。"""

from itertools import product

import numpy as np
import pandas as pd
import pytest

from sglib.analysis import c4, c5, c6

pytestmark = pytest.mark.consume


def test_c4_has_nine_fixed_members_and_does_not_duplicate_static_seeds():
    assert len(c4.definitions("MW")) == 9
    assert [d["metric"] for d in c4.definitions("MW")[:3]] == ["rmse", "WSD", "RSD_median"]
    spec = {"country": "au", "regions": ["r"], "methods": {"GPM": {"seeds": [None]}, "GNN": {"seeds": [42, 123, 456]}}}
    table = pd.DataFrame([{"region": "r", "candidate": candidate, "seed": seed, "status": "VALID", "value": value}
                          for candidate, seed, value in (("GPM", None, 2.), ("GNN", 42, 1.), ("GNN", 123, 2.), ("GNN", 456, 3.))])
    result = c4.summarize_realizations(table, spec, value_column="value", unit="MW")
    assert result.n_realizations.tolist() == [1, 3] and result.value.tolist() == [2., 2.]
    with pytest.raises(ValueError, match="缺失"):
        c4.summarize_realizations(table.iloc[:-1], spec, value_column="value", unit="MW")


def panel():
    return pd.DataFrame([{"country": "nz", "region": "r", "radius_km": radius, "lambda": load, "gnn_seed": seed,
                          "distribution": name, "cell_spearman": i/8, "aggregate_spearman": i/8,
                          "L_S_over_X": 0., "L_E_over_X": float(8-i), "ref_eligible": True, "scenario_defined": True}
                         for radius, load, seed in product((10., 20.), (.25, .5, 1.), (42, 123, 456)) for i, name in enumerate(c5.PANEL)])


def test_c5_constant_regret_remains_not_assessable_and_keeps_secondary_loss():
    result = c5.panel_associations(panel(), {"country": "nz", "regions": ["r"]})
    assert result[result.primary].status.eq("METRIC_NOT_ASSESSABLE").all()
    assert len(result[result.primary]) == 2
    assert result[result.loss.eq("L_E_over_X")].status.eq("VALID").all()
    assert result[result.loss.eq("L_E_over_X")].cell_rho.eq(-1.).all()
    assert set(result.distribution_count) == {8, 9}


def test_c5_missing_distribution_never_becomes_eight_distribution_primary():
    with pytest.raises(ValueError, match="缺件"):
        c5.panel_associations(panel().iloc[:-1], {"country": "nz", "regions": ["r"]})


def bounds():
    records = []
    for region, (candidate, seed), radius, load, eta in product(("a", "b"), (("Uni", None), ("GPM", None), ("GNN", 42), ("GNN", 123), ("GNN", 456)),
                                                              (10., 20.), (.25, .5, 1.), (.5, .8, .9, 1.)):
        records.append({"country": "uk", "region": region, "candidate": candidate, "seed": seed, "radius_km": radius,
                        "lambda": load, "eta": eta, "assessable": True, "budget_realized": region == "a", "conditional_violation": False,
                        "L_E_over_X": 0., "L_S_over_X": 0., "B_E_over_X": .2, "B_S_over_X": .4})
    return pd.DataFrame(records)


def test_c6_keeps_zero_all_breakpoints_and_both_denominators():
    result = c6.analyze(bounds(), {"country": "uk", "regions": ["a", "b"]})
    curves = result["tolerance_curves"]
    assert set(curves.tau_normalized) == {0., .2, .4}
    terminal = curves[curves.tau_normalized.gt(0)]
    assert terminal.conditional_share.eq(1.).all()
    assert terminal.all_registered_share.eq(.5).all()
    assert terminal.n_total.eq(2).all() and terminal.n_budget_realized.eq(1).all()
    assert result["bounds_audit"].actual_planner_tolerance.isna().all()


def test_c6_rejects_missing_matrix_and_conditional_violation():
    frame = bounds()
    with pytest.raises(ValueError, match="不完整"):
        c6.analyze(frame.iloc[:-1], {"country": "uk", "regions": ["a", "b"]})
    frame.loc[0, "conditional_violation"] = True
    with pytest.raises(ValueError, match="违规"):
        c6.analyze(frame, {"country": "uk", "regions": ["a", "b"]})
