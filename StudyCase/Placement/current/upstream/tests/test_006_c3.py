"""C3 不按回退标签强制零差，五对比与量纲独立验证。"""

from types import SimpleNamespace

import numpy as np
import pytest

from sglib.experiment.allocator_observations import observe
from sglib.analysis.c3 import analyze

pytestmark = pytest.mark.consume


def test_matched_fallback_can_have_nonzero_difference_against_fixed():
    regions, fixed, matched = [], [], []
    names = ["r0", "r1", "r2"]
    for name in names:
        fields = [SimpleNamespace(label=label, seed=42, fold=1, qa_only=False, values=np.array(values), lineage={"sha256": "a" * 64})
                  for label, values in (("MLP", [2., 2.]), ("GNN", [1., 3.]))]
        regions.append(SimpleNamespace(region=name, country="fixture", unit="kW", observed=np.array([1., 3.]), assignment=np.array([0, 1]),
                       station_ids=np.array(["a", "b"]), station_sources=np.array(["s", "s"]), fields=fields, input_fingerprint="b" * 64, map_selected=False))
        gate = {"selected_mode": "matched_idr", "sha256": "c" * 64, "g0_pass": True, "g1_pass": True,
                "source_total_basis": "public_activity", "transport_budget": .1, "tv_mass": 0.}
        fixed.append(SimpleNamespace(region=name, assignment=np.array([1, 0]), raw_assignment=np.array([1, 0]), gate=gate))
        for label in ("MLP", "GNN"):
            matched.append(SimpleNamespace(region=name, candidate=label, seed=42, assignment=np.array([0, 1]), raw_assignment=np.array([1, 0]),
                           gate={**gate, "g0_pass": False, "selected_mode": "fallback_vd", "candidate_sha256": "a" * 64}))
    canonical = [SimpleNamespace(region=r.region, assignment=r.assignment, target_ids=r.station_ids, lineage={"sha256": "b" * 64}) for r in regions]
    tables = observe(regions, fixed, matched, [], canonical)
    spec = {"country": "fixture", "unit": "kW", "regions": names, "methods": {m: {"seeds": [42]} for m in ("MLP", "GNN")},
            "queen_adjacency": np.zeros((3, 3), bool), "blocks": [[i] for i in range(3)]}
    result = analyze(tables, spec)
    assert len(result["contrasts"]) == 5
    assert result["contrasts"].iloc[2].effect == 2.
    assert result["contrasts"].iloc[4].effect == -2.
    assert tables["gates"].query("allocator == 'IDR-matched'").fallback.all()
    assert result["contrasts"].iloc[2].relative_pct is None or np.isnan(result["contrasts"].iloc[2].relative_pct)


def _civd_observation_inputs(station_cluster):
    values = np.array([2., 4., 3., 5., 12., 7.])
    field = SimpleNamespace(label="GNN", seed=42, fold=1, qa_only=False, values=values,
                            lineage={"sha256": "a" * 64})
    region = SimpleNamespace(region="r0", country="fixture", unit="kW", observed=np.array([9., 2., 4., 6., 9., 0.]),
        assignment=np.arange(6), station_ids=np.array(["f", "c", "a", "b", "e", "d"]),
        station_sources=np.array(["s"] * 6), fields=[field], grid_xy=np.zeros((6, 2)), map_selected=True)
    gate = {"selected_mode": "fixed_idr", "sha256": "c" * 64, "g0_pass": True, "g1_pass": True}
    fixed = SimpleNamespace(region=region.region, assignment=region.assignment, raw_assignment=region.assignment, gate=gate)
    matched = SimpleNamespace(region=region.region, candidate="GNN", seed=42, assignment=region.assignment,
        raw_assignment=region.assignment, gate={**gate, "selected_mode": "matched_idr", "candidate_sha256": "a" * 64})
    civd = SimpleNamespace(region=region.region, assignment=np.array([0, 0, 1, 1, 1, 2]),
        grid_cluster=np.array([0, 0, 1, 1, 1, 2]), station_cluster=np.asarray(station_cluster),
        metadata={"sha256": "d" * 64, "n_clusters": 4})
    vd = SimpleNamespace(region=region.region, assignment=region.assignment, target_ids=region.station_ids,
                         lineage={"sha256": "b" * 64})
    return [region], [fixed], [matched], [civd], [vd]


@pytest.mark.parametrize("station_cluster", ([1, 0, 0, 2, 1, 3], [30, 10, 10, 70, 30, 90]))
def test_civd_splits_cluster_demand_among_stations_including_unassigned_cluster(station_cluster):
    inputs = _civd_observation_inputs(station_cluster)
    tables = observe(*inputs)
    predicted = tables["predictions"].query("allocator == 'CIVD'")
    expected = np.array([10., 3., 3., 7., 10., 0.])
    np.testing.assert_array_equal(predicted.target_id, ["f", "c", "a", "b", "e", "d"])
    np.testing.assert_allclose(predicted.predicted, expected)
    assert predicted.predicted.sum() == inputs[0][0].fields[0].values.sum()
    for allocator in ("VD", "IDR-fixed", "IDR-matched"):
        np.testing.assert_array_equal(tables["predictions"].query("allocator == @allocator").predicted,
                                      inputs[0][0].fields[0].values)
    rmse = tables["metrics"].query("allocator == 'CIVD' and metric == 'rmse'").iloc[0]
    assert rmse.value == pytest.approx(np.sqrt(np.mean((expected - predicted.observed.to_numpy()) ** 2)))
    gate = tables["gates"].query("allocator == 'CIVD'").iloc[0]
    assert gate.changed_grid_count == 4
    assert gate.changed_grid_basis == "canonical_vd_station_cluster"
    assert gate.station_prediction_rule == "equal_split_cluster_demand"
    assert gate.candidate_tv_mass == pytest.approx(20 / 66)
    assert gate.max_station_mass_change == 8
    maps = tables["maps"].query("allocator == 'CIVD'")
    assert maps.assignment_kind.eq("cluster_ordinal").all()
    np.testing.assert_array_equal(maps.selected_assignment, inputs[3][0].assignment)


def test_civd_singletons_follow_station_cluster_identity_after_station_reordering():
    inputs = _civd_observation_inputs([5, 3, 1, 4, 0, 2])
    inputs[3][0].assignment = inputs[3][0].grid_cluster = np.arange(6)
    tables = observe(*inputs)
    np.testing.assert_array_equal(tables["predictions"].query("allocator == 'CIVD'").predicted,
                                  [7., 5., 4., 12., 2., 3.])


def test_civd_one_cluster_uniformly_splits_total_instead_of_crediting_station_zero():
    inputs = _civd_observation_inputs([0] * 6)
    inputs[3][0].assignment = inputs[3][0].grid_cluster = np.zeros(6, dtype=int)
    tables = observe(*inputs)
    np.testing.assert_array_equal(tables["predictions"].query("allocator == 'CIVD'").predicted,
                                  np.full(6, 33 / 6))
    assert tables["gates"].query("allocator == 'CIVD'").iloc[0].changed_grid_count == 0


@pytest.mark.parametrize("station_cluster", ([0, 1], [0., 0., 1., 1., 2., 2.], [-1, 0, 0, 1, 1, 2]))
def test_civd_rejects_misaligned_or_invalid_station_clusters(station_cluster):
    with pytest.raises(ValueError, match="CIVD station_cluster"):
        observe(*_civd_observation_inputs(station_cluster))


def test_civd_rejects_station_ordinal_outside_cluster_range():
    inputs = _civd_observation_inputs([1, 0, 0, 2, 1, 3])
    inputs[3][0].assignment = inputs[3][0].grid_cluster = np.array([0, 0, 1, 1, 1, 4])
    with pytest.raises(ValueError, match="assignment 与场 / target 身份不符"):
        observe(*inputs)


def test_civd_rejects_disagreement_between_assignment_and_grid_clusters():
    inputs = _civd_observation_inputs([1, 0, 0, 2, 1, 3])
    inputs[3][0].grid_cluster = np.array([0, 0, 1, 1, 1, 3])
    with pytest.raises(ValueError, match="assignment 与 grid_cluster 不符"):
        observe(*inputs)
