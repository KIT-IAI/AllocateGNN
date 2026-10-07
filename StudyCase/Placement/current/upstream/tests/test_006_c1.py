"""C1 估计量、分辨率、零分母与空间敏感性回归。"""

import numpy as np
import pytest

from sglib.analysis.paired_inference import compare, greedy_blocks, holm, moran, sign_flip
from sglib.core.algorithms.reconstruction_metrics import reconstruction_metrics, equal_region

pytestmark = pytest.mark.consume


def test_nz_exact_resolution_and_block_interval_support_are_distinct():
    assert sign_flip(np.ones(9))["p"] == 2/512
    assert np.allclose(holm([2/512]*10), .0390625)
    adjacency = np.eye(9, k=1, dtype=bool) | np.eye(9, k=-1, dtype=bool)
    blocks = greedy_blocks(adjacency)
    assert len(blocks) == 5
    result = compare(np.ones(9), np.ones(9)*2, adjacency, blocks)
    assert result["block_sign_flip"]["p_min"] == 2/32
    assert result["block_support"]
    assert result["block"]["native_ci"] == [-1., -1.]


def test_relative_ratio_is_not_the_median_of_region_percentages():
    result = compare([2., 110.], [1., 100.], np.zeros((2, 2), bool), [[0], [1]])
    assert result["relative_pct"] == pytest.approx(100*11/101)
    assert result["median_region_pct"] == 55.
    assert result["effect"] == 5.5


def test_bootstrap_zero_denominator_invalidates_only_relative_interval():
    result = compare([1., 1., 2.], [0., 0., 1.], np.zeros((3, 3), bool), [[0], [1], [2]])
    assert result["relative_pct"] == 300.
    assert result["main"]["relative_ci"] is None
    assert result["main"]["zero_denominator_draws"] > 0
    assert result["main"]["native_ci"] == [1., 1.]
    assert result["sign_flip"]["p"] == .25


def test_islands_are_excluded_only_from_moran_and_retain_singleton_blocks():
    adjacency = np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]], bool)
    assert greedy_blocks(adjacency) == [[0, 1], [2], [3]]
    result = moran([1, 2, 4, 100], adjacency)
    assert result["n"] == 3 and result["islands"] == 1
    assert result["p"] * 1000 == pytest.approx(round(result["p"] * 1000))


def test_metrics_preserve_negative_predictive_r2_and_undefined_corr():
    result = reconstruction_metrics([1., 3.], [10., 10.])
    assert result["predictive_r2"]["value"] < 0
    assert result["corr"]["value"] is None
    assert reconstruction_metrics([0., 0.], [0., 0.])["wape"]["status"] == "METRIC_NOT_ASSESSABLE"
    assert np.array_equal(equal_region(["s", "s", "t"], {"s": 10., "t": 2.}), [5., 5., 2.])
    assert np.array_equal(equal_region(["s", "s", "t"], {"s": 10., "t": 2., "empty": 0.}), [5., 5., 2.])
    with pytest.raises(ValueError, match="正需求"):
        equal_region(["s"], {"s": 10., "empty": 1.})


def synthetic_c1_tables():
    from types import SimpleNamespace
    from sglib.experiment.reconstruction import ReconstructionRegion, observe
    from sglib.analysis.c1 import METHODS, analyze

    names = [f"r{i}" for i in range(9)]
    methods = {m: {"seeds": [42, 123, 456] if m in {"MLP", "GNN"} else [None]} for m in METHODS}
    inputs = []
    for i, name in enumerate(names):
        fields = []
        for label in METHODS[:5]:
            for seed in methods[label]["seeds"]:
                predicted = [5., 5.] if label in {"Uni", "EqualGrid"} else [4., 6.] if label == "GPM" else [2., 8.]
                if label == "MLP":
                    predicted = {42: [1., 9.], 123: [3., 7.], 456: [2., 8.]}[seed]
                fields.append(SimpleNamespace(label=label, seed=seed, fold=None if seed is None else 1,
                    values=np.array(predicted), qa_only=False, lineage={"sha256": "a"*64}))
        inputs.append(ReconstructionRegion("nz", name, "MVA", np.array(["a", "b"]), np.array(["s", "s"]), np.array([2., 8.]),
            {"s": 10.}, np.array(["s", "s"]), np.array([0, 1]), tuple(fields), "b"*64,
            np.array([[0., 0.], [1., 1.]]), np.array([[0., 0.], [1., 1.]]), ("POLYGON ((0 0, 2 0, 0 2, 0 0))",), i == 0))
    tables = observe(inputs)
    specification = {"country": "nz", "regions": names, "unit": "MVA", "methods": methods,
        "queen_adjacency": np.zeros((9, 9), bool).tolist(), "blocks": [[i] for i in range(9)], "representative_region": names[0]}
    return analyze(tables, specification), specification


def test_observation_analysis_chain_keeps_seed_metrics_before_region_average():
    result, _ = synthetic_c1_tables()
    regions = result["region_metrics"]
    observed = regions[(regions.candidate == "MLP") & (regions.metric == "rmse")].value.to_numpy()
    assert np.allclose(observed, 2/3)
    assert len(result["contrasts"]) == 10
    assert len(result["secondary_effects"]) == 40
    assert len(result["inference_resolution_audit"]) == 20
    assert result["coordinate_audit"].status.eq("VALID").all()
