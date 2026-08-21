\
\
\
\
\
\
\
\
\
\
\
\
   
import numpy as np
import pandas as pd
import pytest

from SpatialPlacement.pipeline.statistics import (
    bootstrap_ci,
    friedman_test,
    holm,
    min_attainable_p,
    paired_compare,
)


class TestMinAttainableP:
                             

    @pytest.mark.parametrize(
        "n, expected",
        [(4, 0.125), (5, 0.0625), (6, 0.03125), (16, 2.0 ** -15)],
    )
    def test_known_values(self, n: int, expected: float) -> None:
        assert min_attainable_p(n) == pytest.approx(expected)

    def test_n4_can_never_reach_alpha_005(self) -> None:
\
\
\
\
           
        assert min_attainable_p(4) > 0.05

    def test_n6_is_the_first_n_that_can_reach_alpha_005(self) -> None:
                                            
        assert min_attainable_p(5) > 0.05
        assert min_attainable_p(6) < 0.05

    def test_degenerate_n(self) -> None:
        assert min_attainable_p(0) == 1.0
        assert min_attainable_p(1) == 1.0


class TestHolm:
                                

    def test_stepdown_weights_then_enforces_monotonicity(self) -> None:
\
\
\
\
\
           
        adj = holm({"a": 0.01, "b": 0.02, "c": 0.03})
        assert adj["a"] == pytest.approx(0.03)             
        assert adj["b"] == pytest.approx(0.04)             
        assert adj["c"] == pytest.approx(0.04)                        

    def test_monotone_nondecreasing(self) -> None:
                                              
        raw = {"a": 0.001, "b": 0.049, "c": 0.05, "d": 0.9}
        adj = holm(raw)
        ordered = [adj[k] for k, _ in sorted(raw.items(), key=lambda kv: kv[1])]
        assert all(x <= y + 1e-12 for x, y in zip(ordered, ordered[1:]))

    def test_capped_at_one(self) -> None:
        assert all(v <= 1.0 for v in holm({"a": 0.6, "b": 0.7, "c": 0.8}).values())

    def test_single_test_is_unchanged(self) -> None:
        assert holm({"only": 0.04})["only"] == pytest.approx(0.04)


class TestBootstrapCI:
                                   

    def test_deterministic_given_seed(self) -> None:
                                             
        v = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert bootstrap_ci(v, n_boot=2000, seed=42) == bootstrap_ci(v, n_boot=2000, seed=42)

    def test_brackets_the_mean(self) -> None:
        rng = np.random.default_rng(0)
        v = rng.normal(10.0, 2.0, 200)
        lo, hi = bootstrap_ci(v, n_boot=2000, seed=1)
        assert lo < v.mean() < hi

    def test_constant_input_gives_degenerate_interval(self) -> None:
        lo, hi = bootstrap_ci([7.0] * 10, n_boot=500, seed=0)
        assert lo == pytest.approx(7.0) and hi == pytest.approx(7.0)

    def test_empty_returns_nan(self) -> None:
        lo, hi = bootstrap_ci([], n_boot=100, seed=0)
        assert np.isnan(lo) and np.isnan(hi)


def _long_df(values: dict) -> pd.DataFrame:
                                                     
    rows = []
    for method, vals in values.items():
        for i, v in enumerate(vals):
            rows.append({"region": f"R{i}", "method": method, "WSD": v})
    return pd.DataFrame(rows)


class TestPairedCompare:
                               

    def test_direction_lower_is_better(self) -> None:
                                             
        df = _long_df({"A": [10.0, 12.0, 11.0, 13.0, 10.5, 12.5],
                       "B": [9.0, 10.0, 10.0, 11.0, 9.5, 11.0]})
        out = paired_compare(df, "WSD", pairs=[("A", "B")], n_boot=500)

        row = out.iloc[0]
        assert row["pair"] == "A_vs_B"
        assert row["n"] == 6
        assert row["n_B_better"] == 6
        assert row["improve_pct_median"] > 0

    def test_direction_reversed_when_a_is_better(self) -> None:
                                       
        df = _long_df({"A": [9.0, 10.0, 10.0, 11.0],
                       "B": [10.0, 12.0, 11.0, 13.0]})
        out = paired_compare(df, "WSD", pairs=[("A", "B")], n_boot=500)
        assert out.iloc[0]["n_B_better"] == 0
        assert out.iloc[0]["improve_pct_median"] < 0

    def test_underpowered_flag_on_n4(self) -> None:
\
\
\
           
        df = _long_df({"A": [10.0, 12.0, 11.0, 13.0],
                       "B": [8.0, 9.0, 9.5, 10.0]})
        out = paired_compare(df, "WSD", pairs=[("A", "B")], n_boot=500, alpha=0.05)
        row = out.iloc[0]

        assert row["n"] == 4
        assert row["n_B_better"] == 4                 
        assert bool(row["underpowered"])                
        assert row["p_wilcoxon"] >= 0.125                    
        assert not bool(row["significant"])

    def test_n16_not_underpowered(self) -> None:
                             
        rng = np.random.default_rng(0)
        a = rng.uniform(10, 14, 16)
        df = _long_df({"A": list(a), "B": list(a - 1.0)})
        out = paired_compare(df, "WSD", pairs=[("A", "B")], n_boot=500)
        assert not bool(out.iloc[0]["underpowered"])

    def test_missing_method_is_skipped(self) -> None:
                                      
        df = _long_df({"A": [1.0, 2.0, 3.0], "B": [1.0, 1.5, 2.5]})
        out = paired_compare(df, "WSD", pairs=[("A", "ZZZ"), ("A", "B")], n_boot=200)
        assert list(out["pair"]) == ["A_vs_B"]

    def test_empty_pairs_returns_empty_frame(self) -> None:
        df = _long_df({"A": [1.0, 2.0], "B": [1.0, 1.5]})
        assert paired_compare(df, "WSD", pairs=[], n_boot=200).empty


class TestFriedman:
                       

    def test_returns_stats_for_three_methods(self) -> None:
        df = _long_df({"A": [10.0, 12.0, 11.0, 13.0],
                       "B": [9.0, 11.0, 10.0, 12.0],
                       "C": [8.0, 10.0, 9.0, 11.0]})
        out = friedman_test(df, "WSD", ["A", "B", "C"])
        assert set(out) == {"friedman_chi2", "friedman_p", "n"}
        assert out["n"] == 4

    def test_returns_empty_for_two_methods(self) -> None:
                                            
        df = _long_df({"A": [1.0, 2.0, 3.0], "B": [1.0, 1.5, 2.5]})
        assert friedman_test(df, "WSD", ["A", "B"]) == {}
