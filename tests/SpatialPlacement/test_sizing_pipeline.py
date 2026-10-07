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
\
\
   
import numpy as np
import geopandas as gpd
import pytest
from shapely.geometry import Point

from SpatialPlacement.pipeline.sizing_pipeline import (
    DEFAULT_SAFETY_MARGIN,
    compute_sizing_metrics,
    match_to_real_substations,
    predict_substation_demand,
    recommend_capacity,
)


class TestPredictDemand:
                                                 

    def test_sum_conservation(self) -> None:
                                     
        assignment = np.array([0, 0, 1, 1, 2])
        weights = np.array([0.3, 0.2, 0.15, 0.15, 0.2])
        pred = predict_substation_demand(assignment, weights, 100.0, k=3)

        assert pred.shape == (3,)
        assert abs(pred.sum() - 100.0) < 1e-9

    def test_uniform_equal_split(self) -> None:
                                    
        pred = predict_substation_demand(
            np.array([0, 0, 1, 1]), np.full(4, 0.25), 200.0, k=2,
        )
        assert pred == pytest.approx([100.0, 100.0])

    def test_weights_need_not_be_normalised(self) -> None:
                                        
        a = np.array([0, 0, 1])
        norm = predict_substation_demand(a, np.array([0.5, 0.25, 0.25]), 80.0, k=2)
        raw = predict_substation_demand(a, np.array([2.0, 1.0, 1.0]), 80.0, k=2)
        assert norm == pytest.approx(raw)

    def test_empty_cluster_gets_zero(self) -> None:
                                      
        pred = predict_substation_demand(np.array([0, 0]), np.array([0.5, 0.5]), 50.0, k=3)
        assert pred[2] == pytest.approx(0.0)
        assert pred.sum() == pytest.approx(50.0)

    def test_zero_weights_fall_back_to_uniform(self) -> None:
                                   
        pred = predict_substation_demand(np.array([0, 1]), np.zeros(2), 60.0, k=2)
        assert np.all(np.isfinite(pred))
        assert pred == pytest.approx([30.0, 30.0])

    @pytest.mark.parametrize("bad", [0.0, -1.0])
    def test_nonpositive_region_demand_rejected(self, bad: float) -> None:
                                      
        with pytest.raises(ValueError):
            predict_substation_demand(np.array([0]), np.array([1.0]), bad, k=1)


class TestDRegionUnitGuard:
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
       

    def _tur(self, d_region: float) -> float:
        assignment = np.array([0, 0, 1, 1])
        weights = np.full(4, 0.25)
                                                    
        actual = np.array([2640.35, 2640.35])

        pred = predict_substation_demand(assignment, weights, d_region, k=2)
        q_rec = recommend_capacity(pred)
        return compute_sizing_metrics(q_rec, actual)["TUR_mean"]

    def test_correct_d_region_lands_near_inverse_gamma(self) -> None:
                                                     
        tur = self._tur(5280.7)
        assert tur == pytest.approx(100.0 / DEFAULT_SAFETY_MARGIN, abs=0.5)

    def test_step_size_as_demand_explodes_tur(self) -> None:
\
\
\
\
\
           
        tur = self._tur(280.0)
        assert tur > 1000.0

    def test_ratio_matches_documented_18_9x(self) -> None:
                                            
        assert self._tur(280.0) / self._tur(5280.7) == pytest.approx(5280.7 / 280.0, rel=1e-6)


class TestRecommendCapacity:
                                                  

    def test_continuous_is_default(self) -> None:
                                    
        q = recommend_capacity(np.array([10.0, 33.3]))
        assert q == pytest.approx([15.0, 49.95])

    def test_gamma_is_one_point_five(self) -> None:
                                                      
        assert DEFAULT_SAFETY_MARGIN == 1.5

    def test_discretise_rounds_up(self) -> None:
                              
        q = recommend_capacity(np.array([10.0]), discretise_to=[8, 15, 23, 32, 38])
        assert q[0] == pytest.approx(15.0)                        

    def test_discretise_beyond_max_uses_multiple_units(self) -> None:
                                         
        q = recommend_capacity(np.array([100.0]), discretise_to=[8, 15, 23, 32, 38])
        assert q[0] >= 150.0                                                        
        assert q[0] % 38 == pytest.approx(0.0)


class TestMatchToReal:
                                                               

    @staticmethod
    def _subs(with_firm: bool = True) -> gpd.GeoDataFrame:
        data = {"Demand (MVA)": [50.0, 30.0]}
        if with_firm:
            data["Firm Capacity (MVA)"] = [75.0, 45.0]
        return gpd.GeoDataFrame(
            data,
            geometry=[Point(-1.0, 52.0), Point(-1.5, 52.5)],
            crs="EPSG:4326",
        )

    def test_exact_overlap(self) -> None:
                                           
        actual, firm, dist, low = match_to_real_substations(
            np.array([[-1.0, 52.0], [-1.5, 52.5]]), self._subs(),
        )
        assert dist == pytest.approx([0.0, 0.0], abs=0.01)
        assert not low.any()
        assert actual == pytest.approx([50.0, 30.0])
        assert firm == pytest.approx([75.0, 45.0])

    def test_far_station_low_confidence(self) -> None:
                                                 
        subs = gpd.GeoDataFrame(
            {"Demand (MVA)": [100.0], "Firm Capacity (MVA)": [150.0]},
            geometry=[Point(-0.1, 51.5)],               
            crs="EPSG:4326",
        )
        actual, firm, dist, low = match_to_real_substations(
            np.array([[-2.24, 53.48]]), subs,                      
        )
        assert bool(low[0])
        assert np.isnan(actual[0]) and np.isnan(firm[0])
        assert dist[0] > 200.0

    def test_missing_firm_column_yields_nan(self) -> None:
                                              
        _, firm, _, _ = match_to_real_substations(
            np.array([[-1.0, 52.0]]), self._subs(with_firm=False),
        )
        assert np.isnan(firm).all()


class TestSizingMetrics:
                                                

    def test_tur_and_ce_basic(self) -> None:
        q = np.array([150.0, 150.0])
        d = np.array([100.0, 50.0])
        m = compute_sizing_metrics(q, d)

        assert m["n_matched"] == 2
        assert m["TUR_mean"] == pytest.approx((100 / 150 + 50 / 150) / 2 * 100)
        assert m["TUR_aggregate"] == pytest.approx(150 / 300 * 100)
        assert m["CE_mean"] == pytest.approx((50 / 100 + 100 / 50) / 2 * 100)

    def test_nan_rows_excluded(self) -> None:
                                          
        m = compute_sizing_metrics(np.array([150.0, 150.0]), np.array([100.0, np.nan]))
        assert m["n_matched"] == 1
        assert m["TUR_mean"] == pytest.approx(100 / 150 * 100)

    def test_all_nan_returns_only_count(self) -> None:
                                   
        assert compute_sizing_metrics(np.array([1.0]), np.array([np.nan])) == {"n_matched": 0}

    def test_over_and_under_provision_rates(self) -> None:
                                            
        q = np.array([300.0, 100.0, 150.0])
        d = np.array([100.0, 200.0, 100.0])
        m = compute_sizing_metrics(q, d)
        assert m["OPR"] == pytest.approx(1 / 3)                   
        assert m["UPR"] == pytest.approx(1 / 3)                   

    def test_firm_capacity_metrics(self) -> None:
                                                           
        m = compute_sizing_metrics(
            np.array([150.0]), np.array([100.0]), firm_capacity_mva=np.array([200.0]),
        )
        assert m["FCE_mean"] == pytest.approx(25.0)                         
        assert m["TUR_actual_observed"] == pytest.approx(50.0)           


class TestRSD:
                                                               

    GAMMA = DEFAULT_SAFETY_MARGIN        

    def test_absent_without_gamma(self) -> None:
                                           
        m = compute_sizing_metrics(np.array([150.0]), np.array([100.0]))
        assert "RSD_mean" not in m and "RSD_median" not in m

    def test_zero_at_perfect_estimate(self) -> None:
                                                           
        d = np.array([100.0, 37.5])
        q = self.GAMMA * d                              
        m = compute_sizing_metrics(q, d, gamma=self.GAMMA)
        assert m["RSD_mean"] == pytest.approx(0.0)
        assert m["RSD_median"] == pytest.approx(0.0)
        assert m["CE_median"] == pytest.approx(50.0)              

    def test_equals_relative_demand_error(self) -> None:
                                           
        d = np.array([100.0, 100.0, 100.0])
        d_hat = np.array([100.0, 150.0, 60.0])
        m = compute_sizing_metrics(self.GAMMA * d_hat, d, gamma=self.GAMMA)
        expected = np.abs(d_hat - d) / d * 100.0                  
        assert m["RSD_mean"] == pytest.approx(expected.mean())
        assert m["RSD_median"] == pytest.approx(np.median(expected))

    def test_minimum_at_r_equals_one_not_two_thirds(self) -> None:
\
\
\
\
           
        d = np.array([100.0])
        q_under = self.GAMMA * (2.0 / 3.0) * d                
        m = compute_sizing_metrics(q_under, d, gamma=self.GAMMA)
        assert m["CE_median"] == pytest.approx(0.0)                    
        assert m["RSD_median"] == pytest.approx(100.0 / 3)                  

    def test_discretised_capacity_uses_definition_not_shortcut(self) -> None:
                                                       
        d = np.array([100.0])
        q_rec = np.array([180.0])                                
        m = compute_sizing_metrics(q_rec, d, gamma=self.GAMMA)
        assert m["RSD_median"] == pytest.approx(abs(180.0 - 150.0) / 150.0 * 100)

    def test_nan_rows_excluded_like_other_metrics(self) -> None:
                            
        m = compute_sizing_metrics(
            np.array([150.0, 999.0]), np.array([100.0, np.nan]), gamma=self.GAMMA,
        )
        assert m["n_matched"] == 1
        assert m["RSD_mean"] == pytest.approx(0.0)


class TestSizingDetailRows:
                                           

    def test_row_contents_and_tags(self) -> None:
        from SpatialPlacement.core.sizing import sizing_detail_rows

        rows = sizing_detail_rows(
            d_hat=np.array([80.0, 120.0]),
            q_rec=np.array([120.0, 180.0]),
            actual_demand=np.array([100.0, np.nan]),
            firm_capacity=np.array([150.0, np.nan]),
            match_dist_km=np.array([1.5, 30.0]),
            low_conf=np.array([False, True]),
            gamma=DEFAULT_SAFETY_MARGIN,
            region="TLC1", method="GNN", protocol="multi",
        )
        assert len(rows) == 2
        assert rows[0]["region"] == "TLC1" and rows[0]["protocol"] == "multi"
        assert rows[0]["Q_bench_mva"] == pytest.approx(150.0)        
        assert rows[0]["D_hat_mva"] == pytest.approx(80.0)
                                          
        assert rows[1]["low_conf"] is True
