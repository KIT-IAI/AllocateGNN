                  
import numpy as np
import pytest

from StudyCase.Placement.pipeline.metrics import (
    compute_all_sizing_metrics,
    compute_dcr,
    compute_lbi,
    compute_tur,
    compute_wsd,
)


class TestWSD:
                       

    def test_single_facility(self) -> None:
                                 
        demand_coords = np.array([
            [-0.1278, 51.5074],          
            [-2.2426, 53.4808],              
        ])
        facility_coords = np.array([[-1.0, 52.5]])
        assignment = np.array([0, 0])
        weights = np.array([0.5, 0.5])

        wsd = compute_wsd(demand_coords, facility_coords, assignment, weights)
        assert wsd > 0
                          
        from StudyCase.Placement.pipeline.metrics import _haversine_vector
        d0 = _haversine_vector(
            np.array([-0.1278]), np.array([51.5074]),
            np.array([-1.0]), np.array([52.5]),
        )[0]
        d1 = _haversine_vector(
            np.array([-2.2426]), np.array([53.4808]),
            np.array([-1.0]), np.array([52.5]),
        )[0]
        expected = 0.5 * d0 + 0.5 * d1
        assert abs(wsd - expected) < 0.01


class TestLBI:
                             

    def test_balanced(self) -> None:
        assignment = np.array([0, 0, 1, 1])
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        lbi = compute_lbi(assignment, weights, k=2)
        assert abs(lbi) < 1e-10


class TestDCR:
                       

    def test_radius_zero(self) -> None:
        demand_coords = np.array([[-0.1, 51.5], [-0.2, 51.6]])
        facility_coords = np.array([[-0.15, 51.55]])
        assignment = np.array([0, 0])
        weights = np.array([0.5, 0.5])

        dcr = compute_dcr(demand_coords, facility_coords, assignment, weights, 0.0)
        assert dcr == 0.0

    def test_radius_huge(self) -> None:
        demand_coords = np.array([[-0.1, 51.5], [-0.2, 51.6]])
        facility_coords = np.array([[-0.15, 51.55]])
        assignment = np.array([0, 0])
        weights = np.array([0.5, 0.5])

        dcr = compute_dcr(demand_coords, facility_coords, assignment, weights, 99999.0)
        assert abs(dcr - 1.0) < 1e-10


class TestTUR:
                     

    def test_specific_case(self) -> None:
        sizing_config = {
            "safety_margin": 1.5,
            "standard_sizes_mva": [7.5, 15, 30, 45, 60],
        }
        predicted = np.array([7.5])
        actual = np.array([8.0])

        tur, q_rec = compute_tur(predicted, actual, sizing_config)
                                          
        assert q_rec[0] == 15.0
                                     
        assert abs(tur[0] - 53.333333) < 0.01


class TestSizingMetrics:
                         

    def test_all_over_provisioned(self) -> None:
        sizing_config = {
            "safety_margin": 1.5,
            "standard_sizes_mva": [7.5, 15, 30, 45, 60],
        }
                                               
        predicted = np.array([40.0, 40.0, 40.0])
        actual = np.array([1.0, 1.0, 1.0])

        result = compute_all_sizing_metrics(predicted, actual, sizing_config)
        assert result["OPR"] == 1.0
