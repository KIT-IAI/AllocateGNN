                         
import numpy as np
import pytest

from StudyCase.Placement.pipeline.pmedian_solver import (
    assign_demand_points,
    haversine_distance_matrix,
    solve_pmedian_greedy,
)


class TestHaversine:
                                           

    def test_london_manchester(self) -> None:
        london = np.array([[-0.1278, 51.5074]])
        manchester = np.array([[-2.2426, 53.4808]])
        dist = haversine_distance_matrix(london, manchester)
        assert abs(dist[0, 0] - 262.0) < 5.0


class TestPMedianGreedy:
                          

    def test_k1_near_weighted_centroid(self) -> None:
                                    
        demand = np.array([[-1.0, 52.0], [-1.0, 52.1], [-1.0, 52.2]])
                           
        facilities = np.array([[-1.0, 52.1], [-5.0, 55.0]])
        weights = np.array([1.0, 1.0, 1.0]) / 3
        config = {"max_iter": 100, "random_restarts": 1}

        selected, assignment = solve_pmedian_greedy(
            demand, facilities, weights, k=1, pmedian_config=config,
        )
                             
        assert 0 in selected

    def test_k2_two_clusters(self) -> None:
                                      
        rng = np.random.RandomState(42)
                   
        cluster1 = np.column_stack([
            -0.1 + rng.randn(20) * 0.01,
            51.5 + rng.randn(20) * 0.01,
        ])
                     
        cluster2 = np.column_stack([
            -2.2 + rng.randn(20) * 0.01,
            53.5 + rng.randn(20) * 0.01,
        ])
        demand = np.vstack([cluster1, cluster2])
        weights = np.ones(40) / 40

                      
        facilities = np.array([
            [-0.1, 51.5],       
            [-2.2, 53.5],         
            [-4.0, 50.0],       
        ])
        config = {"max_iter": 100, "random_restarts": 2}

        selected, assignment = solve_pmedian_greedy(
            demand, facilities, weights, k=2, pmedian_config=config,
        )
                       
        assert set(selected) == {0, 1}

    def test_assignment_range(self) -> None:
                             
        demand = np.array([[-1.0, 52.0], [-1.5, 52.5], [-2.0, 53.0]])
        facilities = np.array([[-1.0, 52.0], [-1.5, 52.5], [-2.0, 53.0]])
        weights = np.ones(3) / 3
        config = {"max_iter": 50, "random_restarts": 1}

        selected, assignment = solve_pmedian_greedy(
            demand, facilities, weights, k=2, pmedian_config=config,
        )
                            
        for a in assignment:
            assert a in selected
