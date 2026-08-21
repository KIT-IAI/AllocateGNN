                              
import numpy as np
import geopandas as gpd
import pytest
from shapely.geometry import Point

from StudyCase.Placement.pipeline.candidate_generator import (
    aggregate_weights_to_candidates,
    compute_buildability_mask,
    compute_n_candidates,
    generate_candidates,
)


def _make_gdf(n: int, wc_others: float = 0.1) -> gpd.GeoDataFrame:
                                       
    rng = np.random.RandomState(42)
    lons = -1.0 + rng.randn(n) * 0.1
    lats = 52.0 + rng.randn(n) * 0.1
    return gpd.GeoDataFrame(
        {"wc_others_ratio": np.full(n, wc_others)},
        geometry=[Point(lon, lat) for lon, lat in zip(lons, lats)],
        crs="EPSG:4326",
    )


class TestBuildabilityMask:

    def test_threshold_zero_all_false(self) -> None:
                                      
        gdf = _make_gdf(100, wc_others=0.1)
        mask = compute_buildability_mask(gdf, threshold=0.0)
        assert mask.sum() == 0

    def test_threshold_one_all_true(self) -> None:
                                       
        gdf = _make_gdf(100, wc_others=0.5)
        mask = compute_buildability_mask(gdf, threshold=1.0)
        assert mask.sum() == 100


class TestComputeNCandidates:

    def test_auto(self) -> None:
        assert compute_n_candidates(50000, None) == 500

    def test_explicit(self) -> None:
        assert compute_n_candidates(50000, 800) == 800


class TestGenerateCandidates:

    def test_candidates_are_buildable(self) -> None:
                                                       
        n = 500
        rng = np.random.RandomState(42)
        lons = -1.0 + rng.randn(n) * 0.1
        lats = 52.0 + rng.randn(n) * 0.1
        wc = rng.uniform(0, 0.8, n)
        gdf = gpd.GeoDataFrame(
            {"wc_others_ratio": wc},
            geometry=[Point(lon, lat) for lon, lat in zip(lons, lats)],
            crs="EPSG:4326",
        )
        weights = np.ones(n) / n
        config = {"buildability_threshold": 0.3, "n_candidates": 10, "random_state": 42}

        result = generate_candidates(gdf, weights, config)
        assert all(gdf.iloc[result.candidate_indices]["wc_others_ratio"].values <= 0.3)

    def test_candidate_count_le_config(self) -> None:
                             
        gdf = _make_gdf(500, wc_others=0.1)
        weights = np.ones(500) / 500
        config = {"buildability_threshold": 0.5, "n_candidates": 20, "random_state": 42}

        result = generate_candidates(gdf, weights, config)
        assert len(result.candidate_indices) <= 20
        assert len(result.candidate_indices) > 0

    def test_aggregate_weights_sum_to_one(self) -> None:
                            
        gdf = _make_gdf(500, wc_others=0.1)
        weights = np.ones(500) / 500
        config = {"buildability_threshold": 0.5, "n_candidates": 20, "random_state": 42}

        result = generate_candidates(gdf, weights, config)
        agg_w = aggregate_weights_to_candidates(weights, result)
        assert abs(agg_w.sum() - 1.0) < 1e-6
        assert len(agg_w) == len(result.candidate_indices)
