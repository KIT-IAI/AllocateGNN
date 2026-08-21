\
\
\
\
\
\
   
import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans


@dataclass
class CandidateResult:
                   
                            
    candidate_indices: np.ndarray
                                  
    candidate_gdf: gpd.GeoDataFrame
                           
    candidate_coords: np.ndarray
                                            
    labels: np.ndarray
                       
    buildable_indices: np.ndarray


def compute_buildability_mask(gdf: gpd.GeoDataFrame,
                              threshold: float) -> pd.Series:
\
\
\
\
\
\
\
\
\
       
    return gdf["wc_others_ratio"] <= threshold


def compute_n_candidates(n_grid: int,
                         n_candidates_config: Optional[int]) -> int:
\
\
\
\
\
\
\
\
\
       
    if n_candidates_config is not None:
        return int(n_candidates_config)
    return min(max(n_grid // 100, 300), 1000)


def generate_candidates(
    gdf: gpd.GeoDataFrame,
    weights: np.ndarray,
    candidates_config: Dict,
) -> CandidateResult:
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
       
    threshold = candidates_config["buildability_threshold"]
    n_candidates_cfg = candidates_config.get("n_candidates")
    random_state = candidates_config.get("random_state", 42)

               
    mask = compute_buildability_mask(gdf, threshold)
    buildable_idx = np.where(mask.values)[0]
    n_buildable = len(buildable_idx)

               
    n_candidates = compute_n_candidates(n_buildable, n_candidates_cfg)
    if n_buildable < n_candidates:
        warnings.warn(
            f"Buildable cells ({n_buildable}) are fewer than requested candidates "
            f"({n_candidates}); all {n_buildable} buildable cells will be used"
        )
        n_candidates = n_buildable

    if n_buildable == 0:
        empty_gdf = gdf.iloc[:0].copy()
        return CandidateResult(
            candidate_indices=np.array([], dtype=int),
            candidate_gdf=empty_gdf,
            candidate_coords=np.empty((0, 2)),
            labels=np.array([], dtype=int),
            buildable_indices=np.array([], dtype=int),
        )

                       
    buildable_gdf = gdf.iloc[buildable_idx].copy()
    buildable_proj = buildable_gdf.to_crs("EPSG:27700")
    proj_coords = np.column_stack([
        buildable_proj.geometry.x.values,
        buildable_proj.geometry.y.values,
    ])

              
    buildable_weights = weights[buildable_idx]
    buildable_weights = np.maximum(buildable_weights, 0.0)
    w_sum = buildable_weights.sum()
    sample_weight = buildable_weights if w_sum > 0 else None

                  
    kmeans = KMeans(
        n_clusters=n_candidates,
        random_state=random_state,
        n_init=10,
    )
    kmeans.fit(proj_coords, sample_weight=sample_weight)
    labels = kmeans.labels_                             

                                   
    centroids = kmeans.cluster_centers_
    dist_matrix = cdist(centroids, proj_coords, metric="euclidean")
    nearest_local = dist_matrix.argmin(axis=1)

                         
    unique_local, inverse = np.unique(nearest_local, return_inverse=True)
    candidate_indices = buildable_idx[unique_local]

                          
    remapped_labels = inverse[labels]

    candidate_gdf = gdf.iloc[candidate_indices].copy().reset_index(drop=True)
    candidate_coords = np.column_stack([
        candidate_gdf.geometry.x.values,
        candidate_gdf.geometry.y.values,
    ])

    return CandidateResult(
        candidate_indices=candidate_indices,
        candidate_gdf=candidate_gdf,
        candidate_coords=candidate_coords,
        labels=remapped_labels,
        buildable_indices=buildable_idx,
    )


def aggregate_weights_to_candidates(
    weights: np.ndarray,
    candidate_result: CandidateResult,
) -> np.ndarray:
\
\
\
\
\
\
\
\
\
       
    buildable_w = weights[candidate_result.buildable_indices]
    buildable_w = np.maximum(buildable_w, 0.0)

    n_cand = len(candidate_result.candidate_indices)
    agg = np.zeros(n_cand)
    np.add.at(agg, candidate_result.labels, buildable_w)

                                
    unbuildable_total = weights.sum() - buildable_w.sum()
    if unbuildable_total > 0 and n_cand > 0:
        agg += unbuildable_total / n_cand

         
    total = agg.sum()
    if total > 0:
        agg /= total
    return agg
