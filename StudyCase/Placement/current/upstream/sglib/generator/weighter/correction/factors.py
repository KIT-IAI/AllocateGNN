from __future__ import annotations

import geopandas as gpd
import numpy as np
from scipy.spatial.distance import cdist

RCI_THRESHOLD = 0.5
PROXIMITY_GAMMA = 2.0
DIST_CLAMP_KM = 0.01


def compute_prox_scores(
    grid_gdf: gpd.GeoDataFrame,
    stations: gpd.GeoDataFrame,
    gamma: float = PROXIMITY_GAMMA,
    *,
    working_crs: str,
) -> np.ndarray:
    grid = grid_gdf.to_crs(working_crs)
    targets = stations.to_crs(working_crs)
    grid_xy = np.column_stack([grid.geometry.x, grid.geometry.y])
    target_xy = np.column_stack([targets.geometry.x, targets.geometry.y])
    distance_km = np.maximum(cdist(grid_xy, target_xy) / 1000.0, DIST_CLAMP_KM)
    return np.sum(distance_km ** (-gamma), axis=1)


def compute_factors(
    grid_gdf: gpd.GeoDataFrame,
    ntl_values: np.ndarray,
    proximity_scores: np.ndarray,
    *,
    source_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    ntl_values = np.asarray(ntl_values, dtype=float)
    proximity_scores = np.asarray(proximity_scores, dtype=float)
    rci = (
        grid_gdf["lu_residential_prop"].to_numpy()
        + grid_gdf["lu_commercial_prop"].to_numpy()
        + grid_gdf["lu_industrial_prop"].to_numpy()
    ) > RCI_THRESHOLD
    ntl_factor = np.ones(len(grid_gdf))
    proximity_factor = np.ones(len(grid_gdf))
    for _, group in grid_gdf.groupby(source_key, sort=False):
        idx = group.index.to_numpy()
        valid_ntl = ntl_values[idx][rci[idx] & (ntl_values[idx] > 0)]
        fallback = ntl_values[idx][ntl_values[idx] > 0]
        epsilon = float(np.percentile(valid_ntl, 5)) if len(valid_ntl) else float(np.percentile(fallback, 5)) if len(fallback) else 0.1
        median_ntl_values = ntl_values[idx][rci[idx]]
        median_ntl = float(np.median(median_ntl_values if len(median_ntl_values) else ntl_values[idx]))
        ntl_factor[idx] = np.log1p(ntl_values[idx] + epsilon) / np.log1p(median_ntl if median_ntl > 0 else epsilon)
        median_proximity_values = proximity_scores[idx][rci[idx]]
        median_proximity = float(np.median(median_proximity_values if len(median_proximity_values) else proximity_scores[idx]))
        proximity_factor[idx] = np.log1p(proximity_scores[idx]) / np.log1p(median_proximity if median_proximity > 0 else 1e-6)
    return ntl_factor, proximity_factor


def factor_bundle(grid_gdf, ntl_values, proximity_scores, *, source_key: str) -> dict[str, np.ndarray]:
    ntl, proximity = compute_factors(grid_gdf, ntl_values, proximity_scores, source_key=source_key)
    return {"N": ntl, "P": proximity, "NP": ntl * proximity}

