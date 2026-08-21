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
   
from __future__ import annotations

import warnings
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

warnings.filterwarnings("ignore")

                                 
DEFAULT_SAFETY_MARGIN = 1.5


def predict_substation_demand(
    assignment: np.ndarray,
    weights: np.ndarray,
    d_region_mva: float,
    k: int,
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
\
\
\
       
    if d_region_mva <= 0:
        raise ValueError(f"d_region_mva must be positive, got {d_region_mva}")

    w_sum = float(np.sum(weights))
    if w_sum <= 0:
        return np.full(k, d_region_mva / max(k, 1))

    cluster_w = np.zeros(k, dtype=float)
    np.add.at(cluster_w, np.asarray(assignment, dtype=int), np.asarray(weights, float))
    return cluster_w / w_sum * float(d_region_mva)


def recommend_capacity(
    predicted_demand_mva: np.ndarray,
    safety_margin: float = DEFAULT_SAFETY_MARGIN,
    discretise_to: Optional[Sequence[float]] = None,
) -> np.ndarray:
\
\
\
\
\
       
    q = np.asarray(predicted_demand_mva, dtype=float) * float(safety_margin)
    if discretise_to is None:
        return q

    sizes = np.sort(np.asarray(discretise_to, dtype=float))
    idx = np.searchsorted(sizes, q, side="left")
    over = idx >= len(sizes)
    out = sizes[np.clip(idx, 0, len(sizes) - 1)]
                            
    out = np.where(over, np.ceil(q / sizes[-1]) * sizes[-1], out)
    return out


def match_to_real_substations(
    rec_coords: np.ndarray,
    subs_gdf,
    max_dist_multiplier: float = 3.0,
    max_dist_km: float = 20.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
\
\
\
\
\
\
\
\
       
    from .pmedian_solver import haversine_distance_matrix

    subs_xy = np.column_stack([subs_gdf.geometry.x.values, subs_gdf.geometry.y.values])
    demand = subs_gdf["Demand (MVA)"].to_numpy(dtype=float)
    firm = (
        subs_gdf["Firm Capacity (MVA)"].to_numpy(dtype=float)
        if "Firm Capacity (MVA)" in subs_gdf.columns
        else np.full(len(subs_gdf), np.nan)
    )

    if len(subs_gdf) > 1:
        dd = haversine_distance_matrix(subs_xy, subs_xy)
        np.fill_diagonal(dd, np.inf)
        mean_nn = float(np.median(dd.min(axis=1)))
    else:
        mean_nn = max_dist_km
    threshold = min(mean_nn * max_dist_multiplier, max_dist_km)

    dist = haversine_distance_matrix(np.asarray(rec_coords), subs_xy)
    nearest = dist.argmin(axis=1)
    match_dist = dist[np.arange(len(rec_coords)), nearest]
    low_conf = match_dist > threshold

    actual = demand[nearest].astype(float)
    firm_matched = firm[nearest].astype(float)
    actual[low_conf] = np.nan
    firm_matched[low_conf] = np.nan

    return actual, firm_matched, match_dist, low_conf


def match_to_real_substations_unique(
    rec_coords: np.ndarray,
    subs_gdf,
    max_dist_multiplier: float = 3.0,
    max_dist_km: float = 20.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
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
       
    from scipy.optimize import linear_sum_assignment

    from .pmedian_solver import haversine_distance_matrix

    subs_xy = np.column_stack([subs_gdf.geometry.x.values,
                               subs_gdf.geometry.y.values])
    demand = subs_gdf["Demand (MVA)"].to_numpy(dtype=float)
    firm = (
        subs_gdf["Firm Capacity (MVA)"].to_numpy(dtype=float)
        if "Firm Capacity (MVA)" in subs_gdf.columns
        else np.full(len(subs_gdf), np.nan)
    )

    if len(subs_gdf) > 1:
        dd = haversine_distance_matrix(subs_xy, subs_xy)
        np.fill_diagonal(dd, np.inf)
        mean_nn = float(np.median(dd.min(axis=1)))
    else:
        mean_nn = max_dist_km
    threshold = min(mean_nn * max_dist_multiplier, max_dist_km)

    dist = haversine_distance_matrix(np.asarray(rec_coords), subs_xy)

                                  
    nearest = dist.argmin(axis=1)
    _, counts = np.unique(nearest, return_counts=True)
    n_dup = int(counts[counts > 1].sum() - (counts > 1).sum())
    collision_rate = float(n_dup / len(rec_coords)) if len(rec_coords) else 0.0

    rows, cols = linear_sum_assignment(dist)
    n_rec = len(rec_coords)
    matched_col = np.full(n_rec, -1, dtype=int)
    matched_col[rows] = cols
                                  
    unassigned = matched_col < 0
    safe_col = np.where(unassigned, 0, matched_col)
    match_dist = np.where(unassigned, np.inf,
                          dist[np.arange(n_rec), safe_col])
    low_conf = match_dist > threshold

    actual = demand[safe_col].astype(float)
    firm_matched = firm[safe_col].astype(float)
    actual[low_conf] = np.nan
    firm_matched[low_conf] = np.nan

    return actual, firm_matched, match_dist, low_conf, collision_rate


def compute_sizing_metrics(
    q_rec_mva: np.ndarray,
    actual_demand_mva: np.ndarray,
    firm_capacity_mva: Optional[np.ndarray] = None,
    gamma: Optional[float] = None,
) -> Dict[str, float]:
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
\
\
\
       
    from ..core.sizing import compute_sizing_metrics as _core_metrics

    return _core_metrics(q_rec_mva, actual_demand_mva, firm_capacity_mva,
                         gamma=gamma)
