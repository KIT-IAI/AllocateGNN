                       
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

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .geometry import haversine_distance_matrix

DEFAULT_SAFETY_MARGIN = 1.5


def predict_substation_demand(
    assignment: np.ndarray,
    weights: np.ndarray,
    d_region: float,
    k: int,
) -> np.ndarray:
                                             
    if d_region <= 0:
        raise ValueError(f"d_region must be positive, got {d_region}")

    w_sum = float(np.sum(weights))
    if w_sum <= 0:
        return np.full(k, d_region / max(k, 1))

    cluster_w = np.zeros(k, dtype=float)
    np.add.at(cluster_w, np.asarray(assignment, dtype=int), np.asarray(weights, float))
    return cluster_w / w_sum * float(d_region)


def recommend_capacity(
    predicted_demand: np.ndarray,
    safety_margin: float = DEFAULT_SAFETY_MARGIN,
    discretise_to: Optional[Sequence[float]] = None,
) -> np.ndarray:
                                               
    q = np.asarray(predicted_demand, dtype=float) * float(safety_margin)
    if discretise_to is None:
        return q

    sizes = np.sort(np.asarray(discretise_to, dtype=float))
    idx = np.searchsorted(sizes, q, side="left")
    over = idx >= len(sizes)
    out = sizes[np.clip(idx, 0, len(sizes) - 1)]
    out = np.where(over, np.ceil(q / sizes[-1]) * sizes[-1], out)
    return out


def _matching_threshold(station_lonlat: np.ndarray,
                        max_dist_multiplier: float,
                        max_dist_km: float) -> float:
                                                         
    if len(station_lonlat) > 1:
        dd = haversine_distance_matrix(station_lonlat, station_lonlat)
        np.fill_diagonal(dd, np.inf)
        mean_nn = float(np.median(dd.min(axis=1)))
    else:
        mean_nn = max_dist_km
    return min(mean_nn * max_dist_multiplier, max_dist_km)


def match_to_real_substations(
    rec_coords: np.ndarray,
    station_lonlat: np.ndarray,
    station_demand: np.ndarray,
    station_firm: np.ndarray,
    max_dist_multiplier: float = 3.0,
    max_dist_km: float = 20.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
\
\
\
       
    demand = np.asarray(station_demand, dtype=float)
    firm = np.asarray(station_firm, dtype=float)
    threshold = _matching_threshold(np.asarray(station_lonlat, float),
                                    max_dist_multiplier, max_dist_km)

    dist = haversine_distance_matrix(np.asarray(rec_coords, float),
                                     np.asarray(station_lonlat, float))
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
    station_lonlat: np.ndarray,
    station_demand: np.ndarray,
    station_firm: np.ndarray,
    max_dist_multiplier: float = 3.0,
    max_dist_km: float = 20.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
\
\
\
       
    from scipy.optimize import linear_sum_assignment

    demand = np.asarray(station_demand, dtype=float)
    firm = np.asarray(station_firm, dtype=float)
    threshold = _matching_threshold(np.asarray(station_lonlat, float),
                                    max_dist_multiplier, max_dist_km)

    dist = haversine_distance_matrix(np.asarray(rec_coords, float),
                                     np.asarray(station_lonlat, float))

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
    q_rec: np.ndarray,
    actual_demand: np.ndarray,
    firm_capacity: Optional[np.ndarray] = None,
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
\
\
\
\
       
    q = np.asarray(q_rec, dtype=float)
    d = np.asarray(actual_demand, dtype=float)

    valid = (d > 0) & (q > 0) & np.isfinite(d) & np.isfinite(q)
    if int(valid.sum()) == 0:
        return {"n_matched": 0}

    qv, dv = q[valid], d[valid]
    tur = dv / qv * 100.0
    ce = np.abs(qv - dv) / dv * 100.0

    rsd: Dict[str, float] = {}
    if gamma is not None:
        bench = float(gamma) * dv                                            
        r = np.abs(qv - bench) / bench * 100.0
        rsd = {"RSD_mean": float(r.mean()), "RSD_median": float(np.median(r))}

    out: Dict[str, float] = {
        "n_matched": int(valid.sum()),
        **rsd,
        "TUR_mean": float(tur.mean()),
        "TUR_median": float(np.median(tur)),
        "TUR_aggregate": float(dv.sum() / qv.sum() * 100.0),
        "CE_mean": float(ce.mean()),
        "CE_median": float(np.median(ce)),
        "OPR": float(np.mean(qv > 2.0 * dv)),
        "UPR": float(np.mean(qv < dv)),
    }

    if firm_capacity is not None:
        f = np.asarray(firm_capacity, dtype=float)[valid]
        ok = (f > 0) & np.isfinite(f)
        if int(ok.sum()) > 0:
            fce = np.abs(qv[ok] - f[ok]) / f[ok] * 100.0
            out["FCE_mean"] = float(fce.mean())
            out["FCE_median"] = float(np.median(fce))
            out["TUR_actual_observed"] = float(np.median(dv[ok] / f[ok] * 100.0))

    return out


def sizing_detail_rows(
    d_hat: np.ndarray,
    q_rec: np.ndarray,
    actual_demand: np.ndarray,
    firm_capacity: np.ndarray,
    match_dist_km: np.ndarray,
    low_conf: np.ndarray,
    gamma: float,
    **tags: object,
) -> list:
\
\
\
\
\
\
\
\
       
    dh = np.asarray(d_hat, dtype=float)
    q = np.asarray(q_rec, dtype=float)
    d = np.asarray(actual_demand, dtype=float)
    f = np.asarray(firm_capacity, dtype=float)
    md = np.asarray(match_dist_km, dtype=float)
    lc = np.asarray(low_conf, dtype=bool)

    return [
        {
            **tags,
            "station": int(i),
            "D_hat_mva": float(dh[i]),
            "D_actual_mva": float(d[i]),
            "Q_rec_mva": float(q[i]),
            "Q_bench_mva": float(gamma) * float(d[i]),
            "Q_firm_mva": float(f[i]),
            "match_dist_km": float(md[i]),
            "low_conf": bool(lc[i]),
        }
        for i in range(len(q))
    ]
