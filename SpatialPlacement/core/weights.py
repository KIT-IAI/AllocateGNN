                       
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

from typing import Dict

import numpy as np
import pandas as pd

from .case import RegionBundle
from .geometry import haversine_distance_matrix


def normalise(w: np.ndarray) -> np.ndarray:
                                
    w = np.asarray(w, dtype=float)
    total = float(w.sum())
    if total <= 0:
        n = max(len(w), 1)
        return np.full(n, 1.0 / n)
    return w / total


def _group_indices(source_key: np.ndarray) -> Dict[str, np.ndarray]:
                                                       
    return pd.Series(source_key).groupby(pd.Series(source_key)).indices


def uniform_weights(bundle: RegionBundle) -> np.ndarray:
\
\
\
       
    base = np.zeros(len(bundle.grid_lonlat), dtype=float)
    for key, idx in _group_indices(bundle.source_key).items():
        d = bundle.source_demand.get(str(key))
        if d is None:
            continue
        base[idx] = float(d) / len(idx)
    return normalise(base)


def gpm_weights(bundle: RegionBundle, mode: str = "categorical") -> np.ndarray:
\
\
\
\
\
\
\
       
    if mode not in ("categorical", "proportional"):
        raise ValueError(f"unknown GPM mode: {mode}")

    lu = np.asarray(bundle.lu_prop, dtype=float)
    scores = np.zeros(len(bundle.grid_lonlat), dtype=float)

    for key, idx in _group_indices(bundle.source_key).items():
        key = str(key)
        pi = bundle.source_pct.get(key)
        if pi is None:
            continue
        pi = np.asarray(pi, dtype=float)
        d = bundle.source_demand.get(key)
        if d is None:
            continue

        if mode == "categorical":
            s = pi[lu[idx].argmax(axis=1)]
        else:
            s = lu[idx] @ pi

        s_sum = float(s.sum())
        inner = s / s_sum if s_sum > 0 else np.full(len(idx), 1.0 / len(idx))
        scores[idx] = inner * float(d)

    return normalise(scores)


def ref_weights(bundle: RegionBundle) -> np.ndarray:
\
\
\
\
\
       
    n = len(bundle.grid_lonlat)
    m = len(bundle.station_lonlat)
    if n == 0 or m == 0:
        return np.full(max(n, 1), 1.0 / max(n, 1))

    nearest = haversine_distance_matrix(
        np.asarray(bundle.grid_lonlat, float),
        np.asarray(bundle.station_lonlat, float)).argmin(axis=1)
    counts = np.bincount(nearest, minlength=m)
    dem = np.asarray(bundle.station_demand, float)

    per_point = np.where(counts[nearest] > 0, dem[nearest] / counts[nearest], 0.0)
    return normalise(per_point)
