                       
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

import numpy as np


def block_permute(w: np.ndarray, xy: np.ndarray, block_km: float,
                  rng: np.random.RandomState) -> np.ndarray:
                                   
    bid = (np.floor(xy[:, 0] / block_km).astype(np.int64) * 1_000_003
           + np.floor(xy[:, 1] / block_km).astype(np.int64))
    out = w.copy()
    order = np.argsort(bid, kind="stable")
    bs = bid[order]
    starts = np.flatnonzero(np.r_[True, bs[1:] != bs[:-1]])
    for a, b in zip(starts, np.r_[starts[1:], len(bs)]):
        idx = order[a:b]
        out[idx] = w[rng.permutation(idx)]
    return out


def smooth_multiplier(xy: np.ndarray, wavelength_km: float, strength: float,
                      rng: np.random.RandomState, n_modes: int = 4) -> np.ndarray:
                                        
    f = 2 * np.pi / wavelength_km
    z = np.zeros(len(xy))
    for _ in range(n_modes):
        th = rng.uniform(0, 2 * np.pi)
        ph = rng.uniform(0, 2 * np.pi)
        z += np.sin(f * (xy[:, 0] * np.cos(th) + xy[:, 1] * np.sin(th)) + ph)
    z /= np.sqrt(n_modes)
    return np.exp(strength * z)
