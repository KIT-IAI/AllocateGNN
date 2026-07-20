# -*- coding: utf-8 -*-
"""
Statistical utility module for robustness analysis.

Provides an inference toolkit used across the experiment scripts in this
directory:

- exact_sign_flip_permutation : exact two-sided sign-flip permutation test
                                over the 2^n sign combinations of n paired
                                differences (primary significance test)
- paired_region_bootstrap     : region-level paired bootstrap confidence
                                interval, resampling only at the region
                                level (no nested resampling over seeds,
                                since seed is a crossed rather than nested
                                factor). The CI is reported as an interval
                                only and is not used to back out a p-value.
- cauchy_combination          : Cauchy combination (ACAT) of per-seed
                                p-values into a single p-value when needed
                                (the median is not used for this purpose)
- morans_i                    : Moran's I with a permutation p-value
                                (implemented directly in numpy, avoiding an
                                additional dependency on esda)
- build_queen_weights         : queen-contiguity spatial weights built by
                                dissolving ITL3_region.gpkg to the study
                                regions (row-standardised, with a check for
                                zero-neighbour rows)
- build_knn_weights           : centroid k-NN (k=3) spatial weights, used
                                as a supplementary weights matrix

`holm` (Holm-Bonferroni step-down correction) and `min_attainable_p`
(the smallest exact-test p-value attainable at a given sample size) are
defined locally below and re-exported for convenience.

The sys.path adjustment below walks up from this file's directory to the
repository root (the directory containing SpatialAllocation/).
"""

import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
import geopandas as gpd

SCRIPT_DIR = Path(__file__).resolve().parent
_p = SCRIPT_DIR
while not (_p / 'SpatialAllocation').exists():
    _p = _p.parent
PROJECT_ROOT = _p
sys.path.insert(0, str(PROJECT_ROOT))

def min_attainable_p(n: int) -> float:
    """Smallest two-sided p-value attainable by the exact Wilcoxon signed-rank
    test at sample size n.

    When all paired differences share the same sign, W = 0 and the two-sided
    p equals 2 / 2**n = 2**(1 - n). Use this to check, before reporting any
    small-sample test, whether significance is even reachable
    (n=4 -> 0.125; n=6 -> 0.03125; n=16 -> 3.05e-05).
    """
    if n < 1:
        return 1.0
    return min(1.0, 2.0 ** (1 - n))


def holm(pvalues: Dict[str, float]) -> Dict[str, float]:
    """Holm-Bonferroni step-down correction.

    Returns corrected p-values keyed identically to the input.
    """
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out: Dict[str, float] = {}
    running = 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)  # keep monotonic non-decreasing
        out[k] = running
    return out

__all__ = [
    'holm', 'min_attainable_p',
    'exact_sign_flip_permutation', 'paired_region_bootstrap',
    'cauchy_combination', 'morans_i',
    'build_queen_weights', 'build_knn_weights',
]

# Upper bound on sample size for exact enumeration (2^n memory guard)
_MAX_EXACT_N = 20


def exact_sign_flip_permutation(diffs: Sequence[float]) -> Dict[str, float]:
    """Exact two-sided sign-flip permutation test on paired differences.

    Enumerates all 2^n sign-flip combinations of n paired differences,
    computing the test statistic T = sum(+-d_i). The two-sided exact
    p-value is #{|T_perm| >= |T_obs|} / 2^n. The observed statistic itself
    is counted in the numerator, so the smallest attainable p-value is
    2/2^n (only the all-positive and all-negative sign combinations reach
    |T_obs| when all differences share the same sign).

    Args:
        diffs: array of paired differences (typically seed-averaged
               per-region paired differences; comparisons with no seed
               dimension pass the region-level differences directly).

    Returns:
        {'p': two-sided exact p-value, 't_obs': observed statistic,
         'n': sample size, 'n_enumerated': 2^n,
         'min_attainable_p': 2/2^n}
    """
    d = np.asarray(diffs, dtype=float)
    if d.ndim != 1 or len(d) == 0:
        raise ValueError('diffs must be a non-empty 1-D array')
    if not np.all(np.isfinite(d)):
        raise ValueError('diffs contains NaN/Inf')
    n = len(d)
    if n > _MAX_EXACT_N:
        raise ValueError(f'n={n} exceeds the exact enumeration limit {_MAX_EXACT_N} (2^n memory guard)')

    t_obs = float(d.sum())

    # Enumerate all 2^n sign combinations: bit k selects the sign of the k-th difference
    masks = np.arange(2 ** n, dtype=np.uint32)
    bits = ((masks[:, None] >> np.arange(n, dtype=np.uint32)) & 1).astype(np.float64)
    signs = 2.0 * bits - 1.0                       # (2^n, n) matrix of +-1 signs
    t_all = signs @ d                              # statistic for every permutation

    # Numerical tolerance: floating-point noise from summation order should not
    # cause a tied extreme value to be missed (conservative direction)
    tol = 1e-12 * max(1.0, float(np.abs(d).sum()))
    p = float(np.mean(np.abs(t_all) >= abs(t_obs) - tol))

    return {
        'p': p,
        't_obs': t_obs,
        'n': n,
        'n_enumerated': int(2 ** n),
        'min_attainable_p': 2.0 / 2 ** n,
    }


def paired_region_bootstrap(diffs: Sequence[float],
                            B: int = 10000,
                            seed: int = 0,
                            alpha: float = 0.05) -> Dict[str, float]:
    """Region-level paired bootstrap percentile confidence interval (outer region-level resampling only, no inner resampling).

    seed is treated as a crossed factor rather than a nested factor: inner
    resampling over seeds (3-choose-3 with replacement) would both
    underestimate variance and only produce 10 distinct multisets, so it is
    not performed here. This function resamples (with replacement) only the
    seed-averaged per-region paired differences. The CI is reported as an
    interval only; it must not be used to back out a p-value (percentile
    bootstrap does not support valid p-value inversion).

    Returns:
        {'mean': mean of the paired differences, 'ci_lo': CI lower bound,
         'ci_hi': CI upper bound, 'B': number of resamples,
         'alpha': significance level, 'seed': random seed, 'n': sample size}
    """
    d = np.asarray(diffs, dtype=float)
    if d.ndim != 1 or len(d) == 0:
        raise ValueError('diffs must be a non-empty 1-D array')
    if not np.all(np.isfinite(d)):
        raise ValueError('diffs contains NaN/Inf')
    n = len(d)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    boot_means = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [100.0 * alpha / 2, 100.0 * (1 - alpha / 2)])

    return {
        'mean': float(d.mean()),
        'ci_lo': float(lo),
        'ci_hi': float(hi),
        'B': int(B),
        'alpha': float(alpha),
        'seed': int(seed),
        'n': n,
    }


def cauchy_combination(pvalues: Sequence[float]) -> float:
    """Cauchy combination (ACAT): combines multiple p-values into a single p-value.

    T = mean(tan((0.5 - p_i) * pi)), p_comb = 0.5 - arctan(T) / pi.
    Used when a single summary number is needed in addition to reporting
    the per-seed permutation p-values individually (the median is
    deliberately not used for this purpose). A single p-value input is
    returned unchanged (up to floating-point tolerance).
    """
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1 or len(p) == 0:
        raise ValueError('pvalues must be a non-empty 1-D array')
    if np.any(~np.isfinite(p)) or np.any(p < 0) or np.any(p > 1):
        raise ValueError('pvalues must lie within [0, 1]')

    # Numerical safeguard: tan diverges as p -> 0 or p -> 1
    eps = 1e-15
    p = np.clip(p, eps, 1 - eps)

    t = float(np.mean(np.tan((0.5 - p) * np.pi)))
    p_comb = 0.5 - np.arctan(t) / np.pi
    return float(min(max(p_comb, 0.0), 1.0))


def morans_i(values: Sequence[float],
             W: np.ndarray,
             n_perm: int = 999,
             seed: int = 0) -> Dict[str, float]:
    """Moran's I with a permutation p-value (implemented directly in numpy; used as a diagnostic rather than a formal test, so no multiple-comparison correction is applied).

    I = (n/S0) * (z^T W z) / (z^T z), z = x - mean(x), S0 = sum(W).
    Permutation p-value: z is randomly permuted n_perm times with a fixed
    seed;
      two-sided p = (1 + #{|I_perm - E[I]| >= |I_obs - E[I]|}) / (n_perm + 1)
      one-sided p = (1 + #extreme in the observed direction) / (n_perm + 1)
      (following the esda convention).
    The two-sided p-value is used when flagging potential spatial
    dependence.

    Args:
        W: (n, n) spatial weights matrix (row-standardised or not; S0 is
           the actual sum of weights).
    """
    x = np.asarray(values, dtype=float)
    W = np.asarray(W, dtype=float)
    n = len(x)
    if W.shape != (n, n):
        raise ValueError(f'W shape {W.shape} does not match values length {n}')
    if n < 3:
        raise ValueError('Moran I requires at least 3 observations')
    if not np.all(np.isfinite(x)):
        raise ValueError('values contains NaN/Inf')

    z = x - x.mean()
    denom = float(z @ z)
    if denom == 0.0:
        raise ValueError('values are all equal; Moran I is undefined')
    s0 = float(W.sum())
    if s0 <= 0.0:
        raise ValueError('W weight sum must be positive')

    i_obs = n / s0 * float(z @ W @ z) / denom
    e_i = -1.0 / (n - 1)

    rng = np.random.default_rng(seed)
    perm = np.empty(n_perm, dtype=float)
    for k in range(n_perm):
        zp = z[rng.permutation(n)]
        perm[k] = n / s0 * float(zp @ W @ zp) / denom

    p_two = float((1 + np.sum(np.abs(perm - e_i) >= abs(i_obs - e_i))) / (n_perm + 1))
    if i_obs >= e_i:
        p_one = float((1 + np.sum(perm >= i_obs)) / (n_perm + 1))
    else:
        p_one = float((1 + np.sum(perm <= i_obs)) / (n_perm + 1))

    return {
        'I': float(i_obs),
        'EI': float(e_i),
        'p_perm_two_sided': p_two,
        'p_perm_one_sided': p_one,
        'n_perm': int(n_perm),
        'seed': int(seed),
    }


def build_queen_weights(gpkg_path: Union[str, Path],
                        region_ids: Sequence[str]) -> Tuple[np.ndarray, dict]:
    """Builds queen-contiguity spatial weights (row-standardised) by dissolving ITL3_region.gpkg to the study regions.

    The gpkg has an 'ITL2' column with London already merged (TLI3-TLI7 ->
    'London'), so filtering and dissolving by that column is
    straightforward. Queen contiguity means polygons share at least one
    boundary point; this is implemented with `intersects` (equivalent to
    `touches` for non-overlapping polygons, and more robust to boundary
    floating-point noise).

    The 16 study regions are not all contiguous, so isolated regions
    (zero-neighbour rows) are not treated as an error here; they are
    returned as-is so the caller can decide whether to fall back to a
    supplementary weights matrix.

    Args:
        region_ids: list of study region ids (e.g.
                    shared_correction_utils.ALL_LOCATIONS); the row/column
                    order of the returned W matches this list.

    Returns:
        (W, info): W is an (n, n) row-standardised weights matrix
        (zero-neighbour rows remain all-zero); info contains 'binary' (the
        0/1 adjacency matrix), 'neighbor_counts', 'zero_rows' (list of
        isolated region ids), 'region_ids', and 'centroids_epsg27700'
        ((n, 2) centroid coordinates, reused by build_knn_weights).
    """
    region_ids = list(region_ids)
    region_gdf = gpd.read_file(str(gpkg_path))
    if 'ITL2' not in region_gdf.columns:
        raise ValueError(f'{gpkg_path} is missing an ITL2 column')

    available = set(region_gdf['ITL2'].unique())
    missing = [r for r in region_ids if r not in available]
    if missing:
        raise ValueError(f'study regions not found in gpkg: {missing}')

    sub = region_gdf[region_gdf['ITL2'].isin(region_ids)]
    dissolved = sub[['ITL2', 'geometry']].dissolve(by='ITL2')
    dissolved = dissolved.loc[region_ids]  # Align row order to region_ids

    n = len(region_ids)
    geoms = dissolved.geometry.values
    binary = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            if geoms[i].intersects(geoms[j]):
                binary[i, j] = binary[j, i] = 1.0

    row_sums = binary.sum(axis=1)
    zero_rows = [region_ids[i] for i in range(n) if row_sums[i] == 0]

    W = np.zeros_like(binary)
    nz = row_sums > 0
    W[nz] = binary[nz] / row_sums[nz, None]

    cent = dissolved.to_crs('EPSG:27700').geometry.centroid
    centroids = np.column_stack([cent.x.values, cent.y.values])

    info = {
        'region_ids': region_ids,
        'binary': binary,
        'neighbor_counts': row_sums.astype(int).tolist(),
        'zero_rows': zero_rows,
        'centroids_epsg27700': centroids,
    }
    return W, info


def build_knn_weights(centroids: np.ndarray, k: int = 3) -> np.ndarray:
    """Centroid k-NN spatial weights (row-standardised) — a supplementary weights matrix alongside the queen-contiguity weights.

    Args:
        centroids: (n, 2) projected coordinates (EPSG:27700 recommended;
                   can be passed directly from build_queen_weights's
                   returned info['centroids_epsg27700']).
        k: number of neighbours per row; each row has exactly k neighbours,
           and after row-standardisation each weight is 1/k.

    Note that k-NN adjacency is generally asymmetric; Moran's I does not
    require a symmetric weights matrix.
    """
    c = np.asarray(centroids, dtype=float)
    if c.ndim != 2 or c.shape[1] != 2:
        raise ValueError('centroids must be an (n, 2) array')
    n = len(c)
    if not (1 <= k < n):
        raise ValueError(f'k={k} must satisfy 1 <= k < n={n}')

    diff = c[:, None, :] - c[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(dist, np.inf)

    W = np.zeros((n, n), dtype=float)
    nn_idx = np.argsort(dist, axis=1)[:, :k]
    for i in range(n):
        W[i, nn_idx[i]] = 1.0 / k
    return W
