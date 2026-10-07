"""Budget-bounded pairwise-distance reductions shared across stages."""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial.distance import cdist


def _xy(values: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1:] != (2,) or len(array) == 0:
        raise ValueError(f"{label} must have shape (n, 2) with n > 0")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must be finite")
    return array


def _chunk_plan(
    n_left: int,
    n_right: int,
    max_workspace_bytes: int,
) -> tuple[int, int]:
    if n_left <= 0 or n_right <= 0:
        raise ValueError("distance inputs must both be non-empty")
    if isinstance(max_workspace_bytes, bool) or int(max_workspace_bytes) <= 0:
        raise ValueError("max_workspace_bytes must be positive")
    # One float64 distance row plus one float64 reduction/argmin output value.
    bytes_per_left_row = (n_right + 1) * np.dtype(np.float64).itemsize
    budget = int(max_workspace_bytes)
    if bytes_per_left_row > budget:
        raise ValueError("one distance row exceeds max_workspace_bytes")
    rows = max(1, min(n_left, budget // bytes_per_left_row))
    return rows, rows * bytes_per_left_row


def chunked_nearest_assignment(
    left_xy: np.ndarray,
    right_xy: np.ndarray,
    *,
    max_workspace_bytes: int,
) -> tuple[np.ndarray, int, int]:
    """Return first-ordinal nearest labels and bounded peak workspace bytes."""

    left = _xy(left_xy, "left_xy")
    right = _xy(right_xy, "right_xy")
    chunk_rows, planned_peak = _chunk_plan(
        len(left), len(right), max_workspace_bytes
    )
    labels = np.empty(len(left), dtype=np.int64)
    observed_peak = 0
    for start in range(0, len(left), chunk_rows):
        stop = min(start + chunk_rows, len(left))
        distances = cdist(left[start:stop], right)
        labels[start:stop] = np.argmin(distances, axis=1)
        observed_peak = max(
            observed_peak,
            distances.nbytes + (stop - start) * np.dtype(np.int64).itemsize,
        )
    if observed_peak > planned_peak or observed_peak > int(max_workspace_bytes):
        raise RuntimeError("distance chunk exceeded its workspace plan")
    return labels, int(observed_peak), int(chunk_rows)


def chunked_inverse_power_sum(
    left_xy: np.ndarray,
    right_xy: np.ndarray,
    *,
    gamma: float,
    distance_scale: float,
    clamp_distance: float,
    max_workspace_bytes: int,
) -> tuple[np.ndarray, int, int]:
    """Compute ``sum(max(cdist/scale, clamp) ** -gamma, axis=1)``.

    All element-wise operations reuse the distance matrix in place.  This keeps
    the peak pairwise workspace within the configured byte budget while
    preserving the full-matrix formula and column reduction order.
    """

    left = _xy(left_xy, "left_xy")
    right = _xy(right_xy, "right_xy")
    exponent = float(gamma)
    scale = float(distance_scale)
    clamp = float(clamp_distance)
    if not math.isfinite(exponent):
        raise ValueError("gamma must be finite")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("distance_scale must be finite and positive")
    if not math.isfinite(clamp) or clamp <= 0:
        raise ValueError("clamp_distance must be finite and positive")
    chunk_rows, planned_peak = _chunk_plan(
        len(left), len(right), max_workspace_bytes
    )
    values = np.empty(len(left), dtype=np.float64)
    observed_peak = 0
    for start in range(0, len(left), chunk_rows):
        stop = min(start + chunk_rows, len(left))
        distances = cdist(left[start:stop], right)
        distances /= scale
        np.maximum(distances, clamp, out=distances)
        np.power(distances, -exponent, out=distances)
        values[start:stop] = np.sum(distances, axis=1)
        observed_peak = max(
            observed_peak,
            distances.nbytes + (stop - start) * np.dtype(np.float64).itemsize,
        )
    if observed_peak > planned_peak or observed_peak > int(max_workspace_bytes):
        raise RuntimeError("inverse-power chunk exceeded its workspace plan")
    return values, int(observed_peak), int(chunk_rows)


__all__ = ["chunked_inverse_power_sum", "chunked_nearest_assignment"]
