"""Standalone coordinate/public-field implementation of IDR-VD.

This exploratory module intentionally does not register with the frozen project
pipeline.  It consumes only grid coordinates, station coordinates, and a
non-negative public activity field aligned to the grid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist


@dataclass(frozen=True)
class IdrVdResult:
    assignment: np.ndarray
    site_weights: np.ndarray
    demand_intensity: np.ndarray
    station_intensity: np.ndarray
    bandwidth_m: float
    alpha: float
    empty_station_count: int
    demand_context_count: int
    station_context_count: int


def _xy(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2 or len(array) == 0:
        raise ValueError(f"{name} must have shape (n, 2) with n > 0")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def median_nearest_neighbor_bandwidth(station_xy: np.ndarray) -> float:
    """Label-free bandwidth derived only from the station coordinate set."""

    stations = _xy(station_xy, "station_xy")
    if len(stations) == 1:
        return 1.0
    distances = cdist(stations, stations)
    np.fill_diagonal(distances, np.inf)
    bandwidth = float(np.median(distances.min(axis=1)))
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("station coordinates must not contain unresolved duplicates")
    return bandwidth


def context_nearest_neighbor_bandwidth(
    target_station_xy: np.ndarray, station_context_xy: np.ndarray
) -> float:
    """Median target-site nearest-neighbour distance in a context containing targets."""

    targets = _xy(target_station_xy, "target_station_xy")
    context = _xy(station_context_xy, "station_context_xy")
    distances = cdist(targets, context)
    represented = np.isclose(distances, 0.0, rtol=0.0, atol=1e-8).any(axis=1)
    if not represented.all():
        raise ValueError("station_context_xy must contain every target station")
    distances[np.isclose(distances, 0.0, rtol=0.0, atol=1e-8)] = np.inf
    nearest = distances.min(axis=1)
    if not np.isfinite(nearest).all() or (nearest <= 0).any():
        if len(targets) == 1:
            return 1.0
        raise ValueError("station context lacks a distinct neighbour for every target")
    return float(np.median(nearest))


def euclidean_assignment(
    grid_xy: np.ndarray, station_xy: np.ndarray, *, chunk_size: int = 50_000
) -> np.ndarray:
    """Return ordinary Euclidean Voronoi labels with stable ordinal tie-breaking."""

    grid = _xy(grid_xy, "grid_xy")
    stations = _xy(station_xy, "station_xy")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    result = np.empty(len(grid), dtype=np.int64)
    for start in range(0, len(grid), chunk_size):
        stop = min(start + chunk_size, len(grid))
        result[start:stop] = np.argmin(cdist(grid[start:stop], stations), axis=1)
    return result


def allocate_idr_vd(
    grid_xy: np.ndarray,
    station_xy: np.ndarray,
    public_activity: np.ndarray,
    *,
    alpha: float = -0.5,
    bandwidth_m: float | None = None,
    bandwidth_multiplier: float = 1.0,
    chunk_size: int = 50_000,
    demand_context_xy: np.ndarray | None = None,
    demand_context_activity: np.ndarray | None = None,
    station_context_xy: np.ndarray | None = None,
) -> IdrVdResult:
    """Allocate an inverse intensity-ratio multiplicatively weighted VD.

    Let rho_D be a Gaussian kernel estimate of public activity at every station
    and rho_S the same estimate for station-point intensity.  The site weight is

        w_k = ((rho_D,k / rho_S,k) / median(rho_D / rho_S)) ** alpha

    and each target grid cell is assigned to
    ``argmin_k distance(cell, k) / w_k``.  Density estimation may use explicit
    demand/station context outside the target assignment domain.  Context must
    contain the target grid field/stations; assignment candidates remain the
    target stations only.
    The selected exploratory candidate uses ``alpha=-0.5``.  ``alpha=0``
    recovers ordinary Euclidean VD exactly.
    """

    grid = _xy(grid_xy, "grid_xy")
    stations = _xy(station_xy, "station_xy")
    activity = np.asarray(public_activity, dtype=np.float64)
    if activity.shape != (len(grid),):
        raise ValueError("public_activity must align one-to-one with grid_xy")
    if not np.isfinite(activity).all() or (activity < 0).any():
        raise ValueError("public_activity must be finite and non-negative")
    if float(activity.sum()) <= 0:
        raise ValueError("public_activity must contain positive mass")
    if (demand_context_xy is None) != (demand_context_activity is None):
        raise ValueError(
            "demand_context_xy and demand_context_activity must be supplied together"
        )
    if demand_context_xy is None:
        density_grid = grid
        density_activity = activity
    else:
        density_grid = _xy(demand_context_xy, "demand_context_xy")
        density_activity = np.asarray(demand_context_activity, dtype=np.float64)
        if density_activity.shape != (len(density_grid),):
            raise ValueError(
                "demand_context_activity must align one-to-one with demand_context_xy"
            )
        if not np.isfinite(density_activity).all() or (density_activity < 0).any():
            raise ValueError(
                "demand_context_activity must be finite and non-negative"
            )
        if float(density_activity.sum()) <= 0:
            raise ValueError("demand context must contain positive mass")
    station_context = (
        stations
        if station_context_xy is None
        else _xy(station_context_xy, "station_context_xy")
    )
    if not np.isfinite(alpha):
        raise ValueError("alpha must be finite")
    if not np.isfinite(bandwidth_multiplier) or bandwidth_multiplier <= 0:
        raise ValueError("bandwidth_multiplier must be finite and positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    base_bandwidth = (
        (
            median_nearest_neighbor_bandwidth(stations)
            if station_context_xy is None
            else context_nearest_neighbor_bandwidth(stations, station_context)
        )
        if bandwidth_m is None
        else float(bandwidth_m)
    )
    bandwidth = base_bandwidth * float(bandwidth_multiplier)
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("bandwidth must be finite and positive")

    demand_intensity = np.zeros(len(stations), dtype=np.float64)
    denominator = 2.0 * bandwidth**2
    for start in range(0, len(density_grid), chunk_size):
        stop = min(start + chunk_size, len(density_grid))
        distances = cdist(density_grid[start:stop], stations)
        kernel = np.exp(-(distances**2) / denominator)
        demand_intensity += kernel.T @ density_activity[start:stop]
    station_distances = cdist(stations, station_context)
    station_intensity = np.exp(-(station_distances**2) / denominator).sum(axis=1)
    ratio = demand_intensity / np.maximum(station_intensity, np.finfo(float).tiny)
    if not np.isfinite(ratio).all() or (ratio <= 0).any():
        raise ValueError("every station must receive positive public kernel intensity")
    ratio /= np.median(ratio)
    site_weights = ratio ** float(alpha)

    assignment = np.empty(len(grid), dtype=np.int64)
    for start in range(0, len(grid), chunk_size):
        stop = min(start + chunk_size, len(grid))
        distances = cdist(grid[start:stop], stations)
        assignment[start:stop] = np.argmin(
            distances / site_weights[None, :], axis=1
        )
    empty = int(len(stations) - len(np.unique(assignment)))
    return IdrVdResult(
        assignment=assignment,
        site_weights=site_weights,
        demand_intensity=demand_intensity,
        station_intensity=station_intensity,
        bandwidth_m=bandwidth,
        alpha=float(alpha),
        empty_station_count=empty,
        demand_context_count=len(density_grid),
        station_context_count=len(station_context),
    )


__all__ = [
    "IdrVdResult",
    "allocate_idr_vd",
    "euclidean_assignment",
    "context_nearest_neighbor_bandwidth",
    "median_nearest_neighbor_bandwidth",
]


