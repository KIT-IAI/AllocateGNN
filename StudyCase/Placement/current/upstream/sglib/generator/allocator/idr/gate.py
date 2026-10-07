"""Minimal active stack: IDR-VD followed by G0 admissibility and G1 TV budget.

This module intentionally excludes G2, Power-OT, buffer selection, candidate
search and outcome-based calibration.  The only transferable fitted constant is
the frozen historical alpha=-0.5.  The transport budget is an explicit caller
input; the module has no hidden default.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .idr import allocate_idr_vd, euclidean_assignment


FROZEN_ALPHA = -0.5
KERNEL = "gaussian"
BANDWIDTH_RULE = "median_station_nearest_neighbor_distance"


@dataclass(frozen=True)
class IdrG01Result:
    assignment: np.ndarray
    raw_idr_assignment: np.ndarray
    selected_mode: str
    fallback_reason: str
    g0_pass: bool
    g1_pass: bool
    tv_mass: float
    transport_budget: float
    raw_empty_station_count: int
    alpha: float
    bandwidth_m: float
    canonical_total_mass: float
    raw_total_mass: float


def _aggregate(
    assignment: np.ndarray, field: np.ndarray, n_stations: int
) -> np.ndarray:
    result = np.zeros(n_stations, dtype=np.float64)
    np.add.at(result, assignment, field)
    return result


def _canonical_assignment(
    assignment: np.ndarray, n_cells: int, n_stations: int
) -> np.ndarray:
    labels = np.asarray(assignment, dtype=np.int64)
    if labels.shape != (n_cells,):
        raise ValueError("canonical_vd_assignment must align with target grid")
    if (labels < 0).any() or (labels >= n_stations).any():
        raise ValueError("canonical_vd_assignment contains an invalid station ordinal")
    return labels


def total_variation(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape or (a < 0).any() or (b < 0).any():
        raise ValueError("TV vectors must be aligned and non-negative")
    if a.sum() <= 0 or b.sum() <= 0:
        raise ValueError("TV vectors must have positive total mass")
    return float(0.5 * np.abs(a / a.sum() - b / b.sum()).sum())


def allocate_idr_g01(
    grid_xy: np.ndarray,
    station_xy: np.ndarray,
    public_activity: np.ndarray,
    canonical_vd_assignment: np.ndarray,
    *,
    transport_budget: float,
    demand_context_xy: np.ndarray | None = None,
    demand_context_activity: np.ndarray | None = None,
    station_context_xy: np.ndarray | None = None,
) -> IdrG01Result:
    """Return raw IDR if G0/G1 pass, otherwise canonical Euclidean VD.

    G0: finite positive weights, valid labels, conserved mass, no empty station.
    G1: TV between raw-IDR and canonical station public-mass distributions is no
        greater than the explicit caller-supplied budget.
    """

    if not np.isfinite(transport_budget) or not 0 <= transport_budget <= 1:
        raise ValueError("transport_budget must be finite and lie in [0, 1]")
    grid = np.asarray(grid_xy, dtype=np.float64)
    stations = np.asarray(station_xy, dtype=np.float64)
    field = np.asarray(public_activity, dtype=np.float64)
    if grid.ndim != 2 or grid.shape[1] != 2 or len(grid) == 0:
        raise ValueError("grid_xy must have shape (n, 2) with n > 0")
    if stations.ndim != 2 or stations.shape[1] != 2 or len(stations) == 0:
        raise ValueError("station_xy must have shape (k, 2) with k > 0")
    if field.shape != (len(grid),) or not np.isfinite(field).all():
        raise ValueError("public_activity must be finite and align with grid")
    if (field < 0).any() or field.sum() <= 0:
        raise ValueError("public_activity must be non-negative with positive mass")
    canonical = _canonical_assignment(
        canonical_vd_assignment, len(grid), len(stations)
    )
    reconstructed_canonical = euclidean_assignment(grid, stations)
    if not np.array_equal(reconstructed_canonical, canonical):
        mismatch = int(np.count_nonzero(reconstructed_canonical != canonical))
        raise ValueError(
            "grid_xy/station_xy do not reproduce canonical_vd_assignment "
            f"under Euclidean argmin ({mismatch} cells differ)"
        )
    raw = allocate_idr_vd(
        grid,
        stations,
        field,
        alpha=FROZEN_ALPHA,
        demand_context_xy=demand_context_xy,
        demand_context_activity=demand_context_activity,
        station_context_xy=station_context_xy,
    )
    raw_labels = np.asarray(raw.assignment, dtype=np.int64)
    canonical_mass = _aggregate(canonical, field, len(stations))
    raw_mass = _aggregate(raw_labels, field, len(stations))
    mass_tolerance = max(1e-12, abs(float(field.sum())) * 1e-12)
    g0_pass = bool(
        raw.empty_station_count == 0
        and np.isfinite(raw.site_weights).all()
        and (raw.site_weights > 0).all()
        and (raw_labels >= 0).all()
        and (raw_labels < len(stations)).all()
        and abs(float(raw_mass.sum() - field.sum())) <= mass_tolerance
    )
    tv = total_variation(raw_mass, canonical_mass)
    g1_pass = bool(tv <= float(transport_budget) + 1e-15)
    if not g0_pass:
        selected = canonical.copy()
        mode = "canonical_vd"
        reason = "g0_failed"
    elif not g1_pass:
        selected = canonical.copy()
        mode = "canonical_vd"
        reason = "g1_transport_budget_exceeded"
    else:
        selected = raw_labels.copy()
        mode = "matched_idr"
        reason = ""
    return IdrG01Result(
        assignment=selected,
        raw_idr_assignment=raw_labels.copy(),
        selected_mode=mode,
        fallback_reason=reason,
        g0_pass=g0_pass,
        g1_pass=g1_pass,
        tv_mass=tv,
        transport_budget=float(transport_budget),
        raw_empty_station_count=raw.empty_station_count,
        alpha=FROZEN_ALPHA,
        bandwidth_m=raw.bandwidth_m,
        canonical_total_mass=float(canonical_mass.sum()),
        raw_total_mass=float(raw_mass.sum()),
    )


__all__ = [
    "BANDWIDTH_RULE",
    "FROZEN_ALPHA",
    "IdrG01Result",
    "KERNEL",
    "allocate_idr_g01",
    "total_variation",
]


