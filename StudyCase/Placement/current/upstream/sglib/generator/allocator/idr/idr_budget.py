"""Truth-free sufficient conversions from planner tolerances to a G1 TV budget."""

from __future__ import annotations

import numpy as np


def budget_from_wape_shift_tolerance(wape_tolerance: float) -> float:
    """TV budget guaranteeing total absolute prediction shift / mass <= tolerance.

    For equal-total station mass vectors y0,y1:
        sum |y1-y0| / M = 2 * TV(p1,p0).
    """

    if not np.isfinite(wape_tolerance) or not 0 <= wape_tolerance <= 2:
        raise ValueError("wape_tolerance must lie in [0, 2]")
    return float(wape_tolerance / 2.0)


def budget_from_absolute_rmse_shift_tolerance(
    rmse_tolerance: float, total_mass: float, n_stations: int
) -> float:
    """Conservative TV budget for an absolute station-prediction RMSE shift.

    ||y1-y0||_2 / sqrt(K) <= 2 M TV / sqrt(K), so a sufficient budget is
    tau * sqrt(K) / (2 M).
    """

    if not np.isfinite(rmse_tolerance) or rmse_tolerance < 0:
        raise ValueError("rmse_tolerance must be finite and non-negative")
    if not np.isfinite(total_mass) or total_mass <= 0:
        raise ValueError("total_mass must be finite and positive")
    if int(n_stations) != n_stations or n_stations <= 0:
        raise ValueError("n_stations must be a positive integer")
    return float(min(1.0, rmse_tolerance * np.sqrt(n_stations) / (2.0 * total_mass)))


def budget_from_relative_rmse_to_mean_tolerance(
    relative_rmse_tolerance: float, n_stations: int
) -> float:
    """TV budget when RMSE tolerance is expressed relative to mean station mass."""

    if not np.isfinite(relative_rmse_tolerance) or relative_rmse_tolerance < 0:
        raise ValueError(
            "relative_rmse_tolerance must be finite and non-negative"
        )
    if int(n_stations) != n_stations or n_stations <= 0:
        raise ValueError("n_stations must be a positive integer")
    return float(min(1.0, relative_rmse_tolerance / (2.0 * np.sqrt(n_stations))))


def realised_shift_metrics(
    canonical_station_mass: np.ndarray, candidate_station_mass: np.ndarray
) -> dict[str, float]:
    canonical = np.asarray(canonical_station_mass, dtype=np.float64)
    candidate = np.asarray(candidate_station_mass, dtype=np.float64)
    if canonical.shape != candidate.shape or canonical.ndim != 1:
        raise ValueError("station mass vectors must be aligned and one-dimensional")
    if (
        not np.isfinite(canonical).all()
        or not np.isfinite(candidate).all()
        or (canonical < 0).any()
        or (candidate < 0).any()
    ):
        raise ValueError("station masses must be finite and non-negative")
    total = float(canonical.sum())
    if total <= 0 or not np.isclose(candidate.sum(), total, rtol=1e-10, atol=1e-12 * total):
        raise ValueError("station mass vectors must have equal positive totals")
    difference = candidate - canonical
    tv = float(0.5 * np.abs(difference / total).sum())
    wape_shift = float(np.abs(difference).sum() / total)
    rmse_shift = float(np.sqrt(np.mean(difference**2)))
    rmse_bound = float(2.0 * total * tv / np.sqrt(len(canonical)))
    return {
        "tv": tv,
        "wape_shift": wape_shift,
        "rmse_shift": rmse_shift,
        "rmse_shift_upper_bound": rmse_bound,
    }


__all__ = [
    "budget_from_absolute_rmse_shift_tolerance",
    "budget_from_relative_rmse_to_mean_tolerance",
    "budget_from_wape_shift_tolerance",
    "realised_shift_metrics",
]


