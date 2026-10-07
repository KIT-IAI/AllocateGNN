                                                                      
from __future__ import annotations

import numpy as np

from .geometry import haversine_distance_matrix


def reconstruct_substation_peak(
    grid_lonlat: np.ndarray,
    grid_weights: np.ndarray,
    station_lonlat: np.ndarray,
    regional_demand: float,
) -> np.ndarray:
                                                                            
    grid = np.asarray(grid_lonlat, dtype=float)
    weights = np.asarray(grid_weights, dtype=float)
    stations = np.asarray(station_lonlat, dtype=float)
    if len(weights) != len(grid):
        raise ValueError("grid coordinates and weights must have equal length")
    if weights.sum() <= 0:
        raise ValueError("grid weights must have a positive sum")
    nearest = haversine_distance_matrix(grid, stations).argmin(axis=1)
    prediction = np.zeros(len(stations), dtype=float)
    np.add.at(prediction, nearest, weights)
    return prediction / weights.sum() * float(regional_demand)


def reconstruction_metrics(predicted: np.ndarray, observed: np.ndarray) -> dict[str, float]:
                                                              
    prediction = np.asarray(predicted, dtype=float)
    reference = np.asarray(observed, dtype=float)
    if prediction.shape != reference.shape:
        raise ValueError("predicted and observed arrays must have equal shape")
    error = prediction - reference
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
    }
