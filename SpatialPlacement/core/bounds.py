                                                                        
from __future__ import annotations

import numpy as np


def connection_cost(
    demand: np.ndarray,
    firm_capacity: np.ndarray,
    incoming_load: float,
    unit_cost: float = 1.0,
    tariff: np.ndarray | float = 0.0,
) -> np.ndarray:
                                                                
    demand = np.asarray(demand, dtype=float)
    firm = np.asarray(firm_capacity, dtype=float)
    return unit_cost * np.maximum(0.0, demand + incoming_load - firm) + tariff


def cost_interval(
    estimated_demand: np.ndarray,
    error_budget: np.ndarray | float,
    firm_capacity: np.ndarray,
    incoming_load: float,
    unit_cost: float = 1.0,
    tariff: np.ndarray | float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
                                                                                  
    estimate = np.asarray(estimated_demand, dtype=float)
    budget = np.asarray(error_budget, dtype=float)
    if np.any(budget < 0):
        raise ValueError("error budgets must be non-negative")
    estimated_cost = connection_cost(
        estimate, firm_capacity, incoming_load, unit_cost, tariff
    )
    lower = connection_cost(
        estimate - budget, firm_capacity, incoming_load, unit_cost, tariff
    )
    upper = connection_cost(
        estimate + budget, firm_capacity, incoming_load, unit_cost, tariff
    )
    return estimated_cost, lower, upper


def fixed_site_error_and_bound(
    observed_cost: np.ndarray,
    estimated_cost: np.ndarray,
    lower_cost: np.ndarray,
    upper_cost: np.ndarray,
) -> tuple[float, float]:
                                                                                
    observed = np.asarray(observed_cost, dtype=float)
    estimate = np.asarray(estimated_cost, dtype=float)
    lower = np.asarray(lower_cost, dtype=float)
    upper = np.asarray(upper_cost, dtype=float)
    realised = float(np.mean(np.abs(estimate - observed)))
    bound = float(np.mean(np.maximum(estimate - lower, upper - estimate)))
    return realised, bound


def selection_regret_and_rectangular_bound(
    observed_cost: np.ndarray,
    estimated_cost: np.ndarray,
    lower_cost: np.ndarray,
    upper_cost: np.ndarray,
    top_fraction: float = 0.01,
) -> tuple[float, float]:
                                                                               
    observed = np.asarray(observed_cost, dtype=float)
    estimate = np.asarray(estimated_cost, dtype=float)
    lower = np.asarray(lower_cost, dtype=float)
    upper = np.asarray(upper_cost, dtype=float)
    if not (len(observed) == len(estimate) == len(lower) == len(upper)):
        raise ValueError("all cost arrays must have the same length")
    if len(observed) < 2:
        raise ValueError("at least two candidates are required")
    k = max(1, int(round(len(observed) * float(top_fraction))))
    selected = np.argsort(estimate, kind="stable")[:k]
    optimum = np.argsort(observed, kind="stable")[:k]
    regret = float(observed[selected].mean() - observed[optimum].mean())

    mask = np.zeros(len(observed), dtype=bool)
    mask[selected] = True
    high_inside = np.sort(upper[mask])[::-1]
    low_outside = np.sort(lower[~mask])
    exchanges = min(k, len(observed) - k)
    if exchanges == 0:
        return regret, 0.0
    cumulative = np.cumsum(high_inside[:exchanges]) - np.cumsum(
        low_outside[:exchanges]
    )
    bound = float(max(0.0, float(cumulative.max())) / k)
    return regret, bound
