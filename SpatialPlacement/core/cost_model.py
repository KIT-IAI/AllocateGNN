                       
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

from typing import Optional

import numpy as np

from .case import CostSpec


def annuity_factor(years: int = 20, r: float = 0.035) -> float:
                                                              
    return (1 - (1 + r) ** (-years)) / r


def reinforcement(x_mw: float, f_cap: np.ndarray, g: np.ndarray) -> np.ndarray:
                                                          
    return np.maximum(0.0, x_mw - (np.asarray(f_cap, float) - np.asarray(g, float)))


def unit_multiplier(cost: CostSpec) -> float:
                                         
    if cost.unit_cost_per_kva is None:
        return 1.0
    return 1000.0 * float(cost.unit_cost_per_kva)


def cost_surface(
    x_mw: float,
    f_cap: np.ndarray,
    g: np.ndarray,
    cost: CostSpec,
    tariff_per_kw: Optional[np.ndarray] = None,
) -> np.ndarray:
\
\
\
\
\
\
\
       
    r = reinforcement(x_mw, f_cap, g)
    c = r * unit_multiplier(cost)
    if tariff_per_kw is not None and np.any(np.asarray(tariff_per_kw) != 0):
        if cost.unit_cost_per_kva is None:
            raise ValueError("a tariff surface requires unit_cost_per_kva; MVA and currency cannot be added")
        c = c + x_mw * 1000.0 * np.asarray(tariff_per_kw, float)\
            * annuity_factor(cost.horizon_years, cost.discount)
    return c
