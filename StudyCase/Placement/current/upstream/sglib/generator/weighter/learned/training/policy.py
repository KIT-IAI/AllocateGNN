from __future__ import annotations

import numpy as np

POLICY = "uniform_region_cyclic_sgd__source_mean_v1"


def source_mean(values: np.ndarray, source_index: np.ndarray) -> float:
    losses = np.asarray(values, dtype=np.float64).reshape(-1)
    sources = np.asarray(source_index).reshape(-1)
    if losses.shape != sources.shape or not np.isfinite(losses).all() or len(losses) == 0:
        raise ValueError("source_mean inputs must be aligned, finite, and non-empty")
    per_source = [float(losses[sources == key].mean()) for key in dict.fromkeys(sources.tolist())]
    return float(np.mean(per_source))


def uniform_region_mean(region_losses: np.ndarray) -> float:
    values = np.asarray(region_losses, dtype=np.float64).reshape(-1)
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("region losses must be finite and non-empty")
    return float(values.mean())


def validate_region_schedule(configured: list[str], observed: list[str]) -> None:
    if observed != configured or len(observed) != len(set(observed)):
        raise ValueError("each configured region must be visited exactly once in configured order")


__all__ = ["POLICY", "source_mean", "uniform_region_mean", "validate_region_schedule"]

