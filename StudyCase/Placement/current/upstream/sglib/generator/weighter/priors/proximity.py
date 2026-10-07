from __future__ import annotations

import numpy as np

from .registry import PriorSpec, prior_registry


def correction_factor(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("proximity prior must be a finite non-negative vector")
    median = float(np.median(values))
    denominator = np.log1p(median if median > 0 else 1e-6)
    return np.log1p(values) / denominator


SPEC = prior_registry.register(
    PriorSpec(
        id="P",
        source_artifact="generator.static.proximity",
        normalization="log1p_over_median_v1",
        support_rule="known_station_geometry__all_cells",
        correction_factor=correction_factor,
        loss_terms=("proximity_prior",),
        feature_channels=("proximity_prior",),
    )
)

