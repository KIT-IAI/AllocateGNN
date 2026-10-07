from __future__ import annotations

import numpy as np

from .registry import PriorSpec, prior_registry


def correction_factor(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("NTL prior must be a finite non-negative vector")
    positive = values[values > 0]
    epsilon = float(np.percentile(positive, 5)) if len(positive) else 0.1
    median = float(np.median(values))
    denominator = np.log1p(median if median > 0 else epsilon)
    return np.log1p(values + epsilon) / denominator


SPEC = prior_registry.register(
    PriorSpec(
        id="N",
        source_artifact="dataoverview.ntl",
        normalization="log1p_p05_epsilon_over_median_v1",
        support_rule="all_cells__statistics_from_valid_positive_support",
        correction_factor=correction_factor,
        loss_terms=("ntl_prior",),
        feature_channels=("ntl_prior",),
    )
)

