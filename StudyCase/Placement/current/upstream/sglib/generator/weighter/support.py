"""Source-aware C/U/Z mass composition shared by all weighting methods."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def compose_support_field(
    source_keys: np.ndarray,
    source_demand: Mapping[str, float],
    covered_mask: np.ndarray,
    unknown_mask: np.ndarray,
    built_fraction: np.ndarray,
    covered_scores: np.ndarray,
) -> np.ndarray:
    """Compose one source-conserving field with fixed C/U mass and exact Z=0."""

    keys = np.asarray(source_keys).astype(str)
    covered = np.asarray(covered_mask, dtype=bool)
    unknown = np.asarray(unknown_mask, dtype=bool)
    capacity = np.asarray(built_fraction, dtype=np.float64)
    scores = np.asarray(covered_scores, dtype=np.float64)
    n = len(keys)
    if any(value.shape != (n,) for value in (covered, unknown, capacity, scores)):
        raise ValueError("all support arrays must be aligned one-dimensional vectors")
    if not np.isfinite(capacity).all() or (capacity < 0).any():
        raise ValueError("built_fraction must be finite and non-negative")
    if not np.isfinite(scores).all():
        raise ValueError("covered_scores must be finite")
    if np.any(covered & unknown):
        raise ValueError("covered and unknown masks must be disjoint")

    result = np.zeros(n, dtype=np.float64)
    for source in dict.fromkeys(keys.tolist()):
        idx = np.flatnonzero(keys == source)
        source_covered = covered[idx]
        source_unknown = unknown[idx]
        support = source_covered | source_unknown
        total_capacity = float(capacity[idx][support].sum())
        try:
            total_demand = float(source_demand[source])
        except KeyError as error:
            raise ValueError(f"missing demand for source {source}") from error
        if not np.isfinite(total_demand) or total_demand < 0:
            raise ValueError(f"source {source}: demand must be finite and non-negative")
        if total_demand == 0:
            continue
        if total_capacity <= 0:
            raise ValueError(f"source {source}: positive demand has no C/U capacity")

        covered_capacity = float(capacity[idx][source_covered].sum())
        covered_mass = total_demand * covered_capacity / total_capacity
        unknown_mass = total_demand - covered_mass
        if source_covered.any() and covered_mass > 0:
            local_scores = np.maximum(scores[idx][source_covered], 0.0)
            if float(local_scores.sum()) <= 0:
                local_scores = np.ones(source_covered.sum(), dtype=np.float64)
            result[idx[source_covered]] = covered_mass * local_scores / local_scores.sum()
        if source_unknown.any() and unknown_mass > 0:
            local_capacity = capacity[idx][source_unknown]
            if float(local_capacity.sum()) <= 0:
                raise ValueError(f"source {source}: unknown mass has no Built-S capacity")
            result[idx[source_unknown]] = unknown_mass * local_capacity / local_capacity.sum()
    return result


def support_block_mass(
    field: np.ndarray,
    source_keys: np.ndarray,
    covered_mask: np.ndarray,
    unknown_mask: np.ndarray,
) -> dict[tuple[str, str], float]:
    values = np.asarray(field, dtype=np.float64)
    keys = np.asarray(source_keys).astype(str)
    covered = np.asarray(covered_mask, dtype=bool)
    unknown = np.asarray(unknown_mask, dtype=bool)
    if any(value.shape != values.shape for value in (keys, covered, unknown)):
        raise ValueError("field and support arrays must be aligned")
    result: dict[tuple[str, str], float] = {}
    for source in dict.fromkeys(keys.tolist()):
        idx = keys == source
        result[(source, "C")] = float(values[idx & covered].sum())
        result[(source, "U")] = float(values[idx & unknown].sum())
    return result


