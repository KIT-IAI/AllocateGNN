from __future__ import annotations

import geopandas as gpd
import numpy as np


def apply_additive(
    base: np.ndarray,
    factor: np.ndarray,
    grid: gpd.GeoDataFrame,
    *,
    source_key: str,
) -> np.ndarray:
    base = np.asarray(base, dtype=float)
    factor = np.asarray(factor, dtype=float)
    result = np.zeros_like(base)
    for _, group in grid.groupby(source_key, sort=False):
        index = group.index.to_numpy()
        blocks = (
            group["covered_mask"].to_numpy(dtype=bool),
            group["unknown_mask"].to_numpy(dtype=bool),
        ) if {"covered_mask", "unknown_mask"} <= set(group.columns) else (np.ones(len(group), dtype=bool),)
        for block in blocks:
            if not block.any():
                continue
            selected = index[block]
            target_mass = float(base[selected].sum())
            if target_mass <= 0:
                continue
            centered = factor[selected] - factor[selected].mean()
            base_std = base[selected].std()
            factor_std = centered.std()
            alpha = base_std / factor_std if factor_std > 0 and base_std > 0 else 0.0
            raw = np.maximum(base[selected] + alpha * centered, 0.0)
            result[selected] = target_mass * raw / raw.sum() if raw.sum() > 0 else base[selected]
    return result

