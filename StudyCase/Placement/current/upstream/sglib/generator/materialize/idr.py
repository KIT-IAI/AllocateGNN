from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping

import geopandas as gpd
import numpy as np

from ..allocator.idr import allocate_idr_g01


PUBLIC_COLUMNS = ("residential_percent", "commercial_percent", "industrial_percent", "agricultural_percent", "others_percent")


def public_activity_field(
    stored_gpm: np.ndarray,
    source_keys: np.ndarray,
    source_regions: gpd.GeoDataFrame,
    *,
    source_column: str,
    demand_column: str = "Demand (MVA)",
) -> tuple[np.ndarray, float]:
    missing = {source_column, demand_column, *PUBLIC_COLUMNS} - set(source_regions)
    if missing:
        raise ValueError(f"public source table is missing {sorted(missing)}")
    if source_regions[source_column].astype(str).duplicated().any():
        raise ValueError("public source table must contain unique source keys")
    activity_values = source_regions[list(PUBLIC_COLUMNS)].astype(float).sum(axis=1)
    demand_values = source_regions[demand_column].astype(float)
    if not np.isfinite(activity_values).all() or (activity_values < 0).any():
        raise ValueError("public source activity must be finite and non-negative")
    if not np.isfinite(demand_values).all() or (demand_values < 0).any():
        raise ValueError("public source demand must be finite and non-negative")
    activity = dict(zip(source_regions[source_column].astype(str), activity_values, strict=True))
    demand = dict(zip(source_regions[source_column].astype(str), demand_values, strict=True))
    keys = np.asarray(source_keys).astype(str)
    stored = np.asarray(stored_gpm, dtype=np.float64)
    if keys.ndim != 1 or stored.shape != keys.shape or not len(keys):
        raise ValueError("stored GPM and source keys must be aligned non-empty vectors")
    if not np.isfinite(stored).all() or (stored < 0).any():
        raise ValueError("stored GPM must be finite and non-negative")
    result = np.zeros_like(stored)
    rescaled = np.zeros_like(stored)
    for position, source in enumerate(dict.fromkeys(keys.tolist()), start=1):
        if source not in activity:
            raise ValueError(f"{source}: public source row is missing")
        index = np.flatnonzero(keys == source)
        block = stored[index]
        block_total = float(block.sum())
        source_demand = float(demand[source])
        if source_demand == 0.0:
            if np.count_nonzero(block):
                raise ValueError(f"{source}: stored GPM violates source conservation")
            # A zero-demand source has no demand-derived spatial shares.  Keep
            # the block strictly zero instead of fabricating an epsilon or a
            # country-specific proxy distribution.
            continue
        if not np.isclose(block_total, source_demand, rtol=1e-8, atol=1e-8):
            raise ValueError(f"{source}: stored GPM violates source conservation")
        if block_total <= 0 or activity[source] <= 0:
            raise ValueError(f"{source}: public activity has no positive mass")
        result[index] = activity[source] * block / block_total
        altered = (0.37 + position * 1.91) * block
        rescaled[index] = activity[source] * altered / altered.sum()
    error = float(np.max(np.abs(result - rescaled)))
    if error > 1e-12:
        raise ValueError(f"source-total invariance failed: {error}")
    return result, error


def _coordinates(frame: gpd.GeoDataFrame, working_crs: str) -> np.ndarray:
    projected = frame.to_crs(working_crs)
    return np.column_stack([projected.geometry.x, projected.geometry.y])


def _require_positive_idr_mass(field: np.ndarray) -> None:
    """Fail closed when the IDR probability/TV gates are undefined."""

    values = np.asarray(field, dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("IDR activity field must be finite and non-negative")
    if float(values.sum()) == 0.0:
        raise ValueError(
            "IDR activity field has zero total mass; refusing to fabricate mass"
        )


def materialize_idr_fixed(
    grid: gpd.GeoDataFrame,
    stations: gpd.GeoDataFrame,
    public_activity: np.ndarray,
    canonical_assignment: np.ndarray,
    *,
    working_crs: str,
    b_tv: float,
) -> dict[str, Any]:
    _require_positive_idr_mass(public_activity)
    result = allocate_idr_g01(
        _coordinates(grid, working_crs),
        _coordinates(stations, working_crs),
        public_activity,
        canonical_assignment,
        transport_budget=b_tv,
    )
    return {**asdict(result), "allocator_version": "idr_g01_v1", "uses_station_geometry": True, "source_total_basis": "public_activity", "prohibited_truth": False}


def materialize_idr_matched(
    grid: gpd.GeoDataFrame,
    stations: gpd.GeoDataFrame,
    fields: Mapping[tuple[str, int], np.ndarray],
    canonical_assignment: np.ndarray,
    *,
    working_crs: str,
    b_tv: float,
) -> dict[tuple[str, int], dict[str, Any]]:
    output = {}
    for key, field in fields.items():
        _require_positive_idr_mass(field)
        result = allocate_idr_g01(
            _coordinates(grid, working_crs),
            _coordinates(stations, working_crs),
            np.asarray(field, dtype=float),
            canonical_assignment,
            transport_budget=b_tv,
        )
        output[key] = {**asdict(result), "allocator_version": "idr_g01_v1", "uses_station_geometry": True, "source_total_basis": "candidate_field", "prohibited_truth": False}
    return output
