from __future__ import annotations

from typing import Any, Mapping

import geopandas as gpd
import numpy as np

from ..allocator.vd import compute_vd_assignment
from ..inputs import GeneratorDataBundle
from ..weighter.correction import apply_additive, apply_standard_multiplicative, factor_bundle
from ..weighter.native import GPMWeighter, UniformWeighter, materialize_equal_grid
from ..weighter.correction import compute_factors, compute_prox_scores


class MaterializationError(ValueError):
    pass


def validate_field(
    field: np.ndarray,
    grid: gpd.GeoDataFrame,
    source_regions: gpd.GeoDataFrame,
    *,
    source_column: str,
    demand_column: str = "Demand (MVA)",
) -> np.ndarray:
    values = np.asarray(field, dtype=np.float64)
    if values.shape != (len(grid),) or not np.isfinite(values).all() or (values < 0).any():
        raise MaterializationError("field must be a finite non-negative grid vector")
    if "zero_mask" in grid and np.count_nonzero(values[grid["zero_mask"].to_numpy(bool)]):
        raise MaterializationError("field violates Z exact-zero")
    demand = dict(zip(source_regions[source_column].astype(str), source_regions[demand_column].astype(float), strict=True))
    keys = grid[source_column].astype(str).to_numpy()
    for source, expected in demand.items():
        if not np.isclose(values[keys == source].sum(), expected, rtol=1e-8, atol=1e-8):
            raise MaterializationError(f"field violates source conservation: {source}")
    return values


def materialize_assignment(
    grid: gpd.GeoDataFrame,
    targets: gpd.GeoDataFrame,
    *,
    working_crs: str,
) -> np.ndarray:
    assignment = compute_vd_assignment(grid, targets, config={"working_crs": working_crs})
    if assignment.shape != (len(grid),) or (assignment < 0).any() or (assignment >= len(targets)).any():
        raise MaterializationError("VD assignment is invalid")
    return assignment


def materialize_native_fields(bundle: GeneratorDataBundle) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    features = bundle.params["features"]
    for region in bundle.regions:
        grid, _ = bundle.grids[region]
        source = bundle.source_regions[region]
        targets = bundle.stations[region]
        uniform = UniformWeighter({"source_column": bundle.source_column, "demand_column": "Demand (MVA)"}).compute(grid, targets, source).weights
        gpm = GPMWeighter(
            {
                "mode": bundle.params["gpm"]["mode"],
                "proportion_columns": features["landuse_columns"],
                "source_feature_columns": features["source_columns"],
                "source_column": bundle.source_column,
                "demand_column": "Demand (MVA)",
            }
        ).compute(grid, targets, source).weights
        assignment = materialize_assignment(grid, targets, working_crs=bundle.params["crs"]["working"])
        equal = materialize_equal_grid(grid, uniform, assignment, source_column=bundle.source_column)
        factors = factor_bundle(
            grid,
            bundle.ntl[region],
            bundle.proximity[region],
            source_key=bundle.source_column,
        )
        output[region] = {
            "Uni": validate_field(uniform, grid, source, source_column=bundle.source_column),
            "GPM": validate_field(gpm, grid, source, source_column=bundle.source_column),
            "EqualGrid": validate_field(equal, grid, source, source_column=bundle.source_column),
            "assignment": assignment,
            "factors": factors,
        }
    return output


def materialize_candidate(
    definition: Mapping[str, Any],
    base_field: np.ndarray,
    factors: Mapping[str, np.ndarray],
    grid: gpd.GeoDataFrame,
    source_regions: gpd.GeoDataFrame,
    *,
    source_column: str,
) -> np.ndarray:
    operator = str(definition["operator"])
    auxiliary = str(definition["auxiliary"])
    if operator == "base":
        result = np.asarray(base_field, dtype=float).copy()
    else:
        signal = {"ntl": "N", "proximity": "P", "ntl_proximity": "NP"}[auxiliary]
        if operator == "multiply":
            result = apply_standard_multiplicative(base_field, factors[signal], grid, source_key=source_column)
        elif operator == "add":
            result = apply_additive(base_field, factors[signal], grid, source_key=source_column)
        else:
            raise MaterializationError(f"unknown candidate operator: {operator}")
    return validate_field(result, grid, source_regions, source_column=source_column)


def sweep_field(base, grid, stations, ntl, ntl_factor, *, source_column, working_crs, parameter, value):
    """四类后处理扫描的纯数值路径；定位与调度不进入本函数。"""
    if parameter == "alpha":
        factor = np.maximum(1.0 + value * (ntl_factor - 1.0), 0.0)
        return apply_standard_multiplicative(base, factor, grid, source_key=source_column)
    if parameter in {"gamma", "beta"}:
        gamma = value if parameter == "gamma" else value * 2.0
        scores = compute_prox_scores(grid, stations, gamma, working_crs=working_crs)
        _, proximity = compute_factors(grid, ntl, scores, source_key=source_column)
        factor = proximity if parameter == "gamma" else np.maximum(1.0 + value * (ntl_factor - 1.0), 0.0) * proximity
        return apply_standard_multiplicative(base, factor, grid, source_key=source_column)
    if parameter != "kappa":
        raise MaterializationError(f"未知扫描参数：{parameter}")
    result = np.zeros_like(base)
    for _, block in grid.groupby(source_column, sort=False):
        indexes = block.index.to_numpy(dtype=np.int64)
        for column in ("covered_mask", "unknown_mask"):
            selected = indexes[block[column].to_numpy(bool)]
            if not len(selected):
                continue
            mass = float(base[selected].sum())
            powered = np.power(np.maximum(base[selected], 0.0), value)
            result[selected] = mass * powered / powered.sum() if powered.sum() > 0 else base[selected]
    return result
