"""Torch-free Generator input assembly from an orchestration-supplied handoff."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

import geopandas as gpd
import numpy as np

from sglib.core.algorithms.chunked_distance import chunked_inverse_power_sum


LANDUSE_CATEGORIES = ("residential", "commercial", "industrial", "agricultural", "others")
AGENT_FEATURE_COLUMNS = tuple(f"lu_{name}_prop" for name in LANDUSE_CATEGORIES) + ("lu_unknown_prop",)


class HandoffRegion(Protocol):
    region: str
    grid: gpd.GeoDataFrame
    grid_metadata: Mapping[str, Any]
    landuse: Mapping[str, np.ndarray]
    built_surface: Mapping[str, np.ndarray]
    cuz_support: Mapping[str, np.ndarray]
    ntl: Mapping[str, np.ndarray]


class DataOverviewHandoff(Protocol):
    country: str
    regions_table: gpd.GeoDataFrame
    stations_table: gpd.GeoDataFrame
    regions: Sequence[HandoffRegion]
    profile: Any
    evidence: Mapping[str, Any]


class GeneratorDataError(ValueError):
    pass


@dataclass
class GeneratorRegionInputs:
    country: str
    region: str
    source_key: str
    grid: gpd.GeoDataFrame
    projected_step_m: float
    ground_step_m: float
    ntl: np.ndarray
    proximity: np.ndarray
    covered_mask: np.ndarray
    unknown_mask: np.ndarray
    zero_mask: np.ndarray
    source_key_order: tuple[str, ...]
    metadata: dict[str, Any]


@dataclass
class GeneratorDataBundle:
    country: str
    params: dict[str, Any]
    regions: tuple[str, ...]
    source_column: str
    demand_column: str
    relation_column: str
    grids: dict[str, tuple[gpd.GeoDataFrame, float]]
    ntl: dict[str, np.ndarray]
    proximity: dict[str, np.ndarray]
    rci: dict[str, np.ndarray]
    source_regions: dict[str, gpd.GeoDataFrame]
    stations: dict[str, gpd.GeoDataFrame]
    region_inputs: dict[str, GeneratorRegionInputs]


def region_zscore_log1p(values: np.ndarray, label: str = "feature") -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all() or (array < 0).any():
        raise GeneratorDataError(f"{label}: values must be finite and non-negative")
    transformed = np.log1p(array)
    standard_deviation = float(transformed.std(ddof=0))
    result = np.zeros_like(transformed) if standard_deviation == 0 else (transformed - transformed.mean()) / standard_deviation
    if not np.isfinite(result).all():
        raise GeneratorDataError(f"{label}: z-score produced non-finite values")
    return result


def compute_proximity_scores(
    grid: gpd.GeoDataFrame,
    stations: gpd.GeoDataFrame,
    *,
    working_crs: str,
    gamma: float = 2.0,
    clamp_km: float = 0.01,
    max_workspace_bytes: int,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    if stations.empty:
        raise GeneratorDataError("proximity requires at least one station")
    projected_grid = grid.to_crs(working_crs)
    projected_stations = stations.to_crs(working_crs)
    grid_xy = np.column_stack([projected_grid.geometry.x, projected_grid.geometry.y])
    station_xy = np.column_stack([projected_stations.geometry.x, projected_stations.geometry.y])
    try:
        values, peak_workspace_bytes, chunk_rows = chunked_inverse_power_sum(
            grid_xy,
            station_xy,
            gamma=gamma,
            distance_scale=1000.0,
            clamp_distance=clamp_km,
            max_workspace_bytes=max_workspace_bytes,
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise GeneratorDataError(f"invalid proximity workspace: {exc}") from exc
    if not np.isfinite(values).all() or (values <= 0).any():
        raise GeneratorDataError("proximity field is not finite and positive")
    if not return_diagnostics:
        return values
    return values, {
        "peak_workspace_bytes": peak_workspace_bytes,
        "chunk_rows": chunk_rows,
        "n_grid_cells": len(grid_xy),
        "n_stations": len(station_xy),
    }


def _array(value: Mapping[str, np.ndarray], key: str, shape: tuple[int, ...]) -> np.ndarray:
    result = np.asarray(value[key])
    if result.shape != shape:
        raise GeneratorDataError(f"array {key} has shape {result.shape}, expected {shape}")
    return result


def assemble_from_handoff(
    handoff: DataOverviewHandoff,
    generator_config: Mapping[str, Any],
    *,
    selected_regions: Sequence[str] | None = None,
) -> GeneratorDataBundle:
    country = str(handoff.country)
    params = deepcopy(dict(generator_config))
    configured_regions = tuple(map(str, params.get("regions", ())))
    handoff_regions = tuple(item.region for item in handoff.regions)
    if configured_regions != handoff_regions:
        raise GeneratorDataError("Generator region order differs from DataOverview handoff")
    observed_regions = configured_regions if selected_regions is None else tuple(map(str, selected_regions))
    if not observed_regions or observed_regions != tuple(region for region in configured_regions if region in set(observed_regions)):
        raise GeneratorDataError("selected regions must be a non-empty configured-order subset")
    params["regions"] = list(observed_regions)
    contract = handoff.profile.station_contract
    source_column = str(contract["source_key"])
    demand_column = str(contract["region_demand_column"])
    station_region_column = str(contract["region_column"])
    if not {source_column, demand_column, "geometry"} <= set(handoff.regions_table):
        raise GeneratorDataError("DataOverview region table lacks Generator source columns")
    if not {station_region_column, "geometry", "station_id"} <= set(handoff.stations_table):
        raise GeneratorDataError("DataOverview station table lacks Generator target columns")
    corrections = params.get("corrections", {})
    try:
        distance_workspace = params["execution"]["distance_workspace"]
        max_workspace_bytes = int(distance_workspace["max_workspace_bytes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise GeneratorDataError(
            "Generator config lacks the resolved engineering distance workspace"
        ) from exc
    if (
        max_workspace_bytes <= 0
        or distance_workspace.get("strategy") != "country_neutral_chunked_rows_v1"
    ):
        raise GeneratorDataError("Generator distance workspace authority is invalid")
    threshold = float(corrections.get("rci_threshold", 0.5))
    region_inputs: dict[str, GeneratorRegionInputs] = {}
    grids: dict[str, tuple[gpd.GeoDataFrame, float]] = {}
    ntl_fields: dict[str, np.ndarray] = {}
    proximity_fields: dict[str, np.ndarray] = {}
    rci_fields: dict[str, np.ndarray] = {}
    sources: dict[str, gpd.GeoDataFrame] = {}
    targets: dict[str, gpd.GeoDataFrame] = {}
    for item in (item for item in handoff.regions if item.region in set(observed_regions)):
        grid = item.grid.copy().reset_index(drop=True)
        support = item.cuz_support
        n_cells = len(grid)
        features = _array(support, "features", (n_cells, 6)).astype(np.float64)
        if not np.isfinite(features).all():
            raise GeneratorDataError(f"{country}/{item.region}: non-finite C/U/Z features")
        masks = {
            key: _array(support, key, (n_cells,)).astype(bool)
            for key in ("covered_mask", "unknown_mask", "zero_mask")
        }
        if not np.all(sum(mask.astype(np.int8) for mask in masks.values()) == 1):
            raise GeneratorDataError(f"{country}/{item.region}: invalid C/U/Z partition")
        if np.count_nonzero(features[masks["zero_mask"]]):
            raise GeneratorDataError(f"{country}/{item.region}: Z rows are not exact zero")
        for index, column in enumerate(AGENT_FEATURE_COLUMNS):
            grid[column] = features[:, index]
        grid["built_fraction"] = _array(support, "built_fraction", (n_cells,)).astype(np.float64)
        for key, values in masks.items():
            grid[key] = values
        categories = np.asarray(LANDUSE_CATEGORIES, dtype=object)
        grid["landuse"] = categories[np.argmax(features[:, :5], axis=1)]
        grid.loc[~grid["covered_mask"], "landuse"] = None
        source_order = tuple(map(str, support["source_key_order"].tolist()))
        grid_keys = grid[source_column].astype(str).to_numpy()
        if not np.array_equal(grid_keys, support["source_keys"].astype(str)):
            raise GeneratorDataError(f"{country}/{item.region}: handoff row identity differs")
        source = handoff.regions_table.loc[
            handoff.regions_table[source_column].astype(str).isin(source_order)
        ].copy().reset_index(drop=True)
        station = handoff.stations_table.loc[
            handoff.stations_table[station_region_column].astype(str).isin(source_order)
        ].copy().reset_index(drop=True)
        if len(source) != len(source_order) or station.empty:
            raise GeneratorDataError(f"{country}/{item.region}: source/target coverage differs")
        source[source_column] = source[source_column].astype(str)
        source["Demand (MVA)"] = source[demand_column].astype(float)
        source["ITL3"] = source[source_column]
        station["ITL3"] = station[station_region_column].astype(str)
        grid["ITL3"] = grid[source_column].astype(str)
        ntl_blob = _array(item.ntl, "data", (n_cells, 1)).astype(np.float64)[:, 0]
        proximity, proximity_diagnostics = compute_proximity_scores(
            grid,
            station,
            working_crs=str(handoff.profile.crs["working"]),
            gamma=float(corrections.get("proximity_gamma", 2.0)),
            clamp_km=float(corrections.get("dist_clamp_km", 0.01)),
            max_workspace_bytes=max_workspace_bytes,
            return_diagnostics=True,
        )
        rci = (
            grid["lu_residential_prop"].to_numpy(float)
            + grid["lu_commercial_prop"].to_numpy(float)
            + grid["lu_industrial_prop"].to_numpy(float)
        ) > threshold
        rci &= masks["covered_mask"]
        projected_step = float(item.grid_metadata["projected_step_m"])
        region_input = GeneratorRegionInputs(
            country=country,
            region=item.region,
            source_key=source_column,
            grid=grid,
            projected_step_m=projected_step,
            ground_step_m=float(item.grid_metadata["target_ground_step_m"]),
            ntl=ntl_blob,
            proximity=proximity,
            covered_mask=masks["covered_mask"],
            unknown_mask=masks["unknown_mask"],
            zero_mask=masks["zero_mask"],
            source_key_order=source_order,
            metadata={
                "grid": dict(item.grid_metadata),
                "support": "C/U/Z",
                "proximity": {
                    **proximity_diagnostics,
                    "strategy": str(distance_workspace["strategy"]),
                    "max_workspace_bytes": max_workspace_bytes,
                    "authority_sha256": str(
                        distance_workspace["authority_sha256"]
                    ),
                },
            },
        )
        region_inputs[item.region] = region_input
        grids[item.region] = (grid, projected_step)
        ntl_fields[item.region] = ntl_blob
        proximity_fields[item.region] = proximity
        rci_fields[item.region] = rci
        sources[item.region] = source
        targets[item.region] = station
    return GeneratorDataBundle(
        country=country,
        params=params,
        regions=observed_regions,
        source_column=source_column,
        demand_column=demand_column,
        relation_column="ITL3",
        grids=grids,
        ntl=ntl_fields,
        proximity=proximity_fields,
        rci=rci_fields,
        source_regions=sources,
        stations=targets,
        region_inputs=region_inputs,
    )


__all__ = [
    "AGENT_FEATURE_COLUMNS",
    "GeneratorDataBundle",
    "GeneratorDataError",
    "GeneratorRegionInputs",
    "assemble_from_handoff",
    "compute_proximity_scores",
    "region_zscore_log1p",
]
