"""Capacity/influence-weighted Voronoi disaggregation without an LP dependency."""

from __future__ import annotations

from typing import Optional

import geopandas as gpd
import numpy as np
from scipy.spatial.distance import cdist

from ..base import AllocationResult, BaseAllocator
from ..registry import allocator_registry


def influence_matrix(
    grid_coords: np.ndarray,
    target_coords: np.ndarray,
    cluster_labels: np.ndarray,
    target_weights: np.ndarray,
    *,
    method: str = "civd",
    distance_floor: float = 1e-10,
    distance_matrix: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(influence, ordered_cluster_labels)``.

    CIVD takes ``max(weight / distance)`` within each cluster; IVD takes the
    sum.  The former Pyomo model had only one simplex constraint per grid row,
    so its optimum is exactly the stable row-wise argmax computed here.
    """

    grid = np.asarray(grid_coords, dtype=float)
    target = np.asarray(target_coords, dtype=float)
    labels = np.asarray(cluster_labels)
    weights = np.asarray(target_weights, dtype=float)
    if grid.ndim != 2 or target.ndim != 2 or grid.shape[1:] != (2,) or target.shape[1:] != (2,):
        raise ValueError("grid_coords and target_coords must be (n, 2)")
    if len(target) == 0 or labels.shape != (len(target),) or weights.shape != (len(target),):
        raise ValueError("target coordinates, labels and weights must be non-empty and aligned")
    if not np.isfinite(grid).all() or not np.isfinite(target).all() or not np.isfinite(weights).all():
        raise ValueError("CIVD inputs must be finite")
    if (weights <= 0).any():
        raise ValueError("CIVD target weights must be strictly positive")
    if method not in {"civd", "ivd"}:
        raise ValueError("method must be 'civd' or 'ivd'")
    if not np.isfinite(distance_floor) or distance_floor <= 0:
        raise ValueError("distance_floor must be positive and finite")
    distances = (
        np.asarray(distance_matrix, dtype=float).copy()
        if distance_matrix is not None
        else cdist(grid, target)
    )
    if distances.shape != (len(grid), len(target)) or not np.isfinite(distances).all():
        raise ValueError("distance_matrix has an invalid shape or value")
    distances = np.maximum(distances, distance_floor)
    ordered_labels = np.unique(labels)
    result = np.empty((len(grid), len(ordered_labels)), dtype=float)
    ratios = weights[None, :] / distances
    for index, label in enumerate(ordered_labels):
        selected = ratios[:, labels == label]
        result[:, index] = selected.max(axis=1) if method == "civd" else selected.sum(axis=1)
    return result, ordered_labels


@allocator_registry.register("civd", description="Capacity-influenced Voronoi disaggregation")
class CIVDAllocator(BaseAllocator):
    def allocate(
        self,
        grid_gdf: gpd.GeoDataFrame,
        target_gdf: gpd.GeoDataFrame,
        weights: Optional[np.ndarray] = None,
        source_gdf: Optional[gpd.GeoDataFrame] = None,
        **kwargs,
    ) -> AllocationResult:
        working_crs = self.config.get("working_crs", "EPSG:3857")
        output_crs = self.config.get("output_crs", "EPSG:4326")
        cluster_column = self.config.get("cluster_label_column", "cluster_label")
        capacity_column = self.config.get("capacity_column")
        method = self.config.get("method", "civd")
        grid = grid_gdf.to_crs(working_crs).copy()
        target = target_gdf.to_crs(working_crs).copy()
        grid_coords = np.column_stack((grid.geometry.x.to_numpy(), grid.geometry.y.to_numpy()))
        target_coords = np.column_stack((target.geometry.x.to_numpy(), target.geometry.y.to_numpy()))
        labels = (
            target_gdf[cluster_column].to_numpy()
            if cluster_column in target_gdf.columns
            else np.arange(len(target_gdf))
        )
        if weights is not None and len(weights) == len(target_gdf):
            target_weights = np.asarray(weights, dtype=float)
        elif capacity_column and capacity_column in target_gdf.columns:
            target_weights = np.maximum(
                target_gdf[capacity_column].to_numpy(dtype=float), 1e-10
            )
        else:
            target_weights = np.ones(len(target_gdf), dtype=float)
        influence, ordered_labels = influence_matrix(
            grid_coords,
            target_coords,
            labels,
            target_weights,
            method=method,
            distance_floor=float(self.config.get("distance_floor", 1e-10)),
            distance_matrix=kwargs.get("distance_matrix"),
        )
        assignment = np.argmax(influence, axis=1).astype(int)
        sorted_influence = np.sort(influence, axis=1)
        confidence = (
            np.ones(len(grid), dtype=float)
            if influence.shape[1] == 1
            else (sorted_influence[:, -1] - sorted_influence[:, -2])
            / np.maximum(sorted_influence[:, -1], 1e-12)
        )
        grid["assigned_target"] = assignment
        regions = grid.dissolve(by="assigned_target")
        regions["geometry"] = regions.convex_hull
        regions = regions.to_crs(output_crs).reset_index()
        return AllocationResult(
            assignment=assignment,
            assignment_column_name="assigned_target",
            voronoi_gdf=regions[["assigned_target", "geometry"]],
            assignment_gdf=grid.to_crs(output_crs),
            confidence=confidence,
            metadata={
                "method": method,
                "n_grid": len(grid_gdf),
                "n_target": len(target_gdf),
                "n_clusters": len(ordered_labels),
                "cluster_labels": ordered_labels.tolist(),
                "solver": "analytic_rowwise_argmax",
            },
        )

    @property
    def requires_weights(self) -> bool:
        return False


