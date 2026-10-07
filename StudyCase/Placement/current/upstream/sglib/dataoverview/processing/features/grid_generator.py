"""Bounded equal-area grid generation (contract B+)."""

from __future__ import annotations

from dataclasses import dataclass
import math

import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from shapely.ops import unary_union


@dataclass(frozen=True)
class GridDesign:
    grid_policy: str
    target_points: int
    area_crs: str
    generation_crs: str
    storage_crs: str
    area_m2: float
    target_ground_step_m: float
    projected_step_m: float
    scale_factor_method: str
    reference_latitude: float
    clamp_branch: str


def design_grid(
    target_points: int,
    regions: gpd.GeoDataFrame,
    *,
    area_crs: str,
    min_ground_step_m: float = 100.0,
    max_ground_step_m: float = 500.0,
    generation_crs: str = "EPSG:3857",
    storage_crs: str = "EPSG:4326",
) -> GridDesign:
    if target_points <= 0:
        raise ValueError("target_points must be positive")
    if min_ground_step_m <= 0 or max_ground_step_m < min_ground_step_m:
        raise ValueError("invalid ground-step bounds")
    if regions.empty or regions.crs is None:
        raise ValueError("regions must be a non-empty georeferenced table")
    area_geometry = unary_union(regions.to_crs(area_crs).geometry)
    area_m2 = float(area_geometry.area)
    if not math.isfinite(area_m2) or area_m2 <= 0:
        raise ValueError("regions have no positive equal-area footprint")
    raw_step = math.sqrt(area_m2 / float(target_points))
    target_step = float(np.clip(raw_step, min_ground_step_m, max_ground_step_m))
    clamp = "minimum" if raw_step < min_ground_step_m else "maximum" if raw_step > max_ground_step_m else "none"
    centroid = unary_union(regions.to_crs(storage_crs).geometry).centroid
    latitude = float(centroid.y)
    cosine = math.cos(math.radians(latitude))
    if cosine <= 0:
        raise ValueError(f"invalid reference latitude: {latitude}")
    projected_step = target_step / cosine
    return GridDesign(
        grid_policy="bounded_equal_area_budget_v1",
        target_points=int(target_points),
        area_crs=area_crs,
        generation_crs=generation_crs,
        storage_crs=storage_crs,
        area_m2=area_m2,
        target_ground_step_m=target_step,
        projected_step_m=projected_step,
        scale_factor_method="web_mercator_secant_at_region_centroid",
        reference_latitude=latitude,
        clamp_branch=clamp,
    )


def calculate_step_size(
    target_points: int,
    regions: gpd.GeoDataFrame,
    min_step_size: float = 100.0,
    *,
    max_step_size: float = 500.0,
    area_crs: str,
) -> float:
    """Return the target ground step under the B+ contract."""

    return design_grid(
        target_points,
        regions,
        area_crs=area_crs,
        min_ground_step_m=min_step_size,
        max_ground_step_m=max_step_size,
    ).target_ground_step_m


def generate_base_grid(
    polygons: gpd.GeoDataFrame,
    step_size_m: float,
    crs_project: str = "EPSG:3857",
) -> gpd.GeoDataFrame:
    """Generate a regular lattice and retain points within source polygons."""

    if not math.isfinite(step_size_m) or step_size_m <= 0:
        raise ValueError("step_size_m must be finite and positive")
    polygons_m = polygons.to_crs(crs_project)
    boundary = unary_union(polygons_m.geometry)
    minx, miny, maxx, maxy = boundary.bounds
    grid_x, grid_y = np.meshgrid(
        np.arange(minx, maxx, step_size_m),
        np.arange(miny, maxy, step_size_m),
    )
    points = [Point(x, y) for x, y in zip(grid_x.ravel(), grid_y.ravel(), strict=True)]
    grid = gpd.GeoDataFrame(geometry=points, crs=crs_project)
    grid = gpd.sjoin(grid, polygons_m, how="inner", predicate="within")
    grid = grid.rename(columns={"index_right": "index_region"})
    grid = grid.drop_duplicates(subset=["geometry"]).reset_index(drop=True)
    return grid.to_crs("EPSG:4326")


def regenerate_grid_reference(
    polygons: gpd.GeoDataFrame,
    *,
    target_points: int,
    min_ground_step_m: float,
    max_ground_step_m: float,
    area_crs: str,
    generation_crs: str = "EPSG:3857",
) -> tuple[gpd.GeoDataFrame, GridDesign]:
    design = design_grid(
        target_points,
        polygons,
        area_crs=area_crs,
        min_ground_step_m=min_ground_step_m,
        max_ground_step_m=max_ground_step_m,
        generation_crs=generation_crs,
    )
    return generate_base_grid(polygons, design.projected_step_m, generation_crs), design
