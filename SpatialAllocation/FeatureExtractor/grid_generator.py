"""
Grid generator - generates a regular grid of points from a region polygon

Logic adapted from:
- the pure grid-generation part of GenerateGrid.generate_grid() (excluding landuse)
- the inline calculate_step_size() from notebook 002
"""
import numpy as np
from shapely.ops import unary_union
from shapely.geometry import Point
import geopandas as gpd


def calculate_step_size(target_points: int, regions: gpd.GeoDataFrame,
                        min_step_size: float = 10) -> float:
    """
    Automatically computes the grid step size from the target point count and region area.

    Formula: step = floor( sqrt( bbox_area / (target_points / fill_ratio) ) / 10 ) * 10

    Args:
        target_points: target number of grid points
        regions: region polygon GeoDataFrame (any CRS, converted internally)
        min_step_size: minimum step size in meters, default 10

    Returns:
        step_size_m: grid step size in meters, rounded down to the nearest multiple of 10
    """
    regions_m = regions.to_crs("EPSG:3857")
    boundary_polygon_m = unary_union(regions_m["geometry"])
    minx, miny, maxx, maxy = boundary_polygon_m.bounds

    box_area = (maxx - minx) * (maxy - miny)
    polygon_area = boundary_polygon_m.area

    if box_area == 0 or polygon_area == 0:
        return min_step_size

    area_ratio = polygon_area / box_area
    adjusted_target = target_points / area_ratio
    ideal_step_size = np.sqrt(box_area / adjusted_target)
    step_size_m = np.floor(ideal_step_size / 10) * 10

    return max(min_step_size, step_size_m)


def generate_base_grid(
    polygons: gpd.GeoDataFrame,
    step_size_m: float,
    crs_project: str = "EPSG:3857",
) -> gpd.GeoDataFrame:
    """
    Generates a regular grid of points within the boundary polygon (no feature computation).

    Args:
        polygons: region polygon GeoDataFrame (with a geometry column, any CRS)
        step_size_m: grid step size in meters
        crs_project: projected coordinate system used to generate the grid, default EPSG:3857

    Returns:
        grid_gdf: GeoDataFrame (CRS=EPSG:4326), containing the geometry column
            and the region column inherited from the sjoin
    """
    # 1. Project to a metric coordinate system
    polygons_m = polygons.to_crs(crs_project)
    boundary_polygon_m = unary_union(polygons_m["geometry"])

    # 2. Generate grid points
    minx, miny, maxx, maxy = boundary_polygon_m.bounds
    grid_x, grid_y = np.meshgrid(
        np.arange(minx, maxx, step_size_m),
        np.arange(miny, maxy, step_size_m),
    )
    grid_points = [Point(x, y) for x, y in zip(grid_x.flatten(), grid_y.flatten())]

    # 3. Spatial join: keep points inside the polygon, inherit the region attribute
    grid_gdf = gpd.GeoDataFrame(geometry=grid_points, crs=crs_project)
    grid_gdf = gpd.sjoin(grid_gdf, polygons_m, how="inner", predicate="within")
    grid_gdf.rename(columns={"index_right": "index_region"}, inplace=True)
    grid_gdf = grid_gdf.drop_duplicates(subset=["geometry"])
    grid_gdf = grid_gdf.reset_index(drop=True)

    # 4. Convert back to WGS84
    grid_gdf = grid_gdf.to_crs("EPSG:4326")

    return grid_gdf
