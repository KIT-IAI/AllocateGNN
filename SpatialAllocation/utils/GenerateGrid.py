import numpy as np
from shapely.ops import unary_union
from shapely.geometry import Point, Polygon
import geopandas as gpd
from .CalcuLanduse import fetch_landuse_data, calculate_landuse_proportions


def generate_grid(polygons, step_size_m, with_tags=None):
    """
    Generate a grid within boundary polygons and optionally calculate landuse type proportions
    for each grid cell.

    Parameters:
    - polygons: GeoDataFrame containing a 'geometry' column.
    - step_size_m: Grid step size in meters.
    - with_tags: Dictionary indicating which tags to compute (e.g., landuse).

    Returns:
    - grid_gdf: GeoDataFrame containing grid geometry and (optionally) landuse type proportion columns.
    """
    # 1. Convert input polygons to meter-based coordinate reference system (CRS) (EPSG:3857)
    polygons_m = polygons.to_crs("EPSG:3857")
    boundary_polygon_m = unary_union(polygons_m["geometry"])

    # 2. Get boundary extent and generate grid points
    minx, miny, maxx, maxy = boundary_polygon_m.bounds
    print("- Generating grid points...")
    grid_x, grid_y = np.meshgrid(
        np.arange(minx, maxx, step_size_m),
        np.arange(miny, maxy, step_size_m)
    )
    grid_points = [Point(x, y) for x, y in zip(grid_x.flatten(), grid_y.flatten())]

    # 3. Convert grid points to GeoDataFrame and filter to points within the boundary
    grid_gdf = gpd.GeoDataFrame(geometry=grid_points, crs="EPSG:3857")
    grid_gdf = gpd.sjoin(grid_gdf, polygons_m, how="inner", predicate="within")
    grid_gdf.rename(columns={"index_right": "index_region"}, inplace=True)
    grid_gdf = grid_gdf.drop_duplicates(subset=['geometry'])
    grid_gdf = grid_gdf.reset_index(drop=True)

    numerical_col_names = []
    categorical_col_members = {}
    grid_gdf_ori = grid_gdf.copy()
    print(grid_gdf_ori.columns)
    # 4. If needed, compute landuse data
    if "landuse" in with_tags.keys():
        print("- Fetching primary landuse...")
        grid_gdf_landuse, numerical_col_names_landuse, categorical_col_members_landuse = fetch_landuse_data(grid_gdf_ori, polygons_m, step_size_m)
        grid_gdf[list(categorical_col_members_landuse.keys())] = grid_gdf_landuse[list(categorical_col_members_landuse.keys())]
        numerical_col_names.extend(numerical_col_names_landuse)
        categorical_col_members.update(categorical_col_members_landuse)
        print("- Computing all landuse proportions...")
        grid_gdf_landuse_proportions, numerical_col_names_landuse_proportions, categorical_col_members_landuse_proportions = calculate_landuse_proportions(grid_gdf_ori, polygons_m, step_size_m)
        grid_gdf[numerical_col_names_landuse_proportions] = grid_gdf_landuse_proportions[numerical_col_names_landuse_proportions]
        numerical_col_names.extend(numerical_col_names_landuse_proportions)
        categorical_col_members.update(categorical_col_members_landuse_proportions)

    grid_gdf = grid_gdf.copy().to_crs("EPSG:4326")

    return grid_gdf, numerical_col_names, categorical_col_members