import numpy as np
from shapely.ops import unary_union
from shapely.geometry import Point, Polygon
import geopandas as gpd
from .CalcuLanduse import fetch_landuse_data, calculate_landuse_proportions


def generate_grid(polygons, step_size_m, with_tags=None):
    """
    在边界多边形内生成网格，并可选地计算每个网格单元的土地利用类型占比。

    参数:
    - polygons: GeoDataFrame，包含 'geometry' 列。
    - step_size_m: 网格步长（米）。
    - with_landuse: 布尔值，指示是否要计算土地利用占比。

    返回:
    - grid_gdf: GeoDataFrame，包含网格几何对象以及（可选的）各土地利用类型的占比列。
    """
    # 1. 将输入多边形转换为米制单位的坐标参考系统 (CRS) (EPSG:3857)
    polygons_m = polygons.to_crs("EPSG:3857")
    boundary_polygon_m = unary_union(polygons_m["geometry"])

    # 2. 获取边界范围并生成网格点
    minx, miny, maxx, maxy = boundary_polygon_m.bounds
    print("- 正在生成网格点...")
    grid_x, grid_y = np.meshgrid(
        np.arange(minx, maxx, step_size_m),
        np.arange(miny, maxy, step_size_m)
    )
    grid_points = [Point(x, y) for x, y in zip(grid_x.flatten(), grid_y.flatten())]

    # 3. 将网格点转换为GeoDataFrame，并筛选出在边界内的点
    grid_gdf = gpd.GeoDataFrame(geometry=grid_points, crs="EPSG:3857")
    grid_gdf = gpd.sjoin(grid_gdf, polygons_m, how="inner", predicate="within")
    grid_gdf.rename(columns={"index_right": "index_region"}, inplace=True)
    grid_gdf = grid_gdf.drop_duplicates(subset=['geometry'])
    grid_gdf = grid_gdf.reset_index(drop=True)

    numerical_col_names = []
    categorical_col_members = {}
    grid_gdf_ori = grid_gdf.copy()
    print(grid_gdf_ori.columns)
    # 4. 如果需要，计算土地利用数据
    if "landuse" in with_tags.keys():
        print("- 正在获取主要土地利用...")
        grid_gdf_landuse, numerical_col_names_landuse, categorical_col_members_landuse = fetch_landuse_data(grid_gdf_ori, polygons_m, step_size_m)
        grid_gdf[list(categorical_col_members_landuse.keys())] = grid_gdf_landuse[list(categorical_col_members_landuse.keys())]
        numerical_col_names.extend(numerical_col_names_landuse)
        categorical_col_members.update(categorical_col_members_landuse)
        print("- 正在计算所有土地利用占比...")
        grid_gdf_landuse_proportions, numerical_col_names_landuse_proportions, categorical_col_members_landuse_proportions = calculate_landuse_proportions(grid_gdf_ori, polygons_m, step_size_m)
        grid_gdf[numerical_col_names_landuse_proportions] = grid_gdf_landuse_proportions[numerical_col_names_landuse_proportions]
        numerical_col_names.extend(numerical_col_names_landuse_proportions)
        categorical_col_members.update(categorical_col_members_landuse_proportions)

    grid_gdf = grid_gdf.copy().to_crs("EPSG:4326")

    return grid_gdf, numerical_col_names, categorical_col_members
