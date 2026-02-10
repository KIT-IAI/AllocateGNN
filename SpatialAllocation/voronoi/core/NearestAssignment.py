import numpy as np
from scipy.spatial.distance import cdist
import geopandas as gpd
import pandas as pd


def simple_nearest_assignment(gdf_grid, gdf_point, sub_columns=None):

    gdf_grid = gdf_grid.to_crs("EPSG:3857").copy()
    gdf_point = gdf_point.to_crs("EPSG:3857").copy()

    # 如果没有提供 sub_columns，执行全局最近邻分配
    if not sub_columns:
        print("- Calculating global nearest assignments...")
        return gpd.sjoin_nearest(gdf_grid, gdf_point, how='left')

    print("- Calculating grouped nearest assignments...")

    # 从字典中获取用于分组的列名
    # 我们假设字典中只有一个键值对
    point_col = list(sub_columns.keys())[0]
    grid_col = list(sub_columns.values())[0]

    # 检查列是否存在
    if point_col not in gdf_point.columns or grid_col not in gdf_grid.columns:
        raise ValueError(f"Grouping columns '{point_col}' or '{grid_col}' not found in the GeoDataFrames.")

    # 存储每个分组计算结果的列表
    all_mappings = []

    # 找到共同的分组ID
    common_groups = set(gdf_grid[grid_col].unique()) & set(gdf_point[point_col].unique())

    print(f"- Found {len(common_groups)} common groups to process.")

    # 对每个分组进行迭代
    for group_id in common_groups:
        # 筛选出当前分组的网格和点
        grid_subset = gdf_grid[gdf_grid[grid_col] == group_id]
        point_subset = gdf_point[gdf_point[point_col] == group_id]

        # 如果某个分组在筛选后为空，则跳过
        if grid_subset.empty or point_subset.empty:
            continue

        # 在分组内部进行最近邻分配
        group_mapping = gpd.sjoin_nearest(grid_subset, point_subset, how='left')
        all_mappings.append(group_mapping)

    mapping_gdf = pd.concat(all_mappings, ignore_index=True)

    return mapping_gdf