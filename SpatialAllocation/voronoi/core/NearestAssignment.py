import numpy as np
from scipy.spatial.distance import cdist
import geopandas as gpd
import pandas as pd


def simple_nearest_assignment(gdf_grid, gdf_point, sub_columns=None):

    gdf_grid = gdf_grid.to_crs("EPSG:3857").copy()
    gdf_point = gdf_point.to_crs("EPSG:3857").copy()

    # If sub_columns is not provided, perform global nearest neighbor assignment
    if not sub_columns:
        print("- Calculating global nearest assignments...")
        return gpd.sjoin_nearest(gdf_grid, gdf_point, how='left')

    print("- Calculating grouped nearest assignments...")

    # Get column names for grouping from the dictionary
    # We assume only one key-value pair in the dictionary
    point_col = list(sub_columns.keys())[0]
    grid_col = list(sub_columns.values())[0]

    # Check if columns exist
    if point_col not in gdf_point.columns or grid_col not in gdf_grid.columns:
        raise ValueError(f"Grouping columns '{point_col}' or '{grid_col}' not found in the GeoDataFrames.")

    # List to store results for each group
    all_mappings = []

    # Find common group IDs
    common_groups = set(gdf_grid[grid_col].unique()) & set(gdf_point[point_col].unique())

    print(f"- Found {len(common_groups)} common groups to process.")

    # Iterate through each group
    for group_id in common_groups:
        # Filter grid and points for the current group
        grid_subset = gdf_grid[gdf_grid[grid_col] == group_id]
        point_subset = gdf_point[gdf_point[point_col] == group_id]

        # If a group is empty after filtering, skip it
        if grid_subset.empty or point_subset.empty:
            continue

        # Perform nearest neighbor assignment within the group
        group_mapping = gpd.sjoin_nearest(grid_subset, point_subset, how='left')
        all_mappings.append(group_mapping)

    mapping_gdf = pd.concat(all_mappings, ignore_index=True)

    return mapping_gdf