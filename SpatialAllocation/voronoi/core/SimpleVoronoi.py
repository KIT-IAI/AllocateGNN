from .NearestAssignment import simple_nearest_assignment


def simple_voronoi(gdf_point, gdf_grid, sub_columns=None, with_landuse=False):
    """

    :param gdf_point:
    :param gdf_polygon:
    :param sub_columns:
    :param step_size_m:
    :param target_points:
    :return:
    """
    if sub_columns is None:
        gdf_point = gdf_point[["cluster_label", "geometry"]].copy()
        gdf_grid = gdf_grid[["geometry"]].copy()
    else:
        # 如果提供了 sub_columns，则只保留指定的列
        gdf_point = gdf_point[list(sub_columns.keys()) + ["cluster_label", "geometry"]].copy()
        gdf_grid = gdf_grid[list(sub_columns.values()) + ["geometry"]].copy()


    print("Step 2: Assigning nearest assignments...")
    mapping_gdf = simple_nearest_assignment(gdf_grid, gdf_point, sub_columns=sub_columns)

    print("Step 3: Aggregating grid cells into Voronoi polygons...")
    # voronoi_gdf = mapping_gdf.dissolve(by='index_right')
    # voronoi_gdf = voronoi_gdf.reset_index()
    # voronoi_gdf = voronoi_gdf.rename(columns={'index_right': 'point_index'})
    voronoi_gdf = mapping_gdf.dissolve(by='cluster_label')
    voronoi_gdf['geometry'] = voronoi_gdf.convex_hull
    voronoi_gdf.to_crs("EPSG:4326", inplace=True)

    return voronoi_gdf[['geometry']]