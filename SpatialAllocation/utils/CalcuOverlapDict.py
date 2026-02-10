import pandas as pd

def calcu_overlap_dict(polygons_gdf, points_gdf, surfaces, weight="equal"):
    # Initialize overlap area matrix
    overlap_matrix = pd.DataFrame(index=polygons_gdf.index, columns=surfaces.index, data=0.0)

    # Calculate overlap areas
    for i, subpolygon in polygons_gdf.iterrows():
        for j, region_polygon in surfaces.iterrows():
            overlap_area = 0.0
            if subpolygon['geometry'].intersects(region_polygon['geometry']):
                intersection = subpolygon['geometry'].intersection(region_polygon['geometry'])
                if intersection.is_empty:
                    overlap_area = 0.0
                elif intersection.geom_type == 'Polygon':
                    overlap_area = intersection.area
                elif intersection.geom_type == 'MultiPolygon':
                    overlap_area = sum([poly.area for poly in intersection.geoms])
            overlap_matrix.loc[i, j] = overlap_area

    # Normalize the overlap matrix
    overlap_matrix_normalized = overlap_matrix.div(overlap_matrix.sum(axis=0), axis=1)

    polygons_gdf['overlap_dict'] = [
        {region_idx: overlap_matrix_normalized.loc[i, region_idx]
         for region_idx in overlap_matrix_normalized.columns
         if overlap_matrix_normalized.loc[i, region_idx] > 0}
        for i in polygons_gdf.index
    ]

    # Initialize the overlap_dict column in resulting_points_gdf
    points_gdf['overlap_dict'] = [{} for _ in range(len(points_gdf))]

    # Iterate over resulting_subpolygons_gdf, assign overlap_dict to corresponding points
    for idx, row in polygons_gdf.iterrows():
        if "points" not in row.keys():
            if "cluster_label" in points_gdf.columns:
                row['points'] = [point for point in points_gdf.index if
                                 idx == points_gdf.at[point, 'cluster_label']]
            else:
                row['points'] = [point for point in points_gdf.index if row['geometry'].contains(points_gdf.at[point, 'geometry'])]
        overlap_dict = row['overlap_dict']
        points = row['points'] if isinstance(row['points'], list) else [row['points']]

        # Calculate weights based on selected method
        if weight == "equal":
            # Equal weights for all points
            point_weights = {point_idx:1.0 / len(points) for point_idx in points}

        elif weight == "distance":
            # Calculate centroid of the polygon
            centroid = row['geometry'].centroid

            # Calculate inverse distances (closer points get higher weights)
            distances = {}
            for point_idx in points:
                point_geom = points_gdf.at[point_idx, 'geometry']
                # Calculate Euclidean distance between point and centroid
                distance = point_geom.distance(centroid)
                # Use inverse distance (add small epsilon to avoid division by zero)
                distances[point_idx] = 1.0 / (distance + 1e-10)

            # Normalize weights to sum to 1
            total_inv_distance = sum(distances.values())
            point_weights = {point_idx:distances[point_idx] / total_inv_distance
                             for point_idx in points}

        else:
            raise ValueError(f"Invalid weight method: {weight}. Choose 'equal' or 'distance'.")

        for point_idx in points:
            temp_dict = points_gdf.at[point_idx, 'overlap_dict'].copy()
            for region_idx, value in overlap_dict.items():
                if region_idx not in temp_dict:
                    temp_dict[region_idx] = 0.0
                temp_dict[region_idx] += value * point_weights[point_idx]
            points_gdf.at[point_idx, 'overlap_dict'] = temp_dict

    return polygons_gdf['overlap_dict'], points_gdf["overlap_dict"]