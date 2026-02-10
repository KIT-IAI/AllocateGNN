import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from scipy.spatial import Voronoi
from ClusterBasedVoronoi.voronoi.voronoi_utils import calcu_overlap_dict


__all__ = ['simple_voronoi']


def simple_voronoi(surfaces, points):
    boundary_polygon = unary_union(surfaces['geometry'])
    points_xy = [[x, y] for x, y in zip(points.geometry.x.values, points.geometry.y.values)]
    polygons_gdf, points_gdf = _create_subpolygon(boundary_polygon, points_xy)
    polygons_gdf['overlap_dict'], points["overlap_dict"] = calcu_overlap_dict(polygons_gdf, points_gdf, surfaces)


    return polygons_gdf, points


def _create_subpolygon(polygon, points):
    """
    Create sub-polygons within a given polygon using a Voronoi diagram, and allocate points to these sub-polygons.

    Parameters:
    polygon (shapely.geometry.Polygon): The main polygon within which sub-polygons are to be created.
    points (list of tuples or array-like): Coordinates of the points to generate the Voronoi diagram.

    Returns:
    polygons_gdf(geopandas.GeoDataFrame): A GeoDataFrame containing sub-polygons and the indices of the points they include.
    points_gdf(geopandas.GeoDataFrame): A GeoDataFrame containing points and the indices of the sub-polygons they belong to.

    The function follows these steps:
    1. Generate a Voronoi diagram from the provided points.
    2. Create sub-polygons from Voronoi regions, intersecting them with the main polygon.
        a. Creates a Voronoi diagram from the given points (PF_substation_xy).
        b. Iterates over each point
        c. If the vertices of the subpolygon corresponding to this point do not contain -1 (the infinity point),
           then this subpolygon is saved. And intersect these subpolygons with the original polygon
           (regions_boundary_polygon) to make sure they are within the original polygon.
        d. If the vertices of the subpolygon corresponding to this point contain -1, then this point is recorded and skipped.
        e. Creates a GeoDataFrame to store the subpolygons and the associated points, calculates the union of all
           subpolygons, and calculates the complement between the original polygon (ethos_regions) and the union.
        f. Breaks the complement into individual polygons and saves them in a new GeoDataFrame.
        g. The skipped points are to be assigned to the nearest complementary polygon
    3. Calculate the difference between the main polygon and the union of sub-polygons.
    4. Assign skipped points to the nearest complement sub-polygon.
    5. Create GeoDataFrames for sub-polygons and points, tracking their relationships.
    """

    subpolygons = []  # List to store sub-polygons
    points_indices = []  # List to store indices of the points
    skipped_points = []  # List to store points that are skipped
    allocated_points = []  # List to store allocated points

    vor = Voronoi(points)  # Generate Voronoi diagram
    for point_index, region_index in enumerate(vor.point_region):
        region = vor.regions[region_index]
        if not -1 in region and len(region) > 0:
            subpolygon = Polygon([vor.vertices[i] for i in region])
            if not subpolygon.within(polygon):
                subpolygon = subpolygon.intersection(polygon)  # Intersect with main polygon if not within
            subpolygons.append(subpolygon)
            points_indices.append(point_index)
            allocated_points.append(Point(vor.points[point_index]))
        else:
            skipped_points.append(Point(vor.points[point_index]))  # Record skipped points

    # Create GeoDataFrame for sub-polygons and allocated points
    subpolygons_gdf = gpd.GeoDataFrame({'geometry': subpolygons, 'points': allocated_points})
    voronoi_union = unary_union(subpolygons_gdf['geometry'])  # Union of all sub-polygons
    complement = polygon.difference(voronoi_union)  # Difference between main polygon and union of sub-polygons

    # Decompose complement into individual polygons
    complement_polygons = []
    if complement.geom_type == 'MultiPolygon':
        for poly in complement.geoms:
            complement_polygons.append(poly)
    else:
        complement_polygons.append(complement)

    # Create GeoDataFrame for complement polygons
    complement_gdf = gpd.GeoDataFrame({'geometry': complement_polygons})

    while len(skipped_points) > 0:
        # Assign skipped points to the nearest complement sub-polygon
        assigned_polygons = {i: [] for i in range(len(complement_gdf))}

        for skipped_point in skipped_points:
            distances = complement_gdf.distance(skipped_point)
            nearest_polygon_index = distances.idxmin()
            assigned_polygons[nearest_polygon_index].append(skipped_point)  # Assign point to nearest complement polygon

        for ap_index, ap_points in assigned_polygons.items():
            complement_polygon = complement_gdf.loc[ap_index, 'geometry']
            for ap_point in ap_points:
                new_subpolygons_gdf = gpd.GeoDataFrame({'geometry': [complement_polygon], 'points': [ap_point]})
                subpolygons_gdf = pd.concat([subpolygons_gdf, new_subpolygons_gdf])  # Merge new sub-polygons
                skipped_points.remove(ap_point)  # Remove assigned points from skipped list

    subpolygons_gdf.reset_index(drop=True, inplace=True)

    # Create new GeoDataFrame to record sub-polygons and their point indices
    polygon_records = []
    for idx, row in subpolygons_gdf.iterrows():
        polygon = row['geometry']
        point_index = idx
        polygon_exists = False

        # Check if polygon already exists
        for record in polygon_records:
            if record['geometry'] == polygon:
                record['points'].append(point_index)
                polygon_exists = True
                break

        # If polygon doesn't exist, create new record
        if not polygon_exists:
            polygon_records.append({'geometry': polygon, 'points': [point_index]})

    polygons_gdf = gpd.GeoDataFrame(polygon_records)

    # Create new GeoDataFrame to record points and their corresponding polygon indices
    point_records = []
    for idx, row in subpolygons_gdf.iterrows():
        point = row['points']
        polygon_index = polygons_gdf[polygons_gdf['geometry'] == row['geometry']].index[0]
        point_records.append({'geometry': point, 'polygons': polygon_index})

    points_gdf = gpd.GeoDataFrame(point_records)

    return polygons_gdf, points_gdf  # Return GeoDataFrames containing sub-polygons and points

