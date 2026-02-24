"""
Author: Xuanhao Mu
Email: xuanhao.mu@kit.edu
Description: This module contains functions for preparing Pyomo parameters.
"""

import numpy as np
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist
from shapely.ops import unary_union
from shapely.geometry import Point
import geopandas as gpd


def generate_grid(regions, step_size_m = None, landuse_gdf = None, target_points=30000):
    """
    Generate grid points directly within the boundary polygon and optionally assign weights.

    Parameters:
    - regions: GeoDataFrame with 'geometry' and 'weights' columns.
    - step_size_m: Grid step size in meters.
    - with_weights: Boolean to indicate if weights should be assigned.
    - n_jobs: Number of jobs for parallel computation (default: -1 for all available cores).

    Returns:
    - grid_points: List of (x, y) tuples representing grid points within the polygon.
    - grid_weights (optional): List of weights for each grid point.
    """
    # Convert regions to meters (EPSG:3857)
    regions = regions.to_crs("EPSG:3857")
    boundary_polygon_m = unary_union(regions["geometry"])
    return_step_size_m = False

    # Get bounds of the boundary polygon
    minx, miny, maxx, maxy = boundary_polygon_m.bounds

    if step_size_m is None:
        print("Step size not provided, calculating optimal step size...")
        print("This function 'target_points' is experimental and estimates the step size based on the square area of the region.")
        # Calculate bounding box area
        box_area = (maxx - minx) * (maxy - miny)
        polygon_area = boundary_polygon_m.area
        area_ratio = polygon_area / box_area
        adjusted_target = target_points / area_ratio
        ideal_step_size = np.sqrt(box_area / adjusted_target)
        step_size_m = np.floor(ideal_step_size / 10) * 10
        step_size_m = max(10, step_size_m)

        print(f"Area ratio (polygon/box): {area_ratio:.2f}")
        print(f"Automatically calculated step size: {step_size_m} meters")
        return_step_size_m = True


    # Create grid points using numpy
    grid_x, grid_y = np.meshgrid(np.arange(minx, maxx, step_size_m), np.arange(miny, maxy, step_size_m))
    grid_points = [Point(x, y) for x, y in zip(grid_x.flatten(), grid_y.flatten())]

    # Convert grid points to GeoDataFrame for spatial join
    grid_gdf = gpd.GeoDataFrame(geometry=grid_points, crs="EPSG:3857")

    # Perform spatial join to find which points are inside the polygons
    grid_gdf = gpd.sjoin(grid_gdf, regions, how="inner", predicate="within")

    if landuse_gdf is not None:
        landuse_slim = landuse_gdf[['landuse', 'geometry']]
        grid_gdf = gpd.sjoin(grid_gdf.to_crs("EPSG:4326"), landuse_slim, how="left", predicate="within")

    grid_gdf = grid_gdf.drop_duplicates(subset=['geometry'])

    if return_step_size_m:
        return grid_gdf, step_size_m

    return grid_gdf


def calculate_distances(grid_points, cluster_points, n_jobs=-1):
    """
    Calculate the distances between grid points and cluster points using parallel processing.

    Parameters:
    grid_points (list of tuples): List of grid points as (x, y) tuples.
    cluster_points (GeoSeries): GeoSeries containing the cluster points.

    Returns:
    numpy.ndarray: A 2D array where each element [i, j] represents the distance between the i-th grid point and the j-th cluster point.
    """

    # Number of grid points
    num_grid = len(grid_points)
    # Number of cluster points
    num_clusters = len(cluster_points)

    # Function to compute distance for a single grid point
    def compute_distance(qx, qy):
        return np.sqrt((qx - cluster_points.x.values) ** 2 + (qy - cluster_points.y.values) ** 2)

    # Use joblib to parallelize the distance computation across all grid points
    distances = Parallel(n_jobs=n_jobs)(delayed(compute_distance)(qx, qy) for qx, qy in grid_points)

    # Convert list of arrays back into a 2D numpy array
    return np.array(distances)



def compute_civd_influence_for_grid(q_idx, unique_labels, cluster_labels, point_weights, distances):
    """
    Compute CIVD influence for a single grid point.

    Parameters:
    q_idx (int): Index of the grid point.

    Returns:
    tuple: A tuple containing the grid point index and a dictionary of influence per cluster label.
    """
    result = {}
    for label in unique_labels:
        label_indices = np.where(cluster_labels == label)[0]
        min_influence = np.max(point_weights[label_indices] / distances[q_idx, label_indices])
        result[label] = min_influence
    return q_idx, result

def compute_ivd_influence_for_grid(q_idx, unique_labels, cluster_labels, point_weights, distances):
    """
    Compute IVD influence for a single grid point.

    Parameters:
    q_idx (int): Index of the grid point.

    Returns:
    tuple: A tuple containing the grid point index and a dictionary of influence per cluster label.
    """
    result = {}
    for label in unique_labels:
        label_indices = np.where(cluster_labels == label)[0]
        total_influence = np.sum(point_weights[label_indices] / distances[q_idx, label_indices])
        result[label] = total_influence
    return q_idx, result


def compute_centroid_influence_for_grid(q_idx, unique_labels, cluster_labels, point_weights, grid_points, centroids):
    """
    Compute influence for a single grid point based on cluster centroids.
    """
    result = {}
    # Convert grid point and centroids to numpy arrays for faster computation
    grid_point = np.array(grid_points[q_idx]).reshape(1, -1)
    centroid_positions = np.array([(centroid.x, centroid.y) for centroid in centroids])

    # Compute all distances at once using cdist
    distances = cdist(grid_point, centroid_positions).flatten()

    for label, distance in zip(unique_labels, distances):
        label_indices = np.where(cluster_labels == label)[0]
        # Influence is weighted by the cluster weight and inversely proportional to the centroid distance
        result[label] = np.mean(point_weights[label_indices]) / distance

    return q_idx, result

def calculate_influence(grid_points, clusters, point_weights, method="civd", n_jobs=-1):
    """
    Calculate the influence of clusters on grid points using specified method.

    Parameters:
    grid_points (list of tuples): List of grid points as (x, y) tuples.
    clusters (GeoDataFrame): GeoDataFrame containing cluster geometries and labels.
    point_weights (numpy.ndarray): Array of weights for each cluster point.
    method (str, optional): Method to calculate influence, either "civd", "ivd", or "centroid". Defaults to "civd".
    n_jobs (int, optional): Number of jobs to run in parallel. Defaults to -1 (use all processors).

    Returns:
    dict: A dictionary where keys are grid point indices and values are dictionaries of influence per cluster label.
    """
    # Extract cluster points and labels
    clusters = clusters.to_crs("EPSG:3857")
    cluster_points = clusters.geometry
    cluster_labels = clusters['cluster_label'].values
    unique_labels = np.unique(cluster_labels)

    # Calculate distances between grid points and cluster points
    distances = calculate_distances(grid_points, cluster_points, n_jobs=n_jobs)


    # Select the appropriate computation method
    if method == "civd":
        results = Parallel(n_jobs=n_jobs)(
            delayed(compute_civd_influence_for_grid)(q_idx, unique_labels, cluster_labels, point_weights, distances)
            for q_idx in range(len(grid_points))
        )
    elif method == "ivd":
        results = Parallel(n_jobs=n_jobs)(
            delayed(compute_ivd_influence_for_grid)(q_idx, unique_labels, cluster_labels, point_weights, distances)
            for q_idx in range(len(grid_points))
        )
    elif method == "centroid":
        centroids = clusters.groupby('cluster_label')['geometry'].apply(lambda x: unary_union(x).centroid)
        # results = Parallel(n_jobs=n_jobs)(
        # delayed(compute_centroid_influence_for_grid)(q_idx, unique_labels, cluster_labels, point_weights, grid_points, centroids)
        # for q_idx in range(len(grid_points)))
        results = []
        for q_idx in range(len(grid_points)):
            result = compute_centroid_influence_for_grid(q_idx, unique_labels, cluster_labels, point_weights,
                                                         grid_points, centroids)
            results.append(result)


    # Populate influence matrix with results
    influence_matrix = {q_idx: result for q_idx, result in results}

    return influence_matrix


def scale_grid_weights(grid_weights, n_points):
    """
    Scale grid_weights so that their sum equals a given value n_points.

    Parameters:
    - grid_weights: A list of weights.
    - n_points: The desired sum of the scaled weights.

    Returns:
    - scaled_weights: A list of weights scaled to sum up to n_points.
    """

    # Step 1: Calculate the current sum of grid_weights
    total_weight = sum(grid_weights)

    if total_weight == 0:
        raise ValueError("Total weight is zero, cannot scale the weights.")

    # Step 2: Calculate the scaling factor
    scaling_factor = n_points / total_weight

    # Step 3: Scale each weight by the scaling factor
    scaled_weights = [weight * scaling_factor for weight in grid_weights]

    return scaled_weights