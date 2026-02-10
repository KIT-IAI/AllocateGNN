import numpy as np
import geopandas as gpd
import warnings
from sklearn.cluster import MeanShift, estimate_bandwidth
from shapely.ops import unary_union
import pandas as pd
from scipy.spatial.distance import cdist


def do_mean_shift_clustering(coords, bandwidth=None, bin_seeding=True, cluster_all=True,
                             min_bin_freq=1, seeds=None, quantile=0.2, n_samples=100,
                             auto_bandwidth=True):
    """
    Perform Mean Shift clustering on a set of coordinates.

    Parameters:
    coords (numpy.ndarray): An array of shape (n_samples, 2) containing the coordinates (longitude, latitude).
    bandwidth (float): The bandwidth parameter, which controls the size of the kernel (default is None, which triggers auto-bandwidth).
    bin_seeding (bool): Whether to use the bin seeding procedure for initialization (default is True).
    cluster_all (bool): Whether to cluster all points, including noise (default is True).
    min_bin_freq (int): The minimum number of samples in a bin (default is 1).
    seeds (numpy.ndarray): The initial kernel centers (default is None).
    quantile (float): The quantile for bandwidth estimation (default is 0.2).
    n_samples (int): The number of samples to use for bandwidth estimation (default is 100).
    auto_bandwidth (bool): Whether to automatically estimate the bandwidth (default is True).

    Returns:
    tuple: A tuple containing:
        - gdf (geopandas.GeoDataFrame): The GeoDataFrame with the clustering results.
        - centroid_gdf (geopandas.GeoDataFrame): The GeoDataFrame with the centroids of each cluster and the point count.
    """

    # Handle cases where coords might be 1D or empty
    if len(coords.shape) == 1:
        if coords.size == 0:
            warnings.warn("No coordinates provided for clustering.")
            return gpd.GeoDataFrame(), gpd.GeoDataFrame()
        coords = np.expand_dims(coords, axis=0)

    # If there are not enough points, return early
    if len(coords) < 2:
        raise ValueError("Mean Shift clustering requires at least 2 samples.")

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(coords[:, 0], coords[:, 1]))
    gdf.crs = "EPSG:4326"

    # Convert to metric projection for distance calculation
    gdf_m = gdf.to_crs("EPSG:3857")
    coords_m = np.column_stack((gdf_m.geometry.x, gdf_m.geometry.y))

    # Estimate bandwidth if needed
    if auto_bandwidth and bandwidth is None:
        bandwidth = estimate_bandwidth(coords_m, quantile=quantile, n_samples=min(n_samples, len(coords_m)))
        if bandwidth <= 0:
            warnings.warn("Estimated bandwidth is non-positive. Using default heuristic.")
            # Fallback to a heuristic based on average distance to nearest neighbors
            distances = cdist(coords_m, coords_m)
            np.fill_diagonal(distances, np.inf)  # Exclude self-distances
            avg_min_dist = np.mean(np.min(distances, axis=1))
            bandwidth = avg_min_dist * 2  # Heuristic: twice the average nearest neighbor distance

    # Perform Mean Shift clustering
    try:
        ms = MeanShift(
            bandwidth=bandwidth,
            bin_seeding=bin_seeding,
            min_bin_freq=min_bin_freq,
            cluster_all=cluster_all,
            seeds=seeds
        )

        ms.fit(coords_m)

        # Add the clustering results to the GeoDataFrame
        gdf['cluster_label'] = ms.labels_

        # Add cluster centers in the original projection
        cluster_centers_m = ms.cluster_centers_
        cluster_centers_df = pd.DataFrame(cluster_centers_m, columns=['x', 'y'])
        centers_gdf = gpd.GeoDataFrame(cluster_centers_df,
                                       geometry=gpd.points_from_xy(cluster_centers_df.x, cluster_centers_df.y))
        centers_gdf.crs = "EPSG:3857"
        centers_gdf = centers_gdf.to_crs("EPSG:4326")

        # Calculate centroids and point counts
        centroids = []
        point_counts = []

        for i, center in enumerate(centers_gdf.geometry):
            cluster_points = gdf[gdf['cluster_label'] == i]
            point_counts.append(len(cluster_points))

            if len(cluster_points) > 1:
                # Calculate geometric centroid from the actual points
                centroid = unary_union(cluster_points.geometry).centroid
            else:
                # Use the cluster center directly
                centroid = center

            centroids.append(centroid)

        # Create the centroid GeoDataFrame
        centroid_gdf = gpd.GeoDataFrame({'geometry': centroids, 'point_count': point_counts})
        centroid_gdf['cluster_id'] = range(len(centroid_gdf))

        # Set the CRS
        centroid_gdf.crs = "EPSG:4326"

        # Add total number of clusters as attribute
        centroid_gdf.attrs['n_clusters'] = len(centroid_gdf)
        centroid_gdf.attrs['bandwidth'] = bandwidth

        return gdf, centroid_gdf

    except Exception as e:
        warnings.warn(f"Mean Shift clustering failed: {str(e)}")
        empty_centroid_gdf = gpd.GeoDataFrame(columns=['geometry', 'point_count', 'cluster_id'])
        empty_centroid_gdf.crs = "EPSG:4326"
        return gdf, empty_centroid_gdf


def do_mean_shift_custom(coords, bandwidth=None, max_iterations=100, epsilon=1e-3,
                         min_cluster_size=2, quantile=0.2, n_samples=100, auto_bandwidth=True):
    """
    Perform custom Mean Shift clustering on a set of coordinates.

    Parameters:
    coords (numpy.ndarray): An array of shape (n_samples, 2) containing the coordinates (longitude, latitude).
    bandwidth (float): The bandwidth parameter, which controls the size of the kernel (default is None, which triggers auto-bandwidth).
    max_iterations (int): Maximum number of iterations for the algorithm (default is 100).
    epsilon (float): Convergence threshold for point shifts (default is 1e-3).
    min_cluster_size (int): Minimum number of points to form a cluster (default is 2).
    quantile (float): The quantile for bandwidth estimation (default is 0.2).
    n_samples (int): The number of samples to use for bandwidth estimation (default is 100).
    auto_bandwidth (bool): Whether to automatically estimate the bandwidth (default is True).

    Returns:
    tuple: A tuple containing:
        - gdf (geopandas.GeoDataFrame): The GeoDataFrame with the clustering results.
        - centroid_gdf (geopandas.GeoDataFrame): The GeoDataFrame with the centroids of each cluster and the point count.
    """

    # Define Gaussian kernel function
    def gaussian_kernel(distance, bandwidth):
        return np.exp(-0.5 * (distance / bandwidth) ** 2)

    # Handle cases where coords might be 1D or empty
    if len(coords.shape) == 1:
        if coords.size == 0:
            warnings.warn("No coordinates provided for clustering.")
            return gpd.GeoDataFrame(), gpd.GeoDataFrame()
        coords = np.expand_dims(coords, axis=0)

    # If there are not enough points, return early
    if len(coords) < min_cluster_size:
        raise ValueError(f"Mean Shift clustering requires at least {min_cluster_size} samples.")

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(coords[:, 0], coords[:, 1]))
    gdf.crs = "EPSG:4326"

    # Convert to metric projection for distance calculation
    gdf_m = gdf.to_crs("EPSG:3857")
    coords_m = np.column_stack((gdf_m.geometry.x, gdf_m.geometry.y))

    # Estimate bandwidth if needed
    if auto_bandwidth and bandwidth is None:
        # Use scikit-learn's bandwidth estimator if available
        try:
            bandwidth = estimate_bandwidth(coords_m, quantile=quantile, n_samples=min(n_samples, len(coords_m)))
        except:
            # Fallback to a simple heuristic based on average distance to nearest neighbors
            distances = cdist(coords_m, coords_m)
            np.fill_diagonal(distances, np.inf)  # Exclude self-distances
            avg_min_dist = np.mean(np.min(distances, axis=1))
            bandwidth = avg_min_dist * 2  # Heuristic: twice the average nearest neighbor distance

        if bandwidth <= 0:
            warnings.warn("Estimated bandwidth is non-positive. Using default heuristic.")
            distances = cdist(coords_m, coords_m)
            np.fill_diagonal(distances, np.inf)
            avg_min_dist = np.mean(np.min(distances, axis=1))
            bandwidth = avg_min_dist * 2

    # Initialize points for shifting
    points = coords_m.copy()
    original_points = coords_m.copy()
    shifted_points = np.zeros_like(points)
    converged = np.zeros(len(points), dtype=bool)

    # Perform Mean Shift iterations
    for iteration in range(max_iterations):
        # Calculate shifts for each point
        for i, point in enumerate(points):
            if converged[i]:
                continue

            # Calculate distances to all points
            distances = cdist([point], points)[0]

            # Apply Gaussian kernel to get weights
            weights = gaussian_kernel(distances, bandwidth)

            # Calculate weighted mean shift
            shifted_point = np.sum(weights[:, np.newaxis] * points, axis=0) / np.sum(weights)
            shifted_points[i] = shifted_point

            # Check convergence
            if np.linalg.norm(shifted_point - point) < epsilon:
                converged[i] = True

        # Update points
        points = shifted_points.copy()

        # Check if all points have converged
        if np.all(converged):
            break

    # Find unique cluster centers by merging close centers
    unique_centers = []
    cluster_ids = []

    for point in points:
        if len(unique_centers) == 0:
            unique_centers.append(point)
            cluster_ids.append(0)
        else:
            distances = cdist([point], unique_centers)[0]
            min_distance = np.min(distances)
            min_idx = np.argmin(distances)

            if min_distance < bandwidth * 0.5:  # Use half bandwidth as merging threshold
                cluster_ids.append(min_idx)
            else:
                unique_centers.append(point)
                cluster_ids.append(len(unique_centers) - 1)

    # Assign labels to original points
    labels = []
    for point in original_points:
        distances = cdist([point], points)[0]
        min_idx = np.argmin(distances)
        labels.append(cluster_ids[min_idx])

    # Convert unique centers to the original CRS
    unique_centers = np.array(unique_centers)
    centers_df = pd.DataFrame(unique_centers, columns=['x', 'y'])
    centers_gdf = gpd.GeoDataFrame(centers_df, geometry=gpd.points_from_xy(centers_df.x, centers_df.y))
    centers_gdf.crs = "EPSG:3857"
    centers_gdf = centers_gdf.to_crs("EPSG:4326")

    # Add cluster labels to the GeoDataFrame
    gdf['cluster_label'] = labels

    # Calculate centroids and point counts
    cluster_data = []

    for i in range(len(unique_centers)):
        cluster_points = gdf[gdf['cluster_label'] == i]

        if len(cluster_points) >= min_cluster_size:
            point_count = len(cluster_points)

            if point_count > 1:
                # Calculate geometric centroid from the actual points
                centroid = unary_union(cluster_points.geometry).centroid
            else:
                # Use the center directly
                centroid = centers_gdf.geometry.iloc[i]

            cluster_data.append({
                'geometry': centroid,
                'point_count': point_count,
                'cluster_id': i
            })

    # Create the centroid GeoDataFrame
    if cluster_data:
        centroid_gdf = gpd.GeoDataFrame(cluster_data)
        centroid_gdf.crs = "EPSG:4326"
        centroid_gdf.attrs['n_clusters'] = len(centroid_gdf)
        centroid_gdf.attrs['bandwidth'] = bandwidth
    else:
        centroid_gdf = gpd.GeoDataFrame(columns=['geometry', 'point_count', 'cluster_id'])
        centroid_gdf.crs = "EPSG:4326"

    return gdf, centroid_gdf