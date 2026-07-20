"""
Author: Xuanhao Mu
Email: xuanhao.mu@kit.edu
Description: This module contains functions for clustering
"""

from .DBSCAN_clustering import do_DBSCAN_clustering
from .hierarchical_clustering import do_hierarchical_clustering
from .kmeans_clustering import do_KMeans_clustering
from .HDBSCAN_clustering import do_hdbscan_clustering
from .mean_shift import do_mean_shift_clustering

def do_clustering(points, method = 'KMeans', n_clusters = 2, distance_threshold = 10, min_cluster_size = 3, **kwargs):
    """
    Perform clustering on a set of points.

    Parameters:
    points (numpy.ndarray): An array of shape (n_samples, 2) containing the coordinates (longitude, latitude).
    method (str): The clustering method to use ('KMeans', 'Hierarchical', or 'DBSCAN').
    n_clusters (int): The number of clusters to form (for KMeans and Hierarchical clustering).
    distance_threshold (float): The distance threshold for the DBSCAN algorithm.
    linkage (str): The linkage criterion to use for hierarchical clustering.

    Returns:
    tuple: A tuple containing:
        - gdf (geopandas.GeoDataFrame): The GeoDataFrame with the clustering results.
        - centroid_gdf (geopandas.GeoDataFrame): The GeoDataFrame with the centroids of each cluster and the point count.
    """
    method = method.lower()
    if method == 'kmeans':
        return do_KMeans_clustering(points, n_clusters)
    elif method == 'hierarchical':
        return do_hierarchical_clustering(points, n_clusters=n_clusters, distance_threshold=distance_threshold, **kwargs)
    elif method == 'dbscan':
        return do_DBSCAN_clustering(points, distance_threshold)
    elif method == 'hdbscan':
        return do_hdbscan_clustering(points, min_cluster_size=min_cluster_size, **kwargs)
    elif method == 'meanshift':
        return do_mean_shift_clustering(points, **kwargs)
    else:
        raise ValueError("Invalid clustering method. Choose from 'kmeans', 'dbscan', 'hierarchical'.")