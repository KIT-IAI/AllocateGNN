"""
Author: Xuanhao Mu
Email: xuanhao.mu@kit.edu
Description: This module contains functions for clustering using KMeans clustering.
"""

from sklearn.cluster import KMeans
import geopandas as gpd
import numpy as np
from shapely.ops import unary_union
from sklearn.metrics import silhouette_score

def do_KMeans_clustering(coords, n_clusters):
    """
    Perform KMeans clustering on a set of coordinates.

    Parameters:
    coords (numpy.ndarray): An array of shape (n_samples, 2) containing the coordinates (longitude, latitude).
    n_clusters (int): The number of clusters to form.

    Returns:
    tuple: A tuple containing:
        - gdf (geopandas.GeoDataFrame): The GeoDataFrame with the clustering results.
        - centroid_gdf (geopandas.GeoDataFrame): The GeoDataFrame with the centroids of each cluster and the point count.
    """
    # Handle edge cases where coords might be 1D or empty
    if len(coords.shape) == 1:
        if coords.size == 0:
            raise ValueError("No points provided for clustering.")
        coords = np.expand_dims(coords, axis=0)

    # Check if n_clusters is greater than the number of points
    if len(coords) < n_clusters:
        raise ValueError(f"n_samples={len(coords)} should be >= n_clusters={n_clusters}.")

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(coords[:, 0], coords[:, 1]))
    gdf.crs = "EPSG:4326"

    # Convert to metric projection for distance calculation
    gdf_m = gdf.to_crs("EPSG:3857")
    coords_m = np.column_stack((gdf_m.geometry.x, gdf_m.geometry.y))

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(coords_m)

    # Add the clustering results to the GeoDataFrame
    gdf['cluster_label'] = kmeans.labels_

    # Calculate the centroid of each cluster
    centroids = gdf.groupby('cluster_label')['geometry'].apply(lambda x: unary_union(x).centroid)
    centroid_gdf = gpd.GeoDataFrame(centroids, geometry='geometry')
    centroid_gdf['point_count'] = gdf.groupby('cluster_label').size().values
    centroid_gdf.crs = "EPSG:4326"

    return gdf, centroid_gdf

def calculate_silhouette_scores_for_n_clusters_kmeans(coords, min_clusters=2, max_clusters=10):
    """
    Calculate Silhouette Scores for a range of cluster numbers.

    Parameters:
    gdf (geopandas.GeoDataFrame): The GeoDataFrame containing the points for clustering.
    min_clusters (int): Minimum number of clusters to consider.
    max_clusters (int): Maximum number of clusters to consider.
    linkage (str): Linkage method to use for Agglomerative Clustering.

    Returns:
    tuple: A tuple containing:
        - best_n_clusters (int): The number of clusters with the best Silhouette Score.
        - best_score (float): The highest Silhouette Score.
        - silhouette_scores (list): List of Silhouette Scores for each number of clusters.
        - cluster_range (range): The range of cluster numbers considered.
    """
    # Extract coordinates

    silhouette_scores = []
    cluster_range = range(min_clusters, max_clusters + 1)

    for n_clusters in cluster_range:
        # Perform hierarchical clustering
        clustering_model = KMeans(n_clusters=n_clusters, random_state=0)
        cluster_labels = clustering_model.fit_predict(coords)

        # Calculate Silhouette Score
        if len(set(cluster_labels)) > 1:  # Ensure at least two clusters
            score = silhouette_score(coords, cluster_labels)
        else:
            score = -1  # Invalid score if there's only one cluster

        silhouette_scores.append(score)

    # Find the best number of clusters based on the highest Silhouette Score
    best_n_clusters = cluster_range[np.argmax(silhouette_scores)]
    best_score = max(silhouette_scores)

    return best_n_clusters, best_score, silhouette_scores, cluster_range



