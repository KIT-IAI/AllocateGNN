"""
Author: Xuanhao Mu
Email: xuanhao.mu@kit.edu
Description: This module contains functions for clustering using Hierarchical clustering.
"""

import warnings
from sklearn.cluster import AgglomerativeClustering
import geopandas as gpd
import numpy as np
from shapely.ops import unary_union
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def do_hierarchical_clustering(coords, n_clusters=15, distance_threshold=10, linkage='ward', use='n_clusters'):
    """
    Perform Hierarchical Agglomerative clustering on a set of coordinates.

    Parameters:
    coords (numpy.ndarray): An array of shape (n_samples, 2) containing the coordinates (longitude, latitude).
    n_clusters (int): The number of clusters to form.
    distance_threshold (float): The distance threshold for the clustering algorithm (in kilometers).
    linkage (str): The linkage criterion to use (default is 'ward').

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
        raise ValueError("Hierarchical clustering requires at least 2 samples.")

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(coords[:, 0], coords[:, 1]))
    gdf.crs = "EPSG:4326"

    # Convert to metric projection for distance calculation
    gdf_m = gdf.to_crs("EPSG:3857")
    coords_m = np.column_stack((gdf_m.geometry.x, gdf_m.geometry.y))

    # Perform hierarchical clustering
    if use == 'n_clusters':
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters, distance_threshold=None, linkage=linkage).fit(coords_m)
    elif use == 'distance_threshold':
        raise NotImplementedError("Distance threshold is not supported yet.")
    else:
        raise ValueError("Invalid value for 'use'. Choose from 'n_clusters' or 'distance_threshold'.")

    # Add the clustering results to the GeoDataFrame
    gdf['cluster_label'] = hierarchical.labels_

    # Calculate the centroid of each cluster
    centroids = gdf.groupby('cluster_label')['geometry'].apply(lambda x: unary_union(x).centroid if len(x) > 1 else x.iloc[0])
    centroid_gdf = gpd.GeoDataFrame(centroids, geometry='geometry')
    centroid_gdf['point_count'] = gdf.groupby('cluster_label').size().values
    centroid_gdf.crs = "EPSG:4326"

    return gdf, centroid_gdf


def calculate_silhouette_scores_for_n_clusters(coords, min_clusters=2, max_clusters=10, linkage='ward'):
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
        clustering_model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
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


from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import numpy as np


def calculate_scores_for_n_clusters(coords, min_clusters=2, max_clusters=10, linkage='ward', methods=['silhouette']):
    """
    Calculate internal clustering scores (Silhouette Score, Calinski-Harabasz Index, Davies-Bouldin Index)
    for a range of cluster numbers using Agglomerative Clustering.

    Parameters:
    coords (array-like): The coordinates for clustering.
    min_clusters (int): Minimum number of clusters to consider.
    max_clusters (int): Maximum number of clusters to consider.
    linkage (str): Linkage method to use for Agglomerative Clustering.
    methods (list): List of methods to calculate scores. Options are 'silhouette', 'calinski_harabasz', 'davies_bouldin'.

    Returns:
    dict: A dictionary where each method (from the methods list) is a key, and the value is another dictionary
          containing 'best_n_clusters', 'best_score', and 'all_scores' for that method.
    """

    cluster_range = range(min_clusters, max_clusters + 1)

    # Mapping method names to corresponding functions and default worst scores
    scoring_methods = {
        'silhouette':{'func':silhouette_score, 'worst':-1, 'best':np.argmax},
        'calinski_harabasz':{'func':calinski_harabasz_score, 'worst':-1, 'best':np.argmax},
        'davies_bouldin':{'func':davies_bouldin_score, 'worst':np.inf, 'best':np.argmin}
    }

    # Initialize score lists for each method
    scores = {method:[] for method in methods}

    # Calculate the scores for each number of clusters
    for n_clusters in cluster_range:
        clustering_model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        cluster_labels = clustering_model.fit_predict(coords)

        # Skip invalid clustering results with fewer than 2 clusters
        if len(set(cluster_labels)) <= 1:
            for method in methods:
                scores[method].append(scoring_methods[method]['worst'])
            continue

        # Calculate scores for each method
        for method in methods:
            score = scoring_methods[method]['func'](coords, cluster_labels)
            scores[method].append(score)

    # Create a dictionary to store the best n_clusters and scores for each method
    results = {}
    for method in methods:
        best_index = scoring_methods[method]['best'](scores[method])
        results[method] = {
            'best_n_clusters':cluster_range[best_index],
            'best_score':scores[method][best_index],
            'all_scores':scores[method]
        }

    return results


def plot_silhouette_vs_n_clusters(cluster_range, silhouette_scores):
    """
    Plot the Silhouette Scores against the number of clusters.

    Parameters:
    cluster_range (range): The range of cluster numbers considered.
    silhouette_scores (list): List of Silhouette Scores for each number of clusters.
    """
    plt.figure(figsize=(3.5, 2.5))
    plt.plot(cluster_range, silhouette_scores, marker='o')
    plt.xticks(cluster_range)
    plt.xlabel('Number of Clusters', fontsize=10)
    plt.ylabel('Silhouette Score', fontsize=10)
    plt.title('Silhouette Score vs Number of Clusters', fontsize=10)
    plt.grid(True)
    plt.show()
