"""
Author: Xuanhao Mu
Email: xuanhao.mu@kit.edu
Description: This module contains functions for clustering using DBSCAN clustering.
"""

import warnings
import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from shapely.ops import unary_union
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt

def do_DBSCAN_clustering(coords, distance_threshold):
    """
    Perform DBSCAN clustering on a set of coordinates.

    Parameters:
    coords (numpy.ndarray): An array of shape (n_samples, 2) containing the coordinates (longitude, latitude).
    distance_threshold (float): The distance threshold for the DBSCAN algorithm, in meters.

    Returns:
    tuple: A tuple containing:
        - gdf (geopandas.GeoDataFrame): The GeoDataFrame with the clustering results.
        - centroid_gdf (geopandas.GeoDataFrame): The GeoDataFrame with the centroids of each cluster and the point count.
    """
    # Handle edge cases where coords might be 1D or empty
    if len(coords.shape) == 1:
        if coords.size == 0:
            warnings.warn("No coordinates provided for clustering.")
            return gpd.GeoDataFrame(), gpd.GeoDataFrame()
        coords = np.expand_dims(coords, axis=0)

    gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(coords[:, 0], coords[:, 1]))
    gdf.crs = 'EPSG:4326'

    # Convert to metric projection for distance calculation
    gdf_m = gdf.to_crs("EPSG:3857")
    coords_m = np.column_stack((gdf_m.geometry.x, gdf_m.geometry.y))

    # Latitude and longitude are typically expressed in degrees (ranging from -180° to 180° longitude and -90° to 90° latitude).
    # When calculating the haversine distance between two points, the formula requires using radians, not degrees.
    db = DBSCAN(eps=distance_threshold, min_samples=2).fit(coords_m)

    # Add the clustering results to the GeoDataFrame
    gdf['cluster_label'] = db.labels_
    gdf['cluster_label_with_unclassified'] = db.labels_

    # Handle unclassified points (-1 indicates unclassified)
    unclassified_points = gdf[gdf['cluster_label'] == -1].copy()
    classified_points = gdf[gdf['cluster_label'] != -1].copy()

    # Assign new unique cluster IDs to unclassified points, starting from the max cluster ID + 1
    if not classified_points.empty:
        max_cluster_id = classified_points['cluster_label'].max() + 1
    else:
        max_cluster_id = 0

    if len(unclassified_points) > 0:
        unclassified_points['cluster_label'] = np.arange(max_cluster_id, max_cluster_id + len(unclassified_points))

    # Merge the classified and unclassified points
    gdf = pd.concat([classified_points, unclassified_points])

    # Calculate the centroid of each cluster
    centroids = gdf.groupby('cluster_label')['geometry'].apply(lambda x: unary_union(x).centroid if len(x) > 1 else x.iloc[0])
    centroid_gdf = gpd.GeoDataFrame(centroids, geometry='geometry')
    centroid_gdf['point_count'] = gdf.groupby('cluster_label').size().values
    centroid_gdf.crs = "EPSG:4326"

    return gdf, centroid_gdf


def calculate_silhouette_for_eps_range(coords, eps_values, min_samples=4):
    """
    Calculate Silhouette Scores for a range of eps values for DBSCAN and select the optimal eps.

    Parameters:
    gdf (geopandas.GeoDataFrame): The GeoDataFrame containing coordinates.
    eps_values (list): A range of different eps values to test.
    min_samples (int): The min_samples parameter for DBSCAN.

    Returns:
    tuple: Returns the best eps value, corresponding Silhouette Score, and the list of Silhouette Scores.
    """
    # Extract coordinates

    silhouette_scores = []

    for eps in eps_values:
        # Perform DBSCAN clustering
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
        labels = db.labels_

        # Skip eps values that result in only one cluster or all noise
        if len(set(labels)) <= 1:
            silhouette_scores.append(-1)
            continue

        # Calculate Silhouette Score
        score = silhouette_score(coords, labels)
        silhouette_scores.append(score)

    # Find the eps with the highest Silhouette Score
    best_eps = eps_values[np.argmax(silhouette_scores)]
    best_score = max(silhouette_scores)

    return best_eps, best_score, silhouette_scores


def calculate_internal_metrics_for_eps_range(coords, eps_values, methods=['silhouette'], min_samples=4):
    """
    Calculate internal metrics (Silhouette Score, Calinski-Harabasz Index, Davies-Bouldin Index)
    for a range of eps values for DBSCAN and return the best eps and scores for specified methods.

    Parameters:
    coords (array-like): The coordinate data for clustering.
    eps_values (list): A range of different eps values to test.
    methods (list): A list of methods to use for selecting the best eps. Can include 'silhouette',
                    'calinski_harabasz', 'davies_bouldin'.
    min_samples (int): The min_samples parameter for DBSCAN.

    Returns:
    dict: A dictionary where each method (from the methods list) is a key, and the value is another dictionary
          containing 'best_eps', 'best_score', and 'all_scores' for that method.
    """

    # Mapping method names to corresponding functions and default worst scores
    scoring_methods = {
        'silhouette':{'func':silhouette_score, 'worst':-1, 'best':np.argmax},
        'calinski_harabasz':{'func':calinski_harabasz_score, 'worst':-1, 'best':np.argmax},
        'davies_bouldin':{'func':davies_bouldin_score, 'worst':np.inf, 'best':np.argmin}
    }

    # Initialize score lists for each method
    scores = {method:[] for method in methods}

    # Calculate the scores for each eps value
    for eps in eps_values:
        # Perform DBSCAN clustering
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
        labels = db.labels_

        # Skip eps values that result in only one cluster or all noise
        if len(set(labels)) <= 1 or len(set(labels)) == len(labels):
            for method in methods:
                scores[method].append(scoring_methods[method]['worst'])
            continue

        # Calculate scores for each method
        for method in methods:
            score = scoring_methods[method]['func'](coords, labels)
            scores[method].append(score)

    # Create a dictionary to store the best eps and scores for each method
    results = {}
    for method in methods:
        best_index = scoring_methods[method]['best'](scores[method])
        results[method] = {
            'best_eps':eps_values[best_index],
            'best_score':scores[method][best_index],
            'all_scores':scores[method]
        }

    return results

def plot_silhouette_vs_eps(eps_values, silhouette_scores):
    """
    Plot the relationship between eps values and Silhouette Scores.

    Parameters:
    eps_values (list): A range of different eps values.
    silhouette_scores (list): The corresponding Silhouette Scores.
    """
    plt.figure(figsize=(3.5, 2.5))
    plt.plot(eps_values, silhouette_scores, marker='o', markersize=3)
    plt.title('Silhouette Score vs eps', fontsize=10)
    plt.xlabel('eps values', fontsize=10)
    plt.ylabel('Silhouette Score', fontsize=10)
    plt.grid(True)
    plt.show()
