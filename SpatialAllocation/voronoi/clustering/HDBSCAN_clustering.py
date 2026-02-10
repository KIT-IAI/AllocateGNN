import numpy as np
import geopandas as gpd
import warnings
import hdbscan
from shapely.ops import unary_union
import pandas as pd


def do_hdbscan_clustering(coords, min_cluster_size=5, min_samples=None, metric='euclidean',
                          cluster_selection_method='eom'):
    """
    Perform HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) on a set of coordinates.

    Parameters:
    coords (numpy.ndarray): An array of shape (n_samples, 2) containing the coordinates (longitude, latitude).
    min_cluster_size (int): The minimum size of clusters (default is 5).
    min_samples (int): The number of samples in a neighborhood for a point to be considered a core point (default is None, which sets it equal to min_cluster_size).
    metric (str): The metric to use for distance computation (default is 'euclidean').
    cluster_selection_method (str): The method to use for cluster selection, 'eom' (excess of mass) or 'leaf' (default is 'eom').

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
    if len(coords) < min_cluster_size:
        raise ValueError(
            f"HDBSCAN clustering requires at least {min_cluster_size} samples for min_cluster_size={min_cluster_size}.")

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(coords[:, 0], coords[:, 1]))
    gdf.crs = "EPSG:4326"

    # Convert to metric projection for distance calculation
    gdf_m = gdf.to_crs("EPSG:3857")
    coords_m = np.column_stack((gdf_m.geometry.x, gdf_m.geometry.y))

    # Perform HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method=cluster_selection_method,
        core_dist_n_jobs=1  # For reproducibility
    )

    clusterer.fit(coords_m)

    # Add the clustering results to the GeoDataFrame
    gdf['cluster_label'] = clusterer.labels_
    gdf['cluster_label_with_unclassified'] = clusterer.labels_

    # Add cluster membership probability
    gdf['cluster_probability'] = clusterer.probabilities_

    # Handle unclassified points (-1 indicates unclassified/noise)
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

    # Filter valid clusters (now all points are in clusters, but we use the original labels for centroid calculation)
    valid_clusters = gdf[gdf['cluster_label_with_unclassified'] != -1]

    # Add noise percentage information
    noise_percentage = (gdf['cluster_label'] == -1).mean() * 100

    # If all points are noise, return the original GeoDataFrame and an empty centroid GeoDataFrame
    if len(valid_clusters) == 0:
        warnings.warn(f"All points were classified as noise. Consider adjusting parameters.")
        empty_centroid_gdf = gpd.GeoDataFrame(columns=['geometry', 'point_count', 'noise_percentage'])
        empty_centroid_gdf.crs = "EPSG:4326"
        return gdf, empty_centroid_gdf

    # Calculate the centroid of each cluster using the original labels
    # (this only includes points that were originally clustered, not the noise points)
    centroids = valid_clusters.groupby('cluster_label_with_unclassified')['geometry'].apply(
        lambda x: unary_union(x).centroid if len(x) > 1 else x.iloc[0]
    )

    centroid_gdf = gpd.GeoDataFrame(centroids, geometry='geometry')

    # Add point count for each cluster
    centroid_gdf['point_count'] = valid_clusters.groupby('cluster_label_with_unclassified').size().values

    # Add average probability for each cluster
    centroid_gdf['avg_probability'] = valid_clusters.groupby('cluster_label')['cluster_probability'].mean().values

    # Add noise percentage information to centroid GeoDataFrame
    centroid_gdf['noise_percentage'] = noise_percentage

    # Set the CRS
    centroid_gdf.crs = "EPSG:4326"

    # Add total number of clusters as attribute
    centroid_gdf.attrs['n_clusters'] = len(centroid_gdf)

    return gdf, centroid_gdf