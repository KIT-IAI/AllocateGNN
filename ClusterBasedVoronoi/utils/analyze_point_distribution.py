import geopandas as gpd
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree


# Assume gdf is a GeoDataFrame containing point data
def analyze_point_distribution(gdf):
    # Ensure point geometry type
    if not all(geom.geom_type == 'Point' for geom in gdf.geometry):
        print("Non-point geometry type detected; please only include point geometries")
        return

    # Extract coordinates
    coords = np.array([(geom.x, geom.y) for geom in gdf.geometry])

    # Use KDTree to compute nearest neighbors
    tree = cKDTree(coords)
    distances, _ = tree.query(coords, k=2)  # k=2 because each point's distance to itself is 0
    mean_distance = np.mean(distances[:, 1])  # Second column is the distance to the nearest neighbor

    # Calculate study area
    area = gdf.total_bounds[2] - gdf.total_bounds[0] * (gdf.total_bounds[3] - gdf.total_bounds[1])

    # Calculate point density
    density = len(gdf) / area

    # Calculate the expected mean nearest neighbor distance (for random distribution)
    expected_mean_distance = 0.5 / np.sqrt(density)

    # Calculate the nearest neighbor index
    nn_index = mean_distance / expected_mean_distance

    print(f"Observed mean nearest neighbor distance: {mean_distance}")
    print(f"Expected mean nearest neighbor distance (random distribution): {expected_mean_distance}")
    print(f"Nearest neighbor index: {nn_index}")

    # Interpret results
    if nn_index < 1:
        print("Point distribution shows a clustered pattern")
    elif nn_index > 1:
        print("Point distribution shows a uniform pattern")
    else:
        print("Point distribution is close to a random pattern")

    return nn_index