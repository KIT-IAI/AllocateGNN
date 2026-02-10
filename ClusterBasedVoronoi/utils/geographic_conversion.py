"""
Author: Xuanhao Mu
Email: xuanhao.mu@kit.edu
Description: This module contains functions for geographic conversion.
"""

import numpy as np
from shapely.geometry import Polygon

# Calculate the geographical distance between two points using the haversine formula
def haversine(lon1, lat1, lon2, lat2):
    # Convert latitude and longitude from degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # Calculate differences in longitude and latitude
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    # Haversine formula
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    # Return the surface distance
    return 6371 * c  # 6371 is the radius of the Earth in kilometers

# Calculate maximum length and width
def calculate_max_length_width(coords):
    # Separate longitude and latitude
    lons = np.array([coord[0] for coord in coords])
    lats = np.array([coord[1] for coord in coords])

    # 1. Calculate maximum length (in the north-south direction)
    max_lat = np.max(lats)
    min_lat = np.min(lats)
    max_length = haversine(0, max_lat, 0, min_lat)  # Same longitude, calculate north-south distance

    # 2. Calculate maximum width (in the east-west direction), using the average latitude for estimation
    avg_lat = np.mean(lats)  # Use average latitude to calculate width
    max_lon = np.max(lons)
    min_lon = np.min(lons)
    max_width = haversine(max_lon, avg_lat, min_lon, avg_lat)  # Same latitude, calculate east-west distance

    return max_length, max_width

# Convert distance in kilometers to latitude/longitude difference
def distance_to_eps(distance_km):
    # 1 degree is approximately equal to 6371 kilometers
    return distance_km / 6371
