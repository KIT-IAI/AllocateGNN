import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.ops import unary_union
import matplotlib.pyplot as plt


def calculate_adjacency_matrix(gdf):
    """Calculate the adjacency matrix of polygons in a GeoDataFrame."""
    n = len(gdf)
    adjacency_matrix = np.zeros((n, n), dtype=bool)

    # Create spatial index for efficiency
    sindex = gdf.sindex

    for i, geom in enumerate(gdf.geometry):
        # Get potentially adjacent polygons
        possible_matches_index = list(sindex.intersection(geom.bounds))
        possible_matches = gdf.iloc[possible_matches_index]

        # Filter for truly adjacent polygons
        for j, possible_match in enumerate(possible_matches_index):
            if i != possible_match:  # Exclude self
                if geom.touches(possible_matches.iloc[j].geometry):
                    adjacency_matrix[i, possible_match] = True

    return adjacency_matrix


def find_merge_pairs(adjacency_matrix):
    """
    Find the best pairwise merge pairs.

    Parameters:
    adjacency_matrix: Adjacency matrix.

    Returns:
    merge_pairs: List of region pairs to merge [(i1,j1), (i2,j2), ...]
    """
    n = len(adjacency_matrix)
    remaining = set(range(n))
    pairs = []

    # For each region, find an adjacent region to pair with
    while len(remaining) >= 2:
        found_pair = False

        # Take the first from the remaining regions
        if not remaining:
            break

        i = min(remaining)
        remaining.remove(i)

        # Find all adjacent regions of i
        neighbors = [j for j in remaining if adjacency_matrix[i, j]]

        if neighbors:
            # Pair with the first adjacent region
            j = neighbors[0]
            pairs.append((i, j))
            remaining.remove(j)
            found_pair = True

        # If no adjacent region found, try pairing with any remaining region (even if not adjacent)
        if not found_pair and remaining:
            j = min(remaining)
            pairs.append((i, j))
            remaining.remove(j)

    # If an odd number of regions remain, keep the last one as-is
    if remaining:
        pairs.append((list(remaining)[0],))

    return pairs


def merge_by_pairs(gdf, id_column='id', attribute_operations=None):
    """
    Pairwise merge of regions.

    Parameters:
    gdf: GeoDataFrame containing the regions.
    id_column: Region ID column name.
    attribute_operations: Dictionary where keys are column names and values are merge operations
                         ('mean', 'sum', 'min', 'max', 'first', 'last', 'count').

    Returns:
    A new GeoDataFrame after merging.
    """
    if len(gdf) <= 1:
        return gdf.copy()

    # Compute adjacency matrix
    adjacency_matrix = calculate_adjacency_matrix(gdf)

    # Find region pairs to merge
    merge_pairs = find_merge_pairs(adjacency_matrix)

    # Create new merged regions
    new_geometries = []
    new_ids = []

    # Prepare data for each attribute
    attribute_data = {}
    if attribute_operations:
        for col, operation in attribute_operations.items():
            if col in gdf.columns and col != 'geometry' and col != id_column:
                attribute_data[col] = []

    for i, pair in enumerate(merge_pairs):
        if len(pair) == 2:
            # Merge two regions
            i1, i2 = pair
            geom1 = gdf.iloc[i1].geometry
            geom2 = gdf.iloc[i2].geometry
            merged_geom = unary_union([geom1, geom2])

            # Process attributes
            if attribute_operations:
                for col, operation in attribute_operations.items():
                    if col in gdf.columns and col != 'geometry' and col != id_column:
                        values = [gdf.iloc[i1][col], gdf.iloc[i2][col]]

                        # Execute the specified operation
                        if operation == 'mean':
                            attribute_data[col].append(np.mean(values))
                        elif operation == 'sum':
                            attribute_data[col].append(sum(values))
                        elif operation == 'min':
                            attribute_data[col].append(min(values))
                        elif operation == 'max':
                            attribute_data[col].append(max(values))
                        elif operation == 'first':
                            attribute_data[col].append(values[0])
                        elif operation == 'last':
                            attribute_data[col].append(values[1])
                        elif operation == 'count':
                            attribute_data[col].append(len(values))
                        else:  # Default to mean
                            attribute_data[col].append(np.mean(values))
        else:
            # Single region stays unchanged
            i1 = pair[0]
            merged_geom = gdf.iloc[i1].geometry

            # Process attributes
            if attribute_operations:
                for col, operation in attribute_operations.items():
                    if col in gdf.columns and col != 'geometry' and col != id_column:
                        attribute_data[col].append(gdf.iloc[i1][col])

        new_geometries.append(merged_geom)
        new_ids.append(i + 1)  # New IDs starting from 1

    # Create data dictionary
    data_dict = {
        id_column: new_ids,
        'geometry': new_geometries
    }

    # Add attribute data
    for col, values in attribute_data.items():
        data_dict[col] = values

    # Create new GeoDataFrame
    new_gdf = gpd.GeoDataFrame(data_dict, crs=gdf.crs)

    return new_gdf


def create_hierarchical_regions(gdf, target_region_count, id_column='id', attribute_operations=None):
    """
    Create hierarchical region merging through pairwise merging.

    Parameters:
    gdf: Original GeoDataFrame.
    target_region_count: Final target number of regions (e.g., 1).
    id_column: Region ID column name.
    attribute_operations: Dictionary where keys are column names and values are merge operations
                         ('mean', 'sum', 'min', 'max', 'first', 'last', 'count').
        Example: {'population': 'sum', 'income': 'mean', 'area_code': 'first'}

    Returns:
    list: A list of GeoDataFrames containing the original GDF and merge results at each level,
          sorted by region count from most to fewest.
    """
    # Ensure input GDF has consecutive IDs starting from 1
    original_gdf = gdf.copy()
    original_gdf[id_column] = range(1, len(original_gdf) + 1)

    # Create result list, first add the original GDF
    results_list = [original_gdf]

    # Current working GDF
    current_gdf = original_gdf.copy()

    # Keep merging until reaching or going below the target count
    while len(current_gdf) > target_region_count:
        # Pairwise merge, passing attribute operation parameters
        current_gdf = merge_by_pairs(current_gdf, id_column, attribute_operations)

        # If no further reduction (e.g., only one region left), stop
        if len(current_gdf) == len(results_list[-1]):
            break

        # Add to result list
        results_list.append(current_gdf.copy())

        # If target count reached, stop merging
        if len(current_gdf) <= target_region_count:
            break

    return results_list