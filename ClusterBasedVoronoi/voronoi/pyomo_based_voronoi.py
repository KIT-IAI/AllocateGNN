"""
Author: Xuanhao Mu
Email: xuanhao.mu@kit.edu
Description: This module contains utility functions for creating weights.
"""
import numpy as np
import pyomo.environ as pyo
from .prepare_pyomo_parameter import generate_grid, calculate_influence, scale_grid_weights
from ClusterBasedVoronoi.utils import create_weights
from shapely.ops import unary_union
import geopandas as gpd
from shapely.geometry import Polygon
import time
import pandas as pd

def pyomo_solution_to_gdf(model, grid_points, regions, step_size_m, clusters):
    """
    Convert Pyomo model solution to a GeoDataFrame, with polygons representing grid cells assigned to clusters.

    Parameters:
    - model: The solved Pyomo model, which contains decision variables 'x[q, c]' representing grid-to-cluster assignments.
    - grid_points: A list of (x, y) tuples representing grid points.
    - boundary_polygon: The boundary of the area, in meters.
    - step_size_m: The resolution of the grid in meters.
    - cluster_points: A dictionary with cluster labels as keys and list of (x, y) points as values.
    - weights_dict: A dictionary mapping cluster labels to their weights.
    - crs: The coordinate reference system for the output GeoDataFrame (default is EPSG:3857).

    Returns:
    - gdf: A GeoDataFrame containing polygons for each cluster's region.
    """

    # Step 1: Initialize lists for polygons and indices (cluster labels)
    start = time.time()
    points_df = pd.DataFrame(grid_points, columns=['x', 'y'])
    points_df['q_idx'] = np.arange(1, len(grid_points) + 1)  # Pyomo索引从1开始

    # 创建一个字典以快速查询每个点的分配情况
    point_cluster_dict = {}

    # 这个循环无法避免，但我们可以减少内循环的工作量
    # for var in model.x:
    #     q_idx, c = var
    #     if pyo.value(model.x[q_idx, c]) > 0.5:
    #         point_cluster_dict[q_idx] = c
    # points_df['cluster_label'] = points_df['q_idx'].map(point_cluster_dict)

    assignment_values = {}
    for var in model.x:
        q_idx, c = var
        value = pyo.value(model.x[q_idx, c])
        if q_idx not in assignment_values:
            assignment_values[q_idx] = {}
        assignment_values[q_idx][c] = value

    # 为每个网格点选择最佳cluster
    point_cluster_dict = {}
    for q_idx, cluster_values in assignment_values.items():
        best_cluster = max(cluster_values, key=cluster_values.get)
        point_cluster_dict[q_idx] = best_cluster

    points_df['cluster_label'] = points_df['q_idx'].map(point_cluster_dict)

    # Step 2.2: Convert boundary_polygon to meters (EPSG:3857)
    regions = regions.to_crs("EPSG:3857")
    boundary_polygon_m = unary_union(regions["geometry"])
    end = time.time()
    print(f"Step preparation time: {end - start}")

    start = time.time()
    def create_polygon(row):
        x0, y0 = row['x'], row['y']
        x1, y1 = x0 + step_size_m, y0 + step_size_m
        return Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])

    points_df['geometry'] = points_df.apply(create_polygon, axis=1)
    gdf = gpd.GeoDataFrame(points_df, geometry='geometry', crs="EPSG:3857")
    end = time.time()
    print(f"Step create gdf time: {end - start}")

    start = time.time()
    # 创建结果DataFrame存储
    result_geometries = []
    result_labels = []

    # 按cluster_label分组并合并
    for label, group in gdf.groupby('cluster_label'):
        # 合并同一cluster的所有几何体
        union_geom = unary_union(group['geometry'].tolist())

        # 与边界相交
        if union_geom.intersects(boundary_polygon_m):
            clipped_geom = union_geom.intersection(boundary_polygon_m)
            result_geometries.append(clipped_geom)
            result_labels.append(label)
    end = time.time()
    print(f"Step clip time: {end - start}")

    # # Step 3: Iterate over the grid points and retrieve cluster assignments from the model
    # start = time.time()
    # total_points = len(grid_points)
    # print(f"Grid points number: {total_points}")
    # grid_points = np.array(grid_points)
    #
    # polygons = []
    # indices = []
    # # Step 3: Iterate over the grid points and retrieve cluster assignments from the model
    # print(f"Grid points number: {len(grid_points)}")
    # for q_idx, (x0, y0) in enumerate(grid_points):
    #     # For each grid point, check which cluster it is assigned to
    #     for c in cluster_labels:
    #         if pyo.value(model.x[q_idx + 1, c]) > 0.5:  # Pyomo indices start from 1
    #             # Create a small square polygon for the grid cell
    #             x1, y1 = x0 + step_size_m, y0 + step_size_m
    #             poly = Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
    #
    #             # Retain only the grid cells that intersect with the boundary polygon
    #             if poly.intersects(boundary_polygon_m):
    #                 polygons.append(poly.intersection(boundary_polygon_m))  # Clip to the boundary polygon
    #                 indices.append(c)
    #             break  # Only one cluster per grid point
    #     print(f"Grid points: {start - end}")
    #
    # print(f"Step 3 time: {end - start}")


    # Step 4: Construct a GeoDataFrame where 'cluster_label' is the centroid index and 'geometry' is the collection of polygons
    gdf = gpd.GeoDataFrame({'cluster_label':result_labels, 'geometry':result_geometries}, crs="EPSG:3857")
    gdf.to_crs(epsg=4326, inplace=True)

    # Step 5: Set the index to the cluster labels
    gdf = gdf.set_index('cluster_label')

    # Step 6: Merge polygons with the same cluster label (i.e., those belonging to the same cluster)
    gdf = gdf.dissolve(by='cluster_label')

    return gdf


def build_model(regions, clusters, step_size_m, weight = "equal", method="civd", solver="scip", penalty_weight=None, **kwargs):
    """
    Build a Pyomo optimization model based on regions and clusters.

    Parameters:
    regions (GeoDataFrame): GeoDataFrame containing the regions.
    clusters (GeoDataFrame): GeoDataFrame containing the cluster geometries and labels.
    step_size_m (float): The step size in meters for generating grid points.
    weights (str, optional): Method to calculate weights. Defaults to None.
    method (str, optional): Method to calculate influence, either "civd" or "ivd". Defaults to "civd".

    Returns:
    ConcreteModel: A Pyomo ConcreteModel object representing the optimization model.
    """
    print(f"Step 1: Combine all region geometries into a single boundary polygon")

    # Generate grid points within the boundary polygon
    grid_gdf = generate_grid(regions, step_size_m)
    print(f"grid_gdf: {grid_gdf.shape}")
    grid_points = grid_gdf["geometry"].apply(lambda p: (p.x, p.y)).tolist()

    # Create weights for each cluster point
    if type(weight) == str:
        point_weights = create_weights(clusters, method = weight, **kwargs)
    elif type(weight) == list or type(weight) == np.ndarray:
        point_weights = weight
        if len(point_weights) != clusters.shape[0]:
            raise ValueError("The length of the point weights must be equal to the number of clusters.")
    else:
        raise ValueError("unexpected type of weight")
    print(f"point_weights: {point_weights}, clusters: {clusters.shape[0]}")

    print(f"Step 2: Calculate the influence matrix based on the specified method")
    # Calculate the influence matrix based on the specified method
    if method == "civd":
        influence_matrix = calculate_influence(grid_points, clusters, point_weights, method='civd')
    elif method == "ivd":
        influence_matrix = calculate_influence(grid_points, clusters, point_weights, method='ivd')
    elif method == "centroid":
        influence_matrix = calculate_influence(grid_points, clusters, point_weights, method='centroid')

    print(f"Step 3: Build the Pyomo optimization model")
    # Initialize a Pyomo ConcreteModel
    model = pyo.ConcreteModel()

    # Define the set of grid points
    model.Q = pyo.RangeSet(len(grid_points))

    # Define the set of clusters
    model.C = pyo.Set(initialize=clusters['cluster_label'].unique())

    # Define binary decision variables for assigning grid points to clusters
    # model.x = pyo.Var(model.Q, model.C, within=pyo.Binary)
    model.x = pyo.Var(model.Q, model.C, within=pyo.UnitInterval)  # 0 <= x <= 1

    def one_cluster_per_point_rule(model, q):
        """
        Ensure each grid point is assigned to exactly one cluster.

        Parameters:
        model (ConcreteModel): The Pyomo model.
        q (int): The index of the grid point.

        Returns:
        Constraint: A Pyomo constraint ensuring one cluster per grid point.
        """
        return sum(model.x[q, c] for c in model.C) == 1

        # Add the constraint to the model

    model.one_cluster_per_point = pyo.Constraint(model.Q, rule=one_cluster_per_point_rule)

    def objective_function(model):
        """
        Define the objective function to maximize the total influence.

        Parameters:
        model (ConcreteModel): The Pyomo model.

        Returns:
        Objective: A Pyomo objective to maximize the total influence.
        """
        # if method == "civd" or method == "ivd":
        #     # Maximize total influence (CIVD approach)
        # 原目标函数部分
        objective_value = sum(model.x[q, c] * influence_matrix[q - 1][c] for q in model.Q for c in model.C)

        if penalty_weight is not None:
            # 添加惩罚项
            penalty_value = penalty_weight * sum((model.x[q, c] - 0.5) ** 2 for q in model.Q for c in model.C)
            return objective_value + penalty_value

        return objective_value

    # Add the objective function to the model
    model.obj = pyo.Objective(rule=objective_function, sense=pyo.maximize)

    print(f"Step 4: Solve the model using the specified solver")
    # Solve the model using the specified solver
    solver = pyo.SolverFactory(solver)
    solver.options['threads'] = 4
    solver.solve(model)

    print(f"Step 5: Convert the Pyomo solution to a GeoDataFrame")
    gdf = pyomo_solution_to_gdf(model, grid_points, regions, step_size_m, clusters)

    return model, gdf, grid_points
