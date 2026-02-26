import time
import pandas as pd
from torch_geometric.data import Data
import torch
import numpy as np
import geopandas as gpd
from typing import Dict, Any, Optional, List
from torch_geometric.data import HeteroData


def _build_source_agent_graph(gdf_s: gpd.GeoDataFrame, gdf_a: gpd.GeoDataFrame,
                              relation_column: str) -> list:
    """
    Build a region-point graph based on pre-computed affiliation relationships.

    Args:
        gdf_s (gpd.GeoDataFrame): Region data (polygon centroids).
        gdf_a (gpd.GeoDataFrame): Point data.
        relation_column (str): Common index column name indicating the affiliation relationship.

    Returns:
        list: Edge list of the graph.
    """
    graph_edges = []

    # Verify that the index column exists
    if relation_column not in gdf_s.columns:
        raise ValueError(f"Index column '{relation_column}' not found in gdf_s")
    if relation_column not in gdf_a.columns:
        raise ValueError(f"Index column '{relation_column}' not found in gdf_a")

    # Create index mappings
    # gdf_s index mapping: region_id -> node_index_in_graph
    region_to_node = {region_id: i for i, region_id in enumerate(gdf_s[relation_column])}

    # gdf_a index mapping: region_id -> [point_indices_in_graph]
    point_groups = {}
    for i, region_id in enumerate(gdf_a[relation_column]):
        if region_id not in point_groups:
            point_groups[region_id] = []
        point_groups[region_id].append(len(gdf_s) + i)  # Point node indices start from len(gdf_s)

    # 1. Build region-point connections (based on affiliation)
    for region_id, region_node_idx in region_to_node.items():
        if region_id in point_groups:
            for point_node_idx in point_groups[region_id]:
                # Bidirectional connection: region <-> point
                graph_edges.append((region_node_idx, point_node_idx))
                graph_edges.append((point_node_idx, region_node_idx))

    return graph_edges


def preprocess_features(
        gdf: gpd.GeoDataFrame,
        numerical_col_names_all: Optional[List[str]] = None,
        categorical_col_members_all: Optional[Dict[str, List[str]]] = None
) -> Dict[str, Any]:
    """
    Automatically or based on a predefined schema, preprocess features in a DataFrame,
    converting categorical text to numerical format.

    Args:
        gdf (gpd.GeoDataFrame): Input GeoDataFrame containing mixed-type features.
        numerical_col_names_all (Optional[List[str]]):
            A list of all expected numerical feature column names. If provided, this list
            will be used instead of auto-detection.
        categorical_col_members_all (Optional[Dict[str, List[str]]]):
            A dictionary where keys are categorical feature names and values are lists of all
            possible category members. If provided, this dictionary will be used for one-hot
            encoding to ensure feature matrix consistency.

    Returns:
        Dict[str, Any]: A dictionary containing the following key-value pairs:
            - 'final_features': pd.DataFrame containing only pure numerical features.
            - 'mapping': Dict mapping feature names to column ranges {feature_name: (start_col, end_col, feature_type)}.
    """
    # 1. Exclude 'geometry' column
    features_gdf = gdf.drop(columns='geometry', errors='ignore')

    # Initialize mapping dictionary and feature list
    mapping = {}
    feature_components = []
    current_col_idx = 0

    # =============================================================================
    # Mode selection: auto-detection vs. predefined schema
    # =============================================================================

    # Mode 1: Auto-detection (if both new parameters are not provided)
    if numerical_col_names_all is None and categorical_col_members_all is None:
        print("Mode: Auto-detecting features...")
        # Automatically identify numerical and categorical columns
        numerical_cols = features_gdf.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = features_gdf.select_dtypes(include=['object', 'category']).columns.tolist()

        print(f"Found {len(numerical_cols)} numerical columns: {numerical_cols}")
        print(f"Found {len(categorical_cols)} categorical columns: {categorical_cols}")

        # Process numerical columns
        if numerical_cols:
            numerical_df = features_gdf[numerical_cols].copy()
            feature_components.append(numerical_df)
            for col in numerical_cols:
                mapping[col] = (current_col_idx, current_col_idx + 1, 'numerical')
                current_col_idx += 1

        # Process categorical columns
        if categorical_cols:
            one_hot_encoded = pd.get_dummies(features_gdf[categorical_cols], prefix=categorical_cols, dtype=float)
            feature_components.append(one_hot_encoded)
            for col in categorical_cols:
                prefix = f"{col}_"
                one_hot_cols = [c for c in one_hot_encoded.columns if c.startswith(prefix)]
                num_categories = len(one_hot_cols)
                if num_categories > 0:
                    mapping[col] = (current_col_idx, current_col_idx + num_categories, 'categorical')
                    current_col_idx += num_categories

    # Mode 2: Use predefined schema
    else:
        print("Mode: Using predefined feature schema...")

        # Process numerical columns
        if numerical_col_names_all:
            print(f"Processing {len(numerical_col_names_all)} numerical features per schema...")
            # Create a DataFrame matching the full schema
            numerical_df = pd.DataFrame(index=features_gdf.index, columns=numerical_col_names_all)

            # Find columns that actually exist in the input gdf
            cols_to_fill = [col for col in numerical_col_names_all if col in features_gdf.columns]

            # Fill data from input gdf
            if cols_to_fill:
                numerical_df[cols_to_fill] = features_gdf[cols_to_fill]

            # Fill all missing values (columns not in gdf) with 0
            numerical_df = numerical_df.fillna(0).astype(float)

            feature_components.append(numerical_df)
            for col in numerical_col_names_all:
                mapping[col] = (current_col_idx, current_col_idx + 1, 'numerical')
                current_col_idx += 1

        # Process categorical columns
        if categorical_col_members_all:
            print(f"Processing {len(categorical_col_members_all)} categorical features per schema...")
            all_categorical_dfs = []
            for feature_name, all_categories in categorical_col_members_all.items():
                # Check if the original categorical column exists in the input gdf
                if feature_name in features_gdf.columns:
                    # Use pd.Categorical to ensure all categories are represented, even if absent in current data
                    cat_series = pd.Categorical(features_gdf[feature_name], categories=all_categories)
                    one_hot_df = pd.get_dummies(cat_series, prefix=feature_name, dtype=float)
                else:
                    # If the original column doesn't exist at all, create all-zero dummy columns
                    one_hot_cols = [f"{feature_name}_{cat}" for cat in all_categories]
                    one_hot_df = pd.DataFrame(0, index=features_gdf.index, columns=one_hot_cols, dtype=float)

                all_categorical_dfs.append(one_hot_df)

                num_categories = len(one_hot_df.columns)
                if num_categories > 0:
                    mapping[feature_name] = (current_col_idx, current_col_idx + num_categories, 'categorical')
                    current_col_idx += num_categories

            if all_categorical_dfs:
                feature_components.append(pd.concat(all_categorical_dfs, axis=1))

    # =============================================================================
    # Merge all features and return
    # =============================================================================
    if feature_components:
        final_features = pd.concat(feature_components, axis=1)
    else:
        final_features = pd.DataFrame(index=features_gdf.index)
        print("Warning: No feature columns found; returning empty feature matrix.")

    print(f"Processing complete. Final feature dimensions: {final_features.shape}")
    print(f"Feature mapping: {mapping}")
    print("-" * 30)

    return {
        'final_features': final_features,
        'mapping': mapping
    }


def _build_landuse_matrices(
        gdf_s: gpd.GeoDataFrame,
        gdf_a: gpd.GeoDataFrame,
        s_to_a_edges: torch.Tensor,
        landuse_categorical_col: str = 'landuse',
        landuse_percent_suffix: str = '_percent'
) -> Dict[str, Any]:
    """
    Build landuse_mapping_matrix and landuse_ratio for LandusePredictionLoss.
    V2: Validates and normalizes based on absolute-value percentage columns.
    """
    print("Building landuse supervision matrices (V2)...")

    # 1. Validate input data
    if landuse_categorical_col not in gdf_a.columns:
        print(f"Warning: Column '{landuse_categorical_col}' missing from gdf_a. Skipping landuse matrix construction.")
        return {}

    # 2. Verify that all required landuse ratio columns exist in gdf_s
    # Extract all landuse types that appear in agent data
    agent_landuse_types = gdf_a[landuse_categorical_col].unique()

    # Construct expected column names
    expected_percent_cols = [f"{lu_type}{landuse_percent_suffix}" for lu_type in agent_landuse_types]

    # Check if these columns exist in source data
    missing_cols = [col for col in expected_percent_cols if col not in gdf_s.columns]
    if missing_cols:
        print(f"Warning: gdf_s is missing the following required landuse ratio columns: {missing_cols}. Skipping landuse matrix construction.")
        return {}

    print(f"Validation passed: gdf_s contains all {len(expected_percent_cols)} required landuse ratio columns.")

    # 3. Build landuse_ratio (true ratios)
    # Sort categories alphabetically for deterministic processing order
    landuse_categories = sorted(list(agent_landuse_types))
    sorted_percent_cols = [f"{cat}{landuse_percent_suffix}" for cat in landuse_categories]

    # Extract absolute values
    absolute_values = gdf_s[sorted_percent_cols].values

    # Row-wise normalization
    row_sums = absolute_values.sum(axis=1, keepdims=True)
    # Prevent division by zero: if a row sum is 0, normalized values remain 0
    row_sums[row_sums == 0] = 1

    normalized_ratios = absolute_values / row_sums
    landuse_ratio = torch.tensor(normalized_ratios, dtype=torch.float32)

    # 4. Build landuse_mapping_matrix
    num_s = len(gdf_s)
    num_edges = s_to_a_edges.shape[1]
    num_landuse_types = len(landuse_categories)

    # Convert landuse text categories to indices
    category_to_idx = {cat: i for i, cat in enumerate(landuse_categories)}
    agent_landuse_indices = gdf_a[landuse_categorical_col].map(category_to_idx).fillna(-1).astype(int)

    # Get source and agent indices for each edge
    source_indices = s_to_a_edges[0].numpy()
    agent_indices = s_to_a_edges[1].numpy()

    # Get the landuse type index for the agent of each edge
    # Use .iloc for positional indexing to avoid index alignment issues
    edge_landuse_indices = agent_landuse_indices.iloc[agent_indices].values

    # Create a sparse matrix [num_edges, num_s * num_landuse_types]
    mapping_matrix = torch.zeros((num_edges, num_s * num_landuse_types), dtype=torch.float32)

    for i in range(num_edges):
        s_idx = source_indices[i]
        lu_idx = edge_landuse_indices[i]

        if lu_idx != -1:  # Ensure the agent's landuse type is known
            # Compute column index in the flattened matrix
            col_idx = s_idx * num_landuse_types + lu_idx
            mapping_matrix[i, col_idx] = 1.0

    print("Landuse supervision matrix construction complete.")
    return {
        'landuse_mapping_matrix': mapping_matrix,
        'landuse_ratio': landuse_ratio
    }


def prepare_hetero_graph_from_processed(
        gdf_s: gpd.GeoDataFrame,
        gdf_a: gpd.GeoDataFrame,
        gdf_t: gpd.GeoDataFrame,
        processed_features_s: Dict[str, Any],
        processed_features_a: Dict[str, Any],
        relation_column: str,
) -> HeteroData:
    print("Building heterogeneous graph (HeteroData) object...")

    # 1. Initialize an empty HeteroData object
    data = HeteroData()

    # 2. Process 'source' nodes
    num_s = len(gdf_s)
    coords_s = np.array([[point.x, point.y] for point in gdf_s.geometry])
    features_s_df = processed_features_s['final_features']
    features_s_np = features_s_df.values if not features_s_df.empty else np.empty((num_s, 0))

    # Store source node features and coordinates
    data['source'].x = torch.tensor(features_s_np, dtype=torch.float32)
    data['source'].coords = torch.tensor(coords_s, dtype=torch.float32)
    # num_nodes is inferred automatically

    # 3. Process 'agent' nodes
    num_a = len(gdf_a)
    coords_a = np.array([[point.x, point.y] for point in gdf_a.geometry])
    features_a_df = processed_features_a['final_features']
    features_a_np = features_a_df.values if not features_a_df.empty else np.empty((num_a, 0))

    # Store agent node features and coordinates
    data['agent'].x = torch.tensor(features_a_np, dtype=torch.float32)
    data['agent'].coords = torch.tensor(coords_a, dtype=torch.float32)

    print(f"Source nodes: {data['source'].num_nodes}, feature dim: {data['source'].num_features}")
    print(f"Agent nodes: {data['agent'].num_nodes}, feature dim: {data['agent'].num_features}")

    # 4. Build edge relations (core change)
    # First, get the global-indexed edge list as before
    global_edges = _build_source_agent_graph(gdf_s, gdf_a, relation_column)

    # Then, convert global indices to local indices for specific edge types
    s_to_a_edges = []
    a_to_s_edges = []
    for u, v in global_edges:
        if u < num_s and v >= num_s:  # This is a Source -> Agent edge
            local_u = u  # Source node index is local
            local_v = v - num_s  # Agent node index needs offset subtracted to become local
            s_to_a_edges.append([local_u, local_v])
        elif u >= num_s and v < num_s:  # This is an Agent -> Source edge
            local_u = u - num_s  # Agent node index is local
            local_v = v  # Source node index is local
            a_to_s_edges.append([local_u, local_v])

    s_to_a_edges_tensor = None
    # Store processed edge lists in the corresponding edge stores
    # We define two relations: 'connects_to' and its reverse 'rev_connects_to'
    if s_to_a_edges:
        s_to_a_edges_tensor = torch.tensor(s_to_a_edges,dtype=torch.long).t().contiguous()
        data['source', 'connects_to', 'agent'].edge_index = s_to_a_edges_tensor
    if a_to_s_edges:
        data['agent', 'rev_connects_to', 'source'].edge_index = torch.tensor(a_to_s_edges,
                                                                             dtype=torch.long).t().contiguous()

    print(f"Source->Agent edges: {data['source', 'connects_to', 'agent'].num_edges}")
    print(f"Agent->Source edges: {data['agent', 'rev_connects_to', 'source'].num_edges}")

    # 5. Build and add landuse supervision matrices
    if s_to_a_edges_tensor is not None:
        landuse_data = _build_landuse_matrices(gdf_s, gdf_a, s_to_a_edges_tensor)
        if landuse_data:
            data.landuse_mapping_matrix = landuse_data['landuse_mapping_matrix']
            data.landuse_ratio = landuse_data['landuse_ratio']
            print("Added landuse_mapping_matrix and landuse_ratio to the graph object.")

    # 6. Attach other metadata
    # These are stored as graph-level global attributes
    data.feature_mapping_s = processed_features_s['mapping']
    data.feature_mapping_a = processed_features_a['mapping']

    # Add source node true demand as evaluation target (used only for post-training evaluation, not for training loss)
    if 'Demand (MVA)' in gdf_s.columns:
        source_demand_true = gdf_s['Demand (MVA)'].values
        data['source'].y = torch.tensor(source_demand_true, dtype=torch.float32)
        print("Added evaluation target 'y' for source nodes.")

    if 'Demand (MVA)' in gdf_a.columns:
        agent_demand = gdf_a['Demand (MVA)'].values
        data['agent'].demand = torch.tensor(agent_demand, dtype=torch.float32)
        print("Added base demand 'demand' for agent nodes.")

    if 'substation_idx' in gdf_a.columns:
        agent_substation_map = gdf_a['substation_idx'].values
        data['agent'].substation_idx = torch.tensor(agent_substation_map, dtype=torch.long)
        print("Added agent-to-substation mapping 'substation_idx'.")

    # Add substation true demand as evaluation target (D_t, used ONLY for post-training evaluation metrics like RMSE/MAE, NOT for training loss)
    if 'Demand (MVA)' in gdf_t.columns:
        substation_demand_true = gdf_t['Demand (MVA)'].values
        data.substation_y = torch.tensor(substation_demand_true, dtype=torch.float32).reshape(-1)  # Ensure 1D tensor
        print(data.substation_y.shape)
        data.num_substations = len(gdf_t)
        print("Added substation true demand 'substation_y' as evaluation target (graph-level attribute).")

    data.source_index_map = pd.Series(gdf_s.index.values)
    data.agent_index_map = pd.Series(gdf_a.index.values)
    print("Added original index mappings 'source_index_map' and 'agent_index_map'.")

    print("\n" + "=" * 50)
    print("Heterogeneous graph (HeteroData) object construction complete!")
    print("=" * 50 + "\n")

    return data