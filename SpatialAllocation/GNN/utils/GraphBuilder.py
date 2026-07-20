import time
import pandas as pd
from torch_geometric.data import Data
import torch
import numpy as np
import geopandas as gpd
from typing import Dict, Any, Optional, List
from torch_geometric.data import HeteroData
from SpatialAllocation.GNN.Layer.PositionalEncoding import sinusoidal_pe


def _build_source_agent_graph(gdf_s: gpd.GeoDataFrame, gdf_a: gpd.GeoDataFrame,
                              relation_column: str) -> list:
    """
    Build a region-point graph based on a precomputed membership relation.

    Args:
        gdf_s (gpd.GeoDataFrame): Region data (polygon centroids)
        gdf_a (gpd.GeoDataFrame): Point data
        relation_column (str): Name of the shared index column expressing the membership relation
        include_region_connections (bool): Whether to include connections between regions (optional)

    Returns:
        list: The graph's edge list
    """
    graph_edges = []

    # Verify that the index column exists
    if relation_column not in gdf_s.columns:
        raise ValueError(f"Index column '{relation_column}' does not exist in gdf_s")
    if relation_column not in gdf_a.columns:
        raise ValueError(f"Index column '{relation_column}' does not exist in gdf_a")

    # Create the index mapping
    # gdf_s index mapping: region_id -> node_index_in_graph
    region_to_node = {region_id: i for i, region_id in enumerate(gdf_s[relation_column])}

    # gdf_a index mapping: region_id -> [point_indices_in_graph]
    point_groups = {}
    for i, region_id in enumerate(gdf_a[relation_column]):
        if region_id not in point_groups:
            point_groups[region_id] = []
        point_groups[region_id].append(len(gdf_s) + i)  # Point node indices start from len(gdf_s)

    # 1. Build region-point connections (based on the membership relation)
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
    Preprocess features in a DataFrame either automatically or according to
    a predefined schema, converting categorical text into numerical format.

    Args:
        gdf (gpd.GeoDataFrame): Input GeoDataFrame containing mixed-type features.
        numerical_col_names_all (Optional[List[str]]):
            A list of all expected numerical feature column names. If provided,
            this list is used instead of automatic detection.
        categorical_col_members_all (Optional[Dict[str, List[str]]]):
            A dictionary whose keys are categorical feature names and whose
            values are lists of all possible category members for that feature.
            If provided, this dict is used for one-hot encoding to ensure
            consistency of the feature matrix.

    Returns:
        Dict[str, Any]: A dictionary containing the following key-value pairs:
            - 'final_features': pd.DataFrame containing only pure numerical features.
            - 'mapping': Dict mapping feature names to column ranges
              {feature_name: (start_col, end_col, feature_type)}.
    """
    # 1. Exclude the 'geometry' column
    features_gdf = gdf.drop(columns='geometry', errors='ignore')

    # Initialize the mapping dict and feature list
    mapping = {}
    feature_components = []
    current_col_idx = 0

    # =============================================================================
    # Mode selection: automatic detection vs. predefined schema
    # =============================================================================

    # Mode 1: automatic detection (if neither of the two new parameters is provided)
    if numerical_col_names_all is None and categorical_col_members_all is None:
        print("Mode: automatic feature detection...")
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

    # Mode 2: use the predefined schema
    else:
        print("Mode: using the predefined feature schema...")

        # Process numerical columns
        if numerical_col_names_all:
            print(f"Processing {len(numerical_col_names_all)} numerical features per the schema...")
            # Create a DataFrame that conforms to the full schema
            numerical_df = pd.DataFrame(index=features_gdf.index, columns=numerical_col_names_all)

            # Find the columns that actually exist in the input gdf
            cols_to_fill = [col for col in numerical_col_names_all if col in features_gdf.columns]

            # Fill in data from the input gdf
            if cols_to_fill:
                numerical_df[cols_to_fill] = features_gdf[cols_to_fill]

            # Fill all missing values (i.e. columns absent from gdf) with 0
            numerical_df = numerical_df.fillna(0).astype(float)

            feature_components.append(numerical_df)
            for col in numerical_col_names_all:
                mapping[col] = (current_col_idx, current_col_idx + 1, 'numerical')
                current_col_idx += 1

        # Process categorical columns
        if categorical_col_members_all:
            print(f"Processing {len(categorical_col_members_all)} categorical features per the schema...")
            all_categorical_dfs = []
            for feature_name, all_categories in categorical_col_members_all.items():
                # Check whether the original categorical column exists in the input gdf
                if feature_name in features_gdf.columns:
                    # Use pd.Categorical to ensure all categories are accounted for,
                    # even if they do not appear in the current data
                    cat_series = pd.Categorical(features_gdf[feature_name], categories=all_categories)
                    one_hot_df = pd.get_dummies(cat_series, prefix=feature_name, dtype=float)
                else:
                    # If the original column does not exist at all, create dummy columns filled with 0
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
        print("Warning: no feature columns found; returning an empty feature matrix.")

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
    V2: validates and normalizes based on absolute-value percentage columns.
    """
    print("Building the land-use supervision matrix (V2)...")

    # 1. Validate the input data
    if landuse_categorical_col not in gdf_a.columns:
        print(f"Warning: gdf_a is missing the '{landuse_categorical_col}' column. Skipping land-use matrix construction.")
        return {}

    # 2. Verify that gdf_s contains all the required land-use ratio columns
    # Extract all land-use types that appear in the agent data
    agent_landuse_types = gdf_a[landuse_categorical_col].unique()

    # Build the expected column names
    expected_percent_cols = [f"{lu_type}{landuse_percent_suffix}" for lu_type in agent_landuse_types]

    # Check whether these columns exist in the source data
    missing_cols = [col for col in expected_percent_cols if col not in gdf_s.columns]
    if missing_cols:
        print(f"Warning: gdf_s is missing the following required land-use ratio columns: {missing_cols}. Skipping land-use matrix construction.")
        return {}

    print(f"Validation passed: gdf_s contains all {len(expected_percent_cols)} required land-use ratio columns.")

    # 3. Build landuse_ratio (the ground-truth ratio)
    # Sort categories alphabetically to ensure deterministic processing order
    landuse_categories = sorted(list(agent_landuse_types))
    sorted_percent_cols = [f"{cat}{landuse_percent_suffix}" for cat in landuse_categories]

    # Extract the absolute values
    absolute_values = gdf_s[sorted_percent_cols].values

    # Row-wise normalization
    row_sums = absolute_values.sum(axis=1, keepdims=True)
    # Prevent division by zero: if a row's sum is 0, it remains 0 after normalization
    row_sums[row_sums == 0] = 1

    normalized_ratios = absolute_values / row_sums
    landuse_ratio = torch.tensor(normalized_ratios, dtype=torch.float32)

    # 4. Build landuse_mapping_matrix
    num_s = len(gdf_s)
    num_edges = s_to_a_edges.shape[1]
    num_landuse_types = len(landuse_categories)

    # Convert land-use text categories to indices
    category_to_idx = {cat: i for i, cat in enumerate(landuse_categories)}
    agent_landuse_indices = gdf_a[landuse_categorical_col].map(category_to_idx).fillna(-1).astype(int)

    # Get the source and agent indices for each edge
    source_indices = s_to_a_edges[0].numpy()
    agent_indices = s_to_a_edges[1].numpy()

    # Get the land-use type index of the agent corresponding to each edge.
    # Use .iloc to guarantee positional indexing, avoiding index-alignment issues
    edge_landuse_indices = agent_landuse_indices.iloc[agent_indices].values

    # Create a sparse matrix [num_edges, num_s * num_landuse_types]
    mapping_matrix = torch.zeros((num_edges, num_s * num_landuse_types), dtype=torch.float32)

    for i in range(num_edges):
        s_idx = source_indices[i]
        lu_idx = edge_landuse_indices[i]

        if lu_idx != -1:  # Ensure the agent's land-use type is known
            # Compute the column index in the flattened matrix
            col_idx = s_idx * num_landuse_types + lu_idx
            mapping_matrix[i, col_idx] = 1.0

    print("Land-use supervision matrix construction complete.")
    return {
        'landuse_mapping_matrix': mapping_matrix,
        'landuse_ratio': landuse_ratio
    }


def build_agent_adjacency(
        coords: np.ndarray,
        threshold: float = None,
) -> torch.Tensor:
    """
    Build agent-agent adjacency edge pairs: pairs agents whose distance is
    below a threshold, based on spatial proximity.

    Used in SemanticLoss to compute spectral similarity between neighboring agents.

    Args:
        coords: (N_a, 2) agent projected coordinates (in meters)
        threshold: Distance threshold. If unspecified, automatically computed
            as median(nearest-neighbor distance) x 1.5

    Returns:
        (2, num_pairs) symmetric edge-pair tensor (bidirectional: both (i,j)
        and (j,i) are included)
    """
    from scipy.spatial import KDTree

    N_a = len(coords)
    if N_a < 2:
        return torch.zeros((2, 0), dtype=torch.long)

    tree = KDTree(coords)

    # Automatic threshold: median(nearest-neighbor distance) x 1.5
    if threshold is None:
        nn_dist, _ = tree.query(coords, k=2)  # k=2: nearest neighbor (excluding itself)
        median_nn_dist = np.median(nn_dist[:, 1])
        threshold = median_nn_dist * 1.5

    # Find all neighbors within the threshold
    neighbors = tree.query_ball_point(coords, r=threshold)

    src_list = []
    dst_list = []
    for i, nbrs in enumerate(neighbors):
        for j in nbrs:
            if i != j:  # Exclude self-loops
                src_list.append(i)
                dst_list.append(j)

    if len(src_list) == 0:
        return torch.zeros((2, 0), dtype=torch.long)

    edge_index = torch.tensor(
        np.stack([src_list, dst_list], axis=0), dtype=torch.long
    )
    return edge_index


def build_grid_adjacency(
        coords: np.ndarray,
        mode: str = 'neumann',
        step_size: float = None,
        tolerance: float = 0.1,
) -> torch.Tensor:
    """
    Build agent-agent adjacency edges based on a regular grid topology.

    Supports two neighborhood modes:
    - Von Neumann (4-connectivity): only orthogonal up/down/left/right neighbors
    - Moore (8-connectivity): orthogonal + diagonal neighbors

    Args:
        coords: (N, 2) agent coordinates (any coordinate system)
        mode: 'neumann' (4-neighborhood) or 'moore' (8-neighborhood)
        step_size: Grid step size; auto-inferred from median(nearest-neighbor
            distance) when None
        tolerance: Distance-matching tolerance ratio, default 0.1 (10%)

    Returns:
        (2, num_edges) bidirectional symmetric edge index
    """
    from scipy.spatial import KDTree

    if mode not in ('neumann', 'moore'):
        raise ValueError(f"Unsupported neighborhood mode: '{mode}'. Supported values: 'neumann' or 'moore'")

    N = len(coords)
    if N < 2:
        return torch.zeros((2, 0), dtype=torch.long)

    tree = KDTree(coords)

    # Automatically infer the step size: median(nearest-neighbor distance)
    if step_size is None:
        nn_dist, _ = tree.query(coords, k=2)  # k=2: excludes itself
        step_size = float(np.median(nn_dist[:, 1]))

    # Search radius: moore needs to cover the diagonal distance step*sqrt(2)
    if mode == 'neumann':
        search_radius = step_size * (1.0 + tolerance)
    else:  # moore
        search_radius = step_size * np.sqrt(2) * (1.0 + tolerance)

    # Distance tolerance range
    ortho_lo = step_size * (1.0 - tolerance)
    ortho_hi = step_size * (1.0 + tolerance)
    diag_dist = step_size * np.sqrt(2)
    diag_lo = diag_dist * (1.0 - tolerance)
    diag_hi = diag_dist * (1.0 + tolerance)

    # Find candidate neighbors
    neighbors = tree.query_ball_point(coords, r=search_radius)

    src_list = []
    dst_list = []
    for i, nbrs in enumerate(neighbors):
        for j in nbrs:
            if i < j:  # Deduplicate: only keep i < j
                dist = np.linalg.norm(coords[i] - coords[j])
                if mode == 'neumann':
                    if ortho_lo <= dist <= ortho_hi:
                        src_list.append(i)
                        dst_list.append(j)
                else:  # moore
                    if ortho_lo <= dist <= ortho_hi or diag_lo <= dist <= diag_hi:
                        src_list.append(i)
                        dst_list.append(j)

    if len(src_list) == 0:
        return torch.zeros((2, 0), dtype=torch.long)

    # Symmetrize: add the reverse direction
    src_arr = np.array(src_list)
    dst_arr = np.array(dst_list)
    edge_index = torch.tensor(
        np.stack([
            np.concatenate([src_arr, dst_arr]),
            np.concatenate([dst_arr, src_arr]),
        ], axis=0),
        dtype=torch.long,
    )
    return edge_index


def prepare_hetero_graph_from_processed(
        gdf_s: gpd.GeoDataFrame,
        gdf_a: gpd.GeoDataFrame,
        processed_features_s: Dict[str, Any],
        processed_features_a: Dict[str, Any],
        relation_column: str,
        spectral_features: Optional[np.ndarray] = None,
        spectral_column_names: Optional[List[str]] = None,
        macro_attributes: Optional[np.ndarray] = None,
        macro_column_names: Optional[List[str]] = None,
        attribute_stds: Optional[np.ndarray] = None,
        agent_connectivity: Optional[str] = None,
) -> HeteroData:
    print("Building the heterogeneous graph (HeteroData) object...")

    # 1. Initialize an empty HeteroData object
    data = HeteroData()

    # 2. Process the 'source' nodes
    num_s = len(gdf_s)
    coords_s = np.array([[geom.centroid.x, geom.centroid.y] for geom in gdf_s.geometry])
    features_s_df = processed_features_s['final_features']
    features_s_np = features_s_df.values if not features_s_df.empty else np.empty((num_s, 0))

    # Store the 'source' node features and coordinates in the 'source' node store
    data['source'].x = torch.tensor(features_s_np, dtype=torch.float32)
    data['source'].coords = torch.tensor(coords_s, dtype=torch.float32)
    # num_nodes is inferred automatically

    # 3. Process the 'agent' nodes
    num_a = len(gdf_a)
    coords_a = np.array([[point.x, point.y] for point in gdf_a.geometry])
    features_a_df = processed_features_a['final_features']
    features_a_np = features_a_df.values if not features_a_df.empty else np.empty((num_a, 0))

    # If a spectral feature array is provided, concatenate it onto the agent features
    if spectral_features is not None:
        if spectral_features.shape[0] != num_a:
            raise ValueError(
                f"Spectral feature row count ({spectral_features.shape[0]}) does not match the number of agents ({num_a})"
            )
        # Concatenate the existing features with the spectral features
        if features_a_np.shape[1] > 0:
            features_a_np = np.concatenate([features_a_np, spectral_features], axis=1)
        else:
            features_a_np = spectral_features
        spec_dim = spectral_features.shape[1]
        if spectral_column_names is not None:
            print(f"Concatenated spectral features ({spec_dim} dims): {spectral_column_names}, total agent feature dimension: {features_a_np.shape[1]}")
        else:
            print(f"Concatenated spectral features ({spec_dim} dims), total agent feature dimension: {features_a_np.shape[1]}")

    # Store the 'agent' node features and coordinates in the 'agent' node store
    data['agent'].x = torch.tensor(features_a_np, dtype=torch.float32)
    data['agent'].coords = torch.tensor(coords_a, dtype=torch.float32)

    # Store the macro attributes (used for the L_recon loss)
    if macro_attributes is not None:
        data.macro_attributes = torch.tensor(macro_attributes, dtype=torch.float32)
        macro_dim = macro_attributes.shape[1]
        if macro_column_names is not None:
            print(f"Added macro attributes ({macro_dim} dims): {macro_column_names}")
        else:
            print(f"Added macro attributes ({macro_dim} dims)")
    if attribute_stds is not None:
        data.attribute_stds = torch.tensor(attribute_stds, dtype=torch.float32)

    print(f"Source nodes: {data['source'].num_nodes}, feature dimension: {data['source'].num_features}")
    print(f"Agent nodes: {data['agent'].num_nodes}, feature dimension: {data['agent'].num_features}")

    # 4. Build the edge relations (the core part of this step)
    # First, get the edge list in global indices as before
    global_edges = _build_source_agent_graph(gdf_s, gdf_a, relation_column)

    # Then convert the global indices into edge-type-local indices
    s_to_a_edges = []
    a_to_s_edges = []
    for u, v in global_edges:
        if u < num_s and v >= num_s:  # This is a Source -> Agent edge
            local_u = u  # The source node index is already local
            local_v = v - num_s  # The agent node index needs the offset subtracted to become local
            s_to_a_edges.append([local_u, local_v])
        elif u >= num_s and v < num_s:  # This is an Agent -> Source edge
            local_u = u - num_s  # The agent node index is already local
            local_v = v  # The source node index is already local
            a_to_s_edges.append([local_u, local_v])

    s_to_a_edges_tensor = None
    # Store the processed edge lists in the corresponding edge stores.
    # We define two relations: 'connects_to' and its reverse 'rev_connects_to'
    if s_to_a_edges:
        s_to_a_edges_tensor = torch.tensor(s_to_a_edges,dtype=torch.long).t().contiguous()
        data['source', 'connects_to', 'agent'].edge_index = s_to_a_edges_tensor
    if a_to_s_edges:
        data['agent', 'rev_connects_to', 'source'].edge_index = torch.tensor(a_to_s_edges,
                                                                             dtype=torch.long).t().contiguous()

    print(f"Source->Agent edges: {data['source', 'connects_to', 'agent'].num_edges}")
    print(f"Agent->Source edges: {data['agent', 'rev_connects_to', 'source'].num_edges}")

    # 5. Build and attach the matrices required for land-use supervision
    if s_to_a_edges_tensor is not None:
        landuse_data = _build_landuse_matrices(gdf_s, gdf_a, s_to_a_edges_tensor)
        if landuse_data:
            data.landuse_mapping_matrix = landuse_data['landuse_mapping_matrix']
            data.landuse_ratio = landuse_data['landuse_ratio']
            print("Added landuse_mapping_matrix and landuse_ratio to the graph object.")

    # 6. Attach other metadata
    # These can serve as graph-level global attributes
    data.feature_mapping_s = processed_features_s['mapping']
    data.feature_mapping_a = processed_features_a['mapping']
    # The original GeoDataFrame could also be stored here, but be mindful of memory usage
    # data.source_gdf = gdf_s
    # data.agent_gdf = gdf_a
    # Add the source node's true demand as a supervision target
    if 'Demand (MVA)' in gdf_s.columns:
        source_demand_true = gdf_s['Demand (MVA)'].values
        data['source'].y = torch.tensor(source_demand_true, dtype=torch.float32)
        print(f"Added the supervision target 'y' for source nodes.")

    if 'Demand (MVA)' in gdf_a.columns:
        agent_demand = gdf_a['Demand (MVA)'].values
        data['agent'].demand = torch.tensor(agent_demand, dtype=torch.float32)
        print(f"Added the base demand 'demand' for agent nodes.")

    data.source_index_map = pd.Series(gdf_s.index.values)
    data.agent_index_map = pd.Series(gdf_a.index.values)
    print("Added the original index mappings 'source_index_map' and 'agent_index_map' for source and agent.")

    # 6.5 Agent-Agent grid adjacency edges (optional)
    if agent_connectivity is not None:
        agent_near_edges = build_grid_adjacency(coords_a, mode=agent_connectivity)
        if agent_near_edges.shape[1] > 0:
            data['agent', 'near', 'agent'].edge_index = agent_near_edges
            # Also set agent_adj_pairs (used by loss functions such as FeatureConsistencyLoss)
            data.agent_adj_pairs = agent_near_edges
            print(f"Agent-Agent adjacency edges ('{agent_connectivity}'): {agent_near_edges.shape[1]}")
            print(f"agent_adj_pairs set: {agent_near_edges.shape[1]} pairs")
        else:
            print(f"Warning: agent_connectivity='{agent_connectivity}' but no adjacency edges were found")
    else:
        # Without grid connectivity, build adjacency pairs using a KDTree distance
        # threshold (loss functions can still make use of them)
        agent_adj_pairs = build_agent_adjacency(coords_a)
        data.agent_adj_pairs = agent_adj_pairs
        print(f"agent_adj_pairs (KDTree): {agent_adj_pairs.shape[1]} pairs")

    print("\n" + "=" * 50)
    print("Heterogeneous graph (HeteroData) object construction complete!")
    print("=" * 50 + "\n")

    return data


