import numpy as np
import torch
import geopandas as gpd
from typing import Dict, Any, Optional, List
from torch_geometric.data import HeteroData
from scipy.spatial import KDTree
from SpatialAllocation.GNN.Layer.PositionalEncoding import sinusoidal_pe
from SpatialAllocation.GNN.utils.GraphBuilder import preprocess_features


def build_agent_target_knn(
        agent_coords: np.ndarray,
        target_coords: np.ndarray,
        k: int = 5,
) -> tuple:
    """
    Builds agent -> target k-NN edge indices.

    Args:
        agent_coords: (N_a, 2) agent projected coordinates (in meters)
        target_coords: (N_t, 2) target projected coordinates (in meters)
        k: number of nearest targets to connect per agent

    Returns:
        edge_index_at: (2, N_a * k) agent->target edge index
        edge_index_ta: (2, N_a * k) target->agent reverse edge index
    """
    N_a = len(agent_coords)
    N_t = len(target_coords)
    actual_k = min(k, N_t)

    # Check coordinate range and warn if needed
    coord_max = max(np.abs(agent_coords).max(), np.abs(target_coords).max())
    if coord_max <= 180.0:
        import warnings
        coord_std = max(agent_coords.std(), target_coords.std())
        if coord_std < 5.0 and coord_max < 10.0:
            warnings.warn(
                f"Detected pre-normalized coordinates (max={coord_max:.2f}, std={coord_std:.2f}); "
                f"k-NN will be computed in normalized space.",
                stacklevel=2,
            )
        else:
            warnings.warn(
                f"Input coordinate range is within [-180, 180] (max={coord_max:.2f}); "
                f"these may be lon/lat coordinates. A projected coordinate system (in meters) is recommended.",
                stacklevel=2,
            )

    tree = KDTree(target_coords)
    _, knn_indices = tree.query(agent_coords, k=actual_k)
    # scipy returns shape (N_a,) when k=1; reshape to 2D for consistency
    if actual_k == 1:
        knn_indices = knn_indices.reshape(-1, 1)

    # Build edge index
    agent_idx = np.repeat(np.arange(N_a), actual_k)
    target_idx = knn_indices.flatten()

    edge_index_at = torch.tensor(
        np.stack([agent_idx, target_idx], axis=0), dtype=torch.long
    )
    edge_index_ta = torch.tensor(
        np.stack([target_idx, agent_idx], axis=0), dtype=torch.long
    )

    return edge_index_at, edge_index_ta


def build_allocation_graph(
        gdf_agent: gpd.GeoDataFrame,
        gdf_target: gpd.GeoDataFrame,
        agent_feature_cols: Optional[List[str]] = None,
        agent_weights: Optional[np.ndarray] = None,
        k: int = 5,
        pe_L: int = 16,
) -> HeteroData:
    """
    Builds an agent-target bipartite graph for Stage 2 allocation training.

    Args:
        gdf_agent: Agent GeoDataFrame (contains coordinates, landuse features, etc.)
        gdf_target: Target GeoDataFrame (contains coordinates, measured demand)
        agent_feature_cols: List of agent numeric feature column names (auto-detected if None)
        agent_weights: Stage 1 predicted agent weighted demand (N_a,), optional feature
        k: number of k-NN connections
        pe_L: number of sinusoidal positional encoding frequencies (output dim = 4L)

    Returns:
        HeteroData containing:
        - data['agent'].x: agent features
        - data['agent'].coords: agent coordinates
        - data['target'].x: target features (coordinate PE)
        - data['target'].coords: target coordinates
        - data['target'].demand: measured demand
        - data['agent', 'connects_to', 'target'].edge_index
        - data['target', 'rev_connects_to', 'agent'].edge_index
    """
    data = HeteroData()
    num_a = len(gdf_agent)
    num_t = len(gdf_target)

    # === Agent nodes ===
    coords_a = np.array([[pt.x, pt.y] for pt in gdf_agent.geometry], dtype=np.float32)

    # Build agent features
    if agent_feature_cols is not None:
        # Use specified columns
        agent_features_np = gdf_agent[agent_feature_cols].values.astype(np.float32)
    else:
        # Auto-detect numeric columns
        result = preprocess_features(gdf_agent)
        features_df = result['final_features']
        agent_features_np = features_df.values.astype(np.float32) if not features_df.empty else np.empty((num_a, 0), dtype=np.float32)

    # Concatenate Stage 1 weight feature (optional)
    if agent_weights is not None:
        if agent_weights.shape[0] != num_a:
            raise ValueError(
                f"agent_weights length ({agent_weights.shape[0]}) does not match agent count ({num_a})"
            )
        w_col = agent_weights.reshape(-1, 1).astype(np.float32)
        if agent_features_np.shape[1] > 0:
            agent_features_np = np.concatenate([agent_features_np, w_col], axis=1)
        else:
            agent_features_np = w_col
        print(f"Concatenated Stage 1 weight feature; total agent feature dimension: {agent_features_np.shape[1]}")

    data['agent'].x = torch.tensor(agent_features_np, dtype=torch.float32)
    data['agent'].coords = torch.tensor(coords_a, dtype=torch.float32)

    # Agent demand (used for allocation computation)
    if 'Demand (MVA)' in gdf_agent.columns:
        data['agent'].demand = torch.tensor(
            gdf_agent['Demand (MVA)'].values, dtype=torch.float32
        )

    # === Target nodes ===
    coords_t = np.array([[pt.x, pt.y] for pt in gdf_target.geometry], dtype=np.float32)
    coords_t_tensor = torch.tensor(coords_t, dtype=torch.float32)

    # Target features: sinusoidal positional encoding of coordinates
    target_pe = sinusoidal_pe(coords_t_tensor, L=pe_L, scale_factor=100000.0)

    data['target'].x = target_pe
    data['target'].coords = coords_t_tensor

    # Target demand (ground truth)
    if 'Demand (MVA)' in gdf_target.columns:
        data['target'].demand = torch.tensor(
            gdf_target['Demand (MVA)'].values, dtype=torch.float32
        )
        print(f"Target demand added, total: {gdf_target['Demand (MVA)'].sum():.1f} MVA")

    print(f"Agent nodes: {num_a}, feature dimension: {data['agent'].x.shape[1]}")
    print(f"Target nodes: {num_t}, feature dimension: {data['target'].x.shape[1]}")

    # === Build k-NN edges ===
    edge_index_at, edge_index_ta = build_agent_target_knn(
        agent_coords=coords_a,
        target_coords=coords_t,
        k=k,
    )

    data['agent', 'connects_to', 'target'].edge_index = edge_index_at
    data['target', 'rev_connects_to', 'agent'].edge_index = edge_index_ta
    print(f"Agent->Target edges: {edge_index_at.shape[1]} (k={k})")
    print(f"Target->Agent edges: {edge_index_ta.shape[1]}")

    # Index mapping (used to recover original GeoDataFrame indices at inference time)
    import pandas as pd
    data.agent_index_map = pd.Series(gdf_agent.index.values)
    data.target_index_map = pd.Series(gdf_target.index.values)

    print(f"\n{'=' * 50}")
    print("Allocation graph (HeteroData) construction complete!")
    print(f"{'=' * 50}\n")

    return data
