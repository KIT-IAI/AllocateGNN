# -*- coding: utf-8 -*-
"""
Road network topological distance computation module

This module provides road-network distance computation based on OSMnx,
used in the spatial load allocation model to compute the true road-network
distance between agent nodes and target nodes.

Main features:
- Download OpenStreetMap road network data
- Map geographic points to road network nodes
- Compute a sparse road-network distance matrix (only the K nearest targets
  are kept for each agent)

Author: AllocateGNN Team
Date: 2026-01-22
"""

import hashlib
import json
import logging
import os
import pickle
import platform
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import Point

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration class
# =============================================================================

@dataclass
class NetworkDistanceConfig:
    """
    Road-network distance computation configuration.

    Attributes:
        network_type: Road network type ('drive', 'walk', 'bike', 'all')
        k_nearest: Number of nearest targets kept for each agent
        cache_dir: Road network cache directory
        use_cache: Whether to use the road-network cache
        use_multiprocess: Whether to use multiprocessing acceleration
        max_workers: Maximum number of worker processes (None means automatic)
        weight: Edge weight attribute name ('length' means distance, in meters)
        max_mapping_distance: Maximum allowed distance for mapping a point to the road network (meters)
    """
    network_type: str = 'drive'
    k_nearest: int = 20
    cache_dir: str = field(default_factory=lambda: _get_default_cache_dir())
    use_cache: bool = True
    use_multiprocess: bool = True
    max_workers: Optional[int] = None
    weight: str = 'length'
    max_mapping_distance: float = 2000.0


def _get_default_cache_dir() -> str:
    """Get the default road-network cache directory."""
    project_root = Path(__file__).parent.parent.parent
    cache_dir = project_root / "data" / "network_cache"
    return str(cache_dir)


def _get_default_max_workers() -> int:
    """Get the default number of worker processes."""
    cpu_count = os.cpu_count() or 1
    # Reserve one core for the system
    return max(1, cpu_count - 1)


# =============================================================================
# Private helper functions
# =============================================================================

def _compute_bounds_hash(bounds: Tuple[float, float, float, float], network_type: str) -> str:
    """
    Compute a hash of the bounding box, used to name cache files.

    Args:
        bounds: (minx, miny, maxx, maxy) bounding box
        network_type: Road network type

    Returns:
        str: Hash string (first 16 characters)
    """
    bounds_str = f"{bounds[0]:.6f}_{bounds[1]:.6f}_{bounds[2]:.6f}_{bounds[3]:.6f}_{network_type}"
    return hashlib.md5(bounds_str.encode()).hexdigest()[:16]


def _ensure_projected_crs(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Ensure the GeoDataFrame uses a projected CRS (for distance computation).

    If it is a geographic CRS (EPSG:4326), convert it to Web Mercator (EPSG:3857).

    Args:
        gdf: The input GeoDataFrame

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame using a projected CRS
    """
    if gdf.crs is None:
        logger.warning("The GeoDataFrame has no CRS set; assuming EPSG:4326")
        gdf = gdf.set_crs("EPSG:4326")

    if gdf.crs.is_geographic:
        gdf = gdf.to_crs("EPSG:3857")
        logger.debug("Converted the CRS from geographic coordinates to EPSG:3857")

    return gdf


def _compute_euclidean_distance(
    point1: Tuple[float, float],
    point2: Tuple[float, float],
    crs: str = "EPSG:4326",
    target_crs: str = "EPSG:3857"
) -> float:
    """
    Compute the Euclidean distance (meters) between two points.

    Args:
        point1: (lon, lat) or (x, y) coordinates
        point2: (lon, lat) or (x, y) coordinates
        crs: Coordinate reference system
        target_crs: Projection target for geographic coordinates (default
            EPSG:3857; EPSG:27700 is recommended for the UK)

    Returns:
        float: Distance (meters)
    """
    import pyproj
    from shapely.geometry import Point as ShapelyPoint
    from shapely.ops import transform

    p1 = ShapelyPoint(point1)
    p2 = ShapelyPoint(point2)

    if crs == "EPSG:4326":
        # Geographic coordinates need to be projected to planar coordinates for distance computation
        project = pyproj.Transformer.from_crs(
            "EPSG:4326", target_crs, always_xy=True
        ).transform
        p1 = transform(project, p1)
        p2 = transform(project, p2)

    return p1.distance(p2)


# =============================================================================
# Road network download functionality
# =============================================================================

def download_road_network(
    bounds: Tuple[float, float, float, float],
    network_type: str = 'drive',
    cache_dir: Optional[str] = None,
    use_cache: bool = True
) -> nx.MultiDiGraph:
    """
    Download the road network data within the given bounding box.

    Uses OSMnx to fetch the road network from OpenStreetMap, with local
    caching support to avoid repeated downloads.

    Args:
        bounds: (minx, miny, maxx, maxy) bounding box, using WGS84 coordinates (EPSG:4326)
            - minx: western boundary longitude
            - miny: southern boundary latitude
            - maxx: eastern boundary longitude
            - maxy: northern boundary latitude
        network_type: Road network type
            - 'drive': drivable roads (default)
            - 'walk': pedestrian paths
            - 'bike': cycling paths
            - 'all': all roads
        cache_dir: Cache directory path, defaults to data/network_cache/
        use_cache: Whether to use the cache, default True

    Returns:
        nx.MultiDiGraph: A NetworkX directed multigraph, where nodes have
                        (x, y) coordinate attributes and edges have a
                        'length' (meters) attribute

    Raises:
        ValueError: Invalid bounding box parameters
        RuntimeError: Road network download failed

    Example:
        >>> bounds = (-0.2, 51.4, 0.0, 51.6)  # Part of London
        >>> G = download_road_network(bounds, network_type='drive')
        >>> print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    """
    import osmnx as ox

    # Validate the bounding box
    minx, miny, maxx, maxy = bounds
    if minx >= maxx or miny >= maxy:
        raise ValueError(
            f"Invalid bounding box: ({minx}, {miny}, {maxx}, {maxy}). "
            f"Requires: minx < maxx and miny < maxy"
        )

    # Set the cache directory
    if cache_dir is None:
        cache_dir = _get_default_cache_dir()

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Generate the cache file name
    bounds_hash = _compute_bounds_hash(bounds, network_type)
    cache_file = cache_path / f"network_{bounds_hash}.graphml"

    # Try to load from the cache
    if use_cache and cache_file.exists():
        logger.info(f"Loading the road network from cache: {cache_file}")
        try:
            G = ox.load_graphml(cache_file)
            logger.info(f"Cache load successful, nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")
            return G
        except Exception as e:
            logger.warning(f"Cache file is corrupted; re-downloading: {e}")

    # Download the road network
    logger.info(f"Downloading the road network from OSM, bounds: {bounds}, type: {network_type}")

    try:
        # OSMnx bbox parameter order: (left, bottom, right, top) = (minx, miny, maxx, maxy)
        G = ox.graph_from_bbox(
            bbox=bounds,  # (minx, miny, maxx, maxy)
            network_type=network_type,
            simplify=True,
            truncate_by_edge=True
        )
    except Exception as e:
        raise RuntimeError(
            f"Road network download failed: {str(e)}\n"
            f"Bounding box: {bounds}\n"
            f"Please check the network connection and whether the bounding box coordinates are valid."
        ) from e

    # OSMnx 2.x adds edge length attributes automatically; no manual call needed
    logger.info(f"Road network downloaded successfully, nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")

    # Save to the cache
    if use_cache:
        try:
            ox.save_graphml(G, cache_file)
            logger.info(f"Road network cached at: {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save the cache: {e}")

    return G


def get_network_connectivity_stats(G: nx.MultiDiGraph) -> Dict[str, Any]:
    """
    Get road-network connectivity statistics.

    Args:
        G: The NetworkX road-network graph

    Returns:
        Dict[str, Any]: Connectivity statistics, containing:
            - 'num_nodes': Total number of nodes
            - 'num_edges': Total number of edges
            - 'num_strongly_connected': Number of strongly connected components
            - 'num_weakly_connected': Number of weakly connected components
            - 'largest_component_ratio': Node share of the largest weakly connected component
            - 'is_strongly_connected': Whether the graph is strongly connected
    """
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    # Weakly connected components (ignoring edge direction)
    weakly_connected = list(nx.weakly_connected_components(G))
    num_weakly = len(weakly_connected)
    largest_component = max(weakly_connected, key=len)
    largest_ratio = len(largest_component) / num_nodes if num_nodes > 0 else 0

    # Strongly connected components
    strongly_connected = list(nx.strongly_connected_components(G))
    num_strongly = len(strongly_connected)

    stats = {
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'num_strongly_connected': num_strongly,
        'num_weakly_connected': num_weakly,
        'largest_component_ratio': largest_ratio,
        'is_strongly_connected': nx.is_strongly_connected(G)
    }

    logger.info(
        f"Road network connectivity: nodes={num_nodes}, edges={num_edges}, "
        f"weakly connected components={num_weakly}, largest component share={largest_ratio:.2%}"
    )

    return stats


# =============================================================================
# Node mapping functionality
# =============================================================================

def _find_nearest_node_simple(
    G: nx.MultiDiGraph,
    x: float,
    y: float
) -> Tuple[int, float]:
    """
    Simple nearest-neighbor node search (does not depend on OSMnx).

    Uses brute-force search over all nodes; used for graphs without a CRS
    (e.g. simulated graphs used in tests).

    Args:
        G: The road-network graph
        x: Query point longitude
        y: Query point latitude

    Returns:
        Tuple[int, float]: (nearest node ID, distance in meters)
    """
    min_dist = float('inf')
    nearest_node = None

    for node, data in G.nodes(data=True):
        node_x = data.get('x', 0)
        node_y = data.get('y', 0)
        # Use a simple Euclidean distance (an approximation under lat/lon coordinates)
        dist = _compute_euclidean_distance((x, y), (node_x, node_y))
        if dist < min_dist:
            min_dist = dist
            nearest_node = node

    return nearest_node, min_dist


def map_points_to_network(
    points_gdf: gpd.GeoDataFrame,
    G: nx.MultiDiGraph,
    id_column: str = 'id',
    max_distance: float = 2000.0
) -> pd.DataFrame:
    """
    Map geographic points to their nearest nodes in the road network.

    Args:
        points_gdf: A GeoDataFrame containing point geometries; must have a 'geometry' column
        G: The NetworkX road-network graph
        id_column: The point ID column name, default 'id'
        max_distance: Maximum allowed mapping distance (meters); a warning
            is logged when exceeded

    Returns:
        pd.DataFrame: Mapping result table, containing the following columns:
            - 'point_id': Original point ID
            - 'network_node': Mapped road-network node ID
            - 'mapping_distance': Mapping distance (meters)
            - 'node_x': Road-network node longitude
            - 'node_y': Road-network node latitude

    Note:
        The input GeoDataFrame should use the WGS84 CRS (EPSG:4326)
    """
    # Validate the input
    if 'geometry' not in points_gdf.columns:
        raise ValueError("The input GeoDataFrame must contain a 'geometry' column")

    if id_column not in points_gdf.columns:
        # If the specified ID column is absent, use the index
        points_gdf = points_gdf.copy()
        points_gdf[id_column] = points_gdf.index.astype(str)
        logger.info(f"Column '{id_column}' not found; using the index as the ID")

    # Extract point coordinates
    coords_x = []
    coords_y = []
    point_ids = []

    for idx, row in points_gdf.iterrows():
        geom = row['geometry']
        if hasattr(geom, 'centroid'):
            center = geom.centroid
        else:
            center = geom

        coords_x.append(center.x)
        coords_y.append(center.y)
        point_ids.append(row[id_column])

    logger.info(f"Mapping {len(point_ids)} points to the road network...")

    # Try to use OSMnx (suitable for real road networks)
    use_osmnx = 'crs' in G.graph if hasattr(G, 'graph') else False

    if use_osmnx:
        try:
            import osmnx as ox
            nearest_nodes = ox.nearest_nodes(G, coords_x, coords_y, return_dist=True)
            node_ids, distances = nearest_nodes
        except (KeyError, AttributeError) as e:
            logger.warning(f"OSMnx lookup failed; falling back to simple search: {e}")
            use_osmnx = False

    if not use_osmnx:
        # Fall back to a simple nearest-neighbor search
        node_ids = []
        distances = []
        for x, y in zip(coords_x, coords_y):
            node_id, dist = _find_nearest_node_simple(G, x, y)
            node_ids.append(node_id)
            distances.append(dist)

    # Get the node coordinates
    node_data = []
    warnings_count = 0

    for i, (point_id, node_id, dist) in enumerate(zip(point_ids, node_ids, distances)):
        node_x = G.nodes[node_id]['x']
        node_y = G.nodes[node_id]['y']

        # Distance warning
        if dist > max_distance:
            warnings_count += 1
            logger.warning(
                f"Point {point_id} has an excessive mapping distance: {dist:.1f}m > {max_distance}m"
            )

        node_data.append({
            'point_id': point_id,
            'network_node': node_id,
            'mapping_distance': dist,
            'node_x': node_x,
            'node_y': node_y
        })

    result_df = pd.DataFrame(node_data)

    # Summary statistics
    avg_dist = result_df['mapping_distance'].mean()
    max_dist = result_df['mapping_distance'].max()
    logger.info(
        f"Mapping complete: average distance={avg_dist:.1f}m, max distance={max_dist:.1f}m, "
        f"over-threshold warnings={warnings_count}"
    )

    return result_df


# =============================================================================
# Shortest-path computation
# =============================================================================

def _compute_distances_from_source(
    G: nx.MultiDiGraph,
    source_node: int,
    target_nodes: List[int],
    weight: str = 'length'
) -> Dict[int, float]:
    """
    Compute the shortest-path distance from a single source node to all
    target nodes.

    Computed via Dijkstra's algorithm; returns infinity for unreachable nodes.

    Args:
        G: The NetworkX road-network graph
        source_node: Source node ID
        target_nodes: List of target node IDs
        weight: Edge weight attribute name, default 'length' (meters)

    Returns:
        Dict[int, float]: {target_node_id: distance} dictionary
            - Reachable nodes return the actual distance (meters)
            - Unreachable nodes return float('inf')
    """
    # Compute the single-source shortest path from the source node
    try:
        # Check whether the source node is in the graph
        if source_node not in G:
            distances = {}
        else:
            distances = nx.single_source_dijkstra_path_length(
                G, source_node, weight=weight
            )
    except (nx.NetworkXError, nx.NodeNotFound):
        # The source node is not in the graph, or some other error occurred
        distances = {}

    # Extract the distances to the target nodes
    result = {}
    for target in target_nodes:
        if target in distances:
            result[target] = distances[target]
        else:
            result[target] = float('inf')

    return result


def _compute_distances_batch(args: Tuple) -> Tuple[int, Dict[int, float]]:
    """
    Worker function for batch computation (used with multiprocessing).

    Args:
        args: A (G_data, source_node, target_nodes, weight) tuple
            - G_data: Serialized graph data
            - source_node: Source node ID
            - target_nodes: List of target nodes
            - weight: Edge weight attribute name

    Returns:
        Tuple[int, Dict[int, float]]: (source_idx, distances_dict)
    """
    G_data, source_idx, source_node, target_nodes, weight = args

    # Deserialize the graph
    G = pickle.loads(G_data)

    distances = _compute_distances_from_source(G, source_node, target_nodes, weight)
    return source_idx, distances


# =============================================================================
# Sparse distance matrix computation
# =============================================================================

def compute_sparse_distance_matrix(
    agents_gdf: gpd.GeoDataFrame,
    targets_gdf: gpd.GeoDataFrame,
    G: nx.MultiDiGraph,
    k: int = 20,
    agent_id_col: str = 'agent_id',
    target_id_col: str = 'target_id',
    weight: str = 'length',
    use_multiprocess: bool = True,
    max_workers: Optional[int] = None,
    show_progress: bool = True
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """
    Compute a sparse road-network distance matrix.

    For each agent, only the K nearest targets are kept, to reduce memory usage.

    Args:
        agents_gdf: GeoDataFrame of agent points
        targets_gdf: GeoDataFrame of target points
        G: The NetworkX road-network graph
        k: Number of nearest targets kept per agent, default 20
        agent_id_col: Agent ID column name
        target_id_col: Target ID column name
        weight: Edge weight attribute name
        use_multiprocess: Whether to use multiprocessing, default True
        max_workers: Maximum number of worker processes
        show_progress: Whether to display progress

    Returns:
        Tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
            - distance_matrix: Distance matrix of shape [Num_Agents, K] (meters)
            - target_indices: Target index matrix of shape [Num_Agents, K]
            - agent_mapping: DataFrame mapping agent ID to matrix row index
            - target_mapping: DataFrame mapping target ID to global index

    Note:
        - If an agent has fewer than K reachable targets, the unreachable
          slots are filled with inf
        - The K distances in each row are sorted in ascending order
    """
    # Map the points to the road network
    logger.info("Mapping agent points to the road network...")
    agent_mapping_df = map_points_to_network(
        agents_gdf, G, id_column=agent_id_col
    )

    logger.info("Mapping target points to the road network...")
    target_mapping_df = map_points_to_network(
        targets_gdf, G, id_column=target_id_col
    )

    # Get the node lists
    agent_nodes = agent_mapping_df['network_node'].tolist()
    target_nodes = target_mapping_df['network_node'].tolist()
    target_nodes_set = set(target_nodes)

    num_agents = len(agent_nodes)
    num_targets = len(target_nodes)
    k_actual = min(k, num_targets)

    logger.info(
        f"Computing the distance matrix: {num_agents} agents x {num_targets} targets, K={k_actual}"
    )

    # Initialize the result arrays
    distance_matrix = np.full((num_agents, k_actual), np.inf, dtype=np.float64)
    target_indices = np.full((num_agents, k_actual), -1, dtype=np.int64)

    # Select the computation strategy
    if num_targets < num_agents:
        # Reverse mode: run Dijkstra outward from each target (call count = num_targets)
        logger.info(
            f"Using the reverse-Dijkstra strategy "
            f"({num_targets} calls vs. {num_agents} originally, "
            f"speedup ~{num_agents // num_targets}x)"
        )
        _compute_from_targets(
            G, agent_nodes, target_nodes, weight,
            distance_matrix, target_indices, k_actual,
            show_progress
        )
    elif use_multiprocess and num_agents > 100 and platform.system() != 'Windows':
        # Multiprocessing mode (multiprocessing can be problematic on Windows)
        _compute_parallel(
            G, agent_nodes, target_nodes, weight,
            distance_matrix, target_indices, k_actual,
            max_workers, show_progress
        )
    else:
        # Single-process mode
        _compute_sequential(
            G, agent_nodes, target_nodes, weight,
            distance_matrix, target_indices, k_actual,
            show_progress
        )

    # Create the mapping tables
    agent_mapping = pd.DataFrame({
        'agent_id': agent_mapping_df['point_id'].tolist(),
        'matrix_row': list(range(num_agents)),
        'network_node': agent_nodes,
        'mapping_distance': agent_mapping_df['mapping_distance'].tolist()
    })

    target_mapping = pd.DataFrame({
        'target_id': target_mapping_df['point_id'].tolist(),
        'global_index': list(range(num_targets)),
        'network_node': target_nodes,
        'mapping_distance': target_mapping_df['mapping_distance'].tolist()
    })

    return distance_matrix, target_indices, agent_mapping, target_mapping


def _compute_sequential(
    G: nx.MultiDiGraph,
    agent_nodes: List[int],
    target_nodes: List[int],
    weight: str,
    distance_matrix: np.ndarray,
    target_indices: np.ndarray,
    k: int,
    show_progress: bool
) -> None:
    """Compute the distance matrix in a single process."""
    num_agents = len(agent_nodes)
    progress_interval = max(1, num_agents // 10)

    for i, agent_node in enumerate(agent_nodes):
        # Compute the distances to all targets
        distances = _compute_distances_from_source(
            G, agent_node, target_nodes, weight
        )

        # Extract and sort the distance values
        dist_array = np.array([distances.get(t, np.inf) for t in target_nodes])

        # Get the K nearest
        sorted_indices = np.argsort(dist_array)[:k]
        sorted_distances = dist_array[sorted_indices]

        distance_matrix[i] = sorted_distances
        target_indices[i] = sorted_indices

        # Progress display
        if show_progress and (i + 1) % progress_interval == 0:
            logger.info(f"Progress: {i + 1}/{num_agents} ({(i + 1) * 100 // num_agents}%)")

    if show_progress:
        logger.info(f"Computation complete: {num_agents}/{num_agents} (100%)")


def _compute_from_targets(
    G: nx.MultiDiGraph,
    agent_nodes: List[int],
    target_nodes: List[int],
    weight: str,
    distance_matrix: np.ndarray,
    target_indices: np.ndarray,
    k: int,
    show_progress: bool,
) -> None:
    """Reverse-Dijkstra strategy: compute the distance matrix starting from
    each target.

    When the number of targets is much smaller than the number of agents,
    running Dijkstra outward from each target substantially reduces the
    number of calls. G.reverse() is used to guarantee correctness for
    directed graphs (one-way streets):
    dist_G_rev(target -> agent) == dist_G(agent -> target).
    """
    num_agents = len(agent_nodes)
    num_targets = len(target_nodes)

    # Reverse the graph (an O(1) view, no data copy)
    G_rev = G.reverse(copy=False)

    # Build an agent node lookup table (multiple agents may map to the same road-network node)
    agent_node_to_indices: Dict[int, List[int]] = {}
    for i, node in enumerate(agent_nodes):
        agent_node_to_indices.setdefault(node, []).append(i)

    # Full distance matrix (num_agents, num_targets)
    full_distances = np.full((num_agents, num_targets), np.inf, dtype=np.float64)

    progress_interval = max(1, num_targets // 10)

    for j, target_node in enumerate(target_nodes):
        try:
            if target_node in G_rev:
                all_distances = dict(
                    nx.single_source_dijkstra_path_length(
                        G_rev, target_node, weight=weight
                    )
                )
            else:
                all_distances = {}
        except (nx.NetworkXError, nx.NodeNotFound):
            all_distances = {}

        # Only extract the distances for agent nodes
        for agent_node, indices in agent_node_to_indices.items():
            dist = all_distances.get(agent_node, np.inf)
            for idx in indices:
                full_distances[idx, j] = dist

        if show_progress and (j + 1) % progress_interval == 0:
            logger.info(
                f"Reverse-Dijkstra progress: {j + 1}/{num_targets} "
                f"({(j + 1) * 100 // num_targets}%)"
            )

    # Extract the top-k nearest targets
    for i in range(num_agents):
        sorted_idx = np.argsort(full_distances[i])[:k]
        distance_matrix[i] = full_distances[i, sorted_idx]
        target_indices[i] = sorted_idx

    if show_progress:
        logger.info(f"Reverse-Dijkstra complete: {num_targets}/{num_targets} (100%)")


def _compute_parallel(
    G: nx.MultiDiGraph,
    agent_nodes: List[int],
    target_nodes: List[int],
    weight: str,
    distance_matrix: np.ndarray,
    target_indices: np.ndarray,
    k: int,
    max_workers: Optional[int],
    show_progress: bool
) -> None:
    """Compute the distance matrix in parallel using multiprocessing."""
    num_agents = len(agent_nodes)

    if max_workers is None:
        max_workers = _get_default_max_workers()

    # Serialize the graph data
    G_data = pickle.dumps(G)

    # Prepare the task arguments
    tasks = [
        (G_data, i, agent_node, target_nodes, weight)
        for i, agent_node in enumerate(agent_nodes)
    ]

    logger.info(f"Computing in parallel using {max_workers} processes...")

    completed = 0
    progress_interval = max(1, num_agents // 10)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_compute_distances_batch, task): task[1] for task in tasks}

        for future in as_completed(futures):
            source_idx, distances = future.result()

            # Extract and sort the distance values
            dist_array = np.array([distances.get(t, np.inf) for t in target_nodes])
            sorted_indices = np.argsort(dist_array)[:k]
            sorted_distances = dist_array[sorted_indices]

            distance_matrix[source_idx] = sorted_distances
            target_indices[source_idx] = sorted_indices

            completed += 1
            if show_progress and completed % progress_interval == 0:
                logger.info(f"Progress: {completed}/{num_agents} ({completed * 100 // num_agents}%)")

    if show_progress:
        logger.info(f"Computation complete: {num_agents}/{num_agents} (100%)")


# =============================================================================
# Result-saving functionality
# =============================================================================

def save_distance_results(
    distance_matrix: np.ndarray,
    target_indices: np.ndarray,
    agent_mapping: pd.DataFrame,
    target_mapping: pd.DataFrame,
    output_dir: str,
    prefix: str = ''
) -> Dict[str, str]:
    """
    Save the distance computation results to files.

    Args:
        distance_matrix: Distance matrix [Num_Agents, K]
        target_indices: Target index matrix [Num_Agents, K]
        agent_mapping: Agent mapping table
        target_mapping: Target mapping table
        output_dir: Output directory path
        prefix: File name prefix (optional)

    Returns:
        Dict[str, str]: Dictionary of saved file paths

    Files saved:
        - {prefix}network_distance_matrix.npy: Distance matrix
        - {prefix}target_indices.npy: Target indices
        - {prefix}agent_mapping.csv: Agent mapping table
        - {prefix}target_mapping.csv: Target mapping table
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Build the file names
    prefix_str = f"{prefix}_" if prefix else ""

    paths = {}

    # Save the distance matrix
    dist_file = output_path / f"{prefix_str}network_distance_matrix.npy"
    np.save(dist_file, distance_matrix)
    paths['distance_matrix'] = str(dist_file)
    logger.info(f"Distance matrix saved: {dist_file}")

    # Save the target indices
    idx_file = output_path / f"{prefix_str}target_indices.npy"
    np.save(idx_file, target_indices)
    paths['target_indices'] = str(idx_file)
    logger.info(f"Target indices saved: {idx_file}")

    # Save the agent mapping table
    agent_file = output_path / f"{prefix_str}agent_mapping.csv"
    agent_mapping.to_csv(agent_file, index=False, encoding='utf-8')
    paths['agent_mapping'] = str(agent_file)
    logger.info(f"Agent mapping table saved: {agent_file}")

    # Save the target mapping table
    target_file = output_path / f"{prefix_str}target_mapping.csv"
    target_mapping.to_csv(target_file, index=False, encoding='utf-8')
    paths['target_mapping'] = str(target_file)
    logger.info(f"Target mapping table saved: {target_file}")

    return paths


def load_distance_results(
    output_dir: str,
    prefix: str = ''
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """
    Load the distance computation results.

    Args:
        output_dir: Results directory path
        prefix: File name prefix

    Returns:
        Tuple: (distance_matrix, target_indices, agent_mapping, target_mapping)
    """
    output_path = Path(output_dir)
    prefix_str = f"{prefix}_" if prefix else ""

    distance_matrix = np.load(output_path / f"{prefix_str}network_distance_matrix.npy")
    target_indices = np.load(output_path / f"{prefix_str}target_indices.npy")
    agent_mapping = pd.read_csv(output_path / f"{prefix_str}agent_mapping.csv")
    target_mapping = pd.read_csv(output_path / f"{prefix_str}target_mapping.csv")

    logger.info(
        f"Loaded results: distance matrix shape={distance_matrix.shape}, "
        f"agents={len(agent_mapping)}, targets={len(target_mapping)}"
    )

    return distance_matrix, target_indices, agent_mapping, target_mapping


# =============================================================================
# Validation functionality
# =============================================================================

def validate_connectivity(
    distance_matrix: np.ndarray,
    threshold: float = 0.01
) -> Dict[str, Any]:
    """
    Validate the connectivity of the distance matrix.

    Checks the proportion of unreachable node pairs (distance = infinity).

    Args:
        distance_matrix: Distance matrix [Num_Agents, K]
        threshold: Unreachable-ratio threshold, default 0.01 (1%)

    Returns:
        Dict[str, Any]: Validation result, containing:
            - 'total_pairs': Total number of node pairs
            - 'unreachable_pairs': Number of unreachable node pairs
            - 'unreachable_ratio': Unreachable ratio
            - 'is_valid': Whether validation passed (unreachable ratio < threshold)
            - 'unreachable_agents': List of agent indices with unreachable targets
    """
    total_pairs = distance_matrix.size
    unreachable_mask = np.isinf(distance_matrix)
    unreachable_pairs = np.sum(unreachable_mask)
    unreachable_ratio = unreachable_pairs / total_pairs if total_pairs > 0 else 0

    # Find agents with unreachable targets
    unreachable_agents = np.where(np.any(unreachable_mask, axis=1))[0].tolist()

    result = {
        'total_pairs': total_pairs,
        'unreachable_pairs': int(unreachable_pairs),
        'unreachable_ratio': unreachable_ratio,
        'is_valid': unreachable_ratio < threshold,
        'unreachable_agents': unreachable_agents
    }

    if result['is_valid']:
        logger.info(f"Connectivity validation passed: unreachable ratio={unreachable_ratio:.4%}")
    else:
        logger.warning(
            f"Connectivity validation failed: unreachable ratio={unreachable_ratio:.4%} > {threshold:.4%}"
        )

    return result


def validate_distance_logic(
    agents_gdf: gpd.GeoDataFrame,
    targets_gdf: gpd.GeoDataFrame,
    distance_matrix: np.ndarray,
    target_indices: np.ndarray,
    agent_id_col: str = 'agent_id',
    target_id_col: str = 'target_id',
    num_samples: int = 100,
    max_ratio: float = 3.0
) -> Dict[str, Any]:
    """
    Validate the logical soundness of the road-network distances.

    Assertion: road-network distance >= Euclidean distance (always holds).

    Args:
        agents_gdf: GeoDataFrame of agent points
        targets_gdf: GeoDataFrame of target points
        distance_matrix: Distance matrix
        target_indices: Target index matrix
        agent_id_col: Agent ID column name
        target_id_col: Target ID column name
        num_samples: Number of random samples
        max_ratio: Maximum expected ratio of road-network distance / Euclidean distance

    Returns:
        Dict[str, Any]: Validation result, containing:
            - 'num_samples': Number of samples
            - 'all_valid': Whether all samples satisfy road-network distance >= Euclidean distance
            - 'ratio_stats': Ratio statistics (min, max, mean, median)
            - 'suspicious_pairs': List of samples with an excessively large ratio
    """
    # CRS-aware: project both to a metric CRS, then compute Euclidean distance directly with numpy
    agents_proj = agents_gdf.copy()
    targets_proj = targets_gdf.copy()

    if agents_proj.crs is not None and agents_proj.crs.is_geographic:
        agents_proj = agents_proj.to_crs("EPSG:27700")
    if targets_proj.crs is not None and targets_proj.crs.is_geographic:
        targets_proj = targets_proj.to_crs("EPSG:27700")

    agents_coords = []
    for idx, row in agents_proj.iterrows():
        geom = row['geometry']
        center = geom.centroid if hasattr(geom, 'centroid') else geom
        agents_coords.append((center.x, center.y))

    targets_coords = []
    for idx, row in targets_proj.iterrows():
        geom = row['geometry']
        center = geom.centroid if hasattr(geom, 'centroid') else geom
        targets_coords.append((center.x, center.y))

    agents_coords_arr = np.array(agents_coords)
    targets_coords_arr = np.array(targets_coords)

    # Random sampling
    num_agents = len(agents_coords)
    sample_size = min(num_samples, num_agents * distance_matrix.shape[1])

    np.random.seed(42)
    sampled_agents = np.random.randint(0, num_agents, sample_size)

    ratios = []
    all_valid = True
    suspicious_pairs = []

    for agent_idx in sampled_agents:
        # Randomly select one target for this agent
        k_idx = np.random.randint(0, distance_matrix.shape[1])
        target_global_idx = target_indices[agent_idx, k_idx]

        if target_global_idx < 0 or target_global_idx >= len(targets_coords):
            continue

        network_dist = distance_matrix[agent_idx, k_idx]
        if np.isinf(network_dist):
            continue

        # Compute the Euclidean distance (coordinates are already metric after projection, use numpy directly)
        diff = agents_coords_arr[agent_idx] - targets_coords_arr[target_global_idx]
        euclidean_dist = float(np.sqrt(np.sum(diff ** 2)))

        # Logical check: road-network distance should be >= Euclidean distance
        if network_dist < euclidean_dist * 0.99:  # Allow a 1% margin of error
            all_valid = False
            logger.warning(
                f"Logic error: agent {agent_idx} -> target {target_global_idx}, "
                f"road-network distance {network_dist:.1f}m < Euclidean distance {euclidean_dist:.1f}m"
            )

        # Compute the ratio
        if euclidean_dist > 0:
            ratio = network_dist / euclidean_dist
            ratios.append(ratio)

            if ratio > max_ratio:
                suspicious_pairs.append({
                    'agent_idx': int(agent_idx),
                    'target_idx': int(target_global_idx),
                    'network_distance': float(network_dist),
                    'euclidean_distance': float(euclidean_dist),
                    'ratio': float(ratio)
                })

    # Statistics
    ratios = np.array(ratios)
    ratio_stats = {
        'min': float(np.min(ratios)) if len(ratios) > 0 else None,
        'max': float(np.max(ratios)) if len(ratios) > 0 else None,
        'mean': float(np.mean(ratios)) if len(ratios) > 0 else None,
        'median': float(np.median(ratios)) if len(ratios) > 0 else None
    }

    result = {
        'num_samples': len(ratios),
        'all_valid': all_valid,
        'ratio_stats': ratio_stats,
        'suspicious_pairs': suspicious_pairs[:10]  # Keep only the first 10
    }

    if ratio_stats['median']:
        logger.info(
            f"Logic validation: samples={len(ratios)}, "
            f"road-network/Euclidean ratio median={ratio_stats['median']:.2f}"
        )

    return result


# =============================================================================
# Convenience function
# =============================================================================

def compute_and_save_distances(
    agents_gdf: gpd.GeoDataFrame,
    targets_gdf: gpd.GeoDataFrame,
    bounds: Tuple[float, float, float, float],
    output_dir: str,
    config: Optional[NetworkDistanceConfig] = None,
    agent_id_col: str = 'agent_id',
    target_id_col: str = 'target_id'
) -> Dict[str, Any]:
    """
    One-stop convenience function to compute and save road-network distances.

    Combines road-network download, distance computation, result saving,
    and validation into a single end-to-end workflow.

    Args:
        agents_gdf: GeoDataFrame of agent points
        targets_gdf: GeoDataFrame of target points
        bounds: (minx, miny, maxx, maxy) bounding box
        output_dir: Output directory
        config: Configuration object; the default configuration is used if None
        agent_id_col: Agent ID column name
        target_id_col: Target ID column name

    Returns:
        Dict[str, Any]: Summary of the computation results, containing:
            - 'saved_files': Paths of the saved files
            - 'connectivity': Connectivity validation result
            - 'logic': Logic validation result
            - 'network_stats': Road-network statistics
    """
    if config is None:
        config = NetworkDistanceConfig()

    # 1. Download the road network
    logger.info("Step 1/4: downloading the road network...")
    G = download_road_network(
        bounds=bounds,
        network_type=config.network_type,
        cache_dir=config.cache_dir,
        use_cache=config.use_cache
    )

    # Road-network statistics
    network_stats = get_network_connectivity_stats(G)

    # 2. Compute the distance matrix
    logger.info("Step 2/4: computing the distance matrix...")
    distance_matrix, target_indices, agent_mapping, target_mapping = \
        compute_sparse_distance_matrix(
            agents_gdf=agents_gdf,
            targets_gdf=targets_gdf,
            G=G,
            k=config.k_nearest,
            agent_id_col=agent_id_col,
            target_id_col=target_id_col,
            weight=config.weight,
            use_multiprocess=config.use_multiprocess,
            max_workers=config.max_workers,
            show_progress=True
        )

    # 3. Save the results
    logger.info("Step 3/4: saving the results...")
    saved_files = save_distance_results(
        distance_matrix=distance_matrix,
        target_indices=target_indices,
        agent_mapping=agent_mapping,
        target_mapping=target_mapping,
        output_dir=output_dir
    )

    # 4. Validate
    logger.info("Step 4/4: validating the results...")
    connectivity = validate_connectivity(distance_matrix)
    logic = validate_distance_logic(
        agents_gdf=agents_gdf,
        targets_gdf=targets_gdf,
        distance_matrix=distance_matrix,
        target_indices=target_indices,
        agent_id_col=agent_id_col,
        target_id_col=target_id_col
    )

    result = {
        'saved_files': saved_files,
        'connectivity': connectivity,
        'logic': logic,
        'network_stats': network_stats,
        'matrix_shape': distance_matrix.shape
    }

    logger.info("Road-network distance computation complete!")
    return result
