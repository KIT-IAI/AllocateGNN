"""
CIVD (Capacity-Influenced Voronoi Diagram) allocator — Pyomo optimization implementation

Uses Pyomo LP to solve the optimal grid-to-cluster assignment:
  max  Σ x[q,c] * influence[q,c]
  s.t. Σ_c x[q,c] = 1  ∀q   (each grid point is assigned to exactly one cluster)
       0 ≤ x[q,c] ≤ 1

Influence computation:
  - CIVD: influence[q][c] = max_{i ∈ c}( w_i / d(q, p_i) )
  - IVD:  influence[q][c] = Σ_{i ∈ c}( w_i / d(q, p_i) )

Reference: ClusterBasedVoronoi/voronoi/prepare_pyomo_parameter.py + pyomo_based_voronoi.py
"""
import logging
from typing import Optional

import numpy as np
import geopandas as gpd
import pyomo.environ as pyo
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist

from .base import BaseAllocator, AllocationResult
from .registry import allocator_registry

logger = logging.getLogger(__name__)


# ─── influence computation ─────────────────────────────────────────────────

def _compute_civd_row(q_idx: int, unique_labels, cluster_labels,
                      weights, distances):
    """CIVD influence for a single grid point: each cluster takes max(w_i / d_qi)"""
    row = {}
    for label in unique_labels:
        mask = cluster_labels == label
        row[label] = np.max(weights[mask] / distances[q_idx, mask])
    return q_idx, row


def _compute_ivd_row(q_idx: int, unique_labels, cluster_labels,
                     weights, distances):
    """IVD influence for a single grid point: each cluster takes Σ(w_i / d_qi)"""
    row = {}
    for label in unique_labels:
        mask = cluster_labels == label
        row[label] = np.sum(weights[mask] / distances[q_idx, mask])
    return q_idx, row


def compute_influence_matrix(
    grid_coords: np.ndarray,
    target_coords: np.ndarray,
    cluster_labels: np.ndarray,
    weights: np.ndarray,
    method: str = "civd",
    n_jobs: int = -1,
    distance_matrix: np.ndarray = None,
) -> dict:
    """
    Compute the influence matrix.

    Args:
        grid_coords: (N_grid, 2) grid point coordinates (after projection)
        target_coords: (N_target, 2) target point coordinates
        cluster_labels: (N_target,) cluster label for each target point
        weights: (N_target,) weight for each target point
        method: "civd" or "ivd"
        n_jobs: number of parallel processes
        distance_matrix: (N_grid, N_target) precomputed distance matrix (optional; if provided, skips cdist)

    Returns:
        {q_idx: {label: influence_value, ...}, ...}
    """
    # Distance matrix (N_grid, N_target)
    if distance_matrix is not None:
        distances = distance_matrix.copy()
    else:
        distances = cdist(grid_coords, target_coords)
    # Avoid division by zero
    distances = np.maximum(distances, 1e-10)

    unique_labels = np.unique(cluster_labels)

    if method == "civd":
        func = _compute_civd_row
    elif method == "ivd":
        func = _compute_ivd_row
    else:
        raise ValueError(f"Unknown influence method: {method}, supported values are 'civd' or 'ivd'")

    results = Parallel(n_jobs=n_jobs)(
        delayed(func)(q, unique_labels, cluster_labels, weights, distances)
        for q in range(len(grid_coords))
    )

    return {q_idx: row for q_idx, row in results}


# ─── Pyomo model construction and solving ──────────────────────────────────

def build_pyomo_model(
    influence_matrix: dict,
    cluster_labels_unique: np.ndarray,
    n_grid: int,
    penalty_weight: Optional[float] = None,
) -> pyo.ConcreteModel:
    """
    Build the Pyomo LP model.

    Objective: max Σ x[q,c] * influence[q][c]
    Constraint: Σ_c x[q,c] = 1  ∀q
    """
    model = pyo.ConcreteModel()

    model.Q = pyo.RangeSet(n_grid)
    model.C = pyo.Set(initialize=list(cluster_labels_unique))

    model.x = pyo.Var(model.Q, model.C, within=pyo.UnitInterval)

    # Constraint: the assignment sum for each grid point = 1
    def one_per_point_rule(m, q):
        return sum(m.x[q, c] for c in m.C) == 1

    model.one_per_point = pyo.Constraint(model.Q, rule=one_per_point_rule)

    # Objective: maximize total influence
    def objective_rule(m):
        obj = sum(
            m.x[q, c] * influence_matrix[q - 1][c]
            for q in m.Q for c in m.C
        )
        if penalty_weight is not None:
            penalty = penalty_weight * sum(
                (m.x[q, c] - 0.5) ** 2
                for q in m.Q for c in m.C
            )
            return obj + penalty
        return obj

    model.obj = pyo.Objective(rule=objective_rule, sense=pyo.maximize)

    return model


def solve_model(
    model: pyo.ConcreteModel,
    solver_name: str = "scip",
) -> pyo.ConcreteModel:
    """Solve the Pyomo model"""
    solver = pyo.SolverFactory(solver_name)
    if solver_name == "scip":
        solver.options["threads"] = 4
    result = solver.solve(model, tee=False)

    if result.solver.termination_condition != pyo.TerminationCondition.optimal:
        logger.warning(
            f"Solver did not reach optimality: {result.solver.termination_condition}"
        )

    return model


def extract_assignment(model: pyo.ConcreteModel, n_grid: int) -> np.ndarray:
    """Extract the assignment result from the Pyomo solution (each grid point takes the cluster with the largest x value)"""
    assignment = np.empty(n_grid, dtype=int)
    for q in model.Q:
        best_c = max(model.C, key=lambda c: pyo.value(model.x[q, c]))
        assignment[q - 1] = best_c  # Pyomo RangeSet is 1-indexed
    return assignment


# ─── CIVDAllocator main class ──────────────────────────────────────────────

@allocator_registry.register(
    "civd",
    description="Capacity-Influenced Voronoi allocation (Pyomo LP optimization)",
)
class CIVDAllocator(BaseAllocator):
    """
    Capacity-Influenced Voronoi allocator (Pyomo optimization implementation).

    Solves the optimal grid-to-cluster assignment by maximizing the sum of
    influence via Pyomo LP, replacing the original argmin(d/capacity^alpha)
    heuristic method.

    Configuration parameters:
        solver: Pyomo solver name (default "scip")
        method: influence computation method "civd" | "ivd" (default "civd")
        penalty_weight: optional penalty term weight (encourages binarization, default None)
        n_jobs: number of processes for parallel influence computation (default -1 = all cores)
        capacity_column: target capacity column name (default "capacity")
        cluster_label_column: cluster label column name (default "cluster_label")
        working_crs: projected CRS used for computation (default "EPSG:3857")
        output_crs: output CRS (default "EPSG:4326")
    """

    def allocate(
        self,
        grid_gdf: gpd.GeoDataFrame,
        target_gdf: gpd.GeoDataFrame,
        weights: Optional[np.ndarray] = None,
        source_gdf: Optional[gpd.GeoDataFrame] = None,
        **kwargs,
    ) -> AllocationResult:
        # ── Read configuration ──
        solver_name = self.config.get("solver", "scip")
        method = self.config.get("method", "civd")
        penalty_weight = self.config.get("penalty_weight", None)
        n_jobs = self.config.get("n_jobs", -1)
        capacity_col = self.config.get("capacity_column", None)
        cluster_col = self.config.get("cluster_label_column", "cluster_label")
        working_crs = self.config.get("working_crs", "EPSG:3857")
        output_crs = self.config.get("output_crs", "EPSG:4326")

        # ── Project to a metric CRS ──
        grid_proj = grid_gdf.to_crs(working_crs)
        target_proj = target_gdf.to_crs(working_crs)

        grid_coords = np.column_stack([
            grid_proj.geometry.x.values,
            grid_proj.geometry.y.values,
        ])
        target_coords = np.column_stack([
            target_proj.geometry.x.values,
            target_proj.geometry.y.values,
        ])

        # ── Cluster labels ──
        if cluster_col in target_gdf.columns:
            cluster_labels = target_gdf[cluster_col].values
        else:
            # When there are no cluster labels, each target forms its own cluster (equivalent to point-wise assignment)
            cluster_labels = np.arange(len(target_gdf))

        # ── Target weights (capacity) ──
        if weights is not None:
            # Externally provided weights (e.g. GPM weights computed by a Weighter)
            # weights is at the grid level, but influence needs target-level weights
            # If len(weights) == number of targets, use directly; otherwise use the capacity column
            if len(weights) == len(target_gdf):
                point_weights = weights.astype(float)
            else:
                logger.info("weights length does not match target count, falling back to capacity_column")
                point_weights = self._get_capacity_weights(target_gdf, capacity_col)
        else:
            point_weights = self._get_capacity_weights(target_gdf, capacity_col)

        # ── Compute the influence matrix ──
        precomputed_distances = kwargs.get('distance_matrix', None)
        logger.info(
            f"Computing influence: method={method}, "
            f"grid={len(grid_coords)}, target={len(target_coords)}"
            f"{', using precomputed distance matrix' if precomputed_distances is not None else ''}"
        )
        influence_matrix = compute_influence_matrix(
            grid_coords, target_coords, cluster_labels, point_weights,
            method=method, n_jobs=n_jobs,
            distance_matrix=precomputed_distances,
        )

        # ── Build + solve the Pyomo model ──
        unique_labels = np.unique(cluster_labels)
        logger.info(
            f"Building Pyomo model: {len(grid_coords)} points x {len(unique_labels)} clusters"
        )
        model = build_pyomo_model(
            influence_matrix, unique_labels, len(grid_coords),
            penalty_weight=penalty_weight,
        )

        logger.info(f"Solving (solver={solver_name})...")
        model = solve_model(model, solver_name=solver_name)

        # ── Extract the assignment result ──
        assignment = extract_assignment(model, len(grid_coords))

        # ── Generate Voronoi polygons ──
        grid_result = grid_proj.copy()
        grid_result["assigned_target"] = assignment
        voronoi_gdf = grid_result.dissolve(by="assigned_target")
        voronoi_gdf["geometry"] = voronoi_gdf.convex_hull
        voronoi_gdf = voronoi_gdf.to_crs(output_crs).reset_index()

        return AllocationResult(
            assignment=assignment,
            assignment_column_name="assigned_target",
            voronoi_gdf=voronoi_gdf[["assigned_target", "geometry"]],
            assignment_gdf=grid_result.to_crs(output_crs),
            confidence=None,
            metadata={
                "method": f"civd_pyomo_{method}",
                "solver": solver_name,
                "penalty_weight": penalty_weight,
                "n_grid": len(grid_gdf),
                "n_target": len(target_gdf),
                "n_clusters": len(unique_labels),
                "n_voronoi_regions": len(voronoi_gdf),
            },
        )

    @staticmethod
    def _get_capacity_weights(
        target_gdf: gpd.GeoDataFrame, capacity_col: Optional[str]
    ) -> np.ndarray:
        """Get capacity weights from target_gdf; uses uniform weights if capacity_col is None or missing"""
        if capacity_col is not None and capacity_col in target_gdf.columns:
            w = target_gdf[capacity_col].values.astype(float)
            return np.maximum(w, 1e-10)
        return np.ones(len(target_gdf))

    @property
    def requires_weights(self) -> bool:
        return False  # weights are optional
