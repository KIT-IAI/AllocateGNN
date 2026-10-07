"""Pure NumPy/SciPy regular-grid adjacency construction."""

from __future__ import annotations

import numpy as np
from scipy.spatial import KDTree


def build_grid_adjacency_indices(
    coords: np.ndarray,
    *,
    mode: str = "neumann",
    step_size: float | None = None,
    tolerance: float = 0.1,
) -> np.ndarray:
    """Return a symmetric ``(2, E)`` int64 edge-index array."""

    if mode not in {"neumann", "moore"}:
        raise ValueError(
            f"unsupported grid-neighbourhood mode {mode!r}; "
            "expected 'neumann' or 'moore'"
        )
    points = np.asarray(coords, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("grid coordinates must have shape (N, 2)")
    if len(points) < 2:
        return np.empty((2, 0), dtype=np.int64)
    tree = KDTree(points)
    if step_size is None:
        nearest, _ = tree.query(points, k=2)
        step_size = float(np.median(nearest[:, 1]))
    step = float(step_size)
    if not np.isfinite(step) or step <= 0:
        raise ValueError("grid step_size must be finite and positive")
    search_radius = step * (
        1.0 + tolerance if mode == "neumann" else np.sqrt(2.0) * (1.0 + tolerance)
    )
    orthogonal = (step * (1.0 - tolerance), step * (1.0 + tolerance))
    diagonal_step = step * np.sqrt(2.0)
    diagonal = (
        diagonal_step * (1.0 - tolerance),
        diagonal_step * (1.0 + tolerance),
    )
    sources: list[int] = []
    destinations: list[int] = []
    for left, neighbours in enumerate(tree.query_ball_point(points, r=search_radius)):
        for right in neighbours:
            if left >= right:
                continue
            distance = float(np.linalg.norm(points[left] - points[right]))
            orthogonal_match = orthogonal[0] <= distance <= orthogonal[1]
            diagonal_match = diagonal[0] <= distance <= diagonal[1]
            if orthogonal_match or (mode == "moore" and diagonal_match):
                sources.append(left)
                destinations.append(right)
    if not sources:
        return np.empty((2, 0), dtype=np.int64)
    source = np.asarray(sources, dtype=np.int64)
    destination = np.asarray(destinations, dtype=np.int64)
    return np.stack(
        [
            np.concatenate([source, destination]),
            np.concatenate([destination, source]),
        ],
        axis=0,
    )


__all__ = ["build_grid_adjacency_indices"]
