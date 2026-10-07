from __future__ import annotations

from typing import Any

import numpy as np

from ..allocator.civd import CIVDAllocator, hdbscan_station_clusters


def materialize_civd(grid, stations, *, working_crs: str, capacity_column: str | None = None) -> dict[str, Any]:
    clusters = hdbscan_station_clusters(stations, working_crs=working_crs)
    targets = stations.copy().reset_index(drop=True)
    targets["cluster_label"] = clusters.labels
    allocator = CIVDAllocator(
        {
            "working_crs": working_crs,
            "cluster_label_column": "cluster_label",
            "capacity_column": capacity_column,
            "method": "civd",
        }
    )
    result = allocator.allocate(grid, targets)
    return {
        "assignment": result.assignment,
        "grid_cluster": result.assignment,
        "station_cluster": clusters.labels,
        "raw_labels": clusters.raw_labels,
        "probabilities": clusters.probabilities,
        "n_clusters": clusters.n_clusters,
        "n_noise": clusters.n_noise,
        "metadata": result.metadata,
    }

