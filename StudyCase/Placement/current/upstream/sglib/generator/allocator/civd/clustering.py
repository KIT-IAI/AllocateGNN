"""CIVD 论文协议使用的确定性 HDBSCAN 站点聚类。"""

from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import hdbscan
import numpy as np


class ClusteringError(ValueError):
    """站点坐标或聚类结果不满足 CIVD 合同。"""


@dataclass(frozen=True)
class HdbscanClusters:
    labels: np.ndarray
    raw_labels: np.ndarray
    probabilities: np.ndarray
    n_clusters: int
    n_noise: int


def hdbscan_station_clusters(
    stations: gpd.GeoDataFrame,
    *,
    min_cluster_size: int = 2,
    working_crs: str = "EPSG:3857",
) -> HdbscanClusters:
    """复现论文 CIVD 的站点聚类，并把每个 noise 点变成独立簇。

    原实现使用 ``hdbscan.HDBSCAN(min_cluster_size=2, min_samples=None,
    cluster_selection_method='eom', core_dist_n_jobs=1)``。HDBSCAN 的正常簇标签
    保留顺序；原始 ``-1`` noise 按站点行序依次获得独立标签，最后整体重编码为
    从零开始的连续整数，保证解析 CIVD 的列序与簇标签一致。
    """

    if (
        not isinstance(stations, gpd.GeoDataFrame)
        or stations.empty
        or stations.crs is None
        or min_cluster_size < 2
    ):
        raise ClusteringError("CIVD clustering requires non-empty georeferenced stations")
    projected = stations.to_crs(working_crs)
    coordinates = np.column_stack(
        (projected.geometry.x.to_numpy(), projected.geometry.y.to_numpy())
    )
    if coordinates.shape != (len(stations), 2) or not np.isfinite(coordinates).all():
        raise ClusteringError("CIVD station coordinates must be finite (N,2)")
    if len(stations) < min_cluster_size:
        raw = np.full(len(stations), -1, dtype=np.int64)
        probabilities = np.zeros(len(stations), dtype=float)
    else:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=int(min_cluster_size),
            min_samples=None,
            metric="euclidean",
            cluster_selection_method="eom",
            core_dist_n_jobs=1,
        )
        raw = np.asarray(clusterer.fit_predict(coordinates), dtype=np.int64)
        probabilities = np.asarray(clusterer.probabilities_, dtype=float)
    labels = raw.copy()
    next_label = int(raw[raw >= 0].max() + 1) if np.any(raw >= 0) else 0
    for index in np.flatnonzero(raw < 0):
        labels[index] = next_label
        next_label += 1
    ordered = tuple(dict.fromkeys(labels.tolist()))
    remap = {label: index for index, label in enumerate(ordered)}
    labels = np.asarray([remap[label] for label in labels], dtype=np.int64)
    if len(np.unique(labels)) == 0 or (labels < 0).any():
        raise ClusteringError("CIVD clustering did not assign every station")
    return HdbscanClusters(
        labels=labels,
        raw_labels=raw,
        probabilities=probabilities,
        n_clusters=int(len(np.unique(labels))),
        n_noise=int(np.count_nonzero(raw < 0)),
    )


__all__ = ["ClusteringError", "HdbscanClusters", "hdbscan_station_clusters"]


