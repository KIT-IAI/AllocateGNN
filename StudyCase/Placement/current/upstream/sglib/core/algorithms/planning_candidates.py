# -*- coding: utf-8 -*-
"""
候选站址生成（planning 公共件；数值冻结，CRS 参数化）
================================================

可建设性过滤 + 加权 KMeans 聚类,把 ~5 万格点降维为数百候选位置;
每个候选聚合其簇内格点权重,作为 p-median/MILP 的需求点。
KMeans 在**国别投影坐标**上运行(UK=EPSG:27700;经 projection_crs 参数化,
调用方从 profile 取,禁在此写死非 UK 值)。

⚠ 候选集完全由 (格点, 权重, 配置, 种子) 决定——确定性;但 KMeans 结果对
sklearn 版本敏感,跨版本对照按带内口径。
"""
import warnings
from dataclasses import dataclass
from typing import Dict, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans


@dataclass
class CandidateResult:
    """候选站址生成结果。"""
    candidate_indices: np.ndarray     # 候选格点在原始 gdf 中的索引
    candidate_gdf: gpd.GeoDataFrame   # 候选格点(EPSG:4326)
    candidate_coords: np.ndarray      # (M, 2) lon/lat
    labels: np.ndarray                # 每个可建设格点所属候选簇
    buildable_indices: np.ndarray     # 可建设格点在原始 gdf 中的索引


def compute_buildability_mask(gdf: gpd.GeoDataFrame,
                              threshold: float) -> pd.Series:
    """可建设性掩码:wc_others_ratio 超过阈值视为不可建设(True = 可建设)。"""
    if "wc_others_ratio" in gdf.columns:
        return gdf["wc_others_ratio"] <= threshold
    if "zero_mask" in gdf.columns:
        return ~gdf["zero_mask"].astype(bool)
    return pd.Series(True, index=gdf.index)


def compute_n_candidates(n_grid: int,
                         n_candidates_config: Optional[int]) -> int:
    """候选站数量;配置为 None 时自动 = min(max(N//100, 300), 1000)。"""
    if n_candidates_config is not None:
        return int(n_candidates_config)
    return min(max(n_grid // 100, 300), 1000)


def generate_candidates(
    gdf: gpd.GeoDataFrame,
    weights: np.ndarray,
    candidates_config: Dict,
    projection_crs: str = "EPSG:27700",
) -> CandidateResult:
    """生成候选站址(KMeans 降维;数值路径冻结,仅 CRS 参数化)。"""
    threshold = candidates_config["buildability_threshold"]
    n_candidates_cfg = candidates_config.get("n_candidates")
    random_state = candidates_config.get("random_state", 42)

    # 1. 可建设性过滤
    mask = compute_buildability_mask(gdf, threshold)
    buildable_idx = np.where(mask.values)[0]
    n_buildable = len(buildable_idx)

    # 2. 确定候选数量
    n_candidates = compute_n_candidates(n_buildable, n_candidates_cfg)
    if n_buildable < n_candidates:
        warnings.warn(
            f"可建设格点数 ({n_buildable}) < 请求候选数 ({n_candidates}),"
            f"将使用全部 {n_buildable} 个可建设格点"
        )
        n_candidates = n_buildable

    if n_buildable == 0:
        empty_gdf = gdf.iloc[:0].copy()
        return CandidateResult(
            candidate_indices=np.array([], dtype=int),
            candidate_gdf=empty_gdf,
            candidate_coords=np.empty((0, 2)),
            labels=np.array([], dtype=int),
            buildable_indices=np.array([], dtype=int),
        )

    # 3. 投影到国别投影
    buildable_gdf = gdf.iloc[buildable_idx].copy()
    buildable_proj = buildable_gdf.to_crs(projection_crs)
    proj_coords = np.column_stack([
        buildable_proj.geometry.x.values,
        buildable_proj.geometry.y.values,
    ])

    # 可建设格点的权重
    buildable_weights = weights[buildable_idx]
    buildable_weights = np.maximum(buildable_weights, 0.0)
    w_sum = buildable_weights.sum()
    sample_weight = buildable_weights if w_sum > 0 else None

    # 4. KMeans 聚类
    kmeans = KMeans(
        n_clusters=n_candidates,
        random_state=random_state,
        n_init=10,
    )
    kmeans.fit(proj_coords, sample_weight=sample_weight)
    labels = kmeans.labels_

    # 5. 每个质心找最近的可建设格点(作为候选位置的实际坐标)
    centroids = kmeans.cluster_centers_
    workspace = candidates_config.get("max_workspace_bytes")
    if workspace is None:
        dist_matrix = cdist(centroids, proj_coords, metric="euclidean")
        nearest_local = dist_matrix.argmin(axis=1)
    else:
        from .chunked_distance import chunked_nearest_assignment

        nearest_local, _, _ = chunked_nearest_assignment(
            centroids, proj_coords, max_workspace_bytes=int(workspace)
        )

    # 6. 去重(多个质心可能映射同一格点)
    unique_local, inverse = np.unique(nearest_local, return_inverse=True)
    candidate_indices = buildable_idx[unique_local]
    remapped_labels = inverse[labels]

    candidate_gdf = gdf.iloc[candidate_indices].copy().reset_index(drop=True)
    candidate_coords = np.column_stack([
        candidate_gdf.geometry.x.values,
        candidate_gdf.geometry.y.values,
    ])

    return CandidateResult(
        candidate_indices=candidate_indices,
        candidate_gdf=candidate_gdf,
        candidate_coords=candidate_coords,
        labels=remapped_labels,
        buildable_indices=buildable_idx,
    )


def aggregate_weights_to_candidates(
    weights: np.ndarray,
    candidate_result: CandidateResult,
) -> np.ndarray:
    """把格点权重聚合到候选位置,返回 sum=1 的 (M,) 数组。"""
    buildable_w = weights[candidate_result.buildable_indices]
    buildable_w = np.maximum(buildable_w, 0.0)

    n_cand = len(candidate_result.candidate_indices)
    agg = np.zeros(n_cand)
    np.add.at(agg, candidate_result.labels, buildable_w)

    # 不可建设格点的权重均匀分配到所有候选(通常占比很小)
    unbuildable_total = weights.sum() - buildable_w.sum()
    if unbuildable_total > 0 and n_cand > 0:
        agg += unbuildable_total / n_cand

    total = agg.sum()
    if total > 0:
        agg /= total
    return agg


def local_refine(
    selected_coords: np.ndarray,
    grid_coords: np.ndarray,
    weights: np.ndarray,
    radius_m: float = 500.0,
    projection_crs: str = "EPSG:27700",
) -> np.ndarray:
    """局部精化:每个选中站址移到其 500 m 邻域加权质心,再吸附最近格点。

    候选集把分辨率从 ~140 m 降到 ~1.4 km;本步把它拉回来,不改变组合选择。
    """
    from pyproj import Transformer
    from scipy.spatial import cKDTree

    to_m = Transformer.from_crs("EPSG:4326", projection_crs, always_xy=True)
    gx, gy = to_m.transform(grid_coords[:, 0], grid_coords[:, 1])
    grid_m = np.column_stack([gx, gy])

    sx, sy = to_m.transform(selected_coords[:, 0], selected_coords[:, 1])
    sel_m = np.column_stack([sx, sy])

    tree = cKDTree(grid_m)

    refined = selected_coords.copy()
    for j, c in enumerate(sel_m):
        idx = tree.query_ball_point(c, r=radius_m)
        if not idx:
            continue
        w = weights[idx]
        if w.sum() <= 0:
            continue
        centroid = (grid_m[idx] * w[:, None]).sum(axis=0) / w.sum()
        nearest = idx[int(np.argmin(np.linalg.norm(grid_m[idx] - centroid, axis=1)))]
        refined[j] = grid_coords[nearest]
    return refined
