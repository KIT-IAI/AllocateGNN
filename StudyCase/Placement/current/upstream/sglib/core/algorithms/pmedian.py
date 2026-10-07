# -*- coding: utf-8 -*-
"""
p-median 近似求解器(选址公共件;数值冻结;全仓库唯一实现)
==========================================================

Phase 1: 贪心构造 O(k·M·N)
Phase 2: 向量化局部交换 —— 对每个 j_in 一次矩阵运算评估所有候选 j_out

收敛拍板(2026-08-10):单一实现直接调用,消费方一律从本模块导入,禁止二次
实现；距离函数从 geometry 导入。当前由 country-first 的 103/203 Planning
Notebook 通过纯 Planning API 调用；历史 exp_milp warm start 已随第一代论文支删除。
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .planning_geometry import haversine_distance_matrix


def _compute_nn_sn(dist_matrix: np.ndarray,
                   selected: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """每个需求点的最近/次近已选站。返回 (nn_idx, nn_dist, sn_idx, sn_dist),局部索引。"""
    sub = dist_matrix[:, selected]
    if len(selected) == 1:
        nn_idx = np.zeros(len(dist_matrix), dtype=int)
        nn_dist = sub[:, 0]
        sn_idx = np.zeros(len(dist_matrix), dtype=int)
        sn_dist = np.full(len(dist_matrix), np.inf)
        return nn_idx, nn_dist, sn_idx, sn_dist

    kth = min(1, len(selected) - 1)
    part = np.argpartition(sub, kth, axis=1)[:, :2]
    d0 = sub[np.arange(len(sub)), part[:, 0]]
    d1 = sub[np.arange(len(sub)), part[:, 1]]

    swap_mask = d1 < d0
    nn_local = np.where(swap_mask, part[:, 1], part[:, 0])
    sn_local = np.where(swap_mask, part[:, 0], part[:, 1])
    nn_dist = np.minimum(d0, d1)
    sn_dist = np.maximum(d0, d1)

    return nn_local, nn_dist, sn_local, sn_dist


def _eval_all_swaps_for_jin(jin_local: int, unselected: np.ndarray,
                            nn_idx: np.ndarray, nn_dist: np.ndarray,
                            sn_dist: np.ndarray, dist_matrix: np.ndarray,
                            weights: np.ndarray) -> np.ndarray:
    """固定 j_in,一次性向量化评估所有 unselected 候选的 swap delta(负 = 改善)。"""
    D_out = dist_matrix[:, unselected]
    nn_col = nn_dist[:, np.newaxis]

    affected = (nn_idx == jin_local)
    delta = np.where(D_out < nn_col, D_out - nn_col, 0.0)

    if affected.any():
        sn_col = sn_dist[affected, np.newaxis]
        d_aff = D_out[affected]
        nn_aff = nn_dist[affected, np.newaxis]
        delta[affected] = np.minimum(sn_col, d_aff) - nn_aff

    return weights @ delta


def solve_pmedian_greedy(demand_coords: np.ndarray, facility_coords: np.ndarray,
                         weights: np.ndarray, k: int,
                         pmedian_config: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """贪心构造 + 向量化局部交换求解 p-median。返回 (selected_indices, assignment)。

    restart_seed 默认 42(与历史结果逐位兼容);种子敏感性实验经 config 覆写。
    """
    max_iter = pmedian_config.get('max_iter', 100)
    random_restarts = pmedian_config.get('random_restarts', 1)

    N = len(demand_coords)
    M = len(facility_coords)
    k = min(k, M)

    if k == 0:
        return np.array([], dtype=int), np.zeros(N, dtype=int)
    if k >= M:
        sel = np.arange(M)
        sub = haversine_distance_matrix(demand_coords, facility_coords)
        return sel, sel[sub.argmin(axis=1)]

    dist_matrix = haversine_distance_matrix(demand_coords, facility_coords)

    # 过滤零权重点加速(对 swap 评估无贡献)
    nz_mask = weights > 0
    dist_nz = dist_matrix[nz_mask]
    w_nz = weights[nz_mask]
    w_sum = w_nz.sum()
    if w_sum == 0:
        w_nz = np.ones(nz_mask.sum()) / nz_mask.sum()
        w_sum = 1.0

    def _wsd_from_selected(sel_arr):
        sub = dist_nz[:, sel_arr]
        return float(np.dot(w_nz, sub.min(axis=1)) / w_sum)

    # === Phase 1: 贪心构造 ===
    selected_list = []
    remaining = set(range(M))
    nearest_dist = np.full(nz_mask.sum(), np.inf)

    for _ in range(k):
        remaining_arr = np.array(list(remaining))
        cand_dists = dist_nz[:, remaining_arr]
        new_nearest = np.minimum(nearest_dist[:, np.newaxis], cand_dists)
        improvements = w_nz @ (nearest_dist[:, np.newaxis] - new_nearest)

        best_local = improvements.argmax()
        best_j = int(remaining_arr[best_local])

        selected_list.append(best_j)
        remaining.remove(best_j)
        nearest_dist = np.minimum(nearest_dist, dist_nz[:, best_j])

    best_selected = np.array(selected_list)
    best_wsd = _wsd_from_selected(best_selected)

    # === Phase 2: 向量化局部交换 ===
    for restart in range(random_restarts):
        if restart > 0:
            rng = np.random.RandomState(
                int(pmedian_config.get('restart_seed', 42)) + restart)
            current = list(best_selected)
            unsel = [j for j in range(M) if j not in set(current)]
            n_swap = max(1, k // 10)
            n_swap = min(n_swap, len(current), len(unsel))
            out_idx = rng.choice(len(current), size=n_swap, replace=False)
            in_idx = rng.choice(len(unsel), size=n_swap, replace=False)
            for i in range(n_swap):
                current[out_idx[i]] = unsel[in_idx[i]]
            selected = np.array(current)
        else:
            selected = best_selected.copy()

        improved = True
        iteration = 0

        while improved and iteration < max_iter:
            improved = False
            iteration += 1

            selected_set = set(selected.tolist())
            unselected = np.array([j for j in range(M) if j not in selected_set])
            if len(unselected) == 0:
                break

            nn_idx, nn_dist_arr, sn_idx, sn_dist_arr = _compute_nn_sn(dist_nz, selected)

            lower_bounds = (
                certified_swap_lower_bounds(nn_idx, nn_dist_arr, sn_dist_arr, dist_nz, w_nz, unselected, len(selected))
                if pmedian_config.get("certified_row_pruning", True) else None
            )

            global_best_delta = 0.0
            global_best_jin_local = -1
            global_best_jout_idx = -1

            for jin_local in range(len(selected)):
                if lower_bounds is not None and np.min(lower_bounds[jin_local]) >= global_best_delta - 1e-10:
                    continue
                deltas = _eval_all_swaps_for_jin(
                    jin_local, unselected, nn_idx, nn_dist_arr, sn_dist_arr,
                    dist_nz, w_nz)
                local_best = deltas.argmin()
                if deltas[local_best] < global_best_delta - 1e-10:
                    global_best_delta = deltas[local_best]
                    global_best_jin_local = jin_local
                    global_best_jout_idx = local_best

            if global_best_jin_local >= 0:
                j_out = unselected[global_best_jout_idx]
                selected[global_best_jin_local] = j_out
                improved = True

        current_wsd = _wsd_from_selected(selected)
        if current_wsd < best_wsd:
            best_wsd = current_wsd
            best_selected = selected.copy()

    # 最终分配(用全部 N 点)
    sub = dist_matrix[:, best_selected]
    assignment = best_selected[sub.argmin(axis=1)]

    return best_selected, assignment


def certified_swap_lower_bounds(nn_idx, nn_dist, sn_dist, dist_matrix, weights, unselected, k):
    """原浮点 swap 值的保守下界，只用于证明一整行不可能触发原改进条件。

    逐点原 delta 可拆为非正 add 项与仅属于被移除簇的非负 correction；两者
    在同一个点不会同时非零。重组求和的误差用保守 gamma_(16*n+128) 包围。
    可能改进的行仍调用原 _eval_all_swaps_for_jin，保留原 argmin 和 tie-break。
    """
    n, columns = len(weights), len(unselected)
    fallback = np.full((k, columns), -np.inf)
    if n == 0 or dist_matrix.dtype != np.float64 or weights.dtype != np.float64:
        return fallback
    if not np.isfinite(weights).all() or (weights < 0).any():
        return fallback
    factor = (16 * n + 128) * np.finfo(np.float64).eps
    if factor >= .5:
        return fallback
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        out = dist_matrix[:, unselected]
        add = np.minimum(out - nn_dist[:, None], 0.)
        correction = np.maximum(np.minimum(out, sn_dist[:, None]) - nn_dist[:, None], 0.)
        base = weights @ add
        grouped = np.zeros((k, columns), dtype=np.float64)
        np.add.at(grouped, nn_idx, weights[:, None] * correction)
        estimate = base[None, :] + grouped
        magnitude = -base[None, :] + grouped
        gamma = factor / (1. - factor)
        error = gamma * magnitude + np.finfo(np.float64).tiny * (4 * n + 16)
        lower = np.nextafter(estimate - error, -np.inf)
    if not np.isfinite(lower).all():
        return fallback
    return lower


def assign_demand_points(demand_coords: np.ndarray,
                         selected_facility_coords: np.ndarray) -> np.ndarray:
    """将格点分配到最近的已选站。"""
    dist = haversine_distance_matrix(demand_coords, selected_facility_coords)
    return dist.argmin(axis=1)
