\
\
\
\
\
   
from typing import Dict, Tuple

import numpy as np


def haversine_distance_matrix(coords_a: np.ndarray,
                              coords_b: np.ndarray) -> np.ndarray:
\
\
\
\
\
\
\
\
\
       
    R = 6371.0
    lon_a = np.radians(coords_a[:, 0:1])
    lat_a = np.radians(coords_a[:, 1:2])
    lon_b = np.radians(coords_b[:, 0:1].T)
    lat_b = np.radians(coords_b[:, 1:2].T)

    dlat = lat_b - lat_a
    dlon = lon_b - lon_a
    a = np.sin(dlat / 2) ** 2 + np.cos(lat_a) * np.cos(lat_b) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _compute_nn_sn(dist_matrix: np.ndarray,
                   selected: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
\
\
\
\
\
\
       
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


def _eval_all_swaps_for_jin(
    jin_local: int,
    unselected: np.ndarray,
    nn_idx: np.ndarray,
    nn_dist: np.ndarray,
    sn_dist: np.ndarray,
    dist_matrix: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
\
\
\
\
\
       
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


def solve_pmedian_greedy(
    demand_coords: np.ndarray,
    facility_coords: np.ndarray,
    weights: np.ndarray,
    k: int,
    pmedian_config: Dict,
) -> Tuple[np.ndarray, np.ndarray]:
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
       
    max_iter = pmedian_config.get("max_iter", 100)
    random_restarts = pmedian_config.get("random_restarts", 1)

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

                              
    for restart in range(random_restarts):
        if restart > 0:
                                                
                                              
            rng = np.random.RandomState(
                int(pmedian_config.get("restart_seed", 42)) + restart)
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

                                                   
            global_best_delta = 0.0
            global_best_jin_local = -1
            global_best_jout_idx = -1

            for jin_local in range(len(selected)):
                deltas = _eval_all_swaps_for_jin(
                    jin_local, unselected,
                    nn_idx, nn_dist_arr, sn_dist_arr,
                    dist_nz, w_nz,
                )
                local_best = deltas.argmin()
                if deltas[local_best] < global_best_delta - 1e-10:
                    global_best_delta = deltas[local_best]
                    global_best_jin_local = jin_local
                    global_best_jout_idx = local_best

            if global_best_jin_local >= 0:
                j_in = selected[global_best_jin_local]
                j_out = unselected[global_best_jout_idx]
                selected[global_best_jin_local] = j_out
                improved = True

        current_wsd = _wsd_from_selected(selected)
        if current_wsd < best_wsd:
            best_wsd = current_wsd
            best_selected = selected.copy()

                   
    sub = dist_matrix[:, best_selected]
    assignment = best_selected[sub.argmin(axis=1)]

    return best_selected, assignment


def assign_demand_points(demand_coords: np.ndarray,
                         selected_facility_coords: np.ndarray) -> np.ndarray:
                       
    dist = haversine_distance_matrix(demand_coords, selected_facility_coords)
    return dist.argmin(axis=1)
