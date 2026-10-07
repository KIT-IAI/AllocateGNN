"""006 D3：统一非 Z 候选池及严格 k 的选址入口。"""

import numpy as np

from sglib.core.algorithms.planning_candidates import generate_candidates
from sglib.core.algorithms.pmedian import solve_pmedian_greedy


def build_pool(grid, k, working_crs, *, max_workspace_bytes=128 * 2**20):
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("规划 k 必须为实际正站点数")
    if "zero_mask" not in grid:
        raise ValueError("规划候选池缺少正式 C/U/Z 支持")
    n_buildable = int((~grid.zero_mask.astype(bool)).sum())
    legacy = min(max(n_buildable // 100, 300), 1000)
    requested = max(legacy, 2 * k)
    record = {"k": k, "n_buildable": n_buildable, "legacy_requested_M": legacy, "requested_M": requested,
              "minimum_unique_M": 2 * k, "buildability": "non_Z_all_countries", "weights": "uniform_not_candidate_field"}
    if n_buildable < requested:
        return None, {**record, "actual_M": None, "status": "INELIGIBLE_BY_DESIGN", "reason": "INSUFFICIENT_NON_Z_GRID_POINTS"}
    # 主动删除旧 wc_others_ratio 分支的输入，四国一律采用非 Z 口径。
    frame = grid[["geometry", "zero_mask"]].copy().reset_index(drop=True)
    result = generate_candidates(frame, np.ones(len(frame)), {"n_candidates": requested, "buildability_threshold": .3,
        "random_state": 42, "max_workspace_bytes": max_workspace_bytes}, projection_crs=working_crs)
    actual = len(result.candidate_indices)
    return result, {**record, "actual_M": actual, "status": "VALID" if actual >= 2 * k else "INELIGIBLE_BY_DESIGN",
                    "reason": "" if actual >= 2 * k else "CENTROID_GRID_MAPPING_DEDUP_BELOW_2K"}


def solve_strict(demand_coords, facility_coords, weights, k, config):
    if k <= 0 or k > len(facility_coords):
        raise ValueError("严格 k 不可满足，禁止缩小实际站点数")
    selected, assignment = solve_pmedian_greedy(demand_coords, facility_coords, weights, k, config)
    if len(selected) != k or len(set(map(int, selected))) != k:
        raise ValueError("规划求解器没有返回严格 k 个不同站点")
    return selected, assignment
