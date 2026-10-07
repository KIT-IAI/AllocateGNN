"""006 的统一邻域接入算子；只消费同单位、同工作坐标系的类型化数组。"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from sglib.core.infra.hashing import sha256_json


@dataclass(frozen=True)
class ConnectionRegion:
    country: str
    region: str
    unit: str
    working_crs: str
    capacity_basis: str
    grid_xy: np.ndarray
    station_xy: np.ndarray
    station_ids: np.ndarray
    demand: np.ndarray
    capacity: np.ndarray
    assignment: np.ndarray


def stride_candidates(n_grid, n_candidates=2000):
    """沿原始 grid 行序取固定数量；不足时拒绝，不悄悄缩小候选池。"""
    if not isinstance(n_grid, int) or not isinstance(n_candidates, int):
        raise ValueError("候选数必须为整数")
    if isinstance(n_grid, bool) or isinstance(n_candidates, bool) or n_candidates <= 0:
        raise ValueError("候选数必须为正整数")
    if n_grid < n_candidates:
        raise ValueError("原始网格不足固定候选数")
    return np.arange(0, n_grid, n_grid // n_candidates, dtype=np.int64)[:n_candidates]


def neighbourhood_members(points, queries, radius_m):
    """工作 CRS 米制平面上的闭球邻域，成员按原始行序返回。"""
    points, queries = np.asarray(points, float), np.asarray(queries, float)
    if points.ndim != 2 or queries.ndim != 2 or points.shape[1:] != (2,) or queries.shape[1:] != (2,):
        raise ValueError("邻域坐标必须为 n×2")
    if not len(points) or not np.isfinite(points).all() or not np.isfinite(queries).all():
        raise ValueError("邻域坐标必须非空且有限")
    if not np.isfinite(radius_m) or radius_m <= 0:
        raise ValueError("半径必须为有限正米数")
    return list(cKDTree(points).query_ball_point(queries, radius_m, return_sorted=True))


def neighbourhood_sum(values, members):
    values = np.asarray(values, float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("邻域求和输入必须是有限向量")
    return np.array([values[np.asarray(indices, dtype=int)].sum() for indices in members])


def reference_field(assignment, demand):
    """固定 VD-Ref：每站台账需求均分到该站的 canonical VD 格点。"""
    labels, demand = np.asarray(assignment), np.asarray(demand, float)
    if labels.ndim != 1 or labels.dtype.kind not in "iu" or demand.ndim != 1 or not len(demand):
        raise ValueError("VD-Ref 输入形状或类型不符")
    if not np.isfinite(demand).all() or np.any(demand < 0) or np.any((labels < 0) | (labels >= len(demand))):
        raise ValueError("VD-Ref 输入值不合法")
    counts = np.bincount(labels, minlength=len(demand))
    if np.any(counts == 0):
        raise ValueError("VD-Ref 存在无支持格点的站点")
    result = demand[labels] / counts[labels]
    if not np.isclose(result.sum(), demand.sum(), rtol=1e-12, atol=1e-8):
        raise ValueError("VD-Ref 质量不守恒")
    return result


def reference_surface(region, radii_km=(10., 20.), n_candidates=2000):
    """不读取模型场，生成台账 / Ref 邻域；返回可追溯的候选和站点成员。"""
    grid, stations = np.asarray(region.grid_xy, float), np.asarray(region.station_xy, float)
    demand, capacity = np.asarray(region.demand, float), np.asarray(region.capacity, float)
    if demand.shape != capacity.shape or demand.shape != (len(stations),):
        raise ValueError("站点台账行未对齐")
    if len(region.station_ids) != len(stations) or len(set(map(str, region.station_ids))) != len(stations):
        raise ValueError("站点身份不唯一或未对齐")
    if not np.isfinite(capacity).all() or np.any(capacity < 0):
        raise ValueError("容量必须是非负有限数")
    if len(region.assignment) != len(grid):
        raise ValueError("VD assignment 与原始 grid 不同长")
    rows = stride_candidates(len(grid), n_candidates)
    queries = grid[rows]
    pool_hash = sha256_json({"grid_row": rows.tolist(), "xy": queries.tolist(), "working_crs": region.working_crs})
    ref = reference_field(region.assignment, demand)
    surface, membership = [], []
    base = {"country": region.country, "region": region.region, "unit": region.unit,
            "working_crs": region.working_crs, "capacity_basis": region.capacity_basis,
            "candidate_pool_hash": pool_hash}
    for radius in radii_km:
        sn = neighbourhood_members(stations, queries, radius * 1000.)
        gn = neighbourhood_members(grid, queries, radius * 1000.)
        g, f, a = neighbourhood_sum(demand, sn), neighbourhood_sum(capacity, sn), neighbourhood_sum(ref, gn)
        for i, row in enumerate(rows):
            identity = {**base, "radius_km": radius, "candidate_id": int(row)}
            surface.append({**identity, "x": queries[i, 0], "y": queries[i, 1],
                            "G": g[i], "F": f[i], "A_ref": a[i], "n_stations": len(sn[i]),
                            "n_grid": len(gn[i]), "grid_coverage_fraction": len(gn[i]) / len(grid)})
            membership.extend({"country": region.country, "region": region.region, "radius_km": radius,
                               "candidate_id": int(row), "station_id": str(region.station_ids[j])} for j in sn[i])
    return pd.DataFrame(surface), pd.DataFrame(membership, columns=["country", "region", "radius_km", "candidate_id", "station_id"])


def scenario_preflight(region, surface, lambdas=(.25, .5, 1.), reference_radius_km=10., ref_threshold=.25):
    """冻结情景前的台账预审；Ref 不准入不排除 C6 的台账上界评价。"""
    reference = surface[surface.radius_km == reference_radius_km]
    if reference.empty:
        raise ValueError("缺少固定负荷参考半径")
    f0 = float(np.median(reference.F))
    c_med = float(np.median(region.capacity))
    per_radius = []
    for radius, rows in surface.groupby("radius_km", sort=True):
        median_g = float(np.median(rows.G))
        residual = float(np.mean(np.abs(rows.A_ref - rows.G)))
        per_radius.append({"radius_km": float(radius), "median_G": median_g,
                           "median_F": float(np.median(rows.F)), "ref_mean_abs_error": residual,
                           "ref_score_radius": residual / median_g if median_g > 0 else None,
                           "zero_station_fraction": float(rows.n_stations.eq(0).mean()),
                           "full_grid_coverage_fraction": float(rows.grid_coverage_fraction.eq(1.).mean())})
    scores = [row["ref_score_radius"] for row in per_radius]
    score = max(scores) if all(value is not None for value in scores) else None
    eligible = score is not None and score <= ref_threshold
    reason = "" if eligible else "REF_ZERO_MEDIAN_DEMAND" if score is None else "REF_SCORE_ABOVE_THRESHOLD"
    output = []
    for load in lambdas:
        x = float(load * f0)
        for row in per_radius:
            rows = surface[surface.radius_km == row["radius_km"]]
            q = np.maximum(x - rows.F.to_numpy() + rows.G.to_numpy(), 0.)
            output.append({"country": region.country, "region": region.region, "unit": region.unit,
                "capacity_basis": region.capacity_basis, "working_crs": region.working_crs,
                "candidate_pool_hash": str(rows.candidate_pool_hash.iloc[0]), "n_candidates": len(rows),
                "lambda": float(load), "reference_radius_km": reference_radius_km, "reference_median_F": f0,
                "X": x, "median_station_capacity": c_med, "X_over_median_station_capacity": x / c_med if c_med > 0 else None,
                **row, "ref_score": score, "ref_threshold": ref_threshold, "ref_eligible": eligible,
                "ref_reason": reason, "scenario_defined": x > 0,
                "scenario_reason": "" if x > 0 else "ZERO_REFERENCE_CAPACITY_MEDIAN",
                "c4_c5_expected_eligible": bool(eligible and x > 0), "c6_expected_eligible": x > 0,
                "ledger_zero_q_fraction": float(np.mean(q == 0.))})
    return pd.DataFrame(output)
