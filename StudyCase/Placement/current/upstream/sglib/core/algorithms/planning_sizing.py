# -*- coding: utf-8 -*-
"""
定容管线(国别无关)
====================

    D̂_k   = ( Σ_{i∈C_k} w_i / Σ_i w_i ) × D_region      预测站级峰值需求
    Q_rec = γ · D̂_k                                      推荐容量(连续)

站表不是带列名的 GeoDataFrame,而是三个数组(坐标 / 需求 / 容量)——
列名翻译由 Experiment adapter 完成；本模块只接收数组。

γ=1.5 的论证是 **UK 台账的经验事实**(利用率中位 68.3% ≈ 1/γ),
其它案例沿用它属于**协议常数**,须并报该案例自己的实测利用率。
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .planning_geometry import haversine_distance_matrix

DEFAULT_SAFETY_MARGIN = 1.5


def predict_substation_demand(
    assignment: np.ndarray,
    weights: np.ndarray,
    d_region: float,
    k: int,
) -> np.ndarray:
    """把区域总需求按格点权重摊到各推荐站。按构造 Σ = d_region。"""
    if d_region <= 0:
        raise ValueError(f"d_region 必须为正，实得 {d_region}")

    w_sum = float(np.sum(weights))
    if w_sum <= 0:
        return np.full(k, d_region / max(k, 1))

    cluster_w = np.zeros(k, dtype=float)
    np.add.at(cluster_w, np.asarray(assignment, dtype=int), np.asarray(weights, float))
    return cluster_w / w_sum * float(d_region)


def recommend_capacity(
    predicted_demand: np.ndarray,
    safety_margin: float = DEFAULT_SAFETY_MARGIN,
    discretise_to: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """推荐容量 Q_rec = γ·D̂;默认连续(主口径),档位表仅供敏感性。"""
    q = np.asarray(predicted_demand, dtype=float) * float(safety_margin)
    if discretise_to is None:
        return q

    sizes = np.sort(np.asarray(discretise_to, dtype=float))
    idx = np.searchsorted(sizes, q, side="left")
    over = idx >= len(sizes)
    out = sizes[np.clip(idx, 0, len(sizes) - 1)]
    out = np.where(over, np.ceil(q / sizes[-1]) * sizes[-1], out)
    return out


def _matching_threshold(station_lonlat: np.ndarray,
                        max_dist_multiplier: float,
                        max_dist_km: float) -> float:
    """动态阈值 = 真实站最近邻距离中位数 × multiplier,上限 max_dist_km。"""
    if len(station_lonlat) > 1:
        dd = haversine_distance_matrix(station_lonlat, station_lonlat)
        np.fill_diagonal(dd, np.inf)
        mean_nn = float(np.median(dd.min(axis=1)))
    else:
        mean_nn = max_dist_km
    return min(mean_nn * max_dist_multiplier, max_dist_km)


def match_to_real_substations(
    rec_coords: np.ndarray,
    station_lonlat: np.ndarray,
    station_demand: np.ndarray,
    station_firm: np.ndarray,
    max_dist_multiplier: float = 3.0,
    max_dist_km: float = 20.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """多对一匹配(主协议):每个推荐站取最近真实站;超阈值置 NaN。

    返回: (actual_demand, firm_capacity, match_dist_km, low_confidence_mask)
    """
    demand = np.asarray(station_demand, dtype=float)
    firm = np.asarray(station_firm, dtype=float)
    threshold = _matching_threshold(np.asarray(station_lonlat, float),
                                    max_dist_multiplier, max_dist_km)

    dist = haversine_distance_matrix(np.asarray(rec_coords, float),
                                     np.asarray(station_lonlat, float))
    nearest = dist.argmin(axis=1)
    match_dist = dist[np.arange(len(rec_coords)), nearest]
    low_conf = match_dist > threshold

    actual = demand[nearest].astype(float)
    firm_matched = firm[nearest].astype(float)
    actual[low_conf] = np.nan
    firm_matched[low_conf] = np.nan

    return actual, firm_matched, match_dist, low_conf


def match_to_real_substations_unique(
    rec_coords: np.ndarray,
    station_lonlat: np.ndarray,
    station_demand: np.ndarray,
    station_firm: np.ndarray,
    max_dist_multiplier: float = 3.0,
    max_dist_km: float = 20.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """一对一(全局最优指派)匹配变体 —— 定容结论的协议敏感性检验。

    额外返回碰撞率 = 多对一协议下「非唯一最近邻」的推荐站占比。
    """
    from scipy.optimize import linear_sum_assignment

    demand = np.asarray(station_demand, dtype=float)
    firm = np.asarray(station_firm, dtype=float)
    threshold = _matching_threshold(np.asarray(station_lonlat, float),
                                    max_dist_multiplier, max_dist_km)

    dist = haversine_distance_matrix(np.asarray(rec_coords, float),
                                     np.asarray(station_lonlat, float))

    nearest = dist.argmin(axis=1)
    _, counts = np.unique(nearest, return_counts=True)
    n_dup = int(counts[counts > 1].sum() - (counts > 1).sum())
    collision_rate = float(n_dup / len(rec_coords)) if len(rec_coords) else 0.0

    rows, cols = linear_sum_assignment(dist)
    n_rec = len(rec_coords)
    matched_col = np.full(n_rec, -1, dtype=int)
    matched_col[rows] = cols
    unassigned = matched_col < 0
    safe_col = np.where(unassigned, 0, matched_col)
    match_dist = np.where(unassigned, np.inf,
                          dist[np.arange(n_rec), safe_col])
    low_conf = match_dist > threshold

    actual = demand[safe_col].astype(float)
    firm_matched = firm[safe_col].astype(float)
    actual[low_conf] = np.nan
    firm_matched[low_conf] = np.nan

    return actual, firm_matched, match_dist, low_conf, collision_rate


def compute_sizing_metrics(
    q_rec: np.ndarray,
    actual_demand: np.ndarray,
    firm_capacity: Optional[np.ndarray] = None,
    predicted_demand: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """定容指标(低置信匹配已由 NaN 排除;数值路径冻结)。

        TUR  = D_actual / Q_rec × 100%
        CE   = |Q_rec − D_actual| / D_actual × 100%
        RSD  = |D̂ − D_actual| / D_actual × 100%   ← 需传 predicted_demand
        OPR / UPR / FCE / TUR_aggregate / TUR_actual_observed 同规范定义。

    **CE 与 RSD 的区别(口径要点,勿混用)**:CE 把带裕度的推荐容量 Q_rec = γ·D̂
    直接与裸需求 D 比,γ 不约去——其最优点落在 γ = 2/3 而非「分配无误差」处,
    因此它度量的是「裕度设得准不准」而非「需求分到得准不准」。RSD 把同一条
    定容规则同时施加在推荐值与观测基准上(Q_rec = γ·D̂ 对 Q_obs = γ·D),γ 代数
    约去,退化为选址与匹配之后的相对峰值分配误差,才是「规则定容」任务要问的量。

    RSD 只在 predicted_demand 给出时计算;不传则输出不含 RSD_* 列,既有调用者
    的产物逐位不变(纯增量,E-2)。
    """
    q = np.asarray(q_rec, dtype=float)
    d = np.asarray(actual_demand, dtype=float)

    valid = (d > 0) & (q > 0) & np.isfinite(d) & np.isfinite(q)
    if int(valid.sum()) == 0:
        return {"n_matched": 0}

    qv, dv = q[valid], d[valid]
    tur = dv / qv * 100.0
    ce = np.abs(qv - dv) / dv * 100.0

    out: Dict[str, float] = {
        "n_matched": int(valid.sum()),
        "TUR_mean": float(tur.mean()),
        "TUR_median": float(np.median(tur)),
        "TUR_aggregate": float(dv.sum() / qv.sum() * 100.0),
        "CE_mean": float(ce.mean()),
        "CE_median": float(np.median(ce)),
        "OPR": float(np.mean(qv > 2.0 * dv)),
        "UPR": float(np.mean(qv < dv)),
    }

    if predicted_demand is not None:
        #: γ 在 Q_rec/Q_obs 之比中约去,故 RSD 直接在 D̂ 与 D 上取;
        #: valid 掩码沿用 q/d 的,保证与 CE 同一批站,两指标可逐站对照。
        p = np.asarray(predicted_demand, dtype=float)[valid]
        ok = np.isfinite(p)
        if int(ok.sum()) > 0:
            rsd = np.abs(p[ok] - dv[ok]) / dv[ok] * 100.0
            out["RSD_mean"] = float(rsd.mean())
            out["RSD_median"] = float(np.median(rsd))

    if firm_capacity is not None:
        f = np.asarray(firm_capacity, dtype=float)[valid]
        ok = (f > 0) & np.isfinite(f)
        if int(ok.sum()) > 0:
            fce = np.abs(qv[ok] - f[ok]) / f[ok] * 100.0
            out["FCE_mean"] = float(fce.mean())
            out["FCE_median"] = float(np.median(fce))
            out["TUR_actual_observed"] = float(np.median(dv[ok] / f[ok] * 100.0))

    return out
