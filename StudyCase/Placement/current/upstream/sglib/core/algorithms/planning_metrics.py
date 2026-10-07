# -*- coding: utf-8 -*-
"""
Planning 指标单一居所

三族指标,一处定义:
1. **选址指标**(WSD/LWFL/LBI/DCR):haversine 距离(km);
2. **定容指标**(TUR/CE/OPR/UPR/FCE):标准容量档位向上取整 + 安全裕度。

统计口径纪律:全程 numpy ``.std()``(ddof=0);任何 pandas Series 入口先转
numpy float64(Series 默认 ddof=1,混入即漂移)。
"""
from typing import Dict, Sequence

import numpy as np

# ════════════════════════════════════════════════════════════
# 选址指标(haversine km;数值冻结)
# ════════════════════════════════════════════════════════════

# haversine 单一居所 = 包内 Downstream.planning.geometry;
# 与此前局部实现的唯一差异 = arcsin 入参 clip 到 [0,1],仅影响浮点噪声边界)
from .planning_geometry import haversine_vector as _haversine_vector


def compute_wsd(demand_coords: np.ndarray, facility_coords: np.ndarray,
                assignment: np.ndarray, weights: np.ndarray) -> float:
    """加权服务距离 WSD = Σ(w_i·d_i)/Σw_i,单位 km。"""
    assigned = facility_coords[assignment]
    dist = _haversine_vector(demand_coords[:, 0], demand_coords[:, 1],
                             assigned[:, 0], assigned[:, 1])
    w_sum = weights.sum()
    if w_sum == 0:
        return 0.0
    return float(np.dot(weights, dist) / w_sum)


def compute_lwfl(demand_coords: np.ndarray, facility_coords: np.ndarray,
                 assignment: np.ndarray, weights: np.ndarray) -> float:
    """加权设施负荷 LWFL = Σ(w_i·d_i)(不归一化),单位 km·weight。"""
    assigned = facility_coords[assignment]
    dist = _haversine_vector(demand_coords[:, 0], demand_coords[:, 1],
                             assigned[:, 0], assigned[:, 1])
    return float(np.dot(weights, dist))


def compute_lbi(assignment: np.ndarray, weights: np.ndarray, k: int) -> float:
    """负荷均衡指数 LBI = std(L_k)/mean(L_k)(numpy std,ddof=0)。"""
    loads = np.zeros(k)
    np.add.at(loads, assignment, weights)
    mean_load = loads.mean()
    if mean_load == 0:
        return 0.0
    return float(loads.std() / mean_load)


def compute_dcr(demand_coords: np.ndarray, facility_coords: np.ndarray,
                assignment: np.ndarray, weights: np.ndarray,
                radius_km: float) -> float:
    """需求覆盖率 DCR(r) = Σ(w_i : d_i ≤ r)/Σw_i。"""
    assigned = facility_coords[assignment]
    dist = _haversine_vector(demand_coords[:, 0], demand_coords[:, 1],
                             assigned[:, 0], assigned[:, 1])
    w_sum = weights.sum()
    if w_sum == 0:
        return 0.0
    return float(weights[dist <= radius_km].sum() / w_sum)


def compute_all_siting_metrics(demand_coords: np.ndarray,
                               facility_coords: np.ndarray,
                               assignment: np.ndarray,
                               weights: np.ndarray,
                               dcr_radii: Sequence[float]) -> Dict[str, float]:
    """全部选址指标 {'WSD','LWFL','LBI','DCR_5km',...}。"""
    k = int(facility_coords.shape[0])
    result = {
        'WSD': compute_wsd(demand_coords, facility_coords, assignment, weights),
        'LWFL': compute_lwfl(demand_coords, facility_coords, assignment, weights),
        'LBI': compute_lbi(assignment, weights, k),
    }
    for r in dcr_radii:
        key = f'DCR_{int(r)}km' if r == int(r) else f'DCR_{r}km'
        result[key] = compute_dcr(demand_coords, facility_coords, assignment,
                                  weights, r)
    return result


# ════════════════════════════════════════════════════════════
# 定容指标(标准档位 + 安全裕度;数值冻结)
# ════════════════════════════════════════════════════════════

def _round_up_to_standard(value: float, standard_sizes: Sequence[float]) -> float:
    """向上取整到最近标准容量;超最大档取最大档整数倍。"""
    sizes = sorted(standard_sizes)
    for s in sizes:
        if s >= value:
            return s
    max_size = sizes[-1]
    return float(np.ceil(value / max_size) * max_size)


def compute_tur(predicted_demand_mva: np.ndarray,
                actual_demand_mva: np.ndarray,
                sizing_config: dict) -> tuple:
    """变压器利用率。返回 (tur %, q_rec MVA);Q_rec = 档位取整(预测 × 裕度)。"""
    safety = sizing_config['safety_margin']
    sizes = sizing_config['standard_sizes_mva']
    q_rec = np.array([_round_up_to_standard(p * safety, sizes)
                      for p in predicted_demand_mva])
    tur = np.where(q_rec > 0, actual_demand_mva / q_rec * 100.0, 0.0)
    return tur, q_rec


def compute_all_sizing_metrics(predicted_demand_mva: np.ndarray,
                               actual_demand_mva: np.ndarray,
                               sizing_config: dict,
                               firm_capacity_mva: np.ndarray = None) -> Dict[str, float]:
    """全部定容指标:TUR_mean/std、CE_mean、OPR、UPR(+FCE_mean/FC_match_rate)。

    numpy std(ddof=0);OPR = P(Q_rec > 2·actual);UPR = P(Q_rec < actual)。
    """
    tur, q_rec = compute_tur(predicted_demand_mva, actual_demand_mva, sizing_config)
    ce = np.where(actual_demand_mva > 0,
                  np.abs(q_rec - actual_demand_mva) / actual_demand_mva * 100.0,
                  0.0)
    result = {
        'TUR_mean': float(tur.mean()),
        'TUR_std': float(tur.std()),
        'CE_mean': float(ce.mean()),
        'OPR': float(np.mean(q_rec > 2 * actual_demand_mva)),
        'UPR': float(np.mean(q_rec < actual_demand_mva)),
    }
    if firm_capacity_mva is not None:
        valid = (firm_capacity_mva > 0) & np.isfinite(firm_capacity_mva)
        if valid.sum() > 0:
            fc, qr = firm_capacity_mva[valid], q_rec[valid]
            result['FCE_mean'] = float((np.abs(qr - fc) / fc * 100.0).mean())
            result['FC_match_rate'] = float(np.mean(np.isclose(qr, fc, rtol=0.01)))
    return result
